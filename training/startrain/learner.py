"""Single-machine replay learner, checkpoint publication, and metrics."""

from __future__ import annotations

import json
import hashlib
import heapq
import os
import random
import time
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from .checkpoint import (
    MODEL_MANIFEST_FORMAT,
    MODEL_MANIFEST_VERSION,
    ExponentialMovingAverage,
    ModelManifest,
    ResumeCheckpoint,
    collect_model_garbage,
    collect_recovery_garbage,
    load_checkpoint,
    load_model_manifest,
    save_checkpoint,
    sha256_file,
    verify_file,
    write_recovery_checkpoint,
    write_model_pointer,
    write_resume_cutover,
)
from .config import (
    DataConfig,
    ExperimentConfig,
    LearnerConfig,
    RingMixtureConfig,
    TrainConfig,
)
from .contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE
from .losses import LossWeights
from .model import MODEL_SCHEMA_VERSION, GraphResTNet
from .optim import build_optimizer
from .replay import (
    DecodedReplayShard,
    ReplaySample,
    augment_sample,
    collate_replay_samples,
    decode_replay_shard,
)
from .replay_store import ReplaySelection, ReplaySpan, ReplayStore
from .runtime import RunIdentity, append_jsonl, atomic_json
from .symmetry import deterministic_transform
from .training import (
    DeviceBatchPrefetcher,
    build_scheduler,
    maybe_compile_model,
    train_step,
    unwrap_model,
)


class AugmentedReplayDataset(Dataset[ReplaySample]):
    def __init__(
        self,
        samples: Sequence[ReplaySample],
        *,
        seed: int,
        epoch: int,
        enabled: bool,
    ) -> None:
        self.samples = list(samples)
        self.seed = seed
        self.epoch = epoch
        self.enabled = enabled

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ReplaySample:
        sample = self.samples[index]
        if not self.enabled:
            return sample
        transform = deterministic_transform(
            seed=self.seed, sample_index=index, epoch=self.epoch
        )
        return augment_sample(sample, transform)


@dataclass(frozen=True, slots=True)
class ShardBatchChunk:
    ring: int
    span_index: int
    dataset_start: int
    sample_count: int


class LazyShardReplayDataset(Dataset[ReplaySample]):
    """Indexes immutable shards and lazily caches a bounded number per worker."""

    def __init__(
        self,
        selection: ReplaySelection,
        *,
        seed: int,
        epoch: int,
        augmentation_enabled: bool,
        shard_cache_size: int,
    ) -> None:
        if not selection.spans or shard_cache_size <= 0:
            raise ValueError("lazy replay requires spans and a positive shard cache")
        self.spans = selection.spans
        self.seed = seed
        self.epoch = epoch
        self.augmentation_enabled = augmentation_enabled
        self.shard_cache_size = shard_cache_size
        self._ends: list[int] = []
        self._starts: list[int] = []
        self._ring_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
        total = 0
        for span in self.spans:
            start = total
            self._starts.append(start)
            total += span.sample_count
            self._ends.append(total)
            self._ring_ranges[span.record.ring].append((start, total))
        self._cache: OrderedDict[int, DecodedReplayShard] = OrderedDict()
        self._verified_shards: set[int] = set()
        self.shard_load_count = 0
        self.checksum_verification_count = 0
        self.sample_materialization_count = 0

    def __len__(self) -> int:
        return self._ends[-1]

    @property
    def rings(self) -> tuple[int, ...]:
        return tuple(sorted(self._ring_ranges))

    def ring_count(self, ring: int) -> int:
        return sum(end - start for start, end in self._ring_ranges.get(ring, ()))

    def ring_offset_to_index(self, ring: int, offset: int) -> int:
        if offset < 0:
            raise IndexError(offset)
        for start, end in self._ring_ranges.get(ring, ()):
            width = end - start
            if offset < width:
                return start + offset
            offset -= width
        raise IndexError(offset)

    def shard_batch_chunks(self, batch_size: int) -> tuple[ShardBatchChunk, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        chunks: list[ShardBatchChunk] = []
        for span_index, span in enumerate(self.spans):
            for batch in range(span.sample_count // batch_size):
                chunks.append(
                    ShardBatchChunk(
                        ring=span.record.ring,
                        span_index=span_index,
                        dataset_start=(self._starts[span_index] + batch * batch_size),
                        sample_count=batch_size,
                    )
                )
        return tuple(chunks)

    @staticmethod
    def indices_for_chunk(chunk: ShardBatchChunk) -> list[int]:
        return list(
            range(
                chunk.dataset_start,
                chunk.dataset_start + chunk.sample_count,
            )
        )

    def __getitem__(self, index: int) -> ReplaySample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        span_index = bisect_right(self._ends, index)
        previous_end = self._ends[span_index - 1] if span_index else 0
        span = self.spans[span_index]
        shard = self._load_span_shard(span)
        sample = shard.sample(span.sample_start + index - previous_end)
        self.sample_materialization_count += 1
        if not self.augmentation_enabled:
            return sample
        transform = deterministic_transform(
            seed=self.seed, sample_index=index, epoch=self.epoch
        )
        return augment_sample(sample, transform)

    def __getitems__(self, indices: list[int]) -> list[ReplaySample]:
        """Bulk Dataset hook used by DataLoader for one shard-local batch."""

        return [self[index] for index in indices]

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_verified_shards"] = set()
        state["shard_load_count"] = 0
        state["checksum_verification_count"] = 0
        state["sample_materialization_count"] = 0
        return state

    def _load_span_shard(self, span: ReplaySpan) -> DecodedReplayShard:
        shard_id = span.record.shard_id
        cached = self._cache.pop(shard_id, None)
        if cached is None:
            if shard_id not in self._verified_shards:
                self.checksum_verification_count += 1
                if _sha256(span.record.path) != span.record.checksum_sha256:
                    raise ValueError(
                        f"replay shard checksum failed: {span.record.path}"
                    )
                self._verified_shards.add(shard_id)
            cached = decode_replay_shard(span.record.path)
            self.shard_load_count += 1
            if len(cached) != span.record.sample_count:
                raise ValueError("replay shard count disagrees with its manifest")
        self._cache[shard_id] = cached
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return cached


class UniqueReplayBatchSampler(Sampler[list[int]]):
    """Deterministic no-replacement batches with explicit DDP rank partitioning."""

    def __init__(
        self,
        dataset: LazyShardReplayDataset,
        *,
        batch_size: int,
        batches: int,
        seed: int,
        epoch: int,
        ring_stratified: bool,
        ring_weights: Mapping[int, float] | None = None,
        shards_per_batch: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if len(dataset) <= 0 or batch_size <= 0 or batches <= 0:
            raise ValueError("dataset, batch_size, and batches must be positive")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("invalid distributed sampler rank")
        if (
            isinstance(shards_per_batch, bool)
            or not isinstance(shards_per_batch, int)
            or shards_per_batch <= 0
        ):
            raise ValueError("shards_per_batch must be positive")
        if shards_per_batch > 1 and not ring_stratified:
            raise ValueError("cross-shard batches require ring-stratified replay")
        required = batches * world_size * batch_size
        if required > len(dataset):
            raise ValueError("replay window lacks enough unique samples")
        self.batch_size = batch_size
        self.batches = batches
        self.seed = seed
        self.epoch = epoch
        self.dataset = dataset
        self.ring_stratified = ring_stratified
        self.ring_weights = (
            {int(ring): float(weight) for ring, weight in ring_weights.items()}
            if ring_weights is not None
            else None
        )
        self.shards_per_batch = shards_per_batch
        self.rank = rank
        self.world_size = world_size
        self.chunks = dataset.shard_batch_chunks(batch_size)
        if len(self.chunks) < batches * world_size * shards_per_batch:
            raise ValueError("replay spans lack enough full shard-local unique batches")
        if ring_stratified:
            if len(self.chunks) < batches * world_size:
                raise ValueError(
                    "ring-stratified replay lacks enough homogeneous unique batches"
                )
            if self.ring_weights is not None and (
                any(weight < 0 for weight in self.ring_weights.values())
                or not any(weight > 0 for weight in self.ring_weights.values())
            ):
                raise ValueError("ring weights must be non-negative with positive sum")

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        total_batches = self.batches * self.world_size
        if not self.ring_stratified:
            chosen_groups = [
                [chunk] for chunk in rng.sample(self.chunks, total_batches)
            ]
        else:
            by_ring: dict[int, list[ShardBatchChunk]] = defaultdict(list)
            for chunk in self.chunks:
                by_ring[chunk.ring].append(chunk)
            groups_by_ring = {
                ring: _cross_shard_chunk_groups(
                    chunks,
                    shards_per_batch=self.shards_per_batch,
                    rng=rng,
                )
                for ring, chunks in by_ring.items()
            }
            capacities = {ring: len(groups) for ring, groups in groups_by_ring.items()}
            if self.ring_weights is None:
                order = []
                used = {ring: 0 for ring in capacities}
                while len(order) < total_batches:
                    available = [
                        ring for ring in capacities if used[ring] < capacities[ring]
                    ]
                    rng.shuffle(available)
                    for ring in available:
                        order.append(ring)
                        used[ring] += 1
                        if len(order) == total_batches:
                            break
            else:
                used = _weighted_ring_quotas(
                    total_batches,
                    capacities=capacities,
                    weights={
                        ring: self.ring_weights.get(ring, 0.0) for ring in capacities
                    },
                )
                order = [
                    ring for ring, count in sorted(used.items()) for _ in range(count)
                ]
                rng.shuffle(order)
            ring_groups: dict[int, Iterator[list[ShardBatchChunk]]] = {}
            for ring, count in used.items():
                ring_groups[ring] = iter(rng.sample(groups_by_ring[ring], count))
            chosen_groups = [next(ring_groups[ring]) for ring in order]
        local_groups = chosen_groups[self.rank :: self.world_size]
        local = []
        for group in local_groups:
            batch_indices: list[int] = []
            base = self.batch_size // len(group)
            remainder = self.batch_size % len(group)
            for offset, chunk in enumerate(group):
                indices = self.dataset.indices_for_chunk(chunk)
                rng.shuffle(indices)
                count = base + int(offset < remainder)
                batch_indices.extend(indices[:count])
            rng.shuffle(batch_indices)
            if len(batch_indices) != self.batch_size:
                raise RuntimeError("cross-shard sampler emitted an incomplete batch")
            local.append(batch_indices)
        if len(local) != self.batches:
            raise RuntimeError("distributed sampler emitted uneven batches")
        yield from local


def _cross_shard_chunk_groups(
    chunks: Sequence[ShardBatchChunk],
    *,
    shards_per_batch: int,
    rng: random.Random,
) -> list[list[ShardBatchChunk]]:
    by_span: dict[int, list[ShardBatchChunk]] = defaultdict(list)
    for chunk in chunks:
        by_span[chunk.span_index].append(chunk)
    heap: list[tuple[int, float, int]] = []
    for span_index, span_chunks in by_span.items():
        rng.shuffle(span_chunks)
        heap.append((-len(span_chunks), rng.random(), span_index))
    heapq.heapify(heap)
    output: list[list[ShardBatchChunk]] = []
    while len(heap) >= shards_per_batch:
        selected = [heapq.heappop(heap) for _ in range(shards_per_batch)]
        group = []
        for negative_count, _tie, span_index in selected:
            group.append(by_span[span_index].pop())
            remaining = -negative_count - 1
            if remaining:
                heapq.heappush(
                    heap,
                    (-remaining, rng.random(), span_index),
                )
        output.append(group)
    return output


def _maximum_cross_shard_groups(
    chunk_counts: Sequence[int],
    *,
    shards_per_batch: int,
) -> int:
    if not chunk_counts:
        return 0
    upper = sum(chunk_counts) // shards_per_batch
    low = 0
    while low < upper:
        middle = (low + upper + 1) // 2
        available = sum(min(count, middle) for count in chunk_counts)
        if available >= middle * shards_per_batch:
            low = middle
        else:
            upper = middle - 1
    return low


def _weighted_ring_quotas(
    total: int,
    *,
    capacities: Mapping[int, int],
    weights: Mapping[int, float],
) -> dict[int, int]:
    if total <= 0:
        raise ValueError("weighted ring quota total must be positive")
    eligible = [ring for ring, weight in weights.items() if weight > 0]
    if any(capacities.get(ring, 0) <= 0 for ring in eligible):
        raise ValueError("weighted ring replay is missing a configured ring")
    if sum(capacities[ring] for ring in eligible) < total:
        raise ValueError("weighted ring replay lacks enough configured capacity")
    total_weight = sum(float(weights[ring]) for ring in eligible)
    targets = {ring: total * float(weights[ring]) / total_weight for ring in eligible}
    quotas = {ring: int(targets[ring]) for ring in eligible}
    remaining = total - sum(quotas.values())
    remainders = sorted(
        eligible,
        key=lambda ring: (
            targets[ring] - quotas[ring],
            float(weights[ring]),
            -ring,
        ),
        reverse=True,
    )
    for ring in remainders[:remaining]:
        quotas[ring] += 1
    if any(quotas[ring] > capacities[ring] for ring in eligible):
        raise ValueError("weighted ring replay cannot satisfy configured proportions")
    return {ring: quotas.get(ring, 0) for ring in capacities}


class JSONLMetrics:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict[str, object]) -> None:
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def plateau_policy_decision(
    *,
    lag_steps: int,
    soft_lag_steps: int,
    hard_replay_lag_steps: int,
    status_matches_candidate: bool,
    terminal_rejection: bool,
    rejection_streak: int,
    reset_after_rejections: int,
    action: str,
    reset_already_applied: bool,
) -> str:
    if lag_steps < soft_lag_steps:
        return "proceed"
    if action == "reduce_lr_keep_weights" and reset_already_applied:
        return "proceed"
    recovery_due = (
        status_matches_candidate
        and terminal_rejection
        and (
            lag_steps >= hard_replay_lag_steps
            or rejection_streak >= reset_after_rejections
        )
    )
    if (
        recovery_due
        and action == "reduce_lr_keep_weights"
        and not reset_already_applied
    ):
        return "recover"
    if (
        status_matches_candidate
        and terminal_rejection
        and lag_steps >= hard_replay_lag_steps
        and action == "reset_from_champion"
        and not reset_already_applied
    ):
        return "reset"
    if (
        status_matches_candidate
        and terminal_rejection
        and rejection_streak >= reset_after_rejections
        and action == "reset_from_champion"
        and not reset_already_applied
    ):
        return "reset"
    if (
        status_matches_candidate
        and terminal_rejection
        and rejection_streak < reset_after_rejections
        and lag_steps < hard_replay_lag_steps
    ):
        return "proceed"
    return "pause"


class ImmutableModelPublisher:
    def __init__(self, root: str | Path, run_identity: RunIdentity) -> None:
        self.root = Path(root)
        self.checkpoint_directory = self.root / "checkpoints"
        self.manifest_directory = self.root / "manifests"
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_directory.mkdir(parents=True, exist_ok=True)
        self.candidate_path = self.root / "candidate.json"
        self.champion_path = self.root / "champion.json"
        self.run_identity = run_identity

    def publish(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        ema: ExponentialMovingAverage,
        step: int,
        epoch: int,
        config: dict[str, object],
        examples_consumed: int | None = None,
        global_batch_size: int | None = None,
    ) -> ModelManifest:
        if self.candidate_path.is_file():
            try:
                current = load_model_manifest(self.candidate_path)
            except ValueError:
                current = None
            if current is not None:
                if (
                    current.model_step == step
                    and current.run_id == self.run_identity.run_id
                    and current.generation_family == self.run_identity.generation_family
                ):
                    self._record_model_history(current)
                    return current
        staged = self.checkpoint_directory / f".candidate-{step:012d}.staging.pt"
        save_checkpoint(
            staged,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            step=step,
            epoch=epoch,
            config=config,
            extra={
                "training_step_version": f"step-{step:012d}",
                "run_id": self.run_identity.run_id,
                "generation_family": self.run_identity.generation_family,
                **(
                    {
                        "examples_consumed": examples_consumed,
                        "global_batch_size": global_batch_size,
                    }
                    if examples_consumed is not None
                    else {}
                ),
            },
        )
        checkpoint_sha256 = sha256_file(staged)
        model_identity = f"sha256-{checkpoint_sha256}"
        checkpoint = self.checkpoint_directory / f"{model_identity}.pt"
        if checkpoint.exists():
            verify_file(
                checkpoint,
                expected_sha256=checkpoint_sha256,
                expected_bytes=staged.stat().st_size,
            )
            staged.unlink()
        else:
            os.replace(staged, checkpoint)
            descriptor = os.open(checkpoint.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        manifest_payload = {
            "format": MODEL_MANIFEST_FORMAT,
            "schema_version": MODEL_MANIFEST_VERSION,
            "model_version": model_identity,
            "model_identity": model_identity,
            "model_step": step,
            "checkpoint": os.path.relpath(checkpoint, self.manifest_directory),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "weights": "ema",
            "run_id": self.run_identity.run_id,
            "generation_family": self.run_identity.generation_family,
            "rules_hash": RULES_HASH_WIRE,
            "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "created_ns": time.time_ns(),
        }
        serialized = (
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        manifest_sha256 = hashlib.sha256(serialized).hexdigest()
        manifest_path = self.manifest_directory / f"manifest-{manifest_sha256}.json"
        if manifest_path.exists():
            verify_file(
                manifest_path,
                expected_sha256=manifest_sha256,
                expected_bytes=len(serialized),
            )
        else:
            atomic_json(manifest_path, manifest_payload)
        manifest = load_model_manifest(manifest_path)
        write_model_pointer(self.candidate_path, manifest, role="candidate")
        published = load_model_manifest(self.candidate_path)
        self._record_model_history(published)
        return published

    def _record_model_history(self, manifest: ModelManifest) -> None:
        append_jsonl(
            self.root / "model-history.jsonl",
            {
                "schema_version": 1,
                "run_id": manifest.run_id,
                "generation_family": manifest.generation_family,
                "model_identity": manifest.model_identity,
                "model_step": manifest.model_step,
                "published_ns": manifest.published_ns,
                "manifest": os.path.relpath(
                    manifest.artifact_manifest or manifest.path,
                    self.root,
                ),
            },
            durable=True,
        )


AtomicModelPublisher = ImmutableModelPublisher


class LearnerLoop:
    def __init__(
        self,
        *,
        store: ReplayStore,
        model: GraphResTNet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        ema: ExponentialMovingAverage,
        output_directory: str | Path,
        learner_config: LearnerConfig,
        train_config: TrainConfig,
        data_config: DataConfig,
        loss_weights: LossWeights,
        seed: int,
        serialized_config: dict[str, object],
        run_identity: RunIdentity,
        ring_mixture_config: RingMixtureConfig = RingMixtureConfig(),
        promotion_status_path: str | Path | None = None,
        gpu_pause_path: str | Path | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("invalid learner distributed rank")
        self.store = store
        self.model = model.to(learner_config.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.learner_config = learner_config
        self.train_config = train_config
        self.data_config = data_config
        self.loss_weights = loss_weights
        self.seed = seed
        self.serialized_config = serialized_config
        self.run_identity = run_identity
        self.ring_mixture_config = ring_mixture_config
        self.promotion_status_path = (
            Path(promotion_status_path) if promotion_status_path is not None else None
        )
        self.gpu_pause_path = (
            Path(gpu_pause_path) if gpu_pause_path is not None else None
        )
        self._last_plateau_reset: tuple[str, str] | None = None
        self.rank = rank
        self.world_size = world_size
        self.store.register_run(run_identity)
        output_root = Path(output_directory)
        self.publisher = ImmutableModelPublisher(output_root, run_identity)
        self.selfplay_publisher = (
            ImmutableModelPublisher(output_root / "selfplay", run_identity)
            if learner_config.selfplay_snapshot_interval_examples is not None
            else None
        )
        self.cadence_path = output_root / "cadence.json"
        self._last_candidate_examples: int | None = None
        self._last_selfplay_examples: int | None = None
        self.metrics = JSONLMetrics(output_root / "metrics.jsonl")
        if rank == 0 and any(store.reconciliation_metrics.values()):
            self.metrics.append(
                {
                    "schema_version": 1,
                    "timestamp_ns": time.time_ns(),
                    "worker": "learner",
                    "event": "replay_reconciliation",
                    **store.reconciliation_metrics,
                }
            )
        self.step = 0
        self.epoch = 0
        self.examples_consumed = 0
        self._last_recovery_step = 0
        self._latest_total_replay_samples = 0
        # Replay batches are fixed-size and ring-homogeneous. Static compilation
        # avoids Inductor's dynamic backward reductions (which fail on variable
        # graph lengths) while allowing one cached graph per encountered ring.
        compiled_model = maybe_compile_model(
            self.model,
            enabled=train_config.compile,
            dynamic=False,
            recompile_limit=len(self.ring_mixture_config.rings),
            isolate_recompiles=True,
        )
        if world_size > 1:
            parameter = next(self.model.parameters())
            device_ids = (
                [parameter.device.index]
                if parameter.device.type == "cuda"
                and parameter.device.index is not None
                else None
            )
            self.compiled_model: torch.nn.Module = DistributedDataParallel(
                compiled_model,
                device_ids=device_ids,
                output_device=device_ids[0] if device_ids else None,
            )
        else:
            self.compiled_model = compiled_model

    @classmethod
    def from_experiment(
        cls,
        config: ExperimentConfig,
        *,
        store: ReplayStore,
        output_directory: str | Path,
        run_identity: RunIdentity,
        promotion_status_path: str | Path | None = None,
        gpu_pause_path: str | Path | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> "LearnerLoop":
        torch.manual_seed(config.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.train.seed)
        model = GraphResTNet(config.model).to(config.learner.device)
        optimizer = build_optimizer(model, config.optimizer)
        scheduler = build_scheduler(optimizer, config.train.scheduler)
        ema = ExponentialMovingAverage(model, decay=config.train.ema_decay)
        return cls(
            store=store,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            output_directory=output_directory,
            learner_config=config.learner,
            train_config=config.train,
            data_config=config.data,
            loss_weights=config.loss,
            seed=config.train.seed,
            serialized_config=config.as_dict(),
            run_identity=run_identity,
            ring_mixture_config=config.orchestration.ring_mixture,
            promotion_status_path=promotion_status_path,
            gpu_pause_path=gpu_pause_path,
            rank=rank,
            world_size=world_size,
        )

    def resume(
        self,
        checkpoint: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
    ) -> None:
        model_config = self.serialized_config.get("model")
        game_config = self.serialized_config.get("game")
        metadata = load_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            map_location=self.learner_config.device,
            expected_model_config=(
                model_config if isinstance(model_config, Mapping) else None
            ),
            expected_game_config=(
                game_config if isinstance(game_config, Mapping) else None
            ),
            expected_run_id=self.run_identity.run_id,
            expected_generation_family=self.run_identity.generation_family,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
            metadata_validator=self._resume_examples_consumed,
        )
        self.step = int(metadata["step"])
        self.epoch = int(metadata["epoch"])
        self.examples_consumed = self._resume_examples_consumed(metadata)
        self._last_recovery_step = self.step

    def _resume_examples_consumed(self, metadata: Mapping[str, object]) -> int:
        extra = metadata.get("extra")
        consumed = (
            extra.get("examples_consumed") if isinstance(extra, Mapping) else None
        )
        if isinstance(consumed, int) and not isinstance(consumed, bool):
            if consumed < 0:
                raise ValueError("checkpoint examples_consumed must be non-negative")
            return consumed
        uses_example_cadence = (
            self.learner_config.target_updates_per_new_sample is not None
            or self.learner_config.candidate_interval_examples is not None
            or self.learner_config.selfplay_snapshot_interval_examples is not None
        )
        if uses_example_cadence:
            raise ValueError(
                "legacy checkpoint lacks examples_consumed required by "
                "example-based learner controls"
            )
        step = metadata.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        return step * self.train_config.global_batch_size(self.world_size)

    def run(
        self,
        *,
        steps: int | None = None,
        stop_requested: Callable[[], bool] = lambda: False,
        progress: Callable[..., None] | None = None,
    ) -> int:
        target = (
            self.step + steps
            if steps is not None
            else (None if self.learner_config.unlimited else self.learner_config.steps)
        )
        completion_path = self.publisher.root / "learner-complete.json"
        if self.rank == 0 and (target is None or self.step < target):
            completion_path.unlink(missing_ok=True)
        if self.rank == 0 and not self.publisher.candidate_path.is_file():
            self._publish()
        if self.rank == 0:
            self._load_cadence_state()
            self._publish_due_models()
        self._distributed_barrier()
        interval_started = time.perf_counter()
        interval_steps = 0
        interval_data_wait_seconds = 0.0
        interval_window_setup_seconds = 0.0
        interval_cpu_device_seconds = 0.0
        interval_device_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        interval_copy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        while target is None or self.step < target:
            if self._collective_stop(stop_requested()):
                break
            if not self._gpu_pause_control(
                stop_requested=stop_requested, progress=progress
            ):
                break
            if not self._plateau_control(
                stop_requested=stop_requested, progress=progress
            ):
                break
            if progress is not None and self.rank == 0:
                progress(phase="replay_wait", step=self.step, epoch=self.epoch)
            if not self._wait_for_replay(
                stop_requested=stop_requested, progress=progress
            ):
                break
            selection = self._select_replay_spans()
            maximum_batches = self._maximum_unique_batches(selection)
            target_budget = (
                self.learner_config.steps_per_window
                if target is None
                else target - self.step
            )
            next_weight_step = self.ring_mixture_config.next_weight_step(self.step)
            weight_budget = (
                self.learner_config.steps_per_window
                if next_weight_step is None
                else next_weight_step - self.step
            )
            recovery_interval = self.learner_config.recovery_interval_steps
            recovery_budget = (
                self.learner_config.steps_per_window
                if recovery_interval is None
                else max(
                    1,
                    self._last_recovery_step + recovery_interval - self.step,
                )
            )
            batches = min(
                self.learner_config.steps_per_window,
                target_budget,
                weight_budget,
                recovery_budget,
                maximum_batches,
                self._plateau_step_budget(),
                self._utd_step_budget(),
            )
            batches = self._collective_min_int(batches)
            if batches <= 0:
                if progress is not None and self.rank == 0:
                    progress(
                        phase="update_to_data_wait",
                        step=self.step,
                        examples_consumed=self.examples_consumed,
                        replay_samples=self._latest_total_replay_samples,
                        target_updates_per_new_sample=(
                            self.learner_config.target_updates_per_new_sample
                        ),
                    )
                time.sleep(self.learner_config.replay_poll_seconds)
                continue
            watermark_name = f"learner-{self.run_identity.run_id}"
            if self.rank == 0:
                self.store.set_gc_watermark(watermark_name, selection)
            setup_started = time.perf_counter()
            loader = self._loader(selection, batches=batches)
            device = next(self.model.parameters()).device
            prefetcher = DeviceBatchPrefetcher(
                loader,
                device=device,
                enabled=self.data_config.pin_memory,
            )
            batch_iterator = iter(prefetcher)
            if self.rank == 0:
                interval_window_setup_seconds += time.perf_counter() - setup_started
            while True:
                data_wait_started = time.perf_counter()
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    break
                if self.rank == 0:
                    interval_data_wait_seconds += (
                        time.perf_counter() - data_wait_started
                    )
                consumed_copy_events = prefetcher.pop_copy_events()
                if self.rank == 0:
                    interval_copy_events.extend(consumed_copy_events)
                if not self._gpu_pause_control(
                    stop_requested=stop_requested, progress=progress
                ):
                    break
                if self._collective_stop(stop_requested()):
                    break
                step_started = time.perf_counter()
                device_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
                if self.rank == 0 and device.type == "cuda":
                    device_events = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    device_events[0].record()
                result = train_step(
                    self.compiled_model,
                    batch,
                    self.optimizer,
                    loss_weights=self.loss_weights,
                    precision=self.train_config.precision,
                    gradient_clip_norm=self.train_config.gradient_clip_norm,
                    scheduler=self.scheduler,
                    ema=self.ema,
                    trusted_batch=True,
                )
                if device_events is not None:
                    device_events[1].record()
                    interval_device_events.append(device_events)
                elif self.rank == 0:
                    interval_cpu_device_seconds += time.perf_counter() - step_started
                self.step += 1
                self.examples_consumed += self.train_config.global_batch_size(
                    self.world_size
                )
                if self.rank == 0:
                    interval_steps += 1
                if (
                    self.rank == 0
                    and self.step % self.learner_config.metrics_interval == 0
                ):
                    host_metrics = result.to_host()
                    h2d_seconds = (
                        sum(
                            started.elapsed_time(completed)
                            for started, completed in interval_copy_events
                        )
                        / 1_000.0
                    )
                    device_seconds = (
                        interval_cpu_device_seconds
                        + sum(
                            started.elapsed_time(completed)
                            for started, completed in interval_device_events
                        )
                        / 1_000.0
                    )
                    measured_at = time.perf_counter()
                    wall_seconds = measured_at - interval_started
                    measured_steps = max(1, interval_steps)
                    global_batch_size = self.train_config.global_batch_size(
                        self.world_size
                    )
                    self.metrics.append(
                        {
                            "schema_version": 1,
                            "timestamp_ns": time.time_ns(),
                            "worker": "learner",
                            "step": self.step,
                            "epoch": self.epoch,
                            "world_size": self.world_size,
                            "losses": host_metrics.losses,
                            "gradient_norm": host_metrics.gradient_norm,
                            "learning_rates": host_metrics.learning_rates,
                            "step_seconds": wall_seconds / measured_steps,
                            "examples_per_second": (
                                global_batch_size * measured_steps / wall_seconds
                            ),
                            "device_step_seconds": (device_seconds / measured_steps),
                            "device_examples_per_second": (
                                global_batch_size * measured_steps / device_seconds
                                if device_seconds
                                else None
                            ),
                            "data_wait_seconds": (
                                interval_data_wait_seconds / measured_steps
                            ),
                            "h2d_seconds": h2d_seconds / measured_steps,
                            "window_setup_seconds": (
                                interval_window_setup_seconds / measured_steps
                            ),
                            "metrics_interval_steps": measured_steps,
                            "metrics_interval_wall_seconds": wall_seconds,
                            "examples_consumed": self.examples_consumed,
                            "total_replay_samples": (self._latest_total_replay_samples),
                            "updates_per_new_sample": (
                                self.examples_consumed
                                / self._latest_total_replay_samples
                                if self._latest_total_replay_samples
                                else None
                            ),
                            "feature_path": batch.feature_path,
                            "replay_samples": selection.sample_count,
                            "replay_samples_by_ring": selection.samples_by_ring,
                            "ring_batch_weights": self._active_ring_weights(),
                            "replay_max_shard_id": selection.max_shard_id,
                            "effective_unique_samples": (
                                batches
                                * self.train_config.global_batch_size(self.world_size)
                            ),
                            "per_rank_batch_size": (
                                self.train_config.per_rank_batch_size
                            ),
                            "global_batch_size": (
                                self.train_config.global_batch_size(self.world_size)
                            ),
                        }
                    )
                    interval_started = measured_at
                    interval_steps = 0
                    interval_data_wait_seconds = 0.0
                    interval_window_setup_seconds = 0.0
                    interval_cpu_device_seconds = 0.0
                    interval_device_events.clear()
                    interval_copy_events.clear()
                if self.rank == 0:
                    self._publish_due_models()
                if progress is not None and self.rank == 0:
                    progress(phase="training", step=self.step, epoch=self.epoch)
            if self.rank == 0:
                self.store.clear_gc_watermark(watermark_name)
                self._maybe_write_recovery_checkpoint()
                self._maybe_collect_replay_garbage()
            self.epoch += 1
        if self.rank == 0:
            completed = target is not None and self.step >= target
            if completed:
                final_manifest = self._publish()
                atomic_json(
                    completion_path,
                    {
                        "schema_version": 1,
                        "run_id": self.run_identity.run_id,
                        "generation_family": (self.run_identity.generation_family),
                        "candidate_identity": final_manifest.model_identity,
                        "candidate_step": final_manifest.model_step,
                        "completed_ns": time.time_ns(),
                    },
                )
            else:
                self._maybe_write_recovery_checkpoint(force=True)
        self._distributed_barrier()
        return self.step

    def _loader(self, selection: ReplaySelection, *, batches: int) -> DataLoader:
        dataset = LazyShardReplayDataset(
            selection,
            seed=self.seed,
            epoch=self.epoch,
            augmentation_enabled=self.data_config.d5_augmentation,
            shard_cache_size=self.data_config.shard_cache_size,
        )
        batch_sampler = UniqueReplayBatchSampler(
            dataset,
            batch_size=self.train_config.per_rank_batch_size,
            batches=batches,
            seed=self.seed,
            epoch=self.epoch,
            ring_stratified=self.data_config.ring_stratified,
            ring_weights=self._active_ring_weights(),
            shards_per_batch=self.data_config.shards_per_batch,
            rank=self.rank,
            world_size=self.world_size,
        )
        if self.data_config.workers:
            return DataLoader(
                dataset=dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_replay_samples,
                num_workers=self.data_config.workers,
                pin_memory=self.data_config.pin_memory,
                prefetch_factor=self.data_config.prefetch_factor,
                persistent_workers=True,
                multiprocessing_context="spawn",
            )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_replay_samples,
            num_workers=0,
            pin_memory=self.data_config.pin_memory,
        )

    def _publish(self) -> ModelManifest:
        return self._publish_to(self.publisher)

    def _publish_to(self, publisher: ImmutableModelPublisher) -> ModelManifest:
        return publisher.publish(
            model=unwrap_model(self.compiled_model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            step=self.step,
            epoch=self.epoch,
            config=self.serialized_config,
            examples_consumed=self.examples_consumed,
            global_batch_size=self.train_config.global_batch_size(self.world_size),
        )

    def _load_cadence_state(self) -> None:
        if self._last_candidate_examples is not None:
            return
        if self.cadence_path.is_file():
            try:
                payload = json.loads(self.cadence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read learner cadence state: {exc}") from exc
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != 1
                or payload.get("run_id") != self.run_identity.run_id
                or payload.get("generation_family")
                != self.run_identity.generation_family
            ):
                raise ValueError("learner cadence state is incompatible")
            candidate_examples = payload.get("candidate_examples")
            selfplay_examples = payload.get("selfplay_examples")
            if (
                isinstance(candidate_examples, bool)
                or not isinstance(candidate_examples, int)
                or candidate_examples < 0
                or (
                    selfplay_examples is not None
                    and (
                        isinstance(selfplay_examples, bool)
                        or not isinstance(selfplay_examples, int)
                        or selfplay_examples < 0
                    )
                )
            ):
                raise ValueError("learner cadence counters are invalid")
            self._last_candidate_examples = candidate_examples
            self._last_selfplay_examples = selfplay_examples
            return

        self._last_candidate_examples = self._pointer_examples(
            self.publisher.candidate_path
        )
        if self.selfplay_publisher is not None:
            if not self.selfplay_publisher.candidate_path.is_file():
                candidate = load_model_manifest(self.publisher.candidate_path)
                write_model_pointer(
                    self.selfplay_publisher.candidate_path,
                    candidate,
                    role="candidate",
                )
            self._last_selfplay_examples = self._pointer_examples(
                self.selfplay_publisher.candidate_path
            )
        self._write_cadence_state()

    def _pointer_examples(self, pointer: Path) -> int:
        manifest = load_model_manifest(pointer)
        return manifest.model_step * self.train_config.global_batch_size(
            self.world_size
        )

    def _write_cadence_state(self) -> None:
        if self._last_candidate_examples is None:
            raise RuntimeError("candidate cadence was not initialized")
        atomic_json(
            self.cadence_path,
            {
                "schema_version": 1,
                "run_id": self.run_identity.run_id,
                "generation_family": self.run_identity.generation_family,
                "candidate_examples": self._last_candidate_examples,
                "selfplay_examples": self._last_selfplay_examples,
                "updated_ns": time.time_ns(),
            },
        )

    def _selfplay_snapshot_interval(self) -> int | None:
        steady = self.learner_config.selfplay_snapshot_interval_examples
        if steady is None:
            return None
        warmup = self.learner_config.selfplay_snapshot_warmup_interval_examples
        if (
            warmup is not None
            and self.examples_consumed
            < self.learner_config.selfplay_snapshot_warmup_examples
        ):
            return warmup
        return steady

    def _candidate_due(self) -> bool:
        interval_examples = self.learner_config.candidate_interval_examples
        if interval_examples is None:
            current_step = (
                load_model_manifest(self.publisher.candidate_path).model_step
                if self.publisher.candidate_path.is_file()
                else None
            )
            return (
                self.step % self.learner_config.candidate_interval == 0
                and current_step != self.step
            )
        if self._last_candidate_examples is None:
            raise RuntimeError("candidate cadence was not initialized")
        return (
            self.examples_consumed - self._last_candidate_examples >= interval_examples
        )

    def _selfplay_snapshot_due(self) -> bool:
        interval = self._selfplay_snapshot_interval()
        if interval is None:
            return False
        if self._last_selfplay_examples is None:
            raise RuntimeError("self-play cadence was not initialized")
        return self.examples_consumed - self._last_selfplay_examples >= interval

    def _publish_due_models(self) -> tuple[ModelManifest | None, ModelManifest | None]:
        self._load_cadence_state()
        candidate = None
        selfplay = None
        if self._candidate_due():
            candidate = self._publish()
            self._last_candidate_examples = self.examples_consumed
            self._last_recovery_step = self.step
            self.metrics.append(
                {
                    "schema_version": 1,
                    "timestamp_ns": time.time_ns(),
                    "worker": "learner",
                    "event": "promotion_candidate",
                    "model_identity": candidate.model_identity,
                    "model_step": candidate.model_step,
                    "examples_consumed": self.examples_consumed,
                }
            )
        if self._selfplay_snapshot_due():
            if self.selfplay_publisher is None:
                raise RuntimeError("self-play publisher is unavailable")
            selfplay = self._publish_to(self.selfplay_publisher)
            self._last_selfplay_examples = self.examples_consumed
            self.metrics.append(
                {
                    "schema_version": 1,
                    "timestamp_ns": time.time_ns(),
                    "worker": "learner",
                    "event": "selfplay_snapshot",
                    "model_identity": selfplay.model_identity,
                    "model_step": selfplay.model_step,
                    "examples_consumed": self.examples_consumed,
                    "interval_examples": self._selfplay_snapshot_interval(),
                }
            )
        if candidate is not None or selfplay is not None:
            self._write_cadence_state()
        return candidate, selfplay

    def _scale_learning_rates(self, scale: float) -> None:
        if scale == 1.0:
            return
        if not 0 < scale <= 1:
            raise ValueError("learning-rate scale must be in (0, 1]")
        for group in self.optimizer.param_groups:
            group["lr"] = float(group["lr"]) * scale
            if "initial_lr" in group:
                group["initial_lr"] = float(group["initial_lr"]) * scale
        self.scheduler.base_lrs = [
            float(learning_rate) * scale for learning_rate in self.scheduler.base_lrs
        ]
        if hasattr(self.scheduler, "_last_lr"):
            self.scheduler._last_lr = [
                float(group["lr"]) for group in self.optimizer.param_groups
            ]

    def _clear_optimizer_state(self) -> None:
        self.optimizer.state.clear()

    def _active_ring_weights(self) -> dict[int, float] | None:
        weights = self.ring_mixture_config.weights_for_step(self.step)
        if weights is None:
            return None
        return dict(zip(self.ring_mixture_config.rings, weights, strict=True))

    def _maybe_write_recovery_checkpoint(
        self, *, force: bool = False
    ) -> ResumeCheckpoint | None:
        interval = self.learner_config.recovery_interval_steps
        if interval is None and not force:
            return None
        if not force and (
            self.step <= self._last_recovery_step
            or interval is None
            or self.step - self._last_recovery_step < interval
        ):
            return None
        recovery = write_recovery_checkpoint(
            self.publisher.root,
            model=unwrap_model(self.compiled_model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            step=self.step,
            epoch=self.epoch,
            config=self.serialized_config,
            run_id=self.run_identity.run_id,
            generation_family=self.run_identity.generation_family,
            examples_consumed=self.examples_consumed,
            global_batch_size=self.train_config.global_batch_size(self.world_size),
        )
        self._last_recovery_step = self.step
        self.metrics.append(
            {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "worker": "learner",
                "event": "recovery_checkpoint",
                "step": recovery.step,
                "checkpoint_sha256": recovery.checkpoint_sha256,
                "checkpoint_bytes": recovery.checkpoint_bytes,
            }
        )
        retention = self._retention_config()
        if self.epoch % retention.gc_interval_windows == 0:
            metrics = collect_recovery_garbage(
                self.publisher.root,
                retain_checkpoints=retention.recovery_checkpoints,
                dry_run=retention.recovery_dry_run,
            )
            self.metrics.append(
                {
                    "schema_version": 1,
                    "timestamp_ns": time.time_ns(),
                    "worker": "learner",
                    "event": "recovery_gc",
                    **metrics,
                }
            )
        return recovery

    def _wait_for_replay(
        self,
        *,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
    ) -> bool:
        started = time.monotonic()
        while True:
            counts = self._eligible_replay_counts()
            active_counts = self._active_replay_counts(counts)
            available = sum(active_counts.values())
            ready = self._replay_is_ready(counts)
            stop, globally_ready = self._collective_flags(stop_requested(), ready)
            if stop:
                return False
            if globally_ready:
                return True
            if progress is not None and self.rank == 0:
                progress(
                    phase="replay_wait",
                    step=self.step,
                    available=available,
                    samples_by_ring=counts,
                )
            timeout = self.learner_config.replay_wait_timeout_seconds
            timed_out = bool(timeout and time.monotonic() - started >= timeout)
            if self._collective_any(timed_out):
                raise TimeoutError(
                    "minimum replay was not reached before the learner timeout"
                )
            time.sleep(self.learner_config.replay_poll_seconds)

    def _maximum_unique_batches(self, selection: ReplaySelection) -> int:
        batch = self.train_config.per_rank_batch_size
        chunk_counts: dict[int, list[int]] = defaultdict(list)
        for span in selection.spans:
            chunk_counts[span.record.ring].append(span.sample_count // batch)
        capacities = {
            ring: _maximum_cross_shard_groups(
                counts,
                shards_per_batch=self.data_config.shards_per_batch,
            )
            for ring, counts in chunk_counts.items()
        }
        capacity = sum(capacities.values()) // self.world_size
        weights = self._active_ring_weights()
        if not self.data_config.ring_stratified or weights is None:
            return capacity
        for batches in range(
            min(capacity, self.learner_config.steps_per_window),
            0,
            -1,
        ):
            try:
                _weighted_ring_quotas(
                    batches * self.world_size,
                    capacities=capacities,
                    weights=weights,
                )
            except ValueError:
                continue
            return batches
        return 0

    def _select_replay_spans(self) -> ReplaySelection:
        rings = self.ring_mixture_config.rings
        if self.learner_config.use_ring_mixture_curriculum and self.rank == 0:
            rings = self._active_replay_rings(self._eligible_replay_counts())
        selection = (
            self.store.select_recent_spans(
                rings=rings,
                per_ring_quota=self.learner_config.recent_samples_per_ring,
                run_id=self.run_identity.run_id,
                generation_family=self.run_identity.generation_family,
                current_model_step=self.step,
                max_model_lag_steps=self.learner_config.max_replay_lag_steps,
            )
            if self.rank == 0
            else None
        )
        selection = self._broadcast_object(selection)
        if not isinstance(selection, ReplaySelection):
            raise RuntimeError("rank 0 broadcast invalid replay selection metadata")
        return selection

    def _eligible_replay_counts(self) -> dict[int, int]:
        return self.store.eligible_sample_counts(
            self.ring_mixture_config.rings,
            run_id=self.run_identity.run_id,
            generation_family=self.run_identity.generation_family,
            current_model_step=self.step,
            max_model_lag_steps=self.learner_config.max_replay_lag_steps,
        )

    def _active_replay_rings(self, counts: Mapping[int, int]) -> tuple[int, ...]:
        step_weights = self.ring_mixture_config.weights_for_step(self.step)
        if step_weights is not None:
            return tuple(
                ring
                for ring, weight in zip(
                    self.ring_mixture_config.rings, step_weights, strict=True
                )
                if weight > 0
            )
        if not self.learner_config.use_ring_mixture_curriculum:
            return self.ring_mixture_config.rings
        total = sum(int(counts.get(ring, 0)) for ring in self.ring_mixture_config.rings)
        return self.ring_mixture_config.active_rings(total)

    def _active_replay_counts(self, counts: Mapping[int, int]) -> dict[int, int]:
        return {
            ring: int(counts.get(ring, 0)) for ring in self._active_replay_rings(counts)
        }

    def _replay_is_ready(self, counts: Mapping[int, int]) -> bool:
        active_counts = self._active_replay_counts(counts)
        per_ring_ready = (
            all(
                count >= self.learner_config.minimum_unique_samples_per_ring
                for count in active_counts.values()
            )
            if self.data_config.ring_stratified
            else True
        )
        return (
            sum(active_counts.values()) >= self.learner_config.minimum_replay_samples
            and per_ring_ready
            and self._available_batch_capacity(active_counts) >= self.world_size
        )

    def _available_batch_capacity(self, counts: Mapping[int, int]) -> int:
        batch = self.train_config.per_rank_batch_size
        if self.data_config.ring_stratified:
            return sum(count // batch for count in counts.values())
        return sum(counts.values()) // batch

    def _utd_step_budget(self) -> int:
        self._latest_total_replay_samples = self.store.total_committed_sample_count(
            run_id=self.run_identity.run_id,
            generation_family=self.run_identity.generation_family,
        )
        target = self.learner_config.target_updates_per_new_sample
        if target is None:
            return self.learner_config.steps_per_window
        if not self.store.committed_sample_history_is_complete(
            run_id=self.run_identity.run_id,
            generation_family=self.run_identity.generation_family,
        ):
            raise ValueError(
                "update-to-data control requires a complete committed-sample history"
            )
        allowed_examples = int(target * self._latest_total_replay_samples)
        remaining = max(0, allowed_examples - self.examples_consumed)
        return remaining // self.train_config.global_batch_size(self.world_size)

    def _plateau_step_budget(self) -> int:
        configured = self._plateau_config()
        budget = self.learner_config.steps_per_window
        if (
            configured.enabled
            and self.rank == 0
            and self.publisher.champion_path.is_file()
        ):
            champion = load_model_manifest(self.publisher.champion_path)
            candidate = (
                load_model_manifest(self.publisher.candidate_path)
                if self.publisher.candidate_path.is_file()
                else None
            )
            recovery_token = (
                champion.model_identity,
                candidate.model_identity if candidate is not None else "",
            )
            recovered_in_place = (
                configured.action == "reduce_lr_keep_weights"
                and self._last_plateau_reset == recovery_token
            )
            if not recovered_in_place:
                budget = max(
                    0,
                    champion.model_step
                    + self.learner_config.max_replay_lag_steps
                    - self.step,
                )
        value = self._broadcast_object(budget if self.rank == 0 else None)
        if not isinstance(value, int):
            raise RuntimeError("distributed plateau step budget is invalid")
        return value

    def _plateau_control(
        self,
        *,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
    ) -> bool:
        configured = self._plateau_config()
        if not configured.enabled or self.promotion_status_path is None:
            return True
        while True:
            action = (
                self._rank_zero_plateau_action(configured) if self.rank == 0 else None
            )
            action = self._broadcast_object(action)
            if not isinstance(action, dict):
                raise RuntimeError("distributed plateau action is invalid")
            kind = action.get("kind")
            if kind == "proceed":
                return True
            if kind == "reset":
                checkpoint = Path(str(action["checkpoint"]))
                previous_step = self.step
                champion_manifest = None
                if self.rank == 0:
                    champion_manifest = load_model_manifest(
                        self.publisher.champion_path
                    )
                self._distributed_barrier()
                self.resume(
                    checkpoint,
                    expected_sha256=str(action["sha256"]),
                    expected_bytes=int(action["bytes"]),
                )
                self._scale_learning_rates(configured.reset_learning_rate_scale)
                if self.rank == 0:
                    assert champion_manifest is not None
                    self._last_recovery_step = max(0, self.step - 1)
                    recovery = self._maybe_write_recovery_checkpoint(force=True)
                    if recovery is None:
                        raise RuntimeError(
                            "plateau reset did not create a recovery checkpoint"
                        )
                    cutover_created_ns = time.time_ns()
                    write_resume_cutover(
                        self.publisher.root,
                        manifest=recovery,
                        run_id=self.run_identity.run_id,
                        generation_family=self.run_identity.generation_family,
                        created_ns=cutover_created_ns,
                    )
                    write_model_pointer(
                        self.publisher.candidate_path,
                        champion_manifest,
                        role="candidate",
                    )
                    if self.promotion_status_path is not None:
                        atomic_json(
                            self.promotion_status_path,
                            {
                                "schema_version": 1,
                                "candidate_identity": champion_manifest.model_identity,
                                "candidate_step": champion_manifest.model_step,
                                "champion_identity": champion_manifest.model_identity,
                                "champion_step": champion_manifest.model_step,
                                "decision": "plateau_reset",
                                "terminal": True,
                                "consecutive_terminal_rejections": 0,
                                "cutover_created_ns": cutover_created_ns,
                                "updated_ns": time.time_ns(),
                            },
                        )
                    self.metrics.append(
                        {
                            "schema_version": 1,
                            "timestamp_ns": time.time_ns(),
                            "worker": "learner",
                            "event": "plateau_reset",
                            "reason": action.get("reset_reason"),
                            "from_step": previous_step,
                            "to_step": self.step,
                            "champion_identity": champion_manifest.model_identity,
                            "learning_rate_scale": (
                                configured.reset_learning_rate_scale
                            ),
                            "learning_rates": [
                                float(group["lr"])
                                for group in self.optimizer.param_groups
                            ],
                        }
                    )
                self._distributed_barrier()
                self._last_plateau_reset = (
                    str(action["champion_identity"]),
                    str(action["candidate_identity"]),
                )
                if progress is not None and self.rank == 0:
                    progress(
                        phase="plateau_reset",
                        step=self.step,
                        champion_identity=action["champion_identity"],
                        reason=action.get("reset_reason"),
                    )
                return True
            if kind == "recover":
                previous_step = self.step
                self._scale_learning_rates(configured.reset_learning_rate_scale)
                if configured.clear_optimizer_state_on_recovery:
                    self._clear_optimizer_state()
                if self.rank == 0:
                    self._last_recovery_step = max(0, self.step - 1)
                    recovery = self._maybe_write_recovery_checkpoint(force=True)
                    if recovery is None:
                        raise RuntimeError(
                            "plateau recovery did not create a recovery checkpoint"
                        )
                    cutover_created_ns = time.time_ns()
                    write_resume_cutover(
                        self.publisher.root,
                        manifest=recovery,
                        run_id=self.run_identity.run_id,
                        generation_family=self.run_identity.generation_family,
                        created_ns=cutover_created_ns,
                    )
                    if self.promotion_status_path is not None:
                        atomic_json(
                            self.promotion_status_path,
                            {
                                "schema_version": 1,
                                "candidate_identity": action["candidate_identity"],
                                "candidate_step": action["candidate_step"],
                                "champion_identity": action["champion_identity"],
                                "champion_step": action["champion_step"],
                                "decision": "plateau_recover",
                                "terminal": True,
                                "consecutive_terminal_rejections": 0,
                                "cutover_created_ns": cutover_created_ns,
                                "updated_ns": time.time_ns(),
                            },
                        )
                    self.metrics.append(
                        {
                            "schema_version": 1,
                            "timestamp_ns": time.time_ns(),
                            "worker": "learner",
                            "event": "plateau_recovery",
                            "reason": action.get("reset_reason"),
                            "from_step": previous_step,
                            "to_step": self.step,
                            "candidate_identity": action["candidate_identity"],
                            "champion_identity": action["champion_identity"],
                            "learning_rate_scale": (
                                configured.reset_learning_rate_scale
                            ),
                            "optimizer_state_cleared": (
                                configured.clear_optimizer_state_on_recovery
                            ),
                            "learning_rates": [
                                float(group["lr"])
                                for group in self.optimizer.param_groups
                            ],
                        }
                    )
                self._distributed_barrier()
                self._last_plateau_reset = (
                    str(action["champion_identity"]),
                    str(action["candidate_identity"]),
                )
                if progress is not None and self.rank == 0:
                    progress(
                        phase="plateau_recovery",
                        step=self.step,
                        champion_identity=action["champion_identity"],
                        candidate_identity=action["candidate_identity"],
                        reason=action.get("reset_reason"),
                    )
                return True
            if kind != "pause":
                raise RuntimeError("unknown plateau action")
            if self._collective_stop(stop_requested()):
                return False
            if progress is not None and self.rank == 0:
                progress(
                    phase="learner_plateau",
                    step=self.step,
                    reason=action.get("reason"),
                    champion_step=action.get("champion_step"),
                )
            time.sleep(configured.poll_seconds)

    def _gpu_pause_control(
        self,
        *,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
    ) -> bool:
        if self.gpu_pause_path is None:
            return True
        while True:
            active = self._rank_zero_gpu_pause_active() if self.rank == 0 else None
            active = self._broadcast_object(active)
            if active is False:
                return True
            if active is not True:
                raise RuntimeError("distributed GPU pause state is invalid")
            if self._collective_stop(stop_requested()):
                return False
            if progress is not None and self.rank == 0:
                progress(phase="arena_gpu_pause", step=self.step)
            time.sleep(self._plateau_config().poll_seconds)

    def _rank_zero_gpu_pause_active(self) -> bool:
        assert self.gpu_pause_path is not None
        try:
            with self.gpu_pause_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.gpu_pause_path.unlink(missing_ok=True)
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            self.gpu_pause_path.unlink(missing_ok=True)
            return False
        except PermissionError:
            return True
        return True

    def _plateau_config(self):
        # The typed object is retained separately from serialized checkpoint
        # configuration so resume cannot silently alter live plateau policy.
        from .config import PlateauConfig

        values = self.serialized_config.get("orchestration")
        if not isinstance(values, Mapping):
            return PlateauConfig()
        plateau = values.get("plateau")
        return (
            PlateauConfig(**plateau) if isinstance(plateau, dict) else PlateauConfig()
        )

    def _retention_config(self):
        from .config import RetentionConfig

        values = self.serialized_config.get("orchestration")
        if not isinstance(values, Mapping):
            return RetentionConfig()
        retention = values.get("retention")
        return (
            RetentionConfig(**retention)
            if isinstance(retention, dict)
            else RetentionConfig()
        )

    def _maybe_collect_replay_garbage(self) -> None:
        retention = self._retention_config()
        if not retention.enabled or self.epoch % retention.gc_interval_windows != 0:
            return
        metrics = self.store.collect_garbage(
            run_id=self.run_identity.run_id,
            generation_family=self.run_identity.generation_family,
            retain_shards_per_ring=retention.replay_shards_per_ring,
            dry_run=retention.dry_run,
        )
        self.metrics.append(
            {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "worker": "learner",
                "event": "replay_gc",
                **metrics,
            }
        )
        if self.selfplay_publisher is not None:
            selfplay_metrics = collect_model_garbage(
                self.selfplay_publisher.root,
                retain_candidate_manifests=retention.candidate_manifests,
                dry_run=retention.dry_run,
            )
            self.metrics.append(
                {
                    "schema_version": 1,
                    "timestamp_ns": time.time_ns(),
                    "worker": "learner",
                    "event": "selfplay_model_gc",
                    **selfplay_metrics,
                }
            )

    def _rank_zero_plateau_action(self, configured) -> dict[str, object]:
        if not self.publisher.champion_path.is_file():
            return {"kind": "proceed"}
        champion = load_model_manifest(self.publisher.champion_path)
        lag = self.step - champion.model_step
        if lag < configured.max_learner_champion_lag_steps:
            return {"kind": "proceed"}
        candidate = (
            load_model_manifest(self.publisher.candidate_path)
            if self.publisher.candidate_path.is_file()
            else None
        )
        status: dict[str, object] = {}
        if (
            self.promotion_status_path is not None
            and self.promotion_status_path.is_file()
        ):
            try:
                with self.promotion_status_path.open("r", encoding="utf-8") as stream:
                    loaded = json.load(stream)
                if isinstance(loaded, dict):
                    status = loaded
            except (OSError, json.JSONDecodeError):
                status = {}
        status_matches = (
            candidate is not None
            and status.get("candidate_identity") == candidate.model_identity
        )
        terminal_rejection = bool(status.get("terminal")) and status.get(
            "decision"
        ) in ("reject", "reject_ring_regression", "reject_max_pairs")
        raw_streak = status.get("consecutive_terminal_rejections", 0)
        streak = (
            raw_streak
            if isinstance(raw_streak, int) and not isinstance(raw_streak, bool)
            else 0
        )
        reset_token = (
            champion.model_identity,
            candidate.model_identity if candidate is not None else "",
        )
        decision = plateau_policy_decision(
            lag_steps=lag,
            soft_lag_steps=configured.max_learner_champion_lag_steps,
            hard_replay_lag_steps=self.learner_config.max_replay_lag_steps,
            status_matches_candidate=status_matches,
            terminal_rejection=terminal_rejection,
            rejection_streak=streak,
            reset_after_rejections=(configured.consecutive_terminal_rejections),
            action=configured.action,
            reset_already_applied=self._last_plateau_reset == reset_token,
        )
        if decision in ("reset", "recover"):
            if candidate is None:
                return {
                    "kind": "pause",
                    "reason": "awaiting_candidate",
                    "champion_step": champion.model_step,
                }
            recovery_kind = "reset" if decision == "reset" else "recover"
            return {
                "kind": recovery_kind,
                "reset_reason": (
                    "hard_replay_lag"
                    if lag >= self.learner_config.max_replay_lag_steps
                    and streak < configured.consecutive_terminal_rejections
                    else "terminal_rejection_streak"
                ),
                "champion_identity": champion.model_identity,
                "champion_step": champion.model_step,
                "candidate_identity": candidate.model_identity,
                "candidate_step": candidate.model_step,
                **(
                    {
                        "checkpoint": str(champion.checkpoint),
                        "sha256": champion.checkpoint_sha256,
                        "bytes": champion.checkpoint_bytes,
                    }
                    if decision == "reset"
                    else {}
                ),
            }
        if decision == "proceed":
            return {"kind": "proceed"}
        return {
            "kind": "pause",
            "reason": (
                "candidate_inconclusive"
                if status_matches and not bool(status.get("terminal"))
                else "awaiting_terminal_promotion"
            ),
            "champion_step": champion.model_step,
        }

    def _collective_stop(self, local_stop: bool) -> bool:
        stop, _ = self._collective_flags(local_stop, True)
        return stop

    def _collective_any(self, value: bool) -> bool:
        if self.world_size == 1:
            return value
        device = next(self.model.parameters()).device
        tensor = torch.tensor(int(value), device=device, dtype=torch.int32)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return bool(tensor.item())

    def _collective_min_int(self, value: int) -> int:
        if self.world_size == 1:
            return value
        device = next(self.model.parameters()).device
        tensor = torch.tensor(value, device=device, dtype=torch.int64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        return int(tensor.item())

    def _broadcast_object(self, value: object) -> object:
        if self.world_size == 1:
            return value
        payload = [value]
        torch.distributed.broadcast_object_list(
            payload,
            src=0,
            device=next(self.model.parameters()).device,
        )
        return payload[0]

    def _collective_flags(
        self, local_stop: bool, local_ready: bool
    ) -> tuple[bool, bool]:
        if self.world_size == 1:
            return local_stop, local_ready
        if not torch.distributed.is_initialized():
            raise RuntimeError("distributed learner process group is not initialized")
        device = next(self.model.parameters()).device
        values = torch.tensor(
            [int(local_stop), int(local_ready)],
            device=device,
            dtype=torch.int32,
        )
        stop_value = values[:1]
        ready_value = values[1:]
        torch.distributed.all_reduce(stop_value, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(ready_value, op=torch.distributed.ReduceOp.MIN)
        return bool(stop_value.item()), bool(ready_value.item())

    def _distributed_barrier(self) -> None:
        if self.world_size > 1:
            if not torch.distributed.is_initialized():
                raise RuntimeError(
                    "distributed learner process group is not initialized"
                )
            torch.distributed.barrier()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
