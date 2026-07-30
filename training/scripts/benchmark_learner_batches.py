#!/usr/bin/env python3
"""Benchmark learner batch sizes without mutating training artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sqlite3
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    sha256_file,
)
from startrain.config import ExperimentConfig, load_config
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH, RULES_HASH_WIRE
from startrain.device import (
    empty_device_cache,
    enable_fast_math,
    resolve_compile,
    resolve_device_string,
    resolve_pin_memory,
    resolve_precision,
    seed_all,
    synchronize_device,
)
from startrain.model import GraphResTNet
from startrain.optim import build_optimizer
from startrain.replay import (
    DecodedReplayShard,
    ReplayBatch,
    augment_sample,
    collate_replay_samples,
    decode_replay_shard,
)
from startrain.replay_store import (
    MANIFEST_SCHEMA_VERSION,
    ReplaySpan,
    ShardRecord,
)
from startrain.runtime import validate_identifier
from startrain.symmetry import deterministic_transform
from startrain.training import (
    HostTrainStepMetrics,
    build_scheduler,
    maybe_compile_model,
    train_step,
)

SCHEMA_VERSION = 1
BENCHMARK_NAME = "learner-batch-size-matrix"
DEFAULT_BATCH_SIZES = (512, 768, 1024)
BASELINE_BATCH_SIZE = 512
MINIMUM_END_TO_END_IMPROVEMENT = 0.15
MAXIMUM_PEAK_ALLOCATED_BYTES = 72 * 1024**3


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    config: Path
    checkpoint: Path
    replay_root: Path
    device: str | None = None
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    warmups: int = 2
    repeats: int = 5
    output: Path | None = None


@dataclass(frozen=True, slots=True)
class CheckpointDescriptor:
    step: int
    epoch: int
    run_id: str
    generation_family: str
    config: Mapping[str, object]
    optimizer_compatible: bool
    scheduler_compatible: bool


@dataclass(frozen=True, slots=True)
class PreparedReplay:
    ring: int
    spans: tuple[tuple[ReplaySpan, DecodedReplayShard], ...]
    references: tuple[tuple[DecodedReplayShard, int], ...]
    decode_seconds: tuple[float, ...]

    @property
    def sample_count(self) -> int:
        return len(self.references)

    @property
    def shard_ids(self) -> tuple[int, ...]:
        return tuple(span.record.shard_id for span, _decoded in self.spans)

    def materialize(
        self,
        batch_size: int,
        *,
        seed: int,
        augmentation_enabled: bool,
        pin_memory: bool,
    ) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self.sample_count:
            raise ValueError(
                f"ring {self.ring} has {self.sample_count} selected samples, "
                f"fewer than batch size {batch_size}"
            )
        first = self.sample_count - batch_size
        samples = []
        for offset, (decoded, index) in enumerate(self.references[first:]):
            sample = decoded.sample(index)
            if sample.rings != self.ring:
                raise ValueError("materialized replay batch is not ring-homogeneous")
            if augmentation_enabled:
                sample = augment_sample(
                    sample,
                    deterministic_transform(
                        seed=seed,
                        sample_index=first + offset,
                        epoch=0,
                    ),
                )
            samples.append(sample)
        batch = collate_replay_samples(samples)
        return batch.pin_memory() if pin_memory else batch


@dataclass(slots=True)
class CandidateState:
    model: GraphResTNet
    compiled_model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    ema: ExponentialMovingAverage
    optimizer_loaded: bool
    scheduler_loaded: bool
    optimizer_state_note: str | None
    scheduler_state_note: str | None


BatchRunner = Callable[[int], dict[str, object]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser


def _settings(arguments: argparse.Namespace) -> BenchmarkSettings:
    return BenchmarkSettings(
        config=arguments.config.expanduser().resolve(),
        checkpoint=arguments.checkpoint.expanduser().resolve(),
        replay_root=arguments.replay_root.expanduser().resolve(),
        device=arguments.device,
        batch_sizes=tuple(sorted(arguments.batch_sizes)),
        warmups=arguments.warmups,
        repeats=arguments.repeats,
        output=(
            arguments.output.expanduser().resolve()
            if arguments.output is not None
            else None
        ),
    )


def validate_settings(settings: BenchmarkSettings) -> None:
    if not settings.config.is_file():
        raise ValueError(f"config is not a file: {settings.config}")
    if not settings.checkpoint.is_file():
        raise ValueError(f"checkpoint is not a file: {settings.checkpoint}")
    if not settings.replay_root.is_dir():
        raise ValueError(f"replay root is not a directory: {settings.replay_root}")
    manifest = settings.replay_root / "manifest.sqlite3"
    if not manifest.is_file():
        raise ValueError(f"replay manifest is not a file: {manifest}")
    if (
        not settings.batch_sizes
        or any(type(size) is not int or size <= 0 for size in settings.batch_sizes)
        or len(set(settings.batch_sizes)) != len(settings.batch_sizes)
    ):
        raise ValueError("batch sizes must be unique positive integers")
    if settings.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if settings.repeats <= 0:
        raise ValueError("repeats must be positive")
    if settings.device is not None and not settings.device.strip():
        raise ValueError("device must not be empty")
    if settings.output is not None:
        _validate_output_destination(settings)


def _validate_output_destination(settings: BenchmarkSettings) -> None:
    assert settings.output is not None
    output = settings.output
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    if output in (settings.config, settings.checkpoint):
        raise ValueError("output cannot replace an input artifact")
    try:
        output.relative_to(settings.replay_root)
    except ValueError:
        pass
    else:
        raise ValueError("output cannot be written inside the replay root")


@contextmanager
def open_replay_manifest_read_only(
    replay_root: str | Path,
) -> Iterator[sqlite3.Connection]:
    """Open the replay manifest with SQLite-enforced read-only/query-only modes."""

    manifest = Path(replay_root).expanduser().resolve() / "manifest.sqlite3"
    if not manifest.is_file():
        raise ValueError(f"replay manifest is not a file: {manifest}")
    connection = sqlite3.connect(
        f"{manifest.as_uri()}?mode=ro",
        uri=True,
        timeout=30.0,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def _validate_manifest_metadata(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT key, value
        FROM store_metadata
        WHERE key IN (
            'manifest_schema_version',
            'rules_hash',
            'feature_schema_hash'
        )
        """
    )
    metadata = {str(row["key"]): str(row["value"]) for row in rows}
    expected = {
        "manifest_schema_version": str(MANIFEST_SCHEMA_VERSION),
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
    }
    if metadata != expected:
        raise ValueError("replay manifest metadata is incompatible")


def _record_from_row(root: Path, row: sqlite3.Row) -> ShardRecord:
    relative_path = Path(str(row["relative_path"]))
    if relative_path.is_absolute():
        raise ValueError("replay manifest contains an absolute shard path")
    resolved_root = root.resolve()
    path = (resolved_root / relative_path).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("replay shard path escapes the replay root") from error
    sample_count = int(row["sample_count"])
    if sample_count <= 0:
        raise ValueError("replay manifest contains a non-positive shard size")
    return ShardRecord(
        shard_id=int(row["id"]),
        path=path,
        created_ns=int(row["created_ns"]),
        sample_count=sample_count,
        ring=int(row["ring"]),
        phase_min=int(row["phase_min"]),
        phase_max=int(row["phase_max"]),
        model_version=str(row["model_version"]),
        model_step=int(row["model_step"]),
        model_identity=str(row["model_identity"]),
        run_id=str(row["run_id"]),
        generation_family=str(row["generation_family"]),
        actor_id=str(row["actor_id"]),
        generation=int(row["generation"]),
        game_count=int(row["game_count"]),
        checksum_sha256=str(row["checksum_sha256"]),
        state=str(row["state"]),
        quarantine_reason=(
            str(row["quarantine_reason"])
            if row["quarantine_reason"] is not None
            else None
        ),
    )


def _recent_records_for_ring(
    connection: sqlite3.Connection,
    *,
    replay_root: Path,
    ring: int,
    descriptor: CheckpointDescriptor,
    max_model_lag_steps: int,
    target_samples: int,
) -> list[ShardRecord]:
    rows = connection.execute(
        """
        SELECT
            id, relative_path, created_ns, sample_count, ring,
            phase_min, phase_max, model_version, model_step,
            model_identity, run_id, generation_family, actor_id,
            generation, game_count, checksum_sha256, state,
            quarantine_reason
        FROM shards
        WHERE state = 'ready'
          AND rules_hash = ?
          AND feature_schema_hash = ?
          AND run_id = ?
          AND generation_family = ?
          AND ring = ?
          AND model_step BETWEEN ? AND ?
        ORDER BY id DESC
        """,
        (
            f"{RULES_HASH:016x}",
            f"{FEATURE_SCHEMA_HASH:016x}",
            descriptor.run_id,
            descriptor.generation_family,
            ring,
            max(0, descriptor.step - max_model_lag_steps),
            descriptor.step,
        ),
    )
    records: list[ShardRecord] = []
    samples = 0
    for row in rows:
        record = _record_from_row(replay_root, row)
        records.append(record)
        samples += record.sample_count
        if samples >= target_samples:
            break
    return records


def select_recent_ready_spans(
    replay_root: str | Path,
    *,
    config: ExperimentConfig,
    descriptor: CheckpointDescriptor,
    minimum_samples: int,
    target_samples: int,
) -> tuple[int, tuple[ReplaySpan, ...]]:
    """Select recent eligible spans without instantiating the writable ReplayStore."""

    root = Path(replay_root).expanduser().resolve()
    records_by_ring: dict[int, list[ShardRecord]] = {}
    with open_replay_manifest_read_only(root) as connection:
        _validate_manifest_metadata(connection)
        for ring in sorted(config.game.rings, reverse=True):
            records_by_ring[ring] = _recent_records_for_ring(
                connection,
                replay_root=root,
                ring=ring,
                descriptor=descriptor,
                max_model_lag_steps=config.learner.max_replay_lag_steps,
                target_samples=target_samples,
            )

    capacities = {
        ring: sum(record.sample_count for record in records)
        for ring, records in records_by_ring.items()
    }
    full_capacity = [
        ring
        for ring in sorted(capacities, reverse=True)
        if capacities[ring] >= target_samples
    ]
    partial_capacity = [
        ring
        for ring in sorted(capacities, reverse=True)
        if capacities[ring] >= minimum_samples
    ]
    choices = full_capacity or partial_capacity
    if not choices:
        detail = ", ".join(
            f"ring {ring}: {capacities[ring]}" for ring in sorted(capacities)
        )
        raise ValueError(
            f"no eligible ring has {minimum_samples} recent samples ({detail})"
        )
    ring = choices[0]
    remaining = min(target_samples, capacities[ring])
    newest_first: list[ReplaySpan] = []
    for record in records_by_ring[ring]:
        take = min(record.sample_count, remaining)
        if take <= 0:
            break
        newest_first.append(
            ReplaySpan(
                record=record,
                sample_start=record.sample_count - take,
                sample_count=take,
            )
        )
        remaining -= take
    return ring, tuple(reversed(newest_first))


def prepare_replay(
    replay_root: str | Path,
    *,
    config: ExperimentConfig,
    descriptor: CheckpointDescriptor,
    minimum_samples: int,
    target_samples: int,
) -> PreparedReplay:
    ring, spans = select_recent_ready_spans(
        replay_root,
        config=config,
        descriptor=descriptor,
        minimum_samples=minimum_samples,
        target_samples=target_samples,
    )
    prepared: list[tuple[ReplaySpan, DecodedReplayShard]] = []
    references: list[tuple[DecodedReplayShard, int]] = []
    decode_seconds: list[float] = []
    for span in spans:
        record = span.record
        if not record.path.is_file():
            raise ValueError(f"ready replay shard is missing: {record.path}")
        if sha256_file(record.path) != record.checksum_sha256:
            raise ValueError(f"replay shard checksum failed: {record.path}")
        started = time.perf_counter()
        decoded = decode_replay_shard(record.path)
        decode_seconds.append(time.perf_counter() - started)
        if len(decoded) != record.sample_count:
            raise ValueError("replay shard count disagrees with its manifest")
        decoded_ring = int(decoded.arrays["rings"][0])
        if decoded_ring != ring or record.ring != ring:
            raise ValueError("replay shard ring disagrees with its manifest")
        prepared.append((span, decoded))
        references.extend(
            (decoded, index)
            for index in range(
                span.sample_start,
                span.sample_start + span.sample_count,
            )
        )
    return PreparedReplay(
        ring=ring,
        spans=tuple(prepared),
        references=tuple(references),
        decode_seconds=tuple(decode_seconds),
    )


def _mapping_section(
    config: Mapping[str, object],
    name: str,
) -> Mapping[str, object] | None:
    value = config.get(name)
    return value if isinstance(value, Mapping) else None


def _scheduler_section(
    config: Mapping[str, object],
) -> Mapping[str, object] | None:
    train = _mapping_section(config, "train")
    if train is None:
        return None
    scheduler = train.get("scheduler")
    return scheduler if isinstance(scheduler, Mapping) else None


def verify_checkpoint(
    checkpoint: str | Path,
    config: ExperimentConfig,
) -> CheckpointDescriptor:
    """Validate checkpoint contracts and load both raw model and EMA state."""

    serialized = config.as_dict()
    model = GraphResTNet(config.model)
    ema = ExponentialMovingAverage(model, decay=config.train.ema_decay)
    metadata = load_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        map_location="cpu",
        require_ema=True,
        expected_model_config=serialized["model"],
        expected_game_config=serialized["game"],
    )
    checkpoint_config = metadata.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("checkpoint configuration is missing")
    if checkpoint_config.get("schema_version") != config.schema_version:
        raise ValueError("checkpoint configuration schema is incompatible")
    checkpoint_loss = _mapping_section(checkpoint_config, "loss")
    if checkpoint_loss is None or dict(checkpoint_loss) != serialized["loss"]:
        raise ValueError("checkpoint loss configuration is incompatible")
    checkpoint_train = _mapping_section(checkpoint_config, "train")
    expected_train = serialized["train"]
    if checkpoint_train is None or not isinstance(expected_train, Mapping):
        raise ValueError("checkpoint train configuration is missing")
    for key in ("precision", "compile", "ema_decay", "gradient_clip_norm"):
        if checkpoint_train.get(key) != expected_train.get(key):
            raise ValueError(f"checkpoint train.{key} configuration is incompatible")
    extra = metadata.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError("checkpoint extra metadata is missing")
    run_id = validate_identifier("run_id", extra.get("run_id"))
    generation_family = validate_identifier(
        "generation_family",
        extra.get("generation_family"),
    )
    expected_optimizer = serialized["optimizer"]
    optimizer_compatible = (
        isinstance(expected_optimizer, Mapping)
        and _mapping_section(checkpoint_config, "optimizer") == expected_optimizer
    )
    scheduler_compatible = optimizer_compatible and _scheduler_section(
        checkpoint_config
    ) == _scheduler_section(serialized)
    return CheckpointDescriptor(
        step=int(metadata["step"]),
        epoch=int(metadata["epoch"]),
        run_id=run_id,
        generation_family=generation_family,
        config=dict(checkpoint_config),
        optimizer_compatible=optimizer_compatible,
        scheduler_compatible=scheduler_compatible,
    )


def _is_cuda_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return (
        isinstance(error, RuntimeError)
        and "cuda" in message
        and "out of memory" in message
    )


def _load_candidate_state(
    *,
    checkpoint: Path,
    config: ExperimentConfig,
    descriptor: CheckpointDescriptor,
    device: torch.device,
    compile_enabled: bool,
) -> CandidateState:
    serialized = config.as_dict()
    seed_all(config.train.seed)
    model = GraphResTNet(config.model).to(device)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.ema_decay)
    load_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        map_location=device,
        require_ema=True,
        expected_model_config=serialized["model"],
        expected_game_config=serialized["game"],
    )

    optimizer_loaded = False
    optimizer_note: str | None = None
    if descriptor.optimizer_compatible:
        try:
            load_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                map_location=device,
                require_ema=True,
                expected_model_config=serialized["model"],
                expected_game_config=serialized["game"],
            )
            optimizer_loaded = True
        except (RuntimeError, TypeError, ValueError) as error:
            if _is_cuda_oom(error):
                raise
            optimizer = build_optimizer(model, config.optimizer)
            scheduler = build_scheduler(optimizer, config.train.scheduler)
            optimizer_note = f"{type(error).__name__}: {error}"
    else:
        optimizer_note = "checkpoint optimizer configuration differs"

    scheduler_loaded = False
    scheduler_note: str | None = None
    if descriptor.scheduler_compatible and optimizer_loaded:
        try:
            load_checkpoint(
                checkpoint,
                model=model,
                scheduler=scheduler,
                map_location=device,
                require_ema=True,
                expected_model_config=serialized["model"],
                expected_game_config=serialized["game"],
            )
            scheduler_loaded = True
        except (RuntimeError, TypeError, ValueError) as error:
            if _is_cuda_oom(error):
                raise
            scheduler = build_scheduler(optimizer, config.train.scheduler)
            scheduler_note = f"{type(error).__name__}: {error}"
    elif not descriptor.scheduler_compatible:
        scheduler_note = "checkpoint scheduler configuration differs"
    elif not optimizer_loaded:
        scheduler_note = "optimizer state was not compatible"

    compiled_model = maybe_compile_model(
        model,
        enabled=compile_enabled,
        dynamic=False,
        recompile_limit=len(config.game.rings),
        isolate_recompiles=True,
    )
    compiled_model.train()
    return CandidateState(
        model=model,
        compiled_model=compiled_model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        optimizer_loaded=optimizer_loaded,
        scheduler_loaded=scheduler_loaded,
        optimizer_state_note=optimizer_note,
        scheduler_state_note=scheduler_note,
    )


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty measurements")
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def _empty_batch_result(
    batch_size: int,
    *,
    ring: int | None,
    warmups: int,
    repeats: int,
    precision: str,
    compile_enabled: bool,
) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "status": "error",
        "error": None,
        "ring": ring,
        "warmups_requested": warmups,
        "warmups_run": 0,
        "repeats": repeats,
        "settings": {
            "precision": precision,
            "compile": compile_enabled,
        },
        "checkpoint_state": {
            "state_reloaded": False,
            "model_loaded": False,
            "ema_loaded": False,
            "optimizer_loaded": False,
            "scheduler_loaded": False,
            "optimizer_state_note": None,
            "scheduler_state_note": None,
        },
        "timing_seconds": {
            "materialization": None,
            "end_to_end": None,
            "cuda_device": None,
        },
        "throughput": {
            "end_to_end_samples_per_second": None,
            "cuda_device_samples_per_second": None,
        },
        "memory_bytes": {
            "peak_allocated": None,
            "peak_reserved": None,
        },
        "numerics": {
            "loss_finite": None,
            "gradient_finite": None,
            "losses": None,
            "gradient_norm": None,
        },
    }


def _host_metrics_are_finite(metrics: HostTrainStepMetrics) -> tuple[bool, bool]:
    return (
        all(math.isfinite(value) for value in metrics.losses.values()),
        math.isfinite(metrics.gradient_norm),
    )


def _execute_candidate(
    batch_size: int,
    *,
    checkpoint: Path,
    config: ExperimentConfig,
    descriptor: CheckpointDescriptor,
    replay: PreparedReplay,
    device: torch.device,
    precision: str,
    compile_enabled: bool,
    pin_memory: bool,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    state = _load_candidate_state(
        checkpoint=checkpoint,
        config=config,
        descriptor=descriptor,
        device=device,
        compile_enabled=compile_enabled,
    )
    warmups_run = max(warmups, int(compile_enabled))
    for _ in range(warmups_run):
        batch = replay.materialize(
            batch_size,
            seed=config.train.seed,
            augmentation_enabled=config.data.d5_augmentation,
            pin_memory=pin_memory,
        )
        warmup_result = train_step(
            state.compiled_model,
            batch,
            state.optimizer,
            loss_weights=config.loss,
            precision=precision,
            gradient_clip_norm=config.train.gradient_clip_norm,
            scheduler=state.scheduler,
            ema=state.ema,
            trusted_batch=True,
        )
        warmup_result.to_host()
        del batch, warmup_result

    synchronize_device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    end_to_end_seconds: list[float] = []
    materialization_seconds: list[float] = []
    cuda_device_seconds: list[float] = []
    host_metrics: list[HostTrainStepMetrics] = []
    for _ in range(repeats):
        started = time.perf_counter()
        materialize_started = time.perf_counter()
        batch = replay.materialize(
            batch_size,
            seed=config.train.seed,
            augmentation_enabled=config.data.d5_augmentation,
            pin_memory=pin_memory,
        )
        materialization_seconds.append(time.perf_counter() - materialize_started)
        events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        if device.type == "cuda":
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            events[0].record(torch.cuda.current_stream(device))
        result = train_step(
            state.compiled_model,
            batch,
            state.optimizer,
            loss_weights=config.loss,
            precision=precision,
            gradient_clip_norm=config.train.gradient_clip_norm,
            scheduler=state.scheduler,
            ema=state.ema,
            trusted_batch=True,
        )
        if events is not None:
            events[1].record(torch.cuda.current_stream(device))
        metrics = result.to_host()
        synchronize_device(device)
        end_to_end_seconds.append(time.perf_counter() - started)
        if events is not None:
            cuda_device_seconds.append(events[0].elapsed_time(events[1]) / 1_000.0)
        host_metrics.append(metrics)
        del batch, result

    loss_finite, gradient_finite = zip(
        *(_host_metrics_are_finite(metrics) for metrics in host_metrics),
        strict=True,
    )
    loss_names = tuple(host_metrics[0].losses)
    mean_losses = {
        name: statistics.fmean(metrics.losses[name] for metrics in host_metrics)
        for name in loss_names
    }
    gradient_norm = statistics.fmean(metrics.gradient_norm for metrics in host_metrics)
    all_losses_finite = all(loss_finite)
    all_gradients_finite = all(gradient_finite)
    end_to_end = _stats(end_to_end_seconds)
    cuda_device = _stats(cuda_device_seconds) if cuda_device_seconds else None
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )
    return {
        "batch_size": batch_size,
        "status": ("ok" if all_losses_finite and all_gradients_finite else "error"),
        "error": (
            None
            if all_losses_finite and all_gradients_finite
            else {
                "type": "FloatingPointError",
                "message": "non-finite loss or gradient was measured",
            }
        ),
        "ring": replay.ring,
        "warmups_requested": warmups,
        "warmups_run": warmups_run,
        "repeats": repeats,
        "settings": {
            "precision": precision,
            "compile": compile_enabled,
        },
        "checkpoint_state": {
            "state_reloaded": True,
            "model_loaded": True,
            "ema_loaded": True,
            "optimizer_loaded": state.optimizer_loaded,
            "scheduler_loaded": state.scheduler_loaded,
            "optimizer_state_note": state.optimizer_state_note,
            "scheduler_state_note": state.scheduler_state_note,
        },
        "timing_seconds": {
            "materialization": _stats(materialization_seconds),
            "end_to_end": end_to_end,
            "cuda_device": cuda_device,
        },
        "throughput": {
            "end_to_end_samples_per_second": (batch_size / float(end_to_end["median"])),
            "cuda_device_samples_per_second": (
                batch_size / float(cuda_device["median"])
                if cuda_device is not None
                else None
            ),
        },
        "memory_bytes": {
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
        },
        "numerics": {
            "loss_finite": all_losses_finite,
            "gradient_finite": all_gradients_finite,
            "losses": mean_losses,
            "gradient_norm": gradient_norm,
        },
    }


def _release_candidate_memory(device: torch.device | None) -> None:
    gc.collect()
    if device is None:
        return
    try:
        empty_device_cache(device)
    except RuntimeError:
        pass


def run_batch_matrix(
    batch_sizes: Sequence[int],
    runner: BatchRunner,
    *,
    ring: int | None = None,
    warmups: int = 0,
    repeats: int = 1,
    precision: str = "unknown",
    compile_enabled: bool = False,
    cleanup_device: torch.device | None = None,
) -> list[dict[str, object]]:
    """Run each batch independently so OOMs and errors cannot abort the matrix."""

    results: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        try:
            result = runner(batch_size)
            if result.get("batch_size") != batch_size:
                raise ValueError("batch runner returned the wrong batch size")
            results.append(result)
        except Exception as error:
            result = _empty_batch_result(
                batch_size,
                ring=ring,
                warmups=warmups,
                repeats=repeats,
                precision=precision,
                compile_enabled=compile_enabled,
            )
            oom = _is_cuda_oom(error)
            result["status"] = "oom" if oom else "error"
            result["error"] = {
                "type": ("CUDAOutOfMemoryError" if oom else type(error).__name__),
                "message": str(error),
            }
            if isinstance(error, FloatingPointError):
                result["numerics"] = {
                    "loss_finite": False,
                    "gradient_finite": False,
                    "losses": None,
                    "gradient_norm": None,
                }
            results.append(result)
        finally:
            _release_candidate_memory(cleanup_device)
    return results


def _successful_throughput(result: Mapping[str, object]) -> float | None:
    if result.get("status") != "ok":
        return None
    throughput = result.get("throughput")
    if not isinstance(throughput, Mapping):
        return None
    value = throughput.get("end_to_end_samples_per_second")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def _peak_allocated(result: Mapping[str, object]) -> int | None:
    memory = result.get("memory_bytes")
    if not isinstance(memory, Mapping):
        return None
    value = memory.get("peak_allocated")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def recommend_batch(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the fixed throughput/HBM gate, choosing the smallest winner."""

    rule = {
        "baseline_batch_size": BASELINE_BATCH_SIZE,
        "minimum_end_to_end_improvement_fraction": (MINIMUM_END_TO_END_IMPROVEMENT),
        "maximum_peak_allocated_bytes": MAXIMUM_PEAK_ALLOCATED_BYTES,
        "prefer_smallest_passing_larger_batch": True,
    }
    baseline = next(
        (
            result
            for result in results
            if result.get("batch_size") == BASELINE_BATCH_SIZE
        ),
        None,
    )
    baseline_throughput = (
        _successful_throughput(baseline) if baseline is not None else None
    )
    if baseline_throughput is None:
        return {
            "status": "unavailable",
            "selected_batch_size": None,
            "baseline_end_to_end_samples_per_second": None,
            "passing_larger_batch_sizes": [],
            "rule": rule,
            "reason": "batch 512 did not produce a successful finite measurement",
        }

    sized_results: list[tuple[int, Mapping[str, object]]] = []
    for result in results:
        value = result.get("batch_size")
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        sized_results.append((value, result))
    passing: list[int] = []
    for batch_size, result in sorted(sized_results, key=lambda item: item[0]):
        if batch_size <= BASELINE_BATCH_SIZE:
            continue
        throughput = _successful_throughput(result)
        peak = _peak_allocated(result)
        if throughput is None or peak is None:
            continue
        improvement = throughput / baseline_throughput - 1.0
        if (
            improvement + 1e-12 >= MINIMUM_END_TO_END_IMPROVEMENT
            and peak <= MAXIMUM_PEAK_ALLOCATED_BYTES
        ):
            passing.append(batch_size)
    if passing:
        selected = passing[0]
        return {
            "status": "larger_batch_selected",
            "selected_batch_size": selected,
            "baseline_end_to_end_samples_per_second": baseline_throughput,
            "passing_larger_batch_sizes": passing,
            "rule": rule,
            "reason": (
                "selected the smallest larger batch passing both throughput "
                "and peak-allocation gates"
            ),
        }
    return {
        "status": "keep_baseline",
        "selected_batch_size": BASELINE_BATCH_SIZE,
        "baseline_end_to_end_samples_per_second": baseline_throughput,
        "passing_larger_batch_sizes": [],
        "rule": rule,
        "reason": "no larger batch passed both throughput and peak-allocation gates",
    }


def benchmark_learner_batches(settings: BenchmarkSettings) -> dict[str, object]:
    validate_settings(settings)
    config = load_config(settings.config)
    requested_device = settings.device or config.learner.device
    resolved_device = resolve_device_string(requested_device)
    device = torch.device(resolved_device)
    precision = resolve_precision(config.train.precision, device)
    compile_enabled = resolve_compile(config.train.compile, device)
    pin_memory = resolve_pin_memory(config.data.pin_memory, device)
    enable_fast_math(device)

    descriptor = verify_checkpoint(settings.checkpoint, config)
    replay = prepare_replay(
        settings.replay_root,
        config=config,
        descriptor=descriptor,
        minimum_samples=min(settings.batch_sizes),
        target_samples=max(settings.batch_sizes),
    )

    def runner(batch_size: int) -> dict[str, object]:
        return _execute_candidate(
            batch_size,
            checkpoint=settings.checkpoint,
            config=config,
            descriptor=descriptor,
            replay=replay,
            device=device,
            precision=precision,
            compile_enabled=compile_enabled,
            pin_memory=pin_memory,
            warmups=settings.warmups,
            repeats=settings.repeats,
        )

    results = run_batch_matrix(
        settings.batch_sizes,
        runner,
        ring=replay.ring,
        warmups=settings.warmups,
        repeats=settings.repeats,
        precision=precision,
        compile_enabled=compile_enabled,
        cleanup_device=device,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": "ok",
        "error": None,
        "inputs": {
            "config": str(settings.config),
            "checkpoint": str(settings.checkpoint),
            "replay_root": str(settings.replay_root),
            "device_requested": requested_device,
            "device_resolved": str(device),
            "batch_sizes": list(settings.batch_sizes),
            "warmups": settings.warmups,
            "repeats": settings.repeats,
            "output": str(settings.output) if settings.output is not None else None,
        },
        "checkpoint": {
            "step": descriptor.step,
            "epoch": descriptor.epoch,
            "run_id": descriptor.run_id,
            "generation_family": descriptor.generation_family,
            "model_loaded_and_verified": True,
            "ema_loaded_and_verified": True,
            "optimizer_configuration_compatible": (descriptor.optimizer_compatible),
            "scheduler_configuration_compatible": (descriptor.scheduler_compatible),
        },
        "replay": {
            "manifest_open_mode": "ro",
            "query_only": True,
            "ring": replay.ring,
            "selected_samples": replay.sample_count,
            "selected_shard_ids": list(replay.shard_ids),
            "decoded_shards": len(replay.spans),
            "decode_seconds": _stats(replay.decode_seconds),
        },
        "settings": {
            "precision": precision,
            "compile": compile_enabled,
            "pin_memory": pin_memory,
            "d5_augmentation": config.data.d5_augmentation,
            "compile_measurements_excluded_by_warmup": True,
        },
        "batches": results,
        "recommendation": recommend_batch(results),
    }


def _error_payload(error: BaseException) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": "error",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "inputs": None,
        "checkpoint": None,
        "replay": None,
        "settings": None,
        "batches": [],
        "recommendation": recommend_batch([]),
    }


def _json_text(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    settings = _settings(arguments)
    try:
        payload = benchmark_learner_batches(settings)
        text = _json_text(payload)
        if settings.output is not None:
            _write_output(settings.output, text)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(_json_text(_error_payload(error)))
        return 2
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
