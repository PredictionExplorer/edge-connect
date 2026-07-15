"""Long-lived, batch-boundary-refreshing self-play actor supervisor."""

from __future__ import annotations

import json
import random
import statistics
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import torch

from .checkpoint import ModelManifest, load_ema_checkpoint, load_model_manifest
from .config import ExperimentConfig, GPUWorkerConfig, RingMixtureConfig
from .inference import GraphInferenceAdapter, InferenceConfig
from .model import GraphResTNet
from .replay_store import ReplayStore
from .runtime import HeartbeatReporter, RunIdentity, append_jsonl
from .selfplay import SelfPlayActor, SelfPlayIdentity, SelfPlayMetrics
from .training import maybe_compile_model


class RingMixtureScheduler:
    def __init__(self, config: RingMixtureConfig, *, seed: int) -> None:
        self.config = config
        self.random = random.Random(seed)

    def choose(self, sample_counts: Mapping[int, int], *, learner_step: int = 0) -> int:
        counts = {ring: int(sample_counts.get(ring, 0)) for ring in self.config.rings}
        if any(value < 0 for value in counts.values()):
            raise ValueError("ring sample counts must be non-negative")
        step_weights = self.config.weights_for_step(learner_step)
        if step_weights is not None:
            return self.random.choices(
                self.config.rings,
                weights=step_weights,
                k=1,
            )[0]
        total = sum(counts.values())
        eligible = self.config.active_rings(total)
        target = max((counts[ring] for ring in eligible), default=0)
        weights = []
        for ring in eligible:
            index = self.config.rings.index(ring)
            deficit = (target - counts[ring]) / target if target > 0 else 0.0
            weights.append(
                self.config.uniform_weight
                + self.config.deficit_weights[index] * deficit
            )
        return self.random.choices(eligible, weights=weights, k=1)[0]


class ManifestModelProvider:
    """Owns one immutable evaluator and swaps it only on explicit refresh calls."""

    def __init__(
        self,
        config: ExperimentConfig,
        manifest_path: str | Path,
        *,
        device: str,
        run_identity: RunIdentity,
        expected_role: Literal["champion", "candidate", "direct"] = "champion",
    ) -> None:
        self.config = config
        self.manifest_path = Path(manifest_path)
        self.device = device
        self.run_identity = run_identity
        self.expected_role = expected_role
        self.manifest: ModelManifest | None = None
        self.evaluator: GraphInferenceAdapter | None = None
        self._pointer_signature: tuple[int, int, int] | None = None

    def wait_for_initial(
        self,
        *,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None = None,
    ) -> GraphInferenceAdapter | None:
        refresh = self.config.orchestration.model_refresh
        started = time.monotonic()
        while not stop_requested():
            if self.manifest_path.is_file():
                return self.refresh()
            if time.monotonic() - started >= refresh.startup_timeout_seconds:
                raise TimeoutError(
                    f"actor timed out waiting for {self.expected_role}.json"
                )
            if progress is not None:
                progress(phase=f"waiting_for_{self.expected_role}")
            time.sleep(refresh.manifest_poll_seconds)
        return None

    def refresh(self) -> GraphInferenceAdapter:
        stat = self.manifest_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if self._pointer_signature == signature and self.evaluator is not None:
            return self.evaluator
        manifest = load_model_manifest(self.manifest_path)
        if manifest.role != self.expected_role:
            raise ValueError(f"self-play expected a {self.expected_role} model pointer")
        if (
            self.manifest is not None
            and manifest.model_version == self.manifest.model_version
            and manifest.model_step == self.manifest.model_step
            and manifest.checkpoint == self.manifest.checkpoint
        ):
            assert self.evaluator is not None
            self._pointer_signature = signature
            return self.evaluator
        model = GraphResTNet(self.config.model).to(self.device)
        metadata = load_ema_checkpoint(
            manifest.checkpoint,
            model=model,
            expected_model_config=asdict(self.config.model),
            expected_game_config=asdict(self.config.game),
            map_location=self.device,
            expected_run_id=self.run_identity.run_id,
            expected_generation_family=self.run_identity.generation_family,
            expected_sha256=manifest.checkpoint_sha256,
            expected_bytes=manifest.checkpoint_bytes,
        )
        if (
            int(metadata["step"]) != manifest.model_step
            or manifest.run_id != self.run_identity.run_id
            or manifest.generation_family != self.run_identity.generation_family
        ):
            raise ValueError("model manifest and checkpoint identity disagree")
        model.eval()
        inference_model = maybe_compile_model(
            model,
            enabled=self.config.train.compile,
            dynamic=True,
            fullgraph=True,
        )
        self.evaluator = GraphInferenceAdapter(
            inference_model,
            device=self.device,
            config=InferenceConfig(
                precision=self.config.train.precision,
                score_utility_weight=self.config.selfplay.score_utility_weight,
            ),
            model_version=manifest.model_version,
            model_step=manifest.model_step,
            model_identity=manifest.model_identity,
        )
        self.manifest = manifest
        self._pointer_signature = signature
        return self.evaluator


class HistoricalModelPool:
    """Selects only immutable checkpoints produced by the active self-play run."""

    def __init__(
        self,
        config: ExperimentConfig,
        manifest_directory: str | Path,
        *,
        device: str,
        run_identity: RunIdentity,
        pool_size: int,
        evaluator_cache_size: int = 2,
        additional_manifest_directories: Sequence[str | Path] = (),
    ) -> None:
        if pool_size <= 0 or evaluator_cache_size <= 0:
            raise ValueError("historical model pool sizes must be positive")
        self.config = config
        self.manifest_directories = tuple(
            dict.fromkeys(
                (
                    Path(manifest_directory),
                    *(Path(path) for path in additional_manifest_directories),
                )
            )
        )
        self.device = device
        self.run_identity = run_identity
        self.pool_size = pool_size
        self.evaluator_cache_size = min(pool_size, evaluator_cache_size)
        self.providers: OrderedDict[str, ManifestModelProvider] = OrderedDict()

    def select(
        self,
        *,
        random_source: random.Random,
        exclude: set[str],
    ) -> ManifestModelProvider | None:
        by_identity: dict[str, ModelManifest] = {}
        for directory in self.manifest_directories:
            for path in sorted(directory.glob("manifest-*.json")):
                try:
                    manifest = load_model_manifest(path)
                except (OSError, ValueError):
                    continue
                if (
                    manifest.role == "direct"
                    and manifest.run_id == self.run_identity.run_id
                    and manifest.generation_family
                    == self.run_identity.generation_family
                    and manifest.model_identity not in exclude
                ):
                    by_identity[manifest.model_identity] = manifest
        manifests = list(by_identity.values())
        manifests.sort(key=lambda item: (item.model_step, item.model_identity))
        if not manifests:
            return None
        candidates = self._spaced_candidates(manifests)
        selected = random_source.choice(candidates)
        provider = self.providers.pop(selected.model_identity, None)
        if provider is None:
            source = selected.artifact_manifest or selected.path
            provider = ManifestModelProvider(
                self.config,
                source,
                device=self.device,
                run_identity=self.run_identity,
                expected_role="direct",
            )
        self.providers[selected.model_identity] = provider
        while len(self.providers) > self.evaluator_cache_size:
            self.providers.popitem(last=False)
        return provider

    def _spaced_candidates(
        self,
        manifests: list[ModelManifest],
    ) -> list[ModelManifest]:
        if len(manifests) <= self.pool_size:
            return manifests
        if self.pool_size == 1:
            return [manifests[0]]
        last = len(manifests) - 1
        indices = {
            round(offset * last / (self.pool_size - 1))
            for offset in range(self.pool_size)
        }
        return [manifests[index] for index in sorted(indices)]


class ActorSupervisor:
    def __init__(
        self,
        *,
        native_module: object,
        experiment: ExperimentConfig,
        gpu: GPUWorkerConfig,
        replay_directory: str | Path,
        manifest_path: str | Path,
        candidate_manifest_path: str | Path,
        run_identity: RunIdentity,
        heartbeat_path: str | Path,
        metrics_path: str | Path,
        learner_heartbeat_path: str | Path | None = None,
        device: str = "cuda",
        lane_id: int = 0,
    ) -> None:
        if gpu.role != "actor" or gpu.actor_batch_size is None:
            raise ValueError("actor supervisor requires an actor GPU assignment")
        if (
            isinstance(lane_id, bool)
            or not isinstance(lane_id, int)
            or lane_id < 0
            or lane_id >= gpu.actor_lanes
        ):
            raise ValueError("actor lane_id is outside the configured lane range")
        self.native = native_module
        self.experiment = experiment
        self.gpu = gpu
        self.replay_directory = Path(replay_directory)
        self.run_identity = run_identity
        self.lane_id = lane_id
        self.actor_id = (
            f"actor-gpu-{gpu.gpu_id}"
            if gpu.actor_lanes == 1
            else f"actor-gpu-{gpu.gpu_id}-lane-{lane_id}"
        )
        self.device = torch.device(device)
        self.candidate_manifest_path = Path(candidate_manifest_path)
        self._candidate_manifest: ModelManifest | None = None
        self._candidate_signature: tuple[int, int, int] | None = None
        self.provider = ManifestModelProvider(
            experiment,
            manifest_path,
            device=device,
            run_identity=run_identity,
            expected_role="champion",
        )
        source = experiment.orchestration.model_refresh.selfplay_source
        self.candidate_provider = (
            ManifestModelProvider(
                experiment,
                candidate_manifest_path,
                device=device,
                run_identity=run_identity,
                expected_role="candidate",
            )
            if source != "champion"
            else None
        )
        self.history_pool = (
            HistoricalModelPool(
                experiment,
                self.candidate_manifest_path.parent / "manifests",
                device=device,
                run_identity=run_identity,
                pool_size=experiment.orchestration.model_refresh.history_pool_size,
                additional_manifest_directories=(
                    self.candidate_manifest_path.parent.parent / "manifests",
                ),
            )
            if source == "candidate_champion_history_mix"
            else None
        )
        self.model_random = random.Random(
            experiment.selfplay.seed
            + gpu.gpu_id * 1_000_003
            + lane_id * 104_729
            + 0x5E1F
        )
        self.heartbeat = HeartbeatReporter(
            heartbeat_path,
            worker=self.actor_id,
            interval_seconds=experiment.orchestration.shutdown.heartbeat_interval_seconds,
        )
        self.metrics_path = Path(metrics_path)
        self.learner_heartbeat_path = (
            Path(learner_heartbeat_path)
            if learner_heartbeat_path is not None
            else run_identity.path.parent / "status" / "learner.heartbeat.json"
        )
        self.scheduler = RingMixtureScheduler(
            experiment.orchestration.ring_mixture,
            seed=(
                experiment.selfplay.seed + gpu.gpu_id * 1_000_003 + lane_id * 104_729
            ),
        )

    def run(self, *, stop_requested: Callable[[], bool]) -> int:
        batches = 0
        self.heartbeat.start()
        final_phase = "stopped"
        try:
            evaluator = self.provider.wait_for_initial(
                stop_requested=stop_requested,
                progress=self.heartbeat.advance,
            )
            if evaluator is None:
                return batches
            if self.candidate_provider is not None:
                candidate_evaluator = self.candidate_provider.wait_for_initial(
                    stop_requested=stop_requested,
                    progress=self.heartbeat.advance,
                )
                if candidate_evaluator is None:
                    return batches
            with ReplayStore(self.replay_directory) as store:
                if any(store.reconciliation_metrics.values()):
                    self.heartbeat.advance(
                        phase="replay_reconciliation",
                        **store.reconciliation_metrics,
                    )
                store.register_run(self.run_identity)
                while not stop_requested():
                    # This is the sole model refresh point. The evaluator object is
                    # never mutated while SelfPlayActor owns active games.
                    model_role, provider = self._select_model_provider()
                    self._reset_peak_cuda_memory()
                    refresh_started = time.perf_counter()
                    evaluator = provider.refresh()
                    model_refresh_latency_seconds = (
                        time.perf_counter() - refresh_started
                    )
                    candidate = self._read_candidate()
                    if (
                        candidate.run_id != self.run_identity.run_id
                        or candidate.generation_family
                        != self.run_identity.generation_family
                    ):
                        raise ValueError(
                            "candidate manifest does not belong to the active run"
                        )
                    lag = candidate.model_step - evaluator.model_step
                    plateau = self.experiment.orchestration.plateau
                    if (
                        model_role == "champion"
                        and plateau.enabled
                        and lag > plateau.max_learner_champion_lag_steps
                    ):
                        self.heartbeat.advance(
                            phase="champion_selfplay_plateau",
                            candidate_step=candidate.model_step,
                            champion_step=evaluator.model_step,
                            model_lag=lag,
                        )
                    counts = store.sample_counts_by_ring(
                        self.experiment.orchestration.ring_mixture.rings,
                        run_id=self.run_identity.run_id,
                        generation_family=self.run_identity.generation_family,
                    )
                    scheduling_step, scheduling_step_source = (
                        self._read_learner_scheduling_step(
                            fallback_step=candidate.model_step
                        )
                    )
                    active_ring_weights = (
                        self.experiment.orchestration.ring_mixture.weights_for_step(
                            scheduling_step
                        )
                    )
                    ring = self.scheduler.choose(
                        counts,
                        learner_step=scheduling_step,
                    )
                    generation = store.lease_generation(
                        self.run_identity, self.actor_id
                    )
                    batch_config = replace(
                        self.experiment.selfplay,
                        rings=ring,
                        batch_size=self.gpu.actor_batch_size,
                        games=self.experiment.orchestration.actor_games_per_batch,
                    )
                    self.heartbeat.advance(
                        phase="selfplay",
                        batch=batches,
                        generation=generation,
                        ring=ring,
                        model_role=model_role,
                        model_version=evaluator.model_version,
                        model_step=evaluator.model_step,
                        scheduling_step=scheduling_step,
                        scheduling_step_source=scheduling_step_source,
                        active_ring_weights=active_ring_weights,
                    )
                    evaluator_calls_before = int(
                        getattr(evaluator, "evaluator_calls", 0)
                    )
                    evaluator_rows_before = int(getattr(evaluator, "evaluator_rows", 0))
                    batch_started_ns = time.time_ns()
                    started = time.monotonic()
                    selfplay = SelfPlayActor(
                        self.native,
                        evaluator,
                        store,
                        batch_config,
                        SelfPlayIdentity(
                            run_id=self.run_identity.run_id,
                            generation_family=(self.run_identity.generation_family),
                            actor_id=self.actor_id,
                            generation=generation,
                        ),
                    )
                    summaries = selfplay.run(
                        stop_requested=stop_requested,
                        progress=self.heartbeat.advance,
                    )
                    elapsed = time.monotonic() - started
                    evaluator_calls = (
                        int(getattr(evaluator, "evaluator_calls", 0))
                        - evaluator_calls_before
                    )
                    evaluator_rows = (
                        int(getattr(evaluator, "evaluator_rows", 0))
                        - evaluator_rows_before
                    )
                    if evaluator_calls < 0 or evaluator_rows < 0:
                        raise RuntimeError("evaluator metrics counters moved backwards")
                    metrics_snapshot = getattr(selfplay, "metrics_snapshot", None)
                    measured_metrics = (
                        metrics_snapshot() if callable(metrics_snapshot) else None
                    )
                    selfplay_metrics = (
                        measured_metrics
                        if isinstance(measured_metrics, SelfPlayMetrics)
                        else SelfPlayMetrics()
                    )
                    wins = sum(summary.winner == 0 for summary in summaries)
                    losses = sum(summary.winner == 1 for summary in summaries)
                    if wins + losses != len(summaries):
                        raise RuntimeError("self-play summaries cannot contain ties")
                    samples = sum(summary.samples for summary in summaries)
                    policy_samples = sum(
                        summary.policy_samples for summary in summaries
                    )
                    search_simulations = sum(
                        summary.search_simulations for summary in summaries
                    )
                    game_lengths = [int(summary.samples) for summary in summaries]
                    game_length_distribution: dict[str, int] = {}
                    for game_length in game_lengths:
                        key = str(game_length)
                        game_length_distribution[key] = (
                            game_length_distribution.get(key, 0) + 1
                        )
                    policy_entropy_mean = (
                        selfplay_metrics.policy_entropy_sum
                        / selfplay_metrics.policy_entropy_count
                        if selfplay_metrics.policy_entropy_count
                        else None
                    )
                    policy_weight_mean = (
                        selfplay_metrics.policy_weight_sum
                        / selfplay_metrics.policy_entropy_count
                        if selfplay_metrics.policy_entropy_count
                        else None
                    )
                    policy_surprise_mean = (
                        selfplay_metrics.policy_surprise_sum
                        / selfplay_metrics.policy_surprise_count
                        if selfplay_metrics.policy_surprise_count
                        else None
                    )
                    sample_weight_mean = (
                        selfplay_metrics.sample_weight_sum
                        / selfplay_metrics.completed_decisions
                        if selfplay_metrics.completed_decisions
                        else None
                    )
                    attempted_decisions = (
                        selfplay_metrics.full_decisions
                        + selfplay_metrics.fast_decisions
                    )
                    (
                        peak_cuda_memory_bytes,
                        peak_cuda_reserved_memory_bytes,
                    ) = self._peak_cuda_memory()
                    batch_completed_ns = time.time_ns()
                    append_jsonl(
                        self.metrics_path,
                        {
                            "schema_version": 1,
                            "timestamp_ns": batch_completed_ns,
                            "batch_started_ns": batch_started_ns,
                            "batch_completed_ns": batch_completed_ns,
                            "worker": self.actor_id,
                            "gpu_id": self.gpu.gpu_id,
                            "lane_id": self.lane_id,
                            "run_id": self.run_identity.run_id,
                            "generation_family": (self.run_identity.generation_family),
                            "generation": generation,
                            "batch": batches,
                            "ring": ring,
                            "scheduling_step": scheduling_step,
                            "scheduling_step_source": scheduling_step_source,
                            "active_ring_weights": active_ring_weights,
                            "games": len(summaries),
                            "samples": samples,
                            "policy_samples": policy_samples,
                            "policy_supervision_rate": (
                                policy_samples / samples if samples else 0.0
                            ),
                            "search_simulations": search_simulations,
                            "search_simulations_per_second": (
                                search_simulations / elapsed if elapsed else 0.0
                            ),
                            "evaluator_calls": evaluator_calls,
                            "evaluator_rows": evaluator_rows,
                            "evaluator_rows_per_second": (
                                evaluator_rows / elapsed if elapsed else 0.0
                            ),
                            "completed_decisions": (
                                selfplay_metrics.completed_decisions
                            ),
                            "attempted_decisions": attempted_decisions,
                            "full_decisions": selfplay_metrics.full_decisions,
                            "fast_decisions": selfplay_metrics.fast_decisions,
                            "game_lengths": game_lengths,
                            "game_length_distribution": game_length_distribution,
                            "mean_game_length": (
                                statistics.fmean(game_lengths) if game_lengths else None
                            ),
                            "policy_entropy_count": (
                                selfplay_metrics.policy_entropy_count
                            ),
                            "policy_entropy_sum": selfplay_metrics.policy_entropy_sum,
                            "policy_entropy_mean": policy_entropy_mean,
                            "policy_entropy_unit": "nats",
                            "policy_weight_sum": (selfplay_metrics.policy_weight_sum),
                            "policy_weight_count": (
                                selfplay_metrics.policy_entropy_count
                            ),
                            "policy_weight_mean": policy_weight_mean,
                            "policy_surprise_count": (
                                selfplay_metrics.policy_surprise_count
                            ),
                            "policy_surprise_sum": (
                                selfplay_metrics.policy_surprise_sum
                            ),
                            "policy_surprise_mean": policy_surprise_mean,
                            "sample_weight_sum": (selfplay_metrics.sample_weight_sum),
                            "sample_weight_mean": sample_weight_mean,
                            "interrupted_cohorts": (
                                selfplay_metrics.interrupted_cohorts
                            ),
                            "dropped_games": selfplay_metrics.dropped_games,
                            "dropped_decisions": selfplay_metrics.dropped_decisions,
                            "model_refresh_latency_seconds": (
                                model_refresh_latency_seconds
                            ),
                            "replay_append_calls": (
                                selfplay_metrics.replay_append_calls
                            ),
                            "replay_append_bytes": (
                                selfplay_metrics.replay_append_bytes
                            ),
                            "replay_append_seconds": (
                                selfplay_metrics.replay_append_seconds
                            ),
                            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
                            "peak_cuda_memory_allocated_bytes": (
                                peak_cuda_memory_bytes
                            ),
                            "peak_cuda_memory_reserved_bytes": (
                                peak_cuda_reserved_memory_bytes
                            ),
                            "games_per_second": (
                                len(summaries) / elapsed if elapsed else 0.0
                            ),
                            "samples_per_second": (
                                samples / elapsed if elapsed else 0.0
                            ),
                            "wins_player_zero": wins,
                            "wins_player_one": losses,
                            "elapsed_seconds": elapsed,
                            "model_role": model_role,
                            "selfplay_source": (
                                self.experiment.orchestration.model_refresh.selfplay_source
                            ),
                            "model_version": evaluator.model_version,
                            "model_identity": evaluator.model_identity,
                            "model_step": evaluator.model_step,
                        },
                    )
                    if not summaries and stop_requested():
                        self.heartbeat.advance(
                            phase="cohort_interrupted",
                            batch=batches,
                            generation=generation,
                            dropped_games=selfplay_metrics.dropped_games,
                            dropped_decisions=selfplay_metrics.dropped_decisions,
                            evaluator_rows=evaluator_rows,
                        )
                        break
                    batches += 1
                    self.heartbeat.advance(
                        phase="cohort_complete",
                        batch=batches,
                        generation=generation,
                        games=len(summaries),
                        samples=samples,
                        evaluator_rows=evaluator_rows,
                    )
            return batches
        except Exception:
            final_phase = "failed"
            raise
        finally:
            self.heartbeat.close(final_phase=final_phase)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _reset_peak_cuda_memory(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_cuda_memory(self) -> tuple[int | None, int | None]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None, None
        return (
            int(torch.cuda.max_memory_allocated(self.device)),
            int(torch.cuda.max_memory_reserved(self.device)),
        )

    def _select_model_provider(
        self,
    ) -> tuple[Literal["champion", "candidate", "history"], ManifestModelProvider]:
        refresh = self.experiment.orchestration.model_refresh
        if refresh.selfplay_source == "champion":
            return "champion", self.provider
        assert self.candidate_provider is not None
        if refresh.selfplay_source == "candidate":
            return "candidate", self.candidate_provider
        draw = self.model_random.random()
        if draw < refresh.candidate_probability:
            return "candidate", self.candidate_provider
        if (
            refresh.selfplay_source == "candidate_champion_history_mix"
            and draw < refresh.candidate_probability + refresh.history_probability
        ):
            assert self.history_pool is not None
            candidate = self._read_candidate()
            champion_identity = (
                self.provider.manifest.model_identity
                if self.provider.manifest is not None
                else None
            )
            historical = self.history_pool.select(
                random_source=self.model_random,
                exclude={
                    candidate.model_identity,
                    *({champion_identity} if champion_identity is not None else set()),
                },
            )
            if historical is not None:
                return "history", historical
        return "champion", self.provider

    def _read_candidate(self) -> ModelManifest:
        stat = self.candidate_manifest_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if (
            self._candidate_signature == signature
            and self._candidate_manifest is not None
        ):
            return self._candidate_manifest
        manifest = load_model_manifest(self.candidate_manifest_path)
        self._candidate_signature = signature
        self._candidate_manifest = manifest
        return manifest

    def _read_learner_scheduling_step(self, *, fallback_step: int) -> tuple[int, str]:
        try:
            stat = self.learner_heartbeat_path.stat()
            if (
                time.time() - stat.st_mtime
                > self.experiment.orchestration.shutdown.stale_heartbeat_seconds
            ):
                raise ValueError("learner heartbeat is stale")
            payload = json.loads(
                self.learner_heartbeat_path.read_text(encoding="utf-8")
            )
            step = payload.get("step") if isinstance(payload, dict) else None
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError("learner heartbeat step is invalid")
            return step, "learner_heartbeat"
        except (OSError, json.JSONDecodeError, ValueError):
            return fallback_step, "candidate_manifest"
