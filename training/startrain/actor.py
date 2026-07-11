"""Long-lived, batch-boundary-refreshing self-play actor supervisor."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
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
from .selfplay import SelfPlayActor, SelfPlayIdentity
from .training import maybe_compile_model


class RingMixtureScheduler:
    def __init__(self, config: RingMixtureConfig, *, seed: int) -> None:
        self.config = config
        self.random = random.Random(seed)

    def choose(self, sample_counts: Mapping[int, int]) -> int:
        counts = {ring: int(sample_counts.get(ring, 0)) for ring in self.config.rings}
        if any(value < 0 for value in counts.values()):
            raise ValueError("ring sample counts must be non-negative")
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
        expected_role: Literal["champion", "candidate"] = "champion",
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
                initial_pass_logit_penalty=(
                    self.config.selfplay.initial_pass_logit_penalty
                ),
            ),
            model_version=manifest.model_version,
            model_step=manifest.model_step,
            model_identity=manifest.model_identity,
        )
        self.manifest = manifest
        self._pointer_signature = signature
        return self.evaluator


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
        device: str = "cuda",
    ) -> None:
        if gpu.role != "actor" or gpu.actor_batch_size is None:
            raise ValueError("actor supervisor requires an actor GPU assignment")
        self.native = native_module
        self.experiment = experiment
        self.gpu = gpu
        self.replay_directory = Path(replay_directory)
        self.run_identity = run_identity
        self.actor_id = f"actor-gpu-{gpu.gpu_id}"
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
        self.model_random = random.Random(
            experiment.selfplay.seed + gpu.gpu_id * 1_000_003 + 0x5E1F
        )
        self.heartbeat = HeartbeatReporter(
            heartbeat_path,
            worker=self.actor_id,
            interval_seconds=experiment.orchestration.shutdown.heartbeat_interval_seconds,
        )
        self.metrics_path = Path(metrics_path)
        self.scheduler = RingMixtureScheduler(
            experiment.orchestration.ring_mixture,
            seed=experiment.selfplay.seed + gpu.gpu_id * 1_000_003,
        )

    def run(self, *, stop_requested: Callable[[], bool]) -> int:
        batches = 0
        self.heartbeat.start()
        final_phase = "stopped"
        try:
            _, initial_provider = self._select_model_provider()
            evaluator = initial_provider.wait_for_initial(
                stop_requested=stop_requested,
                progress=self.heartbeat.advance,
            )
            if evaluator is None:
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
                    evaluator = provider.refresh()
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
                    ring = self.scheduler.choose(counts)
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
                    )
                    started = time.monotonic()
                    summaries = SelfPlayActor(
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
                    ).run(
                        stop_requested=stop_requested,
                        progress=self.heartbeat.advance,
                    )
                    if not summaries and stop_requested():
                        break
                    elapsed = time.monotonic() - started
                    wins = sum(summary.winner == 0 for summary in summaries)
                    losses = sum(summary.winner == 1 for summary in summaries)
                    draws = sum(summary.winner == -1 for summary in summaries)
                    samples = sum(summary.samples for summary in summaries)
                    policy_samples = sum(
                        summary.policy_samples for summary in summaries
                    )
                    search_simulations = sum(
                        summary.search_simulations for summary in summaries
                    )
                    append_jsonl(
                        self.metrics_path,
                        {
                            "schema_version": 1,
                            "timestamp_ns": time.time_ns(),
                            "worker": self.actor_id,
                            "run_id": self.run_identity.run_id,
                            "generation_family": (self.run_identity.generation_family),
                            "generation": generation,
                            "batch": batches,
                            "ring": ring,
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
                            "games_per_second": (
                                len(summaries) / elapsed if elapsed else 0.0
                            ),
                            "samples_per_second": (
                                samples / elapsed if elapsed else 0.0
                            ),
                            "wins_player_zero": wins,
                            "wins_player_one": losses,
                            "draws": draws,
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
                    batches += 1
                    self.heartbeat.advance(
                        phase="cohort_complete",
                        batch=batches,
                        generation=generation,
                        games=len(summaries),
                        samples=samples,
                    )
            return batches
        except Exception:
            final_phase = "failed"
            raise
        finally:
            self.heartbeat.close(final_phase=final_phase)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _select_model_provider(
        self,
    ) -> tuple[Literal["champion", "candidate"], ManifestModelProvider]:
        refresh = self.experiment.orchestration.model_refresh
        if refresh.selfplay_source == "champion":
            return "champion", self.provider
        assert self.candidate_provider is not None
        if refresh.selfplay_source == "candidate":
            return "candidate", self.candidate_provider
        if self.model_random.random() < refresh.candidate_probability:
            return "candidate", self.candidate_provider
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
