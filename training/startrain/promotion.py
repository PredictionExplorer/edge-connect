"""Immutable candidate evaluation and atomic champion promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import TracebackType

import torch

from .arena import (
    ARENA_RESULT_SCHEMA_VERSION,
    ArenaPair,
    ArenaRunner,
    summarize_completed_arena_pairs,
)
from .checkpoint import (
    ModelManifest,
    collect_model_garbage,
    extract_verified_manifest_config,
    load_ema_checkpoint,
    load_model_manifest,
)
from .checkpoint import write_model_pointer
from .config import ArenaConfig, ExperimentConfig, load_config
from .device import (
    empty_device_cache,
    peak_memory_stats,
    reset_peak_memory_stats,
    resolve_device_string,
    synchronize_device,
)
from .inference import GraphInferenceAdapter, InferenceConfig
from .historical_evaluation import (
    HISTORICAL_CROSSPLAY_RESULT_KIND,
    load_arena_results,
    load_historical_manifests,
    select_historical_evaluation,
)
from .model import GraphResTNet
from .native import load_star_native
from .orchestration import gpu_pause_ack_path
from .runtime import (
    HeartbeatReporter,
    RunIdentity,
    SignalLatch,
    append_jsonl,
    atomic_json,
    load_run_identity,
)
from .training import maybe_compile_model


def load_manifest_evaluator(
    experiment: ExperimentConfig,
    manifest: ModelManifest,
    *,
    device: str,
    allow_heterogeneous_model: bool = False,
) -> GraphInferenceAdapter:
    model_config = experiment.model
    if allow_heterogeneous_model:
        verified = extract_verified_manifest_config(
            manifest,
            expected_game_config=asdict(experiment.game),
        )
        model_config = verified.model
    model = GraphResTNet(model_config).to(device)
    metadata = load_ema_checkpoint(
        manifest.checkpoint,
        model=model,
        expected_model_config=asdict(model_config),
        expected_game_config=asdict(experiment.game),
        expected_run_id=manifest.run_id,
        expected_generation_family=manifest.generation_family,
        expected_sha256=manifest.checkpoint_sha256,
        expected_bytes=manifest.checkpoint_bytes,
        map_location=device,
    )
    if int(metadata["step"]) != manifest.model_step:
        raise ValueError("manifest and checkpoint step disagree")
    model.eval()
    refresh = experiment.orchestration.model_refresh
    inference_model = maybe_compile_model(
        model,
        enabled=experiment.train.compile,
        dynamic=refresh.inference_compile_dynamic,
        fullgraph=True,
        mode=refresh.inference_compile_mode,
        recompile_limit=(
            None if refresh.inference_compile_dynamic else len(experiment.game.rings)
        ),
        isolate_recompiles=not refresh.inference_compile_dynamic,
    )
    return GraphInferenceAdapter(
        inference_model,
        device=device,
        config=InferenceConfig(
            precision=experiment.train.precision,
            score_utility_weight=experiment.selfplay.score_utility_weight,
        ),
        model_version=manifest.model_version,
        model_step=manifest.model_step,
        model_identity=manifest.model_identity,
    )


class PauseLeaseError(RuntimeError):
    pass


class PauseLeaseInterrupted(PauseLeaseError):
    pass


class CoordinatorPauseLease:
    """Tokenized two-phase lease acquired before any shared-GPU allocation."""

    def __init__(
        self,
        *,
        request_path: str | Path,
        gpu_id: int,
        candidate_identity: str,
        ready_timeout_seconds: float,
        release_timeout_seconds: float,
        heartbeat_interval_seconds: float,
        poll_seconds: float,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
        events_path: str | Path,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.request_path = Path(request_path)
        self.ack_path = gpu_pause_ack_path(self.request_path)
        self.gpu_id = gpu_id
        self.candidate_identity = candidate_identity
        self.ready_timeout_seconds = ready_timeout_seconds
        self.release_timeout_seconds = release_timeout_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.poll_seconds = poll_seconds
        self.stop_requested = stop_requested
        self.progress = progress
        self.events_path = Path(events_path)
        self.clock = clock
        self.sleep = sleep
        self.token = uuid.uuid4().hex
        self.owner_pid = os.getpid()
        self.requested_ns = 0
        self.state = "new"
        self._outcome: str | None = None
        self._lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_error: OSError | None = None
        self._heartbeat_thread: threading.Thread | None = None

    def __enter__(self) -> "CoordinatorPauseLease":
        self.requested_ns = time.time_ns()
        self.state = "requested"
        self._write_request()
        self._event("requested")
        if self.progress is not None:
            self.progress(
                phase="waiting_for_gpu_pause",
                lease_token=self.token,
                gpu_id=self.gpu_id,
            )
        self._start_heartbeat()
        try:
            self._wait_until_ready()
        except BaseException:
            self._cancel()
            raise
        self.state = "active"
        self._write_request()
        self._event("ready")
        if self.progress is not None:
            self.progress(
                phase="arena_gpu_lease_ready",
                lease_token=self.token,
                gpu_id=self.gpu_id,
            )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        self._stop_heartbeat()
        self.state = "released"
        self._outcome = (
            "completed"
            if exception_type is None
            else f"error:{exception_type.__name__}"
        )
        release_error: BaseException | None = None
        try:
            self._write_request()
            self._event("release_requested", outcome=self._outcome)
            acknowledgement = self._wait_until_released()
            self._event("released", acknowledgement=acknowledgement)
            if self.progress is not None:
                self.progress(
                    phase="arena_gpu_lease_released",
                    lease_token=self.token,
                    acknowledgement=acknowledgement,
                )
        except BaseException as error:
            release_error = error
            self._event("release_failed", error=str(error))
        if exception is None and release_error is not None:
            raise release_error
        return False

    def _start_heartbeat(self) -> None:
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"gpu-pause-lease-{self.token[:8]}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self.heartbeat_interval_seconds + 0.5))

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_interval_seconds):
            try:
                self._write_request()
            except OSError as error:
                self._heartbeat_error = error
                return

    def _write_request(self) -> None:
        with self._lock:
            payload: dict[str, object] = {
                "schema_version": 1,
                "protocol": "coordinator-pause-v1",
                "token": self.token,
                "pid": self.owner_pid,
                "gpu_id": self.gpu_id,
                "candidate_identity": self.candidate_identity,
                "state": self.state,
                "requested_ns": self.requested_ns,
                "heartbeat_ns": time.time_ns(),
            }
            if self._outcome is not None:
                payload["outcome"] = self._outcome
            atomic_json(self.request_path, payload)

    def _wait_until_ready(self) -> None:
        deadline = self.clock() + self.ready_timeout_seconds
        while self.clock() < deadline:
            self._raise_heartbeat_error()
            if self.stop_requested():
                raise PauseLeaseInterrupted(
                    "shutdown requested while awaiting GPU lease"
                )
            acknowledgement = self._read_ack()
            if acknowledgement is not None:
                state = acknowledgement.get("state")
                if state == "ready":
                    return
                if state in ("failed", "stopping", "draining", "recovered"):
                    reason = acknowledgement.get("reason")
                    raise PauseLeaseError(
                        f"coordinator rejected GPU pause lease: {state}: {reason}"
                    )
            self.sleep(self.poll_seconds)
        raise TimeoutError(
            "timed out before token-matched coordinator GPU-ready acknowledgement"
        )

    def _wait_until_released(self) -> str:
        deadline = self.clock() + self.release_timeout_seconds
        while self.clock() < deadline:
            if self.stop_requested():
                return "stopping"
            acknowledgement = self._read_ack()
            if acknowledgement is not None:
                state = acknowledgement.get("state")
                if state in ("released", "recovered", "draining", "stopping"):
                    return str(state)
                if state == "failed":
                    raise PauseLeaseError(
                        "coordinator could not safely restore pause-shared worker: "
                        f"{acknowledgement.get('reason')}"
                    )
            self.sleep(self.poll_seconds)
        raise TimeoutError(
            "timed out awaiting token-matched pause-worker release acknowledgement"
        )

    def _cancel(self) -> None:
        self._stop_heartbeat()
        self.state = "cancelled"
        self._outcome = "cancelled"
        try:
            self._write_request()
            self._event("cancelled")
            self._wait_until_released()
        except (OSError, PauseLeaseError, TimeoutError) as error:
            self._event("cancel_cleanup_failed", error=str(error))

    def _read_ack(self) -> dict[str, object] | None:
        try:
            with self.ack_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("protocol") != "coordinator-pause-v1"
            or payload.get("token") != self.token
            or payload.get("gpu_id") != self.gpu_id
        ):
            return None
        return payload

    def _raise_heartbeat_error(self) -> None:
        if self._heartbeat_error is not None:
            raise PauseLeaseError(
                f"GPU pause lease heartbeat failed: {self._heartbeat_error}"
            )

    def _event(self, state: str, **details: object) -> None:
        append_jsonl(
            self.events_path,
            {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "event": f"pause_lease_{state}",
                "token": self.token,
                "pid": self.owner_pid,
                "gpu_id": self.gpu_id,
                "candidate_identity": self.candidate_identity,
                **details,
            },
            durable=True,
        )


class PromotionSupervisor:
    def __init__(
        self,
        *,
        experiment: ExperimentConfig,
        run_identity: RunIdentity,
        candidate_path: str | Path,
        champion_path: str | Path,
        results_directory: str | Path,
        native_module: object,
        device: str,
        gpu_pause_path: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.experiment = experiment
        self.run_identity = run_identity
        self.candidate_path = Path(candidate_path)
        self.champion_path = Path(champion_path)
        self.manifest_directory = self.candidate_path.parent / "manifests"
        self.results_directory = Path(results_directory)
        self.status_path = self.results_directory / "promotion-status.json"
        self.native = native_module
        self.device = device
        self.gpu_pause_path = (
            Path(gpu_pause_path) if gpu_pause_path is not None else None
        )
        self.pause_events_path = self.results_directory / "pause-lease-events.jsonl"
        self.cooldown_path = (
            self.gpu_pause_path.parent / "arena-inter-wave-cooldown.json"
            if self.gpu_pause_path is not None
            else self.results_directory / ".inter-wave-cooldown.json"
        )
        self.clock = clock
        self.wall_clock_ns = wall_clock_ns
        self.sleep = sleep
        self._manifest_cache: dict[Path, ModelManifest] = {}

    def run(
        self,
        *,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None = None,
        once: bool = False,
    ) -> int:
        evaluated = 0
        promotion = self.experiment.orchestration.promotion
        self.results_directory.mkdir(parents=True, exist_ok=True)
        while not stop_requested():
            candidates = self._candidate_manifests()
            if not candidates:
                if progress is not None:
                    progress(phase="waiting_for_candidate")
                if once:
                    return evaluated
                self.sleep(promotion.poll_seconds)
                continue
            champion = (
                load_model_manifest(self.champion_path)
                if self.champion_path.is_file()
                else None
            )
            if champion is None:
                if not promotion.bootstrap_initial_champion:
                    raise RuntimeError(
                        "champion is absent and explicit bootstrap is disabled"
                    )
                champion = candidates[0]
                write_model_pointer(
                    self.champion_path,
                    champion,
                    role="champion",
                    promotion_result="bootstrap",
                )
                self._write_status(
                    candidate=champion,
                    champion=champion,
                    decision="bootstrap",
                    terminal=True,
                )
                if progress is not None:
                    progress(
                        phase="bootstrapped_champion",
                        model_identity=champion.model_identity,
                        model_step=champion.model_step,
                    )
            for stale in candidates:
                if (
                    stale.model_identity != champion.model_identity
                    and stale.model_step < champion.model_step
                ):
                    if (
                        self._mark_superseded(stale, champion, superseded_by=champion)
                        and progress is not None
                    ):
                        progress(
                            phase="candidate_superseded",
                            candidate_step=stale.model_step,
                            superseded_by_step=champion.model_step,
                        )
            viable = [
                item
                for item in candidates
                if item.model_identity != champion.model_identity
                and item.model_step >= champion.model_step
            ]
            started: list[tuple[ModelManifest, dict[str, object]]] = []
            for item in viable:
                item_result = self._read_result(item, champion)
                if (
                    item_result is not None
                    and not bool(item_result.get("terminal"))
                    and self._pairs_from_result(item_result)
                ):
                    started.append((item, item_result))
            if started:
                candidate, previous = min(
                    started,
                    key=lambda item: (
                        item[0].model_step,
                        item[0].model_identity,
                    ),
                )
            else:
                candidate = max(
                    viable,
                    key=lambda item: (item.model_step, item.model_identity),
                    default=None,
                )
                previous = (
                    self._read_result(candidate, champion)
                    if candidate is not None
                    else None
                )
            if candidate is None:
                try:
                    historical_waves = self._evaluate_historical_if_due(
                        champion=champion,
                        stop_requested=stop_requested,
                        progress=progress,
                        once=once,
                    )
                except PauseLeaseInterrupted:
                    return evaluated
                evaluated += historical_waves
                if historical_waves:
                    if once:
                        return evaluated
                    continue
                if progress is not None:
                    progress(
                        phase="waiting_for_candidate",
                        champion_step=champion.model_step,
                    )
                if once:
                    return evaluated
                self.sleep(promotion.poll_seconds)
                continue
            if not started:
                for skipped in viable:
                    if skipped.model_identity != candidate.model_identity:
                        marked = self._mark_superseded(
                            skipped,
                            champion,
                            superseded_by=candidate,
                        )
                        if marked and progress is not None:
                            progress(
                                phase="candidate_superseded",
                                candidate_step=skipped.model_step,
                                superseded_by_step=candidate.model_step,
                            )
            if previous is not None and bool(previous.get("terminal")):
                if progress is not None:
                    previous_promotion = previous.get("promotion")
                    decision = (
                        previous_promotion.get("decision")
                        if isinstance(previous_promotion, dict)
                        else None
                    )
                    progress(
                        phase="awaiting_new_candidate",
                        champion_step=champion.model_step,
                        candidate_step=candidate.model_step,
                        last_decision=decision,
                    )
                if once:
                    return evaluated
                self.sleep(promotion.poll_seconds)
                continue
            try:
                session_evaluated, session_state = self._evaluate_candidate_session(
                    candidate=candidate,
                    champion=champion,
                    previous=previous,
                    stop_requested=stop_requested,
                    progress=progress,
                    once=once,
                )
                evaluated += session_evaluated
            except PauseLeaseInterrupted:
                return evaluated
            if stop_requested():
                return evaluated
            if session_state == "lease_yield":
                if not self._wait_between_leases(
                    candidate=candidate,
                    champion=champion,
                    stop_requested=stop_requested,
                    progress=progress,
                ):
                    return evaluated
            if once and session_state != "superseded":
                return evaluated
        return evaluated

    def _wait_between_leases(
        self,
        *,
        candidate: ModelManifest,
        champion: ModelManifest,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
    ) -> bool:
        promotion = self.experiment.orchestration.promotion
        if not self.cooldown_path.is_file():
            return True
        try:
            payload = json.loads(self.cooldown_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read arena inter-wave cooldown: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("run_id") != self.run_identity.run_id
            or payload.get("generation_family") != self.run_identity.generation_family
            or isinstance(payload.get("not_before_ns"), bool)
            or not isinstance(payload.get("not_before_ns"), int)
        ):
            raise ValueError("arena inter-wave cooldown is incompatible")
        not_before_ns = int(payload["not_before_ns"])
        while self.wall_clock_ns() < not_before_ns:
            if stop_requested():
                return False
            remaining = max(0.0, (not_before_ns - self.wall_clock_ns()) / 1e9)
            if progress is not None:
                progress(
                    phase="arena_inter_wave_cooldown",
                    candidate_step=candidate.model_step,
                    champion_step=champion.model_step,
                    remaining_seconds=remaining,
                )
            interval = min(promotion.poll_seconds, remaining)
            self.sleep(interval)
        self.cooldown_path.unlink(missing_ok=True)
        return True

    def _record_inter_wave_cooldown(self, candidate: ModelManifest) -> None:
        seconds = self.experiment.orchestration.promotion.inter_wave_cooldown_seconds
        if seconds <= 0:
            return
        created_ns = self.wall_clock_ns()
        atomic_json(
            self.cooldown_path,
            {
                "schema_version": 1,
                "run_id": self.run_identity.run_id,
                "generation_family": self.run_identity.generation_family,
                "candidate_identity": candidate.model_identity,
                "candidate_step": candidate.model_step,
                "created_ns": created_ns,
                "not_before_ns": created_ns + int(seconds * 1_000_000_000),
            },
        )

    def _evaluate_historical_if_due(
        self,
        *,
        champion: ModelManifest,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
        once: bool,
    ) -> int:
        configured = self.experiment.orchestration.historical_evaluation
        if not configured.enabled or stop_requested():
            return 0
        manifests = load_historical_manifests(
            self.manifest_directory,
            run_identity=self.run_identity,
        )
        manifests[champion.model_identity] = champion
        plan = select_historical_evaluation(
            config=configured,
            champion=champion,
            manifests=manifests,
            arena_results=load_arena_results(self.results_directory),
            results_directory=self.results_directory,
        )
        if plan is None:
            return 0
        with self._gpu_pause(
            stop_requested=stop_requested,
            progress=progress,
            candidate_identity=plan.candidate.model_identity,
        ):
            return self._evaluate_historical_waves(
                candidate=plan.candidate,
                baseline=plan.baseline,
                result_path=plan.result_path,
                previous=plan.previous,
                stop_requested=stop_requested,
                progress=progress,
                once=once,
            )

    def _evaluate_historical_waves(
        self,
        *,
        candidate: ModelManifest,
        baseline: ModelManifest,
        result_path: Path,
        previous: dict[str, object] | None,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
        once: bool,
    ) -> int:
        configured = self.experiment.orchestration.historical_evaluation
        material = (
            f"historical-crossplay-v1\0{self.experiment.arena.seed}\0"
            f"{candidate.model_identity}\0{baseline.model_identity}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        arena_config = replace(
            self.experiment.arena,
            pairs_per_ring=configured.pairs_per_ring,
            minimum_pairs_per_ring=configured.max_pairs_per_ring,
            max_pairs_per_ring=configured.max_pairs_per_ring,
            promotion_pair_ratios={},
            required_regression_rings=None,
            weighted_initial_blocks=0,
            weighted_continuation_blocks=0,
            weighted_max_blocks=0,
            seed=seed,
        )
        accumulated = self._pairs_from_result(previous)
        candidate_evaluator = load_manifest_evaluator(
            self.experiment, candidate, device=self.device
        )
        try:
            baseline_evaluator = load_manifest_evaluator(
                self.experiment, baseline, device=self.device
            )
            try:
                runner = ArenaRunner(
                    native_module=self.native,
                    candidate=candidate_evaluator,
                    baseline=baseline_evaluator,
                    config=arena_config,
                )
                waves = 0
                previous_result = previous
                while not stop_requested():
                    if self._newer_candidate(candidate, candidate) is not None:
                        return waves
                    starts = {
                        ring: (
                            max(
                                (
                                    pair.pair
                                    for pair in accumulated
                                    if pair.ring == ring
                                ),
                                default=-1,
                            )
                            + 1
                        )
                        for ring in arena_config.rings
                    }
                    counts = {
                        ring: min(
                            configured.pairs_per_ring,
                            configured.max_pairs_per_ring
                            - sum(pair.ring == ring for pair in accumulated),
                        )
                        for ring in arena_config.rings
                    }
                    if all(count <= 0 for count in counts.values()):
                        return waves
                    if progress is not None:
                        progress(
                            phase="historical_crossplay",
                            candidate_step=candidate.model_step,
                            baseline_step=baseline.model_step,
                            pairs=len(accumulated),
                        )
                    started = time.perf_counter()
                    result = runner.run(
                        progress=progress,
                        pair_starts=starts,
                        pair_counts=counts,
                        stop_requested=stop_requested,
                    )
                    completed = self._pairs_from_result(result)
                    if not completed:
                        if stop_requested():
                            return waves
                        raise RuntimeError(
                            "historical arena completed no role-reversed pairs"
                        )
                    unique = {(pair.ring, pair.pair): pair for pair in accumulated}
                    for pair in completed:
                        key = (pair.ring, pair.pair)
                        existing = unique.get(key)
                        if existing is not None and existing != pair:
                            raise ValueError(
                                "historical evaluation changed a persisted pair"
                            )
                        unique[key] = pair
                    accumulated[:] = [unique[key] for key in sorted(unique)]
                    result["schema_version"] = ARENA_RESULT_SCHEMA_VERSION
                    result["pairs"] = [asdict(pair) for pair in accumulated]
                    result.update(
                        summarize_completed_arena_pairs(accumulated, arena_config)
                    )
                    if previous_result is not None:
                        old_games = previous_result.get("games", [])
                        new_games = result.get("games", [])
                        if not isinstance(old_games, list) or not isinstance(
                            new_games, list
                        ):
                            raise ValueError(
                                "persisted historical arena games are invalid"
                            )
                        result["games"] = [*old_games, *new_games]
                    result["result_kind"] = HISTORICAL_CROSSPLAY_RESULT_KIND
                    result["candidate_manifest"] = str(
                        (candidate.artifact_manifest or candidate.path).resolve()
                    )
                    result["baseline_manifest"] = str(
                        (baseline.artifact_manifest or baseline.path).resolve()
                    )
                    result["arena_seed_block"] = arena_config.seed
                    metrics = result.get("evaluation_metrics")
                    if not isinstance(metrics, dict):
                        metrics = {}
                        result["evaluation_metrics"] = metrics
                    metrics["round_wall_seconds"] = time.perf_counter() - started
                    terminal = all(
                        sum(pair.ring == ring for pair in accumulated)
                        >= configured.max_pairs_per_ring
                        for ring in arena_config.rings
                    )
                    assessment = result.get("promotion")
                    if isinstance(assessment, dict):
                        assessment["decision"] = "evaluation"
                    result["terminal"] = terminal
                    atomic_json(result_path, result)
                    previous_result = result
                    waves += 1
                    if terminal or once or stop_requested():
                        return waves
                return waves
            finally:
                del baseline_evaluator
        finally:
            del candidate_evaluator
            synchronize_device(self.device)
            empty_device_cache(self.device)

    def _evaluate_candidate_session(
        self,
        *,
        candidate: ModelManifest,
        champion: ModelManifest,
        previous: dict[str, object] | None,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
        once: bool,
    ) -> tuple[int, str]:
        if not self._wait_between_leases(
            candidate=candidate,
            champion=champion,
            stop_requested=stop_requested,
            progress=progress,
        ):
            return 0, "stopped"
        accumulated = self._pairs_from_result(previous)
        newer = self._newer_candidate(candidate, champion)
        if newer is not None and not (
            self.experiment.orchestration.promotion.finish_inflight_candidate
            and accumulated
        ):
            marked = self._mark_superseded(
                candidate,
                champion,
                superseded_by=newer,
            )
            if marked and progress is not None:
                progress(
                    phase="candidate_superseded",
                    candidate_step=candidate.model_step,
                    superseded_by_step=newer.model_step,
                )
            return 0, "superseded"

        starts, counts = self._wave_plan(accumulated)
        if all(count <= 0 for count in counts.values()):
            if previous is None:
                raise ValueError("max-pair promotion result is missing")
            self._reject_max_pairs(
                candidate=candidate,
                champion=champion,
                result=previous,
                progress=progress,
            )
            return 0, "terminal"
        if stop_requested():
            return 0, "stopped"

        with self._gpu_pause(
            stop_requested=stop_requested,
            progress=progress,
            candidate_identity=candidate.model_identity,
        ):
            result = self._evaluate_waves(
                candidate=candidate,
                champion=champion,
                previous=previous,
                accumulated=accumulated,
                stop_requested=stop_requested,
                progress=progress,
                once=once,
            )
        if result[1] == "lease_yield":
            self._record_inter_wave_cooldown(candidate)
        return result

    def _evaluate_waves(
        self,
        *,
        candidate: ModelManifest,
        champion: ModelManifest,
        previous: dict[str, object] | None,
        accumulated: list[ArenaPair],
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
        once: bool,
    ) -> tuple[int, str]:
        session_started = time.perf_counter()
        metric_device = torch.device(self.device)
        collect_cuda_metrics = (
            metric_device.type == "cuda" and torch.cuda.is_available()
        )
        reset_peak_memory_stats(metric_device)
        arena_config = self._arena_config(candidate, champion)
        if progress is not None:
            progress(
                phase="arena",
                candidate_step=candidate.model_step,
                champion_step=champion.model_step,
            )
            progress(
                phase="arena_loading_candidate",
                candidate_step=candidate.model_step,
                champion_step=champion.model_step,
            )
        candidate_evaluator = load_manifest_evaluator(
            self.experiment, candidate, device=self.device
        )
        try:
            if stop_requested():
                return 0, "stopped"
            if progress is not None:
                progress(
                    phase="arena_loading_champion",
                    candidate_step=candidate.model_step,
                    champion_step=champion.model_step,
                )
            champion_evaluator = load_manifest_evaluator(
                self.experiment, champion, device=self.device
            )
            runner: ArenaRunner | None = None
            try:
                runner = ArenaRunner(
                    native_module=self.native,
                    candidate=candidate_evaluator,
                    baseline=champion_evaluator,
                    config=arena_config,
                )
                waves = 0
                previous_result = previous
                while True:
                    if stop_requested():
                        return waves, "stopped"
                    newer = self._newer_candidate(candidate, champion)
                    if newer is not None and not (
                        self.experiment.orchestration.promotion.finish_inflight_candidate
                        and accumulated
                    ):
                        marked = self._mark_superseded(
                            candidate,
                            champion,
                            superseded_by=newer,
                        )
                        if marked and progress is not None:
                            progress(
                                phase="candidate_superseded",
                                candidate_step=candidate.model_step,
                                superseded_by_step=newer.model_step,
                            )
                        return waves, "superseded"
                    starts, counts = self._wave_plan(accumulated)
                    if all(count <= 0 for count in counts.values()):
                        if previous_result is None:
                            raise ValueError("max-pair promotion result is missing")
                        self._reject_max_pairs(
                            candidate=candidate,
                            champion=champion,
                            result=previous_result,
                            progress=progress,
                        )
                        return waves, "terminal"
                    if progress is not None:
                        progress(
                            phase="arena_search_start",
                            candidate_step=candidate.model_step,
                            champion_step=champion.model_step,
                        )
                    wave_started = (
                        session_started if waves == 0 else time.perf_counter()
                    )
                    persisted_chunk = False
                    pair_ratios = arena_config.promotion_pair_ratios
                    for chunk_starts, chunk_counts in self._pair_chunks(
                        starts,
                        counts,
                        chunk_size=(
                            arena_config.pair_chunk_size
                            or max(counts.values(), default=1)
                        ),
                        pair_ratios=pair_ratios,
                        existing_counts=self._ring_pair_counts(accumulated),
                    ):
                        if stop_requested():
                            return waves + int(persisted_chunk), "stopped"
                        chunk_started = (
                            wave_started if not persisted_chunk else time.perf_counter()
                        )
                        result = runner.run(
                            progress=progress,
                            pair_starts=chunk_starts,
                            pair_counts=chunk_counts,
                            stop_requested=stop_requested,
                        )
                        completed = self._pairs_from_result(result)
                        if not completed:
                            if stop_requested():
                                return waves + int(persisted_chunk), "stopped"
                            raise RuntimeError(
                                "promotion arena completed no role-reversed pairs"
                            )
                        decision, terminal = self._persist_wave(
                            candidate=candidate,
                            champion=champion,
                            previous=previous_result,
                            accumulated=accumulated,
                            result=result,
                            arena_config=arena_config,
                            round_started=chunk_started,
                            metric_device=metric_device,
                            collect_cuda_metrics=collect_cuda_metrics,
                            progress=progress,
                            wave_index=waves,
                            pair_starts=chunk_starts,
                            pair_counts=chunk_counts,
                        )
                        persisted_chunk = True
                        previous_result = result
                        if terminal:
                            return waves + 1, "terminal"
                        if stop_requested():
                            return waves + 1, "stopped"
                    waves += 1
                    if once:
                        return waves, "once"
                    max_waves = (
                        self.experiment.orchestration.promotion.max_waves_per_lease
                    )
                    if max_waves is not None and waves >= max_waves:
                        return waves, "lease_yield"
            finally:
                del runner
                del champion_evaluator
        finally:
            del candidate_evaluator
            if collect_cuda_metrics:
                torch.cuda.synchronize(metric_device)
                torch.cuda.empty_cache()

    def _persist_wave(
        self,
        *,
        candidate: ModelManifest,
        champion: ModelManifest,
        previous: dict[str, object] | None,
        accumulated: list[ArenaPair],
        result: dict[str, object],
        arena_config: ArenaConfig,
        round_started: float,
        metric_device: torch.device,
        collect_cuda_metrics: bool,
        progress: Callable[..., None] | None,
        wave_index: int,
        pair_starts: Mapping[int, int],
        pair_counts: Mapping[int, int],
    ) -> tuple[str, bool]:
        synchronize_device(metric_device)
        completed_pairs = self._pairs_from_result(result)
        evaluation_metrics = result.get("evaluation_metrics")
        if evaluation_metrics is None:
            evaluation_metrics = {}
            result["evaluation_metrics"] = evaluation_metrics
        elif not isinstance(evaluation_metrics, dict):
            raise ValueError("arena result evaluator metrics are invalid")
        evaluation_metrics["round_wall_seconds"] = time.perf_counter() - round_started
        peak_allocated, peak_reserved = peak_memory_stats(metric_device)
        evaluation_metrics["peak_cuda_allocated_bytes"] = (
            peak_allocated if collect_cuda_metrics else None
        )
        evaluation_metrics["peak_cuda_reserved_bytes"] = (
            peak_reserved if collect_cuda_metrics else None
        )
        evaluation_metrics["requested_pairs"] = sum(pair_counts.values())
        evaluation_metrics["completed_pairs"] = len(completed_pairs)
        previous_history = (
            previous.get("wave_history", []) if previous is not None else []
        )
        if not isinstance(previous_history, list) or not all(
            isinstance(item, dict) for item in previous_history
        ):
            raise ValueError("persisted arena wave history is invalid")
        if previous is not None and not previous_history:
            legacy_pairs = self._pairs_from_result(previous)
            if legacy_pairs:
                previous_history = [
                    {
                        "schema_version": 0,
                        "wave_index": 0,
                        "phase": "legacy",
                        "pair_counts": {
                            str(ring): sum(pair.ring == ring for pair in legacy_pairs)
                            for ring in self.experiment.arena.rings
                        },
                    }
                ]
        pair_ratios = arena_config.promotion_pair_ratios
        existing_pair_counts = self._ring_pair_counts(accumulated)
        weighted_metadata: dict[str, object] = {}
        if pair_ratios:
            complete_blocks_before = self._complete_blocks(
                existing_pair_counts,
                pair_ratios,
            )
            requested_pair_counts = {
                ring: existing_pair_counts[ring] + int(pair_counts.get(ring, 0))
                for ring in arena_config.rings
            }
            complete_blocks_target = self._complete_blocks(
                requested_pair_counts,
                pair_ratios,
            )
            phase = (
                "initial"
                if complete_blocks_before < arena_config.weighted_initial_blocks
                else "continuation"
            )
            weighted_metadata = {
                "allocation_mode": "weighted_complete_blocks",
                "pair_ratios": {
                    str(ring): int(pair_ratios[ring]) for ring in arena_config.rings
                },
                "complete_blocks_before": complete_blocks_before,
                "target_complete_blocks": complete_blocks_target,
                "pair_counts_before": {
                    str(ring): existing_pair_counts[ring] for ring in arena_config.rings
                },
                "pair_count_targets": {
                    str(ring): complete_blocks_target * int(pair_ratios[ring])
                    for ring in arena_config.rings
                },
                "pair_deficits": {
                    str(ring): int(pair_counts.get(ring, 0))
                    for ring in arena_config.rings
                },
                "pair_counts_completed": {
                    str(ring): sum(pair.ring == ring for pair in completed_pairs)
                    for ring in arena_config.rings
                },
            }
        else:
            phase = (
                "initial"
                if any(
                    pair_starts.get(ring, 0)
                    < self.experiment.arena.minimum_pairs_per_ring
                    for ring in self.experiment.arena.rings
                )
                else "continuation"
            )
        wave_plan = {
            "schema_version": 1,
            "wave_index": len(previous_history),
            "lease_wave_index": wave_index,
            "phase": phase,
            "pair_starts": {
                str(ring): int(pair_starts.get(ring, 0))
                for ring in self.experiment.arena.rings
            },
            "pair_counts": {
                str(ring): int(pair_counts.get(ring, 0))
                for ring in self.experiment.arena.rings
            },
            **weighted_metadata,
        }
        result["wave_plan"] = wave_plan
        result["wave_history"] = [*previous_history, wave_plan]
        result["arena_seed_block"] = arena_config.seed
        unique = {(pair.ring, pair.pair): pair for pair in accumulated}
        for pair in completed_pairs:
            key = (pair.ring, pair.pair)
            existing = unique.get(key)
            if existing is not None and existing != pair:
                raise ValueError("arena wave changed a persisted pair")
            unique[key] = pair
        accumulated[:] = [unique[key] for key in sorted(unique)]
        if pair_ratios:
            pair_counts_after = self._ring_pair_counts(accumulated)
            wave_plan["complete_blocks_after"] = self._complete_blocks(
                pair_counts_after,
                pair_ratios,
            )
            wave_plan["pair_counts_after"] = {
                str(ring): pair_counts_after[ring] for ring in arena_config.rings
            }
        result["schema_version"] = ARENA_RESULT_SCHEMA_VERSION
        result["pairs"] = [asdict(pair) for pair in accumulated]
        result.update(summarize_completed_arena_pairs(accumulated, arena_config))
        if previous is not None:
            previous_games = previous.get("games", [])
            result_games = result.get("games", [])
            if not isinstance(previous_games, list) or not isinstance(
                result_games, list
            ):
                raise ValueError("persisted arena games are invalid")
            result["games"] = [
                *previous_games,
                *result_games,
            ]
        self._annotate_result(result, candidate, champion)
        max_reached = self._max_allocation_reached(accumulated)
        promotion_result = result.get("promotion")
        if not isinstance(promotion_result, dict) or not isinstance(
            promotion_result.get("decision"), str
        ):
            raise ValueError("arena result promotion is invalid")
        decision = promotion_result["decision"]
        if decision == "continue" and max_reached:
            decision = "reject_max_pairs"
            promotion_result["decision"] = decision
        terminal = decision != "continue"
        result["terminal"] = terminal
        result_path = self._result_path(candidate, champion)
        atomic_json(result_path, result)
        if decision == "promote":
            write_model_pointer(
                self.champion_path,
                candidate,
                role="champion",
                promotion_result=str(result_path.resolve()),
            )
            if progress is not None:
                progress(
                    phase="promoted",
                    model_identity=candidate.model_identity,
                    model_step=candidate.model_step,
                )
        elif progress is not None:
            progress(
                phase="arena_terminal" if terminal else "arena_continue",
                model_identity=candidate.model_identity,
                decision=decision,
                pairs=len(accumulated),
            )
        self._write_status(
            candidate=candidate,
            champion=(candidate if decision == "promote" else champion),
            decision=decision,
            terminal=terminal,
        )
        retention = self.experiment.orchestration.retention
        if terminal and retention.enabled:
            gc_metrics = collect_model_garbage(
                self.candidate_path.parent,
                retain_candidate_manifests=(retention.candidate_manifests),
                dry_run=retention.dry_run,
                referenced_result_directory=self.results_directory,
            )
            result["gc"] = gc_metrics
            atomic_json(result_path, result)
            if progress is not None:
                progress(phase="model_gc", **gc_metrics)
        return decision, terminal

    @staticmethod
    def _pair_chunks(
        starts: Mapping[int, int],
        counts: Mapping[int, int],
        *,
        chunk_size: int,
        pair_ratios: Mapping[int, int] | None = None,
        existing_counts: Mapping[int, int] | None = None,
    ) -> list[tuple[dict[int, int], dict[int, int]]]:
        maximum = max(counts.values(), default=0)
        legacy_chunks = [
            (
                {ring: int(start) + offset for ring, start in starts.items()},
                {
                    ring: min(chunk_size, max(0, int(count) - offset))
                    for ring, count in counts.items()
                },
            )
            for offset in range(0, maximum, chunk_size)
        ]
        if not pair_ratios or maximum <= chunk_size:
            return legacy_chunks

        current_counts = {
            ring: int((existing_counts or {}).get(ring, 0)) for ring in starts
        }
        complete_blocks_before = PromotionSupervisor._complete_blocks(
            current_counts,
            pair_ratios,
        )
        final_counts = {
            ring: current_counts[ring] + max(0, int(counts.get(ring, 0)))
            for ring in starts
        }
        complete_blocks_target = PromotionSupervisor._complete_blocks(
            final_counts,
            pair_ratios,
        )
        if complete_blocks_target <= complete_blocks_before:
            return legacy_chunks

        # A weighted persistence boundary must finish whole macro blocks. Treat
        # a single ratio block as atomic even when it exceeds pair_chunk_size.
        blocks_per_chunk = max(1, chunk_size // max(pair_ratios.values()))
        scheduled = {ring: 0 for ring in starts}
        chunks: list[tuple[dict[int, int], dict[int, int]]] = []
        chunk_target = complete_blocks_before
        while chunk_target < complete_blocks_target:
            chunk_target = min(
                complete_blocks_target,
                chunk_target + blocks_per_chunk,
            )
            chunk_counts = {}
            for ring in starts:
                running = current_counts[ring] + scheduled[ring]
                remaining = max(0, int(counts.get(ring, 0)) - scheduled[ring])
                needed = max(0, chunk_target * int(pair_ratios[ring]) - running)
                chunk_counts[ring] = min(remaining, needed)
            if not any(chunk_counts.values()):
                break
            chunks.append(
                (
                    {ring: int(starts[ring]) + scheduled[ring] for ring in starts},
                    chunk_counts,
                )
            )
            for ring, count in chunk_counts.items():
                scheduled[ring] += count

        remaining_counts = {
            ring: max(0, int(counts.get(ring, 0)) - scheduled[ring]) for ring in starts
        }
        if any(remaining_counts.values()):
            chunks.append(
                (
                    {ring: int(starts[ring]) + scheduled[ring] for ring in starts},
                    remaining_counts,
                )
            )
        return chunks

    def _ring_pair_counts(
        self,
        accumulated: list[ArenaPair],
    ) -> dict[int, int]:
        return {
            ring: sum(pair.ring == ring for pair in accumulated)
            for ring in self.experiment.arena.rings
        }

    @staticmethod
    def _complete_blocks(
        pair_counts: Mapping[int, int],
        pair_ratios: Mapping[int, int],
    ) -> int:
        return min(
            (
                int(pair_counts.get(ring, 0)) // int(ratio)
                for ring, ratio in pair_ratios.items()
            ),
            default=0,
        )

    def _max_allocation_reached(
        self,
        accumulated: list[ArenaPair],
    ) -> bool:
        pair_ratios = self.experiment.arena.promotion_pair_ratios
        if pair_ratios:
            return (
                self._complete_blocks(
                    self._ring_pair_counts(accumulated),
                    pair_ratios,
                )
                >= self.experiment.arena.weighted_max_blocks
            )
        return all(
            sum(pair.ring == ring for pair in accumulated)
            >= self.experiment.arena.max_pairs_per_ring
            for ring in self.experiment.arena.rings
        )

    def _wave_plan(
        self,
        accumulated: list[ArenaPair],
    ) -> tuple[dict[int, int], dict[int, int]]:
        existing_counts = self._ring_pair_counts(accumulated)
        starts = {
            ring: (
                max(
                    (pair.pair for pair in accumulated if pair.ring == ring),
                    default=-1,
                )
                + 1
            )
            for ring in self.experiment.arena.rings
        }
        pair_ratios = self.experiment.arena.promotion_pair_ratios
        if pair_ratios:
            complete_blocks = self._complete_blocks(existing_counts, pair_ratios)
            if complete_blocks >= self.experiment.arena.weighted_max_blocks:
                target_blocks = self.experiment.arena.weighted_max_blocks
            elif complete_blocks < self.experiment.arena.weighted_initial_blocks:
                target_blocks = self.experiment.arena.weighted_initial_blocks
            else:
                target_blocks = min(
                    self.experiment.arena.weighted_max_blocks,
                    complete_blocks
                    + self.experiment.arena.weighted_continuation_blocks,
                )
            return starts, {
                ring: max(
                    0,
                    target_blocks * int(pair_ratios[ring]) - existing_counts[ring],
                )
                for ring in self.experiment.arena.rings
            }

        counts = {}
        for ring, existing in existing_counts.items():
            remaining = self.experiment.arena.max_pairs_per_ring - existing
            required_for_minimum = max(
                0,
                self.experiment.arena.minimum_pairs_per_ring - existing,
            )
            wave = (
                min(self.experiment.arena.pairs_per_ring, required_for_minimum)
                if required_for_minimum
                else (
                    self.experiment.arena.continuation_pairs_per_ring
                    or self.experiment.arena.pairs_per_ring
                )
            )
            counts[ring] = min(wave, remaining)
        return starts, counts

    def _reject_max_pairs(
        self,
        *,
        candidate: ModelManifest,
        champion: ModelManifest,
        result: dict[str, object],
        progress: Callable[..., None] | None,
    ) -> None:
        result["schema_version"] = ARENA_RESULT_SCHEMA_VERSION
        promotion_result = result.get("promotion")
        if not isinstance(promotion_result, dict):
            raise ValueError("persisted arena promotion is invalid")
        promotion_result["decision"] = "reject_max_pairs"
        result["terminal"] = True
        self._annotate_result(result, candidate, champion)
        atomic_json(self._result_path(candidate, champion), result)
        self._write_status(
            candidate=candidate,
            champion=champion,
            decision="reject_max_pairs",
            terminal=True,
        )
        if progress is not None:
            progress(
                phase="candidate_terminal",
                champion_step=champion.model_step,
                candidate_step=candidate.model_step,
                decision="reject_max_pairs",
            )

    @staticmethod
    def _annotate_result(
        result: dict[str, object],
        candidate: ModelManifest,
        champion: ModelManifest,
    ) -> None:
        result["result_kind"] = "promotion"
        result["candidate_manifest"] = str(
            (candidate.artifact_manifest or candidate.path).resolve()
        )
        result["champion_manifest"] = str(
            (champion.artifact_manifest or champion.path).resolve()
        )

    def _arena_config(
        self,
        candidate: ModelManifest,
        champion: ModelManifest,
    ) -> ArenaConfig:
        material = (
            f"promotion-arena-v1\0{self.experiment.arena.seed}\0"
            f"{candidate.model_identity}\0{champion.model_identity}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return replace(self.experiment.arena, seed=seed)

    def _resume_cutover(self) -> tuple[int, str] | None:
        cutover_path = self.candidate_path.parent / "resume-cutover.json"
        if not cutover_path.is_file():
            return None
        try:
            payload = json.loads(cutover_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read resume cutover: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("format") != "startrain.resume-cutover"
            or payload.get("schema_version") != 1
            or payload.get("run_id") != self.run_identity.run_id
            or payload.get("generation_family") != self.run_identity.generation_family
            or isinstance(payload.get("created_ns"), bool)
            or not isinstance(payload.get("created_ns"), int)
            or not isinstance(payload.get("checkpoint_sha256"), str)
        ):
            raise ValueError("resume cutover is invalid for promotion")
        return int(payload["created_ns"]), str(payload["checkpoint_sha256"])

    def _candidate_manifests(self) -> list[ModelManifest]:
        output: list[ModelManifest] = []
        cutover = self._resume_cutover()
        cutover_ns = cutover[0] if cutover is not None else 0
        cutover_sha256 = cutover[1] if cutover is not None else None
        for path in self.manifest_directory.glob("manifest-*.json"):
            manifest = self._manifest_cache.get(path)
            if manifest is None:
                manifest = load_model_manifest(path)
                self._manifest_cache[path] = manifest
            if (
                manifest.run_id == self.run_identity.run_id
                and manifest.generation_family == self.run_identity.generation_family
                and (
                    cutover_sha256 is None
                    or manifest.checkpoint_sha256 == cutover_sha256
                    or manifest.published_ns >= cutover_ns
                )
            ):
                output.append(manifest)
        return sorted(output, key=lambda item: (item.model_step, item.model_identity))

    def _newer_candidate(
        self,
        candidate: ModelManifest,
        champion: ModelManifest,
    ) -> ModelManifest | None:
        candidate_key = (candidate.model_step, candidate.model_identity)
        return max(
            (
                item
                for item in self._candidate_manifests()
                if item.model_identity != champion.model_identity
                and (item.model_step, item.model_identity) > candidate_key
            ),
            key=lambda item: (item.model_step, item.model_identity),
            default=None,
        )

    @contextmanager
    def _gpu_pause(
        self,
        *,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
        candidate_identity: str,
    ):
        if self.gpu_pause_path is None:
            yield
            return
        promotion = self.experiment.orchestration.promotion
        lease = CoordinatorPauseLease(
            request_path=self.gpu_pause_path,
            gpu_id=promotion.gpu_id,
            candidate_identity=candidate_identity,
            ready_timeout_seconds=promotion.pause_ready_timeout_seconds,
            release_timeout_seconds=promotion.pause_release_timeout_seconds,
            heartbeat_interval_seconds=(
                self.experiment.orchestration.shutdown.heartbeat_interval_seconds
            ),
            poll_seconds=max(
                0.01,
                min(
                    0.25,
                    self.experiment.orchestration.shutdown.monitor_interval_seconds,
                ),
            ),
            stop_requested=stop_requested,
            progress=progress,
            events_path=self.pause_events_path,
            clock=self.clock,
            sleep=self.sleep,
        )
        with lease:
            yield

    def _result_path(self, candidate: ModelManifest, champion: ModelManifest) -> Path:
        return self.results_directory / (
            f"{candidate.model_identity}-vs-{champion.model_identity}.json"
        )

    def _read_result(
        self, candidate: ModelManifest, champion: ModelManifest
    ) -> dict[str, object] | None:
        path = self._result_path(candidate, champion)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            return None
        result_kind = payload.get("result_kind") if isinstance(payload, dict) else None
        valid = (
            isinstance(payload, dict)
            and payload.get("schema_version") == ARENA_RESULT_SCHEMA_VERSION
            and payload.get("candidate") == candidate.model_identity
            and payload.get("baseline") == champion.model_identity
            and isinstance(payload.get("promotion"), dict)
            and (not isinstance(result_kind, str) or result_kind == "promotion")
        )
        return payload if valid else None

    @staticmethod
    def _pairs_from_result(
        result: dict[str, object] | None,
    ) -> list[ArenaPair]:
        if result is None:
            return []
        payload = result.get("pairs", [])
        if not isinstance(payload, list):
            raise ValueError("persisted arena pairs are invalid")
        output = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("persisted arena pair is invalid")
            values = dict(item)
            outcomes = values.get("outcomes")
            if not isinstance(outcomes, (list, tuple)) or len(outcomes) != 2:
                raise ValueError("persisted arena pair outcomes are invalid")
            values["outcomes"] = (int(outcomes[0]), int(outcomes[1]))
            output.append(ArenaPair(**values))
        return output

    def _mark_superseded(
        self,
        candidate: ModelManifest,
        champion: ModelManifest,
        *,
        superseded_by: ModelManifest,
    ) -> bool:
        previous = self._read_result(candidate, champion)
        if previous is not None and bool(previous.get("terminal")):
            return False
        payload: dict[str, object] = previous or {
            "schema_version": ARENA_RESULT_SCHEMA_VERSION,
            "candidate": candidate.model_identity,
            "baseline": champion.model_identity,
            "pairs": [],
            "games": [],
            "promotion": {},
        }
        payload["schema_version"] = ARENA_RESULT_SCHEMA_VERSION
        payload["terminal"] = True
        payload["promotion"] = {
            "decision": "superseded",
            "superseded_by": superseded_by.model_identity,
        }
        self._annotate_result(payload, candidate, champion)
        atomic_json(self._result_path(candidate, champion), payload)
        return True

    def _write_status(
        self,
        *,
        candidate: ModelManifest,
        champion: ModelManifest,
        decision: str,
        terminal: bool,
    ) -> None:
        prior: dict[str, object] = {}
        if self.status_path.is_file():
            try:
                with self.status_path.open("r", encoding="utf-8") as stream:
                    loaded = json.load(stream)
                if isinstance(loaded, dict):
                    prior = loaded
            except (OSError, json.JSONDecodeError):
                prior = {}
        raw_streak = prior.get("consecutive_terminal_rejections", 0)
        streak = (
            raw_streak
            if isinstance(raw_streak, int) and not isinstance(raw_streak, bool)
            else 0
        )
        cutover = self._resume_cutover()
        cutover_created_ns = cutover[0] if cutover is not None else None
        if prior.get("cutover_created_ns") != cutover_created_ns:
            streak = 0
        prior_candidate = prior.get("candidate_identity")
        prior_terminal = bool(prior.get("terminal"))
        prior_decision = prior.get("decision")
        rejection = decision in (
            "reject",
            "reject_ring_regression",
            "reject_max_pairs",
        )
        if decision in ("promote", "bootstrap"):
            streak = 0
        elif (
            terminal
            and rejection
            and not (
                prior_candidate == candidate.model_identity
                and prior_terminal
                and prior_decision
                in ("reject", "reject_ring_regression", "reject_max_pairs")
            )
        ):
            streak += 1
        atomic_json(
            self.status_path,
            {
                "schema_version": 1,
                "candidate_identity": candidate.model_identity,
                "candidate_step": candidate.model_step,
                "champion_identity": champion.model_identity,
                "champion_step": champion.model_step,
                "decision": decision,
                "terminal": terminal,
                "consecutive_terminal_rejections": streak,
                "cutover_created_ns": cutover_created_ns,
                "updated_ns": time.time_ns(),
            },
        )


def promotion_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate immutable candidates and atomically promote champions"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-identity", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--heartbeat", required=True)
    parser.add_argument("--device")
    parser.add_argument("--gpu-pause")
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args(argv)

    experiment = load_config(arguments.config)
    run_identity = load_run_identity(arguments.run_identity)
    device = resolve_device_string(
        arguments.device or experiment.orchestration.promotion.device
    )
    native = load_star_native(required=True)
    assert native is not None
    stop = SignalLatch()
    stop.install()
    heartbeat = HeartbeatReporter(
        arguments.heartbeat,
        worker="arena-promotion",
        interval_seconds=(experiment.orchestration.shutdown.heartbeat_interval_seconds),
    )
    heartbeat.start()
    try:
        evaluated = PromotionSupervisor(
            experiment=experiment,
            run_identity=run_identity,
            candidate_path=arguments.candidate,
            champion_path=arguments.champion,
            results_directory=arguments.results,
            native_module=native,
            device=device,
            gpu_pause_path=arguments.gpu_pause,
        ).run(
            stop_requested=stop.is_set,
            progress=heartbeat.advance,
            once=arguments.once,
        )
    finally:
        heartbeat.close(final_phase="stopped" if stop.is_set() else "completed")
    print(json.dumps({"evaluated": evaluated}, sort_keys=True))


if __name__ == "__main__":
    promotion_main()
