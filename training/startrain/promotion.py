"""Immutable candidate evaluation and atomic champion promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import TracebackType

import torch

from .arena import (
    ARENA_RESULT_SCHEMA_VERSION,
    ArenaPair,
    ArenaRunner,
    summarize_arena_pairs,
)
from .checkpoint import (
    ModelManifest,
    collect_model_garbage,
    load_ema_checkpoint,
    load_model_manifest,
)
from .checkpoint import write_model_pointer
from .config import ArenaConfig, ExperimentConfig, load_config
from .inference import GraphInferenceAdapter, InferenceConfig
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
) -> GraphInferenceAdapter:
    model = GraphResTNet(experiment.model).to(device)
    metadata = load_ema_checkpoint(
        manifest.checkpoint,
        model=model,
        expected_model_config=asdict(experiment.model),
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
    inference_model = maybe_compile_model(
        model,
        enabled=experiment.train.compile,
        dynamic=True,
        fullgraph=True,
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
        self.clock = clock
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
                    self._mark_superseded(stale, champion, superseded_by=champion)
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
                        self._mark_superseded(
                            skipped,
                            champion,
                            superseded_by=candidate,
                        )
            if previous is not None and bool(previous.get("terminal")):
                if once:
                    return evaluated
                self.sleep(promotion.poll_seconds)
                continue
            accumulated = self._pairs_from_result(previous)
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
            counts = {
                ring: min(
                    self.experiment.arena.pairs_per_ring,
                    self.experiment.arena.max_pairs_per_ring
                    - sum(pair.ring == ring for pair in accumulated),
                )
                for ring in self.experiment.arena.rings
            }
            if all(count <= 0 for count in counts.values()):
                assert previous is not None
                previous["schema_version"] = ARENA_RESULT_SCHEMA_VERSION
                previous_promotion = previous.get("promotion")
                if not isinstance(previous_promotion, dict):
                    raise ValueError("persisted arena promotion is invalid")
                previous_promotion["decision"] = "reject_max_pairs"
                previous["terminal"] = True
                atomic_json(self._result_path(candidate, champion), previous)
                self._write_status(
                    candidate=candidate,
                    champion=champion,
                    decision="reject_max_pairs",
                    terminal=True,
                )
                if once:
                    return evaluated
                continue
            try:
                with self._gpu_pause(
                    stop_requested=stop_requested,
                    progress=progress,
                    candidate_identity=candidate.model_identity,
                ):
                    self._evaluate_round(
                        candidate=candidate,
                        champion=champion,
                        previous=previous,
                        accumulated=accumulated,
                        starts=starts,
                        counts=counts,
                        progress=progress,
                    )
                    evaluated += 1
            except PauseLeaseInterrupted:
                return evaluated
            if once:
                return evaluated
        return evaluated

    def _evaluate_round(
        self,
        *,
        candidate: ModelManifest,
        champion: ModelManifest,
        previous: dict[str, object] | None,
        accumulated: list[ArenaPair],
        starts: dict[int, int],
        counts: dict[int, int],
        progress: Callable[..., None] | None,
    ) -> None:
        round_started = time.perf_counter()
        metric_device = torch.device(self.device)
        collect_cuda_metrics = (
            metric_device.type == "cuda" and torch.cuda.is_available()
        )
        if collect_cuda_metrics:
            torch.cuda.reset_peak_memory_stats(metric_device)
        arena_config = self._arena_config(candidate, champion)
        if progress is not None:
            progress(
                phase="arena",
                candidate_step=candidate.model_step,
                champion_step=champion.model_step,
            )
        candidate_evaluator = load_manifest_evaluator(
            self.experiment, candidate, device=self.device
        )
        try:
            champion_evaluator = load_manifest_evaluator(
                self.experiment, champion, device=self.device
            )
            try:
                result = ArenaRunner(
                    native_module=self.native,
                    candidate=candidate_evaluator,
                    baseline=champion_evaluator,
                    config=arena_config,
                ).run(
                    progress=progress,
                    pair_starts=starts,
                    pair_counts=counts,
                )
                if collect_cuda_metrics:
                    torch.cuda.synchronize(metric_device)
                evaluation_metrics = result.get("evaluation_metrics")
                if evaluation_metrics is None:
                    evaluation_metrics = {}
                    result["evaluation_metrics"] = evaluation_metrics
                elif not isinstance(evaluation_metrics, dict):
                    raise ValueError("arena result evaluator metrics are invalid")
                evaluation_metrics["round_wall_seconds"] = (
                    time.perf_counter() - round_started
                )
                evaluation_metrics["peak_cuda_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(metric_device)
                    if collect_cuda_metrics
                    else None
                )
                evaluation_metrics["peak_cuda_reserved_bytes"] = (
                    torch.cuda.max_memory_reserved(metric_device)
                    if collect_cuda_metrics
                    else None
                )
                result["arena_seed_block"] = arena_config.seed
                accumulated.extend(self._pairs_from_result(result))
                unique = {(pair.ring, pair.pair): pair for pair in accumulated}
                accumulated[:] = [unique[key] for key in sorted(unique)]
                result["schema_version"] = ARENA_RESULT_SCHEMA_VERSION
                result["pairs"] = [asdict(pair) for pair in accumulated]
                result.update(summarize_arena_pairs(accumulated, arena_config))
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
                max_reached = all(
                    sum(pair.ring == ring for pair in accumulated)
                    >= self.experiment.arena.max_pairs_per_ring
                    for ring in self.experiment.arena.rings
                )
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
            finally:
                del champion_evaluator
        finally:
            del candidate_evaluator
            if collect_cuda_metrics:
                torch.cuda.synchronize(metric_device)
                torch.cuda.empty_cache()

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

    def _candidate_manifests(self) -> list[ModelManifest]:
        output: list[ModelManifest] = []
        for path in self.manifest_directory.glob("manifest-*.json"):
            manifest = self._manifest_cache.get(path)
            if manifest is None:
                manifest = load_model_manifest(path)
                self._manifest_cache[path] = manifest
            if (
                manifest.run_id == self.run_identity.run_id
                and manifest.generation_family == self.run_identity.generation_family
            ):
                output.append(manifest)
        return sorted(output, key=lambda item: (item.model_step, item.model_identity))

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
        valid = (
            isinstance(payload, dict)
            and payload.get("schema_version") == ARENA_RESULT_SCHEMA_VERSION
            and payload.get("candidate") == candidate.model_identity
            and payload.get("baseline") == champion.model_identity
            and isinstance(payload.get("promotion"), dict)
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
    ) -> None:
        previous = self._read_result(candidate, champion)
        if previous is not None and bool(previous.get("terminal")):
            return
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
        atomic_json(self._result_path(candidate, champion), payload)

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
    device = arguments.device or experiment.orchestration.promotion.device
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
