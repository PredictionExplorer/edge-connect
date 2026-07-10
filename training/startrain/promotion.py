"""Immutable candidate evaluation and atomic champion promotion."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

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
from .config import ExperimentConfig, load_config
from .inference import GraphInferenceAdapter, InferenceConfig
from .model import GraphResTNet
from .native import load_star_native
from .runtime import (
    HeartbeatReporter,
    RunIdentity,
    SignalLatch,
    atomic_json,
    load_run_identity,
)


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
    return GraphInferenceAdapter(
        model.eval(),
        device=device,
        config=InferenceConfig(
            precision=experiment.train.precision,
            score_utility_weight=experiment.selfplay.score_utility_weight,
            initial_pass_logit_penalty=(experiment.selfplay.initial_pass_logit_penalty),
        ),
        model_version=manifest.model_version,
        model_step=manifest.model_step,
        model_identity=manifest.model_identity,
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
                time.sleep(promotion.poll_seconds)
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
            candidate = max(
                viable,
                key=lambda item: (item.model_step, item.model_identity),
                default=None,
            )
            if candidate is None:
                if progress is not None:
                    progress(
                        phase="waiting_for_candidate",
                        champion_step=champion.model_step,
                    )
                if once:
                    return evaluated
                time.sleep(promotion.poll_seconds)
                continue
            for skipped in viable:
                if skipped.model_identity != candidate.model_identity:
                    self._mark_superseded(skipped, champion, superseded_by=candidate)
            previous = self._read_result(candidate, champion)
            if previous is not None and bool(previous.get("terminal")):
                if once:
                    return evaluated
                time.sleep(promotion.poll_seconds)
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
            if progress is not None:
                progress(
                    phase="arena",
                    candidate_step=candidate.model_step,
                    champion_step=champion.model_step,
                )
            with self._gpu_pause():
                candidate_evaluator = load_manifest_evaluator(
                    self.experiment, candidate, device=self.device
                )
                champion_evaluator = load_manifest_evaluator(
                    self.experiment, champion, device=self.device
                )
                result = ArenaRunner(
                    native_module=self.native,
                    candidate=candidate_evaluator,
                    baseline=champion_evaluator,
                    config=self.experiment.arena,
                ).run(
                    progress=progress,
                    pair_starts=starts,
                    pair_counts=counts,
                )
            accumulated.extend(self._pairs_from_result(result))
            unique = {(pair.ring, pair.pair): pair for pair in accumulated}
            accumulated = [unique[key] for key in sorted(unique)]
            result["schema_version"] = ARENA_RESULT_SCHEMA_VERSION
            result["pairs"] = [asdict(pair) for pair in accumulated]
            result.update(summarize_arena_pairs(accumulated, self.experiment.arena))
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
            evaluated += 1
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
            del candidate_evaluator, champion_evaluator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if once:
                return evaluated
        return evaluated

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
    def _gpu_pause(self):
        if self.gpu_pause_path is None:
            yield
            return
        atomic_json(
            self.gpu_pause_path,
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "started_ns": time.time_ns(),
            },
        )
        try:
            yield
        finally:
            self.gpu_pause_path.unlink(missing_ok=True)

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
            and payload.get("schema_version") in (1, ARENA_RESULT_SCHEMA_VERSION)
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
