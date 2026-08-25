from __future__ import annotations

import hashlib
import grp
import json
import os
import pwd
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.run_terminal_boundary_pipeline import (
    DefaultTerminalBoundaryAdapters,
    POLICY_REPORT,
    STATE_REPORT,
    TerminalBoundaryExecutionError,
    TerminalBoundaryManifestError,
    main,
    recover_terminal_boundary_pipeline,
    run_terminal_boundary_pipeline,
    verify_queue_activation_manifest,
)


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _model_manifest(
    root: Path,
    *,
    name: str,
    identity: str,
    step: int,
) -> Path:
    path = root / "learner" / "manifests" / f"{name}.json"
    _write_json(
        path,
        {
            "format": "startrain.model-manifest",
            "schema_version": 3,
            "model_identity": identity,
            "model_step": step,
            "model_version": f"step-{step}",
            "published_ns": step,
            "run_id": "source-run",
            "generation_family": "source-family",
        },
    )
    return path


def _model_pointer(
    path: Path,
    *,
    role: str,
    manifest: Path,
    identity: str,
    step: int,
    updated_ns: int,
    promotion_result: Path | None = None,
) -> None:
    document = {
        "format": "startrain.model-pointer",
        "schema_version": 2,
        "role": role,
        "manifest": str(manifest.relative_to(path.parent)),
        "manifest_sha256": _sha256(manifest),
        "manifest_bytes": manifest.stat().st_size,
        "model_identity": identity,
        "model_step": step,
        "run_id": "source-run",
        "generation_family": "source-family",
        "updated_ns": updated_ns,
    }
    if promotion_result is not None:
        document["promotion_result"] = os.path.relpath(promotion_result, path.parent)
    _write_json(path, document)


class BoundaryFixture:
    def __init__(self, tmp_path: Path) -> None:
        self.root = (tmp_path / "source").resolve()
        self.root.mkdir()
        self.candidate_identity = "sha256-" + "c" * 64
        self.champion_identity = "sha256-" + "b" * 64
        self.candidate_manifest = _model_manifest(
            self.root,
            name="candidate",
            identity=self.candidate_identity,
            step=20,
        )
        self.baseline_manifest = _model_manifest(
            self.root,
            name="baseline",
            identity=self.champion_identity,
            step=10,
        )
        self.candidate_pointer = self.root / "learner" / "candidate.json"
        self.champion_pointer = self.root / "learner" / "champion.json"
        _model_pointer(
            self.candidate_pointer,
            role="candidate",
            manifest=self.candidate_manifest,
            identity=self.candidate_identity,
            step=20,
            updated_ns=90,
        )
        _model_pointer(
            self.champion_pointer,
            role="champion",
            manifest=self.baseline_manifest,
            identity=self.champion_identity,
            step=10,
            updated_ns=80,
        )
        _write_json(
            self.root / "run.json",
            {
                "schema_version": 1,
                "run_id": "source-run",
                "generation_family": "source-family",
                "created_ns": 10,
            },
        )
        self.status_path = self.root / "arena" / "promotion-status.json"
        _write_json(
            self.status_path,
            {
                "schema_version": 1,
                "candidate_identity": self.candidate_identity,
                "candidate_step": 20,
                "champion_identity": self.champion_identity,
                "champion_step": 10,
                "decision": "continue",
                "terminal": False,
                "updated_ns": 100,
            },
        )
        self.result_path = (
            self.root
            / "arena"
            / f"{self.candidate_identity}-vs-{self.champion_identity}.json"
        )
        self.arm_sha256 = _sha256(self.status_path)

        self.profile = tmp_path / "release" / "training" / "configs" / "source.yaml"
        self.profile.parent.mkdir(parents=True)
        self.profile.write_text("profile: continuous\n", encoding="utf-8")
        self.training_dir = self.profile.parents[1]
        runtime_paths = {}
        for name in (
            "run_terminal_boundary_pipeline.py",
            "run_frozen_replay_optimizer_calibration.py",
            "run_frozen_replay_optimizer_calibration_queue.py",
            "compare_frozen_replay_optimizer_calibration.py",
            "prepare_elo_ablation.py",
            "fork_elo_ablation.py",
            "prepare_champion_warm_start.py",
        ):
            path = self.training_dir / "scripts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# fixture {name}\n", encoding="utf-8")
            runtime_paths[name] = path
        for name in (
            "replay_manifest_backup.py",
            "training_disaster_recovery.py",
            "run_elo_ablation_queue.py",
            "run_elo_ablation.py",
            "compare_elo_ablation.py",
        ):
            path = self.training_dir / "scripts" / name
            path.write_text(f"# fixture {name}\n", encoding="utf-8")
            runtime_paths[name] = path
        for name in (
            "runtime.py",
            "training.py",
            "learner.py",
            "checkpoint.py",
            "continuity.py",
        ):
            path = self.training_dir / "startrain" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# fixture {name}\n", encoding="utf-8")
            runtime_paths[f"startrain/{name}"] = path
        snapshot_module = self.training_dir / "starserve" / "snapshot.py"
        snapshot_module.parent.mkdir(parents=True, exist_ok=True)
        snapshot_module.write_text("# fixture snapshot.py\n", encoding="utf-8")
        runtime_paths["starserve/snapshot.py"] = snapshot_module
        release_manifest = self.training_dir.parent / "release-manifest.json"
        _write_json(release_manifest, {"commit": "a" * 40})
        self.source_unit = tmp_path / "release" / "source.service"
        self.source_unit.write_text(
            "[Service]\nExecStart=/bin/true\n", encoding="utf-8"
        )
        self.base_config = tmp_path / "release" / "base.yaml"
        self.base_config.write_text("profile: continuous\n", encoding="utf-8")
        self.queue_unit = tmp_path / "release" / "queue.service"
        self.queue_unit.write_text("[Service]\nExecStart=/bin/true\n", encoding="utf-8")
        self.finalize_unit = tmp_path / "release" / "finalize.service"
        self.finalize_unit.write_text(
            "[Service]\nExecStart=/bin/true\n", encoding="utf-8"
        )
        self.environment = tmp_path / "release" / "queue.env"
        self.environment.write_text("PYTHONUNBUFFERED=1\n", encoding="utf-8")
        self.continuity = tmp_path / "release" / "continuity.json"
        _write_json(self.continuity, {"fixture": "continuity"})

        state_root = (tmp_path / "boundary-state").resolve()
        backup_mount = (tmp_path / "backup-mount").resolve()
        backup_root = backup_mount / "source"
        self.policy_path = (tmp_path / "terminal-boundary-policy.json").resolve()
        self.policy: dict[str, Any] = {
            "schema_version": 1,
            "report": POLICY_REPORT,
            "policy_id": "terminal-calibration",
            "policy_version": 1,
            "state_path": str(state_root / "state.json"),
            "runtime": {
                "training_dir": str(self.training_dir),
                "release_manifest": _artifact(release_manifest),
                "terminal_runner": _artifact(
                    runtime_paths["run_terminal_boundary_pipeline.py"]
                ),
                "calibration_runner": _artifact(
                    runtime_paths["run_frozen_replay_optimizer_calibration.py"]
                ),
                "calibration_queue": _artifact(
                    runtime_paths["run_frozen_replay_optimizer_calibration_queue.py"]
                ),
                "calibration_comparator": _artifact(
                    runtime_paths["compare_frozen_replay_optimizer_calibration.py"]
                ),
                "ablation_preparer": _artifact(
                    runtime_paths["prepare_elo_ablation.py"]
                ),
                "ablation_forker": _artifact(runtime_paths["fork_elo_ablation.py"]),
                "warm_starter": _artifact(
                    runtime_paths["prepare_champion_warm_start.py"]
                ),
                "runtime_module": _artifact(runtime_paths["startrain/runtime.py"]),
                "training_module": _artifact(runtime_paths["startrain/training.py"]),
                "learner_module": _artifact(runtime_paths["startrain/learner.py"]),
                "checkpoint_module": _artifact(
                    runtime_paths["startrain/checkpoint.py"]
                ),
                "continuity_module": _artifact(
                    runtime_paths["startrain/continuity.py"]
                ),
                "snapshot_module": _artifact(runtime_paths["starserve/snapshot.py"]),
                "replay_backup": _artifact(runtime_paths["replay_manifest_backup.py"]),
                "disaster_recovery": _artifact(
                    runtime_paths["training_disaster_recovery.py"]
                ),
                "queue_generator": _artifact(
                    runtime_paths["run_elo_ablation_queue.py"]
                ),
                "queue_runner": _artifact(runtime_paths["run_elo_ablation.py"]),
                "elo_comparator": _artifact(runtime_paths["compare_elo_ablation.py"]),
            },
            "arm": {
                "promotion_status_sha256": self.arm_sha256,
                "promotion_status_updated_ns": 100,
            },
            "source": {
                "run_root": str(self.root),
                "run_identity": {
                    "run_id": "source-run",
                    "generation_family": "source-family",
                    "created_ns": 10,
                },
                "profile": _artifact(self.profile),
                "unit": {
                    **_artifact(self.source_unit),
                    "name": "edgeconnect-source.service",
                },
                "promotion_status": str(self.status_path),
                "candidate_pointer": str(self.candidate_pointer),
                "champion_pointer": str(self.champion_pointer),
                "arena_root": str(self.root / "arena"),
                "coordinator_status": str(self.root / "status" / "coordinator.json"),
                "coordinator_lock": str(self.root / "coordinator.lock"),
                "arena_pause_request": str(
                    self.root / "status" / "arena-gpu-pause.json"
                ),
                "hardware_report": str(self.root / "status" / "hardware-health.json"),
                "hardware_max_age_seconds": 180,
                "gpu_ids": [0, 1],
                "stop_timeout_seconds": 60,
            },
            "operator_hold_path": str(state_root / "operator-hold.json"),
            "backup": {
                "replay_retain": 3,
                "replay_max_total_bytes": 1024 * 1024,
                "disaster_backup_root": str(backup_root),
                "disaster_backup_mount": str(backup_mount),
                "off_host_acknowledgement_path": str(
                    backup_root / "acknowledgements" / "mac.json"
                ),
                "off_host_ack_timeout_seconds": 60,
                "off_host_ack_poll_seconds": 1,
            },
            "snapshot": {
                "destination": str(tmp_path / "champion-export"),
                "pin_path": str(state_root / "champion-export-pin.json"),
            },
            "calibration": {
                "base_config": _artifact(self.base_config),
                "output_dir": str(tmp_path / "calibration-plan"),
                "run_root_parent": str(tmp_path / "calibration-runs"),
                "run_id": "source-run",
                "runtime_user": "ubuntu",
                "runtime_group": "ubuntu",
                "prefix": "terminal",
                "seed": 47,
                "wall_budget_hours": 1,
                "leaf_budget": 100,
                "guard_floor_elo": -35,
                "treatments": ["control"],
                "frozen_replay": {
                    "output_root": str(tmp_path / "frozen-calibration"),
                    "screen_plan_path": str(
                        tmp_path / "screen-plan" / "ablation-plan.json"
                    ),
                    "steps": 2,
                    "device": "cpu",
                    "budget_h100_hours": 2,
                    "batch_size": 2,
                    "evaluation_batch_size": 2,
                    "max_samples": 16,
                    "holdout_fraction": 0.25,
                    "checkpoint_interval": 1,
                    "screen_wall_budget_hours": 1,
                    "screen_leaf_budget": 100,
                },
            },
            "queue": {
                "training_dir": str(self.training_dir),
                "deployment_manifest": str(state_root / "queue" / "deployment.json"),
                "activation_manifest": str(state_root / "queue-activation.json"),
                "queue_unit": {
                    **_artifact(self.queue_unit),
                    "name": "edgeconnect-calibration-queue.service",
                },
                "finalize_unit": _artifact(self.finalize_unit),
                "environment": _artifact(self.environment),
                "state_path": str(state_root / "queue" / "state.json"),
                "comparison_output": str(state_root / "queue" / "comparison.json"),
                "continuity_handoff_output": str(
                    state_root / "queue" / "continuity-handoff.json"
                ),
                "execution_lock_path": str(state_root / "execution.lock"),
                "source_commit": "a" * 40,
                "orchestrator": "startrain-orchestrate",
                "poll_seconds": 1,
                "launch_timeout_seconds": 30,
                "max_transient_retries": 2,
                "retry_delay_seconds": 0,
                "continue_after_fatal": False,
                "provisioned_gpus": 2,
            },
            "fallback": {
                "handoff_path": str(state_root / "fallback-request.json"),
                "continuity_manifest": _artifact(self.continuity),
            },
        }
        _write_json(self.policy_path, self.policy)

    @property
    def state_path(self) -> Path:
        return Path(self.policy["state_path"])

    def make_terminal(self, *, decision: str = "reject", updated_ns: int = 200) -> None:
        champion_identity = self.champion_identity
        champion_step = 10
        if decision == "promote":
            champion_identity = self.candidate_identity
            champion_step = 20
            _model_pointer(
                self.champion_pointer,
                role="champion",
                manifest=self.candidate_manifest,
                identity=self.candidate_identity,
                step=20,
                updated_ns=updated_ns - 1,
                promotion_result=self.result_path,
            )
        _write_json(
            self.result_path,
            {
                "schema_version": 3,
                "result_kind": "promotion",
                "candidate": self.candidate_identity,
                "baseline": self.champion_identity,
                "candidate_manifest": str(self.candidate_manifest),
                "champion_manifest": str(self.baseline_manifest),
                "promotion": {"decision": decision},
                "terminal": True,
            },
        )
        _write_json(
            self.status_path,
            {
                "schema_version": 1,
                "candidate_identity": self.candidate_identity,
                "candidate_step": 20,
                "champion_identity": champion_identity,
                "champion_step": champion_step,
                "decision": decision,
                "terminal": True,
                "updated_ns": updated_ns,
            },
        )

    def race_status(self) -> None:
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        status["updated_ns"] += 1
        _write_json(self.status_path, status)


class FakeAdapters:
    def __init__(
        self,
        fixture: BoundaryFixture,
        *,
        fail_at: str | None = None,
        mutate_after_hold: bool = False,
        crash_snapshot_once: bool = False,
        arena_active: bool = False,
        frozen_fallback: bool = False,
        hardware_unavailable: bool = False,
    ) -> None:
        self.fixture = fixture
        self.fail_at = fail_at
        self.mutate_after_hold = mutate_after_hold
        self.crash_snapshot_once = crash_snapshot_once
        self.arena_active = arena_active
        self.frozen_fallback = frozen_fallback
        self.hardware_unavailable = hardware_unavailable
        self.snapshot_crashed = False
        self.snapshot_side_effects = 0
        self.events: list[str] = []

    def _event(self, name: str) -> None:
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"injected {name} failure")

    def inspect_source(self, _policy) -> dict[str, object]:
        self.events.append("inspect_source")
        return {
            "current": True,
            "service_active": True,
            "main_pid": 123,
            "coordinator_lock_matches_service": True,
            "coordinator_status_matches_service": True,
            "run_identity_current": True,
            "process_ids": [123, 124],
        }

    def inspect_hardware(self, _policy) -> dict[str, object]:
        self.events.append("inspect_hardware")
        if self.hardware_unavailable:
            return {
                "healthy": False,
                "current": False,
                "status": "unavailable",
            }
        return {"healthy": True, "current": True}

    def inspect_arena_pause(self, _policy) -> dict[str, object]:
        self.events.append("inspect_pause")
        return {"active": self.arena_active}

    def place_operator_hold(self, path: Path, document) -> dict[str, object]:
        self._event("operator_hold")
        if not path.exists():
            _write_json(path, dict(document))
        if self.mutate_after_hold:
            self.fixture.race_status()
            self.mutate_after_hold = False
        return {
            "status": "active",
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    def final_replay_backup(self, _policy) -> dict[str, object]:
        self._event("final_replay_backup")
        path = self.fixture.state_path.parent / "replay-backup.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"replay-backup")
        return {
            "status": "ok",
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    def disaster_snapshot(self, policy) -> dict[str, object]:
        self.events.append("disaster_snapshot")
        backup = policy["backup"]
        snapshot = (
            Path(backup["disaster_backup_root"])
            / "snapshots"
            / "source-run"
            / "snapshot.json"
        )
        if not snapshot.exists():
            _write_json(snapshot, {"snapshot": "fixture"})
            self.snapshot_side_effects += 1
        evidence = {
            "status": "ok",
            "snapshot": str(snapshot),
            "path": str(snapshot),
            "snapshot_sha256": _sha256(snapshot),
            "snapshot_bytes": snapshot.stat().st_size,
            "created_ns": 300,
        }
        if self.fail_at == "disaster_snapshot":
            raise RuntimeError("injected disaster_snapshot failure")
        if self.crash_snapshot_once and not self.snapshot_crashed:
            self.snapshot_crashed = True
            raise RuntimeError("synthetic crash after snapshot publication")
        return evidence

    def off_host_acknowledgement(self, _policy, snapshot) -> dict[str, object]:
        self._event("off_host_acknowledgement")
        path = Path(self.fixture.policy["backup"]["off_host_acknowledgement_path"])
        _write_json(path, {"snapshot_sha256": snapshot["snapshot_sha256"]})
        return {
            "status": "verified",
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "snapshot_sha256": snapshot["snapshot_sha256"],
        }

    def stop_source(self, _policy) -> dict[str, object]:
        self._event("stop_source")
        return {"status": "stopped", "previous_main_pid": 123}

    def prove_source_release(
        self,
        _policy,
        _source_evidence,
        _stop_evidence,
    ) -> dict[str, object]:
        self._event("prove_source_release")
        return {
            "service_inactive": True,
            "main_pid_released": True,
            "coordinator_lock_released": True,
            "process_groups_released": True,
            "gpus_released": True,
        }

    def export_champion(self, policy, winner_snapshot) -> dict[str, object]:
        self._event("export_champion")
        snapshot = policy["snapshot"]
        destination = Path(snapshot["destination"])
        _write_json(
            destination / "champion.json",
            {"identity": winner_snapshot["champion"]["model_identity"]},
        )
        champion_file = destination / "champion.json"
        pin = Path(snapshot["pin_path"])
        _write_json(
            pin,
            {
                "status": "verified",
                "destination": str(destination),
                "model_identity": winner_snapshot["champion"]["model_identity"],
                "files": [
                    {
                        "path": "champion.json",
                        "sha256": _sha256(champion_file),
                        "bytes": champion_file.stat().st_size,
                    }
                ],
            },
        )
        return {
            "status": "verified",
            "destination": str(destination),
            "pin": {
                "path": str(pin),
                "sha256": _sha256(pin),
                "bytes": pin.stat().st_size,
            },
        }

    def prepare_calibration(self, policy, winner_snapshot) -> dict[str, object]:
        self._event("prepare_calibration")
        calibration = policy["calibration"]
        output = Path(calibration["output_dir"])
        profile = output / "control.yaml"
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_text("profile: fixture\n", encoding="utf-8")
        root = Path(calibration["run_root_parent"]) / "terminal-control-seed47"
        plan = {
            "schema_version": 1,
            "report": "startrain-elo-ablation-plan",
            "initialization": "fork",
            "source_run_root": policy["source"]["run_root"],
            "source_winner_snapshot": dict(winner_snapshot),
            "treatments": [
                {
                    "treatment": "control",
                    "profile": str(profile),
                    "profile_sha256": _sha256(profile),
                    "run_root": str(root),
                }
            ],
        }
        plan_path = output / "ablation-plan.json"
        _write_json(plan_path, plan)
        return {"status": "prepared", "path": str(plan_path)}

    def run_frozen_calibration(
        self,
        policy,
        plan,
        _winner_snapshot,
    ) -> dict[str, object]:
        self._event("run_frozen_calibration")
        comparison_path = (
            Path(policy["calibration"]["frozen_replay"]["output_root"])
            / "comparison.json"
        )
        selection = {
            "selected_arm": "control",
            "fallback_to_control": self.frozen_fallback,
        }
        _write_json(
            comparison_path,
            {"status": "ok", "selection": selection},
        )
        if self.frozen_fallback:
            return {
                "status": "completed",
                "comparison": {
                    "path": str(comparison_path),
                    "sha256": _sha256(comparison_path),
                    "selection": selection,
                },
                "selection": selection,
                "screen_plan": None,
            }
        screen_path = Path(policy["calibration"]["frozen_replay"]["screen_plan_path"])
        _write_json(screen_path, plan["plan"])
        return {
            "status": "completed",
            "comparison": {
                "path": str(comparison_path),
                "sha256": _sha256(comparison_path),
                "selection": selection,
            },
            "selection": selection,
            "screen_plan": {
                "path": str(screen_path),
                "sha256": _sha256(screen_path),
            },
        }

    def fork_calibration(
        self,
        _policy,
        _plan,
        treatment,
        winner_snapshot,
    ) -> dict[str, object]:
        label = treatment["treatment"]
        self._event(f"fork:{label}")
        root = Path(treatment["run_root"])
        champion = winner_snapshot["champion"]
        metadata = {
            "schema_version": 1,
            "report": "startrain-elo-ablation-branch",
            "treatment": label,
            "source_winner_snapshot": dict(winner_snapshot),
            "anchor": {
                "model_identity": champion["model_identity"],
                "model_step": champion["model_step"],
            },
        }
        _write_json(root / "ablation.json", metadata)
        return {"status": "forked", "run_root": str(root)}

    def warm_start(
        self,
        run_root: Path,
        _profile: Path,
        *,
        prepare_only: bool,
    ) -> dict[str, object]:
        phase = "warm_prepare" if prepare_only else "warm_activate"
        self._event(f"{phase}:control")
        marker = run_root / "learner" / "champion-warm-start.json"
        checkpoint = run_root / "learner" / "recovery" / "prepared.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists():
            checkpoint.write_bytes(b"prepared-checkpoint")
        champion = json.loads(
            (self.fixture.champion_pointer).read_text(encoding="utf-8")
        )
        _write_json(
            marker,
            {
                "format": "startrain.champion-warm-start",
                "schema_version": 1,
                "status": "prepared" if prepare_only else "active",
                "source_model_identity": champion["model_identity"],
                "source_model_step": champion["model_step"],
                "absolute_model_step": champion["model_step"],
                "checkpoint": "recovery/prepared.pt",
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
            },
        )
        if not prepare_only:
            _write_json(
                run_root / "learner" / "recovery.json",
                {"step": champion["model_step"]},
            )
            _write_json(
                run_root / "learner" / "resume-cutover.json",
                {"step": champion["model_step"]},
            )
        return {
            "status": "ok",
            "mode": "prepare_only" if prepare_only else "apply",
            "marker": str(marker),
        }

    def prepare_runtime_ownership(self, policy, _plan) -> dict[str, object]:
        self._event("prepare_runtime_ownership")
        calibration = policy["calibration"]
        return {
            "status": "verified",
            "user": calibration["runtime_user"],
            "group": calibration["runtime_group"],
            "uid": 1000,
            "gid": 1000,
            "paths_updated": 1,
        }

    def generate_queue_manifest(self, policy, plan) -> dict[str, object]:
        self._event("generate_queue_manifest")
        path = Path(policy["queue"]["deployment_manifest"])
        _write_json(
            path,
            {
                "schema_version": 1,
                "report": "startrain-elo-ablation-deployment",
                "plan": plan["path"],
            },
        )
        return {"status": "generated", "path": str(path)}

    def verify_queue_manifest(self, _policy, manifest) -> dict[str, object]:
        self._event("verify_queue_manifest")
        assert _sha256(Path(manifest["path"])) == manifest["sha256"]
        return {"status": "verified", "sha256": manifest["sha256"]}

    def launch_queue(self, _policy, activation_manifest: Path) -> dict[str, object]:
        self._event("launch_queue")
        assert activation_manifest.is_file()
        return {"status": "launched", "unit": "fixture-queue.service"}

    def request_continuity_fallback(self, policy, request) -> dict[str, object]:
        self.events.append("request_continuity_fallback")
        path = Path(policy["fallback"]["handoff_path"])
        _write_json(path, request)
        hold = Path(policy["operator_hold_path"])
        if hold.exists():
            hold.unlink()
        return {"status": "requested", "path": str(path)}


def _side_effects(adapters: FakeAdapters) -> list[str]:
    return [event for event in adapters.events if not event.startswith("inspect_")]


def test_stale_terminal_status_waits_without_side_effects(tmp_path: Path) -> None:
    fixture = BoundaryFixture(tmp_path)
    adapters = FakeAdapters(fixture)

    state = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=adapters,
    )

    assert state["status"] == "waiting"
    assert state["waiting_reason"] == "terminal_decision_not_strictly_newer"
    assert _side_effects(adapters) == []
    assert json.loads(fixture.state_path.read_text())["report"] == STATE_REPORT


def test_runtime_drift_is_rejected_before_terminal_side_effects(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    runtime = fixture.policy["runtime"]
    Path(runtime["terminal_runner"]["path"]).write_text(
        "# changed runtime\n",
        encoding="utf-8",
    )
    adapters = FakeAdapters(fixture)

    with pytest.raises(TerminalBoundaryManifestError, match="digest changed"):
        run_terminal_boundary_pipeline(
            fixture.policy_path,
            adapters=adapters,
        )
    assert _side_effects(adapters) == []


def test_runtime_effective_optimizer_is_derived_from_recovery_state(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    checkpoint = fixture.root / "learner" / "recovery" / "runtime.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": {
                "param_groups": [
                    {"algorithm": "muon", "initial_lr": 0.0003125},
                    {"algorithm": "adamw", "initial_lr": 0.0000046875},
                    {"algorithm": "adamw", "initial_lr": 0.0000046875},
                ]
            },
            "scheduler": {
                "base_lrs": [0.0003125, 0.0000046875, 0.0000046875],
                "last_epoch": 123,
            },
        },
        checkpoint,
    )
    _write_json(
        fixture.root / "learner" / "recovery.json",
        {
            "checkpoint": "recovery/runtime.pt",
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
        },
    )

    evidence = DefaultTerminalBoundaryAdapters()._runtime_effective_optimizer(
        fixture.policy
    )

    assert evidence["muon_lr"] == pytest.approx(0.0003125)
    assert evidence["adamw_lr"] == pytest.approx(0.0000046875)
    assert evidence["scheduler_last_epoch"] == 123


def test_cli_does_not_restart_after_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    monkeypatch.setattr(
        "scripts.run_terminal_boundary_pipeline.run_terminal_boundary_pipeline",
        lambda _path: (_ for _ in ()).throw(
            TerminalBoundaryExecutionError("failed safely")
        ),
    )

    assert main(["run", "--manifest", str(fixture.policy_path)]) == 3


def test_cli_treats_normal_waiting_state_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    monkeypatch.setattr(
        "scripts.run_terminal_boundary_pipeline.run_terminal_boundary_pipeline",
        lambda _path: {"status": "waiting"},
    )

    assert main(["run", "--manifest", str(fixture.policy_path)]) == 0


def test_current_terminal_is_accepted_and_launched(tmp_path: Path) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal(decision="reject")
    adapters = FakeAdapters(fixture)

    state = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=adapters,
    )

    accepted = state["accepted_terminal"]
    assert state["status"] == "completed"
    assert accepted["updated_ns"] == 200
    assert accepted["decision"] == "reject"
    assert accepted["status"]["sha256"] == _sha256(fixture.status_path)
    assert state["automatic_launch_authorized"] is True
    assert (
        verify_queue_activation_manifest(
            fixture.policy["queue"]["activation_manifest"],
            policy_path=fixture.policy_path,
        )["status"]
        == "verified"
    )


def test_terminal_result_remains_valid_when_candidate_pointer_has_advanced(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal(decision="reject")
    newer_identity = "sha256-" + "d" * 64
    newer_manifest = _model_manifest(
        fixture.root,
        name="newer-candidate",
        identity=newer_identity,
        step=30,
    )
    _model_pointer(
        fixture.candidate_pointer,
        role="candidate",
        manifest=newer_manifest,
        identity=newer_identity,
        step=30,
        updated_ns=201,
    )

    state = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=FakeAdapters(fixture),
    )

    assert state["status"] == "completed"
    assert state["accepted_terminal"]["candidate"]["model_identity"] == (
        fixture.candidate_identity
    )


def test_newer_terminal_followed_by_active_arena_waits_for_next_boundary(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    adapters = FakeAdapters(fixture, arena_active=True)

    state = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=adapters,
    )

    assert state["status"] == "waiting"
    assert state["phase"] == "awaiting_quiescent_terminal_boundary"
    assert state["waiting_reason"] == (
        "newer_terminal_already_followed_by_active_arena"
    )
    assert state["accepted_terminal"] is None
    assert _side_effects(adapters) == []


def test_transient_missing_hardware_report_waits_without_fallback(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    adapters = FakeAdapters(fixture, hardware_unavailable=True)

    state = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=adapters,
    )

    assert state["status"] == "waiting"
    assert state["phase"] == "awaiting_current_hardware_report"
    assert state["waiting_reason"] == "hardware_health_report_refreshing"
    assert state["fallback"] is None
    assert _side_effects(adapters) == []


def test_terminal_evidence_race_fails_before_backup_or_stop(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    adapters = FakeAdapters(fixture, mutate_after_hold=True)

    with pytest.raises(
        TerminalBoundaryExecutionError,
        match="accepted terminal evidence changed",
    ):
        run_terminal_boundary_pipeline(
            fixture.policy_path,
            adapters=adapters,
        )

    assert _side_effects(adapters) == [
        "operator_hold",
        "request_continuity_fallback",
    ]
    state = json.loads(fixture.state_path.read_text())
    assert state["status"] == "failed"
    assert state["fallback"]["status"] == "requested"


@pytest.mark.parametrize(
    "failure",
    ["final_replay_backup", "off_host_acknowledgement"],
)
def test_backup_or_ack_failure_requests_fallback_after_source_release(
    tmp_path: Path,
    failure: str,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    adapters = FakeAdapters(fixture, fail_at=failure)

    with pytest.raises(TerminalBoundaryExecutionError, match=f"injected {failure}"):
        run_terminal_boundary_pipeline(
            fixture.policy_path,
            adapters=adapters,
        )

    assert "stop_source" in adapters.events
    assert "prove_source_release" in adapters.events
    assert "launch_queue" not in adapters.events
    assert adapters.events[-1] == "request_continuity_fallback"
    state = json.loads(fixture.state_path.read_text())
    assert state["automatic_launch_authorized"] is False
    assert state["fallback"]["status"] == "requested"


def test_stop_failure_never_prepares_or_launches(tmp_path: Path) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    adapters = FakeAdapters(fixture, fail_at="stop_source")

    with pytest.raises(TerminalBoundaryExecutionError, match="injected stop_source"):
        run_terminal_boundary_pipeline(
            fixture.policy_path,
            adapters=adapters,
        )

    assert "off_host_acknowledgement" not in adapters.events
    assert "prove_source_release" not in adapters.events
    assert "prepare_calibration" not in adapters.events
    assert "launch_queue" not in adapters.events
    assert adapters.events[-1] == "request_continuity_fallback"


def test_failed_fallback_request_never_resumes_cutover(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    adapters = FakeAdapters(fixture, fail_at="stop_source")

    def fail_fallback(_policy, _request):
        raise RuntimeError("fallback unavailable")

    adapters.request_continuity_fallback = fail_fallback  # type: ignore[method-assign]
    with pytest.raises(TerminalBoundaryExecutionError, match="injected stop_source"):
        run_terminal_boundary_pipeline(
            fixture.policy_path,
            adapters=adapters,
        )
    failed = json.loads(fixture.state_path.read_text(encoding="utf-8"))
    assert failed["fallback"]["status"] == "failed"

    repeated = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=FakeAdapters(fixture),
    )
    assert repeated["status"] == "blocked"
    assert repeated["phase"] == "continuity_fallback_requested"
    assert "prepare_calibration" not in repeated["steps"]


def test_default_fallback_releases_owned_operator_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    policy_sha256 = _sha256(fixture.policy_path)
    hold_path = Path(fixture.policy["operator_hold_path"])
    _write_json(hold_path, {"policy_sha256": policy_sha256})
    reconciled: list[str] = []
    monkeypatch.setattr(
        "startrain.continuity.reconcile_training_continuity",
        lambda path: reconciled.append(str(path)) or {"status": "active_fallback"},
    )
    request = {
        "failure_id": "failure-1",
        "source": {"policy_sha256": policy_sha256},
    }

    result = DefaultTerminalBoundaryAdapters().request_continuity_fallback(
        fixture.policy,
        request,
    )

    assert not hold_path.exists()
    assert result["operator_hold"]["status"] == "released"
    assert reconciled == [fixture.policy["fallback"]["continuity_manifest"]["path"]]


def test_failed_snapshot_requests_fallback_and_does_not_restart_cutover(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    adapters = FakeAdapters(fixture, crash_snapshot_once=True)

    with pytest.raises(TerminalBoundaryExecutionError, match="synthetic crash"):
        run_terminal_boundary_pipeline(
            fixture.policy_path,
            adapters=adapters,
        )
    first_state = json.loads(fixture.state_path.read_text())
    assert first_state["steps"]["disaster_snapshot"]["status"] == "failed"
    assert first_state["fallback"]["status"] == "requested"

    repeated = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=adapters,
    )

    assert repeated["status"] == "blocked"
    assert repeated["phase"] == "continuity_fallback_requested"
    assert adapters.snapshot_side_effects == 1
    assert adapters.events.count("operator_hold") == 1
    assert adapters.events.count("final_replay_backup") == 1
    assert adapters.events.count("disaster_snapshot") == 1
    assert adapters.events.count("request_continuity_fallback") == 1
    assert adapters.events.count("launch_queue") == 0


def test_recover_requests_fallback_after_uncaught_service_interruption(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    adapters = FakeAdapters(fixture)

    def interrupt(_policy):
        raise KeyboardInterrupt

    adapters.disaster_snapshot = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        run_terminal_boundary_pipeline(
            fixture.policy_path,
            adapters=adapters,
        )
    assert Path(fixture.policy["operator_hold_path"]).exists()

    recovered = recover_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=adapters,
    )

    assert recovered["status"] == "failed"
    assert recovered["fallback"]["status"] == "requested"
    assert not Path(fixture.policy["operator_hold_path"]).exists()


def test_frozen_calibration_no_winner_resumes_runtime_control(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    adapters = FakeAdapters(fixture, frozen_fallback=True)

    state = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=adapters,
    )

    assert state["status"] == "completed"
    assert state["phase"] == "frozen_calibration_retained_control"
    assert state["automatic_launch_authorized"] is False
    assert state["fallback"]["status"] == "requested"
    assert "fork:control" not in adapters.events
    assert "launch_queue" not in adapters.events
    assert not Path(fixture.policy["operator_hold_path"]).exists()

    repeated = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=adapters,
    )
    assert repeated["phase"] == "frozen_calibration_retained_control"
    assert adapters.events.count("request_continuity_fallback") == 1


def test_success_side_effect_order_is_fail_closed(tmp_path: Path) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal(decision="promote")
    adapters = FakeAdapters(fixture)

    state = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=adapters,
    )

    assert state["status"] == "completed"
    assert _side_effects(adapters) == [
        "operator_hold",
        "stop_source",
        "prove_source_release",
        "final_replay_backup",
        "disaster_snapshot",
        "off_host_acknowledgement",
        "export_champion",
        "prepare_calibration",
        "run_frozen_calibration",
        "fork:control",
        "warm_prepare:control",
        "warm_activate:control",
        "generate_queue_manifest",
        "prepare_runtime_ownership",
        "verify_queue_manifest",
        "launch_queue",
    ]
    assert not Path(fixture.policy["operator_hold_path"]).exists()
    assert state["steps"]["release_operator_hold"]["status"] == "completed"


def test_default_queue_launch_requires_pinned_unit_and_durable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    state = run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=FakeAdapters(fixture),
    )
    assert state["status"] == "completed"
    queue = fixture.policy["queue"]
    queue_state = Path(queue["state_path"])

    class Manager:
        def __init__(self, **_kwargs) -> None:
            self.active = False

        def status(self, _unit):
            return SimpleNamespace(
                query_error=None,
                active=self.active,
                active_state="active" if self.active else "inactive",
                main_pid=123 if self.active else 0,
                fragment_path=queue["queue_unit"]["path"],
                as_dict=lambda: {"active": self.active},
            )

        def start(self, _unit) -> None:
            self.active = True
            _write_json(
                queue_state,
                {
                    "manifest": queue["deployment_manifest"],
                    "manifest_sha256": _sha256(Path(queue["deployment_manifest"])),
                    "queue_status": "running",
                },
            )

    monkeypatch.setattr(
        "startrain.continuity.SystemdUnitManager",
        Manager,
    )
    result = DefaultTerminalBoundaryAdapters().launch_queue(
        fixture.policy,
        Path(queue["activation_manifest"]),
    )

    assert result["status"] == "launched"
    assert result["queue_status"] == "running"


def test_activation_rejects_tampered_champion_snapshot_file(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=FakeAdapters(fixture),
    )
    champion_file = Path(fixture.policy["snapshot"]["destination"]) / "champion.json"
    champion_file.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(TerminalBoundaryManifestError, match="champion snapshot file"):
        verify_queue_activation_manifest(
            fixture.policy["queue"]["activation_manifest"],
            policy_path=fixture.policy_path,
        )


def test_activation_rejects_tampered_warm_start_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=FakeAdapters(fixture),
    )
    activation = json.loads(
        Path(fixture.policy["queue"]["activation_manifest"]).read_text(encoding="utf-8")
    )
    root = activation["calibration"]["roots"][0]
    marker = json.loads(
        Path(root["warm_start_active"]["marker"]["path"]).read_text(encoding="utf-8")
    )
    checkpoint = Path(root["run_root"]) / "learner" / marker["checkpoint"]
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(TerminalBoundaryManifestError, match="warm-start checkpoint"):
        verify_queue_activation_manifest(
            fixture.policy["queue"]["activation_manifest"],
            policy_path=fixture.policy_path,
        )


def test_default_runtime_ownership_handoff_covers_profiles_and_queue_manifest(
    tmp_path: Path,
) -> None:
    fixture = BoundaryFixture(tmp_path)
    fixture.make_terminal()
    run_terminal_boundary_pipeline(
        fixture.policy_path,
        adapters=FakeAdapters(fixture),
    )
    current_user = pwd.getpwuid(os.getuid()).pw_name
    current_group = grp.getgrgid(os.getgid()).gr_name
    fixture.policy["calibration"]["runtime_user"] = current_user
    fixture.policy["calibration"]["runtime_group"] = current_group
    activation = json.loads(
        Path(fixture.policy["queue"]["activation_manifest"]).read_text(encoding="utf-8")
    )
    plan_path = Path(activation["calibration"]["plan"]["path"])
    plan_document = json.loads(plan_path.read_text(encoding="utf-8"))

    evidence = DefaultTerminalBoundaryAdapters().prepare_runtime_ownership(
        fixture.policy,
        {"path": str(plan_path), "plan": plan_document},
    )

    deployment = Path(fixture.policy["queue"]["deployment_manifest"])
    profile = Path(plan_document["treatments"][0]["profile"])
    assert evidence["status"] == "verified"
    assert deployment.stat().st_uid == os.getuid()
    assert profile.stat().st_uid == os.getuid()
    assert stat.S_IMODE(deployment.stat().st_mode) == 0o440
    assert stat.S_IMODE(profile.stat().st_mode) == 0o440
