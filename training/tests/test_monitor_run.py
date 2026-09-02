from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import monitor_run as monitor
from startrain import continuity

CONFIGS = Path(__file__).parents[1] / "configs"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _continuity_fixture(tmp_path: Path) -> dict[str, Path]:
    state_root = (tmp_path / "continuity-state").resolve()
    primary_root = (tmp_path / "primary-run").resolve()
    fallback_root = (tmp_path / "fallback-run").resolve()
    disaster_mount = (tmp_path / "disaster").resolve()
    disaster_root = (disaster_mount / "primary").resolve()
    fallback_disaster_root = (disaster_mount / "fallback").resolve()
    training_dir = (tmp_path / "release" / "training").resolve()
    for path in (
        state_root,
        primary_root,
        fallback_root,
        disaster_root,
        fallback_disaster_root,
        training_dir,
    ):
        path.mkdir(parents=True)
    primary_profile = (primary_root / "profile.yaml").resolve()
    fallback_profile = (fallback_root / "profile.yaml").resolve()
    primary_profile.write_text("{}\n", encoding="utf-8")
    fallback_profile.write_text("{}\n", encoding="utf-8")
    digest = "a" * 64
    runtime_manifest = (tmp_path / "release" / "release.json").resolve()
    orchestrator = (training_dir / ".venv" / "bin" / "orchestrator").resolve()
    unit_root = (tmp_path / "units").resolve()
    unit_root.mkdir()

    def workload(
        workload_id: str,
        role: str,
        root: Path,
        profile: Path,
        unit: str,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": workload_id,
            "role": role,
            "unit": unit,
            "profile": {"path": str(profile), "sha256": digest},
            "run_root": {"path": str(root), "sha256": digest},
            "runtime": {
                "manifest": str(runtime_manifest),
                "sha256": digest,
                "training_dir": str(training_dir),
                "orchestrator": str(orchestrator),
                "orchestrator_sha256": digest,
                "unit_path": str(unit_root / unit),
                "unit_sha256": digest,
            },
        }
        if role == "fallback":
            payload["last_known_good"] = {"verified_ns": 1}
        return payload

    primary = workload(
        "primary",
        "primary",
        primary_root,
        primary_profile,
        "edgeconnect-primary.service",
    )

    def protection(owner: str, root: Path, backup_root: Path) -> dict[str, object]:
        return {
            "replay_backup_timer": (f"edgeconnect-startrain-{owner}-backup.timer"),
            "disaster_backup_timer": (
                f"edgeconnect-startrain-{owner}-disaster-backup.timer"
            ),
            "disaster_backup_root": str(backup_root),
            "disaster_backup_mount": str(disaster_mount),
            "telemetry_service": (f"edgeconnect-startrain-{owner}-monitor.service"),
            "telemetry_output": str(root / "status" / "monitor-5s.jsonl"),
        }

    primary["protection"] = protection("primary", primary_root, disaster_root)
    fallback = workload(
        "fallback-lkg",
        "fallback",
        fallback_root,
        fallback_profile,
        "edgeconnect-fallback.service",
    )
    fallback["protection"] = protection(
        "fallback-lkg",
        fallback_root,
        fallback_disaster_root,
    )
    manifest_path = (tmp_path / "continuity.json").resolve()
    _write_json(
        manifest_path,
        {
            "format": continuity.MANIFEST_FORMAT,
            "schema_version": 1,
            "state_root": str(state_root),
            "locks": {
                "transition": str(state_root / "transition.lock"),
                "execution": str(tmp_path / "execution.lock"),
            },
            "hardware": {
                "report_path": str(state_root / "hardware.json"),
                "max_age_seconds": 180,
                "probe_workload": "fallback-lkg",
            },
            "primary": "primary",
            "workloads": [primary, fallback],
        },
    )
    manifest = continuity.load_continuity_manifest(manifest_path)
    _write_json(
        manifest.state_path,
        {
            "format": continuity.STATE_FORMAT,
            "schema_version": 1,
            "manifest_sha256": manifest.sha256,
            "active_workload_id": "primary",
            "active_profile_sha256": digest,
            "active_run_root_sha256": digest,
        },
    )
    return {
        "manifest": manifest_path,
        "run_root": primary_root,
        "profile": primary_profile,
        "state": manifest.state_path,
        "disaster_root": disaster_root,
        "fallback_run_root": fallback_root,
        "fallback_profile": fallback_profile,
        "fallback_disaster_root": fallback_disaster_root,
    }


def _fixture(tmp_path: Path, *, now_ns: int) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    _write_json(
        root / "run.json",
        {
            "schema_version": 1,
            "run_id": "monitor-run",
            "generation_family": "monitor-family",
            "created_ns": now_ns - 1_000_000_000,
        },
    )
    (root / "profile.yaml").write_text(
        yaml.safe_dump(
            {
                "learner": {"steps": 100},
                "orchestration": {
                    "shutdown": {
                        "stale_heartbeat_seconds": 100,
                        "stall_timeout_seconds": 200,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    learner_heartbeat = root / "status" / "learner.heartbeat.json"
    actor_heartbeat = root / "status" / "actor-gpu-1.heartbeat.json"
    _write_json(
        learner_heartbeat,
        {
            "heartbeat_ns": now_ns - 1_000_000_000,
            "progress_ns": now_ns - 2_000_000_000,
            "phase": "training",
            "progress": 10,
            "step": 10,
            "epoch": 1,
        },
    )
    _write_json(
        actor_heartbeat,
        {
            "heartbeat_ns": now_ns - 1_000_000_000,
            "progress_ns": now_ns - 2_000_000_000,
            "phase": "selfplay",
            "progress": 4,
        },
    )
    _write_json(
        root / "status" / "coordinator.json",
        {
            "state": "running",
            "pause_lease": None,
            "workers": {
                "learner": {
                    "role": "learner",
                    "state": "running",
                    "pid": 11,
                    "restart_count": 0,
                    "heartbeat": str(learner_heartbeat),
                },
                "actor-gpu-1": {
                    "role": "actor",
                    "state": "running",
                    "pid": 12,
                    "restart_count": 0,
                    "heartbeat": str(actor_heartbeat),
                },
            },
        },
    )
    (root / "learner").mkdir()
    (root / "learner" / "metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 10,
                "epoch": 1,
                "examples_per_second": 1234.0,
                "step_seconds": 0.1,
                "losses": {"total": 1.5},
                "gradient_norm": 0.5,
                "feature_path": "rust",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "metrics").mkdir()
    (root / "metrics" / "actor-gpu-1.jsonl").write_text(
        json.dumps(
            {
                "worker": "actor-gpu-1",
                "ring": 4,
                "batch": 3,
                "model_role": "champion",
                "model_step": 0,
                "games_per_second": 2.0,
                "samples_per_second": 80.0,
                "evaluator_rows_per_second": 5000.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "arena" / "promotion-status.json",
        {"decision": "bootstrap", "terminal": True, "champion_step": 0},
    )
    replay = root / "replay"
    replay.mkdir()
    connection = sqlite3.connect(replay / "manifest.sqlite3")
    connection.executescript(
        """
        CREATE TABLE shards (
            id INTEGER PRIMARY KEY,
            state TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            ring INTEGER NOT NULL
        );
        CREATE TABLE games (game_id TEXT PRIMARY KEY);
        INSERT INTO shards(state, sample_count, ring) VALUES ('ready', 1000, 4);
        INSERT INTO games(game_id) VALUES ('game-1');
        """
    )
    connection.commit()
    connection.close()
    return root


def _healthy_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        monitor,
        "_systemd_status",
        lambda _unit: {
            "configured": True,
            "active_state": "active",
            "sub_state": "running",
            "restart_count": 0,
        },
    )
    monkeypatch.setattr(
        monitor,
        "_gpu_status",
        lambda: (
            [
                {
                    "index": 0,
                    "temperature.gpu": 45.0,
                    "ecc.errors.uncorrected.volatile.total": 0.0,
                }
            ],
            None,
        ),
    )
    monkeypatch.setattr(
        monitor,
        "_disk_status",
        lambda _root: {"used_fraction": 0.1, "inode_used_fraction": 0.1},
    )


def _install_ring10_profile(root: Path) -> None:
    (root / "profile.yaml").write_text(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _install_ring10_runtime_evidence(root: Path) -> None:
    learner_heartbeat_path = root / "status" / "learner.heartbeat.json"
    learner_heartbeat = json.loads(learner_heartbeat_path.read_text(encoding="utf-8"))
    learner_heartbeat.update(
        {
            "active_rings": [10],
            "active_ring_weights": [0.0, 0.0, 0.0, 1.0],
        }
    )
    _write_json(learner_heartbeat_path, learner_heartbeat)
    actor_heartbeat_path = root / "status" / "actor-gpu-1.heartbeat.json"
    actor_heartbeat = json.loads(actor_heartbeat_path.read_text(encoding="utf-8"))
    actor_heartbeat.update(
        {
            "ring": 10,
            "active_rings": [10],
            "active_ring_weights": [0.0, 0.0, 0.0, 1.0],
        }
    )
    _write_json(actor_heartbeat_path, actor_heartbeat)
    learner_metrics_path = root / "learner" / "metrics.jsonl"
    learner_metric = json.loads(learner_metrics_path.read_text(encoding="utf-8"))
    learner_metric["ring_batch_weights"] = {
        "4": 0.0,
        "6": 0.0,
        "8": 0.0,
        "10": 1.0,
    }
    learner_metrics_path.write_text(json.dumps(learner_metric) + "\n", encoding="utf-8")
    actor_metrics_path = root / "metrics" / "actor-gpu-1.jsonl"
    actor_metric = json.loads(actor_metrics_path.read_text(encoding="utf-8"))
    actor_metric.update(
        {
            "ring": 10,
            "active_rings": [10],
            "active_ring_weights": [0.0, 0.0, 0.0, 1.0],
        }
    )
    actor_metrics_path.write_text(json.dumps(actor_metric) + "\n", encoding="utf-8")


def test_collect_snapshot_reports_healthy_run(tmp_path, monkeypatch) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    (root / "learner" / "model-history.jsonl").write_text(
        json.dumps(
            {
                "model_identity": "candidate",
                "model_step": 10,
                "published_ns": now_ns - 2_000_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "arena" / "evaluation.json",
        {
            "schema_version": 1,
            "candidate": "candidate",
            "baseline": "baseline",
            "completed_ns": now_ns,
            "promotion": {"decision": "promote"},
            "aggregate": {
                "elo_difference": 42.0,
                "anytime_confidence_sequence": [0.51, 0.7],
                "wins": 60,
                "losses": 40,
                "games": 100,
            },
            "per_ring": {"10": {"elo_difference": 25.0}},
        },
    )
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(
        root, unit="startrain.service", now_ns=now_ns
    )

    assert snapshot["status"] == "OK"
    assert snapshot["warnings"] == []
    assert snapshot["learner"]["step"] == 10
    assert snapshot["learner"]["target_steps"] == 100
    assert snapshot["actors"]["latest_batch_rate_sum"] == {
        "games_per_second": 2.0,
        "samples_per_second": 80.0,
        "evaluator_rows_per_second": 5000.0,
    }
    assert snapshot["replay"]["states"]["ready"]["samples"] == 1000
    assert snapshot["replay"]["games"] == 1
    assert snapshot["arena_history"]["recent"][-1]["elo_difference"] == 42.0
    assert snapshot["arena_history"]["candidate_publications"] == 1
    assert snapshot["arena_history"]["candidate_arrival_service_ratio"] == 1
    assert snapshot["arena_history"]["recent"][-1]["publish_to_terminal_seconds"] == 2
    assert snapshot["arena_history"]["recent"][-1]["per_ring_elo"]["10"] == 25.0
    assert "learner=10/100" in monitor.format_text(snapshot)
    assert "elo=42.00" in monitor.format_text(snapshot)


def test_monitor_warns_about_reduced_and_collapsed_learning_rates(
    tmp_path, monkeypatch
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    _healthy_dependencies(monkeypatch)
    profile = yaml.safe_load((root / "profile.yaml").read_text(encoding="utf-8"))
    profile["optimizer"] = {"kind": "muon_adamw", "muon_lr": 5e-3, "adamw_lr": 7.5e-5}
    (root / "profile.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    metrics_path = root / "learner" / "metrics.jsonl"
    metric = json.loads(metrics_path.read_text(encoding="utf-8"))

    # A governed learner running below its reference is a warning.
    metric.update(
        {
            "learning_rates": [1.5e-4, 2.25e-6, 2.25e-6],
            "learning_rate_multiplier": 0.5,
            "reference_learning_rates": [3e-4, 4.5e-6, 4.5e-6],
            "scheduler_segment": "cosine",
        }
    )
    metrics_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    codes = {warning["code"] for warning in snapshot["warnings"]}
    assert "learning_rate_reduced" in codes
    assert "learning_rate_collapsed" not in codes
    assert snapshot["learner"]["learning_rate_multiplier"] == 0.5
    assert snapshot["learner"]["reference_learning_rates"] == [3e-4, 4.5e-6, 4.5e-6]

    # A legacy learner whose live rate sits far below the profile schedule has
    # compounded plateau reductions; that is an error, not a warning.
    metric.pop("learning_rate_multiplier")
    metric.pop("reference_learning_rates")
    metric["learning_rates"] = [1.35e-5, 2e-7, 2e-7]
    metrics_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    snapshot = monitor.collect_snapshot(root, now_ns=now_ns)
    collapsed = [
        warning
        for warning in snapshot["warnings"]
        if warning["code"] == "learning_rate_collapsed"
    ]
    assert len(collapsed) == 1
    assert collapsed[0]["severity"] == "ERROR"

    # Warmup legitimately runs below the schedule.
    metric["scheduler_segment"] = "warmup"
    metrics_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    snapshot = monitor.collect_snapshot(root, now_ns=now_ns)
    assert not any(
        warning["code"].startswith("learning_rate") for warning in snapshot["warnings"]
    )


def test_monitor_exposes_training_objective(tmp_path, monkeypatch) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    _install_ring10_profile(root)
    _install_ring10_runtime_evidence(root)
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)

    assert snapshot["training_objective"] == "ring10_only"
    assert snapshot["objective_contract"]["validated"] is True
    assert not any(
        warning["code"].startswith("ring10_") for warning in snapshot["warnings"]
    )
    assert "objective=ring10_only" in monitor.format_text(snapshot)


def test_monitor_rejects_malformed_ring10_profile(tmp_path, monkeypatch) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    _install_ring10_profile(root)
    profile = yaml.safe_load((root / "profile.yaml").read_text(encoding="utf-8"))
    profile["selfplay"]["rings"] = 8
    (root / "profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)

    assert snapshot["status"] == "ERROR"
    assert snapshot["objective_contract"]["validated"] is False
    assert "objective_profile_invalid" in {
        warning["code"] for warning in snapshot["warnings"]
    }


def test_monitor_flags_current_non_ring10_runtime_evidence(
    tmp_path, monkeypatch
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    _install_ring10_profile(root)
    _install_ring10_runtime_evidence(root)
    actor_metrics_path = root / "metrics" / "actor-gpu-1.jsonl"
    actor_metric = json.loads(actor_metrics_path.read_text(encoding="utf-8"))
    actor_metric["ring"] = 8
    actor_metrics_path.write_text(json.dumps(actor_metric) + "\n", encoding="utf-8")
    learner_metrics_path = root / "learner" / "metrics.jsonl"
    learner_metric = json.loads(learner_metrics_path.read_text(encoding="utf-8"))
    learner_metric["ring_batch_weights"] = {"8": 1.0}
    learner_metrics_path.write_text(json.dumps(learner_metric) + "\n", encoding="utf-8")
    _write_json(
        root / "arena" / "current-evaluation.json",
        {
            "started_ns": now_ns - 2_000_000_000,
            "completed_ns": now_ns - 1_000_000_000,
            "per_ring": {"8": {"games": 2}},
        },
    )
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    codes = {warning["code"] for warning in snapshot["warnings"]}

    assert snapshot["status"] == "ERROR"
    assert {
        "ring10_actor_evidence_mismatch",
        "ring10_learner_evidence_mismatch",
        "ring10_arena_evidence_mismatch",
    } <= codes


def test_monitor_ignores_pre_objective_and_rotated_arena_evidence(
    tmp_path, monkeypatch
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    _install_ring10_profile(root)
    _install_ring10_runtime_evidence(root)
    _write_json(root / "ablation.json", {"prepared_ns": now_ns - 2_000_000_000})
    _write_json(
        root / "arena" / "pre-objective-diagnostic.json",
        {
            "completed_ns": now_ns - 3_000_000_000,
            "per_ring": {"4": {"games": 2}},
        },
    )
    _write_json(
        root / "ablation-parent" / "arena" / "inherited.json",
        {
            "completed_ns": now_ns,
            "per_ring": {"6": {"games": 2}},
        },
    )
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)

    assert "ring10_arena_evidence_mismatch" not in {
        warning["code"] for warning in snapshot["warnings"]
    }


def test_snapshot_warns_about_fragmented_arena_continuation(
    tmp_path, monkeypatch
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    profile = yaml.safe_load((root / "profile.yaml").read_text(encoding="utf-8"))
    profile["arena"] = {
        "pairs_per_ring": 50,
        "minimum_pairs_per_ring": 50,
        "max_pairs_per_ring": 200,
    }
    (root / "profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(
        root / "arena" / "superseded.json",
        {
            "schema_version": 3,
            "candidate": "candidate",
            "baseline": "baseline",
            "completed_ns": now_ns,
            "promotion": {"decision": "superseded"},
            "aggregate": {
                "elo_difference": 10.0,
                "games": 400,
                "wins": 210,
                "losses": 190,
            },
        },
    )
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)

    warning_codes = {item["code"] for item in snapshot["warnings"]}
    assert "arena_continuation_fragmented" in warning_codes
    assert snapshot["arena_history"]["completed_superseded_evaluations"] == 1
    assert snapshot["arena_history"]["completed_superseded_fraction"] == 1.0

    profile["orchestration"]["promotion"] = {"finish_inflight_candidate": True}
    profile["arena"]["continuation_pairs_per_ring"] = 25
    (root / "profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )
    migrated: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    migrated_codes = {item["code"] for item in migrated["warnings"]}
    assert "arena_continuation_fragmented" not in migrated_codes


def test_monitor_surfaces_weighted_block_progress_and_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    profile = yaml.safe_load((root / "profile.yaml").read_text(encoding="utf-8"))
    profile["arena"] = {
        "promotion_pair_ratios": {4: 1, 6: 1, 8: 1, 10: 7},
        "required_regression_rings": [],
        "weighted_initial_blocks": 15,
        "weighted_continuation_blocks": 10,
        "weighted_max_blocks": 50,
    }
    (root / "profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(
        root / "arena" / "weighted-evaluation.json",
        {
            "schema_version": 4,
            "candidate": "candidate",
            "baseline": "baseline",
            "started_ns": 1_000_000_000,
            "completed_ns": now_ns,
            "terminal": False,
            "promotion": {"decision": "continue"},
            "aggregate": {
                "elo_difference": 10,
                "wins": 10,
                "losses": 10,
                "games": 20,
            },
            "per_ring": {"10": {"elo_difference": 20}},
            "weighted_aggregate": {
                "pair_ratios": {"4": 1, "6": 1, "8": 1, "10": 7},
                "complete_blocks": 25,
                "incomplete_pair_counts": {"4": 0, "6": 0, "8": 0, "10": 3},
                "score_rate": 0.58,
                "elo_difference": 56,
                "anytime_elo_interval": [12, 90],
                "evidence_state": "continue",
            },
            "wave_plan": {"target_complete_blocks": 25},
        },
    )
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    weighted = snapshot["weighted_promotion"]
    text = monitor.format_text(snapshot)

    assert weighted["enabled"] is True
    assert weighted["pair_ratios"] == {"4": 1, "6": 1, "8": 1, "10": 7}
    assert weighted["complete_blocks"] == 25
    assert weighted["remaining_blocks"] == 25
    assert weighted["wave_target_blocks"] == 25
    assert weighted["anytime_lower_elo"] == 12
    assert snapshot["arena_history"]["weighted"] == weighted
    assert "weighted_blocks=25/50" in text
    assert "weighted_lcb=12.00" in text
    assert "weighted_state=continue" in text


def test_collect_snapshot_reports_unlimited_recovery_state(
    tmp_path, monkeypatch
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    profile = yaml.safe_load((root / "profile.yaml").read_text(encoding="utf-8"))
    profile["learner"].update({"unlimited": True, "recovery_interval_steps": 5})
    (root / "profile.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    recovery_checkpoint = root / "learner" / "recovery" / ("sha256-" + "a" * 64 + ".pt")
    recovery_checkpoint.parent.mkdir(parents=True)
    recovery_checkpoint.write_bytes(b"checkpoint")
    recovery_sha = hashlib.sha256(b"checkpoint").hexdigest()
    _write_json(
        root / "learner" / "recovery.json",
        {
            "format": "startrain.recovery-pointer",
            "schema_version": 1,
            "checkpoint": f"recovery/{recovery_checkpoint.name}",
            "checkpoint_sha256": recovery_sha,
            "checkpoint_bytes": len(b"checkpoint"),
            "step": 10,
            "epoch": 1,
        },
    )
    backup = root / "recovery" / "replay-manifest" / "manifest-1.sqlite3"
    backup.parent.mkdir(parents=True)
    backup.write_bytes(b"backup")
    _write_json(
        backup.parent / "latest.json",
        {
            "schema_version": 1,
            "path": backup.name,
            "bytes": len(b"backup"),
            "sha256": hashlib.sha256(b"backup").hexdigest(),
            "created_ns": now_ns,
        },
    )
    with (root / "learner" / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "recovery_checkpoint", "step": 10}) + "\n")
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    assert snapshot["status"] == "OK"
    assert snapshot["learner"]["target_steps"] == "unlimited"
    assert snapshot["recovery"]["step"] == 10
    assert snapshot["recovery"]["replay_backup_valid"] is True
    assert snapshot["learner"]["examples_per_second"] == 1234.0
    assert "learner=10/unlimited" in monitor.format_text(snapshot)


def test_snapshot_surfaces_per_lane_training_policy_drift(
    tmp_path, monkeypatch
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    profile = yaml.safe_load((root / "profile.yaml").read_text(encoding="utf-8"))
    profile["selfplay"] = {"record_fast_policy_targets": True}
    profile["orchestration"]["ring_mixture"] = {
        "step_weights": [{"from_step": 0, "weights": [0.15, 0.15, 0.15, 0.55]}]
    }
    (root / "profile.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    metrics_path = root / "metrics" / "actor-gpu-1.jsonl"
    metric = json.loads(metrics_path.read_text())
    metric.update(
        {
            "samples": 100,
            "policy_samples": 0,
            "active_ring_weights": [0.15, 0.15, 0.15, 0.55],
        }
    )
    metrics_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    codes = {warning["code"] for warning in snapshot["warnings"]}
    assert "actor_ring_weight_mismatch" in codes
    assert "policy_supervision_low" in codes
    assert snapshot["status"] == "ERROR"


def test_snapshot_surfaces_stale_restart_quarantine_and_hardware(
    tmp_path, monkeypatch
) -> None:
    now_ns = 300_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    actor_path = root / "status" / "actor-gpu-1.heartbeat.json"
    actor = json.loads(actor_path.read_text())
    actor["heartbeat_ns"] = now_ns - 150_000_000_000
    _write_json(actor_path, actor)
    coordinator_path = root / "status" / "coordinator.json"
    coordinator = json.loads(coordinator_path.read_text())
    coordinator["workers"]["actor-gpu-1"]["restart_count"] = 1
    _write_json(coordinator_path, coordinator)
    connection = sqlite3.connect(root / "replay" / "manifest.sqlite3")
    connection.execute(
        "INSERT INTO shards(state, sample_count, ring) VALUES ('quarantined', 5, 4)"
    )
    connection.commit()
    connection.close()
    _write_json(root / "status" / "arena-gpu-pause.json", {"token": "one"})
    _write_json(root / "status" / "arena-gpu-pause.ack.json", {"token": "two"})
    monkeypatch.setattr(
        monitor,
        "_systemd_status",
        lambda _unit: {
            "configured": True,
            "active_state": "active",
            "restart_count": 1,
        },
    )
    monkeypatch.setattr(
        monitor,
        "_gpu_status",
        lambda: (
            [
                {
                    "index": 0,
                    "temperature.gpu": 85.0,
                    "ecc.errors.uncorrected.volatile.total": 1.0,
                }
            ],
            None,
        ),
    )
    monkeypatch.setattr(
        monitor,
        "_disk_status",
        lambda _root: {"used_fraction": 0.9, "inode_used_fraction": 0.2},
    )

    snapshot: Any = monitor.collect_snapshot(
        root, unit="startrain.service", now_ns=now_ns
    )
    codes = {warning["code"] for warning in snapshot["warnings"]}

    assert snapshot["status"] == "ERROR"
    assert {
        "service_restarted",
        "worker_restarted",
        "heartbeat_stale",
        "replay_quarantine",
        "pause_token_mismatch",
        "disk_high",
        "gpu_temperature",
        "gpu_ecc",
    } <= codes


def test_snapshot_surfaces_persistent_sram_threshold_with_zero_volatile(
    tmp_path, monkeypatch
) -> None:
    now_ns = 300_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    _healthy_dependencies(monkeypatch)
    _write_json(
        root / "status" / "hardware-health.json",
        {
            "schema_version": 1,
            "healthy": False,
            "gpus": [
                {
                    "index": 0,
                    "volatile_sram_uncorrectable_parity": 0,
                    "aggregate_sram_uncorrectable_parity": 65_535,
                    "sram_threshold_exceeded": True,
                    "reasons": [
                        "aggregate_uncorrectable_ecc",
                        "sram_threshold_exceeded",
                    ],
                }
            ],
        },
    )

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)

    assert snapshot["status"] == "ERROR"
    assert "gpu_health_gate" in {warning["code"] for warning in snapshot["warnings"]}


def test_snapshot_and_text_surface_durable_coordinator_failure(
    tmp_path,
    monkeypatch,
) -> None:
    now_ns = 300_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    reason = "checkpoint schema is incompatible"
    _write_json(
        root / "status" / "fatal.json",
        {
            "format": "startrain.coordinator-fatal",
            "schema_version": 1,
            "timestamp_ns": now_ns,
            "terminal_reason": "fatal_worker_failure",
            "failure_class": "fatal",
            "reason": reason,
            "exception_type": "ValueError",
            "worker": "learner",
            "role": "learner",
            "worker_exit_code": 78,
            "coordinator_exit_code": 78,
            "restart_count": 0,
        },
    )
    coordinator_path = root / "status" / "coordinator.json"
    coordinator = json.loads(coordinator_path.read_text(encoding="utf-8"))
    coordinator["workers"]["learner"].update(
        {
            "state": "fatal",
            "failure_class": "fatal",
            "failure_reason": reason,
            "last_exit_code": 78,
        }
    )
    _write_json(coordinator_path, coordinator)
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    text = monitor.format_text(snapshot)

    assert snapshot["status"] == "ERROR"
    assert snapshot["coordinator"]["failure"]["failure_class"] == "fatal"
    assert snapshot["coordinator"]["failure"]["reason"] == reason
    learner = next(
        worker for worker in snapshot["workers"] if worker["name"] == "learner"
    )
    assert learner["failure_class"] == "fatal"
    assert learner["failure_reason"] == reason
    warnings = {warning["code"]: warning["message"] for warning in snapshot["warnings"]}
    assert reason in warnings["coordinator_fatal"]
    assert "failure=fatal" in text


def test_actor_throughput_uses_completed_counters_and_merged_wall_intervals(
    tmp_path,
) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    for lane in range(2):
        (metrics / f"actor-gpu-1-lane-{lane}.jsonl").write_text(
            json.dumps(
                {
                    "worker": f"actor-gpu-1-lane-{lane}",
                    "gpu_id": 1,
                    "batch_started_ns": 10_000_000_000,
                    "batch_completed_ns": 20_000_000_000,
                    "games": 10,
                    "samples": 100,
                    "evaluator_rows": 1_000,
                    "samples_per_second": 999_999,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    throughput = monitor._actor_throughput_window(
        metrics,
        now_ns=20_000_000_000,
        window_seconds=60,
    )

    assert throughput["fleet"]["wall_seconds"] == 10
    assert throughput["fleet"]["samples"] == 200
    assert throughput["fleet"]["samples_per_second"] == 20
    assert throughput["by_gpu"]["1"]["wall_seconds"] == 10


def test_actor_throughput_handles_window_baseline_and_process_restart(
    tmp_path,
) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "actor-gpu-1-lane-0.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "worker": "actor-gpu-1-lane-0",
                    "gpu_id": 1,
                    "process_started_ns": 1,
                    "batch_started_ns": 50_000_000_000,
                    "batch_completed_ns": 60_000_000_000,
                    "cumulative_games": 10,
                    "cumulative_samples": 100,
                    "cumulative_evaluator_rows": 1_000,
                },
                {
                    "worker": "actor-gpu-1-lane-0",
                    "gpu_id": 1,
                    "process_started_ns": 1,
                    "batch_started_ns": 110_000_000_000,
                    "batch_completed_ns": 120_000_000_000,
                    "cumulative_games": 30,
                    "cumulative_samples": 300,
                    "cumulative_evaluator_rows": 3_000,
                },
                {
                    "worker": "actor-gpu-1-lane-0",
                    "gpu_id": 1,
                    "process_started_ns": 90_000_000_000,
                    "batch_started_ns": 100_000_000_000,
                    "batch_completed_ns": 115_000_000_000,
                    "cumulative_games": 5,
                    "cumulative_samples": 50,
                    "cumulative_evaluator_rows": 500,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    throughput = monitor._actor_throughput_window(
        metrics,
        now_ns=120_000_000_000,
        window_seconds=60,
    )

    assert throughput["fleet"]["wall_seconds"] == 60
    assert throughput["fleet"]["samples"] == 250
    assert throughput["fleet"]["samples_per_second"] == pytest.approx(250 / 60)
    assert throughput["partial_processes"] == []


def test_latest_jsonl_ignores_partial_tail(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"step":1}\n{"step":2')
    assert monitor._latest_jsonl(path) == {"step": 1}


def test_replay_query_is_read_only(tmp_path) -> None:
    root = _fixture(tmp_path, now_ns=10_000_000_000)
    path = root / "replay" / "manifest.sqlite3"
    before = path.stat().st_mtime_ns
    result: Any
    result, error = monitor._replay_status(path)
    after = path.stat().st_mtime_ns
    assert error is None
    assert result["states"]["ready"]["shards"] == 1
    assert before == after


def test_replay_reports_sample_weighted_model_step_lag(tmp_path) -> None:
    root = _fixture(tmp_path, now_ns=10_000_000_000)
    path = root / "replay" / "manifest.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "ALTER TABLE shards ADD COLUMN model_step INTEGER NOT NULL DEFAULT 0"
    )
    connection.execute("UPDATE shards SET model_step = 80")
    connection.execute(
        """
        INSERT INTO shards(state, sample_count, ring, model_step)
        VALUES ('ready', 3000, 4, 95)
        """
    )
    connection.execute(
        """
        INSERT INTO shards(state, sample_count, ring, model_step)
        VALUES ('ready', 500, 4, 105)
        """
    )
    connection.commit()
    connection.close()

    result, error = monitor._replay_status(path, current_model_step=100)

    assert error is None
    assert result["model_step_lag"] == {
        "current_model_step": 100,
        "ready_samples": 4500,
        "ahead_samples": 500,
        "ahead_max_steps": 5,
        "minimum": -5,
        "weighted_mean": pytest.approx(7.222222222222222),
        "weighted_p50": 5,
        "weighted_p90": 20,
        "maximum": 20,
    }


def test_continuity_manifest_resolves_active_monitor_target(tmp_path: Path) -> None:
    paths = _continuity_fixture(tmp_path)

    target = monitor.resolve_monitor_target(
        paths["manifest"],
        run_root=paths["run_root"],
        profile_path=paths["profile"],
        unit="edgeconnect-primary.service",
        continuity_state_path=paths["state"],
        disaster_backup_root=paths["disaster_root"],
    )

    assert target.run_root == paths["run_root"]
    assert target.profile_path == paths["profile"]
    assert target.unit == "edgeconnect-primary.service"
    assert target.continuity_state_path == paths["state"]
    assert target.disaster_backup_root == paths["disaster_root"]


def test_continuity_manifest_follows_fallback_active_workload(
    tmp_path: Path,
) -> None:
    paths = _continuity_fixture(tmp_path)
    manifest = continuity.load_continuity_manifest(paths["manifest"])
    _write_json(
        paths["state"],
        {
            "format": continuity.STATE_FORMAT,
            "schema_version": 1,
            "manifest_sha256": manifest.sha256,
            "active_workload_id": "fallback-lkg",
            "active_profile_sha256": "a" * 64,
            "active_run_root_sha256": "a" * 64,
        },
    )

    target = monitor.resolve_monitor_target(paths["manifest"])

    assert target.run_root == paths["fallback_run_root"]
    assert target.profile_path == paths["fallback_profile"]
    assert target.unit == "edgeconnect-fallback.service"
    assert target.disaster_backup_root == paths["fallback_disaster_root"]


def test_continuity_manifest_rejects_conflicting_explicit_monitor_inputs(
    tmp_path: Path,
) -> None:
    paths = _continuity_fixture(tmp_path)
    conflicts = (
        ({"run_root": tmp_path / "other-run"}, "--run-root"),
        ({"profile_path": tmp_path / "other.yaml"}, "--profile"),
        ({"unit": "edgeconnect-other.service"}, "--unit"),
        (
            {"continuity_state_path": tmp_path / "other-state.json"},
            "--continuity-state",
        ),
        (
            {"disaster_backup_root": tmp_path / "other-disaster"},
            "--disaster-backup-root",
        ),
    )

    for arguments, option in conflicts:
        with pytest.raises(ValueError, match=option):
            monitor.resolve_monitor_target(paths["manifest"], **arguments)


def test_main_uses_manifest_resolved_active_workload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _continuity_fixture(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_monitor(run_root: Path, **arguments: object) -> None:
        captured["run_root"] = run_root
        captured.update(arguments)

    monkeypatch.setattr(monitor, "run_monitor", fake_run_monitor)
    monkeypatch.setattr(monitor.signal, "signal", lambda *_args: None)

    result = monitor.main(
        [
            "--continuity-manifest",
            str(paths["manifest"]),
            "--once",
        ]
    )

    assert result == 0
    assert captured["run_root"] == paths["run_root"]
    assert captured["profile_path"] == paths["profile"]
    assert captured["unit"] == "edgeconnect-primary.service"
    assert captured["continuity_state_path"] == paths["state"]
    assert captured["disaster_backup_root"] == paths["disaster_root"]


def test_run_monitor_once_emits_one_json_record(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        monitor,
        "collect_snapshot",
        lambda _root, unit=None, profile_path=None: {
            "schema_version": 1,
            "timestamp": "2026-07-11T00:00:00Z",
            "status": "OK",
            "warnings": [],
        },
    )
    monitor.run_monitor(
        tmp_path,
        profile_path=None,
        unit="unit",
        interval=60,
        once=True,
        output_format="jsonl",
        stop_requested=lambda: False,
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "OK"


def test_run_monitor_persists_durable_jsonl_telemetry(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        monitor,
        "collect_snapshot",
        lambda _root, unit=None, profile_path=None: {
            "schema_version": 1,
            "timestamp": "2026-07-11T00:00:00Z",
            "status": "OK",
            "gpus": [{"index": 0, "utilization.gpu": 75.0}],
            "warnings": [],
        },
    )
    output = tmp_path / "telemetry" / "monitor.jsonl"

    monitor.run_monitor(
        tmp_path,
        profile_path=None,
        unit="unit",
        interval=5,
        once=True,
        output_format="text",
        stop_requested=lambda: False,
        telemetry_output=output,
    )

    persisted = [json.loads(line) for line in output.read_text().splitlines()]
    assert persisted[0]["gpus"][0]["utilization.gpu"] == 75.0
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_telemetry_append_repairs_partial_tail(tmp_path) -> None:
    output = tmp_path / "monitor.jsonl"
    output.write_bytes(b'{"status":"OLD"}\n{"partial":')

    monitor._append_snapshot_jsonl(output, {"status": "OK"})

    assert [json.loads(line)["status"] for line in output.read_text().splitlines()] == [
        "OLD",
        "OK",
    ]


def test_telemetry_tail_repair_preserves_history_beyond_scan_window(tmp_path) -> None:
    output = tmp_path / "monitor.jsonl"
    output.write_bytes(b'{"status":"OLD"}\n{"partial":"' + b"x" * (1024 * 1024 + 128))

    monitor._append_snapshot_jsonl(output, {"status": "OK"})

    assert [json.loads(line)["status"] for line in output.read_text().splitlines()] == [
        "OLD",
        "OK",
    ]


def test_telemetry_rotation_retains_complete_bounded_archives(tmp_path) -> None:
    output = tmp_path / "monitor-5s.jsonl"
    unrelated = tmp_path / "monitor-5s.manual.jsonl"
    unrelated.write_text('{"preserve":true}\n', encoding="utf-8")

    for index in range(8):
        monitor._append_snapshot_jsonl(
            output,
            {"index": index},
            maximum_bytes=25,
            retain_files=2,
        )

    archives = sorted(tmp_path.glob("monitor-5s.*.jsonl"))
    generated = [path for path in archives if path != unrelated]
    assert len(generated) == 2
    assert unrelated.read_text(encoding="utf-8") == '{"preserve":true}\n'
    for path in [*generated, output]:
        assert path.read_bytes().endswith(b"\n")
        assert all(json.loads(line) for line in path.read_text().splitlines())
        assert path.stat().st_size <= 25
    assert json.loads(output.read_text().splitlines()[-1])["index"] == 7


def test_telemetry_rotation_sequence_survives_wall_clock_rollback(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "monitor-5s.jsonl"
    output.write_text('{"current":true}\n', encoding="utf-8")
    (tmp_path / "monitor-5s.100.jsonl").write_text(
        '{"old":true}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(monitor.time, "time_ns", lambda: 50)

    monitor._rotate_telemetry_jsonl(output, retain_files=2)

    assert (tmp_path / "monitor-5s.100.jsonl").is_file()
    assert (tmp_path / "monitor-5s.101.jsonl").is_file()


def test_telemetry_failure_does_not_stop_stdout_monitor(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        monitor,
        "collect_snapshot",
        lambda _root, unit=None, profile_path=None: {
            "schema_version": 1,
            "timestamp": "2026-07-11T00:00:00Z",
            "status": "OK",
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        monitor,
        "_append_snapshot_jsonl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    monitor.run_monitor(
        tmp_path,
        profile_path=None,
        unit=None,
        interval=5,
        once=True,
        output_format="jsonl",
        stop_requested=lambda: False,
        telemetry_output=tmp_path / "telemetry.jsonl",
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "WARN"
    assert payload["warnings"][-1]["code"] == "telemetry_persistence_failed"


def test_arena_latency_retains_pre_measurement_publication(tmp_path) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    (root / "learner" / "model-history.jsonl").write_text(
        json.dumps(
            {
                "model_identity": "candidate",
                "model_step": 10,
                "published_ns": now_ns - 2_000_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "arena" / "evaluation.json",
        {
            "candidate": "candidate",
            "baseline": "baseline",
            "completed_ns": now_ns,
            "promotion": {"decision": "promote"},
            "aggregate": {
                "elo_difference": 42.0,
                "wins": 60,
                "losses": 40,
                "games": 100,
            },
            "per_ring": {"10": {"elo_difference": 42.0}},
        },
    )

    history = monitor._arena_history(root, started_ns=now_ns - 1_000_000_000)

    assert history["candidate_publications"] == 0
    assert history["recent"][0]["publish_to_terminal_seconds"] == 2


def test_monitor_shows_headline_segment_loader_and_result_kind_counts(
    tmp_path,
    monkeypatch,
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    learner_metric_path = root / "learner" / "metrics.jsonl"
    learner_metric = json.loads(learner_metric_path.read_text(encoding="utf-8"))
    learner_metric.update(
        {
            "updates_per_new_sample": 1.05,
            "lifetime_updates_per_new_sample": 1.05,
            "segment_updates_per_new_sample": 1.2,
            "utd_segment_target_updates_per_new_sample": 1.25,
            "loader_workers_effective": 8,
            "window_reuse": True,
            "window_reuse_spins": 3,
            "window_setup_seconds": 0.02,
            "window_setup_amortized_seconds": 0.004,
        }
    )
    learner_metric_path.write_text(
        json.dumps(learner_metric)
        + "\n"
        + json.dumps(
            {
                "timestamp_ns": now_ns,
                "worker": "learner",
                "event": "replay_loader_pool_rebound",
                "loader_lifecycle": "process",
                "loader_pool_starts": 1,
                "loader_pool_rebinds": 4,
                "loader_pool_shutdowns": 0,
                "loader_worker_pids": [101, 102, 103, 104, 105, 106, 107, 108],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "strength-efficiency.json",
        {
            "report": "startrain-strength-efficiency",
            "schema_version": 1,
            "status": "complete",
            "run_id": "monitor-run",
            "generation_family": "monitor-family",
            "run_root": str(root),
            "started_ns": now_ns - 1_000_000_000,
            "observed_until_ns": now_ns,
            "autonomous_elo": {
                "headline": {
                    "source": "aggregate",
                    "rating": 321.5,
                    "confidence_interval": [300.0, 343.0],
                },
                "headline_elo": 321.5,
            },
        },
    )
    common = {
        "schema_version": 3,
        "candidate": "candidate",
        "baseline": "baseline",
        "aggregate": {
            "elo_difference": 10.0,
            "wins": 6,
            "losses": 4,
            "games": 10,
        },
        "per_ring": {},
    }
    _write_json(
        root / "arena" / "legacy-promotion.json",
        {
            **common,
            "completed_ns": now_ns - 1,
            "promotion": {"decision": "reject"},
        },
    )
    _write_json(
        root / "arena" / "crossplay.json",
        {
            **common,
            "candidate": "candidate-new",
            "completed_ns": now_ns,
            "result_kind": "historical_crossplay",
        },
    )
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    text = monitor.format_text(snapshot)

    assert snapshot["strength_efficiency"]["headline_elo"] == 321.5
    assert snapshot["strength_efficiency"]["headline_source"] == "aggregate"
    assert snapshot["arena_history"]["promotion_evaluations"] == 1
    assert snapshot["arena_history"]["crossplay_evaluations"] == 1
    assert snapshot["arena_history"]["result_kind_counts"]["historical_crossplay"] == 1
    assert snapshot["learner"]["segment_updates_per_new_sample"] == 1.2
    assert snapshot["learner"]["loader_workers_effective"] == 8
    assert snapshot["learner"]["loader_lifecycle"] == "process"
    assert snapshot["learner"]["loader_pool_starts"] == 1
    assert snapshot["learner"]["loader_pool_rebinds"] == 4
    assert snapshot["learner"]["loader_pool_shutdowns"] == 0
    assert "utd_segment=1.20/1.25" in text
    assert "loader_workers=8" in text
    assert "loader_pool=1/4/0" in text
    assert "window_reuse=yes" in text
    assert "window_setup=0.0040s" in text
    assert "promotion_evals=1" in text
    assert "crossplay_evals=1" in text
    assert "elo=321.50" in text
    assert "elo_source=aggregate" in text


def test_monitor_warns_on_live_process_loader_pool_violations(
    tmp_path,
    monkeypatch,
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    metric_path = root / "learner" / "metrics.jsonl"
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    metric.update(
        {
            "loader_workers_effective": 4,
            "loader_lifecycle": "process",
            "loader_pool_starts": 2,
            "loader_pool_rebinds": 3,
            "loader_pool_shutdowns": 1,
            "loader_worker_pids": [101],
        }
    )
    metric_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    codes = {warning["code"] for warning in snapshot["warnings"]}

    assert {
        "loader_pool_respawned",
        "loader_pool_shutdown_live",
        "loader_pool_pid_mismatch",
    } <= codes


def test_monitor_softly_ignores_malformed_strength_report(
    tmp_path,
    monkeypatch,
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    (root / "strength-efficiency.json").write_text("{partial", encoding="utf-8")
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)

    assert snapshot["strength_efficiency"] == {
        "available": False,
        "path": str(root / "strength-efficiency.json"),
        "present": True,
    }
    assert "strength_report_invalid" in {
        warning["code"] for warning in snapshot["warnings"]
    }
    assert "elo=n/a" in monitor.format_text(snapshot)


def test_monitor_derives_aggregate_headline_from_legacy_report(tmp_path) -> None:
    _write_json(
        tmp_path / "run.json",
        {
            "run_id": "legacy-run",
            "generation_family": "legacy-family",
            "created_ns": 1,
        },
    )
    _write_json(
        tmp_path / "strength-efficiency.json",
        {
            "report": "startrain-strength-efficiency",
            "schema_version": 1,
            "status": "complete",
            "run_id": "legacy-run",
            "generation_family": "legacy-family",
            "run_root": str(tmp_path),
            "started_ns": 1,
            "observed_until_ns": 10,
            "autonomous_elo": {
                "latest": {"source": "ring_10", "rating": 500.0},
                "latest_elo": 500.0,
                "aggregate": {
                    "status": "available",
                    "latest": {
                        "rating": 300.0,
                        "confidence_interval": [250.0, 350.0],
                    },
                },
            },
        },
    )

    status = monitor._strength_efficiency_status(tmp_path, now_ns=10)

    assert status["headline_elo"] == 300.0
    assert status["headline_source"] == "aggregate"
    assert status["headline_confidence_interval"] == [250.0, 350.0]


def test_monitor_escalates_stale_strength_report_for_active_run(
    tmp_path,
    monkeypatch,
) -> None:
    now_ns = 4_000_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    _write_json(
        root / "strength-efficiency.json",
        {
            "report": "startrain-strength-efficiency",
            "schema_version": 1,
            "status": "complete",
            "run_id": "monitor-run",
            "generation_family": "monitor-family",
            "run_root": str(root),
            "started_ns": now_ns - 1_000_000_000,
            "observed_until_ns": now_ns
            - int((monitor.STRENGTH_REPORT_ERROR_SECONDS + 1) * 1e9),
            "autonomous_elo": {
                "headline": {"source": "aggregate", "rating": 100.0},
                "headline_elo": 100.0,
                "aggregate": {
                    "statistical_role": "descriptive_only",
                    "adoption_ranking_authorized": False,
                },
            },
        },
    )
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)

    warning = next(
        item for item in snapshot["warnings"] if item["code"] == "strength_report_stale"
    )
    assert warning["severity"] == "ERROR"
    assert snapshot["strength_efficiency"]["statistical_role"] == "descriptive_only"
    assert snapshot["strength_efficiency"]["adoption_ranking_authorized"] is False


def test_strength_report_requires_exact_run_root_and_allows_small_clock_skew(
    tmp_path,
) -> None:
    now_ns = 20_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    report = {
        "report": "startrain-strength-efficiency",
        "schema_version": 1,
        "status": "complete",
        "run_id": "monitor-run",
        "generation_family": "monitor-family",
        "run_root": str(root),
        "started_ns": now_ns - 1_000_000_000,
        "observed_until_ns": now_ns + 1_000_000_000,
        "autonomous_elo": {},
    }
    _write_json(root / "strength-efficiency.json", report)

    accepted = monitor._strength_efficiency_status(root, now_ns=now_ns)
    assert accepted["available"] is True
    assert accepted["age_seconds"] == 0

    report["run_root"] = str(tmp_path / "sibling")
    _write_json(root / "strength-efficiency.json", report)
    rejected = monitor._strength_efficiency_status(root, now_ns=now_ns)
    assert rejected["available"] is False
    assert rejected["reason"] == "report_contract_invalid"


def test_monitor_surfaces_optimizer_ema_and_training_health(
    tmp_path,
    monkeypatch,
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    metric_path = root / "learner" / "metrics.jsonl"
    metric = json.loads(metric_path.read_text(encoding="utf-8").splitlines()[-1])
    metric.update(
        {
            "gradient_clipped": True,
            "gradient_clipped_steps": 8,
            "gradient_clipping_frequency": 0.8,
            "nonfinite_loss_count": 1,
            "nonfinite_gradient_count": 0,
            "optimizer_routing_hash": "sha256-" + "a" * 64,
            "optimizer_weight_norm": 10.0,
            "optimizer_update_norm": 0.1,
            "scheduler_age_steps": 42,
            "scheduler_segment": "cosine",
            "scheduler_segment_position": 0.25,
            "raw_vs_ema_distance": 2.0,
            "raw_vs_ema_relative_distance": 0.2,
            "ema_effective_turnover": 0.5,
            "ema_interval_effective_turnover": 0.01,
            "replay_minimum_shard_id_exclusive": 123,
        }
    )
    metric_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)
    learner = snapshot["learner"]
    codes = {warning["code"] for warning in snapshot["warnings"]}

    assert learner["optimizer_routing_hash"].endswith("a" * 64)
    assert learner["scheduler_age_steps"] == 42
    assert learner["ema_effective_turnover"] == 0.5
    assert learner["replay_minimum_shard_id_exclusive"] == 123
    assert {"learner_nonfinite", "gradient_clipping_high"} <= codes
    assert snapshot["status"] == "ERROR"


def test_disaster_recovery_status_verifies_lambda_snapshot(tmp_path) -> None:
    now_ns = 200_000_000_000
    run_id = "run-1"
    backup_root = tmp_path / "backup"
    active_root = tmp_path / "run"
    _write_json(
        active_root / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": "family-1",
            "created_ns": 1,
        },
    )
    _write_json(
        backup_root / "namespace.json",
        {
            "schema_version": 1,
            "report": "startrain-disaster-recovery-namespace",
            "run_id": run_id,
            "generation_family": "family-1",
            "source_run_root": str(active_root),
        },
    )
    run_snapshots = backup_root / "snapshots" / run_id
    snapshot = {
        "schema_version": 1,
        "report": "startrain-disaster-recovery-snapshot",
        "run_id": run_id,
        "generation_family": "family-1",
        "created_ns": now_ns - 60_000_000_000,
        "source": {
            "run_root": str(active_root),
            "replay_backup": {
                "created_ns": now_ns - 90_000_000_000,
            },
        },
        "catalog": {"run.json": {"sha256": "a" * 64, "bytes": 1, "kind": "run"}},
    }
    snapshot_payload = (
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    snapshot_sha256 = hashlib.sha256(snapshot_payload).hexdigest()
    snapshot_name = f"{snapshot['created_ns']}-{snapshot_sha256}.json"
    run_snapshots.mkdir(parents=True)
    (run_snapshots / snapshot_name).write_bytes(snapshot_payload)
    _write_json(
        run_snapshots / "latest.json",
        {
            "schema_version": 1,
            "report": "startrain-disaster-recovery-latest",
            "run_id": run_id,
            "generation_family": "family-1",
            "path": snapshot_name,
            "sha256": snapshot_sha256,
            "bytes": len(snapshot_payload),
            "created_ns": snapshot["created_ns"],
        },
    )
    status = monitor._disaster_recovery_status(
        backup_root,
        run_id=run_id,
        run_root=active_root,
        now_ns=now_ns,
    )

    assert status["valid"] is True
    assert status["snapshot_age_seconds"] == 60.0
    assert status["source_cutoff_age_seconds"] == 90.0
    assert status["catalog_files"] == 1

    newer = {
        **snapshot,
        "created_ns": now_ns - 10_000_000_000,
        "source": {
            "run_root": str(active_root),
            "replay_backup": {
                "created_ns": now_ns - 20_000_000_000,
            },
        },
    }
    newer_payload = (
        json.dumps(newer, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    newer_sha256 = hashlib.sha256(newer_payload).hexdigest()
    newer_name = f"{newer['created_ns']}-{newer_sha256}.json"
    (run_snapshots / newer_name).write_bytes(newer_payload)
    _write_json(
        run_snapshots / "latest.json",
        {
            "schema_version": 1,
            "report": "startrain-disaster-recovery-latest",
            "run_id": run_id,
            "generation_family": "family-1",
            "path": newer_name,
            "sha256": newer_sha256,
            "bytes": len(newer_payload),
            "created_ns": newer["created_ns"],
        },
    )

    pending = monitor._disaster_recovery_status(
        backup_root,
        run_id=run_id,
        run_root=active_root,
        now_ns=now_ns,
    )

    assert pending["valid"] is True
    assert pending["snapshot_age_seconds"] == 10.0
    assert pending["source_cutoff_age_seconds"] == 20.0

    other_root = tmp_path / "other-run"
    _write_json(
        other_root / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": "family-1",
            "created_ns": 1,
        },
    )
    wrong_workload = monitor._disaster_recovery_status(
        backup_root,
        run_id=run_id,
        run_root=other_root,
        now_ns=now_ns,
    )
    assert wrong_workload["valid"] is False
    assert wrong_workload["reason"] == "namespace_or_run_identity_invalid"


def test_disaster_recovery_status_rejects_tampered_snapshot(tmp_path) -> None:
    now_ns = 20_000_000_000
    run_id = "run-1"
    active_root = tmp_path / "run"
    _write_json(
        active_root / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": "family-1",
            "created_ns": 1,
        },
    )
    _write_json(
        tmp_path / "namespace.json",
        {
            "schema_version": 1,
            "report": "startrain-disaster-recovery-namespace",
            "run_id": run_id,
            "generation_family": "family-1",
            "source_run_root": str(active_root),
        },
    )
    run_snapshots = tmp_path / "snapshots" / run_id
    run_snapshots.mkdir(parents=True)
    snapshot_name = f"1-{'a' * 64}.json"
    (run_snapshots / snapshot_name).write_text("{}\n", encoding="utf-8")
    _write_json(
        run_snapshots / "latest.json",
        {
            "schema_version": 1,
            "report": "startrain-disaster-recovery-latest",
            "run_id": run_id,
            "generation_family": "family-1",
            "path": snapshot_name,
            "sha256": "a" * 64,
            "bytes": 3,
            "created_ns": 1,
        },
    )

    status = monitor._disaster_recovery_status(
        tmp_path,
        run_id=run_id,
        run_root=active_root,
        now_ns=now_ns,
    )

    assert status["valid"] is False
    assert status["reason"] == "snapshot_invalid"
