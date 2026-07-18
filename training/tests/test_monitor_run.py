from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import monitor_run as monitor


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path, *, now_ns: int) -> Path:
    root = tmp_path / "run"
    root.mkdir()
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


def test_collect_snapshot_reports_healthy_run(tmp_path, monkeypatch) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
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
    assert snapshot["arena_history"]["recent"][-1]["per_ring_elo"]["10"] == 25.0
    assert "learner=10/100" in monitor.format_text(snapshot)
    assert "elo=42.00" in monitor.format_text(snapshot)


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
        json.dumps(learner_metric) + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "strength-efficiency.json",
        {
            "status": "complete",
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
    assert "utd_segment=1.20/1.25" in text
    assert "loader_workers=8" in text
    assert "window_reuse=yes" in text
    assert "window_setup=0.0040s" in text
    assert "promotion_evals=1" in text
    assert "crossplay_evals=1" in text
    assert "elo=321.50" in text
    assert "elo_source=aggregate" in text


def test_monitor_softly_ignores_malformed_strength_report(
    tmp_path,
    monkeypatch,
) -> None:
    now_ns = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now_ns)
    (root / "strength-efficiency.json").write_text("{partial", encoding="utf-8")
    _healthy_dependencies(monkeypatch)

    snapshot: Any = monitor.collect_snapshot(root, now_ns=now_ns)

    assert snapshot["strength_efficiency"] == {"available": False}
    assert "elo=n/a" in monitor.format_text(snapshot)


def test_monitor_derives_aggregate_headline_from_legacy_report(tmp_path) -> None:
    _write_json(
        tmp_path / "strength-efficiency.json",
        {
            "status": "complete",
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

    status = monitor._strength_efficiency_status(tmp_path)

    assert status["headline_elo"] == 300.0
    assert status["headline_source"] == "aggregate"
    assert status["headline_confidence_interval"] == [250.0, 350.0]
