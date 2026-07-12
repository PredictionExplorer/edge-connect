from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

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
    assert "learner=10/100" in monitor.format_text(snapshot)


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
