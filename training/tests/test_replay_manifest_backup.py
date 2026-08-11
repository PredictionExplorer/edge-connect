from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

import scripts.replay_manifest_backup as backup_module
from scripts.replay_manifest_backup import create_backup, restore_if_corrupt
from scripts.replay_manifest_backup import (
    BackupLockBusy,
    backup_active_arm,
    backup_lock,
    create_backup_with_evidence,
)
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity, atomic_json


def _database(run_root: Path, value: str) -> None:
    identity = RunIdentity(
        run_root / "run.json",
        "run-test",
        "family-test",
        1,
    )
    atomic_json(
        identity.path,
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "created_ns": identity.created_ns,
        },
    )
    with ReplayStore(run_root / "replay") as store:
        store.register_run(identity)
        store.connection.execute("CREATE TABLE state(value TEXT NOT NULL)")
        store.connection.execute("INSERT INTO state(value) VALUES (?)", (value,))


def test_online_backup_rotates_and_restores_corrupt_manifest(tmp_path) -> None:
    run_root = tmp_path / "run"
    manifest = run_root / "replay" / "manifest.sqlite3"
    _database(run_root, "durable")
    first = create_backup(run_root, retain=2)
    second = create_backup(run_root, retain=2)
    third = create_backup(run_root, retain=2)
    assert not first.exists()
    assert second.exists()
    assert third.exists()

    orphan = run_root / "replay" / "shards" / "post-backup.npz"
    orphan.write_bytes(b"newer replay")
    manifest.write_bytes(b"not sqlite")
    restored = restore_if_corrupt(run_root)
    assert restored == third
    assert (run_root / "replay" / "restore-marker.json").is_file()
    with sqlite3.connect(manifest) as connection:
        assert connection.execute("SELECT value FROM state").fetchone()[0] == "durable"
    with ReplayStore(run_root / "replay") as store:
        assert store.reconciliation_metrics["post_restore_orphans"] == 1
    assert not orphan.exists()
    assert list(
        (run_root / "replay" / "quarantine").glob(
            "post-restore-orphan-*-post-backup.npz"
        )
    )
    damaged = list((run_root / "recovery").glob("damaged-replay-*"))
    assert len(damaged) == 1
    assert (damaged[0] / "manifest.sqlite3").read_bytes() == b"not sqlite"


def test_online_backup_preserves_create_once_initialization_marker(tmp_path) -> None:
    run_root = tmp_path / "stable-marker-run"
    _database(run_root, "durable")
    marker = run_root / "replay" / "initialized.json"
    before = marker.read_bytes()

    create_backup(run_root, retain=2)
    create_backup(run_root, retain=2)

    assert marker.read_bytes() == before


def test_missing_manifest_is_allowed_until_replay_initialization(tmp_path) -> None:
    run_root = tmp_path / "new-run"
    assert restore_if_corrupt(run_root) is None
    (run_root / "run.json").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "run.json").write_text("{}", encoding="utf-8")
    assert restore_if_corrupt(run_root) is None
    atomic_json(
        run_root / "replay" / "initialized.json",
        {
            "schema_version": 1,
            "run_id": "run-test",
            "generation_family": "family-test",
            "initialized_ns": 1,
        },
    )
    try:
        restore_if_corrupt(run_root)
    except RuntimeError as exc:
        assert "no valid backup" in str(exc)
    else:
        raise AssertionError("existing run without a manifest must fail closed")


def test_backup_rejects_structurally_valid_empty_database(tmp_path) -> None:
    run_root = tmp_path / "empty-run"
    atomic_json(
        run_root / "run.json",
        {
            "schema_version": 1,
            "run_id": "run-test",
            "generation_family": "family-test",
            "created_ns": 1,
        },
    )
    database = run_root / "replay" / "manifest.sqlite3"
    database.parent.mkdir(parents=True)
    sqlite3.connect(database).close()
    with pytest.raises(RuntimeError, match="required replay tables"):
        create_backup(run_root, retain=3)


def test_backup_enforces_hard_byte_cap(tmp_path) -> None:
    run_root = tmp_path / "capped-run"
    _database(run_root, "durable")
    with pytest.raises(RuntimeError, match="above backup cap"):
        create_backup(run_root, retain=3, max_total_bytes=1)


def test_markerless_empty_bootstrap_database_is_safely_reinitialized(tmp_path) -> None:
    run_root = tmp_path / "bootstrap-run"
    atomic_json(
        run_root / "run.json",
        {
            "schema_version": 1,
            "run_id": "run-test",
            "generation_family": "family-test",
            "created_ns": 1,
        },
    )
    with ReplayStore(run_root / "replay"):
        pass
    assert (run_root / "replay" / "manifest.sqlite3").is_file()
    assert not (run_root / "replay" / "initialized.json").exists()
    assert restore_if_corrupt(run_root) is None
    assert not (run_root / "replay" / "manifest.sqlite3").exists()
    assert list((run_root / "recovery").glob("uninitialized-replay-*"))


def test_backup_lock_serializes_overlapping_writers(tmp_path) -> None:
    run_root = tmp_path / "locked-run"
    _database(run_root, "durable")

    with backup_lock(run_root):
        with pytest.raises(BackupLockBusy, match="already active"):
            create_backup(run_root, retain=3, blocking=False)

    assert create_backup(run_root, retain=3).is_file()


def test_backup_evidence_is_captured_under_the_backup_lock(tmp_path) -> None:
    run_root = tmp_path / "evidence-run"
    _database(run_root, "durable")

    destination, evidence = create_backup_with_evidence(run_root, retain=3)

    assert evidence["path"] == str(destination)
    assert evidence["bytes"] == destination.stat().st_size
    assert evidence["sha256"]
    assert evidence["created_ns"]


def test_active_arm_backup_obeys_queue_state_and_interval(tmp_path) -> None:
    run_root = tmp_path / "active-run"
    _database(run_root, "durable")
    queue_state = tmp_path / "queue.json"
    atomic_json(
        queue_state,
        {
            "schema_version": 1,
            "report": "startrain-elo-ablation-queue",
            "queue_status": "running",
            "arms": [
                {
                    "treatment": "control",
                    "status": "running",
                    "run_root": str(run_root),
                },
                {
                    "treatment": "candidate",
                    "status": "pending",
                    "run_root": str(tmp_path / "candidate"),
                },
            ],
        },
    )

    with backup_lock(run_root):
        assert (
            backup_active_arm(
                queue_state,
                interval_seconds=3_600,
                retain=3,
                now_ns=9_000_000_000_000,
            )
            is None
        )

    first = backup_active_arm(
        queue_state,
        interval_seconds=3_600,
        retain=3,
        now_ns=10_000_000_000_000,
    )
    assert first is not None and first.is_file()
    latest = json.loads(
        (run_root / "recovery" / "replay-manifest" / "latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        backup_active_arm(
            queue_state,
            interval_seconds=3_600,
            retain=3,
            now_ns=int(latest["created_ns"]) + 3_599_000_000_000,
        )
        is None
    )

    state = json.loads(queue_state.read_text(encoding="utf-8"))
    state["queue_status"] = "completed"
    atomic_json(queue_state, state)
    assert (
        backup_active_arm(
            queue_state,
            interval_seconds=3_600,
            retain=3,
        )
        is None
    )


def test_active_arm_backup_rejects_ambiguous_or_future_state(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _database(first_root, "first")
    _database(second_root, "second")
    queue_state = tmp_path / "queue.json"
    atomic_json(
        queue_state,
        {
            "queue_status": "running",
            "arms": [
                {"status": "running", "run_root": str(first_root)},
                {"status": "running", "run_root": str(second_root)},
            ],
        },
    )
    with pytest.raises(RuntimeError, match="multiple running arms"):
        backup_active_arm(queue_state, interval_seconds=60, retain=3)

    state = json.loads(queue_state.read_text(encoding="utf-8"))
    state["arms"][1]["status"] = "pending"
    atomic_json(queue_state, state)
    create_backup(first_root, retain=3)
    latest_path = first_root / "recovery" / "replay-manifest" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["created_ns"] = 10_000
    atomic_json(latest_path, latest)
    with pytest.raises(RuntimeError, match="in the future"):
        backup_active_arm(
            queue_state,
            interval_seconds=60,
            retain=3,
            now_ns=9_999,
        )


def test_active_arm_rechecks_interval_after_acquiring_lock(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "active-run"
    _database(run_root, "durable")
    queue_state = tmp_path / "queue.json"
    atomic_json(
        queue_state,
        {
            "queue_status": "running",
            "arms": [{"status": "running", "run_root": str(run_root)}],
        },
    )
    latest_path = run_root / "recovery" / "replay-manifest" / "latest.json"

    @contextmanager
    def competing_backup(_run_root: Path, *, blocking: bool):
        assert blocking is False
        atomic_json(
            latest_path,
            {
                "schema_version": 1,
                "path": "manifest-competing.sqlite3",
                "bytes": 1,
                "sha256": "d" * 64,
                "created_ns": 10_000,
            },
        )
        yield latest_path.with_name(".backup.lock")

    monkeypatch.setattr(backup_module, "backup_lock", competing_backup)
    assert (
        backup_active_arm(
            queue_state,
            interval_seconds=60,
            retain=3,
            now_ns=10_001,
        )
        is None
    )
