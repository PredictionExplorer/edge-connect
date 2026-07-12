from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.replay_manifest_backup import create_backup, restore_if_corrupt
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
