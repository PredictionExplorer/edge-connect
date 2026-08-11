#!/usr/bin/env python3
"""Create and validate rotating online backups of the replay SQLite ledger."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE
from startrain.replay_store import MANIFEST_SCHEMA_VERSION
from startrain.runtime import atomic_json


class BackupLockBusy(RuntimeError):
    """Raised when another process owns a run's replay-backup lock."""


@contextmanager
def backup_lock(run_root: Path, *, blocking: bool = True) -> Iterator[Path]:
    backup_directory = run_root / "recovery" / "replay-manifest"
    backup_directory.mkdir(parents=True, exist_ok=True)
    lock_path = backup_directory / ".backup.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as exc:
            raise BackupLockBusy(
                f"replay backup is already active for {run_root}"
            ) from exc
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _run_identity(run_root: Path) -> tuple[str, str]:
    path = run_root / "run.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read run identity {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("run identity is not an object")
    run_id = payload.get("run_id")
    family = payload.get("generation_family")
    if not isinstance(run_id, str) or not isinstance(family, str):
        raise RuntimeError("run identity is incomplete")
    return run_id, family


def _integrity_ok(path: Path, *, run_root: Path, full: bool = True) -> tuple[bool, str]:
    if not path.is_file():
        return False, "database is missing"
    try:
        run_id, family = _run_identity(run_root)
        uri = f"{path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=30.0) as connection:
            if full:
                rows = connection.execute("PRAGMA integrity_check").fetchall()
                messages = [str(row[0]) for row in rows]
                if messages != ["ok"]:
                    return False, "; ".join(messages)
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            required_tables = {
                "store_metadata",
                "shards",
                "runs",
                "actor_generations",
                "games",
                "run_counters",
                "gc_watermarks",
                "cursors",
            }
            if not required_tables <= tables:
                return False, "required replay tables are missing"
            metadata = dict(
                connection.execute("SELECT key, value FROM store_metadata").fetchall()
            )
            expected_metadata = {
                "manifest_schema_version": str(MANIFEST_SCHEMA_VERSION),
                "rules_hash": RULES_HASH_WIRE,
                "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
            }
            if any(
                metadata.get(key) != value for key, value in expected_metadata.items()
            ):
                return False, "replay metadata is incompatible"
            registered = connection.execute(
                """
                SELECT 1 FROM runs
                WHERE run_id = ? AND generation_family = ?
                """,
                (run_id, family),
            ).fetchone()
            if registered is None:
                return False, "active run identity is absent from replay ledger"
    except sqlite3.Error as exc:
        return False, str(exc)
    except RuntimeError as exc:
        return False, str(exc)
    return True, "ok"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unregistered_database_is_empty(path: Path) -> bool:
    if not path.is_file():
        return True
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=30.0) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            for table in ("runs", "shards", "games"):
                if table in tables and int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                ):
                    return False
    except sqlite3.Error:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_backup_locked(
    run_root: Path,
    *,
    retain: int,
    max_total_bytes: int = 20 * 1024 * 1024 * 1024,
) -> Path:
    source_path = run_root / "replay" / "manifest.sqlite3"
    source_ok, source_reason = _integrity_ok(source_path, run_root=run_root, full=False)
    if not source_ok:
        raise RuntimeError(f"replay manifest is not backup-safe: {source_reason}")
    backup_directory = run_root / "recovery" / "replay-manifest"
    backup_directory.mkdir(parents=True, exist_ok=True)
    source_bytes = source_path.stat().st_size
    wal_path = Path(f"{source_path}-wal")
    if wal_path.is_file():
        source_bytes += wal_path.stat().st_size
    if source_bytes > max_total_bytes:
        raise RuntimeError(
            f"replay manifest requires {source_bytes} bytes, above backup cap "
            f"{max_total_bytes}"
        )
    existing = sorted(
        backup_directory.glob("manifest-*.sqlite3"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    while (
        len(existing) > 1
        and sum(path.stat().st_size for path in existing) + source_bytes
        > max_total_bytes
    ):
        existing.pop().unlink(missing_ok=True)
    required_free = max(source_bytes * 2, 1024 * 1024 * 1024)
    if shutil.disk_usage(backup_directory).free < required_free:
        raise RuntimeError(
            f"insufficient free space for replay backup; need {required_free} bytes"
        )
    destination = backup_directory / f"manifest-{time.time_ns()}.sqlite3"
    temporary = destination.with_suffix(".sqlite3.tmp")
    try:
        source_uri = f"{source_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(source_uri, uri=True, timeout=30.0) as source:
            with sqlite3.connect(temporary) as target:
                source.backup(target, pages=1024, sleep=0.05)
        ok, reason = _integrity_ok(temporary, run_root=run_root)
        if not ok:
            raise RuntimeError(f"new replay backup failed integrity check: {reason}")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(backup_directory)
        atomic_json(
            backup_directory / "latest.json",
            {
                "schema_version": 1,
                "path": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "created_ns": time.time_ns(),
            },
        )
    except sqlite3.Error as exc:
        raise RuntimeError(f"SQLite replay backup failed: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    backups = sorted(
        backup_directory.glob("manifest-*.sqlite3"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for stale in backups[retain:]:
        stale.unlink(missing_ok=True)
    retained = [path for path in backups[:retain] if path.exists()]
    while (
        len(retained) > 1
        and sum(path.stat().st_size for path in retained) > max_total_bytes
    ):
        retained.pop().unlink(missing_ok=True)
    _fsync_directory(backup_directory)
    return destination


def _backup_evidence_locked(destination: Path) -> dict[str, object]:
    latest_path = destination.parent / "latest.json"
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read replay backup pointer {latest_path}: {exc}"
        ) from exc
    if not isinstance(latest, dict) or latest.get("schema_version") != 1:
        raise RuntimeError("replay backup pointer is invalid")
    if latest.get("path") != destination.name:
        raise RuntimeError("replay backup pointer does not reference the new backup")
    size = destination.stat().st_size
    digest = _sha256(destination)
    if latest.get("bytes") != size or latest.get("sha256") != digest:
        raise RuntimeError("replay backup pointer metadata disagrees with the artifact")
    created_ns = latest.get("created_ns")
    if (
        isinstance(created_ns, bool)
        or not isinstance(created_ns, int)
        or created_ns <= 0
    ):
        raise RuntimeError("replay backup pointer timestamp is invalid")
    return {
        "schema_version": 1,
        "path": str(destination),
        "bytes": size,
        "sha256": digest,
        "created_ns": created_ns,
    }


def create_backup_with_evidence(
    run_root: Path,
    *,
    retain: int,
    max_total_bytes: int = 20 * 1024 * 1024 * 1024,
    blocking: bool = True,
) -> tuple[Path, dict[str, object]]:
    if retain <= 0:
        raise ValueError("retain must be positive")
    if max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be positive")
    root = run_root.expanduser().resolve()
    with backup_lock(root, blocking=blocking):
        destination = _create_backup_locked(
            root,
            retain=retain,
            max_total_bytes=max_total_bytes,
        )
        return destination, _backup_evidence_locked(destination)


def create_backup(
    run_root: Path,
    *,
    retain: int,
    max_total_bytes: int = 20 * 1024 * 1024 * 1024,
    blocking: bool = True,
) -> Path:
    destination, _ = create_backup_with_evidence(
        run_root,
        retain=retain,
        max_total_bytes=max_total_bytes,
        blocking=blocking,
    )
    return destination


def _restore_if_corrupt_locked(run_root: Path) -> Path | None:
    replay_directory = run_root / "replay"
    manifest = replay_directory / "manifest.sqlite3"
    initialized = replay_directory / "initialized.json"
    if not initialized.exists() and _unregistered_database_is_empty(manifest):
        if manifest.exists():
            preserved = run_root / "recovery" / f"uninitialized-replay-{time.time_ns()}"
            preserved.mkdir(parents=True, exist_ok=False)
            for suffix in ("", "-wal", "-shm"):
                artifact = Path(f"{manifest}{suffix}")
                if artifact.exists():
                    os.replace(artifact, preserved / artifact.name)
        print("replay manifest does not exist yet; allowing a new run to initialize")
        return None
    ok, reason = _integrity_ok(manifest, run_root=run_root)
    if ok:
        print(f"replay manifest integrity: ok ({manifest})")
        return None
    backup_directory = run_root / "recovery" / "replay-manifest"
    backups = sorted(
        backup_directory.glob("manifest-*.sqlite3"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    selected: Path | None = None
    for candidate in backups:
        candidate_ok, _ = _integrity_ok(candidate, run_root=run_root)
        if candidate_ok:
            selected = candidate
            break
    if selected is None:
        raise RuntimeError(
            f"replay manifest is invalid ({reason}) and no valid backup exists"
        )
    stamp = time.time_ns()
    damaged_directory = run_root / "recovery" / f"damaged-replay-{stamp}"
    damaged_directory.mkdir(parents=True, exist_ok=False)
    for suffix in ("", "-wal", "-shm"):
        artifact = Path(f"{manifest}{suffix}")
        if artifact.exists():
            os.replace(artifact, damaged_directory / artifact.name)
    temporary = replay_directory / f".manifest.restore-{stamp}.sqlite3"
    try:
        shutil.copyfile(selected, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        restored_ok, restored_reason = _integrity_ok(temporary, run_root=run_root)
        if not restored_ok:
            raise RuntimeError(
                f"selected replay backup failed restore validation: {restored_reason}"
            )
        os.replace(temporary, manifest)
        atomic_json(
            replay_directory / "restore-marker.json",
            {
                "schema_version": 1,
                "restored_from": str(selected),
                "restored_ns": time.time_ns(),
            },
        )
        _fsync_directory(replay_directory)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"restored replay manifest from {selected}; "
        f"preserved damaged files in {damaged_directory}"
    )
    return selected


def restore_if_corrupt(run_root: Path, *, blocking: bool = True) -> Path | None:
    root = run_root.expanduser().resolve()
    with backup_lock(root, blocking=blocking):
        return _restore_if_corrupt_locked(root)


def backup_active_arm(
    queue_state_path: Path,
    *,
    interval_seconds: float,
    retain: int,
    max_total_bytes: int = 20 * 1024 * 1024 * 1024,
    now_ns: int | None = None,
) -> Path | None:
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("backup interval must be positive")
    if retain <= 0:
        raise ValueError("retain must be positive")
    if max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be positive")
    if now_ns is not None and (
        isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns <= 0
    ):
        raise ValueError("now_ns must be a positive integer")
    state_path = queue_state_path.expanduser().resolve()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read ablation queue state {state_path}: {exc}"
        ) from exc
    if not isinstance(state, dict):
        raise RuntimeError("ablation queue state is not an object")
    if state.get("queue_status") != "running":
        return None
    arms = state.get("arms")
    if not isinstance(arms, list):
        raise RuntimeError("ablation queue arms are missing")
    running = [
        arm for arm in arms if isinstance(arm, dict) and arm.get("status") == "running"
    ]
    if not running:
        return None
    if len(running) != 1:
        raise RuntimeError("ablation queue has multiple running arms")
    run_root_value = running[0].get("run_root")
    if not isinstance(run_root_value, str) or not run_root_value:
        raise RuntimeError("active ablation arm run root is invalid")
    run_root = Path(run_root_value).expanduser().resolve()
    captured_ns = time.time_ns() if now_ns is None else now_ns
    try:
        with backup_lock(run_root, blocking=False):
            latest_path = run_root / "recovery" / "replay-manifest" / "latest.json"
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                latest = None
            if isinstance(latest, dict):
                created_ns = latest.get("created_ns")
                if isinstance(created_ns, int) and not isinstance(created_ns, bool):
                    if created_ns > captured_ns:
                        raise RuntimeError(
                            "latest replay backup timestamp is in the future"
                        )
                    interval_ns = int(interval_seconds * 1_000_000_000)
                    if captured_ns - created_ns < interval_ns:
                        return None
            return _create_backup_locked(
                run_root,
                retain=retain,
                max_total_bytes=max_total_bytes,
            )
    except BackupLockBusy:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("backup", "check-and-restore", "backup-active-arm"),
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--queue-state", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=3_600.0)
    parser.add_argument("--retain", type=int, default=3)
    parser.add_argument(
        "--max-total-bytes",
        type=int,
        default=20 * 1024 * 1024 * 1024,
    )
    arguments = parser.parse_args()
    if arguments.retain <= 0:
        parser.error("--retain must be positive")
    if arguments.max_total_bytes <= 0:
        parser.error("--max-total-bytes must be positive")
    if arguments.action == "backup-active-arm":
        if arguments.queue_state is None:
            parser.error("--queue-state is required for backup-active-arm")
        if arguments.run_root is not None:
            parser.error("--run-root cannot be used with backup-active-arm")
        if (
            not math.isfinite(arguments.interval_seconds)
            or arguments.interval_seconds <= 0
        ):
            parser.error("--interval-seconds must be positive")
        destination = backup_active_arm(
            arguments.queue_state,
            interval_seconds=arguments.interval_seconds,
            retain=arguments.retain,
            max_total_bytes=arguments.max_total_bytes,
        )
        print(destination if destination is not None else "no backup required")
        return
    if arguments.run_root is None:
        parser.error("--run-root is required")
    if arguments.queue_state is not None:
        parser.error("--queue-state requires backup-active-arm")
    run_root = arguments.run_root.expanduser().resolve()
    if arguments.action == "backup":
        destination = create_backup(
            run_root,
            retain=arguments.retain,
            max_total_bytes=arguments.max_total_bytes,
        )
        print(destination)
    else:
        restore_if_corrupt(run_root)


if __name__ == "__main__":
    main()
