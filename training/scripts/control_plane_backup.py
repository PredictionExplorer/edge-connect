#!/usr/bin/env python3
"""Snapshot, verify, and restore small mutable campaign control-plane state."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from scripts.training_disaster_recovery import (
    LATEST_REPORT,
    NAMESPACE_REPORT,
    COMMIT_REPORT,
    SCHEMA_VERSION,
    CatalogEntry,
    DisasterRecoveryError,
    _atomic_write_bytes,
    _backup_lock,
    _canonical_json,
    _copy_object_to_destination,
    _fsync_directory,
    _fsync_tree,
    _json_loads,
    _object_path,
    _positive_int,
    _publish_immutable_bytes,
    _rename_noreplace,
    _require_backup_mount,
    _require_separate_filesystems,
    _sha256_bytes,
    _sha256_text,
    _store_object,
    _validate_object,
)

CONTROL_REPORT = "startrain-control-plane-snapshot"
CONTROL_FAMILY = "control-plane-v1"
_CAPTURE_ATTEMPTS = 4


def _identifier(value: str) -> str:
    if (
        not value
        or len(value) > 128
        or value in {".", ".."}
        or any(not (character.isalnum() or character in "._-") for character in value)
    ):
        raise DisasterRecoveryError("namespace ID is invalid")
    return value


def _source_files(root: Path) -> tuple[list[Path], dict[str, tuple[int, ...]]]:
    files = []
    fence = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise DisasterRecoveryError(f"control-plane symlink refused: {relative}")
        if path.is_dir():
            continue
        if path.name.endswith(".lock"):
            continue
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise DisasterRecoveryError(
                f"control-plane path is not a regular file: {relative}"
            )
        logical = PurePosixPath(relative.as_posix())
        if logical.is_absolute() or any(
            part in {"", ".", ".."} for part in logical.parts
        ):
            raise DisasterRecoveryError(f"unsafe control-plane path: {relative}")
        files.append(path)
        fence[logical.as_posix()] = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
    if not files:
        raise DisasterRecoveryError("control-plane source root has no files")
    return files, fence


def _namespace(
    backup_root: Path,
    *,
    namespace_id: str,
    source_root: Path,
) -> None:
    payload = {
        "report": NAMESPACE_REPORT,
        "schema_version": SCHEMA_VERSION,
        "run_id": namespace_id,
        "generation_family": CONTROL_FAMILY,
        "source_run_root": str(source_root),
    }
    data = _canonical_json(payload)
    path = backup_root / "namespace.json"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise DisasterRecoveryError(
                "control-plane backup namespace belongs to another source root"
            )
    else:
        _publish_immutable_bytes(path, data)


def _snapshot_document(
    path: Path,
    backup_root: Path,
    *,
    full_objects: bool,
) -> tuple[dict[str, object], dict[str, CatalogEntry], str, int]:
    payload = _json_loads(path.read_bytes(), name="control-plane snapshot")
    if not isinstance(payload, dict) or set(payload) != {
        "report",
        "schema_version",
        "run_id",
        "generation_family",
        "created_ns",
        "source",
        "catalog",
    }:
        raise DisasterRecoveryError("control-plane snapshot fields are incompatible")
    if (
        payload.get("report") != CONTROL_REPORT
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("generation_family") != CONTROL_FAMILY
    ):
        raise DisasterRecoveryError("control-plane snapshot schema is unsupported")
    run_id = _identifier(str(payload.get("run_id")))
    created_ns = _positive_int("snapshot created_ns", payload.get("created_ns"))
    source = payload.get("source")
    if (
        not isinstance(source, dict)
        or set(source) != {"source_root"}
        or not isinstance(source.get("source_root"), str)
        or not Path(str(source["source_root"])).is_absolute()
    ):
        raise DisasterRecoveryError("control-plane source metadata is invalid")
    catalog_raw = payload.get("catalog")
    if not isinstance(catalog_raw, dict):
        raise DisasterRecoveryError("control-plane catalog must be a mapping")
    catalog = {}
    for logical, raw in catalog_raw.items():
        if (
            not isinstance(logical, str)
            or not isinstance(raw, dict)
            or set(raw) != {"sha256", "bytes", "kind"}
            or raw.get("kind") != "control-plane-file"
        ):
            raise DisasterRecoveryError("control-plane catalog entry is invalid")
        relative = PurePosixPath(logical)
        if (
            relative.is_absolute()
            or str(relative) != logical
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise DisasterRecoveryError("control-plane catalog path is unsafe")
        entry = CatalogEntry(
            _sha256_text("control-plane SHA-256", raw.get("sha256")),
            _positive_int("control-plane bytes", raw.get("bytes")),
            "control-plane-file",
        )
        if full_objects:
            _validate_object(
                _object_path(backup_root, entry.sha256),
                expected_sha256=entry.sha256,
                expected_bytes=entry.bytes,
            )
        catalog[logical] = entry
    data = _canonical_json(payload)
    digest = _sha256_bytes(data)
    if path.read_bytes() != data or path.name != f"{created_ns}-{digest}.json":
        raise DisasterRecoveryError("control-plane snapshot identity is invalid")
    namespace = _json_loads(
        (backup_root / "namespace.json").read_bytes(),
        name="control-plane namespace",
    )
    if not isinstance(namespace, dict) or namespace != {
        "report": NAMESPACE_REPORT,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generation_family": CONTROL_FAMILY,
        "source_run_root": source["source_root"],
    }:
        raise DisasterRecoveryError("control-plane namespace is invalid")
    return payload, catalog, digest, len(data)


def _commit_payload(
    path: Path,
    payload: Mapping[str, object],
    digest: str,
    size: int,
) -> bytes:
    return _canonical_json(
        {
            "report": COMMIT_REPORT,
            "schema_version": SCHEMA_VERSION,
            "path": path.name,
            "sha256": digest,
            "bytes": size,
            "created_ns": payload["created_ns"],
            "run_id": payload["run_id"],
            "generation_family": CONTROL_FAMILY,
        }
    )


def _is_committed(path: Path, backup: Path) -> bool:
    payload, _, digest, size = _snapshot_document(path, backup, full_objects=False)
    marker = path.with_name(f"{path.name}.commit")
    return marker.is_file() and marker.read_bytes() == _commit_payload(
        path,
        payload,
        digest,
        size,
    )


def create_control_snapshot(
    source_root: str | Path,
    backup_root: str | Path,
    *,
    namespace_id: str,
    expected_backup_mount: str | Path | None = None,
    enforce_separate_filesystem: bool = True,
) -> Path:
    source = Path(source_root).expanduser().resolve()
    backup = Path(backup_root).expanduser().resolve()
    identity = _identifier(namespace_id)
    if not source.is_dir() or source.is_symlink():
        raise DisasterRecoveryError("control-plane source root is missing or unsafe")
    if enforce_separate_filesystem:
        _require_separate_filesystems(source, backup)
    _require_backup_mount(
        backup,
        Path(expected_backup_mount) if expected_backup_mount is not None else None,
    )
    with _backup_lock(backup):
        _namespace(backup, namespace_id=identity, source_root=source)
        catalog: dict[str, CatalogEntry] | None = None
        for attempt in range(1, _CAPTURE_ATTEMPTS + 1):
            files, before = _source_files(source)
            catalog = {}
            for path in files:
                logical = path.relative_to(source).as_posix()
                stored = _store_object(path, backup)
                catalog[logical] = CatalogEntry(
                    stored.sha256,
                    stored.bytes,
                    "control-plane-file",
                )
            _, after = _source_files(source)
            if before == after:
                break
            if attempt == _CAPTURE_ATTEMPTS:
                raise DisasterRecoveryError(
                    "control-plane state changed during every capture attempt"
                )
        assert catalog is not None
        snapshots = backup / "snapshots" / identity
        snapshots.mkdir(parents=True, exist_ok=True)
        newest_created_ns = max(
            (
                _positive_int(
                    "existing control-plane snapshot created_ns",
                    _snapshot_document(
                        candidate,
                        backup,
                        full_objects=False,
                    )[0].get("created_ns"),
                )
                for candidate in snapshots.glob("*.json")
                if candidate.name != "latest.json"
            ),
            default=0,
        )
        created_ns = time.time_ns()
        if created_ns <= newest_created_ns:
            raise DisasterRecoveryError(
                "system clock did not advance beyond latest control-plane snapshot"
            )
        document = {
            "report": CONTROL_REPORT,
            "schema_version": SCHEMA_VERSION,
            "run_id": identity,
            "generation_family": CONTROL_FAMILY,
            "created_ns": created_ns,
            "source": {"source_root": str(source)},
            "catalog": {
                logical: entry.as_dict() for logical, entry in sorted(catalog.items())
            },
        }
        data = _canonical_json(document)
        digest = _sha256_bytes(data)
        snapshot = snapshots / f"{created_ns}-{digest}.json"
        _publish_immutable_bytes(snapshot, data)
        _snapshot_document(snapshot, backup, full_objects=False)
        latest = {
            "report": LATEST_REPORT,
            "schema_version": SCHEMA_VERSION,
            "run_id": identity,
            "generation_family": CONTROL_FAMILY,
            "path": snapshot.name,
            "sha256": digest,
            "bytes": len(data),
            "created_ns": created_ns,
        }
        _atomic_write_bytes(snapshots / "latest.json", _canonical_json(latest))
        _atomic_write_bytes(backup / "latest.json", _canonical_json(latest))
        _publish_immutable_bytes(
            snapshot.with_name(f"{snapshot.name}.commit"),
            _commit_payload(snapshot, document, digest, len(data)),
        )
        return snapshot


def _resolve_control_snapshot(
    snapshot: str | Path,
    backup: Path,
) -> tuple[Path, dict[str, object], dict[str, CatalogEntry], str, int]:
    path = Path(snapshot).expanduser().resolve()
    latest: dict[str, object] | None = None
    if path.name == "latest.json":
        loaded = _json_loads(path.read_bytes(), name="control-plane latest pointer")
        latest = loaded if isinstance(loaded, dict) else None
        if not isinstance(latest, dict):
            raise DisasterRecoveryError("control-plane latest pointer is invalid")
        run_id = _identifier(str(latest.get("run_id")))
        filename = latest.get("path")
        if (
            set(latest)
            != {
                "report",
                "schema_version",
                "run_id",
                "generation_family",
                "path",
                "sha256",
                "bytes",
                "created_ns",
            }
            or latest.get("report") != LATEST_REPORT
            or latest.get("schema_version") != SCHEMA_VERSION
            or latest.get("generation_family") != CONTROL_FAMILY
            or not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
        ):
            raise DisasterRecoveryError("control-plane latest pointer is incompatible")
        per_run = backup / "snapshots" / run_id
        if path.parent == backup:
            path = per_run / filename
        elif path.parent == per_run:
            path = path.parent / filename
        else:
            raise DisasterRecoveryError(
                "control-plane latest pointer is outside its namespace"
            )
    payload, catalog, digest, size = _snapshot_document(path, backup, full_objects=True)
    if latest is not None and (
        latest.get("run_id") != payload.get("run_id")
        or latest.get("sha256") != digest
        or latest.get("bytes") != size
        or latest.get("created_ns") != payload.get("created_ns")
    ):
        raise DisasterRecoveryError(
            "control-plane latest pointer disagrees with snapshot"
        )
    snapshots = [
        candidate
        for candidate in path.parent.glob("*.json")
        if candidate.name != "latest.json"
        and (candidate == path or _is_committed(candidate, backup))
    ]
    newest_created_ns = max(
        _positive_int(
            "control-plane snapshot created_ns",
            _snapshot_document(candidate, backup, full_objects=False)[0].get(
                "created_ns"
            ),
        )
        for candidate in snapshots
    )
    if (
        _positive_int(
            "control-plane snapshot created_ns",
            payload.get("created_ns"),
        )
        != newest_created_ns
    ):
        raise DisasterRecoveryError("control-plane latest pointer is stale")
    return path, payload, catalog, digest, size


def verify_control_snapshot(
    snapshot: str | Path,
    backup_root: str | Path,
) -> dict[str, object]:
    backup = Path(backup_root).expanduser().resolve()
    path, payload, catalog, digest, size = _resolve_control_snapshot(
        snapshot,
        backup,
    )
    return {
        "status": "ok",
        "snapshot": str(path),
        "snapshot_sha256": digest,
        "snapshot_bytes": size,
        "run_id": payload["run_id"],
        "catalog_files": len(catalog),
    }


def restore_control_snapshot(
    snapshot: str | Path,
    backup_root: str | Path,
    destination: str | Path,
) -> Path:
    backup = Path(backup_root).expanduser().resolve()
    _, _, catalog, _, _ = _resolve_control_snapshot(snapshot, backup)
    target = Path(destination).expanduser()
    if os.path.lexists(target):
        raise FileExistsError(target)
    target = target.resolve()
    staging = target.with_name(f".{target.name}.restore-{os.getpid()}")
    staging.mkdir(mode=0o700)
    renamed = False
    try:
        for logical, entry in sorted(catalog.items()):
            _copy_object_to_destination(
                _object_path(backup, entry.sha256),
                staging / logical,
                expected_sha256=entry.sha256,
                expected_bytes=entry.bytes,
            )
        _fsync_tree(staging)
        _rename_noreplace(staging, target)
        renamed = True
        _fsync_directory(target.parent)
        return target
    except Exception:
        if not renamed:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--source-root", required=True, type=Path)
    snapshot.add_argument("--backup-root", required=True, type=Path)
    snapshot.add_argument("--namespace-id", required=True)
    snapshot.add_argument("--expected-backup-mount", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--snapshot", required=True, type=Path)
    verify.add_argument("--backup-root", required=True, type=Path)
    restore = commands.add_parser("restore")
    restore.add_argument("--snapshot", required=True, type=Path)
    restore.add_argument("--backup-root", required=True, type=Path)
    restore.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command == "snapshot":
            path = create_control_snapshot(
                arguments.source_root,
                arguments.backup_root,
                namespace_id=arguments.namespace_id,
                expected_backup_mount=arguments.expected_backup_mount,
            )
            report = verify_control_snapshot(path, arguments.backup_root)
        elif arguments.command == "verify":
            report = verify_control_snapshot(
                arguments.snapshot,
                arguments.backup_root,
            )
        else:
            restored = restore_control_snapshot(
                arguments.snapshot,
                arguments.backup_root,
                arguments.destination,
            )
            report = {"status": "ok", "destination": str(restored)}
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
