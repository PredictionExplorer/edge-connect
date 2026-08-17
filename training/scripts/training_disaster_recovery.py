#!/usr/bin/env python3
"""Create, verify, restore, and collect immutable training snapshots."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from scripts.replay_manifest_backup import create_backup_with_evidence
from startrain.checkpoint import (
    MODEL_MANIFEST_FORMAT,
    MODEL_MANIFEST_VERSION,
    MODEL_POINTER_FORMAT,
    MODEL_POINTER_VERSION,
    RECOVERY_POINTER_FORMAT,
    RECOVERY_POINTER_VERSION,
    RESUME_CUTOVER_FORMAT,
    RESUME_CUTOVER_VERSION,
    inspect_checkpoint,
    load_model_manifest,
    load_recovery_pointer,
    load_resume_cutover,
    sha256_file,
)
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE
from startrain.model import MODEL_SCHEMA_VERSION
from startrain.replay_store import MANIFEST_SCHEMA_VERSION
from startrain.runtime import load_run_identity, validate_identifier

SNAPSHOT_REPORT = "startrain-disaster-recovery-snapshot"
LATEST_REPORT = "startrain-disaster-recovery-latest"
RESTORE_REPORT = "startrain-disaster-recovery-restore"
NAMESPACE_REPORT = "startrain-disaster-recovery-namespace"
COMMIT_REPORT = "startrain-disaster-recovery-snapshot-commit"
SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_NAME_RE = re.compile(r"^([1-9][0-9]*)-([0-9a-f]{64})\.json$")
_MAX_JSON_BYTES = 256 * 1024 * 1024
_SNAPSHOT_CAPTURE_ATTEMPTS = 4
_RECOVERY_JOURNAL_LIMIT = 64
_REQUIRED_REPLAY_TABLES = {
    "store_metadata",
    "shards",
    "runs",
    "actor_generations",
    "games",
    "run_counters",
    "gc_watermarks",
    "cursors",
}
_MODEL_POINTERS = (
    "learner/champion.json",
    "learner/candidate.json",
    "learner/selfplay/champion.json",
    "learner/selfplay/candidate.json",
)
_LEARNER_METADATA = (
    "learner/cadence.json",
    "learner/utd-segment.json",
    "learner/champion-warm-start.json",
    "learner/recovery.journal.jsonl",
    "learner/model-history.jsonl",
    "learner/selection-cutover.json",
    "learner/state-rebase.json",
    "learner/state-rebase.pending.json",
    "learner/learner-complete.json",
)
_PYTHON_ENVIRONMENT_NAMES = (
    "python-env.json",
    "python-env.txt",
    "python-environment.json",
    "python-environment.txt",
    "python-version.txt",
    "pip-freeze.txt",
    "requirements.freeze.txt",
)
_MANIFEST_REFERENCE_KEYS = {
    "baseline_manifest",
    "candidate_manifest",
    "champion_manifest",
}
_ALLOWED_KINDS = {
    "arena-json",
    "checkpoint",
    "learner-metadata",
    "model-manifest",
    "model-pointer",
    "profile",
    "profile-checksum",
    "python-environment",
    "recovery-pointer",
    "replay-initialization",
    "replay-ledger",
    "replay-shard",
    "resume-cutover",
    "run-identity",
    "run-metadata",
    "source-commit",
    "status-json",
}


class DisasterRecoveryError(RuntimeError):
    """A fail-closed disaster-recovery validation error."""


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    sha256: str
    bytes: int
    kind: str

    def as_dict(self) -> dict[str, object]:
        return {"sha256": self.sha256, "bytes": self.bytes, "kind": self.kind}


@dataclass(frozen=True, slots=True)
class VerifiedSnapshot:
    path: Path
    backup_root: Path
    sha256: str
    bytes: int
    payload: dict[str, Any]
    catalog: dict[str, CatalogEntry]

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])

    @property
    def generation_family(self) -> str:
        return str(self.payload["generation_family"])

    @property
    def created_ns(self) -> int:
        return int(self.payload["created_ns"])


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise DisasterRecoveryError(f"JSON object repeats key {key!r}")
        output[key] = value
    return output


def _invalid_json_constant(value: str) -> None:
    raise DisasterRecoveryError(f"JSON contains non-finite number {value}")


def _json_loads(data: bytes, *, name: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DisasterRecoveryError(f"{name} is not UTF-8: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_invalid_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, DisasterRecoveryError):
            raise
        raise DisasterRecoveryError(f"cannot parse {name}: {exc}") from exc


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DisasterRecoveryError(f"cannot encode canonical JSON: {exc}") from exc
    return f"{encoded}\n".encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DisasterRecoveryError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DisasterRecoveryError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DisasterRecoveryError(f"{name} must be a non-negative integer")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_regular_file(path: Path, *, name: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DisasterRecoveryError(f"cannot inspect {name} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DisasterRecoveryError(
            f"{name} must be a regular non-symlink file: {path}"
        )
    return metadata


def _hash_file(path: Path) -> tuple[str, int]:
    _require_regular_file(path, name="artifact")
    digest = hashlib.sha256()
    total = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DisasterRecoveryError(f"artifact is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise DisasterRecoveryError(f"artifact changed while hashing: {path}")
        if total != after.st_size:
            raise DisasterRecoveryError(
                f"artifact changed length while hashing: {path}"
            )
    finally:
        os.close(descriptor)
    return digest.hexdigest(), total


def _logical_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DisasterRecoveryError(f"invalid logical path {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise DisasterRecoveryError(f"logical path is not normalized: {value!r}")
    return value


def _join_logical(parent: str, value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DisasterRecoveryError(f"invalid artifact reference {value!r}")
    reference = PurePosixPath(value)
    if reference.is_absolute():
        raise DisasterRecoveryError(f"artifact reference must be relative: {value}")
    parts = list(PurePosixPath(parent).parts)
    for part in reference.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise DisasterRecoveryError(f"artifact reference escapes root: {value}")
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        raise DisasterRecoveryError(f"artifact reference resolves to root: {value}")
    return _logical_path(str(PurePosixPath(*parts)))


def _path_within(root: Path, path: Path, *, name: str) -> tuple[Path, str]:
    _require_regular_file(path, name=name)
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise DisasterRecoveryError(f"{name} escapes run root: {path}") from exc
    logical = _logical_path(relative.as_posix())
    _require_regular_file(resolved, name=name)
    return resolved, logical


def _object_path(backup_root: Path, digest: str) -> Path:
    digest = _sha256_text("object SHA-256", digest)
    object_root = backup_root / "objects"
    digest_root = object_root / "sha256"
    prefix = digest_root / digest[:2]
    for directory in (object_root, digest_root, prefix):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise DisasterRecoveryError(
                f"object-store directory is unsafe: {directory}"
            )
    return prefix / digest


def _validate_object(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
) -> int:
    digest, size = _hash_file(path)
    if expected_bytes is not None and size != expected_bytes:
        raise DisasterRecoveryError(f"object byte length failed: {path}")
    if digest != expected_sha256:
        raise DisasterRecoveryError(f"object SHA-256 failed: {path}")
    return size


def _validate_immutable_object_metadata(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
) -> int:
    metadata = _require_regular_file(path, name="immutable backup object")
    if path.name != expected_sha256:
        raise DisasterRecoveryError(f"object path does not match SHA-256: {path}")
    if metadata.st_mode & 0o222:
        raise DisasterRecoveryError(f"immutable backup object is writable: {path}")
    if expected_bytes is not None and metadata.st_size != expected_bytes:
        raise DisasterRecoveryError(f"object byte length failed: {path}")
    return metadata.st_size


def _install_immutable(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        _rename_noreplace(temporary, destination)
    except FileExistsError:
        return
    except OSError as exc:
        raise DisasterRecoveryError(
            f"cannot publish immutable artifact {destination}: {exc}"
        ) from exc
    _fsync_directory(destination.parent)


def _store_object(
    source: Path,
    backup_root: Path,
    *,
    expected_sha256: str | None = None,
) -> CatalogEntry:
    _require_regular_file(source, name="snapshot source")
    temporary_name: str | None = None
    source_descriptor = -1
    try:
        source_descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DisasterRecoveryError(f"snapshot source is not regular: {source}")
        object_base = backup_root / "objects" / "sha256"
        object_base.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".object-",
            suffix=".tmp",
            dir=object_base,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(source_descriptor, "rb", closefd=False) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.fchmod(temporary.fileno(), 0o444)
        after = os.fstat(source_descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != after.st_size:
            raise DisasterRecoveryError(
                f"snapshot source changed while copying: {source}"
            )
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise DisasterRecoveryError(
                f"snapshot source checksum disagrees with ledger: {source}"
            )
        destination = _object_path(backup_root, actual_sha256)
        _install_immutable(Path(temporary_name), destination)
        _validate_object(
            destination,
            expected_sha256=actual_sha256,
            expected_bytes=size,
        )
        return CatalogEntry(actual_sha256, size, "payload")
    except OSError as exc:
        raise DisasterRecoveryError(
            f"cannot copy snapshot source {source}: {exc}"
        ) from exc
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _reuse_or_store_shard(
    source: Path,
    backup_root: Path,
    *,
    expected_sha256: str,
) -> CatalogEntry:
    source_metadata = _require_regular_file(source, name="ready replay shard")
    destination = _object_path(backup_root, expected_sha256)
    if destination.exists():
        size = _validate_immutable_object_metadata(
            destination,
            expected_sha256=expected_sha256,
            expected_bytes=None,
        )
        if source_metadata.st_size != size:
            raise DisasterRecoveryError(
                f"ready replay shard byte length disagrees with object: {source}"
            )
        return CatalogEntry(expected_sha256, size, "replay-shard")
    stored = _store_object(
        source,
        backup_root,
        expected_sha256=expected_sha256,
    )
    return CatalogEntry(stored.sha256, stored.bytes, "replay-shard")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _publish_immutable_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require_regular_file(path, name="immutable artifact")
        existing = path.read_bytes()
        if existing != data:
            raise DisasterRecoveryError(
                f"refusing to overwrite differing immutable artifact: {path}"
            )
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.fchmod(temporary.fileno(), 0o444)
        _install_immutable(Path(temporary_name), path)
        if path.read_bytes() != data:
            raise DisasterRecoveryError(
                f"immutable artifact publication failed: {path}"
            )
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise DisasterRecoveryError(f"no existing ancestor for {path}")
        current = current.parent
    return current


def _require_separate_filesystems(run_root: Path, backup_root: Path) -> None:
    try:
        run_device = run_root.stat().st_dev
        backup_device = _nearest_existing(backup_root).stat().st_dev
    except OSError as exc:
        raise DisasterRecoveryError(f"cannot inspect backup filesystem: {exc}") from exc
    if run_device == backup_device:
        raise DisasterRecoveryError(
            "backup root must be on a different filesystem from the active run"
        )


def _require_backup_mount(backup_root: Path, expected_mount: Path | None) -> None:
    if expected_mount is None:
        return
    mount = expected_mount.expanduser().resolve()
    if not mount.is_dir() or mount.is_symlink() or not mount.is_mount():
        raise DisasterRecoveryError(f"expected backup mount is not mounted: {mount}")
    try:
        backup_root.resolve().relative_to(mount)
    except ValueError as exc:
        raise DisasterRecoveryError(
            f"backup root is outside expected mount {mount}"
        ) from exc
    if _nearest_existing(backup_root).stat().st_dev != mount.stat().st_dev:
        raise DisasterRecoveryError("backup root device differs from expected mount")


def _bind_backup_namespace(
    backup_root: Path,
    *,
    run_root: Path,
    run_id: str,
    generation_family: str,
) -> None:
    payload = {
        "report": NAMESPACE_REPORT,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generation_family": generation_family,
        "source_run_root": str(run_root),
    }
    path = backup_root / "namespace.json"
    data = _canonical_json(payload)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise DisasterRecoveryError(
                "backup namespace belongs to a different workload root"
            )
        return
    _publish_immutable_bytes(path, data)


def _verify_backup_namespace(
    backup_root: Path,
    *,
    run_root: str,
    run_id: str,
    generation_family: str,
) -> None:
    expected = {
        "report": NAMESPACE_REPORT,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generation_family": generation_family,
        "source_run_root": run_root,
    }
    candidates = [backup_root / "namespace.json"]
    history = backup_root / "namespace-history"
    if history.is_dir() and not history.is_symlink():
        candidates.extend(sorted(history.glob("*.json")))
    if any(
        path.is_file()
        and not path.is_symlink()
        and _read_json_file(path, name="backup namespace") == expected
        for path in candidates
    ):
        return
    else:
        raise DisasterRecoveryError(
            "backup namespace disagrees with snapshot workload root"
        )


@contextmanager
def _backup_lock(backup_root: Path) -> Iterator[None]:
    backup_root.mkdir(parents=True, exist_ok=True)
    if backup_root.is_symlink() or not backup_root.is_dir():
        raise DisasterRecoveryError("backup root must be a non-symlink directory")
    lock_path = backup_root / ".disaster-recovery.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class _SnapshotBuilder:
    def __init__(self, run_root: Path, backup_root: Path) -> None:
        self.run_root = run_root
        self.backup_root = backup_root
        self.catalog: dict[str, CatalogEntry] = {}

    def add(
        self,
        source: Path,
        logical: str,
        kind: str,
        *,
        expected_sha256: str | None = None,
    ) -> CatalogEntry:
        logical = _logical_path(logical)
        current = self.catalog.get(logical)
        if current is not None:
            if expected_sha256 is not None and current.sha256 != expected_sha256:
                raise DisasterRecoveryError(
                    f"logical artifact has conflicting checksums: {logical}"
                )
            return current
        stored = _store_object(
            source,
            self.backup_root,
            expected_sha256=expected_sha256,
        )
        entry = CatalogEntry(stored.sha256, stored.bytes, kind)
        self.catalog[logical] = entry
        return entry

    def add_run_file(
        self,
        source: Path,
        kind: str,
        *,
        expected_sha256: str | None = None,
    ) -> tuple[str, CatalogEntry]:
        resolved, logical = _path_within(
            self.run_root,
            source,
            name=kind,
        )
        return logical, self.add(
            resolved,
            logical,
            kind,
            expected_sha256=expected_sha256,
        )

    def add_bytes(self, data: bytes, logical: str, kind: str) -> CatalogEntry:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".snapshot-bytes-",
            suffix=".tmp",
            dir=self.backup_root,
            delete=False,
        ) as temporary:
            path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            return self.add(path, logical, kind)
        finally:
            path.unlink(missing_ok=True)

    def add_shard(
        self,
        source: Path,
        logical: str,
        expected_sha256: str,
    ) -> CatalogEntry:
        logical = _logical_path(logical)
        current = self.catalog.get(logical)
        if current is not None:
            if current.sha256 != expected_sha256 or current.kind != "replay-shard":
                raise DisasterRecoveryError(
                    f"replay shard has conflicting catalog entry: {logical}"
                )
            return current
        entry = _reuse_or_store_shard(
            source,
            self.backup_root,
            expected_sha256=expected_sha256,
        )
        self.catalog[logical] = entry
        return entry


def _read_json_file(path: Path, *, name: str) -> dict[str, Any]:
    _require_regular_file(path, name=name)
    data = path.read_bytes()
    if len(data) > _MAX_JSON_BYTES:
        raise DisasterRecoveryError(f"{name} is too large: {path}")
    payload = _json_loads(data, name=f"{name} {path}")
    if not isinstance(payload, dict):
        raise DisasterRecoveryError(f"{name} must be a JSON object: {path}")
    return payload


def _read_catalog_json(
    builder: _SnapshotBuilder,
    entry: CatalogEntry,
    *,
    name: str,
) -> dict[str, Any]:
    if entry.bytes > _MAX_JSON_BYTES:
        raise DisasterRecoveryError(f"{name} is too large")
    path = _object_path(builder.backup_root, entry.sha256)
    payload = _json_loads(path.read_bytes(), name=name)
    if not isinstance(payload, dict):
        raise DisasterRecoveryError(f"{name} must be a JSON object")
    return payload


def _require_relative_reference(
    payload: Mapping[str, Any],
    key: str,
    *,
    name: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise DisasterRecoveryError(f"{name} {key} is invalid")
    if Path(value).is_absolute():
        raise DisasterRecoveryError(
            f"{name} {key} must be relative so the snapshot is relocatable"
        )
    return value


def _add_model_manifest(
    builder: _SnapshotBuilder,
    path: Path,
    *,
    run_id: str,
    generation_family: str,
) -> str:
    source, logical = _path_within(
        builder.run_root,
        path,
        name="immutable model manifest",
    )
    existing = builder.catalog.get(logical)
    if existing is not None:
        if existing.kind != "model-manifest":
            raise DisasterRecoveryError(f"model manifest kind conflict: {logical}")
        return logical
    try:
        manifest = load_model_manifest(source)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DisasterRecoveryError(f"invalid model manifest {source}: {exc}") from exc
    if manifest.run_id != run_id or manifest.generation_family != generation_family:
        raise DisasterRecoveryError(f"model manifest belongs to another run: {source}")
    payload = _read_json_file(source, name="model manifest")
    _require_relative_reference(
        payload,
        "checkpoint",
        name="model manifest",
    )
    builder.add(
        source,
        logical,
        "model-manifest",
        expected_sha256=manifest.manifest_sha256,
    )
    builder.add_run_file(
        manifest.checkpoint,
        "checkpoint",
        expected_sha256=manifest.checkpoint_sha256,
    )
    return logical


def _add_model_pointer(
    builder: _SnapshotBuilder,
    path: Path,
    *,
    run_id: str,
    generation_family: str,
) -> None:
    source, logical = _path_within(builder.run_root, path, name="model pointer")
    pointer_entry = builder.add(source, logical, "model-pointer")
    pointer = _read_catalog_json(
        builder,
        pointer_entry,
        name=f"model pointer {logical}",
    )
    if (
        pointer.get("format") != MODEL_POINTER_FORMAT
        or pointer.get("schema_version") != MODEL_POINTER_VERSION
    ):
        raise DisasterRecoveryError(f"unsupported model pointer: {logical}")
    _require_relative_reference(pointer, "manifest", name="model pointer")
    try:
        manifest = load_model_manifest(source)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DisasterRecoveryError(f"invalid model pointer {source}: {exc}") from exc
    if sha256_file(source) != pointer_entry.sha256:
        raise DisasterRecoveryError(f"model pointer changed during snapshot: {source}")
    if manifest.run_id != run_id or manifest.generation_family != generation_family:
        raise DisasterRecoveryError(f"model pointer belongs to another run: {source}")
    artifact = manifest.artifact_manifest or manifest.path
    _add_model_manifest(
        builder,
        artifact,
        run_id=run_id,
        generation_family=generation_family,
    )


def _add_recovery_pointer(
    builder: _SnapshotBuilder,
    path: Path,
    *,
    run_id: str,
    generation_family: str,
) -> None:
    source, logical = _path_within(builder.run_root, path, name="recovery pointer")
    entry = builder.add(source, logical, "recovery-pointer")
    payload = _read_catalog_json(builder, entry, name="recovery pointer")
    _require_relative_reference(payload, "checkpoint", name="recovery pointer")
    try:
        recovery = load_recovery_pointer(
            source,
            expected_run_id=run_id,
            expected_generation_family=generation_family,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise DisasterRecoveryError(
            f"invalid recovery pointer {source}: {exc}"
        ) from exc
    if sha256_file(source) != entry.sha256:
        raise DisasterRecoveryError(
            f"recovery pointer changed during snapshot: {source}"
        )
    builder.add_run_file(
        recovery.checkpoint,
        "checkpoint",
        expected_sha256=recovery.checkpoint_sha256,
    )


def _add_resume_cutover(
    builder: _SnapshotBuilder,
    path: Path,
    *,
    run_id: str,
    generation_family: str,
) -> None:
    source, logical = _path_within(builder.run_root, path, name="resume cutover")
    entry = builder.add(source, logical, "resume-cutover")
    payload = _read_catalog_json(builder, entry, name="resume cutover")
    _require_relative_reference(payload, "checkpoint", name="resume cutover")
    try:
        cutover = load_resume_cutover(
            source,
            expected_run_id=run_id,
            expected_generation_family=generation_family,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise DisasterRecoveryError(f"invalid resume cutover {source}: {exc}") from exc
    if sha256_file(source) != entry.sha256:
        raise DisasterRecoveryError(f"resume cutover changed during snapshot: {source}")
    builder.add_run_file(
        cutover.checkpoint,
        "checkpoint",
        expected_sha256=cutover.checkpoint_sha256,
    )


def _pointer_checkpoint(
    builder: _SnapshotBuilder,
    payload: Mapping[str, Any],
    *,
    pointer_logical: str,
    run_id: str,
    generation_family: str,
    name: str,
) -> None:
    if (
        payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
    ):
        raise DisasterRecoveryError(f"{name} belongs to another run")
    checkpoint_value = _require_relative_reference(
        payload,
        "checkpoint",
        name=name,
    )
    checkpoint_sha256 = _sha256_text(
        f"{name} checkpoint_sha256",
        payload.get("checkpoint_sha256"),
    )
    _positive_int(f"{name} checkpoint_bytes", payload.get("checkpoint_bytes"))
    source = builder.run_root / PurePosixPath(pointer_logical).parent / checkpoint_value
    builder.add_run_file(
        source,
        "checkpoint",
        expected_sha256=checkpoint_sha256,
    )


def _add_recovery_journal(
    builder: _SnapshotBuilder,
    path: Path,
    *,
    run_id: str,
    generation_family: str,
) -> None:
    source, logical = _path_within(builder.run_root, path, name="recovery journal")
    data = source.read_bytes()
    retained: list[dict[str, Any]] = []
    for line_number, line in reversed(list(enumerate(data.splitlines(), start=1))):
        if not line.strip():
            raise DisasterRecoveryError(
                f"recovery journal has blank line {line_number}"
            )
        payload = _json_loads(line, name=f"recovery journal line {line_number}")
        if not isinstance(payload, dict):
            raise DisasterRecoveryError(
                f"recovery journal line {line_number} is not an object"
            )
        if (
            payload.get("format") != RECOVERY_POINTER_FORMAT
            or payload.get("schema_version") != RECOVERY_POINTER_VERSION
        ):
            raise DisasterRecoveryError(
                f"recovery journal line {line_number} is incompatible"
            )
        checkpoint_value = _require_relative_reference(
            payload,
            "checkpoint",
            name=f"recovery journal line {line_number}",
        )
        checkpoint = (
            builder.run_root / PurePosixPath(logical).parent / checkpoint_value
        ).resolve()
        try:
            checkpoint.relative_to(builder.run_root)
        except ValueError as exc:
            raise DisasterRecoveryError(
                f"recovery journal line {line_number} checkpoint escaped run root"
            ) from exc
        if not checkpoint.is_file():
            continue
        retained.append(payload)
        if len(retained) >= _RECOVERY_JOURNAL_LIMIT:
            break
    retained.reverse()
    if not retained:
        return
    builder.add_bytes(
        b"".join(_canonical_json(payload) for payload in retained),
        logical,
        "learner-metadata",
    )
    for line_number, payload in enumerate(retained, start=1):
        _pointer_checkpoint(
            builder,
            payload,
            pointer_logical=logical,
            run_id=run_id,
            generation_family=generation_family,
            name=f"recovery journal line {line_number}",
        )


def _resolve_source_reference(
    run_root: Path,
    source: Path,
    value: object,
    *,
    name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise DisasterRecoveryError(f"{name} path is invalid")
    reference = Path(value)
    candidate = reference if reference.is_absolute() else source.parent / reference
    resolved, _ = _path_within(run_root, candidate, name=name)
    return resolved


def _add_warm_start(
    builder: _SnapshotBuilder,
    path: Path,
    *,
    run_id: str,
    generation_family: str,
) -> None:
    source, logical = _path_within(builder.run_root, path, name="warm-start marker")
    entry = builder.add(source, logical, "learner-metadata")
    payload = _read_catalog_json(builder, entry, name="champion warm-start marker")
    if (
        payload.get("format") != "startrain.champion-warm-start"
        or payload.get("schema_version") != 1
        or payload.get("status") not in ("prepared", "active")
        or payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
    ):
        raise DisasterRecoveryError("champion warm-start marker is incompatible")
    _pointer_checkpoint(
        builder,
        payload,
        pointer_logical=logical,
        run_id=run_id,
        generation_family=generation_family,
        name="champion warm-start marker",
    )
    source_manifest = payload.get("source_manifest")
    if source_manifest is not None:
        manifest_path = _resolve_source_reference(
            builder.run_root,
            source,
            source_manifest,
            name="warm-start source manifest",
        )
        manifest_logical = _add_model_manifest(
            builder,
            manifest_path,
            run_id=run_id,
            generation_family=generation_family,
        )
        manifest_entry = builder.catalog[manifest_logical]
        if manifest_entry.sha256 != _sha256_text(
            "source_manifest_sha256",
            payload.get("source_manifest_sha256"),
        ) or manifest_entry.bytes != _positive_int(
            "source_manifest_bytes",
            payload.get("source_manifest_bytes"),
        ):
            raise DisasterRecoveryError(
                "warm-start source manifest evidence disagrees with artifact"
            )


def _manifest_references(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _MANIFEST_REFERENCE_KEYS and isinstance(child, str) and child:
                yield child
            yield from _manifest_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _manifest_references(child)


def _walk_files(root: Path, *, suffixes: tuple[str, ...]) -> Iterator[Path]:
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise DisasterRecoveryError(f"snapshot directory is unsafe: {root}")
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise DisasterRecoveryError(f"cannot scan {directory}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise DisasterRecoveryError(f"snapshot tree contains symlink: {path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False) and path.suffix in suffixes:
                yield path


def _add_json_tree(
    builder: _SnapshotBuilder,
    root: Path,
    *,
    kind: str,
    run_id: str,
    generation_family: str,
    collect_manifests: bool,
) -> None:
    for source in _walk_files(root, suffixes=(".json",)):
        logical, entry = builder.add_run_file(source, kind)
        payload = _read_catalog_json(builder, entry, name=logical)
        if collect_manifests:
            for value in _manifest_references(payload):
                manifest = _resolve_source_reference(
                    builder.run_root,
                    source,
                    value,
                    name=f"manifest referenced by {logical}",
                )
                _add_model_manifest(
                    builder,
                    manifest,
                    run_id=run_id,
                    generation_family=generation_family,
                )


def _validate_profile_checksum(path: Path, profile_sha256: str) -> None:
    _require_regular_file(path, name="profile checksum")
    try:
        fields = path.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeDecodeError) as exc:
        raise DisasterRecoveryError(
            f"cannot read profile checksum {path}: {exc}"
        ) from exc
    if not fields or _SHA256_RE.fullmatch(fields[0]) is None:
        raise DisasterRecoveryError(f"profile checksum is malformed: {path}")
    if fields[0] != profile_sha256:
        raise DisasterRecoveryError(f"profile checksum disagrees with profile: {path}")


def _ready_shards(
    ledger: Path,
    *,
    run_id: str,
    generation_family: str,
) -> list[tuple[int, str, str]]:
    try:
        uri = f"{ledger.resolve().as_uri()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True, timeout=30.0) as connection:
            connection.row_factory = sqlite3.Row
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(shards)")
            }
            required = {
                "id",
                "relative_path",
                "checksum_sha256",
                "state",
                "run_id",
                "generation_family",
            }
            if not required <= columns:
                raise DisasterRecoveryError("replay shards schema is incomplete")
            output: list[tuple[int, str, str]] = []
            seen_paths: set[str] = set()
            for row in connection.execute(
                """
                SELECT id, relative_path, checksum_sha256, run_id, generation_family
                FROM shards
                WHERE state = 'ready'
                ORDER BY id
                """
            ):
                shard_id = _positive_int("replay shard id", row["id"])
                relative = row["relative_path"]
                if not isinstance(relative, str):
                    raise DisasterRecoveryError("replay shard path is not text")
                logical_relative = _logical_path(relative)
                if PurePosixPath(logical_relative).parts[0] != "shards":
                    raise DisasterRecoveryError(
                        f"ready replay shard is outside replay/shards: {relative}"
                    )
                if logical_relative in seen_paths:
                    raise DisasterRecoveryError(
                        f"replay ledger repeats ready shard path: {relative}"
                    )
                if (
                    row["run_id"] != run_id
                    or row["generation_family"] != generation_family
                ):
                    raise DisasterRecoveryError(
                        f"ready replay shard {shard_id} belongs to another run"
                    )
                seen_paths.add(logical_relative)
                output.append(
                    (
                        shard_id,
                        logical_relative,
                        _sha256_text(
                            f"replay shard {shard_id} checksum",
                            row["checksum_sha256"],
                        ),
                    )
                )
            return output
    except sqlite3.Error as exc:
        raise DisasterRecoveryError(f"cannot read replay backup ledger: {exc}") from exc


def _capture_state_fence(run_root: Path, profile: Path) -> dict[str, tuple[int, ...]]:
    paths = {
        profile.expanduser().resolve(),
        run_root / "run.json",
        run_root / "replay" / "initialized.json",
        run_root / "learner" / "recovery.json",
        run_root / "learner" / "recovery.journal.jsonl",
        run_root / "learner" / "resume-cutover.json",
        *(run_root / logical for logical in _MODEL_POINTERS),
        *(run_root / logical for logical in _LEARNER_METADATA),
    }
    fence: dict[str, tuple[int, ...]] = {}
    for path in sorted(paths):
        try:
            metadata = path.stat()
        except FileNotFoundError:
            fence[str(path)] = ()
            continue
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise DisasterRecoveryError(f"mutable state fence path is unsafe: {path}")
        fence[str(path)] = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
    return fence


def _collect_payloads(
    run_root: Path,
    profile: Path,
    backup_root: Path,
    *,
    replay_backup_retain: int,
    allow_legacy_missing_initialized: bool,
) -> tuple[dict[str, CatalogEntry], dict[str, object], str, str]:
    identity = load_run_identity(run_root / "run.json")
    builder = _SnapshotBuilder(run_root, backup_root)
    builder.add_run_file(identity.path, "run-identity")

    profile_original = profile.expanduser()
    if profile_original.is_symlink():
        raise DisasterRecoveryError("profile may not be a symbolic link")
    profile_path = profile_original.resolve()
    _require_regular_file(profile_path, name="profile")
    profile_digest, _ = _hash_file(profile_path)
    try:
        profile_logical = _logical_path(profile_path.relative_to(run_root).as_posix())
    except ValueError:
        profile_logical = _logical_path(f"profile/{profile_path.name}")
    builder.add(profile_path, profile_logical, "profile")

    checksum_paths: list[Path] = []
    for checksum in (profile_path.with_suffix(".sha256"), run_root / "profile.sha256"):
        if checksum.is_file() and checksum not in checksum_paths:
            _validate_profile_checksum(checksum, profile_digest)
            checksum_paths.append(checksum)
            try:
                logical = _logical_path(
                    checksum.resolve().relative_to(run_root).as_posix()
                )
            except ValueError:
                logical = _logical_path(f"profile/{checksum.name}")
            builder.add(checksum.resolve(), logical, "profile-checksum")

    for name in ("source-commit.txt", *_PYTHON_ENVIRONMENT_NAMES):
        path = run_root / name
        if path.is_file():
            builder.add_run_file(
                path,
                "source-commit"
                if name == "source-commit.txt"
                else "python-environment",
            )

    initialized = run_root / "replay" / "initialized.json"
    legacy_initialized_missing = not initialized.is_file()
    if legacy_initialized_missing and not allow_legacy_missing_initialized:
        raise DisasterRecoveryError(
            "active run is missing required replay/initialized.json"
        )
    if not legacy_initialized_missing and allow_legacy_missing_initialized:
        raise DisasterRecoveryError(
            "legacy markerless mode cannot be used when initialized.json exists"
        )
    if not legacy_initialized_missing:
        initialized_payload = _read_json_file(
            initialized,
            name="replay initialization marker",
        )
        if (
            initialized_payload.get("schema_version") != 1
            or initialized_payload.get("run_id") != identity.run_id
            or initialized_payload.get("generation_family")
            != identity.generation_family
        ):
            raise DisasterRecoveryError(
                "replay initialization marker identity is invalid"
            )
        _positive_int("initialized_ns", initialized_payload.get("initialized_ns"))
        builder.add_run_file(initialized, "replay-initialization")

    replay_capture_started_ns = time.time_ns()
    try:
        ledger_backup, evidence = create_backup_with_evidence(
            run_root,
            retain=replay_backup_retain,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise DisasterRecoveryError(f"online replay backup failed: {exc}") from exc
    ledger_entry = builder.add(
        ledger_backup,
        "replay/manifest.sqlite3",
        "replay-ledger",
        expected_sha256=_sha256_text(
            "replay backup SHA-256",
            evidence.get("sha256"),
        ),
    )
    if ledger_entry.bytes != _positive_int(
        "replay backup bytes",
        evidence.get("bytes"),
    ):
        raise DisasterRecoveryError("replay backup evidence has the wrong byte length")
    for _, relative, checksum in _ready_shards(
        ledger_backup,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
    ):
        source = run_root / "replay" / relative
        resolved, _ = _path_within(
            run_root / "replay", source, name="ready replay shard"
        )
        builder.add_shard(resolved, f"replay/{relative}", checksum)

    for logical in _MODEL_POINTERS:
        path = run_root / logical
        if path.is_file():
            _add_model_pointer(
                builder,
                path,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
            )
    recovery = run_root / "learner" / "recovery.json"
    if recovery.is_file():
        _add_recovery_pointer(
            builder,
            recovery,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )
    cutover = run_root / "learner" / "resume-cutover.json"
    if cutover.is_file():
        _add_resume_cutover(
            builder,
            cutover,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )

    for logical in _LEARNER_METADATA:
        path = run_root / logical
        if not path.is_file():
            continue
        if logical.endswith("recovery.journal.jsonl"):
            _add_recovery_journal(
                builder,
                path,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
            )
        elif logical.endswith("champion-warm-start.json"):
            _add_warm_start(
                builder,
                path,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
            )
        else:
            builder.add_run_file(path, "learner-metadata")
    history_root = run_root / "learner" / "champion-warm-start-history"
    for path in _walk_files(history_root, suffixes=(".json",)):
        builder.add_run_file(path, "learner-metadata")

    for path in sorted(run_root.iterdir(), key=lambda item: item.name):
        if (
            path.is_file()
            and path.name != "run.json"
            and path.suffix in (".json", ".jsonl")
        ):
            builder.add_run_file(path, "run-metadata")

    _add_json_tree(
        builder,
        run_root / "arena",
        kind="arena-json",
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        collect_manifests=True,
    )
    _add_json_tree(
        builder,
        run_root / "status",
        kind="status-json",
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        collect_manifests=False,
    )

    source = {
        "run_root": str(run_root),
        "profile": str(profile_path),
        "profile_logical_path": profile_logical,
        "legacy_initialized_missing": legacy_initialized_missing,
        "replay_backup": {
            "cutoff_started_ns": replay_capture_started_ns,
            "created_ns": _positive_int(
                "replay backup created_ns",
                evidence.get("created_ns"),
            ),
            "sha256": ledger_entry.sha256,
            "bytes": ledger_entry.bytes,
        },
    }
    return (
        dict(sorted(builder.catalog.items())),
        source,
        identity.run_id,
        identity.generation_family,
    )


def _snapshot_envelope(
    path: Path,
    backup_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, CatalogEntry],
    bytes,
    str,
]:
    _require_regular_file(path, name="snapshot document")
    data = path.read_bytes()
    payload = _json_loads(data, name=f"snapshot document {path}")
    if not isinstance(payload, dict):
        raise DisasterRecoveryError("snapshot document must be a JSON object")
    expected_keys = {
        "report",
        "schema_version",
        "run_id",
        "generation_family",
        "created_ns",
        "source",
        "catalog",
    }
    if set(payload) != expected_keys:
        raise DisasterRecoveryError("snapshot document fields are incompatible")
    if (
        payload.get("report") != SNAPSHOT_REPORT
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise DisasterRecoveryError("unsupported snapshot document")
    try:
        run_id = validate_identifier("run_id", payload.get("run_id"))
        family = validate_identifier(
            "generation_family",
            payload.get("generation_family"),
        )
    except ValueError as exc:
        raise DisasterRecoveryError(str(exc)) from exc
    created_ns = _positive_int("snapshot created_ns", payload.get("created_ns"))
    if data != _canonical_json(payload):
        raise DisasterRecoveryError("snapshot document is not canonical JSON")
    digest = _sha256_bytes(data)
    match = _SNAPSHOT_NAME_RE.fullmatch(path.name)
    if (
        match is None
        or int(match.group(1)) != created_ns
        or match.group(2) != digest
        or path.parent.name != run_id
        or path.parent.parent.name != "snapshots"
        or path.parent.parent.parent.resolve() != backup_root
    ):
        raise DisasterRecoveryError(
            "snapshot path does not match its immutable identity"
        )
    source = payload.get("source")
    if not isinstance(source, dict) or set(source) != {
        "run_root",
        "profile",
        "profile_logical_path",
        "legacy_initialized_missing",
        "replay_backup",
    }:
        raise DisasterRecoveryError("snapshot source information is incompatible")
    for key in ("run_root", "profile"):
        value = source.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise DisasterRecoveryError(f"snapshot source {key} must be absolute")
    _logical_path(str(source.get("profile_logical_path")))
    if not isinstance(source.get("legacy_initialized_missing"), bool):
        raise DisasterRecoveryError("legacy initialization flag is invalid")
    replay_backup = source.get("replay_backup")
    replay_backup_fields = (
        set(replay_backup) if isinstance(replay_backup, dict) else set()
    )
    if (
        not isinstance(replay_backup, dict)
        or not {"created_ns", "sha256", "bytes"} <= replay_backup_fields
        or not replay_backup_fields
        <= {"cutoff_started_ns", "created_ns", "sha256", "bytes"}
    ):
        raise DisasterRecoveryError("snapshot replay backup evidence is invalid")
    _positive_int("replay backup created_ns", replay_backup.get("created_ns"))
    _sha256_text("replay backup sha256", replay_backup.get("sha256"))
    _positive_int("replay backup bytes", replay_backup.get("bytes"))

    raw_catalog = payload.get("catalog")
    if not isinstance(raw_catalog, dict) or not raw_catalog:
        raise DisasterRecoveryError("snapshot catalog must be a non-empty mapping")
    catalog: dict[str, CatalogEntry] = {}
    for raw_logical, raw_entry in raw_catalog.items():
        logical = _logical_path(raw_logical)
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "sha256",
            "bytes",
            "kind",
        }:
            raise DisasterRecoveryError(f"invalid catalog entry: {logical}")
        kind = raw_entry.get("kind")
        if not isinstance(kind, str) or kind not in _ALLOWED_KINDS:
            raise DisasterRecoveryError(f"invalid catalog kind: {logical}")
        catalog[logical] = CatalogEntry(
            _sha256_text(f"{logical} sha256", raw_entry.get("sha256")),
            _nonnegative_int(f"{logical} bytes", raw_entry.get("bytes")),
            kind,
        )
    if list(raw_catalog) != sorted(raw_catalog):
        raise DisasterRecoveryError("snapshot catalog is not sorted by logical path")
    if run_id != str(payload["run_id"]) or family != str(payload["generation_family"]):
        raise DisasterRecoveryError("snapshot identity is invalid")
    return payload, catalog, data, digest


class _CatalogReader:
    def __init__(
        self,
        backup_root: Path,
        catalog: Mapping[str, CatalogEntry],
        source_root: str,
    ) -> None:
        self.backup_root = backup_root
        self.catalog = catalog
        self.source_root = Path(source_root)
        self._json: dict[str, dict[str, Any]] = {}

    def data(self, logical: str) -> bytes:
        logical = _logical_path(logical)
        entry = self.catalog.get(logical)
        if entry is None:
            raise DisasterRecoveryError(f"snapshot catalog is missing {logical}")
        return _object_path(self.backup_root, entry.sha256).read_bytes()

    def json(self, logical: str, *, name: str) -> dict[str, Any]:
        cached = self._json.get(logical)
        if cached is not None:
            return cached
        entry = self.catalog.get(logical)
        if entry is None:
            raise DisasterRecoveryError(f"snapshot catalog is missing {logical}")
        if entry.bytes > _MAX_JSON_BYTES:
            raise DisasterRecoveryError(f"{name} is too large: {logical}")
        payload = _json_loads(self.data(logical), name=name)
        if not isinstance(payload, dict):
            raise DisasterRecoveryError(f"{name} must be a JSON object")
        self._json[logical] = payload
        return payload

    def reference(
        self,
        source_logical: str,
        value: object,
        *,
        name: str,
        allow_absolute: bool = False,
    ) -> str:
        if not isinstance(value, str) or not value:
            raise DisasterRecoveryError(f"{name} reference is invalid")
        reference = Path(value)
        if reference.is_absolute():
            if not allow_absolute:
                raise DisasterRecoveryError(f"{name} reference must be relative")
            try:
                relative = reference.relative_to(self.source_root)
            except ValueError as exc:
                raise DisasterRecoveryError(
                    f"{name} absolute reference escapes source run root"
                ) from exc
            return _logical_path(relative.as_posix())
        return _join_logical(str(PurePosixPath(source_logical).parent), value)


def _validate_replay_database(
    path: Path,
    *,
    run_id: str,
    generation_family: str,
    run_created_ns: int,
    catalog: Mapping[str, CatalogEntry],
) -> tuple[int, int]:
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True, timeout=30.0) as connection:
            connection.row_factory = sqlite3.Row
            integrity = [
                str(row[0]) for row in connection.execute("PRAGMA integrity_check")
            ]
            if integrity != ["ok"]:
                raise DisasterRecoveryError(
                    "replay SQLite integrity failed: " + "; ".join(integrity)
                )
            tables = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not _REQUIRED_REPLAY_TABLES <= tables:
                raise DisasterRecoveryError("required replay tables are missing")
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM store_metadata")
            }
            expected_metadata = {
                "manifest_schema_version": str(MANIFEST_SCHEMA_VERSION),
                "rules_hash": RULES_HASH_WIRE,
                "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
            }
            if any(
                metadata.get(key) != value for key, value in expected_metadata.items()
            ):
                raise DisasterRecoveryError("replay metadata is incompatible")
            registered = connection.execute(
                """
                SELECT generation_family, created_ns FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if (
                registered is None
                or registered["generation_family"] != generation_family
                or int(registered["created_ns"]) != run_created_ns
            ):
                raise DisasterRecoveryError(
                    "active run identity is absent from replay ledger"
                )
            counter = connection.execute(
                """
                SELECT committed_samples FROM run_counters
                WHERE run_id = ? AND generation_family = ?
                """,
                (run_id, generation_family),
            ).fetchone()
            if counter is None:
                raise DisasterRecoveryError("replay run counter is missing")
            committed_samples = _nonnegative_int(
                "committed replay samples",
                counter["committed_samples"],
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(shards)")
            }
            required_columns = {
                "id",
                "relative_path",
                "checksum_sha256",
                "state",
                "run_id",
                "generation_family",
            }
            if not required_columns <= columns:
                raise DisasterRecoveryError("replay shards schema is incomplete")
            expected_shards: set[str] = set()
            for row in connection.execute(
                """
                SELECT id, relative_path, checksum_sha256, run_id, generation_family
                FROM shards WHERE state = 'ready' ORDER BY id
                """
            ):
                shard_id = _positive_int("replay shard id", row["id"])
                relative = _logical_path(str(row["relative_path"]))
                if PurePosixPath(relative).parts[0] != "shards":
                    raise DisasterRecoveryError(
                        f"ready replay shard {shard_id} escaped replay/shards"
                    )
                if (
                    row["run_id"] != run_id
                    or row["generation_family"] != generation_family
                ):
                    raise DisasterRecoveryError(
                        f"ready replay shard {shard_id} belongs to another run"
                    )
                logical = f"replay/{relative}"
                if logical in expected_shards:
                    raise DisasterRecoveryError(
                        f"replay ledger repeats ready shard {logical}"
                    )
                expected_shards.add(logical)
                entry = catalog.get(logical)
                checksum = _sha256_text(
                    f"replay shard {shard_id} checksum",
                    row["checksum_sha256"],
                )
                if (
                    entry is None
                    or entry.kind != "replay-shard"
                    or entry.sha256 != checksum
                ):
                    raise DisasterRecoveryError(
                        f"ready replay shard is missing from catalog: {logical}"
                    )
            catalog_shards = {
                logical
                for logical, entry in catalog.items()
                if entry.kind == "replay-shard"
            }
            if catalog_shards != expected_shards:
                raise DisasterRecoveryError(
                    "replay shard catalog disagrees with ledger ready rows"
                )
            return len(expected_shards), committed_samples
    except sqlite3.Error as exc:
        raise DisasterRecoveryError(
            f"cannot validate replay SQLite ledger: {exc}"
        ) from exc


def _validate_model_manifest(
    reader: _CatalogReader,
    logical: str,
    *,
    run_id: str,
    generation_family: str,
    referenced_checkpoints: set[str],
) -> dict[str, Any]:
    entry = reader.catalog.get(logical)
    if entry is None or entry.kind != "model-manifest":
        raise DisasterRecoveryError(
            f"model manifest catalog entry is missing: {logical}"
        )
    payload = reader.json(logical, name=f"model manifest {logical}")
    if (
        payload.get("format") != MODEL_MANIFEST_FORMAT
        or payload.get("schema_version") != MODEL_MANIFEST_VERSION
        or payload.get("rules_hash") != RULES_HASH_WIRE
        or payload.get("feature_schema_hash") != f"{FEATURE_SCHEMA_HASH:016x}"
        or payload.get("model_schema_version") != MODEL_SCHEMA_VERSION
        or payload.get("weights") != "ema"
        or payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
    ):
        raise DisasterRecoveryError(f"model manifest is incompatible: {logical}")
    model_identity = payload.get("model_identity")
    if (
        not isinstance(model_identity, str)
        or not model_identity.startswith("sha256-")
        or payload.get("model_version") != model_identity
    ):
        raise DisasterRecoveryError(f"model identity is invalid: {logical}")
    _nonnegative_int("model_step", payload.get("model_step"))
    _positive_int("model manifest created_ns", payload.get("created_ns"))
    if PurePosixPath(logical).name != f"manifest-{entry.sha256}.json":
        raise DisasterRecoveryError(
            f"model manifest filename is not content-addressed: {logical}"
        )
    checkpoint = reader.reference(
        logical,
        payload.get("checkpoint"),
        name="model checkpoint",
    )
    checkpoint_entry = reader.catalog.get(checkpoint)
    expected_sha256 = _sha256_text(
        "model checkpoint SHA-256",
        payload.get("checkpoint_sha256"),
    )
    expected_bytes = _positive_int(
        "model checkpoint bytes",
        payload.get("checkpoint_bytes"),
    )
    if (
        checkpoint_entry is None
        or checkpoint_entry.kind != "checkpoint"
        or checkpoint_entry.sha256 != expected_sha256
        or checkpoint_entry.bytes != expected_bytes
        or PurePosixPath(checkpoint).name != f"sha256-{expected_sha256}.pt"
    ):
        raise DisasterRecoveryError(
            f"model checkpoint is missing or incomplete: {checkpoint}"
        )
    referenced_checkpoints.add(checkpoint)
    return payload


def _validate_model_pointer(
    reader: _CatalogReader,
    logical: str,
    *,
    run_id: str,
    generation_family: str,
    referenced_manifests: set[str],
    referenced_checkpoints: set[str],
) -> dict[str, Any]:
    payload = reader.json(logical, name=f"model pointer {logical}")
    if (
        payload.get("format") != MODEL_POINTER_FORMAT
        or payload.get("schema_version") != MODEL_POINTER_VERSION
        or payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
        or payload.get("role") not in ("candidate", "champion")
        or payload.get("role") != PurePosixPath(logical).stem
    ):
        raise DisasterRecoveryError(f"model pointer is incompatible: {logical}")
    _positive_int("model pointer updated_ns", payload.get("updated_ns"))
    manifest_logical = reader.reference(
        logical,
        payload.get("manifest"),
        name="model manifest",
    )
    manifest_entry = reader.catalog.get(manifest_logical)
    expected_sha256 = _sha256_text(
        "model manifest SHA-256",
        payload.get("manifest_sha256"),
    )
    expected_bytes = _positive_int(
        "model manifest bytes",
        payload.get("manifest_bytes"),
    )
    if (
        manifest_entry is None
        or manifest_entry.kind != "model-manifest"
        or manifest_entry.sha256 != expected_sha256
        or manifest_entry.bytes != expected_bytes
    ):
        raise DisasterRecoveryError(
            f"model pointer manifest is missing: {manifest_logical}"
        )
    manifest = _validate_model_manifest(
        reader,
        manifest_logical,
        run_id=run_id,
        generation_family=generation_family,
        referenced_checkpoints=referenced_checkpoints,
    )
    for key in ("model_identity", "model_step", "run_id", "generation_family"):
        if payload.get(key) != manifest.get(key):
            raise DisasterRecoveryError(
                f"model pointer identity disagrees with manifest: {logical}"
            )
    referenced_manifests.add(manifest_logical)
    return manifest


def _model_pointer_examples(
    reader: _CatalogReader,
    logical: str,
    *,
    run_id: str,
    generation_family: str,
) -> int | None:
    pointer = reader.json(logical, name=f"model pointer {logical}")
    manifest_logical = reader.reference(
        logical,
        pointer.get("manifest"),
        name="model manifest",
    )
    manifest = reader.json(manifest_logical, name=f"model manifest {manifest_logical}")
    checkpoint_logical = reader.reference(
        manifest_logical,
        manifest.get("checkpoint"),
        name="model checkpoint",
    )
    checkpoint_entry = reader.catalog.get(checkpoint_logical)
    if checkpoint_entry is None:
        return None
    try:
        metadata = inspect_checkpoint(
            _object_path(reader.backup_root, checkpoint_entry.sha256),
            expected_run_id=run_id,
            expected_generation_family=generation_family,
            expected_sha256=checkpoint_entry.sha256,
            expected_bytes=checkpoint_entry.bytes,
        )
    except Exception:
        return None
    extra = metadata.get("extra")
    if not isinstance(extra, Mapping):
        return None
    value = extra.get("examples_consumed")
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _validate_checkpoint_pointer(
    reader: _CatalogReader,
    logical: str,
    payload: Mapping[str, Any],
    *,
    run_id: str,
    generation_family: str,
    name: str,
    referenced_checkpoints: set[str],
) -> str:
    if (
        payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
    ):
        raise DisasterRecoveryError(f"{name} belongs to another run")
    checkpoint = reader.reference(
        logical,
        payload.get("checkpoint"),
        name=f"{name} checkpoint",
    )
    entry = reader.catalog.get(checkpoint)
    expected_sha256 = _sha256_text(
        f"{name} checkpoint SHA-256",
        payload.get("checkpoint_sha256"),
    )
    expected_bytes = _positive_int(
        f"{name} checkpoint bytes",
        payload.get("checkpoint_bytes"),
    )
    if (
        entry is None
        or entry.kind != "checkpoint"
        or entry.sha256 != expected_sha256
        or entry.bytes != expected_bytes
    ):
        raise DisasterRecoveryError(f"{name} checkpoint is incomplete: {checkpoint}")
    referenced_checkpoints.add(checkpoint)
    return checkpoint


def _validate_jsonl(
    reader: _CatalogReader,
    logical: str,
    *,
    run_id: str,
    generation_family: str,
    recovery_journal: bool,
    referenced_checkpoints: set[str],
) -> None:
    data = reader.data(logical)
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            raise DisasterRecoveryError(f"{logical} has blank line {line_number}")
        payload = _json_loads(line, name=f"{logical} line {line_number}")
        if not isinstance(payload, dict):
            raise DisasterRecoveryError(
                f"{logical} line {line_number} is not an object"
            )
        if payload.get("run_id") not in (None, run_id) or payload.get(
            "generation_family"
        ) not in (None, generation_family):
            raise DisasterRecoveryError(
                f"{logical} line {line_number} belongs to another run"
            )
        if recovery_journal:
            if (
                payload.get("format") != RECOVERY_POINTER_FORMAT
                or payload.get("schema_version") != RECOVERY_POINTER_VERSION
            ):
                raise DisasterRecoveryError(
                    f"{logical} line {line_number} is not a recovery pointer"
                )
            _nonnegative_int(
                f"{logical} line {line_number} step",
                payload.get("step"),
            )
            _nonnegative_int(
                f"{logical} line {line_number} epoch",
                payload.get("epoch"),
            )
            _nonnegative_int(
                f"{logical} line {line_number} examples_consumed",
                payload.get("examples_consumed"),
            )
            _positive_int(
                f"{logical} line {line_number} updated_ns",
                payload.get("updated_ns"),
            )
            checkpoint = _validate_checkpoint_pointer(
                reader,
                logical,
                payload,
                run_id=run_id,
                generation_family=generation_family,
                name=f"recovery journal line {line_number}",
                referenced_checkpoints=referenced_checkpoints,
            )
            if str(PurePosixPath(checkpoint).parent) != "learner/recovery":
                raise DisasterRecoveryError(
                    f"{logical} line {line_number} checkpoint escaped recovery"
                )


def _verify_snapshot_document(
    snapshot: Path,
    backup_root: Path,
    *,
    full_objects: bool = True,
) -> VerifiedSnapshot:
    payload, catalog, data, digest = _snapshot_envelope(snapshot, backup_root)
    for logical, entry in catalog.items():
        parts = PurePosixPath(logical).parts
        if "logs" in parts or logical.endswith(("-wal", "-shm")):
            raise DisasterRecoveryError(
                f"snapshot contains excluded mutable artifact: {logical}"
            )
        object_path = _object_path(backup_root, entry.sha256)
        if full_objects:
            _validate_object(
                object_path,
                expected_sha256=entry.sha256,
                expected_bytes=entry.bytes,
            )
        else:
            _validate_immutable_object_metadata(
                object_path,
                expected_sha256=entry.sha256,
                expected_bytes=entry.bytes,
            )

    run_id = str(payload["run_id"])
    family = str(payload["generation_family"])
    source = payload["source"]
    if not isinstance(source, dict):
        raise DisasterRecoveryError("snapshot source information is invalid")
    _verify_backup_namespace(
        backup_root,
        run_root=str(source.get("run_root")),
        run_id=run_id,
        generation_family=family,
    )
    reader = _CatalogReader(
        backup_root,
        catalog,
        str(source["run_root"]),
    )
    run_entry = catalog.get("run.json")
    if run_entry is None or run_entry.kind != "run-identity":
        raise DisasterRecoveryError("snapshot is missing run.json")
    run_payload = reader.json("run.json", name="run identity")
    if (
        run_payload.get("schema_version") != 1
        or run_payload.get("run_id") != run_id
        or run_payload.get("generation_family") != family
    ):
        raise DisasterRecoveryError("snapshot run identity is incompatible")
    run_created_ns = _positive_int("run created_ns", run_payload.get("created_ns"))

    initialized = catalog.get("replay/initialized.json")
    legacy_missing = bool(source["legacy_initialized_missing"])
    if initialized is None:
        if not legacy_missing:
            raise DisasterRecoveryError(
                "snapshot is missing replay initialization marker"
            )
    else:
        if legacy_missing or initialized.kind != "replay-initialization":
            raise DisasterRecoveryError("replay initialization legacy flag disagrees")
        initialized_payload = reader.json(
            "replay/initialized.json",
            name="replay initialization marker",
        )
        if (
            initialized_payload.get("schema_version") != 1
            or initialized_payload.get("run_id") != run_id
            or initialized_payload.get("generation_family") != family
        ):
            raise DisasterRecoveryError(
                "replay initialization marker identity is invalid"
            )
        _positive_int("initialized_ns", initialized_payload.get("initialized_ns"))

    profile_logical = _logical_path(str(source["profile_logical_path"]))
    profile_entry = catalog.get(profile_logical)
    if profile_entry is None or profile_entry.kind != "profile":
        raise DisasterRecoveryError("snapshot source profile is missing")
    for logical, entry in catalog.items():
        if entry.kind != "profile-checksum":
            continue
        try:
            fields = reader.data(logical).decode("utf-8").strip().split()
        except UnicodeDecodeError as exc:
            raise DisasterRecoveryError(
                f"profile checksum is not UTF-8: {logical}"
            ) from exc
        if not fields or fields[0] != profile_entry.sha256:
            raise DisasterRecoveryError(
                f"profile checksum disagrees with profile: {logical}"
            )
    for logical, entry in catalog.items():
        if entry.kind != "source-commit":
            continue
        try:
            fields = reader.data(logical).decode("utf-8").strip().split()
        except UnicodeDecodeError as exc:
            raise DisasterRecoveryError(
                f"source commit is not UTF-8: {logical}"
            ) from exc
        if (
            not fields
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", fields[0]) is None
        ):
            raise DisasterRecoveryError(f"source commit is malformed: {logical}")

    ledger_entry = catalog.get("replay/manifest.sqlite3")
    replay_evidence = source["replay_backup"]
    if not isinstance(replay_evidence, dict):
        raise DisasterRecoveryError("snapshot replay backup evidence is invalid")
    replay_created_ns = _positive_int(
        "replay backup created_ns",
        replay_evidence.get("created_ns"),
    )
    cutoff_started_ns = _positive_int(
        "replay cutoff started_ns",
        replay_evidence.get("cutoff_started_ns", replay_created_ns),
    )
    if not cutoff_started_ns <= replay_created_ns <= int(payload["created_ns"]):
        raise DisasterRecoveryError("snapshot replay cutoff timestamps are invalid")
    if (
        ledger_entry is None
        or ledger_entry.kind != "replay-ledger"
        or ledger_entry.sha256 != replay_evidence["sha256"]
        or ledger_entry.bytes != replay_evidence["bytes"]
    ):
        raise DisasterRecoveryError("snapshot replay ledger evidence is incomplete")
    _, committed_samples = _validate_replay_database(
        _object_path(backup_root, ledger_entry.sha256),
        run_id=run_id,
        generation_family=family,
        run_created_ns=run_created_ns,
        catalog=catalog,
    )

    referenced_manifests: set[str] = set()
    referenced_checkpoints: set[str] = set()
    recovery_examples: int | None = None
    candidate_examples: int | None = None
    for logical, entry in catalog.items():
        if entry.kind == "model-pointer":
            _validate_model_pointer(
                reader,
                logical,
                run_id=run_id,
                generation_family=family,
                referenced_manifests=referenced_manifests,
                referenced_checkpoints=referenced_checkpoints,
            )
            if PurePosixPath(logical).stem == "candidate":
                examples = _model_pointer_examples(
                    reader,
                    logical,
                    run_id=run_id,
                    generation_family=family,
                )
                if examples is not None:
                    candidate_examples = max(candidate_examples or 0, examples)
        elif entry.kind == "recovery-pointer":
            recovery_payload = reader.json(logical, name="recovery pointer")
            if (
                recovery_payload.get("format") != RECOVERY_POINTER_FORMAT
                or recovery_payload.get("schema_version") != RECOVERY_POINTER_VERSION
            ):
                raise DisasterRecoveryError("recovery pointer is incompatible")
            _nonnegative_int("recovery step", recovery_payload.get("step"))
            _nonnegative_int("recovery epoch", recovery_payload.get("epoch"))
            recovery_examples = _nonnegative_int(
                "recovery examples_consumed",
                recovery_payload.get("examples_consumed"),
            )
            _positive_int("recovery updated_ns", recovery_payload.get("updated_ns"))
            checkpoint = _validate_checkpoint_pointer(
                reader,
                logical,
                recovery_payload,
                run_id=run_id,
                generation_family=family,
                name="recovery pointer",
                referenced_checkpoints=referenced_checkpoints,
            )
            if str(PurePosixPath(checkpoint).parent) != "learner/recovery":
                raise DisasterRecoveryError(
                    "recovery pointer checkpoint escaped recovery directory"
                )
        elif entry.kind == "resume-cutover":
            cutover_payload = reader.json(logical, name="resume cutover")
            if (
                cutover_payload.get("format") != RESUME_CUTOVER_FORMAT
                or cutover_payload.get("schema_version") != RESUME_CUTOVER_VERSION
            ):
                raise DisasterRecoveryError("resume cutover is incompatible")
            _nonnegative_int("resume cutover step", cutover_payload.get("step"))
            _positive_int(
                "resume cutover created_ns",
                cutover_payload.get("created_ns"),
            )
            checkpoint = _validate_checkpoint_pointer(
                reader,
                logical,
                cutover_payload,
                run_id=run_id,
                generation_family=family,
                name="resume cutover",
                referenced_checkpoints=referenced_checkpoints,
            )
            allowed = {
                "learner/checkpoints",
                "learner/recovery",
            }
            if str(PurePosixPath(checkpoint).parent) not in allowed:
                raise DisasterRecoveryError(
                    "resume cutover checkpoint escaped artifact directories"
                )

    warm_logical = "learner/champion-warm-start.json"
    warm_examples: int | None = None
    if warm_logical in catalog:
        warm = reader.json(warm_logical, name="champion warm-start marker")
        if (
            warm.get("format") != "startrain.champion-warm-start"
            or warm.get("schema_version") != 1
            or warm.get("status") not in ("prepared", "active")
        ):
            raise DisasterRecoveryError("champion warm-start marker is incompatible")
        warm_examples = _nonnegative_int(
            "champion warm-start examples_consumed",
            warm.get("examples_consumed"),
        )
        warm_checkpoint = _validate_checkpoint_pointer(
            reader,
            warm_logical,
            warm,
            run_id=run_id,
            generation_family=family,
            name="champion warm-start marker",
            referenced_checkpoints=referenced_checkpoints,
        )
        if str(PurePosixPath(warm_checkpoint).parent) != "learner/recovery":
            raise DisasterRecoveryError(
                "champion warm-start checkpoint escaped recovery directory"
            )
        if warm.get("source_manifest") is not None:
            source_manifest = reader.reference(
                warm_logical,
                warm.get("source_manifest"),
                name="warm-start source manifest",
                allow_absolute=True,
            )
            source_entry = catalog.get(source_manifest)
            if (
                source_entry is None
                or source_entry.sha256
                != _sha256_text(
                    "warm-start source manifest SHA-256",
                    warm.get("source_manifest_sha256"),
                )
                or source_entry.bytes
                != _positive_int(
                    "warm-start source manifest bytes",
                    warm.get("source_manifest_bytes"),
                )
            ):
                raise DisasterRecoveryError(
                    "warm-start source manifest evidence is incomplete"
                )
            _validate_model_manifest(
                reader,
                source_manifest,
                run_id=run_id,
                generation_family=family,
                referenced_checkpoints=referenced_checkpoints,
            )
            referenced_manifests.add(source_manifest)
        if warm.get("status") == "active":
            cutover = reader.json(
                "learner/resume-cutover.json",
                name="active warm-start resume cutover",
            )
            if cutover.get("checkpoint_sha256") != catalog[
                warm_checkpoint
            ].sha256 or cutover.get("checkpoint_sha256") != warm.get(
                "checkpoint_sha256"
            ):
                raise DisasterRecoveryError(
                    "active warm-start marker disagrees with resume cutover"
                )

    for logical, entry in catalog.items():
        if entry.kind == "arena-json":
            arena = reader.json(logical, name=f"arena JSON {logical}")
            for value in _manifest_references(arena):
                manifest = reader.reference(
                    logical,
                    value,
                    name=f"arena manifest in {logical}",
                    allow_absolute=True,
                )
                _validate_model_manifest(
                    reader,
                    manifest,
                    run_id=run_id,
                    generation_family=family,
                    referenced_checkpoints=referenced_checkpoints,
                )
                referenced_manifests.add(manifest)
        elif entry.kind == "status-json":
            reader.json(logical, name=f"status JSON {logical}")
        elif entry.kind in ("learner-metadata", "run-metadata"):
            if logical.endswith(".jsonl"):
                _validate_jsonl(
                    reader,
                    logical,
                    run_id=run_id,
                    generation_family=family,
                    recovery_journal=logical.endswith("recovery.journal.jsonl"),
                    referenced_checkpoints=referenced_checkpoints,
                )
            else:
                metadata = reader.json(logical, name=f"metadata {logical}")
                if metadata.get("run_id") not in (None, run_id) or metadata.get(
                    "generation_family"
                ) not in (None, family):
                    raise DisasterRecoveryError(
                        f"metadata belongs to another run: {logical}"
                    )

    durable_examples = max(
        (
            value
            for value in (recovery_examples, warm_examples, candidate_examples)
            if value is not None
        ),
        default=None,
    )
    cadence_logical = "learner/cadence.json"
    if cadence_logical in catalog:
        cadence = reader.json(cadence_logical, name="learner cadence")
        if durable_examples is None:
            raise DisasterRecoveryError(
                "learner cadence exists without durable example evidence"
            )
        for field in ("candidate_examples", "selfplay_examples"):
            value = cadence.get(field)
            if value is None:
                continue
            examples = _nonnegative_int(f"cadence {field}", value)
            if examples > durable_examples:
                raise DisasterRecoveryError(
                    f"learner cadence {field} is ahead of durable state"
                )
    utd_logical = "learner/utd-segment.json"
    if utd_logical in catalog:
        utd = reader.json(utd_logical, name="learner UTD segment")
        if durable_examples is None:
            raise DisasterRecoveryError(
                "UTD segment exists without durable example evidence"
            )
        if (
            _nonnegative_int(
                "UTD baseline examples",
                utd.get("baseline_examples_consumed"),
            )
            > durable_examples
        ):
            raise DisasterRecoveryError("UTD baseline is ahead of durable examples")
        if (
            _nonnegative_int(
                "UTD baseline replay samples",
                utd.get("baseline_committed_replay_samples"),
            )
            > committed_samples
        ):
            raise DisasterRecoveryError("UTD baseline is ahead of replay cutoff")

    manifest_entries = {
        logical for logical, entry in catalog.items() if entry.kind == "model-manifest"
    }
    checkpoint_entries = {
        logical for logical, entry in catalog.items() if entry.kind == "checkpoint"
    }
    if manifest_entries != referenced_manifests:
        raise DisasterRecoveryError(
            "model manifest catalog contains unreferenced or missing artifacts"
        )
    if checkpoint_entries != referenced_checkpoints:
        raise DisasterRecoveryError(
            "checkpoint catalog contains unreferenced or missing artifacts"
        )
    return VerifiedSnapshot(
        snapshot,
        backup_root,
        digest,
        len(data),
        payload,
        catalog,
    )


def _infer_backup_root(snapshot: Path) -> Path:
    resolved = snapshot.expanduser().resolve()
    if resolved.name == "latest.json" and (resolved.parent / "snapshots").is_dir():
        return resolved.parent
    if len(resolved.parents) < 3 or resolved.parent.parent.name != "snapshots":
        raise DisasterRecoveryError(
            "cannot infer backup root from snapshot path; pass backup_root"
        )
    return resolved.parents[2]


def _snapshot_commit_path(snapshot: Path) -> Path:
    return snapshot.with_name(f"{snapshot.name}.commit")


def _snapshot_commit_payload(snapshot: VerifiedSnapshot) -> bytes:
    return _canonical_json(
        {
            "report": COMMIT_REPORT,
            "schema_version": SCHEMA_VERSION,
            "path": snapshot.path.name,
            "sha256": snapshot.sha256,
            "bytes": snapshot.bytes,
            "created_ns": snapshot.created_ns,
            "run_id": snapshot.run_id,
            "generation_family": snapshot.generation_family,
        }
    )


def _publish_snapshot_commit(snapshot: VerifiedSnapshot) -> None:
    _publish_immutable_bytes(
        _snapshot_commit_path(snapshot.path),
        _snapshot_commit_payload(snapshot),
    )


def _snapshot_is_committed(snapshot: VerifiedSnapshot) -> bool:
    marker = _snapshot_commit_path(snapshot.path)
    return (
        marker.is_file()
        and not marker.is_symlink()
        and marker.read_bytes() == _snapshot_commit_payload(snapshot)
    )


def _snapshot_headers(run_directory: Path, backup_root: Path) -> list[VerifiedSnapshot]:
    snapshots: list[VerifiedSnapshot] = []
    for path in sorted(run_directory.iterdir()):
        if path.name == "latest.json":
            continue
        if path.name.endswith(".json.commit"):
            continue
        if path.name.startswith(".") and path.name.endswith(".tmp"):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise DisasterRecoveryError(f"unexpected snapshot-directory entry: {path}")
        payload, catalog, data, digest = _snapshot_envelope(path, backup_root)
        snapshots.append(
            VerifiedSnapshot(
                path,
                backup_root,
                digest,
                len(data),
                payload,
                catalog,
            )
        )
    snapshots.sort(key=lambda item: (item.created_ns, item.path.name))
    return snapshots


def _resolve_latest(
    latest_path: Path,
    backup_root: Path,
    *,
    verify_payload: bool,
    require_newest: bool = True,
) -> VerifiedSnapshot:
    _require_regular_file(latest_path, name="latest snapshot pointer")
    data = latest_path.read_bytes()
    payload = _json_loads(data, name="latest snapshot pointer")
    if not isinstance(payload, dict) or set(payload) != {
        "report",
        "schema_version",
        "run_id",
        "generation_family",
        "path",
        "sha256",
        "bytes",
        "created_ns",
    }:
        raise DisasterRecoveryError("latest snapshot pointer fields are invalid")
    if (
        payload.get("report") != LATEST_REPORT
        or payload.get("schema_version") != SCHEMA_VERSION
        or data != _canonical_json(payload)
    ):
        raise DisasterRecoveryError("latest snapshot pointer is incompatible")
    try:
        run_id = validate_identifier("run_id", payload.get("run_id"))
        family = validate_identifier(
            "generation_family",
            payload.get("generation_family"),
        )
    except ValueError as exc:
        raise DisasterRecoveryError(str(exc)) from exc
    filename = payload.get("path")
    per_run_latest = (backup_root / "snapshots" / run_id / "latest.json").resolve()
    global_latest = (backup_root / "latest.json").resolve()
    if (
        not isinstance(filename, str)
        or PurePosixPath(filename).name != filename
        or _SNAPSHOT_NAME_RE.fullmatch(filename) is None
        or latest_path not in (per_run_latest, global_latest)
    ):
        raise DisasterRecoveryError("latest snapshot path is invalid")
    run_directory = backup_root / "snapshots" / run_id
    target = run_directory / filename
    headers = _snapshot_headers(run_directory, backup_root)
    if not headers:
        raise DisasterRecoveryError("latest pointer has no snapshot documents")
    selected = next((item for item in headers if item.path == target), None)
    if selected is None:
        raise DisasterRecoveryError("latest pointer target does not exist")
    committed = [
        item for item in headers if item.path == target or _snapshot_is_committed(item)
    ]
    newest = committed[-1]
    if latest_path == global_latest:
        global_headers: list[VerifiedSnapshot] = []
        snapshots_root = backup_root / "snapshots"
        for candidate in sorted(snapshots_root.iterdir()):
            if candidate.is_symlink() or not candidate.is_dir():
                raise DisasterRecoveryError(
                    f"unexpected snapshots-root entry: {candidate}"
                )
            global_headers.extend(
                item
                for item in _snapshot_headers(candidate, backup_root)
                if item.path == target or _snapshot_is_committed(item)
            )
        if not global_headers:
            raise DisasterRecoveryError("global latest pointer has no snapshots")
        newest = max(
            global_headers,
            key=lambda item: (item.created_ns, item.run_id, item.path.name),
        )
    if require_newest and selected.path != newest.path:
        raise DisasterRecoveryError("latest snapshot pointer is stale")
    if (
        selected.sha256 != _sha256_text("latest snapshot sha256", payload.get("sha256"))
        or selected.bytes
        != _positive_int("latest snapshot bytes", payload.get("bytes"))
        or selected.created_ns
        != _positive_int("latest snapshot created_ns", payload.get("created_ns"))
        or selected.run_id != run_id
        or selected.generation_family != family
    ):
        raise DisasterRecoveryError("latest snapshot pointer evidence disagrees")
    return (
        _verify_snapshot_document(selected.path, backup_root)
        if verify_payload
        else selected
    )


def _verified_snapshot_report(verified: VerifiedSnapshot) -> dict[str, object]:
    return {
        "status": "ok",
        "snapshot": str(verified.path),
        "snapshot_sha256": verified.sha256,
        "snapshot_bytes": verified.bytes,
        "run_id": verified.run_id,
        "generation_family": verified.generation_family,
        "created_ns": verified.created_ns,
        "catalog_files": len(verified.catalog),
        "catalog_bytes": sum(entry.bytes for entry in verified.catalog.values()),
        "objects": len({entry.sha256 for entry in verified.catalog.values()}),
    }


def verify_snapshot(
    snapshot: str | Path,
    *,
    backup_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify one immutable snapshot, or a latest.json pointer, end to end."""

    snapshot_input = Path(snapshot).expanduser()
    if snapshot_input.is_symlink():
        raise DisasterRecoveryError("snapshot path may not be a symbolic link")
    snapshot_path = snapshot_input.resolve()
    if backup_root is not None:
        backup_input = Path(backup_root).expanduser()
        if backup_input.is_symlink():
            raise DisasterRecoveryError("backup root may not be a symbolic link")
        root = backup_input.resolve()
    else:
        root = _infer_backup_root(snapshot_path)
    if snapshot_path.name == "latest.json":
        verified = _resolve_latest(snapshot_path, root, verify_payload=True)
    else:
        verified = _verify_snapshot_document(snapshot_path, root)
    return _verified_snapshot_report(verified)


verify = verify_snapshot


def create_snapshot(
    run_root: str | Path,
    profile: str | Path,
    backup_root: str | Path,
    *,
    replay_backup_retain: int = 3,
    enforce_separate_filesystem: bool = True,
    allow_legacy_missing_initialized: bool = False,
    expected_backup_mount: str | Path | None = None,
) -> Path:
    """Create and atomically publish a fully verified immutable snapshot."""

    if replay_backup_retain <= 0:
        raise ValueError("replay_backup_retain must be positive")
    root_input = Path(run_root).expanduser()
    destination_input = Path(backup_root).expanduser()
    if root_input.is_symlink():
        raise DisasterRecoveryError("run root may not be a symbolic link")
    if destination_input.is_symlink():
        raise DisasterRecoveryError("backup root may not be a symbolic link")
    root = root_input.resolve()
    destination = destination_input.resolve()
    if not root.is_dir() or root.is_symlink():
        raise DisasterRecoveryError("run root must be a non-symlink directory")
    if (
        destination == root
        or destination in root.parents
        or root in destination.parents
    ):
        raise DisasterRecoveryError("run root and backup root must not overlap")
    if enforce_separate_filesystem:
        _require_separate_filesystems(root, destination)
    _require_backup_mount(
        destination,
        (Path(expected_backup_mount) if expected_backup_mount is not None else None),
    )
    with _backup_lock(destination):
        identity = load_run_identity(root / "run.json")
        _bind_backup_namespace(
            destination,
            run_root=root,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )
        profile_path = Path(profile)
        catalog: dict[str, CatalogEntry] | None = None
        source: dict[str, object] | None = None
        run_id: str | None = None
        family: str | None = None
        for attempt in range(1, _SNAPSHOT_CAPTURE_ATTEMPTS + 1):
            before = _capture_state_fence(root, profile_path)
            catalog, source, run_id, family = _collect_payloads(
                root,
                profile_path,
                destination,
                replay_backup_retain=replay_backup_retain,
                allow_legacy_missing_initialized=allow_legacy_missing_initialized,
            )
            after = _capture_state_fence(root, profile_path)
            if before == after:
                break
            if attempt == _SNAPSHOT_CAPTURE_ATTEMPTS:
                raise DisasterRecoveryError(
                    "learner state changed during every snapshot capture attempt"
                )
        assert catalog is not None
        assert source is not None
        assert run_id is not None
        assert family is not None
        run_directory = destination / "snapshots" / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        latest_path = run_directory / "latest.json"
        global_latest_path = destination / "latest.json"
        previous: VerifiedSnapshot | None = None
        if latest_path.exists():
            previous = _resolve_latest(
                latest_path,
                destination,
                verify_payload=False,
                require_newest=False,
            )
        if global_latest_path.exists():
            _resolve_latest(
                global_latest_path,
                destination,
                verify_payload=False,
                require_newest=False,
            )
        existing_headers: list[VerifiedSnapshot] = []
        snapshots_root = destination / "snapshots"
        for candidate in sorted(snapshots_root.iterdir()):
            if candidate.is_symlink() or not candidate.is_dir():
                raise DisasterRecoveryError(
                    f"unexpected snapshots-root entry: {candidate}"
                )
            existing_headers.extend(_snapshot_headers(candidate, destination))
        newest_created_ns = max(
            (item.created_ns for item in existing_headers),
            default=0,
        )
        created_ns = time.time_ns()
        if created_ns <= newest_created_ns or (
            previous is not None and created_ns <= previous.created_ns
        ):
            raise DisasterRecoveryError(
                "system clock did not advance beyond the latest snapshot"
            )
        document: dict[str, object] = {
            "report": SNAPSHOT_REPORT,
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "generation_family": family,
            "created_ns": created_ns,
            "source": source,
            "catalog": {
                logical: entry.as_dict() for logical, entry in sorted(catalog.items())
            },
        }
        data = _canonical_json(document)
        digest = _sha256_bytes(data)
        snapshot_path = run_directory / f"{created_ns}-{digest}.json"
        _publish_immutable_bytes(snapshot_path, data)
        verified = _verify_snapshot_document(
            snapshot_path,
            destination,
            full_objects=False,
        )
        latest = {
            "report": LATEST_REPORT,
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "generation_family": family,
            "path": snapshot_path.name,
            "sha256": verified.sha256,
            "bytes": verified.bytes,
            "created_ns": created_ns,
        }
        _atomic_write_bytes(latest_path, _canonical_json(latest))
        _resolve_latest(latest_path, destination, verify_payload=False)
        _atomic_write_bytes(global_latest_path, _canonical_json(latest))
        _resolve_latest(global_latest_path, destination, verify_payload=False)
        _publish_snapshot_commit(verified)
        return snapshot_path


snapshot = create_snapshot
snapshot_run = create_snapshot


def _copy_object_to_destination(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise DisasterRecoveryError(
            f"restore target unexpectedly exists: {destination}"
        )
    source_descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    target_descriptor = -1
    try:
        target_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_descriptor, view)
                if written <= 0:
                    raise OSError("short restore write")
                view = view[written:]
        os.fsync(target_descriptor)
        if total != expected_bytes or digest.hexdigest() != expected_sha256:
            raise DisasterRecoveryError(f"object changed during restore copy: {source}")
    finally:
        os.close(source_descriptor)
        if target_descriptor >= 0:
            os.close(target_descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    directories.sort(key=lambda path: len(path.parts), reverse=True)
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _rename_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        renameat2 = library.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        renamex_np = library.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004)
    else:
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(source, destination)
        return
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(destination)
    unsupported = {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    if error_number in unsupported:
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        if source.is_file():
            os.link(source, destination, follow_symlinks=False)
            source.unlink()
        else:
            os.rename(source, destination)
        return
    raise OSError(error_number, os.strerror(error_number), destination)


def _verify_restored_tree(
    root: Path,
    verified: VerifiedSnapshot,
) -> None:
    for logical, entry in verified.catalog.items():
        _validate_object(
            root / logical,
            expected_sha256=entry.sha256,
            expected_bytes=entry.bytes,
        )
    run = load_run_identity(root / "run.json")
    if (
        run.run_id != verified.run_id
        or run.generation_family != verified.generation_family
    ):
        raise DisasterRecoveryError("restored run identity disagrees with snapshot")
    _validate_replay_database(
        root / "replay" / "manifest.sqlite3",
        run_id=run.run_id,
        generation_family=run.generation_family,
        run_created_ns=run.created_ns,
        catalog=verified.catalog,
    )
    for logical, entry in verified.catalog.items():
        path = root / logical
        try:
            if entry.kind == "model-pointer":
                manifest = load_model_manifest(path)
                if (
                    manifest.run_id != run.run_id
                    or manifest.generation_family != run.generation_family
                ):
                    raise DisasterRecoveryError(
                        f"restored model pointer identity failed: {logical}"
                    )
            elif entry.kind == "model-manifest":
                manifest = load_model_manifest(path)
                if (
                    manifest.run_id != run.run_id
                    or manifest.generation_family != run.generation_family
                ):
                    raise DisasterRecoveryError(
                        f"restored model manifest identity failed: {logical}"
                    )
            elif entry.kind == "recovery-pointer":
                load_recovery_pointer(
                    path,
                    expected_run_id=run.run_id,
                    expected_generation_family=run.generation_family,
                )
            elif entry.kind == "resume-cutover":
                load_resume_cutover(
                    path,
                    expected_run_id=run.run_id,
                    expected_generation_family=run.generation_family,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, DisasterRecoveryError):
                raise
            raise DisasterRecoveryError(
                f"restored model/recovery validation failed for {logical}: {exc}"
            ) from exc


def _relocate_restored_profile(
    staging: Path,
    target: Path,
    verified: VerifiedSnapshot,
) -> str:
    source = verified.payload.get("source")
    if not isinstance(source, Mapping):
        raise DisasterRecoveryError("snapshot source information is invalid")
    profile_logical = _logical_path(str(source.get("profile_logical_path")))
    profile_path = staging / profile_logical
    try:
        loaded = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DisasterRecoveryError(f"cannot read restored profile: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DisasterRecoveryError("restored profile must be a mapping")
    orchestration = loaded.get("orchestration")
    if not isinstance(orchestration, dict):
        raise DisasterRecoveryError("restored profile lacks orchestration")
    directories = orchestration.get("directories")
    if not isinstance(directories, dict):
        raise DisasterRecoveryError("restored profile lacks orchestration directories")
    if directories.get("root") != source.get("run_root"):
        raise DisasterRecoveryError(
            "restored profile root disagrees with snapshot source run root"
        )
    directories["root"] = str(target)
    relative = "profile-relocated.yaml"
    destination = staging / relative
    if destination.exists() and destination != profile_path:
        raise DisasterRecoveryError("relocated profile target already exists")
    _atomic_write_bytes(
        destination,
        yaml.safe_dump(loaded, sort_keys=False).encode("utf-8"),
    )
    original_checksum = staging / "profile.sha256"
    if original_checksum.is_file():
        preserved = staging / "recovery" / "original-profile.sha256"
        preserved.parent.mkdir(parents=True, exist_ok=True)
        if preserved.exists():
            raise DisasterRecoveryError("original profile checksum archive exists")
        shutil.copy2(original_checksum, preserved)
    digest, _ = _hash_file(destination)
    checksum = f"{digest}  {relative}\n".encode()
    _atomic_write_bytes(destination.with_suffix(".sha256"), checksum)
    _atomic_write_bytes(staging / "profile.sha256", checksum)
    return relative


def _relocate_restored_metadata(
    staging: Path,
    target: Path,
    verified: VerifiedSnapshot,
) -> int:
    source = verified.payload.get("source")
    if not isinstance(source, Mapping):
        raise DisasterRecoveryError("snapshot source information is invalid")
    source_root = Path(str(source.get("run_root")))
    if not source_root.is_absolute():
        raise DisasterRecoveryError("snapshot source run root is not absolute")

    def relocate(value: object) -> tuple[object, int]:
        if isinstance(value, str) and Path(value).is_absolute():
            try:
                relative = Path(value).relative_to(source_root)
            except ValueError:
                return value, 0
            return str(target / relative), 1
        if isinstance(value, list):
            output = []
            changed = 0
            for item in value:
                relocated, count = relocate(item)
                output.append(relocated)
                changed += count
            return output, changed
        if isinstance(value, dict):
            output = {}
            changed = 0
            for key, item in value.items():
                relocated, count = relocate(item)
                output[key] = relocated
                changed += count
            return output, changed
        return value, 0

    rewritten = 0
    for logical, entry in verified.catalog.items():
        if entry.kind not in {
            "arena-json",
            "learner-metadata",
            "run-metadata",
            "status-json",
        }:
            continue
        path = staging / logical
        if logical.endswith(".jsonl"):
            output = []
            changed = 0
            for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
                payload = _json_loads(line, name=f"{logical} line {line_number}")
                relocated, count = relocate(payload)
                if not isinstance(relocated, Mapping):
                    raise DisasterRecoveryError(
                        f"relocated JSONL entry is not an object: {logical}"
                    )
                output.append(_canonical_json(relocated))
                changed += count
            if changed:
                _atomic_write_bytes(path, b"".join(output))
                rewritten += changed
            continue
        payload = _read_json_file(path, name=f"relocatable metadata {logical}")
        relocated, changed = relocate(payload)
        if changed:
            if not isinstance(relocated, Mapping):
                raise DisasterRecoveryError(
                    f"relocated metadata is not an object: {logical}"
                )
            _atomic_write_bytes(path, _canonical_json(relocated))
            rewritten += changed
    return rewritten


def restore_snapshot(
    snapshot: str | Path,
    destination: str | Path,
    *,
    backup_root: str | Path | None = None,
    recreate_initialized: bool = False,
    relocate_profile: bool = False,
) -> Path:
    """Restore a verified snapshot through a sibling staging directory."""

    snapshot_input = Path(snapshot).expanduser()
    if snapshot_input.is_symlink():
        raise DisasterRecoveryError("snapshot path may not be a symbolic link")
    snapshot_path = snapshot_input.resolve()
    if backup_root is not None:
        backup_input = Path(backup_root).expanduser()
        if backup_input.is_symlink():
            raise DisasterRecoveryError("backup root may not be a symbolic link")
        root = backup_input.resolve()
    else:
        root = _infer_backup_root(snapshot_path)
    verified = (
        _resolve_latest(snapshot_path, root, verify_payload=True)
        if snapshot_path.name == "latest.json"
        else _verify_snapshot_document(snapshot_path, root)
    )
    target = Path(destination).expanduser()
    if os.path.lexists(target):
        raise FileExistsError(f"restore destination already exists: {target}")
    target = target.resolve()
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise DisasterRecoveryError("restore destination parent is unsafe")
    source = verified.payload["source"]
    if not isinstance(source, dict):
        raise DisasterRecoveryError("snapshot source information is invalid")
    legacy_missing = bool(source["legacy_initialized_missing"])
    marker_missing = "replay/initialized.json" not in verified.catalog
    if marker_missing and (not legacy_missing or not recreate_initialized):
        raise DisasterRecoveryError(
            "legacy snapshot lacks initialized.json; use --recreate-initialized"
        )
    if recreate_initialized and (not legacy_missing or not marker_missing):
        raise DisasterRecoveryError(
            "--recreate-initialized is only valid for a legacy markerless snapshot"
        )

    staging = parent / f".{target.name}.restore-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    renamed = False
    try:
        for logical, entry in sorted(verified.catalog.items()):
            _copy_object_to_destination(
                _object_path(root, entry.sha256),
                staging / logical,
                expected_sha256=entry.sha256,
                expected_bytes=entry.bytes,
            )
        if marker_missing:
            run_payload = _read_json_file(staging / "run.json", name="run identity")
            initialized = {
                "schema_version": 1,
                "run_id": run_payload["run_id"],
                "generation_family": run_payload["generation_family"],
                "initialized_ns": time.time_ns(),
            }
            _atomic_write_bytes(
                staging / "replay" / "initialized.json",
                _canonical_json(initialized),
            )
        _verify_restored_tree(staging, verified)
        relocated_metadata_paths = (
            _relocate_restored_metadata(staging, target, verified)
            if relocate_profile
            else 0
        )
        relocated_profile = (
            _relocate_restored_profile(staging, target, verified)
            if relocate_profile
            else None
        )
        restore_marker = {
            "report": RESTORE_REPORT,
            "schema_version": SCHEMA_VERSION,
            "snapshot": str(verified.path),
            "snapshot_sha256": verified.sha256,
            "run_id": verified.run_id,
            "generation_family": verified.generation_family,
            "restored_ns": time.time_ns(),
            "recreated_initialized": marker_missing,
            "relocated_profile": relocated_profile,
            "relocated_metadata_paths": relocated_metadata_paths,
        }
        _atomic_write_bytes(
            staging / "recovery" / "disaster-recovery-restore.json",
            _canonical_json(restore_marker),
        )
        _fsync_tree(staging)
        _rename_noreplace(staging, target)
        renamed = True
        try:
            _fsync_directory(parent)
        except OSError:
            _rename_noreplace(target, staging)
            renamed = False
            try:
                _fsync_directory(parent)
            except OSError:
                pass
            raise
        return target
    except Exception:
        if not renamed:
            shutil.rmtree(staging, ignore_errors=True)
        raise


restore = restore_snapshot


def rebind_backup_namespace(
    backup_root: str | Path,
    restored_run_root: str | Path,
) -> dict[str, object]:
    """Explicitly hand one verified backup namespace to its relocated restore."""

    root = Path(backup_root).expanduser().resolve()
    restored = Path(restored_run_root).expanduser().resolve()
    if not restored.is_dir() or restored.is_symlink():
        raise DisasterRecoveryError("restored run root is missing or unsafe")
    identity = load_run_identity(restored / "run.json")
    marker = _read_json_file(
        restored / "recovery" / "disaster-recovery-restore.json",
        name="disaster recovery restore marker",
    )
    if (
        marker.get("report") != RESTORE_REPORT
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("run_id") != identity.run_id
        or marker.get("generation_family") != identity.generation_family
    ):
        raise DisasterRecoveryError("restore marker identity is invalid")
    snapshot_value = marker.get("snapshot")
    if not isinstance(snapshot_value, str) or not Path(snapshot_value).is_absolute():
        raise DisasterRecoveryError("restore marker snapshot path is invalid")
    snapshot_path = Path(snapshot_value).resolve()
    try:
        snapshot_path.relative_to(root)
    except ValueError as exc:
        raise DisasterRecoveryError(
            "restore marker snapshot is outside backup root"
        ) from exc
    verified = _verify_snapshot_document(snapshot_path, root)
    if (
        marker.get("snapshot_sha256") != verified.sha256
        or verified.run_id != identity.run_id
        or verified.generation_family != identity.generation_family
    ):
        raise DisasterRecoveryError("restore marker does not pin the source snapshot")
    source = verified.payload.get("source")
    if not isinstance(source, Mapping):
        raise DisasterRecoveryError("source snapshot metadata is invalid")
    with _backup_lock(root):
        namespace_path = root / "namespace.json"
        current = _read_json_file(namespace_path, name="backup namespace")
        expected_current = {
            "report": NAMESPACE_REPORT,
            "schema_version": SCHEMA_VERSION,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "source_run_root": str(source.get("run_root")),
        }
        replacement = {
            **expected_current,
            "source_run_root": str(restored),
        }
        if current == replacement:
            return {"status": "ok", "changed": False, "namespace": replacement}
        if current != expected_current:
            raise DisasterRecoveryError(
                "current backup namespace does not match restored source"
            )
        current_data = _canonical_json(current)
        history = root / "namespace-history"
        history.mkdir(parents=True, exist_ok=True)
        archived = history / f"{time.time_ns()}-{_sha256_bytes(current_data)}.json"
        _publish_immutable_bytes(archived, current_data)
        _atomic_write_bytes(namespace_path, _canonical_json(replacement))
        _verify_backup_namespace(
            root,
            run_root=str(restored),
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )
        return {
            "status": "ok",
            "changed": True,
            "namespace": replacement,
            "archived_namespace": str(archived),
        }


def _retention_bucket(created_ns: int, period: str) -> object:
    seconds = created_ns // 1_000_000_000
    if period == "hourly":
        return seconds // 3_600
    if period == "daily":
        return seconds // 86_400
    if period == "monthly":
        try:
            value = dt.datetime.fromtimestamp(seconds, tz=dt.UTC)
        except (OSError, OverflowError, ValueError) as exc:
            raise DisasterRecoveryError(
                f"snapshot timestamp is outside supported range: {created_ns}"
            ) from exc
        return value.year, value.month
    raise AssertionError(period)


def _retain_bucketed(
    snapshots: list[VerifiedSnapshot],
    count: int,
    period: str,
) -> set[Path]:
    retained: set[Path] = set()
    seen: set[object] = set()
    for snapshot in reversed(snapshots):
        bucket = _retention_bucket(snapshot.created_ns, period)
        if bucket in seen:
            continue
        seen.add(bucket)
        retained.add(snapshot.path)
        if len(seen) >= count:
            break
    return retained


def _all_objects(backup_root: Path) -> Iterator[Path]:
    base = backup_root / "objects" / "sha256"
    if not base.exists():
        return
    if base.is_symlink() or not base.is_dir():
        raise DisasterRecoveryError("object store root is unsafe")
    for prefix in sorted(base.iterdir()):
        if prefix.name.startswith(".") and prefix.name.endswith(".tmp"):
            continue
        if prefix.is_symlink() or not prefix.is_dir():
            raise DisasterRecoveryError(f"object prefix is unsafe: {prefix}")
        if re.fullmatch(r"[0-9a-f]{2}", prefix.name) is None:
            raise DisasterRecoveryError(f"invalid object prefix: {prefix}")
        for path in sorted(prefix.iterdir()):
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                continue
            if (
                path.is_symlink()
                or not path.is_file()
                or _SHA256_RE.fullmatch(path.name) is None
                or path.name[:2] != prefix.name
            ):
                raise DisasterRecoveryError(f"invalid object-store entry: {path}")
            yield path


def garbage_collect(
    backup_root: str | Path,
    *,
    retain_latest: int = 96,
    retain_hourly: int = 24,
    retain_daily: int = 30,
    retain_monthly: int = 12,
    grace_seconds: float = 7 * 86_400,
    apply: bool = False,
    now_ns: int | None = None,
) -> dict[str, object]:
    """Prune unretained snapshots and old unreachable objects."""

    for name, value in (
        ("retain_latest", retain_latest),
        ("retain_hourly", retain_hourly),
        ("retain_daily", retain_daily),
        ("retain_monthly", retain_monthly),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise ValueError("grace_seconds must be finite and non-negative")
    captured_ns = time.time_ns() if now_ns is None else now_ns
    if (
        isinstance(captured_ns, bool)
        or not isinstance(captured_ns, int)
        or captured_ns <= 0
    ):
        raise ValueError("now_ns must be a positive integer")
    root_input = Path(backup_root).expanduser()
    if root_input.is_symlink():
        raise DisasterRecoveryError("backup root may not be a symbolic link")
    root = root_input.resolve()
    with _backup_lock(root):
        snapshots_root = root / "snapshots"
        if not snapshots_root.is_dir():
            raise DisasterRecoveryError("backup contains no snapshots")
        all_snapshots: list[VerifiedSnapshot] = []
        retained_paths: set[Path] = set()
        run_directories: list[Path] = []
        for path in sorted(snapshots_root.iterdir()):
            if path.is_symlink() or not path.is_dir():
                raise DisasterRecoveryError(f"unexpected snapshots-root entry: {path}")
            run_directories.append(path)
        if not run_directories:
            raise DisasterRecoveryError("backup contains no run snapshots")
        global_latest = _resolve_latest(
            root / "latest.json",
            root,
            verify_payload=True,
        )
        retained_paths.add(global_latest.path)
        for run_directory in run_directories:
            headers = _snapshot_headers(run_directory, root)
            if not headers:
                raise DisasterRecoveryError(
                    f"run snapshot directory is empty: {run_directory}"
                )
            latest = _resolve_latest(
                run_directory / "latest.json",
                root,
                verify_payload=True,
            )
            verified_for_run = [
                (
                    latest
                    if header.path == latest.path
                    else _verify_snapshot_document(header.path, root)
                )
                for header in headers
            ]
            all_snapshots.extend(verified_for_run)
            retained_paths.add(latest.path)
            if retain_latest:
                retained_paths.update(
                    item.path for item in verified_for_run[-retain_latest:]
                )
            if retain_hourly:
                retained_paths.update(
                    _retain_bucketed(verified_for_run, retain_hourly, "hourly")
                )
            if retain_daily:
                retained_paths.update(
                    _retain_bucketed(verified_for_run, retain_daily, "daily")
                )
            if retain_monthly:
                retained_paths.update(
                    _retain_bucketed(verified_for_run, retain_monthly, "monthly")
                )

        if any(item.created_ns > captured_ns for item in all_snapshots):
            raise DisasterRecoveryError("snapshot timestamp is in the future")
        retained = [item for item in all_snapshots if item.path in retained_paths]
        pruned = [item for item in all_snapshots if item.path not in retained_paths]
        reachable = {
            entry.sha256 for snapshot in retained for entry in snapshot.catalog.values()
        }
        grace_ns = int(grace_seconds * 1_000_000_000)
        candidates: list[Path] = []
        candidate_bytes = 0
        for path in _all_objects(root):
            metadata = path.stat()
            if path.name not in reachable and metadata.st_mtime_ns < (
                captured_ns - grace_ns
            ):
                candidates.append(path)
                candidate_bytes += metadata.st_size
        if apply:
            for snapshot in pruned:
                snapshot.path.unlink()
                _snapshot_commit_path(snapshot.path).unlink(missing_ok=True)
            for directory in run_directories:
                _fsync_directory(directory)
            for path in candidates:
                path.unlink()
            for prefix in {path.parent for path in candidates}:
                _fsync_directory(prefix)
        return {
            "status": "ok",
            "mode": "apply" if apply else "dry-run",
            "snapshots": len(all_snapshots),
            "retained_snapshots": len(retained),
            "prunable_snapshots": len(pruned),
            "reachable_objects": len(reachable),
            "deletable_objects": len(candidates),
            "deletable_bytes": candidate_bytes,
            "deleted_snapshots": len(pruned) if apply else 0,
            "deleted_objects": len(candidates) if apply else 0,
            "deleted_bytes": candidate_bytes if apply else 0,
        }


gc_snapshots = garbage_collect
gc = garbage_collect


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser("snapshot", help="create and publish a snapshot")
    snapshot.add_argument("--run-root", required=True, type=Path)
    snapshot.add_argument("--profile", required=True, type=Path)
    snapshot.add_argument("--backup-root", required=True, type=Path)
    snapshot.add_argument(
        "--expected-backup-mount",
        type=Path,
        help="fail unless this exact mounted filesystem contains --backup-root",
    )
    snapshot.add_argument("--replay-backup-retain", type=int, default=3)
    snapshot.add_argument(
        "--allow-legacy-missing-initialized",
        action="store_true",
        help="mark a verified markerless source as legacy",
    )

    verify = commands.add_parser("verify", help="verify a snapshot")
    verify.add_argument("--snapshot", type=Path)
    verify.add_argument("--backup-root", type=Path)
    verify.add_argument("--run-id")

    restore_command = commands.add_parser("restore", help="restore a snapshot")
    restore_command.add_argument("--snapshot", required=True, type=Path)
    restore_command.add_argument("--backup-root", type=Path)
    restore_command.add_argument("--destination", required=True, type=Path)
    restore_command.add_argument("--recreate-initialized", action="store_true")
    restore_command.add_argument(
        "--relocate-profile",
        action="store_true",
        help="write profile-relocated.yaml with the destination run root",
    )

    rebind = commands.add_parser(
        "rebind-namespace",
        help="hand a backup namespace to a verified relocated restore",
    )
    rebind.add_argument("--backup-root", required=True, type=Path)
    rebind.add_argument("--restored-run-root", required=True, type=Path)

    gc_command = commands.add_parser("gc", help="collect old snapshots and objects")
    gc_command.add_argument("--backup-root", required=True, type=Path)
    gc_command.add_argument("--retain-latest", type=int, default=96)
    gc_command.add_argument("--retain-hourly", type=int, default=24)
    gc_command.add_argument("--retain-daily", type=int, default=30)
    gc_command.add_argument("--retain-monthly", type=int, default=12)
    gc_command.add_argument("--grace-seconds", type=float, default=7 * 86_400)
    gc_mode = gc_command.add_mutually_exclusive_group()
    gc_mode.add_argument("--dry-run", action="store_true")
    gc_mode.add_argument("--apply", action="store_true")
    return parser


def _verify_arguments(arguments: argparse.Namespace) -> Path:
    if arguments.snapshot is not None:
        if arguments.run_id is not None:
            raise DisasterRecoveryError("--run-id cannot be combined with --snapshot")
        return arguments.snapshot
    if arguments.backup_root is None:
        raise DisasterRecoveryError("verify requires --snapshot or --backup-root")
    if arguments.run_id is None:
        return arguments.backup_root / "latest.json"
    try:
        run_id = validate_identifier("run_id", arguments.run_id)
    except ValueError as exc:
        raise DisasterRecoveryError(str(exc)) from exc
    return arguments.backup_root / "snapshots" / run_id / "latest.json"


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command == "snapshot":
            path = create_snapshot(
                arguments.run_root,
                arguments.profile,
                arguments.backup_root,
                replay_backup_retain=arguments.replay_backup_retain,
                allow_legacy_missing_initialized=(
                    arguments.allow_legacy_missing_initialized
                ),
                expected_backup_mount=arguments.expected_backup_mount,
            )
            report = _verified_snapshot_report(
                _verify_snapshot_document(
                    path,
                    arguments.backup_root.expanduser().resolve(),
                    full_objects=False,
                )
            )
        elif arguments.command == "verify":
            snapshot_path = _verify_arguments(arguments)
            report = verify_snapshot(
                snapshot_path,
                backup_root=arguments.backup_root,
            )
        elif arguments.command == "restore":
            destination = restore_snapshot(
                arguments.snapshot,
                arguments.destination,
                backup_root=arguments.backup_root,
                recreate_initialized=arguments.recreate_initialized,
                relocate_profile=arguments.relocate_profile,
            )
            report = {"status": "ok", "destination": str(destination)}
        elif arguments.command == "rebind-namespace":
            report = rebind_backup_namespace(
                arguments.backup_root,
                arguments.restored_run_root,
            )
        else:
            report = garbage_collect(
                arguments.backup_root,
                retain_latest=arguments.retain_latest,
                retain_hourly=arguments.retain_hourly,
                retain_daily=arguments.retain_daily,
                retain_monthly=arguments.retain_monthly,
                grace_seconds=arguments.grace_seconds,
                apply=arguments.apply,
            )
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
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
