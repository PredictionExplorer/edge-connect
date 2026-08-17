#!/usr/bin/env python3
"""Pull and verify a StarTrain disaster-recovery snapshot on macOS."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import stat
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

SNAPSHOT_REPORT = "startrain-disaster-recovery-snapshot"
CONTROL_PLANE_REPORT = "startrain-control-plane-snapshot"
LATEST_REPORT = "startrain-disaster-recovery-latest"
NAMESPACE_REPORT = "startrain-disaster-recovery-namespace"
COMMIT_REPORT = "startrain-disaster-recovery-snapshot-commit"
SCHEMA_VERSION = 1
LATEST_RELATIVE_PATH = PurePosixPath("latest.json")
_LATEST_MAX_BYTES = 1024 * 1024
_MANIFEST_MAX_BYTES = 256 * 1024 * 1024
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_HOST_PATTERN = re.compile(
    r"(?:(?:[A-Za-z0-9_][A-Za-z0-9_.-]*)@)?"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?"
    r"|\[[0-9A-Fa-f:.%]+\])"
)
_REMOTE_PATH_PATTERN = re.compile(r"/[A-Za-z0-9._/-]*")
_SNAPSHOT_NAME_PATTERN = re.compile(
    r"(?P<created_ns>[1-9][0-9]*)-(?P<sha256>[0-9a-f]{64})\.json"
)

_REMOTE_READ_SCRIPT = r"""
import json
import os
import stat
import sys

request = json.load(sys.stdin)
root = os.path.realpath(request["root"])
relative = request["relative_path"]
maximum = request["max_bytes"]
if (
    not isinstance(relative, str)
    or not relative
    or relative.startswith("/")
    or "\\" in relative
):
    raise SystemExit("invalid relative path")
parts = relative.split("/")
if any(part in ("", ".", "..") for part in parts):
    raise SystemExit("invalid relative path")
if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
    raise SystemExit("invalid byte limit")
path = os.path.abspath(os.path.join(root, *parts))
if os.path.commonpath((root, path)) != root:
    raise SystemExit("path escapes backup root")
if os.path.realpath(path) != path:
    raise SystemExit("symlinked backup path refused")
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags)
with os.fdopen(descriptor, "rb") as stream:
    metadata = os.fstat(stream.fileno())
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("backup path is not a regular file")
    if metadata.st_size > maximum:
        raise SystemExit("backup metadata exceeds byte limit")
    payload = stream.read(maximum + 1)
if len(payload) > maximum:
    raise SystemExit("backup metadata exceeds byte limit")
sys.stdout.buffer.write(payload)
""".strip()

_REMOTE_ACK_SCRIPT = r"""
import base64
import json
import os
import secrets
import stat
import sys

request = json.load(sys.stdin)
root = os.path.realpath(request["root"])
relative = request["relative_path"]
if (
    not isinstance(relative, str)
    or not relative
    or relative.startswith("/")
    or "\\" in relative
):
    raise SystemExit("invalid acknowledgement path")
parts = relative.split("/")
if any(part in ("", ".", "..") for part in parts):
    raise SystemExit("invalid acknowledgement path")
current = root
for part in parts[:-1]:
    candidate = os.path.join(current, part)
    try:
        metadata = os.lstat(candidate)
    except FileNotFoundError:
        os.mkdir(candidate, mode=0o700)
        metadata = os.lstat(candidate)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit("unsafe acknowledgement parent")
    current = candidate
parent = current
target = os.path.join(parent, parts[-1])
payload = base64.b64decode(request["payload_base64"], validate=True)
document = json.loads(payload)
if not isinstance(document, dict):
    raise SystemExit("acknowledgement payload is not an object")
temporary = os.path.join(
    parent,
    "." + os.path.basename(target) + "." + secrets.token_hex(8) + ".partial",
)
try:
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    directory_descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
""".strip()

_REMOTE_STAT_SCRIPT = r"""
import json
import os
import stat
import sys

request = json.load(sys.stdin)
root = os.path.realpath(request["root"])
relative = request["relative_path"]
if (
    not isinstance(relative, str)
    or not relative
    or relative.startswith("/")
    or "\\" in relative
):
    raise SystemExit("invalid relative path")
parts = relative.split("/")
if any(part in ("", ".", "..") for part in parts):
    raise SystemExit("invalid relative path")
path = os.path.abspath(os.path.join(root, *parts))
if os.path.commonpath((root, path)) != root or os.path.realpath(path) != path:
    raise SystemExit("unsafe backup path")
descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("backup path is not a regular file")
    print(json.dumps({"bytes": metadata.st_size}, sort_keys=True))
finally:
    os.close(descriptor)
""".strip()

# OpenSSH sends its remote command through a shell. Only these fixed, locally
# quoted programs enter that command; every path and JSON document travels on
# stdin instead.
_REMOTE_READ_COMMAND = f"python3 -I -c {shlex.quote(_REMOTE_READ_SCRIPT)}"
_REMOTE_ACK_COMMAND = f"python3 -I -c {shlex.quote(_REMOTE_ACK_SCRIPT)}"
_REMOTE_STAT_COMMAND = f"python3 -I -c {shlex.quote(_REMOTE_STAT_SCRIPT)}"


class SnapshotPullError(RuntimeError):
    """Raised when a snapshot cannot be safely pulled or verified."""


class SnapshotValidationError(SnapshotPullError):
    """Raised when remote or local backup data fails validation."""


class SnapshotTransportError(SnapshotPullError):
    """Raised when OpenSSH or rsync cannot complete an operation."""


@dataclass(frozen=True)
class CatalogEntry:
    logical_path: str
    sha256: str
    bytes: int
    kind: str


@dataclass(frozen=True)
class SnapshotPointer:
    run_id: str
    generation_family: str
    relative_path: PurePosixPath
    sha256: str
    bytes: int
    created_ns: int


@dataclass(frozen=True)
class SnapshotManifest:
    report: str
    run_id: str
    generation_family: str
    created_ns: int
    source_run_root: str
    catalog: tuple[CatalogEntry, ...]


@dataclass(frozen=True)
class PullConfig:
    host: str
    remote_backup_root: PurePosixPath
    local_backup_root: Path
    known_hosts_file: Path
    identity_file: Path | None = None
    run_id: str | None = None
    ack_remote_path: PurePosixPath | None = None

    @classmethod
    def from_values(
        cls,
        *,
        host: str,
        remote_backup_root: str,
        local_backup_root: str | Path,
        known_hosts_file: str | Path,
        identity_file: str | Path | None = None,
        run_id: str | None = None,
        ack_remote_path: str | None = None,
    ) -> PullConfig:
        validated_host = _validate_host(host)
        remote_root = _validate_remote_absolute_path(
            remote_backup_root,
            label="remote backup root",
        )
        local_root = _validate_local_absolute_path(
            local_backup_root,
            label="local backup root",
        )
        known_hosts = _validate_local_absolute_path(
            known_hosts_file,
            label="known-hosts file",
        )
        identity = (
            None
            if identity_file is None
            else _validate_local_absolute_path(
                identity_file,
                label="identity file",
            )
        )
        selected_run_id = None if run_id is None else _validate_run_id(run_id)
        acknowledgement = (
            None
            if ack_remote_path is None
            else _validate_remote_absolute_path(
                ack_remote_path,
                label="acknowledgement path",
            )
        )
        if acknowledgement is not None:
            try:
                acknowledgement.relative_to(remote_root)
            except ValueError as exc:
                raise ValueError(
                    "acknowledgement path must be under the remote backup root"
                ) from exc
            if acknowledgement == remote_root:
                raise ValueError("acknowledgement path must name a file")
        return cls(
            host=validated_host,
            remote_backup_root=remote_root,
            local_backup_root=local_root,
            known_hosts_file=known_hosts,
            identity_file=identity,
            run_id=selected_run_id,
            ack_remote_path=acknowledgement,
        )


@dataclass(frozen=True)
class PullResult:
    run_id: str
    snapshot_path: str
    snapshot_sha256: str
    object_count: int
    transferred_objects: int
    acknowledged: bool


class SnapshotTransport(Protocol):
    def read_file(
        self,
        relative_path: PurePosixPath,
        *,
        max_bytes: int,
    ) -> bytes:
        """Read bounded backup metadata without changing the remote."""
        ...

    def fetch_file(
        self,
        relative_path: PurePosixPath,
        local_partial_path: Path,
        *,
        expected_bytes: int,
    ) -> None:
        """Fetch one immutable file into a new local partial path."""
        ...

    def write_acknowledgement(
        self,
        relative_path: PurePosixPath,
        payload: bytes,
    ) -> None:
        """Publish an acknowledgement atomically on the remote."""
        ...


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


class OpenSshTransport:
    """OpenSSH/rsync transport with pinned host-key verification."""

    def __init__(
        self,
        config: PullConfig,
        *,
        runner: Runner = subprocess.run,
    ) -> None:
        self._host = config.host
        self._remote_root = config.remote_backup_root
        self._known_hosts_file = config.known_hosts_file
        self._identity_file = config.identity_file
        self._runner = runner
        self._control_path = self._known_hosts_file.parent / ".edgeconnect-dr-%C"

    def _common_options(self) -> list[str]:
        options = [
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            _ssh_path_option(
                "UserKnownHostsFile",
                self._known_hosts_file,
            ),
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            "CheckHostIP=yes",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "RequestTTY=no",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=120",
            "-o",
            _ssh_path_option("ControlPath", self._control_path),
        ]
        if self._identity_file is not None:
            options.extend(
                [
                    "-o",
                    "IdentitiesOnly=yes",
                    "-i",
                    str(self._identity_file),
                ]
            )
        return options

    def _ssh_command(self, remote_command: str) -> list[str]:
        return [
            "ssh",
            "-T",
            *self._common_options(),
            self._host,
            remote_command,
        ]

    def read_file(
        self,
        relative_path: PurePosixPath,
        *,
        max_bytes: int,
    ) -> bytes:
        relative = _validate_transport_relative_path(relative_path)
        request = _json_bytes(
            {
                "root": str(self._remote_root),
                "relative_path": str(relative),
                "max_bytes": max_bytes,
            }
        )
        result = self._invoke(
            self._ssh_command(_REMOTE_READ_COMMAND),
            input=request,
            timeout=60,
        )
        if len(result.stdout) > max_bytes:
            raise SnapshotTransportError(
                f"remote metadata {relative} exceeded {max_bytes} bytes"
            )
        return result.stdout

    def fetch_file(
        self,
        relative_path: PurePosixPath,
        local_partial_path: Path,
        *,
        expected_bytes: int,
    ) -> None:
        relative = _validate_transport_relative_path(relative_path)
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise ValueError("expected_bytes must be a non-negative integer")
        stat_request = _json_bytes(
            {
                "root": str(self._remote_root),
                "relative_path": str(relative),
            }
        )
        stat_result = self._invoke(
            self._ssh_command(_REMOTE_STAT_COMMAND),
            input=stat_request,
            timeout=60,
        )
        remote_stat = _load_json_object(stat_result.stdout, label="remote object stat")
        if remote_stat != {"bytes": expected_bytes}:
            raise SnapshotTransportError(
                f"remote object {relative} size differs from snapshot catalog"
            )
        required_free = expected_bytes + 1024 * 1024 * 1024
        if shutil.disk_usage(local_partial_path.parent).free < required_free:
            raise SnapshotTransportError(
                f"insufficient local space for {relative}; need {required_free} bytes"
            )
        remote_path = self._remote_root.joinpath(relative)
        remote_shell = shlex.join(["ssh", *self._common_options()])
        command = [
            "rsync",
            "-a",
            "--partial",
            f"--max-size={expected_bytes}",
            "-e",
            remote_shell,
            f"{self._host}:{remote_path}",
            str(local_partial_path),
        ]
        timeout = max(300, min(21_600, expected_bytes // (1024 * 1024) + 120))
        self._invoke(command, timeout=timeout)

    def write_acknowledgement(
        self,
        relative_path: PurePosixPath,
        payload: bytes,
    ) -> None:
        relative = _validate_transport_relative_path(relative_path)
        request = _json_bytes(
            {
                "root": str(self._remote_root),
                "relative_path": str(relative),
                "payload_base64": base64.b64encode(payload).decode("ascii"),
            }
        )
        self._invoke(
            self._ssh_command(_REMOTE_ACK_COMMAND),
            input=request,
            timeout=60,
        )

    def _invoke(
        self,
        command: list[str],
        *,
        input: bytes | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            result = self._runner(
                command,
                input=input,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SnapshotTransportError(
                f"{command[0]} transport failed: {exc}"
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            if len(detail) > 500:
                detail = f"{detail[:500]}..."
            suffix = f": {detail}" if detail else ""
            raise SnapshotTransportError(
                f"{command[0]} exited with status {result.returncode}{suffix}"
            )
        return result


def _ssh_path_option(name: str, path: Path) -> str:
    value = str(path)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} path contains a control character")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{name}="{escaped}"'


def _validate_host(host: str) -> str:
    if not isinstance(host, str) or _HOST_PATTERN.fullmatch(host) is None:
        raise ValueError("host must be a user/hostname or user/IP value")
    if host.startswith("-"):
        raise ValueError("host cannot begin with '-'")
    return host


def _validate_remote_absolute_path(
    value: str,
    *,
    label: str,
) -> PurePosixPath:
    if not isinstance(value, str) or _REMOTE_PATH_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be an absolute POSIX path containing only "
            "letters, digits, '.', '_', '-', and '/'"
        )
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be absolute and cannot contain '..'")
    return path


def _validate_local_absolute_path(
    value: str | Path,
    *,
    label: str,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if any(ord(character) < 32 or ord(character) == 127 for character in str(path)):
        raise ValueError(f"{label} contains a control character")
    return path.resolve(strict=False)


def _validate_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or _RUN_ID_PATTERN.fullmatch(run_id) is None
        or run_id in {".", ".."}
    ):
        raise ValueError(
            "run ID must contain 1-128 safe letters, digits, '.', '_', or '-'"
        )
    return run_id


def _validate_transport_relative_path(
    value: PurePosixPath,
) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in str(path)
    ):
        raise SnapshotValidationError("transport path is not a safe relative path")
    return path


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotValidationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        decoded = payload.decode("utf-8")
        document = json.loads(
            decoded,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise SnapshotValidationError(f"{label} must be a JSON object")
    return document


def _schema_version(document: Mapping[str, object], *, label: str) -> None:
    value = document.get("schema_version")
    if isinstance(value, bool) or value != SCHEMA_VERSION:
        raise SnapshotValidationError(
            f"{label} must use schema_version {SCHEMA_VERSION}"
        )


def _one_alias(
    mappings: tuple[Mapping[str, object], ...],
    names: tuple[str, ...],
    *,
    label: str,
) -> object:
    values: list[object] = []
    for mapping in mappings:
        values.extend(mapping[name] for name in names if name in mapping)
    if not values:
        raise SnapshotValidationError(f"latest pointer is missing {label}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise SnapshotValidationError(f"latest pointer has conflicting {label} fields")
    return first


def _parse_pointer(
    payload: bytes,
    *,
    expected_run_id: str | None,
) -> SnapshotPointer:
    document = _load_json_object(payload, label="latest pointer")
    _schema_version(document, label="latest pointer")
    expected_fields = {
        "report",
        "schema_version",
        "run_id",
        "generation_family",
        "path",
        "sha256",
        "bytes",
        "created_ns",
    }
    if set(document) != expected_fields:
        raise SnapshotValidationError("latest pointer fields are incompatible")
    if document.get("report") != LATEST_REPORT:
        raise SnapshotValidationError(
            f"latest pointer report must be {LATEST_REPORT!r}"
        )
    if payload != _json_bytes(document):
        raise SnapshotValidationError("latest pointer is not canonical JSON")
    run_id_value = document.get("run_id")
    if not isinstance(run_id_value, str):
        raise SnapshotValidationError("latest pointer run_id is missing")
    try:
        run_id = _validate_run_id(run_id_value)
    except ValueError as exc:
        raise SnapshotValidationError(str(exc)) from exc
    if expected_run_id is not None and run_id != expected_run_id:
        raise SnapshotValidationError(
            f"latest snapshot belongs to run {run_id!r}, "
            f"not requested run {expected_run_id!r}"
        )
    family_value = document.get("generation_family")
    if not isinstance(family_value, str):
        raise SnapshotValidationError("latest pointer generation_family is missing")
    try:
        generation_family = _validate_run_id(family_value)
    except ValueError as exc:
        raise SnapshotValidationError(f"invalid generation family: {exc}") from exc
    path_value = document.get("path")
    if not isinstance(path_value, str):
        raise SnapshotValidationError("snapshot path must be a string")
    filename = PurePosixPath(path_value)
    if (
        filename.is_absolute()
        or filename.name != path_value
        or len(filename.parts) != 1
    ):
        raise SnapshotValidationError(
            "latest pointer path must be an immutable snapshot filename"
        )
    relative_path = PurePosixPath("snapshots", run_id, path_value)
    digest_value = document.get("sha256")
    if (
        not isinstance(digest_value, str)
        or _DIGEST_PATTERN.fullmatch(digest_value) is None
    ):
        raise SnapshotValidationError(
            "snapshot SHA-256 must be 64 lowercase hexadecimal characters"
        )
    name_match = _SNAPSHOT_NAME_PATTERN.fullmatch(relative_path.name)
    if name_match is None:
        raise SnapshotValidationError("snapshot manifest name is not immutable")
    if name_match.group("sha256") != digest_value:
        raise SnapshotValidationError(
            "snapshot manifest filename does not match pointer SHA-256"
        )
    byte_value = document.get("bytes")
    if (
        isinstance(byte_value, bool)
        or not isinstance(byte_value, int)
        or byte_value <= 0
    ):
        raise SnapshotValidationError("latest pointer bytes must be a positive integer")
    created_value = document.get("created_ns")
    if (
        isinstance(created_value, bool)
        or not isinstance(created_value, int)
        or created_value <= 0
    ):
        raise SnapshotValidationError(
            "latest pointer created_ns must be a positive integer"
        )
    return SnapshotPointer(
        run_id=run_id,
        generation_family=generation_family,
        relative_path=relative_path,
        sha256=digest_value,
        bytes=byte_value,
        created_ns=created_value,
    )


def _parse_namespace(
    payload: bytes,
    *,
    pointer: SnapshotPointer,
) -> dict[str, object]:
    document = _load_json_object(payload, label="backup namespace")
    _schema_version(document, label="backup namespace")
    if payload != _json_bytes(document):
        raise SnapshotValidationError("backup namespace is not canonical JSON")
    if set(document) != {
        "report",
        "schema_version",
        "run_id",
        "generation_family",
        "source_run_root",
    }:
        raise SnapshotValidationError("backup namespace fields are incompatible")
    source_root = document.get("source_run_root")
    if (
        document.get("report") != NAMESPACE_REPORT
        or document.get("run_id") != pointer.run_id
        or document.get("generation_family") != pointer.generation_family
        or not isinstance(source_root, str)
        or not PurePosixPath(source_root).is_absolute()
    ):
        raise SnapshotValidationError("backup namespace identity is invalid")
    return document


def _parse_manifest(
    payload: bytes,
    *,
    pointer: SnapshotPointer,
) -> SnapshotManifest:
    if _sha256_bytes(payload) != pointer.sha256:
        raise SnapshotValidationError(
            "snapshot manifest SHA-256 does not match latest pointer"
        )
    if len(payload) != pointer.bytes:
        raise SnapshotValidationError(
            "snapshot manifest byte count does not match latest pointer"
        )
    document = _load_json_object(payload, label="snapshot manifest")
    _schema_version(document, label="snapshot manifest")
    if payload != _json_bytes(document):
        raise SnapshotValidationError("snapshot manifest is not canonical JSON")
    report = document.get("report")
    if report not in {SNAPSHOT_REPORT, CONTROL_PLANE_REPORT}:
        raise SnapshotValidationError("snapshot manifest report is unsupported")
    run_id = document.get("run_id")
    if run_id != pointer.run_id:
        raise SnapshotValidationError(
            "snapshot manifest run_id does not match latest pointer"
        )
    generation_family = document.get("generation_family")
    if generation_family != pointer.generation_family:
        raise SnapshotValidationError(
            "snapshot manifest generation_family does not match latest pointer"
        )
    created_ns = document.get("created_ns")
    if (
        isinstance(created_ns, bool)
        or not isinstance(created_ns, int)
        or created_ns <= 0
    ):
        raise SnapshotValidationError(
            "snapshot manifest created_ns must be a positive integer"
        )
    if created_ns != pointer.created_ns:
        raise SnapshotValidationError(
            "snapshot manifest created_ns does not match latest pointer"
        )
    name_match = _SNAPSHOT_NAME_PATTERN.fullmatch(pointer.relative_path.name)
    if name_match is None or int(name_match.group("created_ns")) != created_ns:
        raise SnapshotValidationError(
            "snapshot manifest timestamp does not match its immutable filename"
        )
    source = document.get("source")
    source_key = "run_root" if report == SNAPSHOT_REPORT else "source_root"
    source_run_root = source.get(source_key) if isinstance(source, dict) else None
    if (
        not isinstance(source_run_root, str)
        or not PurePosixPath(source_run_root).is_absolute()
    ):
        raise SnapshotValidationError("snapshot source root is invalid")
    catalog_value = document.get("catalog")
    raw_entries: list[tuple[str | None, object]]
    if isinstance(catalog_value, list):
        raw_entries = [(None, value) for value in catalog_value]
    elif isinstance(catalog_value, dict):
        raw_entries = list(catalog_value.items())
    else:
        raise SnapshotValidationError(
            "snapshot manifest catalog must be a list or mapping"
        )
    if not raw_entries:
        raise SnapshotValidationError("snapshot manifest catalog cannot be empty")
    entries: list[CatalogEntry] = []
    logical_paths: set[str] = set()
    digest_sizes: dict[str, int] = {}
    for catalog_key, raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise SnapshotValidationError("catalog entry must be an object")
        logical_value = raw_entry.get("logical_path", catalog_key)
        if catalog_key is not None and logical_value != catalog_key:
            raise SnapshotValidationError(
                "mapped catalog key must equal entry logical_path"
            )
        logical_path = _validate_logical_path(logical_value)
        if logical_path in logical_paths:
            raise SnapshotValidationError(
                f"duplicate catalog logical path {logical_path!r}"
            )
        logical_paths.add(logical_path)
        digest = raw_entry.get("sha256")
        if not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None:
            raise SnapshotValidationError(
                f"catalog entry {logical_path!r} has an invalid SHA-256"
            )
        byte_count = raw_entry.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise SnapshotValidationError(
                f"catalog entry {logical_path!r} has an invalid byte count"
            )
        previous_size = digest_sizes.setdefault(digest, byte_count)
        if previous_size != byte_count:
            raise SnapshotValidationError(
                f"catalog digest {digest} has conflicting byte counts"
            )
        kind = raw_entry.get("kind")
        if (
            not isinstance(kind, str)
            or not kind
            or len(kind) > 128
            or any(ord(character) < 32 for character in kind)
        ):
            raise SnapshotValidationError(
                f"catalog entry {logical_path!r} has an invalid kind"
            )
        entries.append(
            CatalogEntry(
                logical_path=logical_path,
                sha256=digest,
                bytes=byte_count,
                kind=kind,
            )
        )
    return SnapshotManifest(
        report=str(report),
        run_id=pointer.run_id,
        generation_family=pointer.generation_family,
        created_ns=created_ns,
        source_run_root=source_run_root,
        catalog=tuple(entries),
    )


def _validate_logical_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SnapshotValidationError("catalog logical_path must be a POSIX string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 for character in value)
    ):
        raise SnapshotValidationError(f"catalog logical_path {value!r} is unsafe")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_open_file(path: Path) -> tuple[str, int]:
    try:
        path_metadata = path.lstat()
    except OSError as exc:
        raise SnapshotValidationError(
            f"cannot inspect local file {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(path_metadata.st_mode):
        raise SnapshotValidationError(
            f"local backup path is not a regular file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SnapshotValidationError(f"cannot open local file {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SnapshotValidationError(
                f"local backup path is not a regular file: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), metadata.st_size
    finally:
        os.close(descriptor)


def _verify_local_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    digest, byte_count = _hash_open_file(path)
    if byte_count != expected_bytes:
        raise SnapshotValidationError(
            f"local file {path} has {byte_count} bytes; expected {expected_bytes}"
        )
    if digest != expected_sha256:
        raise SnapshotValidationError(f"local file {path} failed SHA-256 verification")


def _ensure_directory(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                metadata = current.lstat()
            else:
                metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SnapshotValidationError(
                f"local backup directory is unsafe: {current}"
            )
    return current


def _prepare_local_root(root: Path) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = root.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SnapshotValidationError(f"local backup root is not a directory: {root}")


def _partial_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.partial")


def _materialize_file(
    transport: SnapshotTransport,
    *,
    remote_relative_path: PurePosixPath,
    local_root: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> bool:
    parent_relative = PurePosixPath(*remote_relative_path.parts[:-1])
    parent = _ensure_directory(local_root, parent_relative)
    destination = parent / remote_relative_path.name
    if os.path.lexists(destination):
        _verify_local_file(
            destination,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        return False
    temporary = _partial_path(destination)
    try:
        transport.fetch_file(
            remote_relative_path,
            temporary,
            expected_bytes=expected_bytes,
        )
        _verify_local_file(
            temporary,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        _fsync_file(temporary)
        if os.path.lexists(destination):
            _verify_local_file(
                destination,
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
            )
            return False
        os.replace(temporary, destination)
        _fsync_directory(parent)
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_file(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = _partial_path(path)
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_local_immutable(
    path: Path,
    payload: bytes,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _verify_local_file(
            path,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        return
    _atomic_write(path, payload)
    _verify_local_file(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )


def _update_local_namespace(root: Path, payload: bytes) -> None:
    target = root / "namespace.json"
    if not target.exists():
        _atomic_write(target, payload)
        return
    if target.is_symlink():
        raise SnapshotValidationError("local backup namespace is a symbolic link")
    current = target.read_bytes()
    if current == payload:
        return
    old = _load_json_object(current, label="local backup namespace")
    new = _load_json_object(payload, label="new backup namespace")
    if (
        old.get("report") != NAMESPACE_REPORT
        or new.get("report") != NAMESPACE_REPORT
        or old.get("schema_version") != SCHEMA_VERSION
        or new.get("schema_version") != SCHEMA_VERSION
        or old.get("run_id") != new.get("run_id")
        or old.get("generation_family") != new.get("generation_family")
    ):
        raise SnapshotValidationError(
            "local backup namespace belongs to another workload"
        )
    history = root / "namespace-history"
    _ensure_directory(root, PurePosixPath("namespace-history"))
    archived = history / f"{_sha256_bytes(current)}.json"
    _publish_local_immutable(
        archived,
        current,
        expected_sha256=_sha256_bytes(current),
        expected_bytes=len(current),
    )
    _atomic_write(target, payload)


def _object_relative_path(digest: str) -> PurePosixPath:
    return PurePosixPath("objects", "sha256", digest[:2], digest)


def _snapshot_commit_payload(pointer: SnapshotPointer) -> bytes:
    return _json_bytes(
        {
            "report": COMMIT_REPORT,
            "schema_version": SCHEMA_VERSION,
            "path": pointer.relative_path.name,
            "sha256": pointer.sha256,
            "bytes": pointer.bytes,
            "created_ns": pointer.created_ns,
            "run_id": pointer.run_id,
            "generation_family": pointer.generation_family,
        }
    )


def pull_snapshot(
    config: PullConfig,
    *,
    transport: SnapshotTransport | None = None,
    completed_ns: int | None = None,
    mac_hostname: str | None = None,
) -> PullResult:
    """Pull one immutable snapshot and publish the local pointer last."""
    active_transport = OpenSshTransport(config) if transport is None else transport
    _prepare_local_root(config.local_backup_root)

    latest_relative_path = (
        LATEST_RELATIVE_PATH
        if config.run_id is None
        else PurePosixPath(
            "snapshots",
            config.run_id,
            LATEST_RELATIVE_PATH.name,
        )
    )
    latest_payload = active_transport.read_file(
        latest_relative_path,
        max_bytes=_LATEST_MAX_BYTES,
    )
    pointer = _parse_pointer(
        latest_payload,
        expected_run_id=config.run_id,
    )
    local_latest_path = config.local_backup_root.joinpath(*latest_relative_path.parts)
    if local_latest_path.is_file():
        local_pointer = _parse_pointer(
            local_latest_path.read_bytes(),
            expected_run_id=pointer.run_id,
        )
        if local_pointer.created_ns > pointer.created_ns:
            raise SnapshotValidationError(
                "remote latest pointer is older than the committed local mirror"
            )
        if (
            local_pointer.created_ns == pointer.created_ns
            and local_pointer.sha256 != pointer.sha256
        ):
            raise SnapshotValidationError(
                "remote latest pointer conflicts with local snapshot history"
            )
    namespace_payload = active_transport.read_file(
        PurePosixPath("namespace.json"),
        max_bytes=_LATEST_MAX_BYTES,
    )
    namespace = _parse_namespace(namespace_payload, pointer=pointer)
    manifest_payload = active_transport.read_file(
        pointer.relative_path,
        max_bytes=_MANIFEST_MAX_BYTES,
    )
    manifest = _parse_manifest(manifest_payload, pointer=pointer)
    if namespace.get("source_run_root") != manifest.source_run_root:
        raise SnapshotValidationError(
            "backup namespace source does not match latest snapshot"
        )

    unique_objects: dict[str, int] = {}
    for entry in manifest.catalog:
        unique_objects.setdefault(entry.sha256, entry.bytes)
    transferred_objects = 0
    for digest, byte_count in sorted(unique_objects.items()):
        if _materialize_file(
            active_transport,
            remote_relative_path=_object_relative_path(digest),
            local_root=config.local_backup_root,
            expected_sha256=digest,
            expected_bytes=byte_count,
        ):
            transferred_objects += 1

    # Re-read every local artifact after transfer. The mutable pointer is not
    # exposed until this complete verification pass succeeds.
    for digest, byte_count in sorted(unique_objects.items()):
        relative = _object_relative_path(digest)
        _verify_local_file(
            config.local_backup_root.joinpath(*relative.parts),
            expected_sha256=digest,
            expected_bytes=byte_count,
        )
    local_manifest = config.local_backup_root.joinpath(*pointer.relative_path.parts)
    _publish_local_immutable(
        local_manifest,
        manifest_payload,
        expected_sha256=pointer.sha256,
        expected_bytes=pointer.bytes,
    )
    _update_local_namespace(config.local_backup_root, namespace_payload)
    _atomic_write(
        local_latest_path,
        latest_payload,
    )
    commit_payload = _snapshot_commit_payload(pointer)
    _publish_local_immutable(
        local_manifest.with_name(f"{local_manifest.name}.commit"),
        commit_payload,
        expected_sha256=_sha256_bytes(commit_payload),
        expected_bytes=len(commit_payload),
    )

    acknowledged = False
    if config.ack_remote_path is not None:
        timestamp = time.time_ns() if completed_ns is None else completed_ns
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp <= 0
        ):
            raise ValueError("completed_ns must be a positive integer")
        hostname = socket.gethostname() if mac_hostname is None else mac_hostname
        if (
            not isinstance(hostname, str)
            or not hostname
            or any(ord(character) < 32 for character in hostname)
        ):
            raise ValueError("Mac hostname must be a non-empty safe string")
        acknowledgement_payload = _json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "report": "startrain-disaster-recovery-acknowledgement",
                "snapshot_sha256": pointer.sha256,
                "snapshot_path": str(pointer.relative_path),
                "completed_ns": timestamp,
                "mac_hostname": hostname,
                "local_verification_status": "verified",
            }
        )
        acknowledgement_relative = config.ack_remote_path.relative_to(
            config.remote_backup_root
        )
        active_transport.write_acknowledgement(
            acknowledgement_relative,
            acknowledgement_payload,
        )
        acknowledged = True

    return PullResult(
        run_id=manifest.run_id,
        snapshot_path=str(pointer.relative_path),
        snapshot_sha256=pointer.sha256,
        object_count=len(unique_objects),
        transferred_objects=transferred_objects,
        acknowledged=acknowledged,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-backup-root", required=True)
    parser.add_argument("--local-backup-root", required=True)
    parser.add_argument("--known-hosts-file", required=True)
    parser.add_argument("--identity-file")
    parser.add_argument("--run-id")
    parser.add_argument("--ack-remote-path")
    return parser


def main() -> int:
    parser = _build_parser()
    arguments = parser.parse_args()
    try:
        config = PullConfig.from_values(
            host=arguments.host,
            remote_backup_root=arguments.remote_backup_root,
            local_backup_root=arguments.local_backup_root,
            known_hosts_file=arguments.known_hosts_file,
            identity_file=arguments.identity_file,
            run_id=arguments.run_id,
            ack_remote_path=arguments.ack_remote_path,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not config.known_hosts_file.is_file():
        parser.error(
            f"--known-hosts-file is not a regular file: {config.known_hosts_file}"
        )
    if config.identity_file is not None and not config.identity_file.is_file():
        parser.error(f"--identity-file is not a regular file: {config.identity_file}")
    try:
        result = pull_snapshot(config)
    except SnapshotPullError as exc:
        parser.exit(1, f"snapshot pull failed: {exc}\n")
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "snapshot_path": result.snapshot_path,
                "snapshot_sha256": result.snapshot_sha256,
                "object_count": result.object_count,
                "transferred_objects": result.transferred_objects,
                "acknowledged": result.acknowledged,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
