"""Host-level training continuity and verified last-known-good fallback.

The coordinator owns recovery inside one run.  This module owns the narrower
host-level policy boundary: exactly one GPU workload may execute, failed runs
are quarantined by metadata only, and a verified LKG workload may replace a
failed primary when current hardware health is known-good.

The continuity manifest and mutable state intentionally live outside every run
root.  Run roots remain evidence and are never renamed, deleted, or edited by
this module.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from startrain.runtime import atomic_json, validate_identifier

MANIFEST_FORMAT = "startrain.training-continuity-manifest"
STATE_FORMAT = "startrain.training-continuity-state"
QUARANTINE_FORMAT = "startrain.training-continuity-quarantine"
ALERT_FORMAT = "startrain.training-continuity-alert"
MANIFEST_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNIT = re.compile(r"^[A-Za-z0-9@_.:-]+\.service$")
_ACTIVE_STATES = frozenset({"active", "activating", "reloading"})


class ContinuityError(RuntimeError):
    """Base class for fail-closed continuity errors."""


class ContinuityManifestError(ContinuityError):
    """The pinned host manifest or a workload identity is invalid."""


class ContinuityStateError(ContinuityError):
    """Mutable host continuity state is corrupt or belongs to another manifest."""


class ContinuityBusyError(ContinuityError):
    """A nonblocking host lock is held by another process."""


class ContinuitySplitBrainError(ContinuityError):
    """More than one known GPU workload appears live."""


@dataclass(frozen=True, slots=True)
class Workload:
    workload_id: str
    role: str
    unit: str
    profile_path: Path
    profile_sha256: str
    run_root: Path
    run_root_sha256: str
    runtime_manifest_path: Path | None
    runtime_manifest_sha256: str | None
    runtime_training_dir: Path | None
    runtime_orchestrator_path: Path | None
    runtime_orchestrator_sha256: str | None
    runtime_unit_path: Path | None
    runtime_unit_sha256: str | None
    lkg_verified_ns: int | None
    lkg_priority: int

    @property
    def is_lkg(self) -> bool:
        return self.lkg_verified_ns is not None

    def state_identity(self) -> dict[str, object]:
        return {
            "id": self.workload_id,
            "role": self.role,
            "unit": self.unit,
            "profile_path": str(self.profile_path),
            "profile_sha256": self.profile_sha256,
            "run_root": str(self.run_root),
            "run_root_sha256": self.run_root_sha256,
            "runtime_manifest_path": (
                str(self.runtime_manifest_path)
                if self.runtime_manifest_path is not None
                else None
            ),
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "runtime_training_dir": (
                str(self.runtime_training_dir)
                if self.runtime_training_dir is not None
                else None
            ),
            "runtime_orchestrator_path": (
                str(self.runtime_orchestrator_path)
                if self.runtime_orchestrator_path is not None
                else None
            ),
            "runtime_orchestrator_sha256": self.runtime_orchestrator_sha256,
            "runtime_unit_path": (
                str(self.runtime_unit_path)
                if self.runtime_unit_path is not None
                else None
            ),
            "runtime_unit_sha256": self.runtime_unit_sha256,
            "lkg_verified_ns": self.lkg_verified_ns,
        }


@dataclass(frozen=True, slots=True)
class FailureArtifact:
    artifact_type: str
    path: Path


@dataclass(frozen=True, slots=True)
class ContinuityManifest:
    path: Path
    sha256: str
    raw: dict[str, object]
    state_root: Path
    transition_lock_path: Path
    execution_lock_path: Path
    hardware_report_path: Path
    hardware_max_age_seconds: float
    hardware_probe_workload: str
    primary_id: str
    workloads: tuple[Workload, ...]
    failure_artifacts: tuple[FailureArtifact, ...]
    automatic_start: bool
    maximum_start_attempts: int
    start_retry_seconds: float
    operator_hold_path: Path | None
    alert_command: tuple[str, ...] | None
    alert_timeout_seconds: float

    @property
    def state_path(self) -> Path:
        return self.state_root / "continuity-state.json"

    @property
    def pinned_manifest_path(self) -> Path:
        return self.state_root / "continuity-manifest.json"

    @property
    def alerts_root(self) -> Path:
        return self.state_root / "alerts"

    @property
    def quarantine_root(self) -> Path:
        return self.state_root / "quarantine"

    def workload(self, workload_id: str) -> Workload:
        for workload in self.workloads:
            if workload.workload_id == workload_id:
                return workload
        raise ContinuityManifestError(f"unknown workload {workload_id!r}")


@dataclass(frozen=True, slots=True)
class WorkloadVerification:
    workload: Workload
    run_id: str
    generation_family: str
    created_ns: int
    run_identity_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            **self.workload.state_identity(),
            "run_id": self.run_id,
            "generation_family": self.generation_family,
            "created_ns": self.created_ns,
            "run_identity_sha256": self.run_identity_sha256,
            "verified": True,
        }


@dataclass(frozen=True, slots=True)
class HardwareAssessment:
    status: str
    safe: bool
    report_path: str
    report_sha256: str | None
    captured_ns: int | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "safe": self.safe,
            "report_path": self.report_path,
            "report_sha256": self.report_sha256,
            "captured_ns": self.captured_ns,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class FailureObservation:
    failure_id: str
    domain: str
    reason: str
    source_type: str
    source_path: str
    source_sha256: str
    occurred_ns: int | None
    terminal_reason: str | None
    evidence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "failure_id": self.failure_id,
            "domain": self.domain,
            "reason": self.reason,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "occurred_ns": self.occurred_ns,
            "terminal_reason": self.terminal_reason,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class UnitStatus:
    unit: str
    active_state: str
    sub_state: str | None = None
    main_pid: int = 0
    result: str | None = None
    exec_main_status: int | None = None
    query_error: str | None = None

    @property
    def active(self) -> bool:
        return self.active_state in _ACTIVE_STATES

    def as_dict(self) -> dict[str, object]:
        return {
            "unit": self.unit,
            "active_state": self.active_state,
            "sub_state": self.sub_state,
            "main_pid": self.main_pid,
            "result": self.result,
            "exec_main_status": self.exec_main_status,
            "query_error": self.query_error,
        }


class UnitManager(Protocol):
    def status(self, unit: str) -> UnitStatus: ...

    def start(self, unit: str) -> None: ...

    def stop(self, unit: str) -> None: ...


class SystemdUnitManager:
    """Small systemctl adapter; all arguments are passed without a shell."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout: float = 15.0,
    ) -> None:
        self.runner = runner
        self.timeout = timeout

    def status(self, unit: str) -> UnitStatus:
        try:
            completed = self.runner(
                [
                    "systemctl",
                    "show",
                    unit,
                    "--property=ActiveState",
                    "--property=SubState",
                    "--property=MainPID",
                    "--property=Result",
                    "--property=ExecMainStatus",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return UnitStatus(
                unit, "unknown", query_error=f"{type(exc).__name__}: {exc}"
            )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit status {completed.returncode}"
            return UnitStatus(unit, "unknown", query_error=detail)
        values: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        try:
            main_pid = int(values.get("MainPID", "0") or "0")
            exec_status = int(values.get("ExecMainStatus", "0") or "0")
        except ValueError:
            return UnitStatus(
                unit, "unknown", query_error="systemctl returned bad integers"
            )
        return UnitStatus(
            unit=unit,
            active_state=values.get("ActiveState", "unknown"),
            sub_state=values.get("SubState"),
            main_pid=main_pid,
            result=values.get("Result"),
            exec_main_status=exec_status,
        )

    def start(self, unit: str) -> None:
        try:
            completed = self.runner(
                ["systemctl", "start", "--no-block", unit],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContinuityError(f"cannot start {unit}: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit status {completed.returncode}"
            raise ContinuityError(f"cannot start {unit}: {detail}")

    def stop(self, unit: str) -> None:
        try:
            completed = self.runner(
                ["systemctl", "stop", "--no-block", unit],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContinuityError(f"cannot stop {unit}: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit status {completed.returncode}"
            raise ContinuityError(f"cannot stop {unit}: {detail}")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_bytes(path: Path, *, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ContinuityManifestError(f"cannot inspect {name} {path}: {exc}") from exc
    if path.is_symlink() or not path.is_file():
        raise ContinuityManifestError(f"{name} is missing or unsafe: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContinuityManifestError(f"cannot open {name} {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    opened_identity = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    )
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity != opened_identity or opened_identity != after_identity:
        raise ContinuityManifestError(f"{name} changed while it was being verified")
    return b"".join(chunks)


def _json_object_bytes(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    data = _read_regular_bytes(path, name=name)
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuityManifestError(f"cannot parse {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContinuityManifestError(f"{name} must be a JSON object")
    return payload, data


def _absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContinuityManifestError(f"{name} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ContinuityManifestError(f"{name} must be an absolute path")
    resolved = path.resolve(strict=False)
    if path != resolved:
        raise ContinuityManifestError(f"{name} must be canonical (no symlink aliases)")
    return path


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContinuityManifestError(f"{name} must be an object")
    return value


def _sequence(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContinuityManifestError(f"{name} must be an array")
    return value


def _strict_keys(
    payload: Mapping[str, object],
    *,
    name: str,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    missing = required - set(payload)
    unknown = set(payload) - required - optional
    if missing:
        raise ContinuityManifestError(f"{name} is missing fields: {sorted(missing)}")
    if unknown:
        raise ContinuityManifestError(f"{name} has unknown fields: {sorted(unknown)}")


def _sha256_value(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContinuityManifestError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContinuityManifestError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContinuityManifestError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContinuityManifestError(f"{name} must be positive")
    result = float(value)
    if result <= 0 or not (result < float("inf")):
        raise ContinuityManifestError(f"{name} must be finite and positive")
    return result


def _identifier(value: object, *, name: str) -> str:
    try:
        return validate_identifier(name, value)
    except ValueError as exc:
        raise ContinuityManifestError(str(exc)) from exc


def _profile_identity(
    profile_path: Path,
) -> tuple[str, str, Path]:
    data = _read_regular_bytes(profile_path, name="workload profile")
    try:
        loaded = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ContinuityManifestError(
            f"cannot parse workload profile {profile_path}: {exc}"
        ) from exc
    profile = _mapping(loaded, name="workload profile")
    orchestration = _mapping(profile.get("orchestration"), name="profile orchestration")
    run_id = _identifier(orchestration.get("run_id"), name="profile run_id")
    directories = _mapping(
        orchestration.get("directories"),
        name="profile orchestration directories",
    )
    configured_root = _absolute_path(
        directories.get("root"),
        name="profile run root",
    )
    return _digest_bytes(data), run_id, configured_root


def workload_fingerprints(
    profile_path: str | Path,
    run_root: str | Path,
) -> dict[str, object]:
    """Return immutable profile and run-root descriptor hashes for deployment."""

    profile = Path(profile_path).expanduser().resolve()
    root = Path(run_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ContinuityManifestError(f"run root is missing or unsafe: {root}")
    profile_sha256, configured_run_id, configured_root = _profile_identity(profile)
    if configured_root != root:
        raise ContinuityManifestError(
            "profile run root does not match requested run root"
        )
    run_payload, run_data = _json_object_bytes(root / "run.json", name="run identity")
    if run_payload.get("schema_version") != 1:
        raise ContinuityManifestError("unsupported run identity")
    run_id = _identifier(run_payload.get("run_id"), name="run_id")
    generation_family = _identifier(
        run_payload.get("generation_family"),
        name="generation_family",
    )
    created_ns = _positive_int(run_payload.get("created_ns"), name="run created_ns")
    if run_id != configured_run_id:
        raise ContinuityManifestError("profile run_id does not match run identity")
    run_identity_sha256 = _digest_bytes(run_data)
    descriptor = {
        "path": str(root),
        "run_id": run_id,
        "generation_family": generation_family,
        "created_ns": created_ns,
        "run_identity_sha256": run_identity_sha256,
    }
    return {
        "profile_sha256": profile_sha256,
        "run_root_sha256": _digest_bytes(_canonical_bytes(descriptor)),
        **descriptor,
    }


def _parse_workload(value: object, *, index: int) -> Workload:
    raw = _mapping(value, name=f"workload {index}")
    _strict_keys(
        raw,
        name=f"workload {index}",
        required={"id", "role", "unit", "profile", "run_root", "runtime"},
        optional={"last_known_good"},
    )
    workload_id = _identifier(raw.get("id"), name=f"workload {index} id")
    role = raw.get("role")
    if role not in {"primary", "fallback"}:
        raise ContinuityManifestError(
            f"workload {workload_id} role must be primary or fallback"
        )
    unit = raw.get("unit")
    if not isinstance(unit, str) or _UNIT.fullmatch(unit) is None:
        raise ContinuityManifestError(f"workload {workload_id} has an invalid unit")
    profile = _mapping(raw.get("profile"), name=f"workload {workload_id} profile")
    root = _mapping(raw.get("run_root"), name=f"workload {workload_id} run root")
    _strict_keys(
        profile,
        name=f"workload {workload_id} profile",
        required={"path", "sha256"},
    )
    _strict_keys(
        root,
        name=f"workload {workload_id} run root",
        required={"path", "sha256"},
    )
    runtime = _mapping(
        raw.get("runtime"),
        name=f"workload {workload_id} runtime",
    )
    _strict_keys(
        runtime,
        name=f"workload {workload_id} runtime",
        required={
            "manifest",
            "sha256",
            "training_dir",
            "orchestrator",
            "orchestrator_sha256",
            "unit_path",
            "unit_sha256",
        },
    )
    runtime_path = _absolute_path(
        runtime.get("manifest"),
        name=f"workload {workload_id} runtime manifest",
    )
    runtime_sha256 = _sha256_value(
        runtime.get("sha256"),
        name=f"workload {workload_id} runtime manifest SHA-256",
    )
    runtime_training_dir = _absolute_path(
        runtime.get("training_dir"),
        name=f"workload {workload_id} runtime training directory",
    )
    runtime_orchestrator = _absolute_path(
        runtime.get("orchestrator"),
        name=f"workload {workload_id} runtime orchestrator",
    )
    runtime_orchestrator_sha256 = _sha256_value(
        runtime.get("orchestrator_sha256"),
        name=f"workload {workload_id} runtime orchestrator SHA-256",
    )
    runtime_unit_path = _absolute_path(
        runtime.get("unit_path"),
        name=f"workload {workload_id} runtime unit path",
    )
    runtime_unit_sha256 = _sha256_value(
        runtime.get("unit_sha256"),
        name=f"workload {workload_id} runtime unit SHA-256",
    )
    try:
        runtime_orchestrator.relative_to(runtime_training_dir)
    except ValueError as exc:
        raise ContinuityManifestError(
            f"workload {workload_id} orchestrator is outside its training directory"
        ) from exc
    if runtime_unit_path.name != unit:
        raise ContinuityManifestError(
            f"workload {workload_id} runtime unit path does not match its unit"
        )
    lkg = raw.get("last_known_good")
    verified_ns: int | None = None
    priority = 0
    if lkg is not None:
        lkg_payload = _mapping(lkg, name=f"workload {workload_id} LKG")
        _strict_keys(
            lkg_payload,
            name=f"workload {workload_id} LKG",
            required={"verified_ns"},
            optional={"priority"},
        )
        verified_ns = _positive_int(
            lkg_payload.get("verified_ns"),
            name=f"workload {workload_id} LKG verified_ns",
        )
        priority = _nonnegative_int(
            lkg_payload.get("priority", 0),
            name=f"workload {workload_id} LKG priority",
        )
    if role == "fallback" and verified_ns is None:
        raise ContinuityManifestError(
            f"fallback workload {workload_id} must be verified as last-known-good"
        )
    if role == "primary" and verified_ns is not None:
        raise ContinuityManifestError(
            f"primary workload {workload_id} may not also be an LKG fallback"
        )
    return Workload(
        workload_id=workload_id,
        role=str(role),
        unit=unit,
        profile_path=_absolute_path(
            profile.get("path"), name=f"workload {workload_id} profile path"
        ),
        profile_sha256=_sha256_value(
            profile.get("sha256"), name=f"workload {workload_id} profile SHA-256"
        ),
        run_root=_absolute_path(
            root.get("path"), name=f"workload {workload_id} run root path"
        ),
        run_root_sha256=_sha256_value(
            root.get("sha256"), name=f"workload {workload_id} run root SHA-256"
        ),
        runtime_manifest_path=runtime_path,
        runtime_manifest_sha256=runtime_sha256,
        runtime_training_dir=runtime_training_dir,
        runtime_orchestrator_path=runtime_orchestrator,
        runtime_orchestrator_sha256=runtime_orchestrator_sha256,
        runtime_unit_path=runtime_unit_path,
        runtime_unit_sha256=runtime_unit_sha256,
        lkg_verified_ns=verified_ns,
        lkg_priority=priority,
    )


def load_continuity_manifest(path: str | Path) -> ContinuityManifest:
    source = Path(path).expanduser().resolve()
    if Path(path).expanduser().is_symlink():
        raise ContinuityManifestError("continuity manifest may not be a symbolic link")
    try:
        source_stat = source.stat()
    except OSError as exc:
        raise ContinuityManifestError(
            f"cannot inspect continuity manifest {source}: {exc}"
        ) from exc
    if os.geteuid() == 0 and (
        source_stat.st_uid != 0 or stat.S_IMODE(source_stat.st_mode) & 0o022
    ):
        raise ContinuityManifestError(
            "a root continuity service requires a root-owned manifest "
            "that is not group/world writable"
        )
    raw, _data = _json_object_bytes(source, name="continuity manifest")
    _strict_keys(
        raw,
        name="continuity manifest",
        required={
            "format",
            "schema_version",
            "state_root",
            "locks",
            "hardware",
            "primary",
            "workloads",
        },
        optional={"failure_artifacts", "policy", "alerts"},
    )
    if (
        raw.get("format") != MANIFEST_FORMAT
        or raw.get("schema_version") != MANIFEST_SCHEMA_VERSION
    ):
        raise ContinuityManifestError("unsupported continuity manifest")
    state_root = _absolute_path(raw.get("state_root"), name="state root")
    locks = _mapping(raw.get("locks"), name="continuity locks")
    _strict_keys(
        locks,
        name="continuity locks",
        required={"transition", "execution"},
    )
    transition_lock = _absolute_path(locks.get("transition"), name="transition lock")
    execution_lock = _absolute_path(locks.get("execution"), name="execution lock")
    if transition_lock == execution_lock:
        raise ContinuityManifestError("transition and execution locks must differ")

    workload_values = _sequence(raw.get("workloads"), name="workloads")
    if len(workload_values) < 2:
        raise ContinuityManifestError(
            "continuity requires a primary and at least one fallback"
        )
    workloads = tuple(
        _parse_workload(value, index=index)
        for index, value in enumerate(workload_values)
    )
    ids = [workload.workload_id for workload in workloads]
    units = [workload.unit for workload in workloads]
    roots = [workload.run_root for workload in workloads]
    if len(ids) != len(set(ids)):
        raise ContinuityManifestError("workload ids must be unique")
    if len(units) != len(set(units)):
        raise ContinuityManifestError("workload units must be unique")
    if len(roots) != len(set(roots)):
        raise ContinuityManifestError("workload run roots must be unique")
    primary = _identifier(raw.get("primary"), name="primary workload id")
    primary_rows = [workload for workload in workloads if workload.role == "primary"]
    if (
        len(primary_rows) != 1
        or primary_rows[0].workload_id != primary
        or not any(workload.role == "fallback" for workload in workloads)
    ):
        raise ContinuityManifestError(
            "manifest primary/fallback roles are inconsistent"
        )
    for workload in workloads:
        try:
            state_root.relative_to(workload.run_root)
        except ValueError:
            pass
        else:
            raise ContinuityManifestError("state root must be outside every run root")

    hardware = _mapping(raw.get("hardware"), name="hardware policy")
    _strict_keys(
        hardware,
        name="hardware policy",
        required={"report_path", "max_age_seconds", "probe_workload"},
    )
    probe_workload = _identifier(
        hardware.get("probe_workload"),
        name="hardware probe workload",
    )
    if probe_workload not in ids:
        raise ContinuityManifestError("hardware probe workload is not registered")
    hardware_report_path = _absolute_path(
        hardware.get("report_path"), name="hardware report path"
    )
    for protected_path, protected_name in (
        (transition_lock, "transition lock"),
        (hardware_report_path, "hardware report"),
    ):
        try:
            protected_path.relative_to(state_root)
        except ValueError as exc:
            raise ContinuityManifestError(
                f"{protected_name} must be inside the protected state root"
            ) from exc

    artifact_rows = _sequence(
        raw.get("failure_artifacts", []), name="failure artifacts"
    )
    artifacts = []
    for index, value in enumerate(artifact_rows):
        artifact = _mapping(value, name=f"failure artifact {index}")
        _strict_keys(
            artifact,
            name=f"failure artifact {index}",
            required={"type", "path"},
        )
        artifact_type = artifact.get("type")
        if artifact_type not in {"queue_state", "continuity_handoff"}:
            raise ContinuityManifestError(
                f"failure artifact {index} has unsupported type"
            )
        artifacts.append(
            FailureArtifact(
                str(artifact_type),
                _absolute_path(
                    artifact.get("path"), name=f"failure artifact {index} path"
                ),
            )
        )

    policy = _mapping(raw.get("policy", {}), name="continuity policy")
    _strict_keys(
        policy,
        name="continuity policy",
        required=set(),
        optional={
            "automatic_start",
            "maximum_start_attempts",
            "start_retry_seconds",
            "operator_hold_path",
        },
    )
    automatic_start = policy.get("automatic_start", True)
    if not isinstance(automatic_start, bool):
        raise ContinuityManifestError("automatic_start must be boolean")
    hold_value = policy.get("operator_hold_path")
    hold_path = (
        _absolute_path(hold_value, name="operator hold path")
        if hold_value is not None
        else None
    )
    if hold_path is not None:
        try:
            hold_path.relative_to(state_root)
        except ValueError as exc:
            raise ContinuityManifestError(
                "operator hold path must be inside the protected state root"
            ) from exc

    alerts = _mapping(raw.get("alerts", {}), name="alert policy")
    _strict_keys(
        alerts,
        name="alert policy",
        required=set(),
        optional={"command", "timeout_seconds"},
    )
    command_value = alerts.get("command")
    command: tuple[str, ...] | None = None
    if command_value is not None:
        command_rows = _sequence(command_value, name="alert command")
        if not command_rows or not all(
            isinstance(item, str) and item for item in command_rows
        ):
            raise ContinuityManifestError(
                "alert command must contain non-empty string arguments"
            )
        command = tuple(command_rows)

    normalized_digest = _digest_bytes(_canonical_bytes(raw))
    return ContinuityManifest(
        path=source,
        sha256=normalized_digest,
        raw=raw,
        state_root=state_root,
        transition_lock_path=transition_lock,
        execution_lock_path=execution_lock,
        hardware_report_path=hardware_report_path,
        hardware_max_age_seconds=_positive_float(
            hardware.get("max_age_seconds"), name="hardware report maximum age"
        ),
        hardware_probe_workload=probe_workload,
        primary_id=primary,
        workloads=workloads,
        failure_artifacts=tuple(artifacts),
        automatic_start=automatic_start,
        maximum_start_attempts=_positive_int(
            policy.get("maximum_start_attempts", 3),
            name="maximum start attempts",
        ),
        start_retry_seconds=_positive_float(
            policy.get("start_retry_seconds", 60.0),
            name="start retry seconds",
        ),
        operator_hold_path=hold_path,
        alert_command=command,
        alert_timeout_seconds=_positive_float(
            alerts.get("timeout_seconds", 10.0),
            name="alert timeout",
        ),
    )


def verify_workload(
    manifest: ContinuityManifest,
    workload_id: str,
) -> WorkloadVerification:
    workload = manifest.workload(workload_id)
    fingerprints = workload_fingerprints(
        workload.profile_path,
        workload.run_root,
    )
    if fingerprints["profile_sha256"] != workload.profile_sha256:
        raise ContinuityManifestError(
            f"workload {workload_id} profile hash does not match manifest"
        )
    if fingerprints["run_root_sha256"] != workload.run_root_sha256:
        raise ContinuityManifestError(
            f"workload {workload_id} run-root hash does not match manifest"
        )
    assert workload.runtime_manifest_path is not None
    assert workload.runtime_manifest_sha256 is not None
    assert workload.runtime_training_dir is not None
    assert workload.runtime_orchestrator_path is not None
    assert workload.runtime_orchestrator_sha256 is not None
    assert workload.runtime_unit_path is not None
    assert workload.runtime_unit_sha256 is not None
    runtime_payload, runtime_data = _json_object_bytes(
        workload.runtime_manifest_path,
        name=f"workload {workload_id} runtime manifest",
    )
    runtime_digest = _digest_bytes(runtime_data)
    if runtime_digest != workload.runtime_manifest_sha256:
        raise ContinuityManifestError(
            f"workload {workload_id} runtime manifest hash does not match"
        )
    if (
        runtime_payload.get("schema_version") != 1
        or runtime_payload.get("report") != "edgeconnect-immutable-release"
    ):
        raise ContinuityManifestError(
            f"workload {workload_id} runtime manifest schema is unsupported"
        )
    release_root = workload.runtime_manifest_path.parent
    if workload.runtime_training_dir != release_root / "training":
        raise ContinuityManifestError(
            f"workload {workload_id} runtime training directory is inconsistent"
        )
    source_files = runtime_payload.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ContinuityManifestError(
            f"workload {workload_id} runtime source manifest is empty"
        )
    required_sources = {
        "training/startrain/orchestration.py",
        "training/startrain/continuity.py",
        "training/scripts/reconcile_training_continuity.py",
    }
    if not required_sources.issubset(source_files):
        raise ContinuityManifestError(
            f"workload {workload_id} runtime manifest omits launch-critical sources"
        )
    for relative, expected_digest in source_files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected_digest, str)
            or _SHA256.fullmatch(expected_digest) is None
        ):
            raise ContinuityManifestError(
                f"workload {workload_id} runtime source entry is invalid"
            )
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContinuityManifestError(
                f"workload {workload_id} runtime source path is unsafe"
            )
        source_path = (release_root / relative_path).resolve(strict=False)
        try:
            source_path.relative_to(release_root)
        except ValueError as exc:
            raise ContinuityManifestError(
                f"workload {workload_id} runtime source escaped its release"
            ) from exc
        source_data = _read_regular_bytes(
            source_path,
            name=f"workload {workload_id} runtime source",
        )
        if _digest_bytes(source_data) != expected_digest:
            raise ContinuityManifestError(
                f"workload {workload_id} runtime source hash does not match: {relative}"
            )
    for path, expected_digest, name in (
        (
            workload.runtime_orchestrator_path,
            workload.runtime_orchestrator_sha256,
            "orchestrator",
        ),
        (
            workload.runtime_unit_path,
            workload.runtime_unit_sha256,
            "systemd unit",
        ),
    ):
        data = _read_regular_bytes(path, name=f"workload {workload_id} {name}")
        if _digest_bytes(data) != expected_digest:
            raise ContinuityManifestError(
                f"workload {workload_id} {name} hash does not match"
            )
    if os.geteuid() == 0:
        for path, name in (
            (release_root, "release root"),
            (workload.runtime_manifest_path, "runtime manifest"),
            (workload.runtime_orchestrator_path, "orchestrator"),
            (workload.runtime_unit_path, "systemd unit"),
        ):
            path_stat = path.stat()
            if path_stat.st_uid != 0 or stat.S_IMODE(path_stat.st_mode) & 0o022:
                raise ContinuityManifestError(
                    f"workload {workload_id} {name} is not root-owned immutable"
                )
    return WorkloadVerification(
        workload=workload,
        run_id=str(fingerprints["run_id"]),
        generation_family=str(fingerprints["generation_family"]),
        created_ns=_positive_int(
            fingerprints["created_ns"],
            name="verified run created_ns",
        ),
        run_identity_sha256=str(fingerprints["run_identity_sha256"]),
    )


def verify_continuity_manifest(
    manifest: ContinuityManifest,
    *,
    workload_id: str | None = None,
) -> dict[str, object]:
    selected = (
        (manifest.workload(workload_id),)
        if workload_id is not None
        else manifest.workloads
    )
    verified = [
        verify_workload(manifest, workload.workload_id).as_dict()
        for workload in selected
    ]
    return {
        "format": MANIFEST_FORMAT,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "ok",
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "workloads": verified,
    }


def _secure_directory(path: Path) -> None:
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise ContinuityStateError(f"host state directory is unsafe: {path}")
    path.mkdir(parents=True, exist_ok=True)
    directory_stat = path.stat()
    if os.geteuid() == 0 and (
        directory_stat.st_uid != 0 or stat.S_IMODE(directory_stat.st_mode) & 0o022
    ):
        raise ContinuityStateError(
            "a root continuity service requires a root-owned state directory "
            "that is not group/world writable"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    mode: int = 0o600,
) -> bool:
    """Atomically create immutable JSON without replacing an existing record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        written = os.write(descriptor, data)
        if written != len(data):
            raise OSError("short immutable JSON write")
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_replace_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    mode: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        written = os.write(descriptor, data)
        if written != len(data):
            raise OSError("short atomic JSON write")
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _pin_manifest(manifest: ContinuityManifest) -> None:
    _secure_directory(manifest.state_root)
    if not manifest.pinned_manifest_path.exists():
        created = _atomic_create_json(
            manifest.pinned_manifest_path,
            manifest.raw,
            mode=0o644,
        )
        if created:
            return
    try:
        pinned, _ = _json_object_bytes(
            manifest.pinned_manifest_path,
            name="pinned continuity manifest",
        )
    except ContinuityManifestError as exc:
        raise ContinuityStateError(str(exc)) from exc
    if _digest_bytes(_canonical_bytes(pinned)) != manifest.sha256:
        raise ContinuityStateError(
            "continuity manifest differs from the atomically pinned host manifest"
        )


@contextmanager
def nonblocking_host_lock(
    path: Path,
    *,
    owner: str,
    mode: int = 0o600,
) -> Iterator[int]:
    """Acquire a host-wide advisory lock without waiting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ContinuityBusyError(f"cannot open {owner} lock {path}: {exc}") from exc
    try:
        current_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
        if current_mode != mode:
            try:
                os.fchmod(descriptor, mode)
            except PermissionError:
                if current_mode & mode != mode:
                    raise
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContinuityBusyError(f"another {owner} owns {path}") from exc
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _execution_lock_available(manifest: ContinuityManifest) -> bool:
    try:
        with nonblocking_host_lock(
            manifest.execution_lock_path,
            owner="GPU workload",
            mode=0o660,
        ):
            return True
    except ContinuityBusyError:
        return False


def _pid_is_live(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def live_coordinator_locks(
    manifest: ContinuityManifest,
) -> dict[str, dict[str, object]]:
    live: dict[str, dict[str, object]] = {}
    for workload in manifest.workloads:
        path = workload.run_root / "coordinator.lock"
        if not path.exists():
            continue
        try:
            payload, _ = _json_object_bytes(path, name="coordinator lock")
            pid_value = payload.get("pid")
            pid = (
                pid_value
                if isinstance(pid_value, int) and not isinstance(pid_value, bool)
                else 0
            )
        except ContinuityManifestError:
            # A malformed lock is not proven live. Workload hash/preflight still
            # fails closed before any launch.
            continue
        if _pid_is_live(pid):
            live[workload.workload_id] = {
                "workload_id": workload.workload_id,
                "path": str(path),
                "pid": pid,
            }
    return live


def assess_hardware(
    manifest: ContinuityManifest,
    *,
    now_ns: int,
    report_path: Path | None = None,
) -> HardwareAssessment:
    path = manifest.hardware_report_path if report_path is None else report_path
    if not path.exists():
        return HardwareAssessment(
            "unavailable",
            False,
            str(path),
            None,
            None,
            ("hardware health report is missing",),
        )
    try:
        payload, data = _json_object_bytes(path, name="hardware health report")
    except ContinuityManifestError as exc:
        return HardwareAssessment(
            "unavailable", False, str(path), None, None, (str(exc),)
        )
    digest = _digest_bytes(data)
    captured_value = payload.get("captured_ns")
    captured_ns = (
        captured_value
        if isinstance(captured_value, int)
        and not isinstance(captured_value, bool)
        and captured_value > 0
        else None
    )
    if payload.get("schema_version") != 1 or captured_ns is None:
        return HardwareAssessment(
            "unavailable",
            False,
            str(path),
            digest,
            captured_ns,
            ("hardware health report schema is invalid",),
        )
    probe = manifest.workload(manifest.hardware_probe_workload)
    try:
        probe_profile_sha256, _run_id, _run_root = _profile_identity(probe.profile_path)
    except ContinuityManifestError as exc:
        return HardwareAssessment(
            "unavailable",
            False,
            str(path),
            digest,
            captured_ns,
            (f"hardware probe profile is invalid: {exc}",),
        )
    if probe_profile_sha256 != probe.profile_sha256:
        return HardwareAssessment(
            "unavailable",
            False,
            str(path),
            digest,
            captured_ns,
            ("hardware probe profile hash does not match the manifest",),
        )
    config = payload.get("config")
    if (
        not isinstance(config, str)
        or Path(config).expanduser().resolve() != probe.profile_path
    ):
        return HardwareAssessment(
            "unavailable",
            False,
            str(path),
            digest,
            captured_ns,
            ("hardware report was produced for a different profile",),
        )
    age_ns = now_ns - captured_ns
    maximum_age_ns = int(manifest.hardware_max_age_seconds * 1_000_000_000)
    if age_ns < -300_000_000_000 or age_ns > maximum_age_ns:
        return HardwareAssessment(
            "unavailable",
            False,
            str(path),
            digest,
            captured_ns,
            ("hardware health report is stale or from the future",),
        )
    healthy = payload.get("healthy")
    if healthy is True:
        return HardwareAssessment("healthy", True, str(path), digest, captured_ns, ())
    reasons: list[str] = []
    missing = payload.get("missing_indices")
    for index in missing if isinstance(missing, list) else []:
        reasons.append(f"GPU {index}: missing")
    rows = payload.get("gpus")
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        row_reasons = row.get("reasons")
        for reason in row_reasons if isinstance(row_reasons, (list, tuple)) else []:
            reasons.append(f"GPU {row.get('index')}: {reason}")
    if reasons:
        return HardwareAssessment(
            "unsafe",
            False,
            str(path),
            digest,
            captured_ns,
            tuple(reasons),
        )
    query_error = payload.get("query_error")
    reason = (
        f"hardware query failed: {query_error}"
        if isinstance(query_error, str) and query_error
        else "hardware report is unhealthy without concrete GPU evidence"
    )
    return HardwareAssessment(
        "unavailable", False, str(path), digest, captured_ns, (reason,)
    )


def _artifact_payload(path: Path) -> tuple[dict[str, Any] | None, str, str | None]:
    try:
        data = _read_regular_bytes(path, name="failure artifact")
    except ContinuityManifestError as exc:
        return None, _digest_bytes(str(exc).encode()), str(exc)
    digest = _digest_bytes(data)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, digest, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, digest, "failure artifact is not a JSON object"
    return value, digest, None


def _failure_domain(payload: Mapping[str, object]) -> str:
    if (
        payload.get("requested_action") == "reconcile_training_continuity"
        or payload.get("status") == "requested"
    ) and payload.get("reason") == "queue_completed":
        return "handoff"
    values = [
        payload.get("failure_domain"),
        payload.get("domain"),
        payload.get("failure_class"),
        payload.get("terminal_reason"),
        payload.get("outcome"),
    ]
    normalized = {str(value).casefold() for value in values if value is not None}
    if any(
        any(token in value for token in ("hardware", "gpu_health", "gpu-unsafe"))
        for value in normalized
    ):
        return "hardware_reported"
    if any(
        any(
            token in value
            for token in (
                "transient",
                "software",
                "fatal",
                "config",
                "invariant",
                "corrupt",
                "restart_budget",
                "runner",
            )
        )
        for value in normalized
    ):
        return "run_failure"
    return "unknown"


def _observation(
    *,
    payload: dict[str, Any] | None,
    digest: str,
    parse_error: str | None,
    source_type: str,
    path: Path,
) -> FailureObservation:
    evidence = {} if payload is None else copy.deepcopy(payload)
    reason_value = None if payload is None else payload.get("reason")
    if reason_value is None and payload is not None:
        reason_value = payload.get("failure") or payload.get("queue_error")
    reason = (
        str(reason_value)
        if reason_value not in (None, "")
        else parse_error or f"{source_type} requested continuity fallback"
    )
    terminal_value = None if payload is None else payload.get("terminal_reason")
    terminal_reason = str(terminal_value) if terminal_value is not None else None
    occurred_value = None
    if payload is not None:
        for key in (
            "timestamp_ns",
            "created_ns",
            "requested_ns",
            "updated_ns",
            "last_stopped_ns",
        ):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                occurred_value = value
                break
    semantic_identity: dict[str, object] = {
        "source_type": source_type,
        "source_path": str(path),
    }
    if payload is not None:
        for key in (
            "report",
            "requested_action",
            "reason",
            "terminal_reason",
            "timestamp_ns",
            "treatment",
            "run_root",
            "last_stopped_ns",
            "outcome",
            "queue_status",
        ):
            value = payload.get(key)
            if value is not None:
                semantic_identity[key] = value
        source = payload.get("source")
        if isinstance(source, dict):
            semantic_identity["source"] = {
                key: source.get(key)
                for key in ("kind", "manifest", "queue_state")
                if source.get(key) is not None
            }
        quarantined = payload.get("quarantined_arms")
        if isinstance(quarantined, list):
            semantic_identity["quarantined_arms"] = [
                {
                    key: arm.get(key)
                    for key in ("treatment", "run_root")
                    if arm.get(key) is not None
                }
                for arm in quarantined
                if isinstance(arm, dict)
            ]
    if len(semantic_identity) == 2:
        semantic_identity["source_sha256"] = digest
    identity = _digest_bytes(_canonical_bytes(semantic_identity))
    return FailureObservation(
        failure_id=identity,
        domain=_failure_domain(evidence),
        reason=reason,
        source_type=source_type,
        source_path=str(path),
        source_sha256=digest,
        occurred_ns=occurred_value,
        terminal_reason=terminal_reason,
        evidence=evidence,
    )


def _queue_failure_payload(payload: Mapping[str, object]) -> dict[str, Any] | None:
    handoff = payload.get("continuity_handoff")
    if isinstance(handoff, dict) and (
        handoff.get("requested") is True
        or handoff.get("status") in {"requested", "pending"}
        or handoff.get("action") in {"fallback", "request_fallback"}
    ):
        return dict(handoff)
    if payload.get("queue_status") not in {"failed", "quarantined"}:
        return None
    raw_arms = payload.get("arms")
    arms = raw_arms if isinstance(raw_arms, list) else []
    failed: list[dict[str, Any]] = [
        arm
        for arm in arms
        if isinstance(arm, dict) and arm.get("status") in {"failed", "quarantined"}
    ]

    def stopped_ns(arm: Mapping[str, object]) -> int:
        value = arm.get("last_stopped_ns")
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    selected = max(
        failed,
        key=stopped_ns,
        default={},
    )
    return {
        "queue_status": payload.get("queue_status"),
        "queue_error": payload.get("queue_error"),
        **selected,
        "updated_ns": payload.get("updated_ns"),
    }


def _coordinator_failure_payload(
    payload: Mapping[str, object],
) -> dict[str, Any] | None:
    failure = payload.get("failure")
    if not isinstance(failure, dict):
        return None
    selected = dict(failure)
    for key in ("timestamp_ns", "updated_ns"):
        if key not in selected and payload.get(key) is not None:
            selected[key] = payload.get(key)
    return selected


def detect_failure(
    manifest: ContinuityManifest,
    workload_id: str,
    *,
    handled_failure_ids: set[str] | None = None,
) -> FailureObservation | None:
    """Classify durable coordinator/fatal/queue evidence for one stopped workload."""

    handled = handled_failure_ids or set()
    workload = manifest.workload(workload_id)
    candidates: list[
        tuple[str, Path, Callable[[dict[str, Any]], dict[str, Any] | None]]
    ] = [
        (
            "coordinator_fatal",
            workload.run_root / "status" / "fatal.json",
            lambda row: row,
        ),
        (
            "continuity_handoff",
            workload.run_root / "status" / "continuity-handoff.json",
            lambda row: (
                row
                if row.get("requested") is True
                or row.get("action") in {"fallback", "request_fallback"}
                else None
            ),
        ),
        (
            "coordinator_status",
            workload.run_root / "status" / "coordinator.json",
            _coordinator_failure_payload,
        ),
    ]
    for artifact in manifest.failure_artifacts:
        parser = (
            _queue_failure_payload
            if artifact.artifact_type == "queue_state"
            else (
                lambda row: (
                    row
                    if row.get("requested") is True
                    or row.get("action") in {"fallback", "request_fallback"}
                    or row.get("status") in {"requested", "pending"}
                    else None
                )
            )
        )
        candidates.append((artifact.artifact_type, artifact.path, parser))
    for source_type, path, parser in candidates:
        if not path.exists():
            continue
        payload, digest, parse_error = _artifact_payload(path)
        selected = None if payload is None else parser(payload)
        if payload is not None and selected is None:
            continue
        observation = _observation(
            payload=selected,
            digest=digest,
            parse_error=parse_error,
            source_type=source_type,
            path=path,
        )
        if (
            workload.lkg_verified_ns is not None
            and observation.occurred_ns is not None
            and observation.occurred_ns <= workload.lkg_verified_ns
        ):
            continue
        if observation.failure_id not in handled:
            return observation
    return None


def _synthetic_failure(
    workload: Workload,
    *,
    reason: str,
    source_type: str,
    now_ns: int,
) -> FailureObservation:
    evidence = {
        "workload_id": workload.workload_id,
        "reason": reason,
        "timestamp_ns": now_ns,
        "failure_domain": "software",
    }
    digest = _digest_bytes(_canonical_bytes(evidence))
    return _observation(
        payload=evidence,
        digest=digest,
        parse_error=None,
        source_type=source_type,
        path=workload.run_root,
    )


def _initial_state(manifest: ContinuityManifest, *, now_ns: int) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "schema_version": STATE_SCHEMA_VERSION,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "hostname": socket.gethostname(),
        "revision": 0,
        "created_ns": now_ns,
        "updated_ns": now_ns,
        "last_reconciled_ns": now_ns,
        "phase": "idle",
        "primary_workload_id": manifest.primary_id,
        "desired_workload_id": manifest.primary_id,
        "active_workload_id": None,
        "selected_lkg_workload_id": None,
        "active_profile_sha256": None,
        "active_run_root_sha256": None,
        "productive_idle_since_ns": now_ns,
        "hardware": None,
        "execution": None,
        "fallback_attempts": 0,
        "start_attempts": {},
        "last_start_requested_ns": {},
        "quarantined_workloads": [],
        "quarantine_records": [],
        "handled_failure_ids": [],
        "last_failure": None,
        "last_handoff": None,
        "last_transition": None,
        "blocked_reason": None,
        "last_alert": None,
    }


def _load_state(
    manifest: ContinuityManifest,
    *,
    now_ns: int,
) -> tuple[dict[str, Any], bool]:
    if not manifest.state_path.exists():
        return _initial_state(manifest, now_ns=now_ns), True
    if manifest.state_path.is_symlink():
        raise ContinuityStateError("continuity state may not be a symbolic link")
    try:
        with manifest.state_path.open("r", encoding="utf-8") as stream:
            state = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuityStateError(f"cannot read continuity state: {exc}") from exc
    if not isinstance(state, dict):
        raise ContinuityStateError("continuity state must be a JSON object")
    if (
        state.get("format") != STATE_FORMAT
        or state.get("schema_version") != STATE_SCHEMA_VERSION
        or state.get("manifest_path") != str(manifest.path)
        or state.get("manifest_sha256") != manifest.sha256
        or state.get("primary_workload_id") != manifest.primary_id
    ):
        raise ContinuityStateError(
            "continuity state is unsupported or belongs to another manifest"
        )
    for name in (
        "revision",
        "created_ns",
        "updated_ns",
        "last_reconciled_ns",
        "fallback_attempts",
    ):
        value = state.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContinuityStateError(f"continuity state {name} is invalid")
    for name in (
        "start_attempts",
        "last_start_requested_ns",
    ):
        if not isinstance(state.get(name), dict):
            raise ContinuityStateError(f"continuity state {name} is invalid")
    for name in (
        "quarantined_workloads",
        "quarantine_records",
        "handled_failure_ids",
    ):
        if not isinstance(state.get(name), list):
            raise ContinuityStateError(f"continuity state {name} is invalid")
    known_ids = {workload.workload_id for workload in manifest.workloads}
    for workload_id, value in state["start_attempts"].items():
        if (
            workload_id not in known_ids
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ContinuityStateError("continuity state start attempts are invalid")
    for workload_id, value in state["last_start_requested_ns"].items():
        if (
            workload_id not in known_ids
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ContinuityStateError(
                "continuity state last start requests are invalid"
            )
    for name in (
        "desired_workload_id",
        "active_workload_id",
        "selected_lkg_workload_id",
    ):
        value = state.get(name)
        if value is not None and value not in known_ids:
            raise ContinuityStateError(f"continuity state {name} is unknown")
    return state, False


def _state_without_metadata(state: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in state.items()
        if key not in {"revision", "updated_ns"}
    }


def _persist_state(
    manifest: ContinuityManifest,
    state: dict[str, Any],
    previous: Mapping[str, object] | None,
    *,
    now_ns: int,
) -> bool:
    if previous is not None and _state_without_metadata(
        state
    ) == _state_without_metadata(previous):
        state.clear()
        state.update(copy.deepcopy(previous))
        return False
    previous_revision_value = previous.get("revision", 0) if previous is not None else 0
    if (
        isinstance(previous_revision_value, bool)
        or not isinstance(previous_revision_value, int)
        or previous_revision_value < 0
    ):
        raise ContinuityStateError("continuity state revision is invalid")
    previous_revision = previous_revision_value
    state["revision"] = previous_revision + 1
    state["updated_ns"] = now_ns
    _atomic_replace_json(manifest.state_path, state, mode=0o644)
    return True


def _write_alert(
    manifest: ContinuityManifest,
    state: dict[str, Any],
    *,
    kind: str,
    dedupe_key: str,
    severity: str,
    message: str,
    now_ns: int,
    details: Mapping[str, object],
) -> tuple[Path, bool]:
    event_id = _digest_bytes(_canonical_bytes({"kind": kind, "dedupe_key": dedupe_key}))
    path = manifest.alerts_root / f"{event_id}.json"
    payload = {
        "format": ALERT_FORMAT,
        "schema_version": 1,
        "event_id": event_id,
        "created_ns": now_ns,
        "hostname": socket.gethostname(),
        "severity": severity,
        "kind": kind,
        "message": message,
        "manifest_sha256": manifest.sha256,
        "details": dict(details),
    }
    created = _atomic_create_json(path, payload)
    event_created_ns = now_ns
    if not created:
        try:
            existing, _ = _json_object_bytes(path, name="continuity alert")
        except ContinuityManifestError as exc:
            raise ContinuityStateError(str(exc)) from exc
        if (
            existing.get("format") != ALERT_FORMAT
            or existing.get("event_id") != event_id
        ):
            raise ContinuityStateError(f"immutable continuity alert is corrupt: {path}")
        existing_created_ns = existing.get("created_ns")
        if (
            isinstance(existing_created_ns, int)
            and not isinstance(existing_created_ns, bool)
            and existing_created_ns > 0
        ):
            event_created_ns = existing_created_ns
    state["last_alert"] = {
        "event_id": event_id,
        "path": str(path),
        "severity": severity,
        "kind": kind,
        "created_ns": event_created_ns,
    }
    return path, created


def _deliver_alert(
    manifest: ContinuityManifest,
    alert_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    if manifest.alert_command is None:
        return
    started_ns = time.time_ns()
    delivery: dict[str, object]
    try:
        completed = runner(
            [*manifest.alert_command, str(alert_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=manifest.alert_timeout_seconds,
            env={
                **os.environ,
                "STARTRAIN_CONTINUITY_ALERT": str(alert_path),
                "STARTRAIN_CONTINUITY_MANIFEST": str(manifest.path),
            },
        )
        delivery = {
            "status": "delivered" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4096:],
        }
    except Exception as exc:  # notification must never gate recovery
        delivery = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    delivery.update(
        {
            "schema_version": 1,
            "alert_path": str(alert_path),
            "started_ns": started_ns,
            "completed_ns": time.time_ns(),
        }
    )
    atomic_json(alert_path.with_suffix(".delivery.json"), delivery)


def _quarantine(
    manifest: ContinuityManifest,
    state: dict[str, Any],
    workload: Workload,
    failure: FailureObservation,
    *,
    now_ns: int,
) -> Path:
    quarantine_id = _digest_bytes(
        _canonical_bytes(
            {
                "workload_id": workload.workload_id,
                "failure_id": failure.failure_id,
            }
        )
    )
    path = manifest.quarantine_root / f"{quarantine_id}.json"
    payload = {
        "format": QUARANTINE_FORMAT,
        "schema_version": 1,
        "quarantine_id": quarantine_id,
        "created_ns": now_ns,
        "metadata_only": True,
        "artifacts_preserved": True,
        "workload": workload.state_identity(),
        "failure": failure.as_dict(),
    }
    created = _atomic_create_json(path, payload)
    if not created:
        try:
            existing, _ = _json_object_bytes(path, name="continuity quarantine")
        except ContinuityManifestError as exc:
            raise ContinuityStateError(str(exc)) from exc
        existing_failure = existing.get("failure")
        if (
            existing.get("format") != QUARANTINE_FORMAT
            or existing.get("quarantine_id") != quarantine_id
            or not isinstance(existing_failure, dict)
            or existing_failure.get("failure_id") != failure.failure_id
        ):
            raise ContinuityStateError(
                f"immutable continuity quarantine is corrupt: {path}"
            )
    if workload.workload_id not in state["quarantined_workloads"]:
        state["quarantined_workloads"].append(workload.workload_id)
        state["quarantined_workloads"].sort()
    if str(path) not in state["quarantine_records"]:
        state["quarantine_records"].append(str(path))
    if failure.failure_id not in state["handled_failure_ids"]:
        state["handled_failure_ids"].append(failure.failure_id)
    state["last_failure"] = failure.as_dict()
    state["last_transition"] = {
        "kind": "quarantine",
        "workload_id": workload.workload_id,
        "failure_id": failure.failure_id,
        "quarantine_path": str(path),
        "timestamp_ns": now_ns,
    }
    state["active_workload_id"] = None
    state["desired_workload_id"] = None
    return path


def select_last_known_good(
    manifest: ContinuityManifest,
    *,
    excluded_workload_ids: set[str],
) -> tuple[Workload | None, dict[str, str]]:
    """Return the newest verified LKG, skipping quarantined or corrupt entries."""

    candidates = sorted(
        (
            workload
            for workload in manifest.workloads
            if workload.is_lkg and workload.workload_id not in excluded_workload_ids
        ),
        key=lambda workload: (
            -(workload.lkg_verified_ns or 0),
            workload.lkg_priority,
            workload.workload_id,
        ),
    )
    rejected: dict[str, str] = {}
    for workload in candidates:
        try:
            verify_workload(manifest, workload.workload_id)
        except ContinuityManifestError as exc:
            rejected[workload.workload_id] = str(exc)
            continue
        return workload, rejected
    return None, rejected


def _execution_snapshot(
    manifest: ContinuityManifest,
    unit_manager: UnitManager,
) -> tuple[dict[str, UnitStatus], dict[str, dict[str, object]], set[str]]:
    statuses = {
        workload.workload_id: unit_manager.status(workload.unit)
        for workload in manifest.workloads
    }
    coordinator_locks = live_coordinator_locks(manifest)
    active = {
        workload_id for workload_id, status in statuses.items() if status.active
    } | set(coordinator_locks)
    return statuses, coordinator_locks, active


def assess_workload_productivity(
    workload: Workload,
    *,
    now_ns: int,
    maximum_heartbeat_age_seconds: float,
) -> dict[str, object]:
    coordinator_path = workload.run_root / "status" / "coordinator.json"
    if not coordinator_path.is_file():
        return {
            "status": "unknown",
            "productive": None,
            "reason": "coordinator status is not available",
        }
    try:
        coordinator, _ = _json_object_bytes(
            coordinator_path,
            name="coordinator status",
        )
    except ContinuityManifestError as exc:
        return {
            "status": "available",
            "productive": False,
            "reason": str(exc),
        }
    workers = coordinator.get("workers")
    if coordinator.get("state") != "running" or not isinstance(workers, dict):
        return {
            "status": "available",
            "productive": False,
            "reason": "coordinator or worker state is not running",
        }

    maximum_age_ns = int(maximum_heartbeat_age_seconds * 1_000_000_000)

    def heartbeat(worker: Mapping[str, object]) -> dict[str, Any] | None:
        raw_path = worker.get("heartbeat")
        if not isinstance(raw_path, str):
            return None
        path = Path(raw_path)
        if not path.is_file():
            return None
        try:
            payload, _ = _json_object_bytes(path, name="worker heartbeat")
        except ContinuityManifestError:
            return None
        observed = payload.get("heartbeat_ns")
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed <= 0
            or now_ns - observed < 0
            or now_ns - observed > maximum_age_ns
        ):
            return None
        return payload

    learner_row = next(
        (
            row
            for row in workers.values()
            if isinstance(row, dict) and row.get("role") == "learner"
        ),
        None,
    )
    learner_heartbeat = (
        heartbeat(learner_row) if isinstance(learner_row, dict) else None
    )
    productive_learner_phases = {
        "training",
        "replay_wait",
        "update_to_data_wait",
        "arena_gpu_pause",
    }
    learner_productive = (
        learner_heartbeat is not None
        and isinstance(learner_heartbeat.get("progress"), int)
        and not isinstance(learner_heartbeat.get("progress"), bool)
        and learner_heartbeat.get("phase") in productive_learner_phases
    )
    progressing_actors = []
    for name, row in workers.items():
        if not isinstance(row, dict) or row.get("role") != "actor":
            continue
        actor_heartbeat = heartbeat(row)
        if (
            actor_heartbeat is not None
            and isinstance(actor_heartbeat.get("progress"), int)
            and not isinstance(actor_heartbeat.get("progress"), bool)
        ):
            progressing_actors.append(str(name))
    productive = learner_productive and bool(progressing_actors)
    return {
        "status": "available",
        "productive": productive,
        "reason": None
        if productive
        else "learner and actor progress are not both current",
        "learner_phase": (
            learner_heartbeat.get("phase") if learner_heartbeat is not None else None
        ),
        "learner_progress": (
            learner_heartbeat.get("progress") if learner_heartbeat is not None else None
        ),
        "progressing_actors": progressing_actors,
    }


def _set_execution_state(
    state: dict[str, Any],
    statuses: Mapping[str, UnitStatus],
    coordinator_locks: Mapping[str, Mapping[str, object]],
    *,
    execution_lock_available: bool,
) -> None:
    state["execution"] = {
        "lock_path": state.get("execution", {}).get("lock_path")
        if isinstance(state.get("execution"), dict)
        else None,
        "lock_available": execution_lock_available,
        "units": {
            workload_id: status.as_dict() for workload_id, status in statuses.items()
        },
        "live_coordinators": dict(coordinator_locks),
    }


def reconcile_training_continuity(
    manifest_path: str | Path,
    *,
    now_ns: int | None = None,
    hardware_report_path: str | Path | None = None,
    unit_manager: UnitManager | None = None,
    alert_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Idempotently reconcile host execution with the pinned continuity policy."""

    manifest = load_continuity_manifest(manifest_path)
    now = time.time_ns() if now_ns is None else now_ns
    manager = SystemdUnitManager() if unit_manager is None else unit_manager
    report_override = (
        Path(hardware_report_path).expanduser().resolve()
        if hardware_report_path is not None
        else None
    )
    alerts_to_deliver: list[Path] = []
    try:
        transition = nonblocking_host_lock(
            manifest.transition_lock_path,
            owner="continuity transition",
        )
        with transition:
            try:
                _pin_manifest(manifest)
                state, is_new = _load_state(manifest, now_ns=now)
            except ContinuityStateError as exc:
                try:
                    _secure_directory(manifest.state_root)
                    emergency_state: dict[str, Any] = {}
                    _write_alert(
                        manifest,
                        emergency_state,
                        kind="continuity_state_blocked",
                        dedupe_key=_digest_bytes(str(exc).encode("utf-8")),
                        severity="critical",
                        message=(
                            "continuity state or pinned manifest validation failed"
                        ),
                        now_ns=now,
                        details={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "state_path": str(manifest.state_path),
                        },
                    )
                except Exception:
                    # Preserve the original fail-closed state error even if the
                    # emergency alert directory is also damaged.
                    pass
                raise
            previous = None if is_new else copy.deepcopy(state)
            state["last_reconciled_ns"] = now
            hardware = assess_hardware(
                manifest,
                now_ns=now,
                report_path=report_override,
            )
            state["hardware"] = hardware.as_dict()
            statuses, coordinator_locks, active_ids = _execution_snapshot(
                manifest, manager
            )
            lock_available = _execution_lock_available(manifest)
            _set_execution_state(
                state,
                statuses,
                coordinator_locks,
                execution_lock_available=lock_available,
            )
            assert isinstance(state["execution"], dict)
            state["execution"]["lock_path"] = str(manifest.execution_lock_path)

            if len(active_ids) > 1:
                state["phase"] = "blocked_split_brain"
                state["blocked_reason"] = {
                    "code": "multiple_live_workloads",
                    "workload_ids": sorted(active_ids),
                }
                state["active_workload_id"] = None
                alert, created = _write_alert(
                    manifest,
                    state,
                    kind="split_brain_blocked",
                    dedupe_key=":".join(sorted(active_ids)),
                    severity="critical",
                    message="multiple registered GPU workloads appear live",
                    now_ns=now,
                    details=state["blocked_reason"],
                )
                if created:
                    alerts_to_deliver.append(alert)
                _persist_state(manifest, state, previous, now_ns=now)
                result: dict[str, object] = copy.deepcopy(state)
            elif len(active_ids) == 1:
                active_id = next(iter(active_ids))
                workload = manifest.workload(active_id)
                state["active_workload_id"] = active_id
                state["active_profile_sha256"] = workload.profile_sha256
                state["active_run_root_sha256"] = workload.run_root_sha256
                state["productive_idle_since_ns"] = None
                expected_id = state.get("desired_workload_id")
                quarantined_ids = set(
                    str(value) for value in state["quarantined_workloads"]
                )
                unexpected = active_id in quarantined_ids or (
                    isinstance(expected_id, str) and active_id != expected_id
                )
                if unexpected:
                    state["phase"] = "blocked_unexpected_active"
                    state["blocked_reason"] = {
                        "code": "unexpected_or_quarantined_workload_active",
                        "active_workload_id": active_id,
                        "desired_workload_id": expected_id,
                        "quarantined": active_id in quarantined_ids,
                        "hardware_status": hardware.status,
                    }
                    alert, created = _write_alert(
                        manifest,
                        state,
                        kind="unexpected_workload_active",
                        dedupe_key=f"{active_id}:{expected_id}",
                        severity="critical",
                        message=(
                            "a quarantined or unselected workload appears active; "
                            "automatic transition is blocked"
                        ),
                        now_ns=now,
                        details=state["blocked_reason"],
                    )
                    if created:
                        alerts_to_deliver.append(alert)
                else:
                    productivity = assess_workload_productivity(
                        workload,
                        now_ns=now,
                        maximum_heartbeat_age_seconds=(
                            manifest.hardware_max_age_seconds
                        ),
                    )
                    assert isinstance(state["execution"], dict)
                    state["execution"]["productivity"] = productivity
                    productivity_known = productivity.get("status") == "available"
                    productive = productivity.get("productive") is True
                    if productivity_known and not productive:
                        if state.get("productive_idle_since_ns") is None:
                            state["productive_idle_since_ns"] = now
                        state["phase"] = (
                            "starting_fallback"
                            if workload.role == "fallback"
                            else "starting_primary"
                        )
                    else:
                        state["productive_idle_since_ns"] = None
                        state["phase"] = (
                            "active_fallback"
                            if workload.role == "fallback"
                            else "active_primary"
                        )
                    state["desired_workload_id"] = active_id
                    state["selected_lkg_workload_id"] = (
                        active_id
                        if workload.is_lkg
                        else state["selected_lkg_workload_id"]
                    )
                    state["blocked_reason"] = (
                        {
                            "code": "active_hardware_not_safe",
                            "hardware_status": hardware.status,
                        }
                        if not hardware.safe
                        else None
                    )
                    if not hardware.safe:
                        alert, created = _write_alert(
                            manifest,
                            state,
                            kind="active_hardware_not_safe",
                            dedupe_key=hardware.report_sha256 or hardware.status,
                            severity="critical",
                            message=(
                                "a registered workload is active while current "
                                "hardware safety is not proven"
                            ),
                            now_ns=now,
                            details={
                                "workload_id": active_id,
                                "hardware": hardware.as_dict(),
                            },
                        )
                        if created:
                            alerts_to_deliver.append(alert)
                    if hardware.status == "unsafe":
                        try:
                            manager.stop(workload.unit)
                        except Exception as exc:
                            state["phase"] = "blocked_hardware_stop_failed"
                            state["blocked_reason"] = {
                                "code": "hardware_unsafe_stop_failed",
                                "workload_id": active_id,
                                "unit": workload.unit,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            alert, created = _write_alert(
                                manifest,
                                state,
                                kind="hardware_unsafe_stop_failed",
                                dedupe_key=(hardware.report_sha256 or hardware.status),
                                severity="critical",
                                message=(
                                    "verified unsafe hardware was detected but "
                                    "the active workload could not be stopped"
                                ),
                                now_ns=now,
                                details=state["blocked_reason"],
                            )
                            if created:
                                alerts_to_deliver.append(alert)
                        else:
                            state["phase"] = "stopping_hardware_unsafe"
                            state["last_transition"] = {
                                "kind": "hardware_unsafe_stop_requested",
                                "workload_id": active_id,
                                "unit": workload.unit,
                                "timestamp_ns": now,
                            }
                    state["start_attempts"][active_id] = 0
                _persist_state(manifest, state, previous, now_ns=now)
                result = copy.deepcopy(state)
            elif not lock_available:
                # A queue or independently wrapped workload owns the host GPU
                # lease.  Unknown owners are never interrupted or raced.
                state["phase"] = "busy_external"
                state["active_workload_id"] = None
                state["blocked_reason"] = {
                    "code": "execution_lock_held",
                    "lock_path": str(manifest.execution_lock_path),
                }
                _persist_state(manifest, state, previous, now_ns=now)
                result = copy.deepcopy(state)
            elif (
                manifest.operator_hold_path is not None
                and manifest.operator_hold_path.exists()
            ):
                state["phase"] = "operator_hold"
                state["active_workload_id"] = None
                state["blocked_reason"] = {
                    "code": "operator_hold",
                    "path": str(manifest.operator_hold_path),
                }
                _persist_state(manifest, state, previous, now_ns=now)
                result = copy.deepcopy(state)
            else:
                result = _reconcile_stopped_host(
                    manifest,
                    state,
                    previous,
                    statuses=statuses,
                    hardware=hardware,
                    manager=manager,
                    now_ns=now,
                    alerts_to_deliver=alerts_to_deliver,
                )
    except ContinuityBusyError:
        return {
            "format": STATE_FORMAT,
            "schema_version": STATE_SCHEMA_VERSION,
            "status": "busy",
            "manifest_sha256": manifest.sha256,
            "phase": "transition_busy",
        }

    # Delivery is deliberately after durable state and any workload start.
    for alert_path in alerts_to_deliver:
        _deliver_alert(manifest, alert_path, runner=alert_runner)
    return result


def _reconcile_stopped_host(
    manifest: ContinuityManifest,
    state: dict[str, Any],
    previous: Mapping[str, object] | None,
    *,
    statuses: Mapping[str, UnitStatus],
    hardware: HardwareAssessment,
    manager: UnitManager,
    now_ns: int,
    alerts_to_deliver: list[Path],
) -> dict[str, object]:
    if state.get("productive_idle_since_ns") is None:
        state["productive_idle_since_ns"] = now_ns
    state["active_workload_id"] = None
    state["active_profile_sha256"] = None
    state["active_run_root_sha256"] = None
    handled = set(str(value) for value in state["handled_failure_ids"])
    desired_value = state.get("desired_workload_id")
    desired = (
        str(desired_value) if isinstance(desired_value, str) else manifest.primary_id
    )
    target = manifest.workload(desired)
    failure = detect_failure(
        manifest,
        target.workload_id,
        handled_failure_ids=handled,
    )
    if failure is None:
        try:
            verify_workload(manifest, target.workload_id)
        except ContinuityManifestError as exc:
            failure = _synthetic_failure(
                target,
                reason=str(exc),
                source_type="workload_integrity",
                now_ns=now_ns,
            )
    if failure is None and state.get("phase") in {"active_primary", "active_fallback"}:
        status = statuses[target.workload_id]
        failure = _synthetic_failure(
            target,
            reason=(
                f"active unit {target.unit} became inactive without durable failure"
                f" (state={status.active_state}, result={status.result},"
                f" exit={status.exec_main_status})"
            ),
            source_type="systemd_inactive_after_active",
            now_ns=now_ns,
        )
    start_attempt_count = int(state["start_attempts"].get(target.workload_id, 0))
    previous_start_request = state["last_start_requested_ns"].get(target.workload_id)
    start_retry_due = not isinstance(previous_start_request, int) or (
        now_ns - previous_start_request
        >= int(manifest.start_retry_seconds * 1_000_000_000)
    )
    if (
        failure is None
        and state.get("phase")
        in {"starting_primary", "starting_fallback", "start_failed"}
        and start_attempt_count >= manifest.maximum_start_attempts
        and start_retry_due
    ):
        failure = _synthetic_failure(
            target,
            reason=(
                "systemd start retry budget exhausted after "
                f"{start_attempt_count} attempt(s)"
            ),
            source_type="start_budget_exhausted",
            now_ns=now_ns,
        )

    if failure is not None and failure.failure_id not in handled:
        if failure.domain == "handoff":
            state["handled_failure_ids"].append(failure.failure_id)
            state["last_handoff"] = failure.as_dict()
            state["desired_workload_id"] = None
            state["last_transition"] = {
                "kind": "continuity_handoff",
                "workload_id": target.workload_id,
                "failure_id": failure.failure_id,
                "timestamp_ns": now_ns,
            }
            alert, created = _write_alert(
                manifest,
                state,
                kind="continuity_handoff_received",
                dedupe_key=failure.failure_id,
                severity="info",
                message="completed workload requested a safe continuity handoff",
                now_ns=now_ns,
                details={
                    "workload_id": target.workload_id,
                    "handoff": failure.as_dict(),
                },
            )
            if created:
                alerts_to_deliver.append(alert)
        else:
            quarantine_path = _quarantine(
                manifest,
                state,
                target,
                failure,
                now_ns=now_ns,
            )
            alert, created = _write_alert(
                manifest,
                state,
                kind="workload_quarantined",
                dedupe_key=failure.failure_id,
                severity="error",
                message=f"quarantined failed workload {target.workload_id}",
                now_ns=now_ns,
                details={
                    "workload_id": target.workload_id,
                    "failure": failure.as_dict(),
                    "quarantine_path": str(quarantine_path),
                },
            )
            if created:
                alerts_to_deliver.append(alert)

    quarantined = set(str(value) for value in state["quarantined_workloads"])
    desired_value = state.get("desired_workload_id")
    desired = str(desired_value) if isinstance(desired_value, str) else None
    if desired is None or desired in quarantined:
        selected, rejected = select_last_known_good(
            manifest,
            excluded_workload_ids=quarantined,
        )
        if selected is None:
            state["phase"] = "blocked_no_verified_lkg"
            state["blocked_reason"] = {
                "code": "no_verified_last_known_good",
                "rejected": rejected,
                "quarantined_workloads": sorted(quarantined),
            }
            alert, created = _write_alert(
                manifest,
                state,
                kind="fallback_unavailable",
                dedupe_key=_digest_bytes(_canonical_bytes(state["blocked_reason"])),
                severity="critical",
                message="no verified last-known-good workload is available",
                now_ns=now_ns,
                details=state["blocked_reason"],
            )
            if created:
                alerts_to_deliver.append(alert)
            _persist_state(manifest, state, previous, now_ns=now_ns)
            return copy.deepcopy(state)
        desired = selected.workload_id
        state["desired_workload_id"] = desired
        state["selected_lkg_workload_id"] = desired
        state["last_transition"] = {
            "kind": "fallback_selected",
            "workload_id": desired,
            "timestamp_ns": now_ns,
            "rejected_lkg": rejected,
        }

    target = manifest.workload(desired)
    try:
        verify_workload(manifest, target.workload_id)
    except ContinuityManifestError as exc:
        # A previously selected candidate may drift between timer ticks.
        drift = _synthetic_failure(
            target,
            reason=str(exc),
            source_type="workload_integrity",
            now_ns=now_ns,
        )
        _quarantine(manifest, state, target, drift, now_ns=now_ns)
        state["phase"] = "blocked_lkg_integrity"
        state["blocked_reason"] = {
            "code": "selected_workload_integrity_failed",
            "workload_id": target.workload_id,
            "reason": str(exc),
        }
        alert, created = _write_alert(
            manifest,
            state,
            kind="fallback_integrity_blocked",
            dedupe_key=drift.failure_id,
            severity="critical",
            message=f"verified workload {target.workload_id} no longer matches hashes",
            now_ns=now_ns,
            details=state["blocked_reason"],
        )
        if created:
            alerts_to_deliver.append(alert)
        _persist_state(manifest, state, previous, now_ns=now_ns)
        return copy.deepcopy(state)

    if not hardware.safe:
        state["phase"] = (
            "blocked_hardware_unsafe"
            if hardware.status == "unsafe"
            else "blocked_hardware_unavailable"
        )
        state["blocked_reason"] = {
            "code": state["phase"],
            "hardware": hardware.as_dict(),
            "selected_workload_id": target.workload_id,
        }
        alert, created = _write_alert(
            manifest,
            state,
            kind="hardware_blocked_fallback",
            dedupe_key=hardware.report_sha256 or hardware.status,
            severity="critical",
            message="hardware safety is not proven; workload start is blocked",
            now_ns=now_ns,
            details=state["blocked_reason"],
        )
        if created:
            alerts_to_deliver.append(alert)
        _persist_state(manifest, state, previous, now_ns=now_ns)
        return copy.deepcopy(state)

    if not manifest.automatic_start:
        state["phase"] = "ready_manual_start"
        state["blocked_reason"] = None
        _persist_state(manifest, state, previous, now_ns=now_ns)
        return copy.deepcopy(state)

    attempts = int(state["start_attempts"].get(target.workload_id, 0))
    last_request = state["last_start_requested_ns"].get(target.workload_id)
    if isinstance(last_request, int) and now_ns - last_request < int(
        manifest.start_retry_seconds * 1_000_000_000
    ):
        state["phase"] = (
            "starting_fallback" if target.role == "fallback" else "starting_primary"
        )
        state["blocked_reason"] = None
        _persist_state(manifest, state, previous, now_ns=now_ns)
        return copy.deepcopy(state)
    if attempts >= manifest.maximum_start_attempts:
        exhausted = _synthetic_failure(
            target,
            reason=f"start retry budget exhausted after {attempts} attempts",
            source_type="start_budget_exhausted",
            now_ns=now_ns,
        )
        _quarantine(manifest, state, target, exhausted, now_ns=now_ns)
        state["phase"] = "blocked_start_budget"
        state["blocked_reason"] = {
            "code": "start_budget_exhausted",
            "workload_id": target.workload_id,
            "attempts": attempts,
        }
        alert, created = _write_alert(
            manifest,
            state,
            kind="start_budget_exhausted",
            dedupe_key=exhausted.failure_id,
            severity="critical",
            message=f"start budget exhausted for {target.workload_id}",
            now_ns=now_ns,
            details=state["blocked_reason"],
        )
        if created:
            alerts_to_deliver.append(alert)
        _persist_state(manifest, state, previous, now_ns=now_ns)
        return copy.deepcopy(state)

    state["start_attempts"][target.workload_id] = attempts + 1
    state["last_start_requested_ns"][target.workload_id] = now_ns
    if target.role == "fallback":
        state["fallback_attempts"] = int(state["fallback_attempts"]) + 1
    state["phase"] = (
        "starting_fallback" if target.role == "fallback" else "starting_primary"
    )
    state["blocked_reason"] = None
    state["last_transition"] = {
        "kind": "start_requested",
        "workload_id": target.workload_id,
        "unit": target.unit,
        "timestamp_ns": now_ns,
    }
    # Persist intent before asking systemd to start the unit.
    _persist_state(manifest, state, previous, now_ns=now_ns)
    persisted = copy.deepcopy(state)
    try:
        manager.start(target.unit)
    except Exception as exc:
        state["phase"] = "start_failed"
        state["blocked_reason"] = {
            "code": "systemd_start_failed",
            "workload_id": target.workload_id,
            "unit": target.unit,
            "error": f"{type(exc).__name__}: {exc}",
        }
        alert, created = _write_alert(
            manifest,
            state,
            kind="workload_start_failed",
            dedupe_key=f"{target.workload_id}:{attempts + 1}",
            severity="error",
            message=f"failed to start {target.unit}",
            now_ns=now_ns,
            details=state["blocked_reason"],
        )
        if created:
            alerts_to_deliver.append(alert)
        _persist_state(manifest, state, persisted, now_ns=now_ns)
        return copy.deepcopy(state)

    observed = manager.status(target.unit)
    if observed.active:
        state["phase"] = (
            "active_fallback" if target.role == "fallback" else "active_primary"
        )
        state["active_workload_id"] = target.workload_id
        state["desired_workload_id"] = target.workload_id
        state["active_profile_sha256"] = target.profile_sha256
        state["active_run_root_sha256"] = target.run_root_sha256
        state["productive_idle_since_ns"] = None
        state["start_attempts"][target.workload_id] = 0
    if target.role == "fallback":
        alert, created = _write_alert(
            manifest,
            state,
            kind="fallback_started",
            dedupe_key=(
                str(state.get("last_failure", {}).get("failure_id"))
                if isinstance(state.get("last_failure"), dict)
                else target.workload_id
            ),
            severity="warning",
            message=f"started verified LKG workload {target.workload_id}",
            now_ns=now_ns,
            details={
                "workload": target.state_identity(),
                "unit_status": observed.as_dict(),
            },
        )
        if created:
            alerts_to_deliver.append(alert)
    _persist_state(manifest, state, persisted, now_ns=now_ns)
    return copy.deepcopy(state)


def _exec_workload(
    executable: str,
    command: list[str],
    environment: dict[str, str],
) -> Any:
    return os.execvpe(executable, command, environment)


def run_locked_workload(
    manifest_path: str | Path,
    workload_id: str,
    *,
    orchestrator: str | Path,
    exec_fn: Callable[[str, list[str], dict[str, str]], Any] = _exec_workload,
    now_ns: int | None = None,
) -> Any:
    """Verify and exec one coordinator while holding the host GPU lock."""

    manifest = load_continuity_manifest(manifest_path)
    _pin_manifest(manifest)
    verification = verify_workload(manifest, workload_id)
    hardware = assess_hardware(
        manifest,
        now_ns=time.time_ns() if now_ns is None else now_ns,
    )
    if not hardware.safe:
        raise ContinuityError(
            f"hardware safety is not proven ({hardware.status}): "
            + "; ".join(hardware.reasons)
        )
    requested_orchestrator = Path(orchestrator).expanduser().resolve()
    if verification.workload.runtime_orchestrator_path != requested_orchestrator:
        raise ContinuityManifestError(
            f"workload {workload_id} requested an unpinned orchestrator"
        )
    executable = str(requested_orchestrator)
    command = [executable, "--config", str(verification.workload.profile_path)]
    with nonblocking_host_lock(
        manifest.execution_lock_path,
        owner="GPU workload",
        mode=0o660,
    ) as descriptor:
        live = live_coordinator_locks(manifest)
        if live:
            raise ContinuitySplitBrainError(
                "refusing GPU launch while coordinator locks are live: "
                + ", ".join(sorted(live))
            )
        if not manifest.state_path.is_file():
            raise ContinuityStateError(
                "continuity state must select a workload before it can run"
            )
        if (
            manifest.operator_hold_path is not None
            and manifest.operator_hold_path.exists()
        ):
            raise ContinuityStateError("operator hold blocks workload execution")
        selected_state, _ = _load_state(
            manifest,
            now_ns=time.time_ns() if now_ns is None else now_ns,
        )
        if selected_state.get("desired_workload_id") != workload_id:
            raise ContinuityStateError(
                f"continuity state does not select workload {workload_id}"
            )
        if workload_id in selected_state.get("quarantined_workloads", []):
            raise ContinuityStateError(
                f"continuity state quarantines workload {workload_id}"
            )
        # Re-verify after obtaining the execution lease to close the normal
        # preflight-to-exec drift window.
        verification = verify_workload(manifest, workload_id)
        hardware = assess_hardware(
            manifest,
            now_ns=time.time_ns() if now_ns is None else now_ns,
        )
        if not hardware.safe:
            raise ContinuityError(
                f"hardware safety changed before exec ({hardware.status}): "
                + "; ".join(hardware.reasons)
            )
        owner = {
            "schema_version": 1,
            "pid": os.getpid(),
            "workload_id": workload_id,
            "unit": verification.workload.unit,
            "profile_sha256": verification.workload.profile_sha256,
            "run_root_sha256": verification.workload.run_root_sha256,
            "acquired_ns": time.time_ns() if now_ns is None else now_ns,
        }
        data = _canonical_bytes(owner) + b"\n"
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.write(descriptor, data) != len(data):
            raise OSError("short execution-lock owner write")
        os.fsync(descriptor)
        os.set_inheritable(descriptor, True)
        environment = {
            **os.environ,
            "STARTRAIN_CONTINUITY_MANIFEST": str(manifest.path),
            "STARTRAIN_CONTINUITY_WORKLOAD_ID": workload_id,
            "STARTRAIN_CONTINUITY_LOCK_FD": str(descriptor),
        }
        return exec_fn(executable, command, environment)
