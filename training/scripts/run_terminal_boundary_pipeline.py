#!/usr/bin/env python3
"""Cut over a terminal training run into a pinned Elo calibration queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from startrain.runtime import atomic_json

SCHEMA_VERSION = 1
POLICY_REPORT = "startrain-terminal-boundary-policy"
STATE_REPORT = "startrain-terminal-boundary-state"
ACTIVATION_REPORT = "startrain-terminal-boundary-queue-activation"
FALLBACK_REPORT = "startrain-continuity-handoff-request"
TERMINAL_DECISIONS = frozenset(
    {
        "promote",
        "reject",
        "reject_ring_regression",
        "reject_max_pairs",
        "plateau_reset",
        "plateau_recover",
    }
)
CONTROL_BOUNDARY_DECISIONS = frozenset({"plateau_reset", "plateau_recover"})
TERMINAL_STATE_STATUSES = frozenset({"blocked", "completed", "failed"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 64 * 1024 * 1024


class TerminalBoundaryError(RuntimeError):
    """A fail-closed terminal-boundary validation or execution error."""


class TerminalBoundaryBusyError(TerminalBoundaryError):
    """Another terminal-boundary runner owns the durable state."""


class TerminalBoundaryManifestError(TerminalBoundaryError):
    """The immutable terminal-boundary policy is malformed or changed."""


class TerminalBoundaryExecutionError(TerminalBoundaryError):
    """A persisted terminal-boundary side effect failed."""


WarmStarter = Callable[..., dict[str, object]]


class TerminalBoundaryAdapters(Protocol):
    """Injected host and deployment operations used by the durable saga."""

    def inspect_source(self, policy: Mapping[str, object]) -> Mapping[str, object]: ...

    def inspect_hardware(
        self, policy: Mapping[str, object]
    ) -> Mapping[str, object]: ...

    def inspect_arena_pause(
        self, policy: Mapping[str, object]
    ) -> Mapping[str, object]: ...

    def place_operator_hold(
        self, path: Path, document: Mapping[str, object]
    ) -> Mapping[str, object]: ...

    def final_replay_backup(
        self, policy: Mapping[str, object]
    ) -> Mapping[str, object]: ...

    def disaster_snapshot(
        self,
        policy: Mapping[str, object],
        required_after_ns: int,
    ) -> Mapping[str, object]: ...

    def stop_source(self, policy: Mapping[str, object]) -> Mapping[str, object]: ...

    def prove_source_release(
        self,
        policy: Mapping[str, object],
        source_evidence: Mapping[str, object],
        stop_evidence: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def export_champion(
        self,
        policy: Mapping[str, object],
        winner_snapshot: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def prepare_calibration(
        self,
        policy: Mapping[str, object],
        winner_snapshot: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def run_frozen_calibration(
        self,
        policy: Mapping[str, object],
        plan: Mapping[str, object],
        winner_snapshot: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def fork_calibration(
        self,
        policy: Mapping[str, object],
        plan: Mapping[str, object],
        treatment: Mapping[str, object],
        winner_snapshot: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def warm_start(
        self,
        run_root: Path,
        profile: Path,
        *,
        prepare_only: bool,
    ) -> Mapping[str, object]: ...

    def prepare_runtime_ownership(
        self,
        policy: Mapping[str, object],
        plan: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def generate_queue_manifest(
        self,
        policy: Mapping[str, object],
        plan: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def verify_queue_manifest(
        self,
        policy: Mapping[str, object],
        manifest: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def launch_queue(
        self,
        policy: Mapping[str, object],
        activation_manifest: Path,
    ) -> Mapping[str, object]: ...

    def request_continuity_fallback(
        self,
        policy: Mapping[str, object],
        request: Mapping[str, object],
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    path: Path
    sha256: str
    data: bytes
    raw: dict[str, Any]
    state_path: Path
    pinned_path: Path


def _canonical_bytes(document: object) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_fence(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _read_json_bytes(path: Path, *, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TerminalBoundaryManifestError(
            f"cannot read {name} {path}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TerminalBoundaryManifestError(f"{name} is not a regular file: {path}")
        if before.st_size > _MAX_JSON_BYTES:
            raise TerminalBoundaryManifestError(f"{name} is too large: {path}")
        blocks = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_fence(before) != _stat_fence(after):
        raise TerminalBoundaryManifestError(f"{name} changed while being read: {path}")
    return b"".join(blocks)


def _hash_regular_file(path: Path, *, name: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TerminalBoundaryManifestError(
            f"cannot hash {name} {path}: {error}"
        ) from error
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TerminalBoundaryManifestError(f"{name} is not a regular file: {path}")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_fence(before) != _stat_fence(after) or total != after.st_size:
        raise TerminalBoundaryManifestError(
            f"{name} changed while being hashed: {path}"
        )
    return digest.hexdigest(), total


def _json_object(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalBoundaryManifestError(
            f"{name} is not valid JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise TerminalBoundaryManifestError(f"{name} must contain a JSON object")
    return document


def _read_json_with_digest(
    path: Path,
    *,
    name: str,
) -> tuple[dict[str, Any], str, bytes]:
    data = _read_json_bytes(path, name=name)
    return _json_object(data, name=name), _sha256_bytes(data), data


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TerminalBoundaryManifestError(f"{name} must be an object")
    return value


def _sequence(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TerminalBoundaryManifestError(f"{name} must be a list")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TerminalBoundaryManifestError(f"{name} must be a normalized string")
    return value


def _identifier(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if _IDENTIFIER.fullmatch(result) is None:
        raise TerminalBoundaryManifestError(f"{name} is not a safe identifier")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TerminalBoundaryManifestError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TerminalBoundaryManifestError(f"{name} must be a nonnegative integer")
    return value


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TerminalBoundaryManifestError(f"{name} must be positive")
    converted = float(value)
    if converted <= 0 or not (converted < float("inf")):
        raise TerminalBoundaryManifestError(f"{name} must be finite and positive")
    return converted


def _digest(value: object, *, name: str) -> str:
    result = _string(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise TerminalBoundaryManifestError(f"{name} must be a SHA-256 digest")
    return result


def _absolute_path(value: object, *, name: str) -> Path:
    text = _string(value, name=name)
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise TerminalBoundaryManifestError(f"{name} must be an absolute path")
    return path.resolve(strict=False)


def _pinned_artifact(value: object, *, name: str) -> tuple[Path, str]:
    artifact = _mapping(value, name=name)
    path = _absolute_path(artifact.get("path"), name=f"{name} path")
    expected = _digest(artifact.get("sha256"), name=f"{name} SHA-256")
    if not path.is_file() or path.is_symlink():
        raise TerminalBoundaryManifestError(f"{name} is missing or unsafe: {path}")
    if _sha256(path) != expected:
        raise TerminalBoundaryManifestError(f"{name} digest changed: {path}")
    return path, expected


_RUNTIME_ARTIFACTS = (
    "release_manifest",
    "terminal_runner",
    "calibration_runner",
    "calibration_queue",
    "calibration_comparator",
    "ablation_preparer",
    "ablation_forker",
    "warm_starter",
    "runtime_module",
    "training_module",
    "learner_module",
    "checkpoint_module",
    "continuity_module",
    "snapshot_module",
    "replay_backup",
    "disaster_recovery",
    "queue_generator",
    "queue_runner",
    "elo_comparator",
)


def _verify_runtime_pins(document: Mapping[str, object]) -> None:
    runtime = _mapping(document.get("runtime"), name="terminal-boundary runtime")
    training_dir = _absolute_path(
        runtime.get("training_dir"),
        name="terminal-boundary training directory",
    )
    if not training_dir.is_dir() or training_dir.is_symlink():
        raise TerminalBoundaryManifestError(
            "terminal-boundary training directory is missing or unsafe"
        )
    for field in _RUNTIME_ARTIFACTS:
        path, _digest_value = _pinned_artifact(
            runtime.get(field),
            name=f"terminal-boundary runtime {field}",
        )
        allowed_root = (
            training_dir.parent if field == "release_manifest" else training_dir
        )
        if path != allowed_root and allowed_root not in path.parents:
            raise TerminalBoundaryManifestError(
                f"terminal-boundary runtime {field} escaped its immutable release"
            )


def _validate_policy(document: dict[str, Any], policy_path: Path) -> LoadedPolicy:
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("report") != POLICY_REPORT
    ):
        raise TerminalBoundaryManifestError("unsupported terminal-boundary policy")
    policy_id = _identifier(document.get("policy_id"), name="policy ID")
    policy_version = _positive_int(
        document.get("policy_version"), name="policy version"
    )
    state_path = _absolute_path(document.get("state_path"), name="state path")
    pinned_path = state_path.with_name(
        f"{policy_id}-v{policy_version}-policy.pinned.json"
    )
    _verify_runtime_pins(document)

    arm = _mapping(document.get("arm"), name="terminal arm")
    _digest(
        arm.get("promotion_status_sha256"),
        name="armed promotion-status SHA-256",
    )
    _positive_int(
        arm.get("promotion_status_updated_ns"),
        name="armed promotion-status updated_ns",
    )

    source = _mapping(document.get("source"), name="source")
    source_root = _absolute_path(source.get("run_root"), name="source run root")
    if not source_root.is_dir() or source_root.is_symlink():
        raise TerminalBoundaryManifestError("source run root is missing or unsafe")
    identity = _mapping(source.get("run_identity"), name="source run identity")
    _identifier(identity.get("run_id"), name="source run ID")
    _identifier(
        identity.get("generation_family"),
        name="source generation family",
    )
    _positive_int(identity.get("created_ns"), name="source created_ns")
    profile_path, _ = _pinned_artifact(source.get("profile"), name="source profile")
    _pinned_artifact(source.get("unit"), name="source unit")
    unit = _mapping(source.get("unit"), name="source unit")
    unit_name = _string(unit.get("name"), name="source unit name")
    if not unit_name.endswith(".service") or "/" in unit_name:
        raise TerminalBoundaryManifestError("source unit name must name one service")
    for field, label in (
        ("promotion_status", "promotion status"),
        ("candidate_pointer", "candidate pointer"),
        ("champion_pointer", "champion pointer"),
        ("arena_root", "arena root"),
        ("coordinator_status", "coordinator status"),
        ("coordinator_lock", "coordinator lock"),
        ("arena_pause_request", "arena pause request"),
        ("hardware_report", "hardware report"),
    ):
        _absolute_path(source.get(field), name=label)
    arena_root = _absolute_path(source.get("arena_root"), name="arena root")
    if (
        arena_root != (source_root / "arena").resolve()
        or not arena_root.is_dir()
        or arena_root.is_symlink()
    ):
        raise TerminalBoundaryManifestError(
            "source arena root must be the run's real arena directory"
        )
    _positive_number(
        source.get("hardware_max_age_seconds"),
        name="hardware maximum age",
    )
    _positive_number(source.get("stop_timeout_seconds"), name="stop timeout")
    gpu_ids = _sequence(source.get("gpu_ids"), name="source GPU IDs")
    if any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids):
        raise TerminalBoundaryManifestError(
            "source GPU IDs must be nonnegative integers"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise TerminalBoundaryManifestError("source GPU IDs contain duplicates")

    hold_path = _absolute_path(
        document.get("operator_hold_path"), name="operator hold path"
    )
    backup = _mapping(document.get("backup"), name="backup policy")
    _positive_int(backup.get("replay_retain"), name="replay backup retention")
    _positive_int(
        backup.get("replay_max_total_bytes"),
        name="replay backup maximum bytes",
    )
    disaster_root = _absolute_path(
        backup.get("disaster_backup_root"),
        name="disaster backup root",
    )
    disaster_mount = _absolute_path(
        backup.get("disaster_backup_mount"),
        name="disaster backup mount",
    )
    disaster_verification = backup.get("disaster_snapshot_verification")
    if disaster_verification != "lambda_attached":
        raise TerminalBoundaryManifestError(
            "disaster snapshot verification must be lambda_attached"
        )
    try:
        disaster_root.relative_to(disaster_mount)
    except ValueError as error:
        raise TerminalBoundaryManifestError(
            "disaster backup root must remain under the backup mount"
        ) from error
    if disaster_root == disaster_mount:
        raise TerminalBoundaryManifestError(
            "disaster backup root must be a child of the backup mount"
        )

    snapshot = _mapping(document.get("snapshot"), name="champion snapshot")
    snapshot_destination = _absolute_path(
        snapshot.get("destination"), name="champion snapshot destination"
    )
    snapshot_pin = _absolute_path(
        snapshot.get("pin_path"), name="champion snapshot pin"
    )

    calibration = _mapping(document.get("calibration"), name="calibration")
    base_config, _ = _pinned_artifact(
        calibration.get("base_config"), name="calibration base config"
    )
    output_dir = _absolute_path(
        calibration.get("output_dir"), name="calibration output directory"
    )
    run_root_parent = _absolute_path(
        calibration.get("run_root_parent"),
        name="calibration run-root parent",
    )
    calibration_run_id = _identifier(
        calibration.get("run_id"), name="calibration run ID"
    )
    _identifier(calibration.get("runtime_user"), name="calibration runtime user")
    _identifier(calibration.get("runtime_group"), name="calibration runtime group")
    if calibration_run_id != identity["run_id"]:
        raise TerminalBoundaryManifestError(
            "calibration run ID differs from the source run identity"
        )
    _identifier(calibration.get("prefix"), name="calibration prefix")
    _positive_int(calibration.get("seed"), name="calibration seed")
    _positive_number(
        calibration.get("wall_budget_hours"),
        name="calibration wall budget",
    )
    _positive_int(calibration.get("leaf_budget"), name="calibration leaf budget")
    guard_floor = calibration.get("guard_floor_elo")
    if (
        isinstance(guard_floor, bool)
        or not isinstance(guard_floor, int | float)
        or float(guard_floor) >= 0
    ):
        raise TerminalBoundaryManifestError("calibration guard floor must be negative")
    treatments = _sequence(calibration.get("treatments"), name="calibration treatments")
    if not treatments or any(
        not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None
        for value in treatments
    ):
        raise TerminalBoundaryManifestError(
            "calibration treatments must be safe identifiers"
        )
    if len(set(treatments)) != len(treatments):
        raise TerminalBoundaryManifestError("calibration treatments contain duplicates")
    suite = calibration.get("suite")
    if suite is not None:
        _identifier(suite, name="calibration suite")
    raw_guard_rings = calibration.get("guard_rings")
    if raw_guard_rings is not None:
        guard_rings = _sequence(raw_guard_rings, name="calibration guard rings")
        if any(type(ring) is not int or ring <= 0 for ring in guard_rings):
            raise TerminalBoundaryManifestError(
                "calibration guard rings must be positive integers"
            )
    frozen = _mapping(
        calibration.get("frozen_replay"),
        name="frozen-replay calibration",
    )
    frozen_output_root = _absolute_path(
        frozen.get("output_root"),
        name="frozen-replay output root",
    )
    screen_plan_path = _absolute_path(
        frozen.get("screen_plan_path"),
        name="optimizer screen plan",
    )
    _positive_int(frozen.get("steps"), name="frozen-replay steps")
    _string(frozen.get("device"), name="frozen-replay device")
    budget_h100_hours = _positive_number(
        frozen.get("budget_h100_hours"),
        name="frozen-replay H100-hour budget",
    )
    if budget_h100_hours > 2.0:
        raise TerminalBoundaryManifestError(
            "frozen-replay budget may not exceed 2 H100-hours per arm"
        )
    batch_size = frozen.get("batch_size")
    if batch_size is not None:
        _positive_int(batch_size, name="frozen-replay batch size")
    _positive_int(
        frozen.get("evaluation_batch_size"),
        name="frozen-replay evaluation batch size",
    )
    _positive_int(
        frozen.get("max_samples"),
        name="frozen-replay maximum samples",
    )
    holdout_fraction = _positive_number(
        frozen.get("holdout_fraction"),
        name="frozen-replay holdout fraction",
    )
    if holdout_fraction >= 1:
        raise TerminalBoundaryManifestError(
            "frozen-replay holdout fraction must be less than one"
        )
    _positive_int(
        frozen.get("checkpoint_interval"),
        name="frozen-replay checkpoint interval",
    )
    _positive_number(
        frozen.get("screen_wall_budget_hours"),
        name="optimizer screen wall budget",
    )
    _positive_int(
        frozen.get("screen_leaf_budget"),
        name="optimizer screen leaf budget",
    )

    queue = _mapping(document.get("queue"), name="queue activation")
    training_dir = _absolute_path(
        queue.get("training_dir"), name="queue training directory"
    )
    if not training_dir.is_dir() or training_dir.is_symlink():
        raise TerminalBoundaryManifestError(
            "queue training directory is missing or unsafe"
        )
    deployment_manifest = _absolute_path(
        queue.get("deployment_manifest"), name="queue deployment manifest"
    )
    activation_manifest = _absolute_path(
        queue.get("activation_manifest"), name="queue activation manifest"
    )
    queue_unit_path, _ = _pinned_artifact(queue.get("queue_unit"), name="queue unit")
    queue_unit = _mapping(queue.get("queue_unit"), name="queue unit")
    queue_unit_name = _string(queue_unit.get("name"), name="queue unit name")
    if not queue_unit_name.endswith(".service") or "/" in queue_unit_name:
        raise TerminalBoundaryManifestError("queue unit name must name one service")
    finalize_path, _ = _pinned_artifact(
        queue.get("finalize_unit"), name="queue finalize unit"
    )
    environment_path, _ = _pinned_artifact(
        queue.get("environment"), name="queue environment"
    )
    for field, label in (
        ("state_path", "queue state"),
        ("comparison_output", "queue comparison output"),
        ("continuity_handoff_output", "queue continuity handoff"),
        ("execution_lock_path", "queue execution lock"),
    ):
        _absolute_path(queue.get(field), name=label)
    queue_owned_paths = [
        _absolute_path(queue.get(field), name=f"queue {field}")
        for field in (
            "deployment_manifest",
            "state_path",
            "comparison_output",
            "continuity_handoff_output",
        )
    ]
    queue_parent = queue_owned_paths[0].parent
    if (
        any(path.parent != queue_parent for path in queue_owned_paths)
        or queue_parent == state_path.parent
    ):
        raise TerminalBoundaryManifestError(
            "queue-owned outputs must share a dedicated directory separate from "
            "terminal-boundary state"
        )
    _string(queue.get("source_commit"), name="queue source commit")
    _string(queue.get("orchestrator"), name="queue orchestrator")
    _positive_number(queue.get("poll_seconds"), name="queue polling interval")
    _positive_number(
        queue.get("launch_timeout_seconds"),
        name="queue launch timeout",
    )
    _nonnegative_int(
        queue.get("max_transient_retries"),
        name="queue maximum transient retries",
    )
    retry_delay = queue.get("retry_delay_seconds")
    if (
        isinstance(retry_delay, bool)
        or not isinstance(retry_delay, int | float)
        or float(retry_delay) < 0
    ):
        raise TerminalBoundaryManifestError("queue retry delay must be nonnegative")
    if type(queue.get("continue_after_fatal")) is not bool:
        raise TerminalBoundaryManifestError(
            "queue continue-after-fatal must be boolean"
        )
    _positive_int(queue.get("provisioned_gpus"), name="queue provisioned GPUs")

    fallback = _mapping(document.get("fallback"), name="continuity fallback")
    fallback_handoff = _absolute_path(
        fallback.get("handoff_path"), name="fallback handoff"
    )
    _pinned_artifact(
        fallback.get("continuity_manifest"),
        name="continuity manifest",
    )

    outputs = (
        state_path,
        pinned_path,
        hold_path,
        snapshot_destination,
        snapshot_pin,
        output_dir,
        frozen_output_root,
        screen_plan_path,
        deployment_manifest,
        activation_manifest,
        _absolute_path(queue["state_path"], name="queue state"),
        _absolute_path(queue["comparison_output"], name="queue comparison output"),
        _absolute_path(
            queue["continuity_handoff_output"],
            name="queue continuity handoff",
        ),
        fallback_handoff,
    )
    if len(set(outputs)) != len(outputs):
        raise TerminalBoundaryManifestError(
            "terminal-boundary state, holds, snapshots, calibration, queue, and "
            "fallback outputs must use distinct paths"
        )
    for output in (
        state_path,
        pinned_path,
        hold_path,
        snapshot_destination,
        snapshot_pin,
        output_dir,
        frozen_output_root,
        screen_plan_path,
        deployment_manifest,
        activation_manifest,
        fallback_handoff,
    ):
        if output == source_root or source_root in output.parents:
            raise TerminalBoundaryManifestError(
                f"terminal-boundary output may not mutate the source root: {output}"
            )
    if source_root == disaster_root or source_root in disaster_root.parents:
        raise TerminalBoundaryManifestError(
            "disaster backup root may not be inside the source run root"
        )
    if source_root == run_root_parent or source_root in run_root_parent.parents:
        raise TerminalBoundaryManifestError(
            "calibration roots may not be inside the source run root"
        )
    del profile_path, base_config, queue_unit_path, finalize_path, environment_path
    data = _read_json_bytes(policy_path, name="terminal-boundary policy")
    return LoadedPolicy(
        path=policy_path,
        sha256=_sha256_bytes(data),
        data=data,
        raw=document,
        state_path=state_path,
        pinned_path=pinned_path,
    )


def load_terminal_boundary_policy(path: str | Path) -> LoadedPolicy:
    policy_path = Path(path).expanduser().resolve()
    data = _read_json_bytes(policy_path, name="terminal-boundary policy")
    document = _json_object(data, name="terminal-boundary policy")
    loaded = _validate_policy(document, policy_path)
    if loaded.data != data:
        raise TerminalBoundaryManifestError(
            "terminal-boundary policy changed during validation"
        )
    return loaded


def verify_terminal_boundary_policy(path: str | Path) -> dict[str, object]:
    policy = load_terminal_boundary_policy(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "report": POLICY_REPORT,
        "status": "verified",
        "policy": str(policy.path),
        "policy_sha256": policy.sha256,
        "policy_id": policy.raw["policy_id"],
        "policy_version": policy.raw["policy_version"],
        "state_path": str(policy.state_path),
        "pinned_path": str(policy.pinned_path),
    }


def _verify_policy_unchanged(policy: LoadedPolicy) -> None:
    current = _read_json_bytes(policy.path, name="terminal-boundary policy")
    if current != policy.data:
        raise TerminalBoundaryManifestError(
            "terminal-boundary policy changed after loading"
        )
    if policy.pinned_path.is_file():
        pinned = _read_json_bytes(policy.pinned_path, name="pinned policy")
        if pinned != policy.data:
            raise TerminalBoundaryManifestError(
                "atomically pinned policy differs from the source policy"
            )


def _write_new_or_verify(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError:
        if (
            _read_json_bytes(path, name=f"existing immutable artifact {path}")
            != data
        ):
            raise TerminalBoundaryManifestError(
                f"immutable artifact already exists with different content: {path}"
            ) from None
        return
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_immutable_json(
    path: Path,
    document: Mapping[str, object],
    *,
    mode: int = 0o600,
) -> dict[str, object]:
    data = _canonical_bytes(document)
    _write_new_or_verify(path, data, mode=mode)
    return {
        "path": str(path),
        "sha256": _sha256_bytes(data),
        "bytes": len(data),
    }


def _release_owned_operator_hold(
    path: Path,
    expected: Mapping[str, object],
) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "status": "absent"}
    document, digest_value, _ = _read_json_with_digest(
        path,
        name="terminal-boundary operator hold",
    )
    if document != dict(expected):
        raise TerminalBoundaryExecutionError(
            "operator hold is not owned by this terminal-boundary transition"
        )
    path.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "path": str(path),
        "status": "released",
        "sha256": digest_value,
    }


@contextmanager
def _pipeline_lock(state_path: Path) -> Iterator[None]:
    lock_path = Path(f"{state_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TerminalBoundaryBusyError(
                f"another terminal-boundary runner owns {lock_path}"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _initial_state(policy: LoadedPolicy) -> dict[str, Any]:
    now = time.time_ns()
    return {
        "schema_version": SCHEMA_VERSION,
        "report": STATE_REPORT,
        "status": "initializing",
        "phase": "pin_policy",
        "policy": str(policy.path),
        "policy_sha256": policy.sha256,
        "policy_id": policy.raw["policy_id"],
        "policy_version": policy.raw["policy_version"],
        "pinned_policy": str(policy.pinned_path),
        "created_ns": now,
        "updated_ns": now,
        "accepted_terminal": None,
        "source_preflight": None,
        "steps": {},
        "failures": [],
        "fallback": None,
        "activation_document": None,
    }


def _read_existing_state(policy: LoadedPolicy) -> dict[str, Any] | None:
    if not policy.state_path.exists():
        return None
    document, _digest_value, _ = _read_json_with_digest(
        policy.state_path,
        name="terminal-boundary state",
    )
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("report") != STATE_REPORT
        or document.get("policy") != str(policy.path)
        or document.get("policy_sha256") != policy.sha256
        or document.get("policy_id") != policy.raw["policy_id"]
        or document.get("policy_version") != policy.raw["policy_version"]
        or document.get("pinned_policy") != str(policy.pinned_path)
        or not isinstance(document.get("steps"), dict)
        or not isinstance(document.get("failures"), list)
    ):
        raise TerminalBoundaryManifestError(
            "durable state does not match the immutable policy"
        )
    return document


def _load_state(policy: LoadedPolicy) -> dict[str, Any]:
    state = _read_existing_state(policy)
    if state is not None:
        return state
    state = _initial_state(policy)
    _persist_state(policy, state)
    return state


def probe_terminal_boundary_policy(path: str | Path) -> dict[str, object]:
    """Report whether a policy has runnable work without creating durable state."""
    policy = load_terminal_boundary_policy(path)
    state = _read_existing_state(policy)
    state_status = state.get("status") if state is not None else None
    terminal = state_status in TERMINAL_STATE_STATUSES
    return {
        "schema_version": SCHEMA_VERSION,
        "report": "startrain-terminal-boundary-probe",
        "status": "terminal" if terminal else "runnable",
        "should_run": not terminal,
        "reason": (
            f"state_{state_status}"
            if terminal
            else "state_absent"
            if state is None
            else f"state_{state_status}"
        ),
        "policy": str(policy.path),
        "policy_sha256": policy.sha256,
        "policy_id": policy.raw["policy_id"],
        "policy_version": policy.raw["policy_version"],
        "state_path": str(policy.state_path),
        "state_status": state_status,
    }


def _persist_state(policy: LoadedPolicy, state: dict[str, Any]) -> None:
    state["updated_ns"] = time.time_ns()
    policy.state_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(policy.state_path, state)


def _json_safe_evidence(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    result = dict(value)
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TerminalBoundaryExecutionError(
            f"{name} returned non-JSON evidence: {error}"
        ) from error
    return result


def _step_record(state: dict[str, Any], name: str) -> dict[str, Any]:
    steps = _mapping(state.get("steps"), name="pipeline steps")
    raw = steps.get(name)
    if raw is None:
        record: dict[str, Any] = {
            "status": "pending",
            "attempts": 0,
            "intent_ns": None,
            "completed_ns": None,
            "evidence": None,
            "error": None,
        }
        steps[name] = record
        return record
    if not isinstance(raw, dict):
        raise TerminalBoundaryManifestError(f"step {name} state is invalid")
    return raw


def _run_step(
    policy: LoadedPolicy,
    state: dict[str, Any],
    name: str,
    operation: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    _verify_policy_unchanged(policy)
    _verify_runtime_pins(policy.raw)
    record = _step_record(state, name)
    if record.get("status") == "completed":
        evidence = record.get("evidence")
        if not isinstance(evidence, dict):
            raise TerminalBoundaryManifestError(
                f"completed step {name} has no evidence"
            )
        return evidence
    record.update(
        {
            "status": "intent",
            "attempts": int(record.get("attempts", 0)) + 1,
            "intent_ns": time.time_ns(),
            "completed_ns": None,
            "error": None,
        }
    )
    state["phase"] = name
    state["status"] = "running"
    _persist_state(policy, state)
    try:
        evidence = _json_safe_evidence(operation(), name=name)
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _persist_state(policy, state)
        raise
    record.update(
        {
            "status": "completed",
            "completed_ns": time.time_ns(),
            "evidence": evidence,
            "error": None,
        }
    )
    _persist_state(policy, state)
    return evidence


def _resolve_pointer_manifest(
    pointer_path: Path, pointer: Mapping[str, object]
) -> Path:
    manifest_value = _string(pointer.get("manifest"), name="model pointer manifest")
    manifest = Path(manifest_value)
    if not manifest.is_absolute():
        manifest = pointer_path.parent / manifest
    resolved = manifest.resolve(strict=False)
    expected_parent = (pointer_path.parent / "manifests").resolve(strict=False)
    if resolved.parent != expected_parent:
        raise TerminalBoundaryManifestError(
            "model pointer manifest must remain in learner/manifests"
        )
    return resolved


def _model_pointer_evidence(
    path: Path,
    *,
    role: str,
    run_identity: Mapping[str, object],
) -> dict[str, object]:
    pointer, pointer_sha256, pointer_bytes = _read_json_with_digest(
        path,
        name=f"{role} pointer",
    )
    identity = _string(pointer.get("model_identity"), name=f"{role} model identity")
    step = _nonnegative_int(pointer.get("model_step"), name=f"{role} model step")
    if (
        pointer.get("format") != "startrain.model-pointer"
        or pointer.get("schema_version") != 2
        or pointer.get("role") != role
        or pointer.get("run_id") != run_identity.get("run_id")
        or pointer.get("generation_family") != run_identity.get("generation_family")
    ):
        raise TerminalBoundaryManifestError(f"{role} pointer identity is incompatible")
    manifest_path = _resolve_pointer_manifest(path, pointer)
    manifest, manifest_sha256, manifest_bytes = _read_json_with_digest(
        manifest_path,
        name=f"{role} immutable model manifest",
    )
    expected_sha256 = _digest(
        pointer.get("manifest_sha256"),
        name=f"{role} pointer manifest SHA-256",
    )
    expected_bytes = _positive_int(
        pointer.get("manifest_bytes"),
        name=f"{role} pointer manifest bytes",
    )
    if (
        manifest_sha256 != expected_sha256
        or len(manifest_bytes) != expected_bytes
        or manifest.get("format") != "startrain.model-manifest"
        or manifest.get("schema_version") != 3
        or manifest.get("model_identity") != identity
        or manifest.get("model_step") != step
        or manifest.get("run_id") != run_identity.get("run_id")
        or manifest.get("generation_family") != run_identity.get("generation_family")
    ):
        raise TerminalBoundaryManifestError(
            f"{role} pointer and immutable manifest are incoherent"
        )
    return {
        "path": str(path),
        "sha256": pointer_sha256,
        "bytes": len(pointer_bytes),
        "document": pointer,
        "model_identity": identity,
        "model_step": step,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "bytes": len(manifest_bytes),
            "document": manifest,
        },
    }


def _direct_manifest_evidence(
    path: Path,
    *,
    identity: str,
    step: int | None = None,
    run_identity: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    document, digest_value, data = _read_json_with_digest(path, name=name)
    if (
        document.get("format") != "startrain.model-manifest"
        or document.get("schema_version") != 3
        or document.get("model_identity") != identity
        or (step is not None and document.get("model_step") != step)
        or document.get("run_id") != run_identity.get("run_id")
        or document.get("generation_family") != run_identity.get("generation_family")
    ):
        raise TerminalBoundaryManifestError(f"{name} identity is incoherent")
    return {
        "path": str(path),
        "sha256": digest_value,
        "bytes": len(data),
        "document": document,
    }


def _terminal_status(
    policy: LoadedPolicy,
) -> tuple[dict[str, Any], str, bytes, str | None]:
    source = _mapping(policy.raw.get("source"), name="source")
    status_path = _absolute_path(
        source.get("promotion_status"), name="promotion status"
    )
    status, digest_value, data = _read_json_with_digest(
        status_path,
        name="promotion status",
    )
    arm = _mapping(policy.raw.get("arm"), name="terminal arm")
    updated_ns = status.get("updated_ns")
    if isinstance(updated_ns, bool) or not isinstance(updated_ns, int):
        raise TerminalBoundaryManifestError(
            "promotion status updated_ns must be an integer"
        )
    armed_ns = int(arm["promotion_status_updated_ns"])
    armed_digest = str(arm["promotion_status_sha256"])
    if updated_ns <= armed_ns or digest_value == armed_digest:
        return status, digest_value, data, "terminal_decision_not_strictly_newer"
    if status.get("terminal") is not True:
        return status, digest_value, data, "strictly_newer_status_is_not_terminal"
    decision = status.get("decision")
    if decision not in TERMINAL_DECISIONS:
        raise TerminalBoundaryManifestError(
            f"unsupported terminal promotion decision: {decision!r}"
        )
    return status, digest_value, data, None


def _capture_terminal_bundle(
    policy: LoadedPolicy,
) -> tuple[dict[str, object] | None, str]:
    source = _mapping(policy.raw.get("source"), name="source")
    status, status_sha256, status_bytes, waiting_reason = _terminal_status(policy)
    if waiting_reason is not None:
        return None, waiting_reason
    run_path = Path(str(source["run_root"])) / "run.json"
    run, run_sha256, run_bytes = _read_json_with_digest(
        run_path,
        name="source run identity",
    )
    expected_run = _mapping(source.get("run_identity"), name="source run identity")
    if any(
        run.get(field) != expected_run.get(field)
        for field in ("run_id", "generation_family", "created_ns")
    ):
        raise TerminalBoundaryManifestError(
            "source run identity differs from the policy"
        )

    champion = _model_pointer_evidence(
        Path(str(source["champion_pointer"])),
        role="champion",
        run_identity=run,
    )
    candidate_identity = _string(
        status.get("candidate_identity"),
        name="terminal candidate identity",
    )
    candidate_step = _nonnegative_int(
        status.get("candidate_step"),
        name="terminal candidate step",
    )
    champion_identity = _string(
        status.get("champion_identity"),
        name="terminal champion identity",
    )
    champion_step = _nonnegative_int(
        status.get("champion_step"),
        name="terminal champion step",
    )
    if (
        champion["model_identity"] != champion_identity
        or champion["model_step"] != champion_step
    ):
        raise TerminalBoundaryManifestError(
            "terminal status differs from the durable champion pointer"
        )
    arena_root = Path(str(source["arena_root"])).resolve()
    decision = status["decision"]
    if status.get("schema_version") != 1:
        raise TerminalBoundaryManifestError(
            "terminal promotion status schema is incompatible"
        )
    if decision in CONTROL_BOUNDARY_DECISIONS:
        if candidate_identity != champion_identity or candidate_step != champion_step:
            raise TerminalBoundaryManifestError(
                "plateau boundary does not point to the current champion"
            )
        champion_manifest = _mapping(
            champion["manifest"],
            name="champion manifest",
        )
        candidate_manifest = Path(str(champion_manifest["path"]))
        candidate_manifest_evidence = dict(champion_manifest)
        baseline_identity = champion_identity
        baseline_manifest_path = candidate_manifest
        baseline_manifest = dict(champion_manifest)
        result_path = Path(str(source["promotion_status"]))
        result = {
            "schema_version": 1,
            "terminal": True,
            "result_kind": "plateau_control",
            "candidate": candidate_identity,
            "baseline": champion_identity,
            "promotion": {"decision": decision},
        }
        result_sha256 = status_sha256
        result_bytes = status_bytes
    else:
        if decision == "promote":
            promotion_result = _string(
                _mapping(champion["document"], name="champion pointer").get(
                    "promotion_result"
                ),
                name="champion promotion result",
            )
            result_path = (
                Path(str(champion["path"])).parent / promotion_result
            ).resolve()
        else:
            result_path = (
                arena_root / f"{candidate_identity}-vs-{champion_identity}.json"
            ).resolve()
        if result_path.parent != arena_root:
            raise TerminalBoundaryManifestError(
                "terminal arena result escaped the source arena root"
            )
        result, result_sha256, result_bytes = _read_json_with_digest(
            result_path,
            name="terminal arena result",
        )
        promotion = _mapping(result.get("promotion"), name="terminal result promotion")
        baseline_identity = _string(
            result.get("baseline"),
            name="terminal result baseline",
        )
        if (
            result.get("schema_version") != 3
            or result.get("terminal") is not True
            or result.get("candidate") != candidate_identity
            or promotion.get("decision") != decision
            or (
                isinstance(result.get("result_kind"), str)
                and result.get("result_kind") != "promotion"
            )
        ):
            raise TerminalBoundaryManifestError(
                "promotion status, candidate, champion, and terminal result are "
                "incoherent"
            )
        if decision == "promote":
            if (
                champion_identity != candidate_identity
                or baseline_identity == candidate_identity
            ):
                raise TerminalBoundaryManifestError(
                    "promoted terminal result has an incoherent champion transition"
                )
        elif baseline_identity != champion_identity:
            raise TerminalBoundaryManifestError(
                "rejected terminal result baseline is not the current champion"
            )

        candidate_manifest = _absolute_path(
            result.get("candidate_manifest"),
            name="terminal result candidate manifest",
        )
        candidate_manifest_evidence = _direct_manifest_evidence(
            candidate_manifest,
            identity=candidate_identity,
            step=candidate_step,
            run_identity=run,
            name="terminal result candidate manifest",
        )
        baseline_manifest_path = _absolute_path(
            result.get("champion_manifest"),
            name="terminal result champion manifest",
        )
        baseline_manifest = _direct_manifest_evidence(
            baseline_manifest_path,
            identity=baseline_identity,
            run_identity=run,
            name="terminal result baseline manifest",
        )
        if decision != "promote":
            current_manifest = Path(
                _mapping(champion["manifest"], name="champion manifest")["path"]
            )
            if baseline_manifest_path != current_manifest:
                raise TerminalBoundaryManifestError(
                    "terminal result baseline differs from the champion pointer"
                )

    status_again = _read_json_bytes(
        Path(source["promotion_status"]),
        name="promotion status recheck",
    )
    if status_again != status_bytes:
        raise TerminalBoundaryManifestError(
            "promotion status raced terminal evidence capture"
        )
    path_digests: dict[str, str] = {
        str(Path(str(source["promotion_status"]))): status_sha256,
        str(run_path): run_sha256,
        str(candidate_manifest): str(candidate_manifest_evidence["sha256"]),
        str(Path(str(champion["path"]))): str(champion["sha256"]),
        str(
            Path(str(_mapping(champion["manifest"], name="champion manifest")["path"]))
        ): str(_mapping(champion["manifest"], name="champion manifest")["sha256"]),
        str(result_path): result_sha256,
        str(baseline_manifest_path): str(baseline_manifest["sha256"]),
    }
    bundle: dict[str, object] = {
        "status": {
            "path": str(source["promotion_status"]),
            "sha256": status_sha256,
            "bytes": len(status_bytes),
            "document": status,
        },
        "run_identity": {
            "path": str(run_path),
            "sha256": run_sha256,
            "bytes": len(run_bytes),
            "document": run,
        },
        "candidate": {
            "model_identity": candidate_identity,
            "model_step": candidate_step,
            "manifest": candidate_manifest_evidence,
        },
        "champion": champion,
        "result": {
            "path": str(result_path),
            "sha256": result_sha256,
            "bytes": len(result_bytes),
            "document": result,
            "baseline_manifest": baseline_manifest,
        },
        "decision": decision,
        "updated_ns": status["updated_ns"],
        "path_digests": dict(sorted(path_digests.items())),
        "winner_snapshot": {
            "schema_version": 1,
            "status": "verified",
            "label": "terminal-boundary-champion",
            "run_root": str(source["run_root"]),
            "run_identity": {
                field: run[field]
                for field in ("run_id", "generation_family", "created_ns")
            },
            "run_identity_artifact": {
                "path": str(run_path),
                "sha256": run_sha256,
            },
            "champion": {
                "model_identity": champion_identity,
                "model_step": champion_step,
                "updated_ns": _mapping(
                    champion["document"], name="champion pointer"
                ).get("updated_ns"),
            },
            "champion_pointer_artifact": {
                "path": str(champion["path"]),
                "sha256": str(champion["sha256"]),
            },
            "source_anchor": {
                "model_identity": baseline_identity,
                "model_step": _mapping(
                    baseline_manifest["document"],
                    name="baseline manifest",
                ).get("model_step"),
            },
            "selection": "terminal_promotion_decision",
        },
    }
    _assert_terminal_bundle_current(bundle)
    return bundle, "accepted"


def _assert_terminal_bundle_current(bundle: Mapping[str, object]) -> None:
    paths = _mapping(bundle.get("path_digests"), name="terminal path digests")
    for raw_path, expected in paths.items():
        path = Path(raw_path)
        observed = _sha256_bytes(
            _read_json_bytes(path, name="accepted terminal evidence")
        )
        if observed != expected:
            raise TerminalBoundaryExecutionError(
                f"accepted terminal evidence changed before cutover: {path}"
            )


def _validate_source_preflight(
    source: Mapping[str, object],
    hardware: Mapping[str, object],
    pause: Mapping[str, object],
) -> None:
    main_pid = source.get("main_pid")
    if (
        source.get("current") is not True
        or source.get("service_active") is not True
        or not isinstance(main_pid, int)
        or isinstance(main_pid, bool)
        or main_pid <= 0
        or source.get("coordinator_lock_matches_service") is not True
        or source.get("coordinator_status_matches_service") is not True
        or source.get("run_identity_current") is not True
    ):
        raise TerminalBoundaryExecutionError(
            "source systemd service or run identity is not current"
        )
    if hardware.get("healthy") is not True or hardware.get("current") is not True:
        raise TerminalBoundaryExecutionError("current hardware health is not proven")
    if pause.get("active") is not False:
        raise TerminalBoundaryExecutionError("an arena pause lease is active or unsafe")


def _validate_same_live_source(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> None:
    if (
        observed.get("current") is not True
        or observed.get("service_active") is not True
        or observed.get("run_identity_current") is not True
        or observed.get("coordinator_lock_matches_service") is not True
        or observed.get("coordinator_status_matches_service") is not True
        or type(observed.get("main_pid")) is not int
        or observed.get("main_pid") != expected.get("main_pid")
    ):
        raise TerminalBoundaryExecutionError(
            "source service identity changed during terminal cutover"
        )


def _validate_launch_preflight(
    hardware: Mapping[str, object],
    pause: Mapping[str, object],
) -> None:
    if hardware.get("healthy") is not True or hardware.get("current") is not True:
        raise TerminalBoundaryExecutionError(
            "hardware health is not current before queue activation"
        )
    if pause.get("active") is not False:
        raise TerminalBoundaryExecutionError(
            "arena pause lease is active before queue activation"
        )


def _validate_release_proof(evidence: Mapping[str, object]) -> None:
    required = (
        "service_inactive",
        "main_pid_released",
        "coordinator_lock_released",
        "process_groups_released",
        "gpus_released",
    )
    failed = [name for name in required if evidence.get(name) is not True]
    if failed:
        raise TerminalBoundaryExecutionError(
            "source release proof is incomplete: " + ", ".join(failed)
        )


def _plan_evidence(
    policy: LoadedPolicy,
    evidence: Mapping[str, object],
    winner_snapshot: Mapping[str, object],
    *,
    expected_path: Path | None = None,
    expected_labels: Sequence[str] | None = None,
) -> dict[str, object]:
    calibration = _mapping(policy.raw.get("calibration"), name="calibration")
    if expected_path is None:
        expected_path = Path(calibration["output_dir"]) / "ablation-plan.json"
    expected_path = expected_path.expanduser().resolve()
    path = Path(str(evidence.get("path", expected_path))).expanduser().resolve()
    if path != expected_path or not path.is_file() or path.is_symlink():
        raise TerminalBoundaryExecutionError(
            "calibration preparer did not publish the expected plan"
        )
    plan, digest_value, data = _read_json_with_digest(path, name="calibration plan")
    if (
        plan.get("report") != "startrain-elo-ablation-plan"
        or plan.get("initialization", "fork") != "fork"
        or plan.get("source_run_root")
        != _mapping(policy.raw.get("source"), name="source").get("run_root")
        or plan.get("source_winner_snapshot") != dict(winner_snapshot)
    ):
        raise TerminalBoundaryExecutionError(
            "calibration plan is not pinned to the terminal champion"
        )
    raw_treatments = _sequence(plan.get("treatments"), name="plan treatments")
    configured_labels = (
        list(calibration["treatments"])
        if expected_labels is None
        else list(expected_labels)
    )
    labels = [
        _mapping(value, name="plan treatment").get("treatment")
        for value in raw_treatments
    ]
    if labels != configured_labels:
        raise TerminalBoundaryExecutionError(
            "calibration plan treatment order differs from policy"
        )
    parent = Path(calibration["run_root_parent"])
    observed_roots: set[Path] = set()
    for raw_treatment in raw_treatments:
        treatment = _mapping(raw_treatment, name="plan treatment")
        root = _absolute_path(
            treatment.get("run_root"), name="calibration treatment root"
        )
        profile = _absolute_path(
            treatment.get("profile"), name="calibration treatment profile"
        )
        if root.parent != parent or root in observed_roots:
            raise TerminalBoundaryExecutionError(
                "calibration treatment roots are not isolated siblings"
            )
        observed_roots.add(root)
        expected_profile_sha256 = _digest(
            treatment.get("profile_sha256"),
            name="calibration profile SHA-256",
        )
        if (
            not profile.is_file()
            or profile.is_symlink()
            or _sha256(profile) != expected_profile_sha256
        ):
            raise TerminalBoundaryExecutionError(
                "calibration treatment profile changed after planning"
            )
    return {
        "path": str(path),
        "sha256": digest_value,
        "bytes": len(data),
        "plan": plan,
    }


def _fork_evidence(
    treatment: Mapping[str, object],
    evidence: Mapping[str, object],
    winner_snapshot: Mapping[str, object],
) -> dict[str, object]:
    label = str(treatment["treatment"])
    run_root = Path(str(treatment["run_root"])).expanduser().resolve()
    metadata_path = run_root / "ablation.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise TerminalBoundaryExecutionError(
            f"{label} fork did not publish ablation metadata"
        )
    metadata, digest_value, data = _read_json_with_digest(
        metadata_path,
        name=f"{label} fork metadata",
    )
    champion = _mapping(winner_snapshot.get("champion"), name="winner champion")
    anchor = _mapping(metadata.get("anchor"), name=f"{label} fork anchor")
    if (
        metadata.get("treatment") != label
        or metadata.get("source_winner_snapshot") != dict(winner_snapshot)
        or anchor.get("model_identity") != champion.get("model_identity")
        or anchor.get("model_step") != champion.get("model_step")
    ):
        raise TerminalBoundaryExecutionError(
            f"{label} fork is not anchored to the terminal champion"
        )
    return {
        **dict(evidence),
        "treatment": label,
        "run_root": str(run_root),
        "metadata": str(metadata_path),
        "metadata_sha256": digest_value,
        "metadata_bytes": len(data),
    }


def _validate_warm_start(
    run_root: Path,
    evidence: Mapping[str, object],
    winner_snapshot: Mapping[str, object],
    *,
    active: bool,
) -> dict[str, object]:
    champion = _mapping(winner_snapshot.get("champion"), name="winner champion")
    marker_path = run_root / "learner" / "champion-warm-start.json"
    marker, marker_sha256, marker_bytes = _read_json_with_digest(
        marker_path,
        name="champion warm-start marker",
    )
    expected_status = "active" if active else "prepared"
    expected_step = champion.get("model_step")
    if (
        marker.get("format") != "startrain.champion-warm-start"
        or marker.get("schema_version") != 1
        or marker.get("status") != expected_status
        or marker.get("source_model_identity") != champion.get("model_identity")
        or marker.get("source_model_step") != expected_step
        or marker.get("absolute_model_step") != expected_step
        or evidence.get("status") != "ok"
    ):
        raise TerminalBoundaryExecutionError(
            f"warm-start marker is not durably {expected_status}: {run_root}"
        )
    normalized: dict[str, object] = {
        **dict(evidence),
        "marker": {
            "path": str(marker_path),
            "sha256": marker_sha256,
            "bytes": len(marker_bytes),
            "status": expected_status,
        },
    }
    if not active:
        return normalized
    recovery_path = run_root / "learner" / "recovery.json"
    cutover_path = run_root / "learner" / "resume-cutover.json"
    recovery, recovery_sha256, recovery_bytes = _read_json_with_digest(
        recovery_path,
        name="active warm-start recovery pointer",
    )
    cutover, cutover_sha256, cutover_bytes = _read_json_with_digest(
        cutover_path,
        name="active warm-start resume cutover",
    )
    if recovery.get("step") != expected_step or cutover.get("step") != expected_step:
        raise TerminalBoundaryExecutionError(
            f"active warm-start recovery chain is incoherent: {run_root}"
        )
    normalized["recovery"] = {
        "path": str(recovery_path),
        "sha256": recovery_sha256,
        "bytes": len(recovery_bytes),
    }
    normalized["resume_cutover"] = {
        "path": str(cutover_path),
        "sha256": cutover_sha256,
        "bytes": len(cutover_bytes),
    }
    return normalized


def _replace_step_evidence(
    policy: LoadedPolicy,
    state: dict[str, Any],
    name: str,
    evidence: Mapping[str, object],
) -> None:
    record = _step_record(state, name)
    if record.get("status") != "completed":
        raise TerminalBoundaryManifestError(
            f"cannot replace evidence for incomplete step {name}"
        )
    record["evidence"] = _json_safe_evidence(evidence, name=name)
    _persist_state(policy, state)


def _deployment_manifest_evidence(
    policy: LoadedPolicy,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    queue = _mapping(policy.raw.get("queue"), name="queue activation")
    expected = Path(queue["deployment_manifest"])
    path = Path(str(evidence.get("path", expected))).expanduser().resolve()
    if path != expected or not path.is_file() or path.is_symlink():
        raise TerminalBoundaryExecutionError(
            "queue generator did not publish the configured deployment manifest"
        )
    data = _read_json_bytes(path, name="queue deployment manifest")
    return {
        **dict(evidence),
        "path": str(path),
        "sha256": _sha256_bytes(data),
        "bytes": len(data),
    }


def _activation_document(
    policy: LoadedPolicy,
    state: Mapping[str, object],
    deployment: Mapping[str, object],
    plan: Mapping[str, object],
) -> dict[str, object]:
    accepted = _mapping(state.get("accepted_terminal"), name="accepted terminal")
    steps = _mapping(state.get("steps"), name="pipeline steps")

    def evidence(name: str) -> dict[str, object]:
        record = _mapping(steps.get(name), name=f"{name} step")
        result = record.get("evidence")
        if record.get("status") != "completed" or not isinstance(result, dict):
            raise TerminalBoundaryExecutionError(
                f"activation cannot be built before {name} completes"
            )
        return result

    source_release_record = _mapping(
        steps.get("prove_source_release"),
        name="prove_source_release step",
    )
    source_release_verified_ns = _positive_int(
        source_release_record.get("completed_ns"),
        name="source release verification timestamp",
    )
    backup = _mapping(policy.raw.get("backup"), name="backup policy")
    disaster_snapshot = {
        **evidence("disaster_snapshot"),
        "verification": backup["disaster_snapshot_verification"],
        "required_after_ns": source_release_verified_ns,
    }
    source_release = {
        **evidence("prove_source_release"),
        "verified_ns": source_release_verified_ns,
    }
    plan_document = _mapping(plan.get("plan"), name="calibration plan")
    treatments = _sequence(plan_document.get("treatments"), name="plan treatments")
    roots = []
    for raw in treatments:
        treatment = _mapping(raw, name="plan treatment")
        label = str(treatment["treatment"])
        roots.append(
            {
                "treatment": label,
                "run_root": treatment["run_root"],
                "profile": treatment["profile"],
                "profile_sha256": treatment["profile_sha256"],
                "fork": evidence(f"fork:{label}"),
                "warm_start_prepared": evidence(f"warm_prepare:{label}"),
                "warm_start_active": evidence(f"warm_activate:{label}"),
            }
        )
    queue = _mapping(policy.raw.get("queue"), name="queue activation")
    return {
        "schema_version": SCHEMA_VERSION,
        "report": ACTIVATION_REPORT,
        "status": "verified_for_launch",
        "created_ns": time.time_ns(),
        "policy": {
            "path": str(policy.path),
            "sha256": policy.sha256,
            "policy_id": policy.raw["policy_id"],
            "policy_version": policy.raw["policy_version"],
        },
        "terminal": {
            "promotion_status_sha256": _mapping(
                accepted.get("status"), name="accepted status"
            )["sha256"],
            "promotion_status_updated_ns": accepted["updated_ns"],
            "decision": accepted["decision"],
            "candidate_identity": _mapping(
                accepted.get("candidate"), name="accepted candidate"
            )["model_identity"],
            "champion_identity": _mapping(
                accepted.get("champion"), name="accepted champion"
            )["model_identity"],
            "result_sha256": _mapping(accepted.get("result"), name="accepted result")[
                "sha256"
            ],
        },
        "safety": {
            "operator_hold": evidence("operator_hold"),
            "replay_backup": evidence("final_replay_backup"),
            "disaster_snapshot": disaster_snapshot,
            "source_stop": evidence("stop_source"),
            "source_release": source_release,
            "launch_preflight": _mapping(
                state.get("launch_preflight"),
                name="launch preflight",
            ),
        },
        "champion_snapshot": evidence("export_champion"),
        "calibration": {
            "frozen_replay": evidence("run_frozen_calibration"),
            "runtime_ownership": evidence("prepare_runtime_ownership"),
            "plan": {key: plan[key] for key in ("path", "sha256", "bytes")},
            "roots": roots,
        },
        "queue": {
            "deployment_manifest": {
                key: deployment[key] for key in ("path", "sha256", "bytes")
            },
            "unit": _mapping(queue.get("queue_unit"), name="queue unit")["name"],
            "state_path": queue["state_path"],
            "execution_lock_path": queue["execution_lock_path"],
        },
    }


def _verify_evidence_file(
    evidence: Mapping[str, object],
    *,
    name: str,
    path_key: str = "path",
    sha256_key: str = "sha256",
    bytes_key: str = "bytes",
) -> Path:
    path = _absolute_path(evidence.get(path_key), name=f"{name} path")
    expected_sha256 = _digest(
        evidence.get(sha256_key),
        name=f"{name} SHA-256",
    )
    expected_bytes = _positive_int(
        evidence.get(bytes_key),
        name=f"{name} bytes",
    )
    observed_sha256, observed_bytes = _hash_regular_file(path, name=name)
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        raise TerminalBoundaryManifestError(
            f"{name} changed after activation was pinned"
        )
    return path


def _verify_lambda_disaster_snapshot(
    policy: Mapping[str, object],
    evidence: Mapping[str, object],
    *,
    required_after_ns: int,
) -> dict[str, object]:
    backup = _mapping(policy.get("backup"), name="backup policy")
    verification = backup.get("disaster_snapshot_verification")
    if verification != "lambda_attached":
        raise TerminalBoundaryManifestError(
            "disaster snapshot verification must be lambda_attached"
        )
    if (
        evidence.get("status") != "ok"
        or evidence.get("verification") != verification
        or evidence.get("required_after_ns") != required_after_ns
    ):
        raise TerminalBoundaryManifestError(
            "Lambda disaster snapshot evidence is incomplete"
        )
    snapshot_path = _verify_evidence_file(
        evidence,
        name="activation Lambda disaster snapshot",
        path_key="snapshot",
        sha256_key="snapshot_sha256",
        bytes_key="snapshot_bytes",
    )
    backup_root = _absolute_path(
        backup.get("disaster_backup_root"),
        name="disaster backup root",
    )
    try:
        from scripts.training_disaster_recovery import verify_snapshot

        verified = verify_snapshot(snapshot_path, backup_root=backup_root)
    except Exception as error:
        raise TerminalBoundaryManifestError(
            f"Lambda disaster snapshot failed end-to-end verification: {error}"
        ) from error
    source_identity = _mapping(
        _mapping(policy.get("source"), name="source").get("run_identity"),
        name="source run identity",
    )
    created_ns = _positive_int(
        verified.get("created_ns"),
        name="verified Lambda disaster snapshot created_ns",
    )
    evidence_created_ns = _positive_int(
        evidence.get("created_ns"),
        name="Lambda disaster snapshot created_ns",
    )
    for field, label in (
        ("catalog_files", "catalog files"),
        ("catalog_bytes", "catalog bytes"),
        ("objects", "objects"),
    ):
        _positive_int(
            verified.get(field),
            name=f"verified Lambda disaster snapshot {label}",
        )
        _positive_int(
            evidence.get(field),
            name=f"Lambda disaster snapshot {label}",
        )
    if (
        verified.get("status") != "ok"
        or Path(str(verified.get("snapshot"))).resolve() != snapshot_path
        or verified.get("snapshot_sha256") != evidence.get("snapshot_sha256")
        or verified.get("snapshot_bytes") != evidence.get("snapshot_bytes")
        or verified.get("run_id") != source_identity.get("run_id")
        or verified.get("generation_family") != source_identity.get("generation_family")
        or evidence.get("run_id") != verified.get("run_id")
        or evidence.get("generation_family") != verified.get("generation_family")
        or evidence_created_ns != created_ns
        or any(
            evidence.get(field) != verified.get(field)
            for field in ("catalog_files", "catalog_bytes", "objects")
        )
    ):
        raise TerminalBoundaryManifestError(
            "Lambda disaster snapshot evidence disagrees with full verification"
        )
    if created_ns < required_after_ns:
        raise TerminalBoundaryManifestError(
            "Lambda disaster snapshot predates verified source release"
        )
    return verified


def verify_queue_activation_manifest(
    path: str | Path,
    *,
    policy_path: str | Path,
) -> dict[str, object]:
    policy = load_terminal_boundary_policy(policy_path)
    activation_path = Path(path).expanduser().resolve()
    expected_path = Path(
        _mapping(policy.raw.get("queue"), name="queue activation")[
            "activation_manifest"
        ]
    )
    if activation_path != expected_path:
        raise TerminalBoundaryManifestError(
            "queue activation manifest path differs from policy"
        )
    document, digest_value, data = _read_json_with_digest(
        activation_path,
        name="queue activation manifest",
    )
    policy_pin = _mapping(document.get("policy"), name="activation policy")
    queue = _mapping(document.get("queue"), name="activation queue")
    deployment = _mapping(
        queue.get("deployment_manifest"),
        name="activation deployment manifest",
    )
    deployment_path = _absolute_path(
        deployment.get("path"), name="activation deployment path"
    )
    deployment_sha256 = _digest(
        deployment.get("sha256"),
        name="activation deployment SHA-256",
    )
    deployment_bytes = _positive_int(
        deployment.get("bytes"), name="activation deployment bytes"
    )
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("report") != ACTIVATION_REPORT
        or document.get("status") != "verified_for_launch"
        or policy_pin.get("path") != str(policy.path)
        or policy_pin.get("sha256") != policy.sha256
        or policy_pin.get("policy_id") != policy.raw["policy_id"]
        or policy_pin.get("policy_version") != policy.raw["policy_version"]
        or deployment_path
        != Path(_mapping(policy.raw.get("queue"), name="queue")["deployment_manifest"])
    ):
        raise TerminalBoundaryManifestError(
            "queue activation manifest does not match the immutable policy"
        )
    terminal = _mapping(document.get("terminal"), name="activation terminal")
    _digest(
        terminal.get("promotion_status_sha256"),
        name="activation promotion-status SHA-256",
    )
    _positive_int(
        terminal.get("promotion_status_updated_ns"),
        name="activation promotion-status updated_ns",
    )
    _digest(
        terminal.get("result_sha256"),
        name="activation result SHA-256",
    )
    if (
        terminal.get("decision") not in TERMINAL_DECISIONS
        or not isinstance(terminal.get("candidate_identity"), str)
        or not isinstance(terminal.get("champion_identity"), str)
    ):
        raise TerminalBoundaryManifestError(
            "queue activation terminal evidence is incomplete"
        )
    safety = _mapping(document.get("safety"), name="activation safety")
    operator_hold = _mapping(
        safety.get("operator_hold"),
        name="activation operator hold",
    )
    replay_backup = _mapping(
        safety.get("replay_backup"),
        name="activation replay backup",
    )
    disaster_snapshot = _mapping(
        safety.get("disaster_snapshot"),
        name="activation disaster snapshot",
    )
    source_stop = _mapping(
        safety.get("source_stop"),
        name="activation source stop",
    )
    source_release = _mapping(
        safety.get("source_release"),
        name="activation source release",
    )
    launch_preflight = _mapping(
        safety.get("launch_preflight"),
        name="activation launch preflight",
    )
    launch_hardware = _mapping(
        launch_preflight.get("hardware"),
        name="activation launch hardware",
    )
    launch_pause = _mapping(
        launch_preflight.get("arena_pause"),
        name="activation launch arena pause",
    )
    source_release_verified_ns = _positive_int(
        source_release.get("verified_ns"),
        name="activation source release verified_ns",
    )
    if (
        operator_hold.get("status") != "active"
        or replay_backup.get("status") != "ok"
        or source_stop.get("status") != "stopped"
        or any(
            source_release.get(field) is not True
            for field in (
                "service_inactive",
                "main_pid_released",
                "coordinator_lock_released",
                "process_groups_released",
                "gpus_released",
            )
        )
        or launch_hardware.get("healthy") is not True
        or launch_hardware.get("current") is not True
        or launch_pause.get("active") is not False
    ):
        raise TerminalBoundaryManifestError(
            "queue activation safety evidence is incomplete"
        )
    operator_hold_path = _absolute_path(
        operator_hold.get("path"),
        name="activation operator hold path",
    )
    _digest(
        operator_hold.get("sha256"),
        name="activation operator hold SHA-256",
    )
    _positive_int(
        operator_hold.get("bytes"),
        name="activation operator hold bytes",
    )
    if operator_hold_path.exists():
        _verify_evidence_file(operator_hold, name="activation operator hold")
    _verify_evidence_file(replay_backup, name="activation replay backup")
    _verify_lambda_disaster_snapshot(
        policy.raw,
        disaster_snapshot,
        required_after_ns=source_release_verified_ns,
    )
    champion_snapshot = _mapping(
        document.get("champion_snapshot"),
        name="activation champion snapshot",
    )
    if champion_snapshot.get("status") != "verified":
        raise TerminalBoundaryManifestError(
            "queue activation champion snapshot is not verified"
        )
    champion_snapshot_pin = _mapping(
        champion_snapshot.get("pin"),
        name="activation champion snapshot pin",
    )
    champion_pin_path = _verify_evidence_file(
        champion_snapshot_pin,
        name="activation champion snapshot pin",
    )
    champion_pin = _json_object(
        _read_json_bytes(
            champion_pin_path,
            name="activation champion snapshot pin",
        ),
        name="activation champion snapshot pin",
    )
    destination = _absolute_path(
        champion_snapshot.get("destination"),
        name="activation champion snapshot destination",
    )
    if (
        champion_pin.get("status") != "verified"
        or Path(str(champion_pin.get("destination"))).resolve() != destination
    ):
        raise TerminalBoundaryManifestError(
            "activation champion snapshot pin is incompatible"
        )
    snapshot_files = _sequence(
        champion_pin.get("files"),
        name="activation champion snapshot files",
    )
    if not snapshot_files:
        raise TerminalBoundaryManifestError(
            "activation champion snapshot contains no files"
        )
    for index, raw_file in enumerate(snapshot_files):
        file_evidence = _mapping(
            raw_file,
            name=f"activation champion snapshot file {index}",
        )
        relative = _string(
            file_evidence.get("path"),
            name=f"activation champion snapshot file {index} path",
        )
        if Path(relative).is_absolute():
            raise TerminalBoundaryManifestError(
                "activation champion snapshot file path must be relative"
            )
        file_path = (destination / relative).resolve()
        if destination not in file_path.parents:
            raise TerminalBoundaryManifestError(
                "activation champion snapshot file escaped its destination"
            )
        _verify_evidence_file(
            {
                **file_evidence,
                "path": str(file_path),
            },
            name=f"activation champion snapshot file {index}",
        )
    policy_queue = _mapping(policy.raw.get("queue"), name="queue")
    if (
        queue.get("unit")
        != _mapping(policy_queue.get("queue_unit"), name="queue unit").get("name")
        or queue.get("state_path") != policy_queue.get("state_path")
        or queue.get("execution_lock_path") != policy_queue.get("execution_lock_path")
    ):
        raise TerminalBoundaryManifestError(
            "queue activation target differs from the immutable policy"
        )
    deployment_data = _read_json_bytes(
        deployment_path,
        name="activation deployment manifest",
    )
    if (
        _sha256_bytes(deployment_data) != deployment_sha256
        or len(deployment_data) != deployment_bytes
    ):
        raise TerminalBoundaryManifestError(
            "queue deployment changed after activation was pinned"
        )
    calibration = _mapping(document.get("calibration"), name="activation calibration")
    runtime_ownership = _mapping(
        calibration.get("runtime_ownership"),
        name="activation runtime ownership",
    )
    policy_calibration = _mapping(
        policy.raw.get("calibration"),
        name="calibration",
    )
    if (
        runtime_ownership.get("status") != "verified"
        or runtime_ownership.get("user") != policy_calibration.get("runtime_user")
        or runtime_ownership.get("group") != policy_calibration.get("runtime_group")
        or not isinstance(runtime_ownership.get("paths_updated"), int)
        or int(runtime_ownership["paths_updated"]) <= 0
    ):
        raise TerminalBoundaryManifestError(
            "activation runtime ownership evidence is incomplete"
        )
    frozen_replay = _mapping(
        calibration.get("frozen_replay"),
        name="activation frozen-replay calibration",
    )
    if frozen_replay.get("status") != "completed":
        raise TerminalBoundaryManifestError(
            "activation manifest lacks completed frozen-replay calibration"
        )
    frozen_comparison = _mapping(
        frozen_replay.get("comparison"),
        name="activation frozen-replay comparison",
    )
    comparison_path = _absolute_path(
        frozen_comparison.get("path"),
        name="activation frozen-replay comparison path",
    )
    if _sha256(comparison_path) != _digest(
        frozen_comparison.get("sha256"),
        name="activation frozen-replay comparison SHA-256",
    ):
        raise TerminalBoundaryManifestError(
            "frozen-replay comparison changed after activation was pinned"
        )
    plan = _mapping(calibration.get("plan"), name="activation plan")
    plan_path = _absolute_path(plan.get("path"), name="activation plan path")
    plan_data = _read_json_bytes(plan_path, name="activation plan")
    if _sha256_bytes(plan_data) != _digest(
        plan.get("sha256"), name="activation plan SHA-256"
    ) or len(plan_data) != _positive_int(
        plan.get("bytes"), name="activation plan bytes"
    ):
        raise TerminalBoundaryManifestError(
            "calibration plan changed after activation was pinned"
        )
    plan_document = _json_object(plan_data, name="activation plan")
    plan_treatments = _sequence(
        plan_document.get("treatments"),
        name="activation plan treatments",
    )
    roots = _sequence(calibration.get("roots"), name="activation roots")
    if not roots or len(roots) != len(plan_treatments):
        raise TerminalBoundaryManifestError(
            "activation manifest lacks fully active warm-start roots"
        )
    for raw_root, raw_treatment in zip(roots, plan_treatments, strict=True):
        root = _mapping(raw_root, name="activation root")
        treatment = _mapping(raw_treatment, name="activation plan treatment")
        active_warm_start = _mapping(
            root.get("warm_start_active"),
            name="activation active warm-start",
        )
        marker = _mapping(
            active_warm_start.get("marker"),
            name="activation warm-start marker",
        )
        recovery = _mapping(
            active_warm_start.get("recovery"),
            name="activation warm-start recovery",
        )
        resume_cutover = _mapping(
            active_warm_start.get("resume_cutover"),
            name="activation warm-start resume cutover",
        )
        marker_path = _verify_evidence_file(
            marker,
            name="activation warm-start marker",
        )
        _verify_evidence_file(
            recovery,
            name="activation warm-start recovery",
        )
        _verify_evidence_file(
            resume_cutover,
            name="activation warm-start resume cutover",
        )
        marker_document = _json_object(
            _read_json_bytes(
                marker_path,
                name="activation warm-start marker",
            ),
            name="activation warm-start marker",
        )
        run_root = _absolute_path(
            root.get("run_root"),
            name="activation warm-start run root",
        )
        checkpoint_value = _string(
            marker_document.get("checkpoint"),
            name="activation warm-start checkpoint",
        )
        checkpoint_path = (run_root / "learner" / checkpoint_value).resolve()
        if checkpoint_path.parent != (run_root / "learner" / "recovery").resolve():
            raise TerminalBoundaryManifestError(
                "activation warm-start checkpoint escaped the recovery directory"
            )
        _verify_evidence_file(
            {
                "path": str(checkpoint_path),
                "sha256": marker_document.get("checkpoint_sha256"),
                "bytes": marker_document.get("checkpoint_bytes"),
            },
            name="activation warm-start checkpoint",
        )
        if (
            root.get("treatment") != treatment.get("treatment")
            or root.get("run_root") != treatment.get("run_root")
            or root.get("profile") != treatment.get("profile")
            or root.get("profile_sha256") != treatment.get("profile_sha256")
            or active_warm_start.get("status") != "ok"
            or marker.get("status") != "active"
        ):
            raise TerminalBoundaryManifestError(
                "activation manifest contains an unverified warm-start root"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "report": ACTIVATION_REPORT,
        "status": "verified",
        "path": str(activation_path),
        "sha256": digest_value,
        "bytes": len(data),
        "deployment_manifest": str(deployment_path),
        "deployment_sha256": deployment_sha256,
    }


def _failure_id(
    policy: LoadedPolicy,
    state: Mapping[str, object],
    error: Exception,
) -> str:
    accepted = state.get("accepted_terminal")
    status_sha256 = (
        _mapping(
            _mapping(accepted, name="accepted terminal").get("status"),
            name="accepted status",
        ).get("sha256")
        if isinstance(accepted, dict)
        else None
    )
    material = {
        "policy_sha256": policy.sha256,
        "terminal_sha256": status_sha256,
        "phase": state.get("phase"),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    return _sha256_bytes(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _control_resume_request(
    policy: LoadedPolicy,
    state: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    fallback = _mapping(policy.raw.get("fallback"), name="continuity fallback")
    return {
        "schema_version": SCHEMA_VERSION,
        "report": FALLBACK_REPORT,
        "status": "requested",
        "requested": True,
        "action": "request_fallback",
        "requested_action": "reconcile_training_continuity",
        "requested_ns": time.time_ns(),
        "reason": "frozen_calibration_retained_control",
        "terminal_reason": "frozen_calibration_retained_control",
        "failure_domain": None,
        "source": {
            "kind": "terminal_boundary",
            "policy": str(policy.path),
            "policy_sha256": policy.sha256,
            "state": str(policy.state_path),
            "phase": state.get("phase"),
        },
        "selection": dict(selection),
        "requires_safe_workload": True,
        "adoption_authorized": False,
        "path": fallback["handoff_path"],
    }


def _record_failure_and_request_fallback(
    policy: LoadedPolicy,
    state: dict[str, Any],
    adapters: TerminalBoundaryAdapters,
    error: Exception,
) -> None:
    failure_id = _failure_id(policy, state, error)
    failure = {
        "failure_id": failure_id,
        "phase": state.get("phase"),
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_ns": time.time_ns(),
    }
    failures = _sequence(state.get("failures"), name="pipeline failures")
    if not any(
        isinstance(existing, dict) and existing.get("failure_id") == failure_id
        for existing in failures
    ):
        failures.append(failure)
    state.update(
        {
            "status": "failed",
            "failure": failure,
            "automatic_launch_authorized": False,
        }
    )
    _persist_state(policy, state)

    previous = state.get("fallback")
    if (
        isinstance(previous, dict)
        and previous.get("failure_id") == failure_id
        and previous.get("status") == "requested"
    ):
        return
    fallback = _mapping(policy.raw.get("fallback"), name="continuity fallback")
    request = {
        "schema_version": SCHEMA_VERSION,
        "report": FALLBACK_REPORT,
        "status": "requested",
        "requested": True,
        "action": "request_fallback",
        "requested_action": "reconcile_training_continuity",
        "requested_ns": time.time_ns(),
        "reason": "terminal_boundary_failed",
        "terminal_reason": "terminal_boundary_failed",
        "failure_domain": "terminal_boundary",
        "failure_id": failure_id,
        "source": {
            "kind": "terminal_boundary",
            "policy": str(policy.path),
            "policy_sha256": policy.sha256,
            "state": str(policy.state_path),
            "phase": state.get("phase"),
        },
        "failure": failure,
        "requires_safe_workload": True,
        "adoption_authorized": False,
        "path": fallback["handoff_path"],
    }
    state["fallback"] = {
        "status": "intent",
        "failure_id": failure_id,
        "intent_ns": time.time_ns(),
        "request": request,
        "evidence": None,
        "error": None,
    }
    _persist_state(policy, state)
    try:
        evidence = _json_safe_evidence(
            adapters.request_continuity_fallback(policy.raw, request),
            name="continuity fallback",
        )
    except Exception as fallback_error:
        state["fallback"].update(
            {
                "status": "failed",
                "error": f"{type(fallback_error).__name__}: {fallback_error}",
            }
        )
        _persist_state(policy, state)
        return
    state["fallback"].update(
        {
            "status": "requested",
            "completed_ns": time.time_ns(),
            "evidence": evidence,
            "error": None,
        }
    )
    _persist_state(policy, state)


def _step_completed(state: Mapping[str, object], name: str) -> bool:
    steps = state.get("steps")
    return (
        isinstance(steps, dict)
        and isinstance(steps.get(name), dict)
        and steps[name].get("status") == "completed"
    )


def run_terminal_boundary_pipeline(
    policy_path: str | Path,
    *,
    adapters: TerminalBoundaryAdapters | None = None,
) -> dict[str, object]:
    """Run or resume the fail-closed terminal-boundary deployment saga."""

    policy = load_terminal_boundary_policy(policy_path)
    operations: TerminalBoundaryAdapters = (
        DefaultTerminalBoundaryAdapters() if adapters is None else adapters
    )
    with _pipeline_lock(policy.state_path):
        state = _load_state(policy)
        try:
            _run_step(
                policy,
                state,
                "pin_policy",
                lambda: (
                    _write_new_or_verify(policy.pinned_path, policy.data, mode=0o444)
                    or {
                        "path": str(policy.pinned_path),
                        "sha256": policy.sha256,
                        "bytes": len(policy.data),
                    }
                ),
            )
            if (
                state.get("status") == "completed"
                and state.get("phase") == "frozen_calibration_retained_control"
            ):
                return state
            fallback = state.get("fallback")
            if isinstance(fallback, dict) and fallback.get("status") in {
                "intent",
                "failed",
                "requested",
            }:
                state.update(
                    {
                        "status": "blocked",
                        "phase": "continuity_fallback_requested",
                        "automatic_launch_authorized": False,
                    }
                )
                _persist_state(policy, state)
                return state
            if state.get("status") == "completed":
                activation = _mapping(
                    _step_record(state, "verify_activation").get("evidence"),
                    name="completed activation evidence",
                )
                verify_queue_activation_manifest(
                    activation["path"],
                    policy_path=policy.path,
                )
                return state
            accepted = state.get("accepted_terminal")
            newly_accepted = False
            if accepted is None:
                bundle, reason = _capture_terminal_bundle(policy)
                if bundle is None:
                    state.update(
                        {
                            "status": "waiting",
                            "phase": "awaiting_strictly_newer_terminal",
                            "waiting_reason": reason,
                            "automatic_launch_authorized": False,
                        }
                    )
                    _persist_state(policy, state)
                    return state
                source_evidence = _json_safe_evidence(
                    operations.inspect_source(policy.raw),
                    name="source inspection",
                )
                hardware_evidence = _json_safe_evidence(
                    operations.inspect_hardware(policy.raw),
                    name="hardware inspection",
                )
                pause_evidence = _json_safe_evidence(
                    operations.inspect_arena_pause(policy.raw),
                    name="arena pause inspection",
                )
                if hardware_evidence.get("status") == "unavailable" or (
                    hardware_evidence.get("healthy") is True
                    and hardware_evidence.get("current") is False
                ):
                    state.update(
                        {
                            "status": "waiting",
                            "phase": "awaiting_current_hardware_report",
                            "waiting_reason": "hardware_health_report_refreshing",
                            "automatic_launch_authorized": False,
                        }
                    )
                    _persist_state(policy, state)
                    return state
                if pause_evidence.get("active") is not False:
                    state.update(
                        {
                            "status": "waiting",
                            "phase": "awaiting_quiescent_terminal_boundary",
                            "waiting_reason": (
                                "newer_terminal_already_followed_by_active_arena"
                            ),
                            "automatic_launch_authorized": False,
                        }
                    )
                    _persist_state(policy, state)
                    return state
                _validate_source_preflight(
                    source_evidence,
                    hardware_evidence,
                    pause_evidence,
                )
                state["accepted_terminal"] = bundle
                state["source_preflight"] = {
                    "source": source_evidence,
                    "hardware": hardware_evidence,
                    "arena_pause": pause_evidence,
                    "accepted_ns": time.time_ns(),
                }
                state.pop("waiting_reason", None)
                state["phase"] = "terminal_accepted"
                _persist_state(policy, state)
                accepted = bundle
                newly_accepted = True
            if not isinstance(accepted, dict):
                raise TerminalBoundaryManifestError(
                    "accepted terminal state is invalid"
                )
            _assert_terminal_bundle_current(accepted)

            if not newly_accepted and _step_record(state, "stop_source").get(
                "status"
            ) not in {
                "intent",
                "failed",
                "completed",
            }:
                source_evidence = _json_safe_evidence(
                    operations.inspect_source(policy.raw),
                    name="source inspection",
                )
                hardware_evidence = _json_safe_evidence(
                    operations.inspect_hardware(policy.raw),
                    name="hardware inspection",
                )
                pause_evidence = _json_safe_evidence(
                    operations.inspect_arena_pause(policy.raw),
                    name="arena pause inspection",
                )
                if hardware_evidence.get("status") == "unavailable" or (
                    hardware_evidence.get("healthy") is True
                    and hardware_evidence.get("current") is False
                ):
                    state.update(
                        {
                            "status": "waiting",
                            "phase": "awaiting_current_hardware_report",
                            "waiting_reason": "hardware_health_report_refreshing",
                            "automatic_launch_authorized": False,
                        }
                    )
                    _persist_state(policy, state)
                    return state
                _validate_source_preflight(
                    source_evidence,
                    hardware_evidence,
                    pause_evidence,
                )
                original_preflight = _mapping(
                    state.get("source_preflight"),
                    name="original source preflight",
                )
                _validate_same_live_source(
                    _mapping(
                        original_preflight.get("source"),
                        name="original source evidence",
                    ),
                    source_evidence,
                )
                state["source_recheck"] = {
                    "source": source_evidence,
                    "hardware": hardware_evidence,
                    "arena_pause": pause_evidence,
                    "verified_ns": time.time_ns(),
                }
                _persist_state(policy, state)

            source_preflight = _mapping(
                state.get("source_preflight"), name="source preflight"
            )
            expected_source = _mapping(
                source_preflight.get("source"),
                name="source preflight evidence",
            )

            def assert_live_source() -> None:
                observed_source = _json_safe_evidence(
                    operations.inspect_source(policy.raw),
                    name="source inspection",
                )
                hardware_deadline = time.monotonic() + 30.0
                while True:
                    observed_hardware = _json_safe_evidence(
                        operations.inspect_hardware(policy.raw),
                        name="hardware inspection",
                    )
                    transient = observed_hardware.get("status") == "unavailable" or (
                        observed_hardware.get("healthy") is True
                        and observed_hardware.get("current") is False
                    )
                    if not transient or time.monotonic() >= hardware_deadline:
                        break
                    time.sleep(0.5)
                observed_pause = _json_safe_evidence(
                    operations.inspect_arena_pause(policy.raw),
                    name="arena pause inspection",
                )
                _validate_same_live_source(expected_source, observed_source)
                _validate_launch_preflight(observed_hardware, observed_pause)

            status_evidence = _mapping(accepted.get("status"), name="accepted status")
            hold_path = Path(policy.raw["operator_hold_path"])
            hold_document = state.get("operator_hold_document")
            if hold_document is None:
                hold_document = {
                    "schema_version": SCHEMA_VERSION,
                    "report": "startrain-terminal-boundary-operator-hold",
                    "status": "active",
                    "policy_sha256": policy.sha256,
                    "promotion_status_sha256": status_evidence["sha256"],
                    "promotion_status_updated_ns": accepted["updated_ns"],
                    "created_ns": time.time_ns(),
                    "reason": "terminal_boundary_cutover",
                }
                state["operator_hold_document"] = hold_document
                _persist_state(policy, state)
            if not isinstance(hold_document, dict):
                raise TerminalBoundaryManifestError(
                    "persisted operator hold document is invalid"
                )
            _run_step(
                policy,
                state,
                "operator_hold",
                lambda: operations.place_operator_hold(hold_path, hold_document),
            )
            _assert_terminal_bundle_current(accepted)
            stop_requires_live_source = (
                _step_record(state, "stop_source").get("status") == "pending"
            )
            stop_evidence = _run_step(
                policy,
                state,
                "stop_source",
                lambda: (
                    _assert_terminal_bundle_current(accepted)
                    or (assert_live_source() if stop_requires_live_source else None)
                    or operations.stop_source(policy.raw)
                ),
            )
            release_evidence = _run_step(
                policy,
                state,
                "prove_source_release",
                lambda: operations.prove_source_release(
                    policy.raw,
                    expected_source,
                    stop_evidence,
                ),
            )
            _validate_release_proof(release_evidence)
            _assert_terminal_bundle_current(accepted)
            _run_step(
                policy,
                state,
                "final_replay_backup",
                lambda: operations.final_replay_backup(policy.raw),
            )
            source_release_verified_ns = _positive_int(
                _step_record(state, "prove_source_release").get("completed_ns"),
                name="source release verification timestamp",
            )
            _run_step(
                policy,
                state,
                "disaster_snapshot",
                lambda: operations.disaster_snapshot(
                    policy.raw,
                    source_release_verified_ns,
                ),
            )
            winner_snapshot = _mapping(
                accepted.get("winner_snapshot"), name="winner snapshot"
            )
            _run_step(
                policy,
                state,
                "export_champion",
                lambda: operations.export_champion(
                    policy.raw,
                    winner_snapshot,
                ),
            )
            raw_plan_evidence = _run_step(
                policy,
                state,
                "prepare_calibration",
                lambda: operations.prepare_calibration(
                    policy.raw,
                    winner_snapshot,
                ),
            )
            plan_evidence = _plan_evidence(
                policy,
                raw_plan_evidence,
                winner_snapshot,
            )
            frozen_evidence = _run_step(
                policy,
                state,
                "run_frozen_calibration",
                lambda: operations.run_frozen_calibration(
                    policy.raw,
                    plan_evidence,
                    winner_snapshot,
                ),
            )
            screen_plan = frozen_evidence.get("screen_plan")
            selection = _mapping(
                frozen_evidence.get("selection"),
                name="frozen calibration selection",
            )
            selected_arm = _string(
                selection.get("selected_arm"),
                name="frozen calibration selected arm",
            )
            if not isinstance(screen_plan, Mapping):
                fallback_evidence = _run_step(
                    policy,
                    state,
                    "resume_runtime_control",
                    lambda: operations.request_continuity_fallback(
                        policy.raw,
                        _control_resume_request(
                            policy,
                            state,
                            selection,
                        ),
                    ),
                )
                state.update(
                    {
                        "status": "completed",
                        "phase": "frozen_calibration_retained_control",
                        "completed_ns": time.time_ns(),
                        "automatic_launch_authorized": False,
                        "fallback": {
                            "status": "requested",
                            "reason": "frozen_calibration_retained_control",
                            "evidence": fallback_evidence,
                        },
                    }
                )
                _persist_state(policy, state)
                return state
            calibration = _mapping(
                policy.raw.get("calibration"),
                name="calibration",
            )
            frozen = _mapping(
                calibration.get("frozen_replay"),
                name="frozen-replay calibration",
            )
            control_arm = _string(
                _sequence(
                    calibration.get("treatments"),
                    name="calibration treatments",
                )[0],
                name="frozen calibration control arm",
            )
            plan_evidence = _plan_evidence(
                policy,
                screen_plan,
                winner_snapshot,
                expected_path=Path(str(frozen["screen_plan_path"])),
                expected_labels=(
                    (control_arm,)
                    if selected_arm == control_arm
                    else (control_arm, selected_arm)
                ),
            )
            plan_document = _mapping(plan_evidence.get("plan"), name="calibration plan")
            treatments = [
                _mapping(raw, name="calibration treatment")
                for raw in _sequence(
                    plan_document.get("treatments"),
                    name="calibration treatments",
                )
            ]
            for treatment in treatments:
                label = str(treatment["treatment"])
                raw_fork = _run_step(
                    policy,
                    state,
                    f"fork:{label}",
                    lambda treatment=treatment: operations.fork_calibration(
                        policy.raw,
                        plan_evidence,
                        treatment,
                        winner_snapshot,
                    ),
                )
                _fork_evidence(treatment, raw_fork, winner_snapshot)
            for treatment in treatments:
                label = str(treatment["treatment"])
                run_root = Path(str(treatment["run_root"]))
                profile = Path(str(treatment["profile"]))
                raw_prepared = _run_step(
                    policy,
                    state,
                    f"warm_prepare:{label}",
                    lambda run_root=run_root, profile=profile: operations.warm_start(
                        run_root,
                        profile,
                        prepare_only=True,
                    ),
                )
                prepared = _validate_warm_start(
                    run_root,
                    raw_prepared,
                    winner_snapshot,
                    active=False,
                )
                _replace_step_evidence(
                    policy,
                    state,
                    f"warm_prepare:{label}",
                    prepared,
                )
            for treatment in treatments:
                label = str(treatment["treatment"])
                run_root = Path(str(treatment["run_root"]))
                profile = Path(str(treatment["profile"]))
                raw_active = _run_step(
                    policy,
                    state,
                    f"warm_activate:{label}",
                    lambda run_root=run_root, profile=profile: operations.warm_start(
                        run_root,
                        profile,
                        prepare_only=False,
                    ),
                )
                active = _validate_warm_start(
                    run_root,
                    raw_active,
                    winner_snapshot,
                    active=True,
                )
                _replace_step_evidence(
                    policy,
                    state,
                    f"warm_activate:{label}",
                    active,
                )

            raw_deployment = _run_step(
                policy,
                state,
                "generate_queue_manifest",
                lambda: operations.generate_queue_manifest(
                    policy.raw,
                    plan_evidence,
                ),
            )
            deployment = _deployment_manifest_evidence(policy, raw_deployment)
            _run_step(
                policy,
                state,
                "prepare_runtime_ownership",
                lambda: operations.prepare_runtime_ownership(
                    policy.raw,
                    plan_evidence,
                ),
            )
            _run_step(
                policy,
                state,
                "verify_queue_manifest",
                lambda: operations.verify_queue_manifest(
                    policy.raw,
                    deployment,
                ),
            )
            launch_hardware = _json_safe_evidence(
                operations.inspect_hardware(policy.raw),
                name="launch hardware inspection",
            )
            launch_pause = _json_safe_evidence(
                operations.inspect_arena_pause(policy.raw),
                name="launch arena pause inspection",
            )
            _validate_launch_preflight(launch_hardware, launch_pause)
            state["launch_preflight"] = {
                "hardware": launch_hardware,
                "arena_pause": launch_pause,
                "verified_ns": time.time_ns(),
            }
            _persist_state(policy, state)
            activation_document = state.get("activation_document")
            if activation_document is None:
                activation_document = _activation_document(
                    policy,
                    state,
                    deployment,
                    plan_evidence,
                )
                state["activation_document"] = activation_document
                _persist_state(policy, state)
            if not isinstance(activation_document, dict):
                raise TerminalBoundaryManifestError(
                    "persisted activation document is invalid"
                )
            activation_path = Path(
                _mapping(policy.raw.get("queue"), name="queue")["activation_manifest"]
            )
            _run_step(
                policy,
                state,
                "publish_activation",
                lambda: _write_immutable_json(
                    activation_path,
                    activation_document,
                    mode=0o440,
                ),
            )
            activation_evidence = _run_step(
                policy,
                state,
                "verify_activation",
                lambda: verify_queue_activation_manifest(
                    activation_path,
                    policy_path=policy.path,
                ),
            )
            _run_step(
                policy,
                state,
                "launch_queue",
                lambda: operations.launch_queue(
                    policy.raw,
                    Path(str(activation_evidence["path"])),
                ),
            )
            _run_step(
                policy,
                state,
                "release_operator_hold",
                lambda: _release_owned_operator_hold(
                    hold_path,
                    hold_document,
                ),
            )
            state.update(
                {
                    "status": "completed",
                    "phase": "queue_launched",
                    "completed_ns": time.time_ns(),
                    "automatic_launch_authorized": True,
                }
            )
            state.pop("failure", None)
            _persist_state(policy, state)
            return state
        except Exception as error:
            _record_failure_and_request_fallback(
                policy,
                state,
                operations,
                error,
            )
            if isinstance(error, TerminalBoundaryError):
                raise
            raise TerminalBoundaryExecutionError(
                f"{type(error).__name__}: {error}"
            ) from error


def recover_terminal_boundary_pipeline(
    policy_path: str | Path,
    *,
    adapters: TerminalBoundaryAdapters | None = None,
) -> dict[str, object]:
    """Release an owned hold and request fallback after an interrupted service."""

    policy = load_terminal_boundary_policy(policy_path)
    operations: TerminalBoundaryAdapters = (
        DefaultTerminalBoundaryAdapters() if adapters is None else adapters
    )
    with _pipeline_lock(policy.state_path):
        state = _load_state(policy)
        if state.get("status") in {"waiting", "completed"}:
            return state
        fallback = state.get("fallback")
        if isinstance(fallback, dict) and fallback.get("status") == "requested":
            return state
        _record_failure_and_request_fallback(
            policy,
            state,
            operations,
            TerminalBoundaryExecutionError(
                "terminal-boundary service exited before completion"
            ),
        )
        return state


class DefaultTerminalBoundaryAdapters:
    """Production adapters backed by existing StarTrain and systemd APIs."""

    def __init__(
        self,
        *,
        warm_starter: WarmStarter | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self._warm_starter = warm_starter
        self.clock_ns = clock_ns
        self.monotonic = monotonic
        self.sleep = sleep
        self.runner = runner

    @staticmethod
    def _source(policy: Mapping[str, object]) -> dict[str, Any]:
        return _mapping(policy.get("source"), name="source")

    @staticmethod
    def _queue(policy: Mapping[str, object]) -> dict[str, Any]:
        return _mapping(policy.get("queue"), name="queue activation")

    @staticmethod
    def _hold_created_ns(policy: Mapping[str, object]) -> int:
        path = _absolute_path(
            policy.get("operator_hold_path"),
            name="operator hold path",
        )
        hold, _digest_value, _ = _read_json_with_digest(
            path,
            name="terminal-boundary operator hold",
        )
        return _positive_int(
            hold.get("created_ns"),
            name="operator hold created_ns",
        )

    @classmethod
    def _frozen_boundary_ns(cls, policy: Mapping[str, object]) -> int:
        source = cls._source(policy)
        coordinator, _digest_value, _ = _read_json_with_digest(
            Path(str(source["coordinator_status"])),
            name="final coordinator status",
        )
        if coordinator.get("state") != "stopped":
            raise TerminalBoundaryExecutionError(
                "source coordinator is not stopped at the backup boundary"
            )
        stopped_ns = _positive_int(
            coordinator.get("timestamp_ns"),
            name="final coordinator status timestamp",
        )
        return max(cls._hold_created_ns(policy), stopped_ns)

    def inspect_source(self, policy: Mapping[str, object]) -> Mapping[str, object]:
        from startrain.continuity import SystemdUnitManager

        source = self._source(policy)
        unit = _mapping(source.get("unit"), name="source unit")
        expected_identity = _mapping(
            source.get("run_identity"), name="source run identity"
        )
        run, run_sha256, _ = _read_json_with_digest(
            Path(source["run_root"]) / "run.json",
            name="source run identity",
        )
        lock, lock_sha256, _ = _read_json_with_digest(
            Path(source["coordinator_lock"]),
            name="source coordinator lock",
        )
        coordinator, coordinator_sha256, _ = _read_json_with_digest(
            Path(source["coordinator_status"]),
            name="source coordinator status",
        )
        status = SystemdUnitManager(runner=self.runner).status(str(unit["name"]))
        lock_pid = lock.get("pid")
        coordinator_pid = coordinator.get("coordinator_pid")
        workers = coordinator.get("workers")
        process_ids = [status.main_pid]
        if isinstance(workers, dict):
            process_ids.extend(
                value["pid"]
                for value in workers.values()
                if isinstance(value, dict)
                and type(value.get("pid")) is int
                and value["pid"] > 0
            )
        return {
            "current": (
                status.query_error is None
                and status.active
                and status.main_pid > 0
                and status.fragment_path == unit["path"]
            ),
            "unit": str(unit["name"]),
            "unit_status": status.as_dict(),
            "service_active": status.active,
            "main_pid": status.main_pid,
            "run_identity_current": all(
                run.get(field) == expected_identity.get(field)
                for field in ("run_id", "generation_family", "created_ns")
            ),
            "run_identity_sha256": run_sha256,
            "coordinator_lock_matches_service": (
                type(lock_pid) is int and lock_pid == status.main_pid
            ),
            "coordinator_lock_sha256": lock_sha256,
            "coordinator_status_matches_service": (
                type(coordinator_pid) is int
                and coordinator_pid == status.main_pid
                and coordinator.get("state") in {"running", "draining"}
            ),
            "coordinator_status_sha256": coordinator_sha256,
            "process_ids": sorted(set(process_ids)),
        }

    def inspect_hardware(self, policy: Mapping[str, object]) -> Mapping[str, object]:
        source = self._source(policy)
        path = Path(source["hardware_report"])
        if not path.exists():
            return {
                "path": str(path),
                "healthy": False,
                "current": False,
                "status": "unavailable",
                "reason": "hardware health report is being refreshed",
            }
        try:
            report, digest_value, _ = _read_json_with_digest(
                path,
                name="hardware health report",
            )
        except TerminalBoundaryManifestError:
            if not path.exists():
                return {
                    "path": str(path),
                    "healthy": False,
                    "current": False,
                    "status": "unavailable",
                    "reason": "hardware health report changed during refresh",
                }
            raise
        captured_ns = report.get("captured_ns")
        now = self.clock_ns()
        maximum_age_ns = int(float(source["hardware_max_age_seconds"]) * 1e9)
        expected_profile = _mapping(source.get("profile"), name="source profile")[
            "path"
        ]
        current = (
            type(captured_ns) is int
            and 0 < captured_ns <= now
            and now - captured_ns <= maximum_age_ns
            and Path(str(report.get("config"))).expanduser().resolve()
            == Path(expected_profile)
        )
        expected_gpus = set(source["gpu_ids"])
        raw_gpus = report.get("gpus")
        observed_gpus = (
            {
                gpu.get("index")
                for gpu in raw_gpus
                if isinstance(gpu, dict) and gpu.get("healthy") is True
            }
            if isinstance(raw_gpus, list)
            else set()
        )
        healthy = report.get("healthy") is True and expected_gpus <= observed_gpus
        if not expected_gpus and report.get("healthy") is True:
            healthy = True
        return {
            "path": str(path),
            "sha256": digest_value,
            "healthy": healthy,
            "current": current,
            "status": "healthy" if healthy and current else "unsafe",
            "captured_ns": captured_ns,
            "expected_gpu_ids": sorted(expected_gpus),
        }

    def inspect_arena_pause(self, policy: Mapping[str, object]) -> Mapping[str, object]:
        path = Path(self._source(policy)["arena_pause_request"])
        if not path.exists():
            return {"path": str(path), "active": False, "state": "absent"}
        try:
            document, digest_value, _ = _read_json_with_digest(
                path,
                name="arena pause request",
            )
        except TerminalBoundaryManifestError as error:
            return {
                "path": str(path),
                "active": True,
                "state": "unsafe",
                "error": str(error),
            }
        state = document.get("state")
        valid = (
            document.get("schema_version") == 1
            and document.get("protocol") == "coordinator-pause-v1"
            and state in {"requested", "active", "released", "cancelled"}
        )
        return {
            "path": str(path),
            "sha256": digest_value,
            "active": not valid or state in {"requested", "active"},
            "state": state if valid else "unsafe",
            "document": document,
        }

    def place_operator_hold(
        self, path: Path, document: Mapping[str, object]
    ) -> Mapping[str, object]:
        evidence = _write_immutable_json(path, document)
        return {"status": "active", **evidence}

    def final_replay_backup(self, policy: Mapping[str, object]) -> Mapping[str, object]:
        from scripts.replay_manifest_backup import create_backup_with_evidence

        source = self._source(policy)
        backup = _mapping(policy.get("backup"), name="backup policy")
        replay_backup_root = (
            Path(str(source["run_root"])) / "recovery" / "replay-manifest"
        )
        latest_path = replay_backup_root / "latest.json"
        frozen_boundary_ns = self._frozen_boundary_ns(policy)
        if latest_path.is_file():
            latest, _latest_sha256, _ = _read_json_with_digest(
                latest_path,
                name="latest replay backup",
            )
            created_ns = latest.get("created_ns")
            relative = latest.get("path")
            if (
                isinstance(created_ns, int)
                and not isinstance(created_ns, bool)
                and created_ns >= frozen_boundary_ns
                and isinstance(relative, str)
                and relative
            ):
                destination = (replay_backup_root / relative).resolve()
                if (
                    destination.parent == replay_backup_root
                    and destination.is_file()
                    and not destination.is_symlink()
                    and destination.stat().st_size == latest.get("bytes")
                    and _sha256(destination) == latest.get("sha256")
                ):
                    return {
                        "status": "ok",
                        "path": str(destination),
                        "bytes": destination.stat().st_size,
                        "sha256": latest["sha256"],
                        "created_ns": created_ns,
                        "reused": True,
                    }
        _destination, evidence = create_backup_with_evidence(
            Path(str(source["run_root"])),
            retain=_positive_int(
                backup.get("replay_retain"),
                name="replay backup retention",
            ),
            max_total_bytes=_positive_int(
                backup.get("replay_max_total_bytes"),
                name="replay backup maximum bytes",
            ),
        )
        return {"status": "ok", **evidence}

    def disaster_snapshot(
        self,
        policy: Mapping[str, object],
        required_after_ns: int,
    ) -> Mapping[str, object]:
        from scripts.training_disaster_recovery import (
            create_snapshot,
            verify_snapshot,
        )

        source = self._source(policy)
        profile = _mapping(source.get("profile"), name="source profile")
        backup = _mapping(policy.get("backup"), name="backup policy")
        backup_root = Path(str(backup["disaster_backup_root"]))
        latest_path = backup_root / "latest.json"
        if latest_path.is_file():
            latest_report = verify_snapshot(
                latest_path,
                backup_root=backup_root,
            )
            identity = _mapping(
                source.get("run_identity"),
                name="source run identity",
            )
            latest_created_ns = latest_report.get("created_ns")
            if (
                latest_report.get("run_id") == identity.get("run_id")
                and latest_report.get("generation_family")
                == identity.get("generation_family")
                and isinstance(latest_created_ns, int)
                and not isinstance(latest_created_ns, bool)
                and latest_created_ns >= required_after_ns
            ):
                return {
                    **latest_report,
                    "path": str(latest_report["snapshot"]),
                    "reused": True,
                }
        snapshot = create_snapshot(
            source["run_root"],
            profile["path"],
            backup_root,
            replay_backup_retain=_positive_int(
                backup.get("replay_retain"),
                name="replay backup retention",
            ),
            expected_backup_mount=backup["disaster_backup_mount"],
        )
        report = verify_snapshot(
            snapshot,
            backup_root=backup_root,
        )
        return {**report, "path": str(snapshot)}

    def stop_source(self, policy: Mapping[str, object]) -> Mapping[str, object]:
        from startrain.continuity import SystemdUnitManager

        source = self._source(policy)
        unit = _mapping(source.get("unit"), name="source unit")
        manager = SystemdUnitManager(runner=self.runner)
        before = manager.status(str(unit["name"]))
        if before.active:
            manager.stop(str(unit["name"]))
        deadline = self.monotonic() + float(source["stop_timeout_seconds"])
        while True:
            status = manager.status(str(unit["name"]))
            if (
                status.query_error is None
                and not status.active
                and status.main_pid == 0
            ):
                return {
                    "status": "stopped",
                    "unit": unit["name"],
                    "previous_main_pid": before.main_pid,
                    "unit_status": status.as_dict(),
                }
            if self.monotonic() >= deadline:
                raise TimeoutError(
                    f"source unit did not stop gracefully: {unit['name']}"
                )
            self.sleep(0.25)

    @staticmethod
    def _pid_live(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _process_group_live(pid: int) -> bool:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def prove_source_release(
        self,
        policy: Mapping[str, object],
        source_evidence: Mapping[str, object],
        stop_evidence: Mapping[str, object],
    ) -> Mapping[str, object]:
        from startrain.continuity import SystemdUnitManager

        del stop_evidence
        source = self._source(policy)
        unit = _mapping(source.get("unit"), name="source unit")
        status = SystemdUnitManager(runner=self.runner).status(str(unit["name"]))
        coordinator_lock = Path(source["coordinator_lock"])
        coordinator, coordinator_sha256, _ = _read_json_with_digest(
            Path(source["coordinator_status"]),
            name="final coordinator status",
        )
        raw_pids = source_evidence.get("process_ids")
        pids = (
            [pid for pid in raw_pids if type(pid) is int and pid > 0]
            if isinstance(raw_pids, list)
            else []
        )
        process_groups_released = all(
            not self._pid_live(pid) and not self._process_group_live(pid)
            for pid in pids
        )
        workers = coordinator.get("workers")
        workers_released = isinstance(workers, dict) and all(
            isinstance(worker, dict)
            and worker.get("pid") is None
            and worker.get("state") != "unkillable"
            for worker in workers.values()
        )
        gpu_ids = source["gpu_ids"]
        gpu_processes: list[str] = []
        gpu_query_error = None
        if gpu_ids:
            try:
                completed = self.runner(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                gpu_query_error = f"{type(error).__name__}: {error}"
            else:
                if completed.returncode == 0:
                    gpu_processes = [
                        line.strip()
                        for line in completed.stdout.splitlines()
                        if line.strip()
                    ]
                else:
                    gpu_query_error = (
                        completed.stderr.strip()
                        or f"nvidia-smi exited {completed.returncode}"
                    )
        gpus_released = not gpu_ids or (gpu_query_error is None and not gpu_processes)
        return {
            "status": "verified",
            "service_inactive": (status.query_error is None and not status.active),
            "main_pid_released": status.main_pid == 0,
            "coordinator_lock_released": not coordinator_lock.exists(),
            "process_groups_released": process_groups_released and workers_released,
            "gpus_released": gpus_released,
            "checked_process_ids": pids,
            "gpu_processes": gpu_processes,
            "gpu_query_error": gpu_query_error,
            "coordinator_status_sha256": coordinator_sha256,
        }

    @staticmethod
    def _snapshot_tree(destination: Path) -> list[dict[str, object]]:
        files = []
        for path in sorted(destination.rglob("*")):
            if path.is_symlink():
                raise TerminalBoundaryExecutionError(
                    f"champion snapshot contains a symlink: {path}"
                )
            if not path.is_file():
                continue
            files.append(
                {
                    "path": str(path.relative_to(destination)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        if not files:
            raise TerminalBoundaryExecutionError("champion snapshot is empty")
        return files

    def export_champion(
        self,
        policy: Mapping[str, object],
        winner_snapshot: Mapping[str, object],
    ) -> Mapping[str, object]:
        from starserve.snapshot import export_champion_snapshot
        from startrain.checkpoint import load_model_manifest

        source = self._source(policy)
        profile = _mapping(source.get("profile"), name="source profile")
        snapshot = _mapping(policy.get("snapshot"), name="champion snapshot")
        destination = Path(snapshot["destination"])
        pin_path = Path(snapshot["pin_path"])
        champion_pointer = Path(source["champion_pointer"])
        champion = _mapping(winner_snapshot.get("champion"), name="winner champion")
        if not destination.exists():
            exported = export_champion_snapshot(
                champion_pointer,
                profile["path"],
                destination,
            )
        else:
            loaded = load_model_manifest(destination / "champion.json")
            if (
                loaded.model_identity != champion.get("model_identity")
                or loaded.model_step != champion.get("model_step")
                or loaded.role != "champion"
            ):
                raise TerminalBoundaryExecutionError(
                    "existing champion snapshot is incomplete or incompatible"
                )
            exported = {
                "destination": str(destination),
                "model_identity": loaded.model_identity,
                "model_step": loaded.model_step,
                "role": loaded.role,
            }
        files = self._snapshot_tree(destination)
        pin = {
            "schema_version": SCHEMA_VERSION,
            "report": "startrain-terminal-boundary-champion-snapshot",
            "status": "verified",
            "destination": str(destination),
            "model_identity": champion["model_identity"],
            "model_step": champion["model_step"],
            "files": files,
        }
        pin_evidence = _write_immutable_json(pin_path, pin, mode=0o440)
        return {
            "status": "verified",
            **dict(exported),
            "pin": pin_evidence,
            "file_count": len(files),
        }

    def _runtime_effective_optimizer(
        self,
        policy: Mapping[str, object],
    ) -> dict[str, object]:
        import torch

        source = self._source(policy)
        learner_root = Path(str(source["run_root"])) / "learner"
        recovery_path = learner_root / "recovery.json"
        recovery, recovery_sha256, _ = _read_json_with_digest(
            recovery_path,
            name="runtime recovery pointer",
        )
        checkpoint_value = _string(
            recovery.get("checkpoint"),
            name="runtime recovery checkpoint",
        )
        checkpoint = (learner_root / checkpoint_value).resolve()
        if checkpoint.parent != (learner_root / "recovery").resolve():
            raise TerminalBoundaryExecutionError(
                "runtime recovery checkpoint escaped the recovery directory"
            )
        expected_sha256 = _digest(
            recovery.get("checkpoint_sha256"),
            name="runtime recovery checkpoint SHA-256",
        )
        expected_bytes = _positive_int(
            recovery.get("checkpoint_bytes"),
            name="runtime recovery checkpoint bytes",
        )
        observed_sha256, observed_bytes = _hash_regular_file(
            checkpoint,
            name="runtime recovery checkpoint",
        )
        if (
            observed_bytes != expected_bytes
            or observed_sha256 != expected_sha256
        ):
            raise TerminalBoundaryExecutionError(
                "runtime recovery checkpoint changed before calibration"
            )
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, Mapping):
            raise TerminalBoundaryExecutionError(
                "runtime recovery checkpoint payload is invalid"
            )
        optimizer = _mapping(payload.get("optimizer"), name="runtime optimizer state")
        groups = optimizer.get("param_groups")
        scheduler = _mapping(payload.get("scheduler"), name="runtime scheduler state")
        base_lrs = scheduler.get("base_lrs")
        if (
            not isinstance(groups, list)
            or not groups
            or not isinstance(base_lrs, list)
            or len(base_lrs) != len(groups)
        ):
            raise TerminalBoundaryExecutionError(
                "runtime optimizer/scheduler group state is incomplete"
            )
        rates: dict[str, set[float]] = {"muon": set(), "adamw": set()}
        for index, raw_group in enumerate(groups):
            group = _mapping(raw_group, name=f"runtime optimizer group {index}")
            algorithm = group.get("algorithm")
            if algorithm not in rates:
                raise TerminalBoundaryExecutionError(
                    "runtime optimizer contains an unexpected algorithm"
                )
            initial_lr = _positive_number(
                group.get("initial_lr"),
                name=f"runtime {algorithm} initial learning rate",
            )
            scheduler_lr = _positive_number(
                base_lrs[index],
                name=f"runtime {algorithm} scheduler base learning rate",
            )
            if not math.isclose(initial_lr, scheduler_lr, rel_tol=0.0, abs_tol=0.0):
                raise TerminalBoundaryExecutionError(
                    "runtime optimizer and scheduler base learning rates differ"
                )
            rates[str(algorithm)].add(initial_lr)
        if len(rates["muon"]) != 1 or len(rates["adamw"]) != 1:
            raise TerminalBoundaryExecutionError(
                "runtime optimizer learning rates are not unique per algorithm"
            )
        return {
            "muon_lr": next(iter(rates["muon"])),
            "adamw_lr": next(iter(rates["adamw"])),
            "source_profile_path": str(
                _mapping(source.get("profile"), name="source profile")["path"]
            ),
            "source_profile_sha256": str(
                _mapping(source.get("profile"), name="source profile")["sha256"]
            ),
            "recovery_pointer": {
                "path": str(recovery_path),
                "sha256": recovery_sha256,
            },
            "recovery_checkpoint": {
                "path": str(checkpoint),
                "sha256": expected_sha256,
                "bytes": expected_bytes,
            },
            "scheduler_last_epoch": scheduler.get("last_epoch"),
            "source": "recovery_optimizer_initial_lr_and_scheduler_base_lrs",
        }

    def prepare_calibration(
        self,
        policy: Mapping[str, object],
        winner_snapshot: Mapping[str, object],
    ) -> Mapping[str, object]:
        from scripts.prepare_elo_ablation import prepare_elo_ablation

        source = self._source(policy)
        calibration = _mapping(policy.get("calibration"), name="calibration")
        output_dir = Path(calibration["output_dir"])
        plan_path = output_dir / "ablation-plan.json"
        runtime_optimizer = self._runtime_effective_optimizer(policy)
        if plan_path.is_file():
            plan, digest_value, data = _read_json_with_digest(
                plan_path,
                name="existing calibration plan",
            )
            if (
                plan.get("source_winner_snapshot") != dict(winner_snapshot)
                or plan.get("runtime_effective_optimizer") != runtime_optimizer
            ):
                raise TerminalBoundaryExecutionError(
                    "existing calibration plan has a different terminal winner"
                )
            return {
                "status": "existing",
                "path": str(plan_path),
                "sha256": digest_value,
                "bytes": len(data),
            }
        if output_dir.exists():
            raise TerminalBoundaryExecutionError(
                "calibration output exists without a complete plan"
            )
        base_config = _mapping(
            calibration.get("base_config"), name="calibration base config"
        )
        plan = prepare_elo_ablation(
            base_config=Path(base_config["path"]),
            output_dir=output_dir,
            run_root_parent=Path(calibration["run_root_parent"]),
            run_id=str(calibration["run_id"]),
            source_run_root=Path(source["run_root"]),
            prefix=str(calibration["prefix"]),
            seed=int(calibration["seed"]),
            wall_budget_hours=float(calibration["wall_budget_hours"]),
            leaf_budget=int(calibration["leaf_budget"]),
            guard_floor_elo=float(calibration["guard_floor_elo"]),
            treatments=tuple(calibration["treatments"]),
            winner_snapshot=winner_snapshot,
            futility_policy=(
                calibration.get("futility_policy")
                if isinstance(calibration.get("futility_policy"), Mapping)
                else None
            ),
            guard_rings=(
                tuple(calibration["guard_rings"])
                if isinstance(calibration.get("guard_rings"), list)
                else None
            ),
            suite=(
                str(calibration["suite"])
                if calibration.get("suite") is not None
                else None
            ),
            runtime_effective_optimizer=runtime_optimizer,
        )
        return {
            "status": "prepared",
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
            "plan_report": plan.get("report"),
        }

    def run_frozen_calibration(
        self,
        policy: Mapping[str, object],
        plan: Mapping[str, object],
        winner_snapshot: Mapping[str, object],
    ) -> Mapping[str, object]:
        from scripts.run_frozen_replay_optimizer_calibration_queue import (
            run_calibration_queue,
        )

        source = self._source(policy)
        calibration = _mapping(policy.get("calibration"), name="calibration")
        frozen = _mapping(
            calibration.get("frozen_replay"),
            name="frozen-replay calibration",
        )
        plan_document = _mapping(plan.get("plan"), name="calibration plan")
        replay_cutoff = plan_document.get("source_replay_cutoff")
        if type(replay_cutoff) is not int or replay_cutoff <= 0:
            raise TerminalBoundaryExecutionError(
                "calibration plan has no positive frozen replay cutoff"
            )
        champion_artifact = _mapping(
            winner_snapshot.get("champion_pointer_artifact"),
            name="winner champion pointer",
        )
        batch_size = frozen.get("batch_size")
        if batch_size is not None and type(batch_size) is not int:
            raise TerminalBoundaryManifestError(
                "frozen-replay batch size must be an integer"
            )
        queue_state = run_calibration_queue(
            plan_path=Path(str(plan["path"])),
            champion=Path(str(champion_artifact["path"])),
            replay_root=Path(str(source["run_root"])) / "replay",
            replay_cutoff=replay_cutoff,
            output_root=Path(str(frozen["output_root"])),
            steps=int(frozen["steps"]),
            device=str(frozen["device"]),
            budget_h100_hours=float(frozen["budget_h100_hours"]),
            screen_plan_path=Path(str(frozen["screen_plan_path"])),
            screen_wall_budget_hours=float(frozen["screen_wall_budget_hours"]),
            screen_leaf_budget=int(frozen["screen_leaf_budget"]),
            batch_size=batch_size,
            evaluation_batch_size=int(frozen["evaluation_batch_size"]),
            max_samples=int(frozen["max_samples"]),
            holdout_fraction=float(frozen["holdout_fraction"]),
            seed=int(calibration["seed"]),
            checkpoint_interval=int(frozen["checkpoint_interval"]),
        )
        if queue_state.get("status") != "completed":
            raise TerminalBoundaryExecutionError(
                "frozen-replay calibration did not complete"
            )
        comparison = _mapping(
            queue_state.get("comparison"),
            name="frozen calibration comparison",
        )
        return {
            "status": "completed",
            "queue_state": str(Path(str(frozen["output_root"])) / "queue-state.json"),
            "comparison": comparison,
            "selection": _mapping(
                comparison.get("selection"),
                name="frozen calibration selection",
            ),
            "screen_plan": queue_state.get("screen_plan"),
        }

    def fork_calibration(
        self,
        policy: Mapping[str, object],
        plan: Mapping[str, object],
        treatment: Mapping[str, object],
        winner_snapshot: Mapping[str, object],
    ) -> Mapping[str, object]:
        from scripts.fork_elo_ablation import fork_elo_ablation

        del policy
        run_root = Path(str(treatment["run_root"]))
        metadata_path = run_root / "ablation.json"
        if metadata_path.is_file():
            metadata, digest_value, data = _read_json_with_digest(
                metadata_path,
                name="existing calibration fork",
            )
            if metadata.get("source_winner_snapshot") != dict(winner_snapshot):
                raise TerminalBoundaryExecutionError(
                    "existing calibration fork has a different winner"
                )
            return {
                "status": "existing",
                "path": str(metadata_path),
                "sha256": digest_value,
                "bytes": len(data),
            }
        metadata = fork_elo_ablation(
            source_run_root=Path(str(winner_snapshot["run_root"])),
            plan_path=Path(str(plan["path"])),
            treatment=str(treatment["treatment"]),
        )
        return {
            "status": "forked",
            "path": str(metadata_path),
            "treatment": metadata.get("treatment"),
        }

    def warm_start(
        self,
        run_root: Path,
        profile: Path,
        *,
        prepare_only: bool,
    ) -> Mapping[str, object]:
        if self._warm_starter is None:
            from scripts.prepare_champion_warm_start import (
                prepare_champion_warm_start,
            )

            starter = prepare_champion_warm_start
        else:
            starter = self._warm_starter
        if prepare_only:
            return starter(
                run_root,
                profile,
                prepare_only=True,
                apply=False,
                replace_existing=True,
            )
        return starter(
            run_root,
            profile,
            prepare_only=False,
            apply=True,
            replace_existing=True,
        )

    def prepare_runtime_ownership(
        self,
        policy: Mapping[str, object],
        plan: Mapping[str, object],
    ) -> Mapping[str, object]:
        import grp
        import pwd

        calibration = _mapping(policy.get("calibration"), name="calibration")
        user = _string(calibration.get("runtime_user"), name="calibration runtime user")
        group = _string(
            calibration.get("runtime_group"),
            name="calibration runtime group",
        )
        uid = pwd.getpwnam(user).pw_uid
        gid = grp.getgrnam(group).gr_gid
        document = _mapping(plan.get("plan"), name="calibration plan")
        roots = _sequence(document.get("treatments"), name="calibration treatments")
        changed = 0

        def chown_tree(root: Path) -> None:
            nonlocal changed
            for path in (root, *sorted(root.rglob("*"))):
                if path.is_symlink():
                    raise TerminalBoundaryExecutionError(
                        f"calibration root contains a symlink: {path}"
                    )
                os.chown(path, uid, gid, follow_symlinks=False)
                changed += 1

        profile_directories: set[Path] = set()
        for raw in roots:
            treatment = _mapping(raw, name="calibration treatment")
            root = _absolute_path(
                treatment.get("run_root"),
                name="calibration run root",
            )
            chown_tree(root)
            profile = _absolute_path(
                treatment.get("profile"),
                name="calibration treatment profile",
            )
            profile_directories.add(profile.parent)

        plan_path = _absolute_path(plan.get("path"), name="calibration plan path")
        plan_directory = plan_path.parent
        chown_tree(plan_directory)
        for path in sorted(plan_directory.rglob("*")):
            if path.is_file():
                path.chmod(0o440)
            elif path.is_dir():
                path.chmod(0o550)
        plan_directory.chmod(0o550)
        for profile_directory in sorted(profile_directories):
            if profile_directory == plan_directory:
                continue
            chown_tree(profile_directory)
            for path in sorted(profile_directory.rglob("*")):
                if path.is_file():
                    path.chmod(0o440)
                elif path.is_dir():
                    path.chmod(0o550)
            profile_directory.chmod(0o550)

        queue = self._queue(policy)
        queue_paths = [
            Path(str(queue[field])).expanduser().resolve()
            for field in (
                "deployment_manifest",
                "state_path",
                "comparison_output",
                "continuity_handoff_output",
            )
        ]
        queue_directory = queue_paths[0].parent
        queue_directory.mkdir(parents=True, exist_ok=True)
        os.chown(queue_directory, uid, gid)
        queue_directory.chmod(0o750)
        deployment = queue_paths[0]
        if not deployment.is_file() or deployment.is_symlink():
            raise TerminalBoundaryExecutionError(
                "queue deployment manifest is missing before ownership handoff"
            )
        os.chown(deployment, uid, gid)
        deployment.chmod(0o440)
        changed += 2
        return {
            "status": "verified",
            "user": user,
            "group": group,
            "uid": uid,
            "gid": gid,
            "paths_updated": changed,
        }

    def generate_queue_manifest(
        self,
        policy: Mapping[str, object],
        plan: Mapping[str, object],
    ) -> Mapping[str, object]:
        from scripts.run_elo_ablation_queue import generate_deployment_manifest

        queue = self._queue(policy)
        output = Path(queue["deployment_manifest"])
        if output.is_file():
            data = _read_json_bytes(output, name="existing queue deployment")
            return {
                "status": "existing",
                "path": str(output),
                "sha256": _sha256_bytes(data),
                "bytes": len(data),
            }
        queue_unit = _mapping(queue.get("queue_unit"), name="queue unit")
        finalize_unit = _mapping(queue.get("finalize_unit"), name="finalize unit")
        environment = _mapping(queue.get("environment"), name="queue environment")
        generate_deployment_manifest(
            plan_path=Path(str(plan["path"])),
            output_path=output,
            training_dir=Path(queue["training_dir"]),
            queue_unit=Path(queue_unit["path"]),
            finalize_unit=Path(finalize_unit["path"]),
            environment_file=Path(environment["path"]),
            state_path=Path(queue["state_path"]),
            comparison_output=Path(queue["comparison_output"]),
            continuity_handoff_output=Path(queue["continuity_handoff_output"]),
            execution_lock_path=Path(queue["execution_lock_path"]),
            source_commit=str(queue["source_commit"]),
            orchestrator=str(queue["orchestrator"]),
            poll_seconds=float(queue["poll_seconds"]),
            max_transient_retries=int(queue["max_transient_retries"]),
            retry_delay_seconds=float(queue["retry_delay_seconds"]),
            continue_after_fatal=bool(queue["continue_after_fatal"]),
            provisioned_gpus=int(queue["provisioned_gpus"]),
        )
        data = _read_json_bytes(output, name="queue deployment manifest")
        return {
            "status": "generated",
            "path": str(output),
            "sha256": _sha256_bytes(data),
            "bytes": len(data),
        }

    def verify_queue_manifest(
        self,
        policy: Mapping[str, object],
        manifest: Mapping[str, object],
    ) -> Mapping[str, object]:
        from scripts.run_elo_ablation_queue import verify_deployment_manifest

        del policy
        return verify_deployment_manifest(
            Path(str(manifest["path"])),
            expected_manifest_sha256=str(manifest["sha256"]),
        )

    def launch_queue(
        self,
        policy: Mapping[str, object],
        activation_manifest: Path,
    ) -> Mapping[str, object]:
        from startrain.continuity import SystemdUnitManager

        activation = _json_object(
            _read_json_bytes(
                activation_manifest,
                name="queue activation manifest",
            ),
            name="queue activation manifest",
        )
        activation_policy = _mapping(activation.get("policy"), name="activation policy")
        verify_queue_activation_manifest(
            activation_manifest,
            policy_path=Path(
                _string(
                    activation_policy.get("path"),
                    name="activation policy path",
                )
            ),
        )
        queue = self._queue(policy)
        unit = _mapping(queue.get("queue_unit"), name="queue unit")
        manager = SystemdUnitManager(runner=self.runner)
        queue_state_path = Path(queue["state_path"])

        def unit_is_pinned(status: object) -> bool:
            return (
                getattr(status, "query_error", None) is None
                and Path(str(getattr(status, "fragment_path", ""))).resolve()
                == Path(str(unit["path"])).resolve()
            )

        def queue_status() -> str | None:
            if not queue_state_path.is_file():
                return None
            state, _digest_value, _ = _read_json_with_digest(
                queue_state_path,
                name="queue state",
            )
            if state.get("manifest") != queue["deployment_manifest"] or state.get(
                "manifest_sha256"
            ) != _sha256(Path(str(queue["deployment_manifest"]))):
                raise TerminalBoundaryExecutionError(
                    "durable queue state belongs to another deployment"
                )
            value = state.get("queue_status")
            return str(value) if isinstance(value, str) else None

        status = manager.status(str(unit["name"]))
        existing_queue_status = queue_status()
        if existing_queue_status == "failed":
            raise TerminalBoundaryExecutionError(
                "durable queue state is already failed"
            )
        if not unit_is_pinned(status):
            raise TerminalBoundaryExecutionError(
                "queue systemd unit fragment differs from the pinned unit"
            )
        if existing_queue_status == "completed":
            return {
                "status": "already_launched",
                "unit": unit["name"],
                "queue_status": existing_queue_status,
                "unit_status": status.as_dict(),
            }
        if not status.active:
            manager.start(str(unit["name"]))
        deadline = self.monotonic() + _positive_number(
            queue.get("launch_timeout_seconds"),
            name="queue launch timeout",
        )
        while True:
            status = manager.status(str(unit["name"]))
            observed_queue_status = queue_status()
            if not unit_is_pinned(status):
                raise TerminalBoundaryExecutionError(
                    "queue systemd unit fragment changed during launch"
                )
            if observed_queue_status in {"running", "completed"}:
                return {
                    "status": "launched",
                    "unit": unit["name"],
                    "unit_status": status.as_dict(),
                    "queue_status": observed_queue_status,
                    "activation_manifest": str(activation_manifest),
                }
            if (
                observed_queue_status == "failed"
                or status.active_state == "failed"
                or status.query_error is not None
            ):
                raise TerminalBoundaryExecutionError(
                    "queue unit failed before a durable launch was proven"
                )
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "queue unit did not become active or publish durable state"
                )
            self.sleep(min(0.25, remaining))

    def request_continuity_fallback(
        self,
        policy: Mapping[str, object],
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        from startrain.continuity import reconcile_training_continuity

        fallback = _mapping(policy.get("fallback"), name="continuity fallback")
        path = Path(fallback["handoff_path"])
        if path.is_file():
            existing, digest_value, _ = _read_json_with_digest(
                path,
                name="existing fallback request",
            )
            if existing.get("failure_id") == request.get("failure_id"):
                persisted = {
                    "path": str(path),
                    "sha256": digest_value,
                    "status": "existing",
                }
            else:
                archive = (
                    path.parent
                    / "terminal-boundary-failures"
                    / (f"{digest_value}.json")
                )
                _write_immutable_json(archive, existing, mode=0o440)
                atomic_json(path, request)
                persisted = {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "status": "replaced_and_archived",
                }
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, request)
            persisted = {
                "path": str(path),
                "sha256": _sha256(path),
                "status": "created",
            }
        hold_path = Path(
            _absolute_path(
                policy.get("operator_hold_path"),
                name="operator hold path",
            )
        )
        hold_release: dict[str, object] = {
            "path": str(hold_path),
            "status": "absent",
        }
        if hold_path.exists():
            hold, hold_sha256, _ = _read_json_with_digest(
                hold_path,
                name="terminal-boundary operator hold",
            )
            request_source = _mapping(
                request.get("source"),
                name="fallback request source",
            )
            if hold.get("policy_sha256") != request_source.get("policy_sha256"):
                raise TerminalBoundaryExecutionError(
                    "operator hold is not owned by this terminal-boundary policy"
                )
            hold_path.unlink()
            directory = os.open(hold_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            hold_release = {
                "path": str(hold_path),
                "status": "released",
                "sha256": hold_sha256,
            }
        continuity = _mapping(
            fallback.get("continuity_manifest"),
            name="continuity manifest",
        )
        reconciliation = reconcile_training_continuity(continuity["path"])
        return {
            "status": "requested",
            "request": persisted,
            "operator_hold": hold_release,
            "reconciliation": reconciliation,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("probe", "verify", "run", "recover"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "probe":
            report = probe_terminal_boundary_policy(arguments.manifest)
            status = 0 if report["should_run"] is True else 1
        elif arguments.command == "verify":
            report = verify_terminal_boundary_policy(arguments.manifest)
            status = 0
        elif arguments.command == "recover":
            report = recover_terminal_boundary_pipeline(arguments.manifest)
            status = 0
        else:
            report = run_terminal_boundary_pipeline(arguments.manifest)
            if report.get("status") == "completed":
                status = 0
            elif report.get("status") == "waiting":
                status = 0
            elif report.get("status") in {"blocked", "failed"}:
                status = 3
            else:
                status = 75
    except TerminalBoundaryBusyError as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "report": STATE_REPORT,
            "status": "busy",
            "error": str(error),
        }
        status = 255 if arguments.command == "probe" else 75
    except TerminalBoundaryExecutionError as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "report": STATE_REPORT,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        status = 255 if arguments.command == "probe" else 3
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        TerminalBoundaryError,
    ) as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "report": STATE_REPORT,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
        status = 255 if arguments.command == "probe" else 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
