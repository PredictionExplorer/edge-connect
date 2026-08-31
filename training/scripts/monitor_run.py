#!/usr/bin/env python3
"""Print periodic, read-only health summaries for one training run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from startrain.config import load_config
from startrain.continuity import ContinuityError, load_continuity_manifest

if __package__:
    from .validate_continuous_profile import validate_continuous_config
else:
    from validate_continuous_profile import validate_continuous_config

SEVERITY = {"OK": 0, "WARN": 1, "ERROR": 2}
CONTINUITY_STALE_SECONDS = 180.0
DISASTER_BACKUP_STALE_SECONDS = 30.0 * 60.0
STRENGTH_REPORT_WARN_SECONDS = 20.0 * 60.0
STRENGTH_REPORT_ERROR_SECONDS = 45.0 * 60.0
STRENGTH_REPORT_CLOCK_SKEW_SECONDS = 5.0
DEFAULT_TELEMETRY_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_TELEMETRY_RETAIN_FILES = 7
_DIGEST_CACHE: dict[Path, tuple[int, int, int, str]] = {}
_ARENA_RESULT_CACHE: dict[Path, tuple[int, int, dict[str, object]]] = {}


@dataclass(frozen=True, slots=True)
class MonitorTarget:
    run_root: Path
    profile_path: Path | None
    unit: str | None
    continuity_state_path: Path | None
    disaster_backup_root: Path | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--unit")
    parser.add_argument(
        "--continuity-manifest",
        type=Path,
        help="resolve the active workload from a pinned continuity manifest",
    )
    parser.add_argument(
        "--continuity-state",
        type=Path,
        help="host-level continuity-state JSON outside the run root",
    )
    parser.add_argument(
        "--disaster-backup-root",
        type=Path,
        help="content-addressed disaster-recovery backup root",
    )
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--format", choices=("text", "jsonl"), default="text")
    parser.add_argument(
        "--telemetry-output",
        type=Path,
        help="append every full snapshot as durable JSONL; use --interval 5 for 5s GPU telemetry",
    )
    parser.add_argument(
        "--telemetry-max-bytes",
        type=int,
        default=DEFAULT_TELEMETRY_MAX_BYTES,
        help="rotate durable telemetry before the active JSONL exceeds this size",
    )
    parser.add_argument(
        "--telemetry-retain-files",
        type=int,
        default=DEFAULT_TELEMETRY_RETAIN_FILES,
        help="number of complete rotated telemetry JSONL files to retain",
    )
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path, *, attempts: int = 3) -> dict[str, object] | None:
    for attempt in range(attempts):
        try:
            with path.open("rb") as stream:
                payload = json.load(stream)
            return payload if isinstance(payload, dict) else None
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            if attempt + 1 < attempts:
                time.sleep(0.02)
    return None


def resolve_monitor_target(
    continuity_manifest_path: Path,
    *,
    run_root: Path | None = None,
    profile_path: Path | None = None,
    unit: str | None = None,
    continuity_state_path: Path | None = None,
    disaster_backup_root: Path | None = None,
) -> MonitorTarget:
    """Resolve and bind monitor inputs to the manifest's active workload."""

    manifest = load_continuity_manifest(continuity_manifest_path)
    state = _read_json(manifest.state_path)
    if (
        state is None
        or state.get("format") != "startrain.training-continuity-state"
        or state.get("schema_version") != 1
        or state.get("manifest_sha256") != manifest.sha256
    ):
        raise ValueError(
            f"continuity state is missing, invalid, or stale: {manifest.state_path}"
        )
    active_workload_id = state.get("active_workload_id")
    if not isinstance(active_workload_id, str) or not active_workload_id:
        raise ValueError("continuity state does not identify an active workload")
    workload = manifest.workload(active_workload_id)
    if (
        state.get("active_profile_sha256") != workload.profile_sha256
        or state.get("active_run_root_sha256") != workload.run_root_sha256
    ):
        raise ValueError(
            "continuity state active workload hashes differ from the manifest"
        )

    def path_value(
        explicit: Path | None,
        expected: Path,
        *,
        option: str,
    ) -> Path:
        if explicit is not None and explicit.expanduser().resolve() != expected:
            raise ValueError(f"{option} conflicts with the active continuity workload")
        return expected

    resolved_run_root = path_value(
        run_root,
        workload.run_root,
        option="--run-root",
    )
    resolved_profile = path_value(
        profile_path,
        workload.profile_path,
        option="--profile",
    )
    if unit is not None and unit != workload.unit:
        raise ValueError("--unit conflicts with the active continuity workload")
    resolved_state = path_value(
        continuity_state_path,
        manifest.state_path,
        option="--continuity-state",
    )
    protection = workload.protection
    expected_disaster_root = (
        protection.disaster_backup_root if protection is not None else None
    )
    if (
        disaster_backup_root is not None
        and expected_disaster_root is not None
        and disaster_backup_root.expanduser().resolve() != expected_disaster_root
    ):
        raise ValueError(
            "--disaster-backup-root conflicts with the active continuity workload"
        )
    resolved_disaster_root = (
        expected_disaster_root
        if expected_disaster_root is not None
        else (
            disaster_backup_root.expanduser().resolve()
            if disaster_backup_root is not None
            else None
        )
    )
    return MonitorTarget(
        run_root=resolved_run_root,
        profile_path=resolved_profile,
        unit=workload.unit,
        continuity_state_path=resolved_state,
        disaster_backup_root=resolved_disaster_root,
    )


def _continuity_status(
    path: Path | None,
    *,
    now_ns: int,
) -> dict[str, object]:
    if path is None:
        return {"configured": False}
    source = path.expanduser().resolve()
    payload = _read_json(source)
    if (
        payload is None
        or payload.get("format") != "startrain.training-continuity-state"
        or payload.get("schema_version") != 1
    ):
        return {
            "configured": True,
            "valid": False,
            "state_path": str(source),
        }
    idle_since = payload.get("productive_idle_since_ns")
    idle_seconds = (
        max(0.0, (now_ns - idle_since) / 1_000_000_000)
        if isinstance(idle_since, int) and not isinstance(idle_since, bool)
        else None
    )
    reconciled_ns = payload.get("last_reconciled_ns")
    reconciliation_age_seconds = (
        max(0.0, (now_ns - reconciled_ns) / 1_000_000_000)
        if isinstance(reconciled_ns, int) and not isinstance(reconciled_ns, bool)
        else None
    )
    return {
        "configured": True,
        "valid": True,
        "state_path": str(source),
        "manifest_sha256": payload.get("manifest_sha256"),
        "revision": payload.get("revision"),
        "updated_ns": payload.get("updated_ns"),
        "last_reconciled_ns": reconciled_ns,
        "reconciliation_age_seconds": reconciliation_age_seconds,
        "phase": payload.get("phase"),
        "primary_workload_id": payload.get("primary_workload_id"),
        "desired_workload_id": payload.get("desired_workload_id"),
        "active_workload_id": payload.get("active_workload_id"),
        "selected_lkg_workload_id": payload.get("selected_lkg_workload_id"),
        "active_profile_sha256": payload.get("active_profile_sha256"),
        "active_run_root_sha256": payload.get("active_run_root_sha256"),
        "fallback_attempts": payload.get("fallback_attempts"),
        "productive_idle_since_ns": idle_since,
        "productive_idle_seconds": idle_seconds,
        "hardware": payload.get("hardware"),
        "execution": payload.get("execution"),
        "protection": payload.get("protection"),
        "blocked_reason": payload.get("blocked_reason"),
        "last_failure": payload.get("last_failure"),
        "last_handoff": payload.get("last_handoff"),
        "last_transition": payload.get("last_transition"),
        "last_alert": payload.get("last_alert"),
        "quarantine_records": payload.get("quarantine_records"),
    }


def _disaster_recovery_status(
    path: Path | None,
    *,
    run_id: object,
    run_root: Path | None = None,
    now_ns: int,
) -> dict[str, object]:
    if path is None:
        return {"configured": False}
    root = path.expanduser().resolve()
    active_root = run_root.expanduser().resolve() if run_root is not None else None
    run_identity = (
        _read_json(active_root / "run.json") if active_root is not None else None
    )
    namespace = _read_json(root / "namespace.json")
    if (
        not isinstance(run_id, str)
        or not run_id
        or active_root is None
        or run_identity is None
        or run_identity.get("run_id") != run_id
        or namespace is None
        or namespace.get("report") != "startrain-disaster-recovery-namespace"
        or namespace.get("schema_version") != 1
        or namespace.get("run_id") != run_id
        or namespace.get("generation_family") != run_identity.get("generation_family")
        or namespace.get("source_run_root") != str(active_root)
    ):
        return {
            "configured": True,
            "valid": False,
            "backup_root": str(root),
            "reason": "namespace_or_run_identity_invalid",
        }
    latest_path = root / "snapshots" / run_id / "latest.json"
    latest = _read_json(latest_path)
    if (
        latest is None
        or latest.get("report") != "startrain-disaster-recovery-latest"
        or latest.get("schema_version") != 1
        or latest.get("run_id") != run_id
    ):
        return {
            "configured": True,
            "valid": False,
            "backup_root": str(root),
            "latest_path": str(latest_path),
            "reason": "latest_pointer_invalid",
        }
    filename = latest.get("path")
    digest = latest.get("sha256")
    expected_bytes = latest.get("bytes")
    created_ns = latest.get("created_ns")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or isinstance(created_ns, bool)
        or not isinstance(created_ns, int)
        or created_ns <= 0
        or created_ns > now_ns
    ):
        return {
            "configured": True,
            "valid": False,
            "backup_root": str(root),
            "latest_path": str(latest_path),
            "reason": "latest_pointer_evidence_invalid",
        }
    snapshot_path = latest_path.parent / filename
    snapshot_valid, _ = _verified_artifact(
        snapshot_path,
        expected_bytes=expected_bytes,
        expected_sha256=digest,
    )
    snapshot = _read_json(snapshot_path) if snapshot_valid else None
    if (
        snapshot is None
        or snapshot.get("report") != "startrain-disaster-recovery-snapshot"
        or snapshot.get("schema_version") != 1
        or snapshot.get("run_id") != run_id
        or snapshot.get("created_ns") != created_ns
    ):
        return {
            "configured": True,
            "valid": False,
            "backup_root": str(root),
            "latest_path": str(latest_path),
            "snapshot_path": str(snapshot_path),
            "reason": "snapshot_invalid",
        }
    snapshot_age = max(0.0, (now_ns - created_ns) / 1_000_000_000)
    source = snapshot.get("source")
    if not isinstance(source, dict) or source.get("run_root") != str(active_root):
        return {
            "configured": True,
            "valid": False,
            "backup_root": str(root),
            "latest_path": str(latest_path),
            "snapshot_path": str(snapshot_path),
            "reason": "snapshot_workload_root_invalid",
        }
    replay_backup = source.get("replay_backup") if isinstance(source, dict) else None
    cutoff_ns = (
        replay_backup.get(
            "cutoff_started_ns",
            replay_backup.get("created_ns"),
        )
        if isinstance(replay_backup, dict)
        else None
    )
    cutoff_age = (
        max(0.0, (now_ns - cutoff_ns) / 1_000_000_000)
        if isinstance(cutoff_ns, int)
        and not isinstance(cutoff_ns, bool)
        and 0 < cutoff_ns <= created_ns
        else None
    )
    if cutoff_age is None:
        return {
            "configured": True,
            "valid": False,
            "backup_root": str(root),
            "latest_path": str(latest_path),
            "snapshot_path": str(snapshot_path),
            "reason": "snapshot_cutoff_invalid",
        }
    catalog = snapshot.get("catalog")
    return {
        "configured": True,
        "valid": True,
        "backup_root": str(root),
        "latest_path": str(latest_path),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": digest,
        "snapshot_created_ns": created_ns,
        "snapshot_age_seconds": snapshot_age,
        "source_cutoff_ns": cutoff_ns,
        "source_cutoff_age_seconds": cutoff_age,
        "catalog_files": len(catalog) if isinstance(catalog, dict) else None,
    }


def _latest_jsonl(
    path: Path,
    *,
    maximum_bytes: int = 2 * 1024 * 1024,
    predicate: Callable[[Mapping[str, object]], bool] | None = None,
):
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            start = max(0, size - maximum_bytes)
            stream.seek(start)
            data = stream.read(size - start)
    except OSError:
        return None
    if start and b"\n" in data:
        data = data.split(b"\n", 1)[1]
    lines = data.splitlines()
    if data and not data.endswith(b"\n") and lines:
        lines.pop()
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and (predicate is None or predicate(payload)):
            return payload
    return None


def _recent_jsonl(path: Path, *, maximum_bytes: int = 2 * 1024 * 1024):
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            start = max(0, size - maximum_bytes)
            stream.seek(start)
            data = stream.read(size - start)
    except OSError:
        return []
    if start and b"\n" in data:
        data = data.split(b"\n", 1)[1]
    lines = data.splitlines()
    if data and not data.endswith(b"\n") and lines:
        lines.pop()
    output = []
    for line in lines:
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            output.append(payload)
    return output


def _merge_interval_seconds(intervals: list[tuple[int, int]]) -> float:
    merged = 0
    end = 0
    for started, completed in sorted(intervals):
        if completed <= started:
            continue
        if started >= end:
            merged += completed - started
        elif completed > end:
            merged += completed - end
        end = max(end, completed)
    return merged / 1_000_000_000


def _actor_throughput_window(
    metrics_root: Path,
    *,
    now_ns: int,
    window_seconds: float = 3_600.0,
) -> dict[str, object]:
    cutoff_ns = now_ns - int(window_seconds * 1_000_000_000)
    all_records = []

    def required_int(row: Mapping[str, object], name: str) -> int:
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"actor metric {name} is not an integer")
        return value

    def counter_int(row: Mapping[str, object], name: str) -> int:
        value = row.get(name, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    for path in sorted(metrics_root.glob("actor-gpu-*.jsonl")):
        for row in _recent_jsonl(path):
            started = row.get("batch_started_ns")
            completed = row.get("batch_completed_ns")
            if (
                isinstance(started, int)
                and not isinstance(started, bool)
                and isinstance(completed, int)
                and not isinstance(completed, bool)
                and completed <= now_ns
                and completed > started
            ):
                all_records.append(row)

    counter_names = ("games", "samples", "evaluator_rows")
    cumulative_names = tuple(f"cumulative_{name}" for name in counter_names)
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    legacy_records = []
    for row in all_records:
        process_started = row.get("process_started_ns")
        if (
            isinstance(process_started, int)
            and not isinstance(process_started, bool)
            and all(
                isinstance(row.get(name), int) and not isinstance(row.get(name), bool)
                for name in cumulative_names
            )
        ):
            groups.setdefault(
                (str(row.get("worker")), process_started),
                [],
            ).append(row)
        elif required_int(row, "batch_completed_ns") >= cutoff_ns:
            legacy_records.append(row)

    totals = {name: 0 for name in counter_names}
    totals_by_gpu: dict[int, dict[str, int]] = {}
    workers_by_gpu: dict[int, set[str]] = {}
    partial_processes = []
    contributing_starts = []
    contributing_records = []

    def add_values(row: Mapping[str, object], values: Mapping[str, int]) -> None:
        gpu_id = row.get("gpu_id")
        for name, value in values.items():
            totals[name] += value
        if isinstance(gpu_id, int) and not isinstance(gpu_id, bool):
            gpu_totals = totals_by_gpu.setdefault(
                gpu_id, {name: 0 for name in counter_names}
            )
            for name, value in values.items():
                gpu_totals[name] += value
            workers_by_gpu.setdefault(gpu_id, set()).add(str(row.get("worker")))

    for (worker, process_started), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: required_int(row, "batch_completed_ns"))
        end = ordered[-1]
        if required_int(end, "batch_completed_ns") < cutoff_ns:
            continue
        baselines = [
            row
            for row in ordered
            if required_int(row, "batch_completed_ns") <= cutoff_ns
        ]
        if baselines:
            baseline = baselines[-1]
            values = {
                name: required_int(end, f"cumulative_{name}")
                - required_int(baseline, f"cumulative_{name}")
                for name in counter_names
            }
            start_ns = cutoff_ns
        elif process_started >= cutoff_ns:
            values = {
                name: required_int(end, f"cumulative_{name}") for name in counter_names
            }
            start_ns = process_started
        else:
            selected = [
                row
                for row in ordered
                if required_int(row, "batch_completed_ns") >= cutoff_ns
            ]
            values = {
                name: sum(counter_int(row, name) for row in selected)
                for name in counter_names
            }
            start_ns = max(
                cutoff_ns,
                min(required_int(row, "batch_started_ns") for row in selected),
            )
            partial_processes.append(worker)
        if any(value < 0 for value in values.values()):
            partial_processes.append(worker)
            continue
        add_values(end, values)
        contributing_starts.append(start_ns)
        contributing_records.extend(
            row
            for row in ordered
            if required_int(row, "batch_completed_ns") >= cutoff_ns
        )

    for row in legacy_records:
        values = {name: counter_int(row, name) for name in counter_names}
        add_values(row, values)
        contributing_starts.append(
            max(cutoff_ns, required_int(row, "batch_started_ns"))
        )
        contributing_records.append(row)
        partial_processes.append(str(row.get("worker")))

    observation_start_ns = (
        max(cutoff_ns, min(contributing_starts)) if contributing_starts else cutoff_ns
    )
    fleet_seconds = max(0.0, (now_ns - observation_start_ns) / 1_000_000_000)
    active_seconds = _merge_interval_seconds(
        [
            (
                max(cutoff_ns, required_int(row, "batch_started_ns")),
                min(now_ns, required_int(row, "batch_completed_ns")),
            )
            for row in contributing_records
        ]
    )
    by_gpu = {}
    for gpu_id, gpu_totals in sorted(totals_by_gpu.items()):
        by_gpu[str(gpu_id)] = {
            "wall_seconds": fleet_seconds,
            **gpu_totals,
            **{
                f"{name}_per_second": value / fleet_seconds if fleet_seconds else None
                for name, value in gpu_totals.items()
            },
            "workers": sorted(workers_by_gpu.get(gpu_id, set())),
        }
    return {
        "schema_version": 1,
        "method": "cumulative_counter_wall_window_v1",
        "requested_window_seconds": window_seconds,
        "observation_start_ns": observation_start_ns,
        "observation_end_ns": now_ns,
        "record_count": len(contributing_records),
        "partial_processes": sorted(set(partial_processes)),
        "active_interval_seconds": active_seconds,
        "fleet": {
            "wall_seconds": fleet_seconds,
            **totals,
            **{
                f"{key}_per_second": value / fleet_seconds if fleet_seconds else None
                for key, value in totals.items()
            },
        },
        "by_gpu": by_gpu,
    }


def _run_command(command: Sequence[str], *, timeout: float = 10.0):
    try:
        return subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def _arena_result_kind(result: Mapping[str, object]) -> str:
    value = result.get("result_kind")
    if value is None:
        return "promotion"
    return (
        value
        if value in ("promotion", "crossplay", "historical_crossplay")
        else "unknown"
    )


def _arena_result_category(result: Mapping[str, object]) -> str:
    kind = _arena_result_kind(result)
    return "crossplay" if kind in ("crossplay", "historical_crossplay") else kind


def _configured_ring_weights(
    profile: Mapping[str, object], step: object
) -> tuple[float, ...] | None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None
    ring_mixture = _mapping(_mapping(profile.get("orchestration")).get("ring_mixture"))
    stages = ring_mixture.get("step_weights")
    if not isinstance(stages, list):
        return None
    selected = None
    for raw_stage in stages:
        stage = _mapping(raw_stage)
        from_step = stage.get("from_step")
        weights = stage.get("weights")
        if (
            isinstance(from_step, int)
            and not isinstance(from_step, bool)
            and from_step <= step
            and isinstance(weights, list)
            and all(_number(weight) is not None for weight in weights)
        ):
            selected = tuple(float(weight) for weight in weights)
    return selected


def _age_seconds(timestamp_ns: object, now_ns: int) -> float | None:
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
        return None
    return max(0.0, (now_ns - timestamp_ns) / 1_000_000_000.0)


def _verified_artifact(
    path: Path, *, expected_bytes: object, expected_sha256: object
) -> tuple[bool, float | None]:
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        return False, None
    try:
        before = path.stat()
        if before.st_size != expected_bytes:
            return False, None
        key = (before.st_ino, before.st_mtime_ns, before.st_size)
        cached = _DIGEST_CACHE.get(path)
        if cached is not None and cached[:3] == key:
            digest = cached[3]
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            after = path.stat()
            if (after.st_ino, after.st_mtime_ns, after.st_size) != key:
                return False, None
            digest = hasher.hexdigest()
            _DIGEST_CACHE[path] = (*key, digest)
        return digest == expected_sha256, before.st_mtime
    except OSError:
        return False, None


def _systemd_status(unit: str | None) -> dict[str, object]:
    if not unit:
        return {"configured": False}
    completed = _run_command(
        [
            "systemctl",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "NRestarts",
            "-p",
            "ActiveEnterTimestamp",
        ]
    )
    if completed is None or completed.returncode != 0:
        return {"configured": True, "query_error": True}
    values = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return {
        "configured": True,
        "active_state": values.get("ActiveState"),
        "sub_state": values.get("SubState"),
        "main_pid": int(values.get("MainPID", "0") or 0),
        "restart_count": int(values.get("NRestarts", "0") or 0),
        "active_since": values.get("ActiveEnterTimestamp"),
    }


def _gpu_status() -> tuple[list[dict[str, object]], str | None]:
    fields = (
        "index",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "ecc.errors.uncorrected.volatile.total",
        "ecc.errors.uncorrected.aggregate.sram",
        "ecc.errors.uncorrected.aggregate.dram",
    )
    completed = _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if completed is None or completed.returncode != 0:
        return [], "gpu_query_failed"
    output = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        row: dict[str, object] = {"index": int(values[0])}
        for name, value in zip(fields[1:], values[1:], strict=True):
            row[name] = None if value in ("N/A", "[N/A]") else _number_string(value)
        output.append(row)
    return output, None


def _number_string(value: str) -> float | None:
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _replay_status(
    path: Path,
    *,
    current_model_step: int | None = None,
) -> tuple[dict[str, object], str | None]:
    if not path.is_file():
        return {}, "replay_manifest_missing"
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        states = {
            str(row["state"]): {
                "shards": int(row["shards"]),
                "samples": int(row["samples"]),
            }
            for row in connection.execute(
                """
                SELECT state, COUNT(*) AS shards,
                       COALESCE(SUM(sample_count), 0) AS samples
                FROM shards GROUP BY state
                """
            )
        }
        rings = {
            str(row["ring"]): int(row["samples"])
            for row in connection.execute(
                """
                SELECT ring, COALESCE(SUM(sample_count), 0) AS samples
                FROM shards WHERE state = 'ready' GROUP BY ring
                """
            )
        }
        games = int(connection.execute("SELECT COUNT(*) FROM games").fetchone()[0])
        columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(shards)")
        }
        model_step_lag = None
        if current_model_step is not None and "model_step" in columns:
            step_rows = [
                (int(row["model_step"]), int(row["samples"]))
                for row in connection.execute(
                    """
                    SELECT model_step, COALESCE(SUM(sample_count), 0) AS samples
                    FROM shards
                    WHERE state = 'ready'
                    GROUP BY model_step
                    ORDER BY model_step DESC
                    """,
                )
                if int(row["samples"]) > 0
            ]
            total_samples = sum(samples for _step, samples in step_rows)
            ahead_samples = sum(
                samples for step, samples in step_rows if step > current_model_step
            )

            def weighted_lag_quantile(quantile: float) -> int | None:
                if not total_samples:
                    return None
                threshold = total_samples * quantile
                observed = 0
                for step, samples in step_rows:
                    observed += samples
                    if observed >= threshold:
                        return current_model_step - step
                return current_model_step - step_rows[-1][0]

            weighted_mean_lag = (
                sum(
                    (current_model_step - step) * samples for step, samples in step_rows
                )
                / total_samples
                if total_samples
                else None
            )
            model_step_lag = {
                "current_model_step": current_model_step,
                "ready_samples": total_samples,
                "ahead_samples": ahead_samples,
                "ahead_max_steps": (
                    max(
                        0,
                        max(
                            (step - current_model_step for step, _samples in step_rows),
                            default=0,
                        ),
                    )
                ),
                "minimum": (
                    current_model_step - step_rows[0][0] if step_rows else None
                ),
                "weighted_mean": weighted_mean_lag,
                "weighted_p50": weighted_lag_quantile(0.50),
                "weighted_p90": weighted_lag_quantile(0.90),
                "maximum": (
                    current_model_step - step_rows[-1][0] if step_rows else None
                ),
            }
        connection.rollback()
        connection.close()
    except (OSError, sqlite3.Error) as error:
        return {}, f"replay_query_failed:{type(error).__name__}"
    return {
        "states": states,
        "samples_by_ring": rings,
        "games": games,
        "model_step_lag": model_step_lag,
    }, None


def _arena_history(
    run_root: Path,
    *,
    limit: int = 5,
    started_ns: int | None = None,
) -> dict[str, object]:
    learner_root = run_root / "learner"
    steps: dict[str, int] = {}
    all_publications: dict[str, int] = {}
    publications: dict[str, int] = {}
    for row in _recent_jsonl(
        learner_root / "model-history.jsonl",
        maximum_bytes=16 * 1024 * 1024,
    ):
        identity = row.get("model_identity")
        step = row.get("model_step")
        published_ns = row.get("published_ns")
        if isinstance(identity, str) and isinstance(step, int):
            steps[identity] = step
        if (
            isinstance(identity, str)
            and isinstance(published_ns, int)
            and not isinstance(published_ns, bool)
        ):
            all_publications[identity] = published_ns
            if started_ns is None or published_ns >= started_ns:
                publications[identity] = published_ns
    for manifest_path in (learner_root / "manifests").glob("manifest-*.json"):
        manifest = _read_json(manifest_path, attempts=1) or {}
        identity = manifest.get("model_identity")
        step = manifest.get("model_step")
        if isinstance(identity, str) and isinstance(step, int):
            steps[identity] = step
    for pointer_name in ("candidate.json", "champion.json"):
        pointer = _read_json(learner_root / pointer_name, attempts=1) or {}
        identity = pointer.get("model_identity")
        step = pointer.get("model_step")
        if isinstance(identity, str) and isinstance(step, int):
            steps[identity] = step

    completed: list[dict[str, object]] = []
    superseded = 0
    existing_paths: set[Path] = set()
    for result_path in (run_root / "arena").glob("*.json"):
        existing_paths.add(result_path)
        try:
            stat = result_path.stat()
        except OSError:
            continue
        cached = _ARENA_RESULT_CACHE.get(result_path)
        if cached is not None and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            summary = cached[2]
        else:
            result = _read_json(result_path, attempts=1) or {}
            promotion = _mapping(result.get("promotion"))
            aggregate = _mapping(result.get("aggregate"))
            weighted_aggregate = _mapping(
                result.get("weighted_aggregate") or promotion.get("weighted_aggregate")
            )
            summary = {
                "result_kind": _arena_result_kind(result),
                "result_category": _arena_result_category(result),
                "candidate": result.get("candidate"),
                "baseline": result.get("baseline"),
                "decision": promotion.get("decision"),
                "started_ns": result.get("started_ns"),
                "completed_ns": result.get("completed_ns"),
                "aggregate": {
                    key: aggregate.get(key)
                    for key in (
                        "anytime_confidence_sequence",
                        "elo_difference",
                        "wins",
                        "losses",
                        "games",
                    )
                },
                "per_ring_elo": {
                    str(ring): _mapping(metrics).get("elo_difference")
                    for ring, metrics in _mapping(result.get("per_ring")).items()
                },
                "weighted_aggregate": (
                    dict(weighted_aggregate) if weighted_aggregate else None
                ),
                "weighted_wave_plan": result.get("wave_plan"),
            }
            _ARENA_RESULT_CACHE[result_path] = (
                stat.st_mtime_ns,
                stat.st_size,
                summary,
            )
        evidence_ns = summary.get("completed_ns", summary.get("started_ns"))
        if (
            started_ns is not None
            and isinstance(evidence_ns, int)
            and not isinstance(evidence_ns, bool)
            and evidence_ns < started_ns
        ):
            continue
        decision = summary.get("decision")
        if decision == "superseded":
            superseded += 1
        aggregate = _mapping(summary.get("aggregate"))
        completed_ns = summary.get("completed_ns")
        if not aggregate or not isinstance(completed_ns, int):
            continue
        confidence = aggregate.get("anytime_confidence_sequence")
        lower_elo = None
        if (
            isinstance(confidence, list)
            and confidence
            and (_number(confidence[0]) is not None)
        ):
            lower_score = float(confidence[0])
            if 0 < lower_score < 1:
                lower_elo = 400 * math.log10(lower_score / (1 - lower_score))
        candidate = str(summary.get("candidate"))
        published_ns = all_publications.get(candidate)
        completed.append(
            {
                "result_kind": summary.get("result_kind", "promotion"),
                "result_category": summary.get("result_category", "promotion"),
                "completed_ns": completed_ns,
                "candidate_step": steps.get(candidate),
                "baseline_step": steps.get(str(summary.get("baseline"))),
                "candidate_published_ns": published_ns,
                "publish_to_terminal_seconds": (
                    (completed_ns - published_ns) / 1_000_000_000
                    if published_ns is not None and completed_ns >= published_ns
                    else None
                ),
                "decision": decision,
                "elo_difference": aggregate.get("elo_difference"),
                "elo_lower": lower_elo,
                "wins": aggregate.get("wins"),
                "losses": aggregate.get("losses"),
                "games": aggregate.get("games"),
                "per_ring_elo": summary.get("per_ring_elo"),
                "weighted_aggregate": summary.get("weighted_aggregate"),
                "weighted_wave_plan": summary.get("weighted_wave_plan"),
            }
        )
    for stale_path in set(_ARENA_RESULT_CACHE) - existing_paths:
        _ARENA_RESULT_CACHE.pop(stale_path, None)
    completed.sort(key=lambda row: _number(row.get("completed_ns")) or 0.0)
    result_kind_counts = {
        kind: sum(row.get("result_kind") == kind for row in completed)
        for kind in ("promotion", "crossplay", "historical_crossplay", "unknown")
    }
    result_category_counts = {
        kind: sum(row.get("result_category") == kind for row in completed)
        for kind in ("promotion", "crossplay", "unknown")
    }
    completed_superseded = sum(
        row.get("result_category") == "promotion"
        and row.get("decision") == "superseded"
        for row in completed
    )
    promotion_evaluations = result_category_counts["promotion"]
    candidate_publications = len(publications)
    return {
        "completed_evaluations": len(completed),
        "result_kind_counts": result_kind_counts,
        "result_category_counts": result_category_counts,
        "promotion_evaluations": promotion_evaluations,
        "candidate_publications": candidate_publications,
        "candidate_arrival_service_ratio": (
            candidate_publications / promotion_evaluations
            if promotion_evaluations
            else None
        ),
        "crossplay_evaluations": result_category_counts["crossplay"],
        "promotions": sum(row.get("decision") == "promote" for row in completed),
        "rejections": sum(
            row.get("decision")
            in ("reject", "reject_ring_regression", "reject_max_pairs")
            for row in completed
        ),
        "superseded_candidates": superseded,
        "completed_superseded_evaluations": completed_superseded,
        "completed_superseded_fraction": (
            completed_superseded / promotion_evaluations
            if promotion_evaluations
            else None
        ),
        "recent": completed[-limit:],
    }


def _weighted_arena_progress(
    arena_config: Mapping[str, object],
    arena_history: Mapping[str, object],
) -> dict[str, object]:
    raw_ratios = arena_config.get("promotion_pair_ratios")
    ratios = (
        {str(ring): ratio for ring, ratio in raw_ratios.items()}
        if isinstance(raw_ratios, Mapping)
        else {}
    )
    recent = arena_history.get("recent")
    recent = recent if isinstance(recent, list) else []
    latest = next(
        (
            row
            for row in reversed(recent)
            if isinstance(row, Mapping)
            and isinstance(row.get("weighted_aggregate"), Mapping)
        ),
        None,
    )
    aggregate = _mapping(latest.get("weighted_aggregate")) if latest is not None else {}
    interval = aggregate.get("anytime_elo_interval")
    lower_elo = (
        _number(interval[0])
        if isinstance(interval, list) and len(interval) == 2
        else None
    )
    complete_blocks = aggregate.get(
        "complete_blocks",
        aggregate.get("completed_blocks"),
    )
    complete_blocks = (
        complete_blocks
        if type(complete_blocks) is int and complete_blocks >= 0
        else None
    )
    maximum_blocks = arena_config.get("weighted_max_blocks")
    maximum_blocks = (
        maximum_blocks if type(maximum_blocks) is int and maximum_blocks > 0 else None
    )
    wave_plan = _mapping(latest.get("weighted_wave_plan")) if latest is not None else {}
    return {
        "enabled": bool(ratios),
        "pair_ratios": ratios,
        "initial_blocks": arena_config.get("weighted_initial_blocks"),
        "continuation_blocks": arena_config.get("weighted_continuation_blocks"),
        "max_blocks": maximum_blocks,
        "complete_blocks": complete_blocks,
        "remaining_blocks": (
            max(0, maximum_blocks - complete_blocks)
            if maximum_blocks is not None and complete_blocks is not None
            else None
        ),
        "wave_target_blocks": wave_plan.get(
            "target_complete_blocks",
            wave_plan.get("target_blocks"),
        ),
        "incomplete_pair_counts": aggregate.get("incomplete_pair_counts"),
        "score_rate": aggregate.get(
            "score_rate",
            aggregate.get("weighted_score_rate"),
        ),
        "elo_difference": aggregate.get(
            "elo_difference",
            aggregate.get("weighted_elo_difference"),
        ),
        "anytime_elo_interval": interval,
        "anytime_lower_elo": lower_elo,
        "sequential_state": aggregate.get(
            "sequential_state",
            aggregate.get("evidence_state"),
        ),
        "decision": latest.get("decision") if latest is not None else None,
        "latest_completed_ns": (
            latest.get("completed_ns") if latest is not None else None
        ),
    }


def _strength_efficiency_status(
    run_root: Path,
    *,
    now_ns: int,
) -> dict[str, object]:
    path = run_root / "strength-efficiency.json"
    report = _read_json(path, attempts=1)
    if report is None:
        return {
            "available": False,
            "path": str(path),
            "present": path.is_file(),
        }
    run_identity = _read_json(run_root / "run.json", attempts=1) or {}
    observed_until_ns = report.get("observed_until_ns")
    run_id = run_identity.get("run_id")
    generation_family = run_identity.get("generation_family")
    started_ns = run_identity.get("created_ns")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(generation_family, str)
        or not generation_family
        or isinstance(started_ns, bool)
        or not isinstance(started_ns, int)
        or started_ns <= 0
        or report.get("report") != "startrain-strength-efficiency"
        or report.get("schema_version") != 1
        or report.get("status") != "complete"
        or report.get("run_id") != run_id
        or report.get("generation_family") != generation_family
        or report.get("run_root") != str(run_root.resolve())
        or report.get("started_ns") != started_ns
        or isinstance(observed_until_ns, bool)
        or not isinstance(observed_until_ns, int)
        or observed_until_ns <= 0
        or observed_until_ns
        > now_ns + int(STRENGTH_REPORT_CLOCK_SKEW_SECONDS * 1_000_000_000)
    ):
        return {
            "available": False,
            "path": str(path),
            "present": True,
            "reason": "report_contract_invalid",
        }
    autonomous = _mapping(report.get("autonomous_elo"))
    headline = _mapping(autonomous.get("headline"))
    headline_elo = _number(autonomous.get("headline_elo"))
    source = headline.get("source")
    if headline_elo is None:
        headline_elo = _number(headline.get("rating"))
    if headline_elo is None:
        aggregate = _mapping(autonomous.get("aggregate"))
        aggregate_latest = _mapping(aggregate.get("latest"))
        headline_elo = _number(aggregate_latest.get("rating"))
        if headline_elo is not None:
            headline = {
                "source": "aggregate",
                **aggregate_latest,
            }
            source = "aggregate"
    confidence_interval = headline.get("confidence_interval")
    if not (
        isinstance(confidence_interval, list)
        and len(confidence_interval) == 2
        and all(_number(value) is not None for value in confidence_interval)
    ):
        confidence_interval = None
    report_age_seconds = (
        max(0.0, (now_ns - observed_until_ns) / 1_000_000_000)
        if isinstance(observed_until_ns, int)
        and not isinstance(observed_until_ns, bool)
        and 0
        < observed_until_ns
        <= now_ns + int(STRENGTH_REPORT_CLOCK_SKEW_SECONDS * 1_000_000_000)
        else None
    )
    aggregate = _mapping(autonomous.get("aggregate"))
    return {
        "available": True,
        "present": True,
        "path": str(path),
        "status": report.get("status"),
        "observed_until_ns": observed_until_ns,
        "age_seconds": report_age_seconds,
        "headline": dict(headline) if headline else None,
        "headline_elo": headline_elo,
        "headline_source": source if isinstance(source, str) else None,
        "headline_confidence_interval": confidence_interval,
        "statistical_role": aggregate.get("statistical_role"),
        "adoption_ranking_authorized": aggregate.get("adoption_ranking_authorized"),
    }


def _disk_status(root: Path) -> dict[str, object]:
    usage = shutil.disk_usage(root)
    stat = root.stat()
    filesystem = root.stat().st_dev
    statvfs = None
    try:
        import os

        statvfs = os.statvfs(root)
    except OSError:
        pass
    result: dict[str, object] = {
        "device": filesystem,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_fraction": usage.used / usage.total if usage.total else 0.0,
        "root_mtime_ns": stat.st_mtime_ns,
    }
    if statvfs is not None:
        inode_total = statvfs.f_files
        inode_free = statvfs.f_ffree
        result.update(
            {
                "inode_total": inode_total,
                "inode_free": inode_free,
                "inode_used_fraction": (
                    (inode_total - inode_free) / inode_total if inode_total else 0.0
                ),
            }
        )
    return result


def _add_warning(
    warnings: list[dict[str, str]], severity: str, code: str, message: str
) -> None:
    warnings.append({"severity": severity, "code": code, "message": message})


def _ring10_weights_match(value: object) -> bool:
    if isinstance(value, Mapping):
        try:
            weights = {int(ring): float(weight) for ring, weight in value.items()}
        except (TypeError, ValueError):
            return False
        return weights.get(10) == 1.0 and all(
            weight == 0.0 for ring, weight in weights.items() if ring != 10
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        try:
            weights = tuple(float(weight) for weight in value)
        except (TypeError, ValueError):
            return False
        return weights == (0.0, 0.0, 0.0, 1.0)
    return False


def _ring10_active_rings_match(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return False
    try:
        return tuple(int(ring) for ring in value) == (10,)
    except (TypeError, ValueError):
        return False


def _ring10_arena_violations(
    root: Path,
    *,
    objective_started_ns: int | None,
) -> list[str]:
    violations = []
    for path in sorted((root / "arena").glob("*.json")):
        result = _read_json(path, attempts=1)
        if result is None:
            continue
        evidence_ns = result.get("completed_ns", result.get("started_ns"))
        if (
            objective_started_ns is not None
            and isinstance(evidence_ns, int)
            and not isinstance(evidence_ns, bool)
            and evidence_ns < objective_started_ns
        ):
            continue
        rings: set[int] = set()
        per_ring = result.get("per_ring")
        if isinstance(per_ring, Mapping):
            try:
                rings.update(int(ring) for ring in per_ring)
            except (TypeError, ValueError):
                violations.append(path.name)
                continue
        weighted = _mapping(
            result.get("weighted_aggregate")
            or _mapping(result.get("promotion")).get("weighted_aggregate")
        )
        ratios = weighted.get("pair_ratios")
        if isinstance(ratios, Mapping):
            try:
                rings.update(int(ring) for ring in ratios)
            except (TypeError, ValueError):
                violations.append(path.name)
                continue
        if any(ring != 10 for ring in rings):
            violations.append(path.name)
    return violations


def collect_snapshot(
    run_root: Path,
    *,
    unit: str | None = None,
    profile_path: Path | None = None,
    continuity_state_path: Path | None = None,
    disaster_backup_root: Path | None = None,
    now_ns: int | None = None,
) -> dict[str, object]:
    root = run_root.expanduser().resolve()
    now = time.time_ns() if now_ns is None else now_ns
    warnings: list[dict[str, str]] = []
    profile_source = (
        profile_path.expanduser().resolve()
        if profile_path is not None
        else root / "profile.yaml"
    )
    profile = _read_json(profile_source) if profile_source.suffix == ".json" else None
    if profile is None:
        try:
            loaded = yaml.safe_load(profile_source.read_text(encoding="utf-8"))
            profile = loaded if isinstance(loaded, dict) else {}
        except (OSError, yaml.YAMLError):
            profile = {}
    orchestration = _mapping(profile.get("orchestration"))
    training_objective = orchestration.get("training_objective", "generalist")
    objective_contract: dict[str, object] = {
        "validated": False,
        "error": None,
        "profile": str(profile_source),
    }
    if "training_objective" in orchestration:
        try:
            validated_profile = load_config(profile_source)
            validate_continuous_config(validated_profile)
        except (OSError, ValueError, yaml.YAMLError) as error:
            objective_contract["error"] = f"{type(error).__name__}: {error}"
            _add_warning(
                warnings,
                "ERROR",
                "objective_profile_invalid",
                f"training objective profile is invalid: {error}",
            )
        else:
            training_objective = validated_profile.orchestration.training_objective
            objective_contract["validated"] = True
    ring10_objective_active = (
        training_objective == "ring10_only" and objective_contract["validated"] is True
    )
    ablation_metadata = _read_json(root / "ablation.json") or {}
    prepared_ns = ablation_metadata.get("prepared_ns")
    objective_started_ns = (
        prepared_ns
        if isinstance(prepared_ns, int) and not isinstance(prepared_ns, bool)
        else None
    )
    shutdown = _mapping(orchestration.get("shutdown"))
    stale_threshold = _number(shutdown.get("stale_heartbeat_seconds")) or 180.0
    stall_threshold = _number(shutdown.get("stall_timeout_seconds")) or 1_800.0
    learner_config = _mapping(profile.get("learner"))
    autonomous_config = _mapping(orchestration.get("autonomous"))
    autonomous_enabled = autonomous_config.get("enabled") is True
    target_steps = (
        "unlimited"
        if learner_config.get("unlimited") is True
        else learner_config.get("steps")
    )
    provenance = _read_json(root / "autonomous-provenance.json") or {}
    run_identity = _read_json(root / "run.json") or {}
    if autonomous_enabled and (
        provenance.get("mode") != "random-init-selfplay-only"
        or provenance.get("run_id") != run_identity.get("run_id")
        or provenance.get("generation_family") != run_identity.get("generation_family")
        or provenance.get("external_weights") is not False
        or provenance.get("external_replay") is not False
        or provenance.get("external_positions") is not False
    ):
        _add_warning(
            warnings,
            "ERROR",
            "autonomous_provenance_invalid",
            "autonomous run provenance is missing or incompatible",
        )

    service = _systemd_status(unit)
    if service.get("query_error"):
        _add_warning(warnings, "WARN", "systemd_query_failed", "systemd query failed")
    elif service.get("configured") and service.get("active_state") != "active":
        _add_warning(
            warnings,
            "ERROR",
            "service_inactive",
            f"service is {service.get('active_state')}",
        )
    if (_number(service.get("restart_count")) or 0) > 0:
        _add_warning(warnings, "WARN", "service_restarted", "systemd restart observed")

    coordinator = _read_json(root / "status" / "coordinator.json") or {}
    fatal = _read_json(root / "status" / "fatal.json")
    if fatal is not None:
        failure_class = fatal.get("failure_class")
        reason = fatal.get("reason")
        _add_warning(
            warnings,
            "ERROR",
            "coordinator_fatal",
            f"{failure_class or 'unknown'} failure: {reason or 'reason unavailable'}",
        )
    if coordinator.get("state") not in ("running", "draining"):
        _add_warning(
            warnings,
            "ERROR",
            "coordinator_unhealthy",
            f"coordinator state is {coordinator.get('state')}",
        )
    workers_output = []
    workers = coordinator.get("workers")
    if isinstance(workers, dict):
        for name, raw in sorted(workers.items()):
            worker = raw if isinstance(raw, dict) else {}
            heartbeat_path = worker.get("heartbeat")
            heartbeat = (
                _read_json(Path(heartbeat_path))
                if isinstance(heartbeat_path, str)
                else None
            ) or {}
            heartbeat_age = _age_seconds(heartbeat.get("heartbeat_ns"), now)
            progress_age = _age_seconds(heartbeat.get("progress_ns"), now)
            state = str(worker.get("state", "unknown"))
            restart_count = int(worker.get("restart_count", 0) or 0)
            if state not in ("running", "paused", "drained", "completed"):
                _add_warning(
                    warnings,
                    "ERROR",
                    "worker_unhealthy",
                    f"{name} state={state}",
                )
            if restart_count:
                _add_warning(
                    warnings,
                    "WARN",
                    "worker_restarted",
                    f"{name} restarts={restart_count}",
                )
            if state == "running" and (
                heartbeat_age is None or heartbeat_age > stale_threshold
            ):
                _add_warning(
                    warnings,
                    "ERROR",
                    "heartbeat_stale",
                    f"{name} heartbeat age={heartbeat_age}",
                )
            elif (
                state == "running"
                and progress_age is not None
                and progress_age > stall_threshold
            ):
                _add_warning(
                    warnings,
                    "ERROR",
                    "worker_stalled",
                    f"{name} progress age={progress_age:.1f}s",
                )
            workers_output.append(
                {
                    "name": name,
                    "role": worker.get("role"),
                    "state": state,
                    "pid": worker.get("pid"),
                    "restart_count": restart_count,
                    "failure_class": worker.get("failure_class"),
                    "failure_reason": worker.get("failure_reason"),
                    "last_exit_code": worker.get("last_exit_code"),
                    "phase": heartbeat.get("phase"),
                    "progress": heartbeat.get("progress"),
                    "active_ring_weights": heartbeat.get("active_ring_weights"),
                    "active_rings": heartbeat.get("active_rings"),
                    "ring": heartbeat.get("ring"),
                    "heartbeat_age_seconds": heartbeat_age,
                    "progress_age_seconds": progress_age,
                }
            )
    else:
        _add_warning(warnings, "ERROR", "workers_missing", "worker map is missing")

    learner_metric = (
        _latest_jsonl(
            root / "learner" / "metrics.jsonl",
            predicate=lambda row: isinstance(row.get("losses"), dict),
        )
        or {}
    )
    loader_pool_metric = (
        _latest_jsonl(
            root / "learner" / "metrics.jsonl",
            predicate=lambda row: (
                row.get("event")
                in {
                    "replay_loader_pool_started",
                    "replay_loader_pool_rebound",
                    "replay_loader_pool_shutdown",
                }
            ),
        )
        or {}
    )
    learner_heartbeat = _read_json(root / "status" / "learner.heartbeat.json") or {}
    losses = learner_metric.get("losses")
    if isinstance(losses, dict) and any(
        _number(value) is None for value in losses.values()
    ):
        _add_warning(warnings, "ERROR", "nonfinite_loss", "learner loss is non-finite")
    if learner_metric.get("feature_path") not in (None, "rust"):
        _add_warning(
            warnings,
            "WARN",
            "python_feature_path",
            f"learner feature path={learner_metric.get('feature_path')}",
        )
    step_seconds = _number(learner_metric.get("step_seconds"))
    data_wait_seconds = _number(learner_metric.get("data_wait_seconds"))
    data_wait_fraction = (
        data_wait_seconds / step_seconds
        if data_wait_seconds is not None and step_seconds
        else None
    )
    if data_wait_fraction is not None and data_wait_fraction > 0.25:
        _add_warning(
            warnings,
            "WARN",
            "learner_data_wait",
            f"learner data wait is {data_wait_fraction:.1%} of wall step time",
        )
    updates_per_new_sample = _number(learner_metric.get("updates_per_new_sample"))
    lifetime_updates = _number(learner_metric.get("lifetime_updates_per_new_sample"))
    if lifetime_updates is None:
        lifetime_updates = updates_per_new_sample
    segment_updates = _number(learner_metric.get("segment_updates_per_new_sample"))
    configured_target_updates = _number(
        learner_config.get("target_updates_per_new_sample")
    )
    segment_target_updates = _number(
        learner_metric.get("utd_segment_target_updates_per_new_sample")
    )
    effective_updates = (
        segment_updates if segment_updates is not None else updates_per_new_sample
    )
    effective_target_updates = (
        segment_target_updates
        if segment_updates is not None and segment_target_updates is not None
        else configured_target_updates
    )
    if (
        effective_target_updates is not None
        and effective_updates is not None
        and effective_updates > effective_target_updates * 1.05
    ):
        _add_warning(
            warnings,
            "WARN",
            "update_to_data_high",
            f"UTD={effective_updates:.3f} target={effective_target_updates:.3f}",
        )

    def loader_value(name: str) -> object:
        value = learner_metric.get(name)
        return value if value is not None else loader_pool_metric.get(name)

    loader_lifecycle = loader_value("loader_lifecycle")
    loader_workers = _number(learner_metric.get("loader_workers_effective"))
    loader_pool_starts = _number(loader_value("loader_pool_starts"))
    loader_pool_shutdowns = _number(loader_value("loader_pool_shutdowns"))
    loader_worker_pids = loader_value("loader_worker_pids")
    learner_worker = workers.get("learner") if isinstance(workers, dict) else None
    learner_worker_state = (
        learner_worker.get("state") if isinstance(learner_worker, Mapping) else None
    )
    if (
        learner_worker_state == "running"
        and loader_pool_starts is not None
        and loader_pool_starts > 1
    ):
        _add_warning(
            warnings,
            "WARN",
            "loader_pool_respawned",
            f"learner loader pool starts={int(loader_pool_starts)}",
        )
    if (
        loader_lifecycle == "process"
        and loader_workers is not None
        and loader_workers > 0
        and learner_worker_state == "running"
    ):
        if loader_pool_starts is None or loader_pool_starts < 1:
            _add_warning(
                warnings,
                "WARN",
                "loader_pool_missing",
                f"process-scoped loader pool starts={loader_pool_starts}",
            )
        if loader_pool_shutdowns not in (None, 0):
            _add_warning(
                warnings,
                "ERROR",
                "loader_pool_shutdown_live",
                f"live learner loader pool shutdowns={int(loader_pool_shutdowns)}",
            )
        if not isinstance(loader_worker_pids, list) or len(loader_worker_pids) != int(
            loader_workers
        ):
            _add_warning(
                warnings,
                "WARN",
                "loader_pool_pid_mismatch",
                "loader worker PID count differs from effective worker count",
            )
    nonfinite_losses = _number(learner_metric.get("nonfinite_loss_count"))
    nonfinite_gradients = _number(learner_metric.get("nonfinite_gradient_count"))
    if (nonfinite_losses or 0) > 0 or (nonfinite_gradients or 0) > 0:
        _add_warning(
            warnings,
            "ERROR",
            "learner_nonfinite",
            "learner reported non-finite loss or gradient events",
        )
    clipping_frequency = _number(learner_metric.get("gradient_clipping_frequency"))
    if clipping_frequency is not None and clipping_frequency > 0.5:
        _add_warning(
            warnings,
            "WARN",
            "gradient_clipping_high",
            f"gradient clipping frequency={clipping_frequency:.1%}",
        )
    learner = {
        "step": learner_heartbeat.get("step", learner_metric.get("step")),
        "target_steps": target_steps,
        "epoch": learner_heartbeat.get("epoch", learner_metric.get("epoch")),
        "phase": learner_heartbeat.get("phase"),
        "examples_per_second": learner_metric.get("examples_per_second"),
        "device_examples_per_second": learner_metric.get("device_examples_per_second"),
        "step_seconds": learner_metric.get("step_seconds"),
        "device_step_seconds": learner_metric.get("device_step_seconds"),
        "data_wait_seconds": learner_metric.get("data_wait_seconds"),
        "data_wait_fraction": data_wait_fraction,
        "h2d_seconds": learner_metric.get("h2d_seconds"),
        "updates_per_new_sample": updates_per_new_sample,
        "target_updates_per_new_sample": configured_target_updates,
        "lifetime_updates_per_new_sample": lifetime_updates,
        "segment_updates_per_new_sample": segment_updates,
        "utd_segment_target_updates_per_new_sample": segment_target_updates,
        "utd_segment_baseline_examples_consumed": learner_metric.get(
            "utd_segment_baseline_examples_consumed"
        ),
        "utd_segment_baseline_committed_replay_samples": learner_metric.get(
            "utd_segment_baseline_committed_replay_samples"
        ),
        "loader_workers_effective": learner_metric.get("loader_workers_effective"),
        "loader_lifecycle": loader_lifecycle,
        "loader_pool_starts": loader_pool_starts,
        "loader_pool_rebinds": loader_value("loader_pool_rebinds"),
        "loader_pool_shutdowns": loader_pool_shutdowns,
        "loader_worker_pids": loader_worker_pids,
        "window_setup_seconds": learner_metric.get("window_setup_seconds"),
        "window_setup_amortized_seconds": learner_metric.get(
            "window_setup_amortized_seconds"
        ),
        "window_batches_allocated": learner_metric.get("window_batches_allocated"),
        "window_batches_consumed": learner_metric.get("window_batches_consumed"),
        "window_batches_consumed_this_spin": learner_metric.get(
            "window_batches_consumed_this_spin"
        ),
        "window_reuse": learner_metric.get("window_reuse"),
        "window_reuse_spins": learner_metric.get("window_reuse_spins"),
        "window_refresh_reason": learner_metric.get("window_refresh_reason"),
        "utd_wait_spins": learner_metric.get("utd_wait_spins"),
        "learning_rates": learner_metric.get("learning_rates"),
        "replay_samples_by_ring": learner_metric.get("replay_samples_by_ring"),
        "ring_batch_weights": learner_metric.get("ring_batch_weights"),
        "active_rings": learner_heartbeat.get("active_rings"),
        "active_ring_weights": learner_heartbeat.get("active_ring_weights"),
        "losses": losses,
        "gradient_norm": learner_metric.get("gradient_norm"),
        "gradient_clipped": learner_metric.get("gradient_clipped"),
        "gradient_clipped_steps": learner_metric.get("gradient_clipped_steps"),
        "gradient_clipping_frequency": learner_metric.get(
            "gradient_clipping_frequency"
        ),
        "nonfinite_loss_count": learner_metric.get("nonfinite_loss_count"),
        "nonfinite_gradient_count": learner_metric.get("nonfinite_gradient_count"),
        "optimizer_routing": learner_metric.get("optimizer_routing"),
        "optimizer_routing_hash": learner_metric.get("optimizer_routing_hash"),
        "optimizer_parameter_tensors": learner_metric.get(
            "optimizer_parameter_tensors"
        ),
        "optimizer_parameter_elements": learner_metric.get(
            "optimizer_parameter_elements"
        ),
        "optimizer_groups": learner_metric.get("optimizer_groups"),
        "optimizer_weight_norm": learner_metric.get("optimizer_weight_norm"),
        "optimizer_update_norm": learner_metric.get("optimizer_update_norm"),
        "scheduler": learner_metric.get("scheduler"),
        "scheduler_age_steps": learner_metric.get("scheduler_age_steps"),
        "scheduler_segment": learner_metric.get("scheduler_segment"),
        "scheduler_segment_position": learner_metric.get("scheduler_segment_position"),
        "ema": learner_metric.get("ema"),
        "raw_vs_ema_distance": learner_metric.get("raw_vs_ema_distance"),
        "raw_vs_ema_relative_distance": learner_metric.get(
            "raw_vs_ema_relative_distance"
        ),
        "ema_effective_turnover": learner_metric.get("ema_effective_turnover"),
        "ema_interval_effective_turnover": learner_metric.get(
            "ema_interval_effective_turnover"
        ),
        "replay_minimum_shard_id_exclusive": learner_metric.get(
            "replay_minimum_shard_id_exclusive"
        ),
        "feature_path": learner_metric.get("feature_path"),
    }
    if ring10_objective_active:
        learner_violations = []
        for source, weights in (
            ("metric ring_batch_weights", learner_metric.get("ring_batch_weights")),
            (
                "heartbeat active_ring_weights",
                learner_heartbeat.get("active_ring_weights"),
            ),
        ):
            if weights is not None and not _ring10_weights_match(weights):
                learner_violations.append(source)
        active_rings = learner_heartbeat.get("active_rings")
        if active_rings is not None and not _ring10_active_rings_match(active_rings):
            learner_violations.append("heartbeat active_rings")
        if learner_violations:
            _add_warning(
                warnings,
                "ERROR",
                "ring10_learner_evidence_mismatch",
                "learner reports non-ring-10 work: " + ", ".join(learner_violations),
            )

    actors = []
    for metrics_path in sorted((root / "metrics").glob("actor-gpu-*.jsonl")):
        metric = _latest_jsonl(metrics_path)
        if metric is not None:
            actors.append(metric)
    actor_samples = sum(int(row.get("samples", 0) or 0) for row in actors)
    actor_policy_samples = sum(int(row.get("policy_samples", 0) or 0) for row in actors)
    worker_map = workers if isinstance(workers, dict) else {}
    worker_health_map = {str(row.get("name")): row for row in workers_output}
    active_actor_rows = [
        row
        for row in actors
        if _mapping(worker_map.get(str(row.get("worker")))).get("state") == "running"
    ]
    if ring10_objective_active:
        ring10_actor_violations = []
        for row in active_actor_rows:
            worker_name = str(row.get("worker"))
            worker_health = _mapping(worker_health_map.get(worker_name))
            for source, weights in (
                ("heartbeat weights", worker_health.get("active_ring_weights")),
                ("metric weights", row.get("active_ring_weights")),
            ):
                if weights is not None and not _ring10_weights_match(weights):
                    ring10_actor_violations.append(f"{worker_name} {source}")
            for source, active_rings in (
                ("heartbeat active_rings", worker_health.get("active_rings")),
                ("metric active_rings", row.get("active_rings")),
            ):
                if active_rings is not None and not _ring10_active_rings_match(
                    active_rings
                ):
                    ring10_actor_violations.append(f"{worker_name} {source}")
            for source, ring in (
                ("heartbeat ring", worker_health.get("ring")),
                ("metric ring", row.get("ring")),
            ):
                if ring is not None:
                    try:
                        matches_ring10 = int(str(ring)) == 10
                    except (TypeError, ValueError):
                        matches_ring10 = False
                    if not matches_ring10:
                        ring10_actor_violations.append(f"{worker_name} {source}")
        if ring10_actor_violations:
            _add_warning(
                warnings,
                "ERROR",
                "ring10_actor_evidence_mismatch",
                "actors report non-ring-10 work: "
                + ", ".join(sorted(set(ring10_actor_violations))),
            )
    weight_variants = {
        tuple(float(value) for value in weights)
        for row in active_actor_rows
        if isinstance(
            (
                weights := _mapping(worker_health_map.get(str(row.get("worker")))).get(
                    "active_ring_weights"
                )
            ),
            list,
        )
    }
    policy_supervision_rate = (
        actor_policy_samples / actor_samples if actor_samples else None
    )
    expected_ring_weights = _configured_ring_weights(profile, learner.get("step"))
    noncompliant_weight_workers = []
    low_policy_workers = []
    fast_targets_enabled = (
        _mapping(profile.get("selfplay")).get("record_fast_policy_targets") is True
    )
    for row in active_actor_rows:
        worker_name = str(row.get("worker"))
        worker_health = _mapping(worker_health_map.get(worker_name))
        weights = worker_health.get("active_ring_weights")
        configured_weights = (
            tuple(float(value) for value in weights)
            if isinstance(weights, list)
            else None
        )
        if (
            expected_ring_weights is not None
            and worker_health.get("phase") != "starting"
            and configured_weights != expected_ring_weights
        ):
            noncompliant_weight_workers.append(worker_name)
        samples = int(row.get("samples", 0) or 0)
        policy_samples = int(row.get("policy_samples", 0) or 0)
        metric_weights = row.get("active_ring_weights")
        metric_weight_tuple = (
            tuple(float(value) for value in metric_weights)
            if isinstance(metric_weights, list)
            else None
        )
        if (
            fast_targets_enabled
            and samples
            and metric_weight_tuple == expected_ring_weights
            and policy_samples / samples < 0.9
        ):
            low_policy_workers.append(worker_name)
    if len(weight_variants) > 1 or noncompliant_weight_workers:
        _add_warning(
            warnings,
            "ERROR",
            "actor_ring_weight_mismatch",
            "actor ring weights mismatch: " + ",".join(noncompliant_weight_workers),
        )
    if low_policy_workers:
        _add_warning(
            warnings,
            "WARN",
            "policy_supervision_low",
            "low actor policy supervision: " + ",".join(low_policy_workers),
        )
    actor_throughput = _actor_throughput_window(
        root / "metrics",
        now_ns=now,
    )
    if (
        active_actor_rows
        and any(isinstance(row.get("batch_completed_ns"), int) for row in actors)
        and actor_throughput.get("record_count") == 0
    ):
        _add_warning(
            warnings,
            "WARN",
            "actor_throughput_stale",
            "no completed actor batches are available in the throughput window",
        )
    actor_fleet = {
        "workers": len(actors),
        "throughput": actor_throughput,
        "policy_supervision_rate": policy_supervision_rate,
        "active_ring_weights": (
            list(next(iter(weight_variants))) if len(weight_variants) == 1 else None
        ),
        "ring_weight_variants": [list(weights) for weights in sorted(weight_variants)],
        "noncompliant_weight_workers": noncompliant_weight_workers,
        "low_policy_workers": low_policy_workers,
        "model_role_counts": {
            role: sum(row.get("model_role") == role for row in actors)
            for role in ("champion", "candidate", "history")
        },
        "latest_batch_rate_sum": {
            "games_per_second": sum(
                _number(row.get("games_per_second")) or 0.0 for row in actors
            ),
            "samples_per_second": sum(
                _number(row.get("samples_per_second")) or 0.0 for row in actors
            ),
            "evaluator_rows_per_second": sum(
                _number(row.get("evaluator_rows_per_second")) or 0.0 for row in actors
            ),
        },
        "latest_batch_rate_sum_deprecated": True,
        "latest": [
            {
                "worker": row.get("worker"),
                "ring": row.get("ring"),
                "batch": row.get("batch"),
                "model_role": row.get("model_role"),
                "model_step": row.get("model_step"),
                "games_per_second": row.get("games_per_second"),
                "samples_per_second": row.get("samples_per_second"),
                "evaluator_rows_per_second": row.get("evaluator_rows_per_second"),
            }
            for row in actors
        ],
    }

    raw_learner_step = learner.get("step")
    current_model_step = (
        raw_learner_step
        if isinstance(raw_learner_step, int) and not isinstance(raw_learner_step, bool)
        else None
    )
    replay, replay_error = _replay_status(
        root / "replay" / "manifest.sqlite3",
        current_model_step=current_model_step,
    )
    if replay_error:
        _add_warning(warnings, "WARN", "replay_query", replay_error)
    replay_states = _mapping(replay.get("states"))
    quarantined = _mapping(replay_states.get("quarantined")).get("shards", 0)
    if quarantined:
        _add_warning(
            warnings,
            "ERROR",
            "replay_quarantine",
            f"quarantined shards={quarantined}",
        )

    learner_root = root / "learner"
    recovery_pointer = _read_json(learner_root / "recovery.json") or {}
    recovery_step = None
    recovery_age = None
    if recovery_pointer:
        checkpoint_value = recovery_pointer.get("checkpoint")
        checkpoint_bytes = recovery_pointer.get("checkpoint_bytes")
        checkpoint_sha256 = recovery_pointer.get("checkpoint_sha256")
        step = recovery_pointer.get("step")
        valid_pointer = (
            recovery_pointer.get("format") == "startrain.recovery-pointer"
            and recovery_pointer.get("schema_version") == 1
            and isinstance(checkpoint_value, str)
            and bool(checkpoint_value)
            and isinstance(checkpoint_bytes, int)
            and not isinstance(checkpoint_bytes, bool)
            and checkpoint_bytes > 0
            and isinstance(checkpoint_sha256, str)
            and len(checkpoint_sha256) == 64
            and isinstance(step, int)
            and not isinstance(step, bool)
            and step >= 0
        )
        checkpoint = (
            (learner_root / checkpoint_value).resolve()
            if valid_pointer and isinstance(checkpoint_value, str)
            else None
        )
        artifact_valid, artifact_mtime = (
            _verified_artifact(
                checkpoint,
                expected_bytes=checkpoint_bytes,
                expected_sha256=checkpoint_sha256,
            )
            if checkpoint is not None
            and checkpoint.parent == (learner_root / "recovery").resolve()
            else (False, None)
        )
        valid_pointer = bool(valid_pointer and artifact_valid)
        if valid_pointer and checkpoint is not None:
            recovery_step = step
            recovery_age = (
                max(0.0, time.time() - artifact_mtime)
                if artifact_mtime is not None
                else None
            )
        else:
            _add_warning(
                warnings,
                "ERROR",
                "recovery_checkpoint_invalid",
                "learner recovery pointer or artifact is invalid",
            )

    candidate_pointer = _read_json(learner_root / "candidate.json") or {}
    candidate_step = None
    if candidate_pointer:
        manifest_value = candidate_pointer.get("manifest")
        manifest = (
            (learner_root / manifest_value).resolve()
            if isinstance(manifest_value, str) and manifest_value
            else None
        )
        manifest_valid, _ = (
            _verified_artifact(
                manifest,
                expected_bytes=candidate_pointer.get("manifest_bytes"),
                expected_sha256=candidate_pointer.get("manifest_sha256"),
            )
            if manifest is not None
            and manifest.parent == (learner_root / "manifests").resolve()
            else (False, None)
        )
        manifest_payload = _read_json(manifest) if manifest_valid and manifest else None
        manifest_payload = manifest_payload or {}
        checkpoint_value = manifest_payload.get("checkpoint")
        checkpoint = (
            (manifest.parent / checkpoint_value).resolve()
            if manifest is not None
            and isinstance(checkpoint_value, str)
            and checkpoint_value
            else None
        )
        checkpoint_valid, _ = (
            _verified_artifact(
                checkpoint,
                expected_bytes=manifest_payload.get("checkpoint_bytes"),
                expected_sha256=manifest_payload.get("checkpoint_sha256"),
            )
            if checkpoint is not None
            and checkpoint.parent == (learner_root / "checkpoints").resolve()
            else (False, None)
        )
        pointer_step = candidate_pointer.get("model_step")
        manifest_step = manifest_payload.get("model_step")
        candidate_valid = (
            candidate_pointer.get("format") == "startrain.model-pointer"
            and candidate_pointer.get("schema_version") == 2
            and manifest_payload.get("format") == "startrain.model-manifest"
            and manifest_valid
            and checkpoint_valid
            and isinstance(pointer_step, int)
            and not isinstance(pointer_step, bool)
            and pointer_step >= 0
            and pointer_step == manifest_step
        )
        if candidate_valid:
            candidate_step = pointer_step
        else:
            _add_warning(
                warnings,
                "ERROR",
                "candidate_checkpoint_invalid",
                "candidate pointer, manifest, or checkpoint is invalid",
            )

    selfplay_pointer = _read_json(learner_root / "selfplay" / "candidate.json") or {}
    selfplay_step = None
    if selfplay_pointer:
        manifest_value = selfplay_pointer.get("manifest")
        manifest = (
            ((learner_root / "selfplay") / manifest_value).resolve()
            if isinstance(manifest_value, str) and manifest_value
            else None
        )
        allowed_manifest_parents = {
            (learner_root / "manifests").resolve(),
            (learner_root / "selfplay" / "manifests").resolve(),
        }
        manifest_valid, _ = (
            _verified_artifact(
                manifest,
                expected_bytes=selfplay_pointer.get("manifest_bytes"),
                expected_sha256=selfplay_pointer.get("manifest_sha256"),
            )
            if manifest is not None and manifest.parent in allowed_manifest_parents
            else (False, None)
        )
        manifest_payload = _read_json(manifest) if manifest_valid and manifest else None
        manifest_payload = manifest_payload or {}
        checkpoint_value = manifest_payload.get("checkpoint")
        checkpoint = (
            (manifest.parent / checkpoint_value).resolve()
            if manifest is not None
            and isinstance(checkpoint_value, str)
            and checkpoint_value
            else None
        )
        allowed_checkpoint_parents = {
            (learner_root / "checkpoints").resolve(),
            (learner_root / "selfplay" / "checkpoints").resolve(),
        }
        checkpoint_valid, _ = (
            _verified_artifact(
                checkpoint,
                expected_bytes=manifest_payload.get("checkpoint_bytes"),
                expected_sha256=manifest_payload.get("checkpoint_sha256"),
            )
            if checkpoint is not None
            and checkpoint.parent in allowed_checkpoint_parents
            else (False, None)
        )
        pointer_step = selfplay_pointer.get("model_step")
        if (
            selfplay_pointer.get("format") == "startrain.model-pointer"
            and selfplay_pointer.get("schema_version") == 2
            and manifest_payload.get("format") == "startrain.model-manifest"
            and manifest_valid
            and checkpoint_valid
            and isinstance(pointer_step, int)
            and not isinstance(pointer_step, bool)
            and pointer_step >= 0
            and pointer_step == manifest_payload.get("model_step")
        ):
            selfplay_step = pointer_step
        else:
            _add_warning(
                warnings,
                "ERROR",
                "selfplay_checkpoint_invalid",
                "self-play pointer, manifest, or checkpoint is invalid",
            )

    backup_directory = root / "recovery" / "replay-manifest"
    latest_backup = _read_json(backup_directory / "latest.json") or {}
    backup_path = None
    backup_age = None
    backup_valid = False
    backup_value = latest_backup.get("path")
    backup_bytes = latest_backup.get("bytes")
    backup_sha256 = latest_backup.get("sha256")
    if isinstance(backup_value, str) and Path(backup_value).name == backup_value:
        backup_path = backup_directory / backup_value
        backup_valid, backup_mtime = _verified_artifact(
            backup_path,
            expected_bytes=backup_bytes,
            expected_sha256=backup_sha256,
        )
        if backup_valid and backup_mtime is not None:
            backup_age = max(0.0, time.time() - backup_mtime)
    recovery_interval = _number(learner_config.get("recovery_interval_steps"))
    learner_step = learner_heartbeat.get("step", learner_metric.get("step"))
    durable_steps = [
        value for value in (recovery_step, candidate_step) if isinstance(value, int)
    ]
    durable_step = max(durable_steps, default=None)
    if recovery_interval is not None and isinstance(learner_step, int):
        if durable_step is None and learner_step > recovery_interval:
            _add_warning(
                warnings,
                "WARN",
                "recovery_checkpoint_missing",
                "learner recovery checkpoint is missing",
            )
        elif (
            durable_step is not None
            and learner_step - durable_step > recovery_interval * 2
        ):
            _add_warning(
                warnings,
                "WARN",
                "recovery_checkpoint_lag",
                f"durable learner state lags by {learner_step - durable_step} steps",
            )
    continuous_recovery = (
        learner_config.get("unlimited") is True
        or learner_config.get("recovery_interval_steps") is not None
    )
    if continuous_recovery and not backup_valid:
        _add_warning(
            warnings,
            "WARN",
            "replay_backup_missing",
            "replay manifest backup is missing",
        )
    elif continuous_recovery and backup_age is not None and backup_age > 2 * 60 * 60:
        _add_warning(
            warnings,
            "WARN",
            "replay_backup_stale",
            f"latest replay backup age={backup_age:.0f}s",
        )
    recovery = {
        "step": recovery_step,
        "candidate_step": candidate_step,
        "selfplay_step": selfplay_step,
        "durable_step": durable_step,
        "checkpoint_age_seconds": recovery_age,
        "replay_backup_age_seconds": backup_age,
        "replay_backup_valid": backup_valid,
    }
    if isinstance(learner_step, int):
        learner["candidate_lag_steps"] = (
            learner_step - candidate_step if isinstance(candidate_step, int) else None
        )
        learner["selfplay_lag_steps"] = (
            learner_step - selfplay_step if isinstance(selfplay_step, int) else None
        )
    cadence = _read_json(learner_root / "cadence.json") or {}
    learner["candidate_examples_published"] = cadence.get("candidate_examples")
    learner["selfplay_examples_published"] = cadence.get("selfplay_examples")

    arena = _read_json(root / "arena" / "promotion-status.json") or {}
    raw_measurement_started_ns = ablation_metadata.get("measurement_started_ns")
    measurement_started_ns = (
        raw_measurement_started_ns
        if isinstance(raw_measurement_started_ns, int)
        and not isinstance(raw_measurement_started_ns, bool)
        else objective_started_ns
    )
    arena_history = _arena_history(root, started_ns=measurement_started_ns)
    arena_worker = _mapping(worker_map.get("arena-promotion"))
    gpu7_actor = _mapping(worker_map.get("actor-gpu-7"))
    arena_history["current_occupancy"] = {
        "arena_running": arena_worker.get("state") == "running",
        "gpu7_actor_paused": gpu7_actor.get("state") == "paused",
        "arena_pid": arena_worker.get("pid"),
        "gpu7_actor_pid": gpu7_actor.get("pid"),
    }
    arena_config = _mapping(profile.get("arena"))
    weighted_promotion = _weighted_arena_progress(arena_config, arena_history)
    arena_history["weighted"] = weighted_promotion
    if ring10_objective_active:
        arena_violations = _ring10_arena_violations(
            root,
            objective_started_ns=objective_started_ns,
        )
        if arena_violations:
            _add_warning(
                warnings,
                "ERROR",
                "ring10_arena_evidence_mismatch",
                "current arena evidence contains non-ring-10 work: "
                + ", ".join(arena_violations),
            )
    promotion_config = _mapping(orchestration.get("promotion"))
    finish_inflight = promotion_config.get("finish_inflight_candidate") is True
    pairs_per_ring = _number(arena_config.get("pairs_per_ring"))
    minimum_pairs = _number(arena_config.get("minimum_pairs_per_ring"))
    maximum_pairs = _number(arena_config.get("max_pairs_per_ring"))
    continuation_pairs = _number(arena_config.get("continuation_pairs_per_ring"))
    if (
        pairs_per_ring is not None
        and minimum_pairs is not None
        and maximum_pairs is not None
        and maximum_pairs > minimum_pairs
    ):
        continuation = continuation_pairs or pairs_per_ring
        if continuation > 0:
            continuation_waves = math.ceil(
                (maximum_pairs - minimum_pairs) / continuation
            )
            if continuation_waves > 1 and not finish_inflight:
                _add_warning(
                    warnings,
                    "WARN",
                    "arena_continuation_fragmented",
                    "arena needs "
                    f"{continuation_waves} post-minimum waves; newer candidates may "
                    "supersede completed evaluation work",
                )
    strength_efficiency = _strength_efficiency_status(root, now_ns=now)
    strength_age = _number(strength_efficiency.get("age_seconds"))
    run_created_ns = run_identity.get("created_ns")
    run_age_seconds = (
        max(0.0, (now - run_created_ns) / 1_000_000_000)
        if isinstance(run_created_ns, int)
        and not isinstance(run_created_ns, bool)
        and 0 < run_created_ns <= now
        else None
    )
    if (
        coordinator.get("state") in ("running", "draining")
        and strength_efficiency.get("present") is True
        and strength_efficiency.get("available") is not True
    ):
        _add_warning(
            warnings,
            "ERROR",
            "strength_report_invalid",
            "strength-efficiency report failed its schema, identity, or timestamp contract",
        )
    elif (
        coordinator.get("state") in ("running", "draining")
        and strength_efficiency.get("present") is False
        and run_age_seconds is not None
        and run_age_seconds > STRENGTH_REPORT_ERROR_SECONDS
    ):
        _add_warning(
            warnings,
            "ERROR",
            "strength_report_missing",
            f"active run has no strength report after {run_age_seconds:.0f}s",
        )
    elif (
        coordinator.get("state") in ("running", "draining")
        and strength_age is not None
        and strength_age > STRENGTH_REPORT_ERROR_SECONDS
    ):
        _add_warning(
            warnings,
            "ERROR",
            "strength_report_stale",
            f"strength-efficiency report age={strength_age:.0f}s",
        )
    elif (
        coordinator.get("state") in ("running", "draining")
        and strength_age is not None
        and strength_age > STRENGTH_REPORT_WARN_SECONDS
    ):
        _add_warning(
            warnings,
            "WARN",
            "strength_report_stale",
            f"strength-efficiency report age={strength_age:.0f}s",
        )
    pause_request = _read_json(root / "status" / "arena-gpu-pause.json")
    pause_ack = _read_json(root / "status" / "arena-gpu-pause.ack.json")
    if (
        pause_request is not None
        and pause_ack is not None
        and pause_request.get("token") != pause_ack.get("token")
    ):
        _add_warning(
            warnings,
            "ERROR",
            "pause_token_mismatch",
            "pause request and acknowledgement tokens differ",
        )
    pause = {
        "coordinator": coordinator.get("pause_lease"),
        "request": pause_request,
        "acknowledgement": pause_ack,
    }

    disk = _disk_status(root)
    disk_fraction = _number(disk.get("used_fraction")) or 0.0
    inode_fraction = _number(disk.get("inode_used_fraction")) or 0.0
    if max(disk_fraction, inode_fraction) >= 0.95:
        _add_warning(warnings, "ERROR", "disk_critical", "disk or inode use >=95%")
    elif max(disk_fraction, inode_fraction) >= 0.85:
        _add_warning(warnings, "WARN", "disk_high", "disk or inode use >=85%")

    gpus, gpu_error = _gpu_status()
    hardware_health = (
        _read_json(root / "status" / "hardware-health.json")
        or _read_json(root / "status" / "hardware-health-startup.json")
        or {}
    )
    if gpu_error:
        _add_warning(warnings, "WARN", gpu_error, "GPU telemetry is unavailable")
    if hardware_health and hardware_health.get("healthy") is not True:
        reasons = []
        hardware_rows = hardware_health.get("gpus")
        for row in hardware_rows if isinstance(hardware_rows, list) else []:
            if isinstance(row, dict):
                reasons.extend(
                    f"GPU {row.get('index')}: {reason}"
                    for reason in row.get("reasons", [])
                )
        detail = "; ".join(reasons) or str(
            hardware_health.get("query_error", "hardware health gate failed")
        )
        _add_warning(warnings, "ERROR", "gpu_health_gate", detail)
    for gpu in gpus:
        temperature = _number(gpu.get("temperature.gpu"))
        ecc = _number(gpu.get("ecc.errors.uncorrected.volatile.total"))
        aggregate_sram = _number(gpu.get("ecc.errors.uncorrected.aggregate.sram"))
        aggregate_dram = _number(gpu.get("ecc.errors.uncorrected.aggregate.dram"))
        if temperature is not None and temperature >= 90:
            _add_warning(
                warnings,
                "ERROR",
                "gpu_temperature",
                f"GPU {gpu['index']} temperature={temperature:g}C",
            )
        elif temperature is not None and temperature >= 80:
            _add_warning(
                warnings,
                "WARN",
                "gpu_temperature",
                f"GPU {gpu['index']} temperature={temperature:g}C",
            )
        if ecc is not None and ecc > 0:
            _add_warning(
                warnings,
                "ERROR",
                "gpu_ecc",
                f"GPU {gpu['index']} volatile uncorrected ECC={ecc:g}",
            )
        if (aggregate_sram or 0.0) > 0 or (aggregate_dram or 0.0) > 0:
            _add_warning(
                warnings,
                "ERROR",
                "gpu_ecc_aggregate",
                f"GPU {gpu['index']} aggregate uncorrected "
                f"SRAM={aggregate_sram or 0:g} DRAM={aggregate_dram or 0:g}",
            )

    continuity = _continuity_status(continuity_state_path, now_ns=now)
    if continuity.get("configured") and continuity.get("valid") is not True:
        _add_warning(
            warnings,
            "ERROR",
            "continuity_state_invalid",
            "host-level continuity state is missing or invalid",
        )
    reconciliation_age = continuity.get("reconciliation_age_seconds")
    if (
        continuity.get("configured")
        and continuity.get("valid") is True
        and (
            not isinstance(reconciliation_age, int | float)
            or float(reconciliation_age) > CONTINUITY_STALE_SECONDS
        )
    ):
        _add_warning(
            warnings,
            "ERROR",
            "continuity_reconciliation_stale",
            "host-level continuity reconciliation heartbeat is stale",
        )
    phase = continuity.get("phase")
    if (isinstance(phase, str) and phase.startswith("blocked_")) or continuity.get(
        "blocked_reason"
    ) is not None:
        _add_warning(
            warnings,
            "ERROR",
            "continuity_blocked",
            f"host-level continuity phase is {phase}",
        )
    disaster_recovery = _disaster_recovery_status(
        disaster_backup_root,
        run_id=orchestration.get("run_id"),
        run_root=root,
        now_ns=now,
    )
    if (
        disaster_recovery.get("configured")
        and disaster_recovery.get("valid") is not True
    ):
        _add_warning(
            warnings,
            "ERROR",
            "disaster_backup_invalid",
            "Lambda disaster-recovery snapshot is missing or invalid",
        )
    disaster_age = disaster_recovery.get("source_cutoff_age_seconds")
    if (
        disaster_recovery.get("valid") is True
        and isinstance(disaster_age, int | float)
        and float(disaster_age) > DISASTER_BACKUP_STALE_SECONDS
    ):
        _add_warning(
            warnings,
            "ERROR",
            "disaster_backup_stale",
            f"latest Lambda snapshot age={float(disaster_age):.0f}s",
        )

    status = max(
        (item["severity"] for item in warnings),
        key=lambda value: SEVERITY[value],
        default="OK",
    )
    return {
        "schema_version": 1,
        "timestamp": _utc_now(),
        "status": status,
        "run_root": str(root),
        "training_objective": training_objective,
        "objective_contract": objective_contract,
        "autonomous": {
            "enabled": autonomous_enabled,
            "provenance": provenance if autonomous_enabled else None,
        },
        "service": service,
        "coordinator": {
            "state": coordinator.get("state"),
            "draining": coordinator.get("draining"),
            "pause_lease": coordinator.get("pause_lease"),
            "failure": fatal or coordinator.get("failure"),
        },
        "workers": workers_output,
        "learner": learner,
        "actors": actor_fleet,
        "replay": replay,
        "recovery": recovery,
        "arena": arena,
        "arena_history": arena_history,
        "weighted_promotion": weighted_promotion,
        "strength_efficiency": strength_efficiency,
        "pause": pause,
        "disk": disk,
        "gpus": gpus,
        "gpu_health": hardware_health,
        "continuity": continuity,
        "disaster_recovery": disaster_recovery,
        "warnings": warnings,
    }


def format_text(snapshot: Mapping[str, object]) -> str:
    coordinator = snapshot.get("coordinator")
    coordinator = coordinator if isinstance(coordinator, Mapping) else {}
    failure = coordinator.get("failure")
    failure = failure if isinstance(failure, Mapping) else {}
    learner = snapshot.get("learner")
    learner = learner if isinstance(learner, Mapping) else {}
    actors = snapshot.get("actors")
    actors = actors if isinstance(actors, Mapping) else {}
    throughput = actors.get("throughput")
    throughput = throughput if isinstance(throughput, Mapping) else {}
    rates = throughput.get("fleet")
    rates = rates if isinstance(rates, Mapping) else {}
    replay = snapshot.get("replay")
    replay = replay if isinstance(replay, Mapping) else {}
    states = replay.get("states")
    states = states if isinstance(states, Mapping) else {}
    ready = states.get("ready")
    ready = ready if isinstance(ready, Mapping) else {}
    arena = snapshot.get("arena")
    arena = arena if isinstance(arena, Mapping) else {}
    recovery = snapshot.get("recovery")
    recovery = recovery if isinstance(recovery, Mapping) else {}
    arena_history = snapshot.get("arena_history")
    arena_history = arena_history if isinstance(arena_history, Mapping) else {}
    strength_efficiency = snapshot.get("strength_efficiency")
    strength_efficiency = (
        strength_efficiency if isinstance(strength_efficiency, Mapping) else {}
    )
    weighted_promotion = snapshot.get("weighted_promotion")
    weighted_promotion = (
        weighted_promotion if isinstance(weighted_promotion, Mapping) else {}
    )
    recent_evaluations = arena_history.get("recent")
    recent_evaluations = (
        recent_evaluations if isinstance(recent_evaluations, list) else []
    )
    latest_evaluation = (
        recent_evaluations[-1]
        if recent_evaluations and isinstance(recent_evaluations[-1], Mapping)
        else {}
    )
    warnings = snapshot.get("warnings")
    warnings = warnings if isinstance(warnings, list) else []
    warning_codes = ",".join(
        str(item.get("code"))
        for item in warnings
        if isinstance(item, Mapping) and item.get("code")
    )
    headline_elo = _number(strength_efficiency.get("headline_elo"))
    displayed_elo = (
        headline_elo
        if headline_elo is not None
        else latest_evaluation.get("elo_difference")
    )
    elo_source = (
        strength_efficiency.get("headline_source")
        if headline_elo is not None
        else "latest_arena"
    )
    segment_target = learner.get("utd_segment_target_updates_per_new_sample")
    if _number(segment_target) is None:
        segment_target = learner.get("target_updates_per_new_sample")
    continuity = snapshot.get("continuity")
    continuity = continuity if isinstance(continuity, Mapping) else {}
    disaster_recovery = snapshot.get("disaster_recovery")
    disaster_recovery = (
        disaster_recovery if isinstance(disaster_recovery, Mapping) else {}
    )
    return (
        f"{snapshot.get('timestamp')} {snapshot.get('status')} "
        f"objective={snapshot.get('training_objective', 'generalist')} "
        f"learner={learner.get('step')}/{learner.get('target_steps')} "
        f"phase={learner.get('phase')} eps={_compact(learner.get('examples_per_second'))} "
        f"utd_segment={_compact(learner.get('segment_updates_per_new_sample'))}/"
        f"{_compact(segment_target)} "
        f"loader_workers={_count(learner.get('loader_workers_effective'))} "
        f"loader_pool={_count(learner.get('loader_pool_starts'))}/"
        f"{_count(learner.get('loader_pool_rebinds'))}/"
        f"{_count(learner.get('loader_pool_shutdowns'))} "
        f"window_reuse={_flag(learner.get('window_reuse'))} "
        f"window_setup={_seconds(learner.get('window_setup_amortized_seconds'))} "
        f"actors={actors.get('workers')} "
        f"policy={_percent(actors.get('policy_supervision_rate'))} "
        f"games/s={_compact(rates.get('games_per_second'))} "
        f"samples/s={_compact(rates.get('samples_per_second'))} "
        f"eval_rows/s={_compact(rates.get('evaluator_rows_per_second'))} "
        f"replay_samples={ready.get('samples', 0)} shards={ready.get('shards', 0)} "
        f"models={recovery.get('selfplay_step')}/"
        f"{recovery.get('candidate_step')}/{arena.get('champion_step')} "
        f"arena={arena.get('decision', arena.get('phase', 'waiting'))} "
        f"promotion_evals={arena_history.get('promotion_evaluations', 0)} "
        f"crossplay_evals={arena_history.get('crossplay_evaluations', 0)} "
        f"elo={_compact(displayed_elo)} elo_source={elo_source or 'n/a'} "
        f"weighted_blocks={_count(weighted_promotion.get('complete_blocks'))}/"
        f"{_count(weighted_promotion.get('max_blocks'))} "
        f"weighted_lcb={_compact(weighted_promotion.get('anytime_lower_elo'))} "
        f"weighted_state={weighted_promotion.get('sequential_state') or 'n/a'} "
        f"failure={failure.get('failure_class', '-')} "
        f"continuity={continuity.get('phase', 'n/a')} "
        f"active_workload={continuity.get('active_workload_id', 'n/a')} "
        f"dr_age={_seconds(disaster_recovery.get('source_cutoff_age_seconds'))} "
        f"warnings={warning_codes or '-'}"
    )


def _compact(value: object) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}m"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.2f}k"
    return f"{number:.2f}"


def _percent(value: object) -> str:
    number = _number(value)
    return f"{number:.0%}" if number is not None else "n/a"


def _count(value: object) -> str:
    number = _number(value)
    return str(int(number)) if number is not None and number.is_integer() else "n/a"


def _flag(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _seconds(value: object) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}s" if abs(number) < 0.01 else f"{_compact(number)}s"


def _repair_partial_jsonl_tail(stream) -> int:
    size = stream.seek(0, os.SEEK_END)
    if not size:
        return 0
    stream.seek(size - 1)
    if stream.read(1) == b"\n":
        return size
    scan_end = size
    complete_offset = 0
    while scan_end > 0:
        scan_start = max(0, scan_end - 1024 * 1024)
        stream.seek(scan_start)
        block = stream.read(scan_end - scan_start)
        last_newline = block.rfind(b"\n")
        if last_newline >= 0:
            complete_offset = scan_start + last_newline + 1
            break
        scan_end = scan_start
    stream.truncate(complete_offset)
    stream.flush()
    os.fsync(stream.fileno())
    return complete_offset


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rotate_telemetry_jsonl(output: Path, *, retain_files: int) -> None:
    pattern = re.compile(
        rf"^{re.escape(output.stem)}\.(?P<sequence>[0-9]+)"
        rf"{re.escape(output.suffix)}$"
    )
    archives: list[tuple[int, Path]] = []
    for path in output.parent.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_file():
            raise OSError(f"telemetry archive is unsafe: {path}")
        archives.append((int(match.group("sequence")), path))
    sequence = max(
        time.time_ns(),
        max((existing for existing, _path in archives), default=0) + 1,
    )
    rotated = output.with_name(f"{output.stem}.{sequence}{output.suffix}")
    if rotated.exists() or rotated.is_symlink():
        raise OSError(f"telemetry rotation destination already exists: {rotated}")
    os.replace(output, rotated)
    archives.append((sequence, rotated))
    archives.sort()
    for _sequence, stale in archives[:-retain_files]:
        if stale.is_symlink() or not stale.is_file():
            raise OSError(f"telemetry archive is unsafe: {stale}")
        stale.unlink()
    _fsync_directory(output.parent)


def _append_snapshot_jsonl(
    path: Path,
    snapshot: Mapping[str, object],
    *,
    maximum_bytes: int = DEFAULT_TELEMETRY_MAX_BYTES,
    retain_files: int = DEFAULT_TELEMETRY_RETAIN_FILES,
) -> None:
    if maximum_bytes <= 0:
        raise ValueError("telemetry maximum bytes must be positive")
    if retain_files <= 0:
        raise ValueError("telemetry retained files must be positive")
    source = path.expanduser()
    if source.is_symlink():
        raise OSError(f"telemetry output may not be a symlink: {source}")
    output = source.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if len(line) > maximum_bytes:
        raise ValueError("one telemetry snapshot exceeds the configured file limit")
    lock_path = output.with_name(f".{output.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            created = not output.exists()
            if output.exists() and (output.is_symlink() or not output.is_file()):
                raise OSError(f"telemetry output is unsafe: {output}")
            with output.open("a+b") as stream:
                size = _repair_partial_jsonl_tail(stream)
            if size and size + len(line) > maximum_bytes:
                _rotate_telemetry_jsonl(output, retain_files=retain_files)
                created = True
            with output.open("a+b") as stream:
                stream.seek(0, os.SEEK_END)
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            if created:
                _fsync_directory(output.parent)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_monitor(
    run_root: Path,
    *,
    profile_path: Path | None,
    unit: str | None,
    interval: float,
    once: bool,
    output_format: str,
    stop_requested: Callable[[], bool],
    continuity_state_path: Path | None = None,
    disaster_backup_root: Path | None = None,
    telemetry_output: Path | None = None,
    telemetry_max_bytes: int = DEFAULT_TELEMETRY_MAX_BYTES,
    telemetry_retain_files: int = DEFAULT_TELEMETRY_RETAIN_FILES,
) -> None:
    next_tick = time.monotonic()
    while not stop_requested():
        try:
            if continuity_state_path is not None and disaster_backup_root is not None:
                snapshot = collect_snapshot(
                    run_root,
                    unit=unit,
                    profile_path=profile_path,
                    continuity_state_path=continuity_state_path,
                    disaster_backup_root=disaster_backup_root,
                )
            elif continuity_state_path is not None:
                snapshot = collect_snapshot(
                    run_root,
                    unit=unit,
                    profile_path=profile_path,
                    continuity_state_path=continuity_state_path,
                )
            elif disaster_backup_root is not None:
                snapshot = collect_snapshot(
                    run_root,
                    unit=unit,
                    profile_path=profile_path,
                    disaster_backup_root=disaster_backup_root,
                )
            else:
                snapshot = collect_snapshot(
                    run_root,
                    unit=unit,
                    profile_path=profile_path,
                )
        except Exception as error:  # monitor must report and continue
            snapshot = {
                "schema_version": 1,
                "timestamp": _utc_now(),
                "status": "ERROR",
                "warnings": [
                    {
                        "severity": "ERROR",
                        "code": "monitor_exception",
                        "message": f"{type(error).__name__}: {error}",
                    }
                ],
            }
        if telemetry_output is not None:
            try:
                _append_snapshot_jsonl(
                    telemetry_output,
                    snapshot,
                    maximum_bytes=telemetry_max_bytes,
                    retain_files=telemetry_retain_files,
                )
            except (OSError, ValueError) as error:
                raw_warnings = snapshot.get("warnings")
                warnings = raw_warnings if isinstance(raw_warnings, list) else []
                warnings.append(
                    {
                        "severity": "WARN",
                        "code": "telemetry_persistence_failed",
                        "message": f"{type(error).__name__}: {error}",
                    }
                )
                snapshot["warnings"] = warnings
                if snapshot.get("status") == "OK":
                    snapshot["status"] = "WARN"
        serialized = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        line = serialized if output_format == "jsonl" else format_text(snapshot)
        print(line, flush=True)
        if once:
            return
        next_tick += interval
        while not stop_requested():
            remaining = next_tick - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.5))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.interval <= 0:
        raise SystemExit("--interval must be positive")
    if arguments.telemetry_max_bytes <= 0:
        raise SystemExit("--telemetry-max-bytes must be positive")
    if arguments.telemetry_retain_files <= 0:
        raise SystemExit("--telemetry-retain-files must be positive")
    if arguments.continuity_manifest is not None:
        try:
            target = resolve_monitor_target(
                arguments.continuity_manifest,
                run_root=arguments.run_root,
                profile_path=arguments.profile,
                unit=arguments.unit,
                continuity_state_path=arguments.continuity_state,
                disaster_backup_root=arguments.disaster_backup_root,
            )
        except (ContinuityError, ValueError) as error:
            parser.error(str(error))
    else:
        if arguments.run_root is None:
            parser.error(
                "--run-root is required unless --continuity-manifest is provided"
            )
        target = MonitorTarget(
            run_root=arguments.run_root,
            profile_path=arguments.profile,
            unit=arguments.unit,
            continuity_state_path=arguments.continuity_state,
            disaster_backup_root=arguments.disaster_backup_root,
        )
    stopped = False

    def request_stop(_signal_number, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run_monitor(
        target.run_root,
        profile_path=target.profile_path,
        unit=target.unit,
        interval=arguments.interval,
        once=arguments.once,
        output_format=arguments.format,
        stop_requested=lambda: stopped,
        continuity_state_path=target.continuity_state_path,
        disaster_backup_root=target.disaster_backup_root,
        telemetry_output=arguments.telemetry_output,
        telemetry_max_bytes=arguments.telemetry_max_bytes,
        telemetry_retain_files=arguments.telemetry_retain_files,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
