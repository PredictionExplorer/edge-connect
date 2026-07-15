#!/usr/bin/env python3
"""Print periodic, read-only health summaries for one training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

SEVERITY = {"OK": 0, "WARN": 1, "ERROR": 2}
_DIGEST_CACHE: dict[Path, tuple[int, int, int, str]] = {}
_ARENA_RESULT_CACHE: dict[Path, tuple[int, int, dict[str, object]]] = {}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--unit")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--format", choices=("text", "jsonl"), default="text")
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


def _replay_status(path: Path) -> tuple[dict[str, object], str | None]:
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
        connection.rollback()
        connection.close()
    except (OSError, sqlite3.Error) as error:
        return {}, f"replay_query_failed:{type(error).__name__}"
    return {"states": states, "samples_by_ring": rings, "games": games}, None


def _arena_history(run_root: Path, *, limit: int = 5) -> dict[str, object]:
    learner_root = run_root / "learner"
    steps: dict[str, int] = {}
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
            summary = {
                "candidate": result.get("candidate"),
                "baseline": result.get("baseline"),
                "decision": promotion.get("decision"),
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
            }
            _ARENA_RESULT_CACHE[result_path] = (
                stat.st_mtime_ns,
                stat.st_size,
                summary,
            )
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
        completed.append(
            {
                "completed_ns": completed_ns,
                "candidate_step": steps.get(str(summary.get("candidate"))),
                "baseline_step": steps.get(str(summary.get("baseline"))),
                "decision": decision,
                "elo_difference": aggregate.get("elo_difference"),
                "elo_lower": lower_elo,
                "wins": aggregate.get("wins"),
                "losses": aggregate.get("losses"),
                "games": aggregate.get("games"),
                "per_ring_elo": summary.get("per_ring_elo"),
            }
        )
    for stale_path in set(_ARENA_RESULT_CACHE) - existing_paths:
        _ARENA_RESULT_CACHE.pop(stale_path, None)
    completed.sort(key=lambda row: _number(row.get("completed_ns")) or 0.0)
    return {
        "completed_evaluations": len(completed),
        "promotions": sum(row.get("decision") == "promote" for row in completed),
        "rejections": sum(
            row.get("decision")
            in ("reject", "reject_ring_regression", "reject_max_pairs")
            for row in completed
        ),
        "superseded_candidates": superseded,
        "recent": completed[-limit:],
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


def collect_snapshot(
    run_root: Path,
    *,
    unit: str | None = None,
    profile_path: Path | None = None,
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
    shutdown = _mapping(orchestration.get("shutdown"))
    stale_threshold = _number(shutdown.get("stale_heartbeat_seconds")) or 180.0
    stall_threshold = _number(shutdown.get("stall_timeout_seconds")) or 1_800.0
    learner_config = _mapping(profile.get("learner"))
    target_steps = (
        "unlimited"
        if learner_config.get("unlimited") is True
        else learner_config.get("steps")
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
                    "phase": heartbeat.get("phase"),
                    "progress": heartbeat.get("progress"),
                    "active_ring_weights": heartbeat.get("active_ring_weights"),
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
        "updates_per_new_sample": learner_metric.get("updates_per_new_sample"),
        "learning_rates": learner_metric.get("learning_rates"),
        "replay_samples_by_ring": learner_metric.get("replay_samples_by_ring"),
        "ring_batch_weights": learner_metric.get("ring_batch_weights"),
        "losses": losses,
        "gradient_norm": learner_metric.get("gradient_norm"),
        "feature_path": learner_metric.get("feature_path"),
    }

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
    actor_fleet = {
        "workers": len(actors),
        "policy_supervision_rate": policy_supervision_rate,
        "active_ring_weights": (
            list(next(iter(weight_variants))) if len(weight_variants) == 1 else None
        ),
        "ring_weight_variants": [list(weights) for weights in sorted(weight_variants)],
        "noncompliant_weight_workers": noncompliant_weight_workers,
        "low_policy_workers": low_policy_workers,
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

    replay, replay_error = _replay_status(root / "replay" / "manifest.sqlite3")
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
        "durable_step": durable_step,
        "checkpoint_age_seconds": recovery_age,
        "replay_backup_age_seconds": backup_age,
        "replay_backup_valid": backup_valid,
    }

    arena = _read_json(root / "arena" / "promotion-status.json") or {}
    arena_history = _arena_history(root)
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
    if gpu_error:
        _add_warning(warnings, "WARN", gpu_error, "GPU telemetry is unavailable")
    for gpu in gpus:
        temperature = _number(gpu.get("temperature.gpu"))
        ecc = _number(gpu.get("ecc.errors.uncorrected.volatile.total"))
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
        "service": service,
        "coordinator": {
            "state": coordinator.get("state"),
            "draining": coordinator.get("draining"),
            "pause_lease": coordinator.get("pause_lease"),
        },
        "workers": workers_output,
        "learner": learner,
        "actors": actor_fleet,
        "replay": replay,
        "recovery": recovery,
        "arena": arena,
        "arena_history": arena_history,
        "pause": pause,
        "disk": disk,
        "gpus": gpus,
        "warnings": warnings,
    }


def format_text(snapshot: Mapping[str, object]) -> str:
    learner = snapshot.get("learner")
    learner = learner if isinstance(learner, Mapping) else {}
    actors = snapshot.get("actors")
    actors = actors if isinstance(actors, Mapping) else {}
    rates = actors.get("latest_batch_rate_sum")
    rates = rates if isinstance(rates, Mapping) else {}
    replay = snapshot.get("replay")
    replay = replay if isinstance(replay, Mapping) else {}
    states = replay.get("states")
    states = states if isinstance(states, Mapping) else {}
    ready = states.get("ready")
    ready = ready if isinstance(ready, Mapping) else {}
    arena = snapshot.get("arena")
    arena = arena if isinstance(arena, Mapping) else {}
    arena_history = snapshot.get("arena_history")
    arena_history = arena_history if isinstance(arena_history, Mapping) else {}
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
    return (
        f"{snapshot.get('timestamp')} {snapshot.get('status')} "
        f"learner={learner.get('step')}/{learner.get('target_steps')} "
        f"phase={learner.get('phase')} eps={_compact(learner.get('examples_per_second'))} "
        f"actors={actors.get('workers')} "
        f"policy={_percent(actors.get('policy_supervision_rate'))} "
        f"games/s={_compact(rates.get('games_per_second'))} "
        f"samples/s={_compact(rates.get('samples_per_second'))} "
        f"eval_rows/s={_compact(rates.get('evaluator_rows_per_second'))} "
        f"replay_samples={ready.get('samples', 0)} shards={ready.get('shards', 0)} "
        f"arena={arena.get('decision', arena.get('phase', 'waiting'))} "
        f"elo={_compact(latest_evaluation.get('elo_difference'))} "
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


def run_monitor(
    run_root: Path,
    *,
    profile_path: Path | None,
    unit: str | None,
    interval: float,
    once: bool,
    output_format: str,
    stop_requested: Callable[[], bool],
) -> None:
    next_tick = time.monotonic()
    while not stop_requested():
        try:
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
        line = (
            json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
            if output_format == "jsonl"
            else format_text(snapshot)
        )
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
    arguments = _parser().parse_args(argv)
    if arguments.interval <= 0:
        raise SystemExit("--interval must be positive")
    stopped = False

    def request_stop(_signal_number, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run_monitor(
        arguments.run_root,
        profile_path=arguments.profile,
        unit=arguments.unit,
        interval=arguments.interval,
        once=arguments.once,
        output_format=arguments.format,
        stop_requested=lambda: stopped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
