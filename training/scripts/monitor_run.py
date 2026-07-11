#!/usr/bin/env python3
"""Print periodic, read-only health summaries for one training run."""

from __future__ import annotations

import argparse
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
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


def _latest_jsonl(path: Path, *, maximum_bytes: int = 2 * 1024 * 1024):
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
        if isinstance(payload, dict):
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


def _age_seconds(timestamp_ns: object, now_ns: int) -> float | None:
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
        return None
    return max(0.0, (now_ns - timestamp_ns) / 1_000_000_000.0)


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
    now_ns: int | None = None,
) -> dict[str, object]:
    root = run_root.expanduser().resolve()
    now = time.time_ns() if now_ns is None else now_ns
    warnings: list[dict[str, str]] = []
    profile = _read_json(root / "profile.json")
    if profile is None:
        try:
            loaded = yaml.safe_load((root / "profile.yaml").read_text(encoding="utf-8"))
            profile = loaded if isinstance(loaded, dict) else {}
        except (OSError, yaml.YAMLError):
            profile = {}
    orchestration = _mapping(profile.get("orchestration"))
    shutdown = _mapping(orchestration.get("shutdown"))
    stale_threshold = _number(shutdown.get("stale_heartbeat_seconds")) or 180.0
    stall_threshold = _number(shutdown.get("stall_timeout_seconds")) or 1_800.0
    target_steps = _mapping(profile.get("learner")).get("steps")

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
                    "heartbeat_age_seconds": heartbeat_age,
                    "progress_age_seconds": progress_age,
                }
            )
    else:
        _add_warning(warnings, "ERROR", "workers_missing", "worker map is missing")

    learner_metric = _latest_jsonl(root / "learner" / "metrics.jsonl") or {}
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
        "losses": losses,
        "gradient_norm": learner_metric.get("gradient_norm"),
        "feature_path": learner_metric.get("feature_path"),
    }

    actors = []
    for metrics_path in sorted((root / "metrics").glob("actor-gpu-*.jsonl")):
        metric = _latest_jsonl(metrics_path)
        if metric is not None:
            actors.append(metric)
    actor_fleet = {
        "workers": len(actors),
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

    arena = _read_json(root / "arena" / "promotion-status.json") or {}
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
        "arena": arena,
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
        f"games/s={_compact(rates.get('games_per_second'))} "
        f"samples/s={_compact(rates.get('samples_per_second'))} "
        f"eval_rows/s={_compact(rates.get('evaluator_rows_per_second'))} "
        f"replay_samples={ready.get('samples', 0)} shards={ready.get('shards', 0)} "
        f"arena={arena.get('decision', arena.get('phase', 'waiting'))} "
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


def run_monitor(
    run_root: Path,
    *,
    unit: str | None,
    interval: float,
    once: bool,
    output_format: str,
    stop_requested: Callable[[], bool],
) -> None:
    next_tick = time.monotonic()
    while not stop_requested():
        try:
            snapshot = collect_snapshot(run_root, unit=unit)
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
        unit=arguments.unit,
        interval=arguments.interval,
        once=arguments.once,
        output_format=arguments.format,
        stop_requested=lambda: stopped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
