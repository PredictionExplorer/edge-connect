#!/usr/bin/env python3
"""Persist telemetry and strength reports for the active ablation queue arm."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path


class ActiveQueueMonitorError(RuntimeError):
    """The active queue monitor encountered unsafe control-plane state."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-state", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--monitor-script", type=Path, required=True)
    parser.add_argument("--strength-script", type=Path, required=True)
    parser.add_argument("--continuity-state", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--strength-interval", type=float, default=900.0)
    return parser


def _read_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ActiveQueueMonitorError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ActiveQueueMonitorError(f"{name} must contain a JSON object")
    return payload


def _active_context(
    state_path: Path,
    mapping_path: Path,
) -> tuple[str, Path, Path, Path] | None:
    if not state_path.is_file():
        return None
    state = _read_json(state_path, name="queue state")
    queue_status = state.get("queue_status")
    if queue_status in {"completed", "failed"}:
        raise SystemExit(0)
    if queue_status not in {"pending", "running"}:
        return None
    raw_arms = state.get("arms")
    if not isinstance(raw_arms, list):
        raise ActiveQueueMonitorError("queue state arms must be a list")
    running = [
        arm
        for arm in raw_arms
        if isinstance(arm, Mapping) and arm.get("status") == "running"
    ]
    if not running:
        return None
    if len(running) != 1:
        raise ActiveQueueMonitorError("queue exposes multiple running arms")
    arm = running[0]
    treatment = arm.get("treatment")
    if not isinstance(treatment, str):
        raise ActiveQueueMonitorError("running arm treatment is invalid")

    mapping = _read_json(mapping_path, name="active-arm mapping")
    raw_mappings = mapping.get("arms")
    if not isinstance(raw_mappings, Mapping):
        raise ActiveQueueMonitorError("active-arm mapping has no arms")
    configured = raw_mappings.get(treatment)
    if not isinstance(configured, Mapping):
        raise ActiveQueueMonitorError("running arm is absent from active-arm mapping")
    values = []
    for name in ("run_root", "profile", "backup_root"):
        value = configured.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ActiveQueueMonitorError(f"mapped {name} must be an absolute path")
        values.append(Path(value))
    run_root, profile, backup_root = values
    if arm.get("run_root") != str(run_root):
        raise ActiveQueueMonitorError("queue run root differs from active-arm mapping")
    return treatment, run_root, profile, backup_root


def monitor_active_queue(
    *,
    queue_state: Path,
    mapping: Path,
    monitor_script: Path,
    strength_script: Path,
    continuity_state: Path,
    interval: float,
    strength_interval: float,
) -> None:
    if interval <= 0 or strength_interval <= 0:
        raise ActiveQueueMonitorError("monitor intervals must be positive")
    stopped = False

    def request_stop(_signal_number, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    last_strength: dict[str, float] = {}
    while not stopped:
        started = time.monotonic()
        context = _active_context(queue_state, mapping)
        if context is not None:
            treatment, run_root, profile, backup_root = context
            telemetry = run_root / "status" / "monitor-5s.jsonl"
            command = [
                sys.executable,
                str(monitor_script),
                "--run-root",
                str(run_root),
                "--profile",
                str(profile),
                "--continuity-state",
                str(continuity_state),
                "--disaster-backup-root",
                str(backup_root),
                "--once",
                "--format",
                "jsonl",
                "--telemetry-output",
                str(telemetry),
            ]
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
            )
            if completed.returncode != 0:
                raise ActiveQueueMonitorError(
                    f"monitor snapshot failed with status {completed.returncode}"
                )
            now = time.monotonic()
            if now - last_strength.get(treatment, 0.0) >= strength_interval:
                report = subprocess.run(
                    [
                        sys.executable,
                        str(strength_script),
                        "--run-root",
                        str(run_root),
                        "--provisioned-gpus",
                        "8",
                        "--output",
                        str(run_root / "strength-efficiency.json"),
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                )
                if report.returncode not in (0, 3):
                    raise ActiveQueueMonitorError(
                        f"strength report failed with status {report.returncode}"
                    )
                last_strength[treatment] = now
        remaining = interval - (time.monotonic() - started)
        if remaining > 0 and not stopped:
            time.sleep(remaining)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        monitor_active_queue(
            queue_state=arguments.queue_state.expanduser().resolve(),
            mapping=arguments.mapping.expanduser().resolve(),
            monitor_script=arguments.monitor_script.expanduser().resolve(),
            strength_script=arguments.strength_script.expanduser().resolve(),
            continuity_state=arguments.continuity_state.expanduser().resolve(),
            interval=arguments.interval,
            strength_interval=arguments.strength_interval,
        )
    except ActiveQueueMonitorError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
