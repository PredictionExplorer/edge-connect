#!/usr/bin/env python3
"""Run one forked Elo ablation until its wall or leaf-evaluation budget is met."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from startrain.config import load_config
from startrain.runtime import SignalLatch, atomic_json, load_run_identity

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-elo-ablation-run"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--orchestrator", default="startrain-orchestrate")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EvaluatorRows:
    """Incrementally sum evaluator rows from append-only actor metric files."""

    def __init__(self, metrics_root: Path) -> None:
        self.metrics_root = metrics_root
        self.offsets: dict[Path, int] = {}
        self.rows = 0

    def refresh(self) -> int:
        for path in sorted(self.metrics_root.glob("actor-*.jsonl")):
            offset = self.offsets.get(path, 0)
            with path.open("r", encoding="utf-8") as stream:
                stream.seek(offset)
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        stream.seek(position)
                        break
                    if line.strip():
                        payload = json.loads(line)
                        value = payload.get("evaluator_rows")
                        if (
                            isinstance(value, int | float)
                            and not isinstance(value, bool)
                            and value >= 0
                        ):
                            self.rows += int(value)
                self.offsets[path] = stream.tell()
        return self.rows


def _positive_number(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be positive")
    return float(value)


def _resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        resolved = candidate.resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise FileNotFoundError(f"orchestrator is not executable: {resolved}")
        return str(resolved)
    resolved_value = shutil.which(value)
    if resolved_value is None:
        raise FileNotFoundError(f"orchestrator is not on PATH: {value}")
    return resolved_value


def _terminate(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_elo_ablation(
    *,
    config_path: Path,
    orchestrator: str,
    poll_seconds: float,
) -> dict[str, object]:
    if poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    profile = config_path.expanduser().resolve()
    if not profile.is_file():
        raise FileNotFoundError(f"ablation profile does not exist: {profile}")
    experiment = load_config(profile)
    root = Path(experiment.orchestration.directories.root).expanduser().resolve()
    metadata_path = root / "ablation.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ablation metadata does not exist: {metadata_path}")
    if (root / "coordinator.lock").exists():
        raise RuntimeError("ablation run root already has a coordinator lock")
    metadata = _read_json(metadata_path)
    if metadata.get("report") != "startrain-elo-ablation-branch":
        raise ValueError("unsupported ablation metadata")
    if metadata.get("measurement_started_ns") is not None:
        raise RuntimeError("ablation measurement has already started")
    configured_profile = metadata.get("profile")
    configured_digest = metadata.get("profile_sha256")
    if (
        not isinstance(configured_profile, str)
        or Path(configured_profile).resolve() != profile
        or not isinstance(configured_digest, str)
        or _sha256(profile) != configured_digest
    ):
        raise ValueError("ablation profile does not match its frozen metadata")
    identity = load_run_identity(root / "run.json")
    if experiment.orchestration.run_id != identity.run_id:
        raise ValueError("ablation profile and run identity disagree")
    wall_budget_seconds = _positive_number(
        "wall_budget_seconds", metadata.get("wall_budget_seconds")
    )
    leaf_budget = int(_positive_number("leaf_budget", metadata.get("leaf_budget")))
    executable = _resolve_executable(orchestrator)
    started_ns = time.time_ns()
    metadata.update(
        {
            "measurement_started_ns": started_ns,
            "measurement_stopped_ns": None,
            "measurement_stop_reason": None,
            "measurement_exit_code": None,
            "measurement_evaluator_rows": 0,
        }
    )
    atomic_json(metadata_path, metadata)

    latch = SignalLatch()
    latch.install()
    tracker = EvaluatorRows(root / experiment.orchestration.directories.metrics)
    process = subprocess.Popen(
        [executable, "--config", str(profile)],
        start_new_session=True,
    )
    stop_reason = "process_exit"
    try:
        while process.poll() is None:
            rows = tracker.refresh()
            elapsed = (time.time_ns() - started_ns) / 1_000_000_000.0
            if latch.is_set():
                stop_reason = f"signal_{latch.signal_number}"
                break
            if rows >= leaf_budget:
                stop_reason = "leaf_budget"
                break
            if elapsed >= wall_budget_seconds:
                stop_reason = "wall_budget"
                break
            time.sleep(poll_seconds)
    finally:
        grace = (
            experiment.orchestration.shutdown.terminate_grace_seconds
            + experiment.orchestration.shutdown.kill_grace_seconds
        )
        _terminate(process, grace_seconds=grace)
    rows = tracker.refresh()
    exit_code = process.returncode
    stopped_ns = time.time_ns()
    metadata = _read_json(metadata_path)
    metadata.update(
        {
            "measurement_stopped_ns": stopped_ns,
            "measurement_stop_reason": stop_reason,
            "measurement_exit_code": exit_code,
            "measurement_evaluator_rows": rows,
        }
    )
    atomic_json(metadata_path, metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": (
            "complete"
            if stop_reason in {"leaf_budget", "wall_budget"} and exit_code in (0, -15)
            else "failed"
        ),
        "treatment": metadata.get("treatment"),
        "run_root": str(root),
        "started_ns": started_ns,
        "stopped_ns": stopped_ns,
        "wall_seconds": (stopped_ns - started_ns) / 1_000_000_000.0,
        "stop_reason": stop_reason,
        "exit_code": exit_code,
        "evaluator_rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = run_elo_ablation(
            config_path=arguments.config,
            orchestrator=arguments.orchestrator,
            poll_seconds=arguments.poll_seconds,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "report": REPORT_NAME,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
