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
from startrain.orchestration import (
    FATAL_WORKER_EXIT_CODE,
    TRANSIENT_WORKER_EXIT_CODE,
)
from startrain.runtime import SignalLatch, atomic_json, load_run_identity

if __package__:
    from .preflight_run_state import run_state_preflight
else:
    from preflight_run_state import run_state_preflight

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-elo-ablation-run"
BUDGET_COMPLETION = "budget_completion"
TRANSIENT_CRASH = "transient_crash"
FATAL_ORCHESTRATOR_EXIT = "fatal_orchestrator_exit"
RUNNER_ERROR = "runner_error"


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


def _coordinator_owner_is_live(path: Path) -> bool:
    try:
        payload = _read_json(path)
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
) -> bool:
    if process.poll() is not None:
        return False
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=10)
        return False
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    return True


def _measurement_attempts(metadata: dict[str, Any]) -> list[dict[str, object]]:
    raw_attempts = metadata.get("measurement_attempts")
    if raw_attempts is None:
        return []
    if not isinstance(raw_attempts, list) or any(
        not isinstance(attempt, dict) for attempt in raw_attempts
    ):
        raise ValueError("ablation measurement attempts are malformed")
    return [dict(attempt) for attempt in raw_attempts]


def _measurement_start(
    metadata: dict[str, Any],
    *,
    now_ns: int,
) -> tuple[int, list[dict[str, object]]]:
    raw_started = metadata.get("measurement_started_ns")
    attempts = _measurement_attempts(metadata)
    if raw_started is None:
        return now_ns, attempts
    if type(raw_started) is not int or raw_started <= 0:
        raise ValueError("ablation measurement start is invalid")
    status = metadata.get("measurement_status")
    outcome = metadata.get("measurement_outcome")
    if status == "complete" or outcome == BUDGET_COMPLETION:
        raise RuntimeError("ablation measurement has already completed")
    if outcome == FATAL_ORCHESTRATOR_EXIT or status == "failed":
        raise RuntimeError("ablation measurement ended with a fatal failure")
    if status not in {"running", "retryable"} or outcome not in {
        None,
        TRANSIENT_CRASH,
    }:
        raise RuntimeError("ablation measurement has already started")
    if attempts and attempts[-1].get("outcome") == "running":
        attempts[-1].update(
            {
                "outcome": TRANSIENT_CRASH,
                "status": "retryable",
                "stop_reason": "runner_restart_recovery",
                "failure": "runner exited before finalizing this attempt",
                "recovered_ns": now_ns,
            }
        )
    return raw_started, attempts


def _finish_attempt(
    *,
    metadata_path: Path,
    metadata: dict[str, Any],
    attempts: list[dict[str, object]],
    attempt: dict[str, object],
    measurement_started_ns: int,
    stopped_ns: int,
    stop_reason: str,
    exit_code: int | None,
    evaluator_rows: int,
    outcome: str,
    status: str,
    failure: str | None,
) -> dict[str, object]:
    attempt.update(
        {
            "stopped_ns": stopped_ns,
            "stop_reason": stop_reason,
            "exit_code": exit_code,
            "evaluator_rows": evaluator_rows,
            "outcome": outcome,
            "status": status,
            "failure": failure,
        }
    )
    metadata.update(
        {
            "measurement_started_ns": measurement_started_ns,
            "measurement_stopped_ns": stopped_ns,
            "measurement_stop_reason": stop_reason,
            "measurement_exit_code": exit_code,
            "measurement_evaluator_rows": evaluator_rows,
            "measurement_status": status,
            "measurement_outcome": outcome,
            "measurement_failure": failure,
            "measurement_attempt_count": len(attempts),
            "measurement_attempts": attempts,
        }
    )
    atomic_json(metadata_path, metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": status,
        "outcome": outcome,
        "treatment": metadata.get("treatment"),
        "run_root": str(metadata_path.parent),
        "started_ns": measurement_started_ns,
        "attempt_started_ns": attempt["started_ns"],
        "stopped_ns": stopped_ns,
        "wall_seconds": (stopped_ns - measurement_started_ns) / 1_000_000_000.0,
        "stop_reason": stop_reason,
        "exit_code": exit_code,
        "evaluator_rows": evaluator_rows,
        "attempt": len(attempts),
        "failure": failure,
    }


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
    coordinator_lock = root / "coordinator.lock"
    if coordinator_lock.exists() and _coordinator_owner_is_live(coordinator_lock):
        raise RuntimeError("ablation run root has a live coordinator lock")
    metadata = _read_json(metadata_path)
    if metadata.get("report") != "startrain-elo-ablation-branch":
        raise ValueError("unsupported ablation metadata")
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
    state_preflight = run_state_preflight(root, profile, apply=True)
    metadata["state_preflight"] = state_preflight
    atomic_json(metadata_path, metadata)
    attempt_started_ns = time.time_ns()
    started_ns, attempts = _measurement_start(
        metadata,
        now_ns=attempt_started_ns,
    )
    executable = _resolve_executable(orchestrator)
    attempt: dict[str, object] = {
        "attempt": len(attempts) + 1,
        "started_ns": attempt_started_ns,
        "stopped_ns": None,
        "stop_reason": None,
        "exit_code": None,
        "evaluator_rows": None,
        "outcome": "running",
        "status": "running",
        "failure": None,
        "orchestrator_started": False,
    }
    attempts.append(attempt)
    metadata.update(
        {
            "measurement_started_ns": started_ns,
            "measurement_stopped_ns": None,
            "measurement_stop_reason": None,
            "measurement_exit_code": None,
            "measurement_evaluator_rows": 0,
            "measurement_status": "running",
            "measurement_outcome": None,
            "measurement_failure": None,
            "measurement_attempt_count": len(attempts),
            "measurement_attempts": attempts,
        }
    )
    atomic_json(metadata_path, metadata)

    tracker = EvaluatorRows(root / experiment.orchestration.directories.metrics)
    try:
        rows = tracker.refresh()
    except (json.JSONDecodeError, OSError, ValueError) as error:
        stopped_ns = time.time_ns()
        return _finish_attempt(
            metadata_path=metadata_path,
            metadata=metadata,
            attempts=attempts,
            attempt=attempt,
            measurement_started_ns=started_ns,
            stopped_ns=stopped_ns,
            stop_reason="runner_error",
            exit_code=None,
            evaluator_rows=tracker.rows,
            outcome=RUNNER_ERROR,
            status="failed",
            failure=f"{type(error).__name__}: {error}",
        )
    elapsed = (time.time_ns() - started_ns) / 1_000_000_000.0
    if rows >= leaf_budget or elapsed >= wall_budget_seconds:
        stop_reason = "leaf_budget" if rows >= leaf_budget else "wall_budget"
        stopped_ns = time.time_ns()
        return _finish_attempt(
            metadata_path=metadata_path,
            metadata=metadata,
            attempts=attempts,
            attempt=attempt,
            measurement_started_ns=started_ns,
            stopped_ns=stopped_ns,
            stop_reason=stop_reason,
            exit_code=None,
            evaluator_rows=rows,
            outcome=BUDGET_COMPLETION,
            status="complete",
            failure=None,
        )

    latch = SignalLatch()
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    latch.install()
    process: subprocess.Popen[bytes] | None = None
    stop_reason = "process_exit"
    runner_failure: str | None = None
    termination_requested = False
    try:
        try:
            process = subprocess.Popen(
                [executable, "--config", str(profile)],
                start_new_session=True,
            )
            attempt["orchestrator_started"] = True
            atomic_json(metadata_path, metadata)
        except OSError as error:
            stop_reason = "spawn_error"
            runner_failure = f"{type(error).__name__}: {error}"
        while process is not None and process.poll() is None:
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
    except (json.JSONDecodeError, OSError, ValueError) as error:
        stop_reason = "runner_error"
        runner_failure = f"{type(error).__name__}: {error}"
    finally:
        try:
            if process is not None:
                grace = (
                    experiment.orchestration.shutdown.terminate_grace_seconds
                    + experiment.orchestration.shutdown.kill_grace_seconds
                )
                termination_requested = _terminate(process, grace_seconds=grace)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
    try:
        rows = tracker.refresh()
    except (json.JSONDecodeError, OSError, ValueError) as error:
        stop_reason = "runner_error"
        runner_failure = f"{type(error).__name__}: {error}"
        rows = tracker.rows
    exit_code = process.returncode if process is not None else None
    stopped_ns = time.time_ns()
    if stop_reason in {"leaf_budget", "wall_budget"} and termination_requested:
        outcome = BUDGET_COMPLETION
        clean_shutdown = exit_code in (0, -signal.SIGTERM)
        status = "complete" if clean_shutdown else "failed"
        failure = (
            None
            if clean_shutdown
            else f"orchestrator did not stop cleanly at budget (exit={exit_code!r})"
        )
    elif not termination_requested and (
        exit_code == TRANSIENT_WORKER_EXIT_CODE
        or (exit_code is not None and exit_code < 0)
    ):
        outcome = TRANSIENT_CRASH
        status = "retryable"
        failure = (
            f"orchestrator reported a transient failure (exit={exit_code})"
            if exit_code == TRANSIENT_WORKER_EXIT_CODE
            else f"orchestrator was terminated by signal {-exit_code}"
        )
    elif not termination_requested and exit_code == FATAL_WORKER_EXIT_CODE:
        outcome = FATAL_ORCHESTRATOR_EXIT
        status = "failed"
        failure = f"orchestrator reported a fatal failure (exit={exit_code})"
    elif not termination_requested and exit_code is not None:
        outcome = FATAL_ORCHESTRATOR_EXIT
        status = "failed"
        failure = f"orchestrator exited before budget with code {exit_code!r}"
    elif stop_reason.startswith("signal_"):
        outcome = TRANSIENT_CRASH
        status = "retryable"
        failure = f"runner interrupted by {stop_reason}"
    elif stop_reason == "spawn_error":
        outcome = TRANSIENT_CRASH
        status = "retryable"
        failure = runner_failure
    elif stop_reason == "runner_error":
        outcome = RUNNER_ERROR
        status = "failed"
        failure = runner_failure
    else:
        outcome = FATAL_ORCHESTRATOR_EXIT
        status = "failed"
        failure = f"orchestrator exited before budget with code {exit_code!r}"
    return _finish_attempt(
        metadata_path=metadata_path,
        metadata=metadata,
        attempts=attempts,
        attempt=attempt,
        measurement_started_ns=started_ns,
        stopped_ns=stopped_ns,
        stop_reason=stop_reason,
        exit_code=exit_code,
        evaluator_rows=rows,
        outcome=outcome,
        status=status,
        failure=failure,
    )


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
    if report["status"] == "complete":
        return 0
    return 75 if report["status"] == "retryable" else 3


if __name__ == "__main__":
    raise SystemExit(main())
