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
    from .replay_manifest_backup import restore_if_corrupt
else:
    from preflight_run_state import run_state_preflight
    from replay_manifest_backup import restore_if_corrupt

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


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate(
    process: subprocess.Popen[bytes],
    *,
    terminate_grace_seconds: float,
    kill_grace_seconds: float,
) -> dict[str, object]:
    """Stop the coordinator first and its isolated process group only on timeout."""

    started_ns = time.time_ns()
    initial_exit_code = process.poll()
    term_sent = False
    kill_sent = False
    escalated = False
    errors: list[str] = []
    process_group_id = process.pid
    terminate_deadline = time.monotonic() + terminate_grace_seconds
    if initial_exit_code is None:
        try:
            os.kill(process.pid, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(f"SIGTERM failed: {type(error).__name__}: {error}")
        try:
            process.wait(timeout=max(0.0, terminate_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        except OSError as error:
            errors.append(f"wait after SIGTERM failed: {type(error).__name__}: {error}")

    exit_code = process.poll()
    group_exists = _process_group_exists(process_group_id)
    while (
        exit_code is not None and group_exists and time.monotonic() < terminate_deadline
    ):
        time.sleep(min(0.05, max(0.0, terminate_deadline - time.monotonic())))
        group_exists = _process_group_exists(process.pid)
    if exit_code is None or group_exists:
        escalated = True
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            kill_sent = True
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(f"SIGKILL failed: {type(error).__name__}: {error}")
        kill_deadline = time.monotonic() + kill_grace_seconds
        if exit_code is None:
            try:
                process.wait(timeout=max(0.0, kill_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                errors.append("coordinator did not exit before kill grace expired")
            except OSError as error:
                errors.append(
                    f"wait after SIGKILL failed: {type(error).__name__}: {error}"
                )
        group_exists = _process_group_exists(process_group_id)
        while group_exists and time.monotonic() < kill_deadline:
            time.sleep(min(0.05, max(0.0, kill_deadline - time.monotonic())))
            group_exists = _process_group_exists(process.pid)
        if group_exists:
            errors.append("process group did not exit before kill grace expired")

    exit_code = process.poll()
    completed_ns = time.time_ns()
    process_group_released = not group_exists
    resource_released_ns = (
        completed_ns if exit_code is not None and process_group_released else None
    )
    if escalated:
        status = "forced" if exit_code is not None else "failed"
        clean = False
    elif initial_exit_code is not None:
        status = "already_exited"
        clean = True
    elif term_sent and exit_code in (0, -signal.SIGTERM) and not errors:
        status = "graceful"
        clean = True
    elif exit_code is not None:
        status = "unexpected_exit"
        clean = False
    else:
        status = "failed"
        clean = False
    warning = None
    if not clean:
        detail = "; ".join(errors)
        warning = f"orchestrator teardown was {status} (exit={exit_code!r})" + (
            f": {detail}" if detail else ""
        )
    return {
        "schema_version": 1,
        "status": status,
        "clean": clean,
        "requested": initial_exit_code is None,
        "coordinator_pid": process.pid,
        "process_group_id": process_group_id,
        "terminate_grace_seconds": terminate_grace_seconds,
        "kill_grace_seconds": kill_grace_seconds,
        "term_target": "coordinator_pid",
        "term_sent": term_sent,
        "kill_target": "process_group",
        "kill_sent": kill_sent,
        "escalated": escalated,
        "process_group_released": process_group_released,
        "initial_exit_code": initial_exit_code,
        "exit_code": exit_code,
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "resource_released_ns": resource_released_ns,
        "warning": warning,
        "errors": errors,
    }


def _inactive_teardown(*, cutoff_ns: int, status: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "clean": True,
        "requested": False,
        "coordinator_pid": None,
        "term_target": "coordinator_pid",
        "term_sent": False,
        "kill_target": "process_group",
        "kill_sent": False,
        "escalated": False,
        "process_group_released": True,
        "initial_exit_code": None,
        "exit_code": None,
        "started_ns": cutoff_ns,
        "completed_ns": cutoff_ns,
        "resource_released_ns": cutoff_ns,
        "warning": None,
        "errors": [],
    }


def _terminal_failure_context(root: Path, cutoff_ns: int) -> dict[str, object] | None:
    candidates = (
        (root / "status" / "fatal.json", None),
        (root / "status" / "coordinator.json", "failure"),
    )
    for path, nested_key in candidates:
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {
                "artifact": str(path),
                "phase": "unknown",
                "timestamp_ns": None,
                "failure": None,
                "parse_error": True,
            }
        failure = payload.get(nested_key) if nested_key is not None else payload
        if not isinstance(failure, dict):
            continue
        timestamp = failure.get("timestamp_ns")
        occurred_ns = (
            timestamp
            if isinstance(timestamp, int)
            and not isinstance(timestamp, bool)
            and timestamp > 0
            else None
        )
        return {
            "artifact": str(path),
            "phase": (
                "unknown"
                if occurred_ns is None
                else ("pre_cutoff" if occurred_ns <= cutoff_ns else "post_cutoff")
            ),
            "timestamp_ns": occurred_ns,
            "failure": failure,
            "parse_error": False,
        }
    return None


def _terminal_failure_domain(root: Path, cutoff_ns: int) -> str:
    context = _terminal_failure_context(root, cutoff_ns)
    failure = context.get("failure") if context is not None else None
    terminal_reason = (
        str(failure.get("terminal_reason", "")).casefold()
        if isinstance(failure, dict)
        else ""
    )
    if any(
        marker in terminal_reason for marker in ("hardware", "gpu_health", "gpu-unsafe")
    ):
        return "host"
    return "run"


def _post_cutoff_integrity(
    root: Path,
    profile: Path,
    *,
    cutoff_ns: int,
) -> dict[str, object]:
    checked_ns = time.time_ns()
    terminal_failure = _terminal_failure_context(root, cutoff_ns)
    try:
        state = run_state_preflight(root, profile, apply=False)
    except Exception as error:
        return {
            "schema_version": 1,
            "status": "failed",
            "valid": False,
            "checked_ns": checked_ns,
            "failure": f"{type(error).__name__}: {error}",
            "state_preflight": None,
            "terminal_failure": terminal_failure,
        }
    terminal_phase = (
        terminal_failure.get("phase") if terminal_failure is not None else None
    )
    valid = state.get("status") == "ok" and terminal_phase in {None, "post_cutoff"}
    return {
        "schema_version": 1,
        "status": "valid" if valid else "failed",
        "valid": valid,
        "checked_ns": checked_ns,
        "failure": (
            None
            if valid
            else (
                f"terminal failure phase is {terminal_phase!r}"
                if terminal_phase not in {None, "post_cutoff"}
                else f"state preflight returned status {state.get('status')!r}"
            )
        ),
        "state_preflight": state,
        "terminal_failure": terminal_failure,
    }


def _measurement_attempts(metadata: dict[str, Any]) -> list[dict[str, object]]:
    raw_attempts = metadata.get("measurement_attempts")
    if raw_attempts is None:
        return []
    if not isinstance(raw_attempts, list) or any(
        not isinstance(attempt, dict) for attempt in raw_attempts
    ):
        raise ValueError("ablation measurement attempts are malformed")
    return [dict(attempt) for attempt in raw_attempts]


def _validate_measurement_resumable(metadata: dict[str, Any]) -> None:
    raw_started = metadata.get("measurement_started_ns")
    _measurement_attempts(metadata)
    if raw_started is None:
        return
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


def _measurement_start(
    metadata: dict[str, Any],
    *,
    now_ns: int,
) -> tuple[int, list[dict[str, object]]]:
    _validate_measurement_resumable(metadata)
    raw_started = metadata.get("measurement_started_ns")
    attempts = _measurement_attempts(metadata)
    if raw_started is None:
        return now_ns, attempts
    assert isinstance(raw_started, int)
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


def _append_replay_restore_evidence(
    metadata: dict[str, Any],
    *,
    run_root: Path,
    restored_from: Path | None,
) -> dict[str, object]:
    previous = metadata.get("replay_restore")
    history: list[dict[str, object]] = []
    if previous is not None:
        if not isinstance(previous, dict):
            raise ValueError("replay restore evidence is malformed")
        raw_history = previous.get("attempts")
        if isinstance(raw_history, list):
            if any(not isinstance(item, dict) for item in raw_history):
                raise ValueError("replay restore attempt history is malformed")
            history = [dict(item) for item in raw_history]
        elif isinstance(previous.get("status"), str):
            history = [dict(previous)]
        else:
            raise ValueError("replay restore evidence is malformed")
    if restored_from is not None:
        status = "restored"
    elif (run_root / "replay" / "manifest.sqlite3").is_file():
        status = "verified"
    else:
        status = "uninitialized"
    record: dict[str, object] = {
        "schema_version": 1,
        "attempt": len(history) + 1,
        "checked_ns": time.time_ns(),
        "status": status,
        "restored_from": str(restored_from) if restored_from is not None else None,
    }
    history.append(record)
    evidence: dict[str, object] = {
        "schema_version": 1,
        "latest": record,
        "attempts": history,
    }
    metadata["replay_restore"] = evidence
    return evidence


def _finish_attempt(
    *,
    metadata_path: Path,
    metadata: dict[str, Any],
    attempts: list[dict[str, object]],
    attempt: dict[str, object],
    measurement_started_ns: int,
    measurement_cutoff_ns: int,
    resource_released_ns: int | None,
    teardown_status: dict[str, object],
    stop_reason: str,
    exit_code: int | None,
    evaluator_rows: int,
    outcome: str,
    status: str,
    completion_status: str,
    failure: str | None,
    warnings: list[str] | None = None,
    integrity: dict[str, object] | None = None,
    failure_domain: str | None = None,
    failure_phase: str | None = None,
) -> dict[str, object]:
    lifecycle_warnings = list(warnings or [])
    teardown_state = (
        "clean"
        if teardown_status.get("clean") is True
        else str(teardown_status.get("status", "failed"))
    )
    integrity_status = integrity.get("status") if integrity is not None else None
    attempt.update(
        {
            "stopped_ns": measurement_cutoff_ns,
            "measurement_cutoff_ns": measurement_cutoff_ns,
            "resource_released_ns": resource_released_ns,
            "teardown_status": teardown_state,
            "teardown": teardown_status,
            "integrity_status": integrity_status,
            "integrity": integrity,
            "stop_reason": stop_reason,
            "exit_code": exit_code,
            "resource_exit_code": teardown_status.get("exit_code"),
            "evaluator_rows": evaluator_rows,
            "outcome": outcome,
            "status": status,
            "completion_status": completion_status,
            "failure": failure,
            "failure_domain": failure_domain,
            "failure_phase": failure_phase,
            "warnings": lifecycle_warnings,
        }
    )
    metadata.update(
        {
            "measurement_started_ns": measurement_started_ns,
            # Legacy readers treat measurement_stopped_ns as the evidence
            # boundary. Keep it as an alias for the explicit cutoff.
            "measurement_stopped_ns": measurement_cutoff_ns,
            "measurement_cutoff_ns": measurement_cutoff_ns,
            "resource_released_ns": resource_released_ns,
            "measurement_teardown_status": teardown_state,
            "measurement_teardown": teardown_status,
            "teardown_status": teardown_state,
            "teardown": teardown_status,
            "integrity_status": integrity_status,
            "integrity": integrity,
            "measurement_stop_reason": stop_reason,
            "measurement_exit_code": exit_code,
            "measurement_resource_exit_code": teardown_status.get("exit_code"),
            "measurement_evaluator_rows": evaluator_rows,
            "measurement_status": status,
            "measurement_completion_status": completion_status,
            "measurement_outcome": outcome,
            "measurement_failure": failure,
            "failure_domain": failure_domain,
            "failure_phase": failure_phase,
            "measurement_warnings": lifecycle_warnings,
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
        "stopped_ns": measurement_cutoff_ns,
        "measurement_cutoff_ns": measurement_cutoff_ns,
        "resource_released_ns": resource_released_ns,
        "wall_seconds": (measurement_cutoff_ns - measurement_started_ns)
        / 1_000_000_000.0,
        "resource_release_seconds": (
            None
            if resource_released_ns is None
            else (resource_released_ns - measurement_cutoff_ns) / 1_000_000_000.0
        ),
        "stop_reason": stop_reason,
        "exit_code": exit_code,
        "resource_exit_code": teardown_status.get("exit_code"),
        "teardown_status": teardown_state,
        "teardown": teardown_status,
        "integrity_status": integrity_status,
        "integrity": integrity,
        "completion_status": completion_status,
        "evaluator_rows": evaluator_rows,
        "attempt": len(attempts),
        "failure": failure,
        "failure_domain": failure_domain,
        "failure_phase": failure_phase,
        "warnings": lifecycle_warnings,
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
    _validate_measurement_resumable(metadata)
    restored_from = restore_if_corrupt(root)
    _append_replay_restore_evidence(
        metadata,
        run_root=root,
        restored_from=restored_from,
    )
    atomic_json(metadata_path, metadata)
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
        "measurement_cutoff_ns": None,
        "resource_released_ns": None,
        "teardown_status": None,
        "teardown": None,
        "integrity_status": None,
        "integrity": None,
        "stop_reason": None,
        "exit_code": None,
        "resource_exit_code": None,
        "evaluator_rows": None,
        "outcome": "running",
        "status": "running",
        "completion_status": "running",
        "failure": None,
        "failure_domain": None,
        "failure_phase": None,
        "warnings": [],
        "orchestrator_started": False,
    }
    attempts.append(attempt)
    metadata.update(
        {
            "measurement_started_ns": started_ns,
            "measurement_stopped_ns": None,
            "measurement_cutoff_ns": None,
            "resource_released_ns": None,
            "measurement_teardown_status": None,
            "measurement_teardown": None,
            "teardown_status": None,
            "teardown": None,
            "integrity_status": None,
            "integrity": None,
            "measurement_stop_reason": None,
            "measurement_exit_code": None,
            "measurement_resource_exit_code": None,
            "measurement_evaluator_rows": 0,
            "measurement_status": "running",
            "measurement_completion_status": "running",
            "measurement_outcome": None,
            "measurement_failure": None,
            "failure_domain": None,
            "failure_phase": None,
            "measurement_warnings": [],
            "measurement_attempt_count": len(attempts),
            "measurement_attempts": attempts,
        }
    )
    atomic_json(metadata_path, metadata)

    tracker = EvaluatorRows(root / experiment.orchestration.directories.metrics)
    try:
        rows = tracker.refresh()
    except (json.JSONDecodeError, OSError, ValueError) as error:
        cutoff_ns = time.time_ns()
        teardown_status = _inactive_teardown(
            cutoff_ns=cutoff_ns,
            status="not_started",
        )
        return _finish_attempt(
            metadata_path=metadata_path,
            metadata=metadata,
            attempts=attempts,
            attempt=attempt,
            measurement_started_ns=started_ns,
            measurement_cutoff_ns=cutoff_ns,
            resource_released_ns=cutoff_ns,
            teardown_status=teardown_status,
            stop_reason="runner_error",
            exit_code=None,
            evaluator_rows=tracker.rows,
            outcome=RUNNER_ERROR,
            status="failed",
            completion_status="failed",
            failure=f"{type(error).__name__}: {error}",
        )
    elapsed = (time.time_ns() - started_ns) / 1_000_000_000.0
    if rows >= leaf_budget or elapsed >= wall_budget_seconds:
        stop_reason = "leaf_budget" if rows >= leaf_budget else "wall_budget"
        cutoff_ns = time.time_ns()
        teardown_status = _inactive_teardown(
            cutoff_ns=cutoff_ns,
            status="not_started",
        )
        return _finish_attempt(
            metadata_path=metadata_path,
            metadata=metadata,
            attempts=attempts,
            attempt=attempt,
            measurement_started_ns=started_ns,
            measurement_cutoff_ns=cutoff_ns,
            resource_released_ns=cutoff_ns,
            teardown_status=teardown_status,
            stop_reason=stop_reason,
            exit_code=None,
            evaluator_rows=rows,
            outcome=BUDGET_COMPLETION,
            status="complete",
            completion_status="complete",
            failure=None,
        )

    latch = SignalLatch()
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    latch.install()
    process: subprocess.Popen[bytes] | None = None
    stop_reason = "process_exit"
    runner_failure: str | None = None
    measurement_cutoff_ns: int | None = None
    measurement_exit_code: int | None = None
    teardown_status: dict[str, object]
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
                measurement_cutoff_ns = time.time_ns()
                measurement_exit_code = process.poll()
                break
            if rows >= leaf_budget:
                stop_reason = "leaf_budget"
                measurement_cutoff_ns = time.time_ns()
                measurement_exit_code = process.poll()
                if measurement_exit_code is not None:
                    stop_reason = "process_exit"
                break
            if elapsed >= wall_budget_seconds:
                stop_reason = "wall_budget"
                measurement_cutoff_ns = time.time_ns()
                measurement_exit_code = process.poll()
                if measurement_exit_code is not None:
                    stop_reason = "process_exit"
                break
            time.sleep(poll_seconds)
        if measurement_cutoff_ns is None:
            try:
                rows = tracker.refresh()
            except (json.JSONDecodeError, OSError, ValueError) as error:
                stop_reason = "runner_error"
                runner_failure = f"{type(error).__name__}: {error}"
                rows = tracker.rows
            measurement_cutoff_ns = time.time_ns()
            measurement_exit_code = process.poll() if process is not None else None
    except (json.JSONDecodeError, OSError, ValueError) as error:
        stop_reason = "runner_error"
        runner_failure = f"{type(error).__name__}: {error}"
        rows = tracker.rows
        measurement_cutoff_ns = time.time_ns()
        measurement_exit_code = process.poll() if process is not None else None
    finally:
        try:
            if measurement_cutoff_ns is None:
                measurement_cutoff_ns = time.time_ns()
            if process is not None:
                teardown_status = _terminate(
                    process,
                    terminate_grace_seconds=(
                        experiment.orchestration.shutdown.terminate_grace_seconds
                        + experiment.orchestration.hardware_health.probe_timeout_seconds
                        + 2 * experiment.orchestration.shutdown.monitor_interval_seconds
                    ),
                    kill_grace_seconds=(
                        experiment.orchestration.shutdown.kill_grace_seconds
                    ),
                )
            else:
                teardown_status = _inactive_teardown(
                    cutoff_ns=measurement_cutoff_ns,
                    status="not_started",
                )
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
    assert measurement_cutoff_ns is not None
    if (
        stop_reason in {"leaf_budget", "wall_budget"}
        and measurement_exit_code is None
        and teardown_status.get("exit_code") not in (0, -signal.SIGTERM)
        and teardown_status.get("clean") is True
    ):
        teardown_status["status"] = "unexpected_exit"
        teardown_status["clean"] = False
        teardown_status["warning"] = (
            "orchestrator teardown exited unexpectedly after measurement cutoff "
            f"(exit={teardown_status.get('exit_code')!r})"
        )
    raw_resource_released_ns = teardown_status.get("resource_released_ns")
    resource_released_ns = (
        raw_resource_released_ns
        if isinstance(raw_resource_released_ns, int)
        and not isinstance(raw_resource_released_ns, bool)
        else None
    )
    if resource_released_ns is None:
        teardown_status["status"] = "resources_not_released"
        teardown_status["clean"] = False
        release_warning = "orchestrator process group release was not confirmed"
        previous_warning = teardown_status.get("warning")
        teardown_status["warning"] = (
            f"{previous_warning}; {release_warning}"
            if isinstance(previous_warning, str) and previous_warning
            else release_warning
        )
    teardown_warning = teardown_status.get("warning")
    warnings = [teardown_warning] if isinstance(teardown_warning, str) else []
    integrity: dict[str, object] | None = None
    failure_domain: str | None = None
    failure_phase: str | None = None
    if stop_reason in {"leaf_budget", "wall_budget"} and measurement_exit_code is None:
        outcome = BUDGET_COMPLETION
        if teardown_status.get("clean") is True and resource_released_ns is not None:
            status = "complete"
            completion_status = "complete"
            failure = None
        else:
            integrity = _post_cutoff_integrity(
                root,
                profile,
                cutoff_ns=measurement_cutoff_ns,
            )
            if integrity.get("valid") is True and resource_released_ns is not None:
                status = "complete"
                completion_status = "complete_with_warning"
                failure = None
                failure_domain = "orchestrator_teardown"
                failure_phase = "post_cutoff"
            else:
                status = "failed"
                completion_status = "failed"
                terminal = integrity.get("terminal_failure")
                terminal_phase = (
                    terminal.get("phase") if isinstance(terminal, dict) else None
                )
                failure_domain = (
                    "orchestrator"
                    if terminal_phase in {"pre_cutoff", "unknown"}
                    else (
                        "process_cleanup"
                        if resource_released_ns is None
                        else "state_integrity"
                    )
                )
                failure_phase = (
                    str(terminal_phase)
                    if terminal_phase in {"pre_cutoff", "unknown"}
                    else "post_cutoff"
                )
                failure = (
                    "post-cutoff teardown anomaly failed state integrity: "
                    f"{integrity.get('failure')}"
                )
                warnings.append(failure)
    elif measurement_exit_code == TRANSIENT_WORKER_EXIT_CODE or (
        measurement_exit_code is not None and measurement_exit_code < 0
    ):
        outcome = TRANSIENT_CRASH
        status = "retryable"
        completion_status = "retryable"
        failure_domain = "orchestrator"
        failure_phase = "measurement"
        failure = (
            f"orchestrator reported a transient failure (exit={measurement_exit_code})"
            if measurement_exit_code == TRANSIENT_WORKER_EXIT_CODE
            else f"orchestrator was terminated by signal {-measurement_exit_code}"
        )
    elif measurement_exit_code == FATAL_WORKER_EXIT_CODE:
        outcome = FATAL_ORCHESTRATOR_EXIT
        status = "failed"
        completion_status = "failed"
        failure_domain = _terminal_failure_domain(root, measurement_cutoff_ns)
        failure_phase = "measurement"
        failure = (
            f"orchestrator reported a fatal failure (exit={measurement_exit_code})"
        )
    elif measurement_exit_code is not None:
        outcome = FATAL_ORCHESTRATOR_EXIT
        status = "failed"
        completion_status = "failed"
        failure_domain = _terminal_failure_domain(root, measurement_cutoff_ns)
        failure_phase = "measurement"
        failure = (
            f"orchestrator exited before budget with code {measurement_exit_code!r}"
        )
    elif stop_reason.startswith("signal_"):
        outcome = TRANSIENT_CRASH
        status = "retryable"
        completion_status = "retryable"
        failure_domain = "runner_signal"
        failure_phase = "measurement"
        failure = f"runner interrupted by {stop_reason}"
    elif stop_reason == "spawn_error":
        outcome = TRANSIENT_CRASH
        status = "retryable"
        completion_status = "retryable"
        failure_domain = "runner_spawn"
        failure_phase = "measurement"
        failure = runner_failure
    elif stop_reason == "runner_error":
        outcome = RUNNER_ERROR
        status = "failed"
        completion_status = "failed"
        failure_domain = "runner"
        failure_phase = "measurement"
        failure = runner_failure
    else:
        outcome = FATAL_ORCHESTRATOR_EXIT
        status = "failed"
        completion_status = "failed"
        failure_domain = "orchestrator"
        failure_phase = "measurement"
        failure = (
            f"orchestrator exited before budget with code {measurement_exit_code!r}"
        )
    return _finish_attempt(
        metadata_path=metadata_path,
        metadata=metadata,
        attempts=attempts,
        attempt=attempt,
        measurement_started_ns=started_ns,
        measurement_cutoff_ns=measurement_cutoff_ns,
        resource_released_ns=resource_released_ns,
        teardown_status=teardown_status,
        stop_reason=stop_reason,
        exit_code=measurement_exit_code,
        evaluator_rows=rows,
        outcome=outcome,
        status=status,
        completion_status=completion_status,
        failure=failure,
        warnings=warnings,
        integrity=integrity,
        failure_domain=failure_domain,
        failure_phase=failure_phase,
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
