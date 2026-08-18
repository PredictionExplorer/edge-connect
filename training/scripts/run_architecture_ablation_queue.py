#!/usr/bin/env python3
"""Run scratch architecture arms, cross-play them, and publish diagnostic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from startrain.checkpoint import load_model_manifest
from startrain.config import load_config
from startrain.runtime import SignalLatch, atomic_json, load_run_identity

if __package__:
    from .compare_architecture_ablation import (
        SUITE_FORMAT,
        SUITE_SCHEMA_VERSION,
        build_architecture_ablation_evidence,
    )
    from .preflight_run_state import run_state_preflight
    from .run_elo_ablation_queue import (
        exclusive_execution_lock,
        exclusive_queue_lock,
    )
else:
    from compare_architecture_ablation import (
        SUITE_FORMAT,
        SUITE_SCHEMA_VERSION,
        build_architecture_ablation_evidence,
    )
    from preflight_run_state import run_state_preflight
    from run_elo_ablation_queue import exclusive_execution_lock, exclusive_queue_lock

SCHEMA_VERSION = 1
QUEUE_REPORT = "startrain-architecture-ablation-queue"
ARM_REPORT = "startrain-scratch-architecture-run"


class ArchitectureQueueError(RuntimeError):
    """Architecture training or evidence failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchitectureQueueError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArchitectureQueueError(f"{name} must be an object: {path}")
    return payload


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArchitectureQueueError(f"{name} must be a non-empty string")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ArchitectureQueueError(f"{name} must be positive")
    return float(value)


def _artifact(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ArchitectureQueueError(f"artifact is missing or unsafe: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _immutable_manifest(pointer: Path) -> Path:
    manifest = load_model_manifest(pointer)
    artifact = (manifest.artifact_manifest or manifest.path).resolve()
    if not artifact.is_file() or artifact.is_symlink():
        raise ArchitectureQueueError("model manifest artifact is missing or unsafe")
    return artifact


def _resolve_executable(value: str) -> str:
    candidate = Path(value)
    if candidate.parent != Path("."):
        resolved = candidate.expanduser().resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ArchitectureQueueError(f"executable is unavailable: {resolved}")
        return str(resolved)
    resolved = shutil.which(value)
    if resolved is None:
        raise ArchitectureQueueError(f"executable is not on PATH: {value}")
    return resolved


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
) -> dict[str, object]:
    term_sent_ns = time.time_ns()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    forced = False
    try:
        exit_code = process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        forced = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        exit_code = process.wait(timeout=max(30.0, grace_seconds))
    return {
        "term_sent_ns": term_sent_ns,
        "forced": forced,
        "exit_code": exit_code,
        "resource_released_ns": time.time_ns(),
    }


def _verify_scratch_authority(
    *,
    plan_path: Path,
    plan_sha256: str,
    treatment: Mapping[str, object],
) -> tuple[Path, Path, dict[str, Any]]:
    label = _string(treatment.get("treatment"), "treatment")
    root = Path(_string(treatment.get("run_root"), f"{label} run root")).resolve()
    profile = root / "profile-architecture-scratch.yaml"
    initialization_path = root / "scratch-initialization.json"
    if not root.is_dir() or root.is_symlink():
        raise ArchitectureQueueError(f"{label} scratch root is missing or unsafe")
    initialization = _json(initialization_path, f"{label} scratch initialization")
    if (
        initialization.get("schema_version") != SCHEMA_VERSION
        or initialization.get("report")
        != "startrain-scratch-architecture-initialization"
        or initialization.get("status") != "prepared"
        or initialization.get("treatment") != label
        or initialization.get("plan") != str(plan_path)
        or initialization.get("plan_sha256") != plan_sha256
        or initialization.get("run_root") != str(root)
        or initialization.get("profile") != str(profile)
    ):
        raise ArchitectureQueueError(f"{label} scratch initialization is incompatible")
    planned_profile = Path(
        _string(treatment.get("profile"), f"{label} plan profile")
    ).resolve()
    planned_sha256 = _string(
        treatment.get("profile_sha256"),
        f"{label} profile SHA-256",
    )
    if (
        not profile.is_file()
        or profile.is_symlink()
        or _sha256(profile) != planned_sha256
        or not planned_profile.is_file()
        or _sha256(planned_profile) != planned_sha256
        or initialization.get("profile_sha256") != planned_sha256
    ):
        raise ArchitectureQueueError(f"{label} scratch profile authority changed")
    experiment = load_config(profile)
    if Path(
        experiment.orchestration.directories.root
    ).resolve() != root or experiment.orchestration.run_id != treatment.get("run_id"):
        raise ArchitectureQueueError(f"{label} scratch profile identity changed")
    return root, profile, initialization


def _complete_scratch_arm(
    *,
    label: str,
    root: Path,
    profile: Path,
    state: dict[str, Any],
    state_path: Path,
    deadline_ns: int,
    resource_released_ns: int,
) -> dict[str, object]:
    if (root / "coordinator.lock").exists():
        state["status"] = "failed"
        state["error"] = "coordinator lock remained after teardown"
        atomic_json(state_path, state)
        raise ArchitectureQueueError(str(state["error"]))
    experiment = load_config(profile)
    identity = load_run_identity(root / "run.json")
    if identity.run_id != experiment.orchestration.run_id:
        raise ArchitectureQueueError(f"{label} run identity differs from profile")
    preflight = run_state_preflight(root, profile, apply=False)
    if preflight.get("status") != "ok":
        state["status"] = "failed"
        state["error"] = "post-budget run-state preflight failed"
        state["state_preflight"] = preflight
        atomic_json(state_path, state)
        raise ArchitectureQueueError(str(state["error"]))
    champion_manifest = _immutable_manifest(root / "learner" / "champion.json")
    champion = _artifact(champion_manifest)
    state.update(
        {
            "status": "completed",
            "measurement_cutoff_ns": deadline_ns,
            "resource_released_ns": resource_released_ns,
            "state_preflight": preflight,
            "champion_manifest": champion["path"],
            "champion_manifest_sha256": champion["sha256"],
            "champion_manifest_bytes": champion["bytes"],
        }
    )
    state.pop("error", None)
    atomic_json(state_path, state)
    return state


def run_scratch_arm(
    *,
    plan_path: Path,
    plan_sha256: str,
    treatment: Mapping[str, object],
    wall_budget_seconds: float,
    orchestrator: str,
    poll_seconds: float,
) -> dict[str, object]:
    if poll_seconds <= 0:
        raise ArchitectureQueueError("poll_seconds must be positive")
    root, profile, initialization = _verify_scratch_authority(
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        treatment=treatment,
    )
    label = _string(treatment.get("treatment"), "treatment")
    state_path = root / "architecture-run.json"
    if not state_path.exists() and any(
        path.exists() or path.is_symlink()
        for path in (
            root / "run.json",
            root / "replay",
            root / "learner",
            root / "arena",
            root / "coordinator.lock",
        )
    ):
        raise ArchitectureQueueError(
            f"{label} scratch root was contaminated before first launch"
        )
    state = (
        _json(state_path, f"{label} architecture run")
        if state_path.is_file()
        else {
            "schema_version": SCHEMA_VERSION,
            "report": ARM_REPORT,
            "status": "pending",
            "treatment": label,
            "plan_sha256": plan_sha256,
            "profile_sha256": initialization["profile_sha256"],
            "measurement_started_ns": None,
            "attempts": [],
        }
    )
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("report") != ARM_REPORT
        or state.get("treatment") != label
        or state.get("plan_sha256") != plan_sha256
        or state.get("profile_sha256") != initialization["profile_sha256"]
    ):
        raise ArchitectureQueueError(f"{label} architecture run state is incompatible")
    if state.get("status") == "completed":
        manifest_path = Path(
            _string(state.get("champion_manifest"), f"{label} champion manifest")
        )
        artifact = _artifact(manifest_path)
        if artifact["sha256"] != state.get("champion_manifest_sha256") or artifact[
            "bytes"
        ] != state.get("champion_manifest_bytes"):
            raise ArchitectureQueueError(f"{label} completed champion evidence changed")
        return state

    initial = state.get("measurement_started_ns")
    if initial is None:
        initial = time.time_ns()
        state["measurement_started_ns"] = initial
    if isinstance(initial, bool) or not isinstance(initial, int) or initial <= 0:
        raise ArchitectureQueueError(f"{label} measurement start is invalid")
    deadline_ns = initial + int(wall_budget_seconds * 1_000_000_000)
    if time.time_ns() >= deadline_ns:
        return _complete_scratch_arm(
            label=label,
            root=root,
            profile=profile,
            state=state,
            state_path=state_path,
            deadline_ns=deadline_ns,
            resource_released_ns=time.time_ns(),
        )
    executable = _resolve_executable(orchestrator)
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        raise ArchitectureQueueError(f"{label} attempts are invalid")
    attempt: dict[str, object] = {
        "attempt": len(attempts) + 1,
        "started_ns": time.time_ns(),
        "orchestrator": executable,
        "exit_code": None,
        "stop_reason": None,
    }
    attempts.append(attempt)
    state["status"] = "running"
    state["wall_budget_seconds"] = wall_budget_seconds
    atomic_json(state_path, state)

    process = subprocess.Popen(
        [executable, "--config", str(profile)],
        start_new_session=True,
    )
    stop = SignalLatch()
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    stop.install()
    stop_reason = "budget"
    try:
        while time.time_ns() < deadline_ns:
            exit_code = process.poll()
            if exit_code is not None:
                stop_reason = "early_exit"
                break
            if stop.is_set():
                stop_reason = f"signal_{stop.signal_number}"
                break
            remaining = max(
                0.0,
                (deadline_ns - time.time_ns()) / 1_000_000_000,
            )
            time.sleep(min(poll_seconds, remaining))
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
    experiment = load_config(profile)
    teardown = _terminate_process_group(
        process,
        grace_seconds=experiment.orchestration.shutdown.terminate_grace_seconds,
    )
    attempt.update(
        {
            "finished_ns": time.time_ns(),
            "exit_code": teardown["exit_code"],
            "stop_reason": stop_reason,
            "teardown": teardown,
        }
    )
    if stop_reason != "budget":
        state["status"] = "retryable" if stop_reason.startswith("signal_") else "failed"
        state["error"] = (
            "architecture arm was interrupted"
            if stop_reason.startswith("signal_")
            else "orchestrator exited before the fixed wall budget"
        )
        atomic_json(state_path, state)
        if stop_reason.startswith("signal_"):
            return state
        raise ArchitectureQueueError(f"{label} orchestrator exited before budget")
    resource_released_ns = teardown["resource_released_ns"]
    assert isinstance(resource_released_ns, int)
    return _complete_scratch_arm(
        label=label,
        root=root,
        profile=profile,
        state=state,
        state_path=state_path,
        deadline_ns=deadline_ns,
        resource_released_ns=resource_released_ns,
    )


def _run_arena(
    *,
    executable: str,
    profile: Path,
    candidate: Path,
    baseline: Path,
    output: Path,
    device: str,
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"architecture arena output already exists: {output}")
    completed = subprocess.run(
        [
            executable,
            "--config",
            str(profile),
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--baseline-kind",
            "checkpoint",
            "--evaluation-mode",
            "architecture",
            "--device",
            device,
            "--output",
            str(output),
        ],
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        raise ArchitectureQueueError(
            f"architecture arena failed with exit {completed.returncode}"
        )


def _write_immutable_json(directory: Path, prefix: str, payload: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    destination = directory / f"{prefix}-sha256-{digest}.json"
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != data:
            raise ArchitectureQueueError("immutable JSON artifact conflicts")
        return destination
    temporary = directory / f".{destination.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def architecture_suite_document(
    *,
    suite_id: str,
    control_manifest: Path,
    treatment_manifest: Path,
    baseline_manifest: Path,
    control_vs_baseline: Path,
    treatment_vs_baseline: Path,
    treatment_vs_control: Path,
) -> dict[str, object]:
    """Build exactly the strict suite schema consumed by the evidence builder."""

    return {
        "format": SUITE_FORMAT,
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_id": suite_id,
        "models": {
            "control": _artifact(control_manifest),
            "treatment": _artifact(treatment_manifest),
            "baseline": _artifact(baseline_manifest),
        },
        "arenas": {
            "control_vs_baseline": _artifact(control_vs_baseline),
            "treatment_vs_baseline": _artifact(treatment_vs_baseline),
            "treatment_vs_control": _artifact(treatment_vs_control),
        },
    }


ArmRunner = Callable[..., dict[str, object]]
ArenaRunner = Callable[..., None]


def _run_architecture_queue_locked(
    *,
    plan_path: Path,
    state_path: Path,
    evidence_directory: Path,
    orchestrator: str,
    arena_executable: str,
    device: str,
    poll_seconds: float,
    arm_runner: ArmRunner = run_scratch_arm,
    arena_runner: ArenaRunner = _run_arena,
) -> dict[str, object]:
    plan_file = plan_path.expanduser().resolve()
    plan = _json(plan_file, "architecture plan")
    plan_sha256 = _sha256(plan_file)
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("report") != "startrain-elo-ablation-plan"
        or plan.get("initialization") != "scratch"
    ):
        raise ArchitectureQueueError("plan is not a scratch architecture plan")
    raw_treatments = plan.get("treatments")
    if not isinstance(raw_treatments, list) or len(raw_treatments) < 2:
        raise ArchitectureQueueError(
            "architecture plan requires one control and at least one treatment"
        )
    treatments = [
        treatment for treatment in raw_treatments if isinstance(treatment, Mapping)
    ]
    if len(treatments) != len(raw_treatments):
        raise ArchitectureQueueError("architecture treatments are invalid")
    controls = [
        treatment
        for treatment in treatments
        if _string(treatment.get("treatment"), "treatment").endswith("-control")
    ]
    if len(controls) != 1:
        raise ArchitectureQueueError(
            "architecture plan must contain exactly one explicit control"
        )
    control = controls[0]
    wall_budget_seconds = _positive_number(
        plan.get("wall_budget_seconds"),
        "wall budget",
    )
    state_file = state_path.expanduser().resolve()
    evidence_root = evidence_directory.expanduser().resolve()
    state = (
        _json(state_file, "architecture queue state")
        if state_file.is_file()
        else {
            "schema_version": SCHEMA_VERSION,
            "report": QUEUE_REPORT,
            "status": "pending",
            "plan": str(plan_file),
            "plan_sha256": plan_sha256,
            "arms": {},
            "comparisons": {},
        }
    )
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("report") != QUEUE_REPORT
        or state.get("plan") != str(plan_file)
        or state.get("plan_sha256") != plan_sha256
        or not isinstance(state.get("arms"), dict)
        or not isinstance(state.get("comparisons"), dict)
    ):
        raise ArchitectureQueueError("architecture queue state is incompatible")
    pinned_baseline = state.get("frozen_baseline")
    if pinned_baseline is None:
        source_root = Path(
            _string(plan.get("source_run_root"), "source run root")
        ).resolve()
        baseline_artifact = _artifact(
            _immutable_manifest(source_root / "learner" / "champion.json")
        )
        state["frozen_baseline"] = baseline_artifact
    else:
        if not isinstance(pinned_baseline, Mapping):
            raise ArchitectureQueueError("frozen baseline pin is invalid")
        baseline_path = Path(
            _string(pinned_baseline.get("path"), "frozen baseline path")
        )
        if _artifact(baseline_path) != dict(pinned_baseline):
            raise ArchitectureQueueError("frozen baseline artifact changed")
        baseline_artifact = dict(pinned_baseline)
    baseline = Path(_string(baseline_artifact["path"], "frozen baseline path"))
    state["status"] = "running"
    atomic_json(state_file, state)
    arms = state["arms"]
    assert isinstance(arms, dict)
    for treatment in treatments:
        label = _string(treatment.get("treatment"), "treatment")
        result = arm_runner(
            plan_path=plan_file,
            plan_sha256=plan_sha256,
            treatment=treatment,
            wall_budget_seconds=wall_budget_seconds,
            orchestrator=orchestrator,
            poll_seconds=poll_seconds,
        )
        if result.get("status") != "completed":
            state["status"] = "retryable"
            arms[label] = result
            atomic_json(state_file, state)
            return state
        arms[label] = {
            "status": "completed",
            "champion_manifest": result["champion_manifest"],
            "champion_manifest_sha256": result["champion_manifest_sha256"],
            "champion_manifest_bytes": result["champion_manifest_bytes"],
        }
        atomic_json(state_file, state)

    control_label = _string(control.get("treatment"), "control treatment")
    control_manifest = Path(
        _string(arms[control_label]["champion_manifest"], "control champion")
    )
    control_profile = (
        Path(_string(control.get("run_root"), "control root"))
        / "profile-architecture-scratch.yaml"
    )
    arena_command = _resolve_executable(arena_executable)
    arena_root = evidence_root / "arenas"
    arena_root.mkdir(parents=True, exist_ok=True)
    shared_control_output = arena_root / "control-vs-baseline.json"
    if not shared_control_output.is_file():
        arena_runner(
            executable=arena_command,
            profile=control_profile,
            candidate=control_manifest,
            baseline=baseline,
            output=shared_control_output,
            device=device,
        )
    comparisons = state["comparisons"]
    assert isinstance(comparisons, dict)
    for treatment in treatments:
        label = _string(treatment.get("treatment"), "treatment")
        if label == control_label:
            continue
        treatment_manifest = Path(
            _string(arms[label]["champion_manifest"], f"{label} champion")
        )
        treatment_root = Path(_string(treatment.get("run_root"), f"{label} root"))
        treatment_profile = treatment_root / "profile-architecture-scratch.yaml"
        versus_baseline = arena_root / f"{label}-vs-baseline.json"
        versus_control = arena_root / f"{label}-vs-control.json"
        if not versus_baseline.is_file():
            arena_runner(
                executable=arena_command,
                profile=treatment_profile,
                candidate=treatment_manifest,
                baseline=baseline,
                output=versus_baseline,
                device=device,
            )
        if not versus_control.is_file():
            arena_runner(
                executable=arena_command,
                profile=treatment_profile,
                candidate=treatment_manifest,
                baseline=control_manifest,
                output=versus_control,
                device=device,
            )
        suite = architecture_suite_document(
            suite_id=f"{plan.get('suite')}.{label}",
            control_manifest=control_manifest,
            treatment_manifest=treatment_manifest,
            baseline_manifest=baseline,
            control_vs_baseline=shared_control_output,
            treatment_vs_baseline=versus_baseline,
            treatment_vs_control=versus_control,
        )
        suite_path = _write_immutable_json(
            evidence_root / "suites",
            label,
            suite,
        )
        evidence = build_architecture_ablation_evidence(suite_path)
        evidence_path = _write_immutable_json(
            evidence_root / "evidence",
            label,
            evidence,
        )
        comparisons[label] = {
            "status": "completed",
            "suite": _artifact(suite_path),
            "evidence": _artifact(evidence_path),
            "production_promotion_authorized": False,
            "adoption_authorized": False,
        }
        atomic_json(state_file, state)
    state.update(
        {
            "status": "completed",
            "completed_ns": time.time_ns(),
            "production_promotion_authorized": False,
            "adoption_authorized": False,
            "operator_review_required": True,
        }
    )
    atomic_json(state_file, state)
    return state


def run_architecture_queue(
    *,
    plan_path: Path,
    state_path: Path,
    evidence_directory: Path,
    orchestrator: str,
    arena_executable: str,
    device: str,
    poll_seconds: float,
    execution_lock_path: Path,
    arm_runner: ArmRunner = run_scratch_arm,
    arena_runner: ArenaRunner = _run_arena,
) -> dict[str, object]:
    state_file = state_path.expanduser().resolve()
    lock = exclusive_queue_lock(state_file)
    lock.__enter__()
    try:
        with exclusive_execution_lock(execution_lock_path.expanduser().resolve()):
            return _run_architecture_queue_locked(
                plan_path=plan_path,
                state_path=state_file,
                evidence_directory=evidence_directory,
                orchestrator=orchestrator,
                arena_executable=arena_executable,
                device=device,
                poll_seconds=poll_seconds,
                arm_runner=arm_runner,
                arena_runner=arena_runner,
            )
    finally:
        lock.__exit__(None, None, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--evidence-directory", required=True, type=Path)
    parser.add_argument("--execution-lock-path", required=True, type=Path)
    parser.add_argument("--orchestrator", default="startrain-orchestrate")
    parser.add_argument("--arena-executable", default="startrain-arena")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    arguments = parser.parse_args(argv)
    try:
        result = run_architecture_queue(
            plan_path=arguments.plan,
            state_path=arguments.state,
            evidence_directory=arguments.evidence_directory,
            orchestrator=arguments.orchestrator,
            arena_executable=arguments.arena_executable,
            device=arguments.device,
            poll_seconds=arguments.poll_seconds,
            execution_lock_path=arguments.execution_lock_path,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "completed" else 75


if __name__ == "__main__":
    raise SystemExit(main())
