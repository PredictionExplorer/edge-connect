#!/usr/bin/env python3
"""Run and compare the frozen-replay optimizer suite before an Elo screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from startrain.runtime import atomic_json

if __package__:
    from .compare_frozen_replay_optimizer_calibration import compare_results
    from .prepare_elo_ablation import (
        RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
    )
    from .run_frozen_replay_optimizer_calibration import (
        CONTROL_ARM,
        PREIMPORT_CACHE_BOOTSTRAP_ENV,
        CalibrationSettings,
    )
else:
    from compare_frozen_replay_optimizer_calibration import compare_results
    from prepare_elo_ablation import RING10_OPTIMIZER_CALIBRATION_TREATMENTS
    from run_frozen_replay_optimizer_calibration import (
        CONTROL_ARM,
        PREIMPORT_CACHE_BOOTSTRAP_ENV,
        CalibrationSettings,
    )
from startrain.training import isolated_compile_cache

FORMAT = "startrain.frozen-replay-optimizer-calibration-queue"
SCREEN_PLAN_REPORT = "startrain-elo-ablation-plan"
SCHEMA_VERSION = 1


def _calibration_command(settings: CalibrationSettings) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_frozen_replay_optimizer_calibration.py")),
        "--config",
        str(settings.config),
        "--champion",
        str(settings.champion),
        "--replay-root",
        str(settings.replay_root),
        "--replay-cutoff",
        str(settings.replay_cutoff),
        "--output-dir",
        str(settings.output_dir),
        "--arm",
        settings.arm,
        "--steps",
        str(settings.steps),
        "--evaluation-batch-size",
        str(settings.evaluation_batch_size),
        "--max-samples",
        str(settings.max_samples),
        "--holdout-fraction",
        str(settings.holdout_fraction),
        "--seed",
        str(settings.seed),
        "--budget-h100-hours",
        str(settings.budget_h100_hours),
        "--checkpoint-interval",
        str(settings.checkpoint_interval),
    ]
    if settings.batch_size is not None:
        command.extend(("--batch-size", str(settings.batch_size)))
    if settings.device is not None:
        command.extend(("--device", settings.device))
    return command


def _run_calibration_process(
    settings: CalibrationSettings,
    *,
    environment: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    command = _calibration_command(settings)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=dict(environment),
        timeout=settings.budget_h100_hours * 3600.0 + 300.0,
    )
    stdout = completed.stdout.encode("utf-8")
    stderr = completed.stderr.encode("utf-8")
    evidence = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_tail": completed.stderr[-4000:],
    }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError as error:
        raise RuntimeError("calibration child emitted invalid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("calibration child emitted no JSON result")
    if completed.returncode != 0 and payload.get("status") != "error":
        raise RuntimeError(
            f"calibration child exited {completed.returncode} without error evidence"
        )
    return payload, evidence


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} is not a regular non-symlink file: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_output_root(path: Path) -> Path:
    source = Path(os.path.abspath(path.expanduser()))
    for component in (source, *source.parents):
        if component.is_symlink():
            raise ValueError(f"calibration output path contains a symlink: {component}")
    return source


def _treatments(plan: Mapping[str, object]) -> tuple[dict[str, Any], ...]:
    raw = plan.get("treatments")
    if not isinstance(raw, list):
        raise ValueError("calibration plan treatments are missing")
    records = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("calibration plan treatment is invalid")
        records.append(dict(item))
    observed = tuple(record.get("treatment") for record in records)
    if observed != RING10_OPTIMIZER_CALIBRATION_TREATMENTS:
        raise ValueError("calibration plan does not contain the exact optimizer suite")
    return tuple(records)


def _write_or_verify(path: Path, document: Mapping[str, object]) -> None:
    if path.exists():
        if _read_json(path, name="existing frozen calibration artifact") != dict(
            document
        ):
            raise ValueError(f"existing artifact is incompatible: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, document)


def build_screen_plan(
    calibration_plan: Mapping[str, object],
    comparison: Mapping[str, object],
    *,
    comparison_path: Path,
    output_path: Path,
    wall_budget_hours: float,
    leaf_budget: int,
) -> dict[str, object] | None:
    """Filter the frozen suite to control plus its unique gate-passing treatment."""

    if wall_budget_hours <= 0 or leaf_budget <= 0:
        raise ValueError("screen wall and leaf budgets must be positive")
    selection = comparison.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("calibration comparison selection is missing")
    selected = selection.get("selected_arm")
    fallback = selection.get("fallback_to_control")
    if fallback is True or selected == CONTROL_ARM:
        return None
    if selected not in RING10_OPTIMIZER_CALIBRATION_TREATMENTS:
        raise ValueError("calibration comparison selected an unknown treatment")

    source_records = _treatments(calibration_plan)
    selected_records = [
        record
        for record in source_records
        if record.get("treatment") in {CONTROL_ARM, selected}
    ]
    if [record.get("treatment") for record in selected_records] != [
        CONTROL_ARM,
        selected,
    ]:
        raise ValueError("screen plan could not preserve control/treatment order")
    screen = dict(calibration_plan)
    screen.update(
        {
            "report": SCREEN_PLAN_REPORT,
            "suite": "ring10-optimizer-screen",
            "wall_budget_seconds": wall_budget_hours * 3600.0,
            "leaf_budget": leaf_budget,
            "treatments": selected_records,
            "frozen_calibration": {
                "comparison": str(comparison_path),
                "comparison_sha256": _sha256(comparison_path),
                "selected_arm": selected,
                "selected_label": selection.get("selected_label"),
                "diagnostic_only": True,
                "production_promotion_authorized": False,
            },
        }
    )
    _write_or_verify(output_path, screen)
    return screen


def run_calibration_queue(
    *,
    plan_path: Path,
    champion: Path,
    replay_root: Path,
    replay_cutoff: int,
    output_root: Path,
    steps: int,
    device: str,
    budget_h100_hours: float,
    screen_plan_path: Path,
    screen_wall_budget_hours: float,
    screen_leaf_budget: int,
    batch_size: int | None = None,
    evaluation_batch_size: int = 64,
    max_samples: int = 65_536,
    holdout_fraction: float = 0.2,
    seed: int = 17,
    checkpoint_interval: int = 100,
) -> dict[str, object]:
    plan_source = plan_path.expanduser().resolve()
    plan = _read_json(plan_source, name="optimizer calibration plan")
    if (
        plan.get("report") != SCREEN_PLAN_REPORT
        or plan.get("suite") != "ring10-optimizer-calibration"
    ):
        raise ValueError("optimizer calibration plan has the wrong suite")
    records = _treatments(plan)
    output = _safe_output_root(output_root)
    replay_source = replay_root.expanduser().resolve()
    canonical_output = output.resolve(strict=False)
    if (
        canonical_output == replay_source
        or canonical_output in replay_source.parents
        or replay_source in canonical_output.parents
    ):
        raise ValueError("calibration output and source replay must not overlap")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "queue-state.json"
    state: dict[str, object] = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "plan": str(plan_source),
        "plan_sha256": _sha256(plan_source),
        "champion": str(champion.expanduser().resolve()),
        "replay_root": str(replay_root.expanduser().resolve()),
        "replay_cutoff": replay_cutoff,
        "execution": "isolated_subprocess_sequential_wave",
        "arms": {},
        "updated_ns": time.time_ns(),
    }
    if state_path.is_file():
        existing = _read_json(state_path, name="optimizer calibration queue state")
        for key in ("format", "schema_version", "plan", "plan_sha256", "champion"):
            if existing.get(key) != state.get(key):
                raise ValueError("existing optimizer calibration queue state changed")
        state = existing
        state["status"] = "running"

    result_paths = []
    arms_state = state.get("arms")
    if not isinstance(arms_state, dict):
        raise ValueError("optimizer calibration queue arm state is invalid")
    for record in records:
        arm = str(record["treatment"])
        profile = Path(str(record["profile"])).expanduser().resolve()
        arm_output = output / arm
        arm_state = arms_state.get(arm)
        if not isinstance(arm_state, dict):
            arm_state = {"status": "pending", "attempts": 0}
            arms_state[arm] = arm_state
        persisted_result = arm_state.get("result")
        persisted_sha256 = arm_state.get("result_sha256")
        if (
            arm_state.get("status") == "completed"
            and isinstance(persisted_result, str)
            and isinstance(persisted_sha256, str)
        ):
            result_path = Path(persisted_result).expanduser().resolve()
            if result_path.is_file() and _sha256(result_path) == persisted_sha256:
                result_paths.append(result_path)
                continue
            raise ValueError(f"completed calibration result changed for {arm}")
        arm_state.update(
            {
                "status": "running",
                "attempts": int(arm_state.get("attempts", 0)) + 1,
                "started_ns": time.time_ns(),
                "profile": str(profile),
                "profile_sha256": _sha256(profile),
            }
        )
        state["updated_ns"] = time.time_ns()
        atomic_json(state_path, state)
        settings = CalibrationSettings(
            config=profile,
            champion=champion.expanduser().resolve(),
            replay_root=replay_source,
            replay_cutoff=replay_cutoff,
            output_dir=arm_output,
            arm=arm,
            steps=steps,
            batch_size=batch_size,
            evaluation_batch_size=evaluation_batch_size,
            max_samples=max_samples,
            holdout_fraction=holdout_fraction,
            seed=seed,
            device=device,
            budget_h100_hours=budget_h100_hours,
            checkpoint_interval=checkpoint_interval,
        )
        planned_cache = None
        try:
            with isolated_compile_cache(arm_output) as planned_cache:
                child_environment = os.environ.copy()
                child_environment.pop(PREIMPORT_CACHE_BOOTSTRAP_ENV, None)
                result, process_evidence = _run_calibration_process(
                    settings,
                    environment=child_environment,
                )
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            arm_state.update(
                {
                    "status": "failed",
                    "completed_ns": time.time_ns(),
                    "compile_cache": (
                        planned_cache.as_dict() if planned_cache is not None else None
                    ),
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            )
            state.update(
                {
                    "status": "failed",
                    "error": {
                        "arm": arm,
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "updated_ns": time.time_ns(),
                }
            )
            atomic_json(state_path, state)
            return state
        if result.get("status") != "complete":
            failed = result.get("status") == "error"
            arm_state.update(
                {
                    "status": "failed" if failed else result.get("status"),
                    "completed_ns": time.time_ns(),
                    "process": process_evidence,
                    "error": result.get("error"),
                }
            )
            state["status"] = "failed" if failed else "paused"
            state["error"] = (
                {"arm": arm, "detail": result.get("error")} if failed else None
            )
            state["updated_ns"] = time.time_ns()
            atomic_json(state_path, state)
            return state
        result_path = arm_output / "result.json"
        result_device = result.get("device")
        compile_cache = (
            result_device.get("compile_cache")
            if isinstance(result_device, Mapping)
            else None
        )
        arm_state.update(
            {
                "status": "completed",
                "completed_ns": time.time_ns(),
                "result": str(result_path),
                "result_sha256": _sha256(result_path),
                "compile_cache": compile_cache,
                "process": process_evidence,
            }
        )
        result_paths.append(result_path)
        state["updated_ns"] = time.time_ns()
        atomic_json(state_path, state)

    comparison = compare_results(result_paths)
    comparison_path = output / "comparison.json"
    _write_or_verify(comparison_path, comparison)
    screen = build_screen_plan(
        plan,
        comparison,
        comparison_path=comparison_path,
        output_path=screen_plan_path.expanduser().resolve(),
        wall_budget_hours=screen_wall_budget_hours,
        leaf_budget=screen_leaf_budget,
    )
    state.update(
        {
            "status": "completed",
            "completed_ns": time.time_ns(),
            "comparison": {
                "path": str(comparison_path),
                "sha256": _sha256(comparison_path),
                "selection": comparison["selection"],
            },
            "screen_plan": (
                {
                    "path": str(screen_plan_path.expanduser().resolve()),
                    "sha256": _sha256(screen_plan_path.expanduser().resolve()),
                }
                if screen is not None
                else None
            ),
            "updated_ns": time.time_ns(),
        }
    )
    atomic_json(state_path, state)
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--champion", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--replay-cutoff", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--budget-h100-hours", required=True, type=float)
    parser.add_argument("--screen-plan", required=True, type=Path)
    parser.add_argument("--screen-wall-budget-hours", required=True, type=float)
    parser.add_argument("--screen-leaf-budget", required=True, type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=65_536)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        state = run_calibration_queue(
            plan_path=arguments.plan,
            champion=arguments.champion,
            replay_root=arguments.replay_root,
            replay_cutoff=arguments.replay_cutoff,
            output_root=arguments.output_root,
            steps=arguments.steps,
            device=arguments.device,
            budget_h100_hours=arguments.budget_h100_hours,
            screen_plan_path=arguments.screen_plan,
            screen_wall_budget_hours=arguments.screen_wall_budget_hours,
            screen_leaf_budget=arguments.screen_leaf_budget,
            batch_size=arguments.batch_size,
            evaluation_batch_size=arguments.evaluation_batch_size,
            max_samples=arguments.max_samples,
            holdout_fraction=arguments.holdout_fraction,
            seed=arguments.seed,
            checkpoint_interval=arguments.checkpoint_interval,
        )
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "format": FORMAT,
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(state, allow_nan=False, sort_keys=True))
    return 0 if state.get("status") == "completed" else 75


if __name__ == "__main__":
    raise SystemExit(main())
