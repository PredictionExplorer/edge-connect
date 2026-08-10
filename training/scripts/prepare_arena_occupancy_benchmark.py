#!/usr/bin/env python3
"""Freeze a benchmark-only ring-10 arena occupancy experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from startrain.checkpoint import sha256_file
from startrain.config import ExperimentConfig, load_config
from startrain.manifest_selection import (
    SelectionPlan,
    freeze_selection_plan,
    verify_selection_plan,
)
from startrain.native import load_star_native
from startrain.runtime import atomic_json

if __package__:
    from .evaluate_archived_manifests import plan_archived_manifest_evaluation
    from .validate_continuous_profile import validate_continuous_config
else:
    from evaluate_archived_manifests import plan_archived_manifest_evaluation
    from validate_continuous_profile import validate_continuous_config

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-arena-occupancy-benchmark-plan"
PLAN_NAME = "benchmark-plan.json"
SELECTION_PLAN_NAME = "selection-plan.json"
CONTROL_ARM = "continuation-25"
TREATMENT_ARM = "occupancy-50"
PAIR_STARTS = (50, 100, 50, 100)
TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRAINING_ROOT.parent
HARNESS_PATHS = (
    Path(__file__).resolve(),
    TRAINING_ROOT / "scripts" / "benchmark_arena_occupancy.py",
    TRAINING_ROOT / "startrain" / "arena.py",
    TRAINING_ROOT / "startrain" / "promotion.py",
)
DEFAULT_HOST_EXECUTION_LOCK = Path("/var/lib/edgeconnect/elo-ablation-execution.lock")


@dataclass(frozen=True, slots=True)
class VerifiedArenaOccupancyPlan:
    path: Path
    payload: dict[str, Any]
    experiment: ExperimentConfig
    selection: SelectionPlan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--physical-gpu-index", type=int, default=7)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    return parser


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _digest_payload(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("plan_digest", None)
    return hashlib.sha256(_canonical_bytes(material)).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"artifact may not be a symbolic link: {requested}")
    resolved = requested.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact does not exist: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_artifact(raw: object, name: str) -> Path:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    path_value = raw.get("path")
    bytes_value = raw.get("bytes")
    digest_value = raw.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{name}.path must be a non-empty string")
    if isinstance(bytes_value, bool) or not isinstance(bytes_value, int):
        raise ValueError(f"{name}.bytes must be an integer")
    if not isinstance(digest_value, str) or len(digest_value) != 64:
        raise ValueError(f"{name}.sha256 must be a SHA-256 digest")
    requested = Path(path_value).expanduser()
    if requested.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    path = requested.resolve()
    if (
        not path.is_file()
        or path.stat().st_size != bytes_value
        or sha256_file(path) != digest_value
    ):
        raise ValueError(f"{name} failed its frozen artifact digest")
    return path


def _release_manifest_artifact(commit: str) -> dict[str, object] | None:
    path = REPOSITORY_ROOT / "release-manifest.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("release manifest must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read release manifest: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("report") != "edgeconnect-immutable-release"
        or payload.get("commit") != commit
    ):
        raise ValueError("release manifest identity does not match the checkout")
    return _artifact(path)


def _git_revision(training_root: Path = TRAINING_ROOT) -> tuple[str, bool]:
    git = ["git", "-c", f"safe.directory={REPOSITORY_ROOT}"]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"],
            cwd=training_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=all"],
            cwd=training_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            f"cannot determine benchmark source revision: {error}"
        ) from error
    if len(commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("benchmark source revision is not a full Git commit")
    release_manifest = _release_manifest_artifact(commit)
    unexpected = [
        line
        for line in status.splitlines()
        if line.strip()
        and not (
            release_manifest is not None and line.strip() == "?? release-manifest.json"
        )
    ]
    return commit, not unexpected


def _assert_stopped_source(
    source: Path,
    *,
    allowed_lock_pid: int | None = None,
) -> None:
    lock = source / "coordinator.lock"
    if not lock.exists() and not lock.is_symlink():
        return
    if allowed_lock_pid is not None:
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("pid") == allowed_lock_pid:
            return
    raise ValueError(
        "source run has a coordinator lock; stop and snapshot it before benchmarking"
    )


def _harness_artifacts() -> list[dict[str, object]]:
    return [_artifact(path) for path in HARNESS_PATHS]


def _native_extension_path(native_module: object) -> Path:
    module_name = getattr(native_module, "__name__", None)
    if not isinstance(module_name, str) or not module_name:
        raise ValueError("native extension module name is unavailable")
    spec = importlib.util.find_spec(f"{module_name}.star_native")
    origin = spec.origin if spec is not None else None
    if not isinstance(origin, str) or not any(
        origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise ValueError("compiled native extension artifact is unavailable")
    return Path(origin).expanduser().resolve()


def _native_extension_artifact() -> dict[str, object]:
    native_module = load_star_native(required=True)
    if native_module is None:
        raise ValueError("native extension is required")
    rules_hash = getattr(native_module, "native_rules_hash", None)
    if not callable(rules_hash):
        raise ValueError("native extension identity is unavailable")
    rules_hash_value = rules_hash()
    if type(rules_hash_value) is not int:
        raise ValueError("native extension rules hash is invalid")
    artifact = _artifact(_native_extension_path(native_module))
    artifact["rules_hash"] = f"fnv1a64:{rules_hash_value:016x}"
    return artifact


def _gpu_index(device: str) -> int:
    prefix, separator, raw_index = device.partition(":")
    if prefix != "cuda" or not separator:
        raise ValueError("arena occupancy benchmark device must be cuda:<index>")
    try:
        index = int(raw_index)
    except ValueError as error:
        raise ValueError("arena occupancy benchmark CUDA index is invalid") from error
    if index < 0:
        raise ValueError("arena occupancy benchmark CUDA index must be non-negative")
    return index


def _validate_profile(experiment: ExperimentConfig, *, gpu_index: int) -> None:
    validate_continuous_config(experiment)
    if experiment.orchestration.training_objective != "ring10_only":
        raise ValueError("arena occupancy benchmark requires ring10_only")
    if experiment.arena.rings != (10,):
        raise ValueError("arena occupancy benchmark requires arena.rings: [10]")
    if (
        experiment.arena.pairs_per_ring != 50
        or experiment.arena.continuation_pairs_per_ring != 25
        or experiment.arena.minimum_pairs_per_ring != 50
        or experiment.arena.max_pairs_per_ring != 200
        or experiment.arena.pair_chunk_size is not None
    ):
        raise ValueError(
            "arena occupancy benchmark requires the unchunked production "
            "50/25/50 initial/continuation/minimum pair contract with a "
            "200-pair range"
        )
    if experiment.orchestration.promotion.gpu_id != gpu_index:
        raise ValueError("benchmark device must match the configured promotion GPU")


def _validate_repeats(repeats: int) -> None:
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise ValueError("repeats must be an integer")
    if repeats != len(PAIR_STARTS):
        raise ValueError(f"repeats must be exactly {len(PAIR_STARTS)}")


def _validate_sample_interval(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("sample interval must be finite and positive")


def _isolated_output(source: Path, output: Path) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("benchmark output must be outside the source run root")
    if output == REPOSITORY_ROOT or REPOSITORY_ROOT in output.parents:
        raise ValueError("benchmark output must be outside the Git repository")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    return output


def _execution_lock_path(path: Path, *, source: Path) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute():
        raise ValueError("execution lock path must be absolute")
    if requested.is_symlink():
        raise ValueError("execution lock path may not be a symbolic link")
    resolved = requested.resolve()
    if resolved == source or source in resolved.parents:
        raise ValueError("execution lock must be outside the source run root")
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise ValueError("execution lock must be outside the Git repository")
    return resolved


def _arms() -> list[dict[str, object]]:
    return [
        {
            "name": CONTROL_ARM,
            "total_pair_count": 50,
            "chunk_pair_count": 25,
            "chunks": 2,
            "concurrent_games": 50,
            "production_equivalent": True,
            "deployment_eligible": True,
        },
        {
            "name": TREATMENT_ARM,
            "total_pair_count": 50,
            "chunk_pair_count": 50,
            "chunks": 1,
            "concurrent_games": 100,
            "production_equivalent": False,
            "deployment_eligible": False,
        },
    ]


def _schedule(repeats: int) -> list[dict[str, object]]:
    schedule = []
    for repeat, pair_start in enumerate(PAIR_STARTS[:repeats]):
        order = (
            [CONTROL_ARM, TREATMENT_ARM]
            if repeat % 4 in (0, 3)
            else [TREATMENT_ARM, CONTROL_ARM]
        )
        schedule.append(
            {
                "repeat": repeat,
                "pair_start": pair_start,
                "arm_order": order,
            }
        )
    return schedule


def _statistical_contract(experiment: ExperimentConfig) -> dict[str, object]:
    return {
        "pairs_per_ring": experiment.arena.pairs_per_ring,
        "continuation_pairs_per_ring": (experiment.arena.continuation_pairs_per_ring),
        "minimum_pairs_per_ring": experiment.arena.minimum_pairs_per_ring,
        "max_pairs_per_ring": experiment.arena.max_pairs_per_ring,
        "simulations": experiment.arena.simulations,
        "max_considered": experiment.arena.max_considered,
        "alpha": experiment.arena.alpha,
        "beta": experiment.arena.beta,
    }


def _deployment_policy() -> dict[str, object]:
    return {
        "treatment_deployable": False,
        "reason": (
            "the production continuous profile caps continuation waves at "
            "25 pairs; the 50-pair arm is benchmark-only"
        ),
        "outcome_equivalence_required": False,
    }


def prepare_arena_occupancy_benchmark(
    *,
    source_run_root: Path,
    profile: Path,
    candidate_manifest: Path,
    output_dir: Path,
    device: str = "cuda:7",
    physical_gpu_index: int = 7,
    execution_lock: Path = DEFAULT_HOST_EXECUTION_LOCK,
    repeats: int = 4,
    sample_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    source = source_run_root.expanduser().resolve()
    profile_path = profile.expanduser().resolve()
    candidate_path = candidate_manifest.expanduser().resolve()
    destination = _isolated_output(source, output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source run root does not exist: {source}")
    _assert_stopped_source(source)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate manifest does not exist: {candidate_path}")
    _validate_repeats(repeats)
    _validate_sample_interval(sample_interval_seconds)
    execution_lock_path = _execution_lock_path(execution_lock, source=source)
    _gpu_index(device)
    if type(physical_gpu_index) is not int or physical_gpu_index < 0:
        raise ValueError("physical GPU index must be a non-negative integer")
    experiment = load_config(profile_path)
    _validate_profile(experiment, gpu_index=physical_gpu_index)
    source_commit, source_clean = _git_revision()
    if not source_clean:
        raise ValueError("benchmark plans require a clean source tree")
    harness_artifacts = _harness_artifacts()
    native_extension = _native_extension_artifact()
    release_manifest = _release_manifest_artifact(source_commit)

    selection = plan_archived_manifest_evaluation(
        source_run_root=source,
        profile=profile_path,
        candidate_manifest_paths=(candidate_path,),
        shortlist_size=1,
    )
    selection = verify_selection_plan(
        selection,
        expected_source_run_root=source,
    )
    if len(selection.candidates) != 1:
        raise ValueError("arena occupancy benchmark requires exactly one candidate")

    destination.mkdir(parents=True)
    selection_path = destination / SELECTION_PLAN_NAME
    freeze_selection_plan(selection_path, selection)
    selection_artifact = _artifact(selection_path)
    selection_artifact["plan_digest"] = selection.plan_digest

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": "prepared",
        "created_ns": time.time_ns(),
        "source_run_root": str(source),
        "profile": _artifact(profile_path),
        "selection_plan": selection_artifact,
        "source_revision": {
            "commit": source_commit,
            "tree_clean": True,
            "training_root": str(TRAINING_ROOT),
        },
        "harness_artifacts": harness_artifacts,
        "release_manifest": release_manifest,
        "native_extension": native_extension,
        "device": device,
        "physical_gpu_index": physical_gpu_index,
        "execution_lock": str(execution_lock_path),
        "ring": 10,
        "search_workers": 2,
        "sample_interval_seconds": sample_interval_seconds,
        "warmup_arm_order": [CONTROL_ARM, TREATMENT_ARM],
        "arms": _arms(),
        "schedule": _schedule(repeats),
        "statistical_contract": _statistical_contract(experiment),
        "deployment_policy": _deployment_policy(),
    }
    payload["plan_digest"] = _digest_payload(payload)
    plan_path = destination / PLAN_NAME
    atomic_json(plan_path, payload)
    os.chmod(plan_path, 0o444)
    return payload


def verify_arena_occupancy_plan(
    path: str | Path,
    *,
    allowed_lock_pid: int | None = None,
) -> VerifiedArenaOccupancyPlan:
    requested_plan = Path(path).expanduser()
    if requested_plan.is_symlink():
        raise ValueError("benchmark plan may not be a symbolic link")
    plan_path = requested_plan.resolve()
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark plan {plan_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("benchmark plan must contain an object")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("report") != REPORT_NAME
        or payload.get("status") != "prepared"
    ):
        raise ValueError("benchmark plan identity is invalid")
    if payload.get("plan_digest") != _digest_payload(payload):
        raise ValueError("benchmark plan digest mismatch")

    profile_path = _verify_artifact(payload.get("profile"), "profile")
    selection_path = _verify_artifact(
        payload.get("selection_plan"),
        "selection_plan",
    )
    source_value = payload.get("source_run_root")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("source_run_root must be a non-empty string")
    source = Path(source_value).expanduser().resolve()
    selection = verify_selection_plan(
        selection_path,
        expected_source_run_root=source,
    )
    raw_selection = payload.get("selection_plan")
    assert isinstance(raw_selection, dict)
    if raw_selection.get("plan_digest") != selection.plan_digest:
        raise ValueError("selection plan digest does not match benchmark plan")
    if len(selection.candidates) != 1:
        raise ValueError("benchmark selection plan must contain one candidate")
    if selection.evaluation_profile.verify() != profile_path:
        raise ValueError("selection and benchmark profiles do not match")

    device = payload.get("device")
    physical_gpu_index = payload.get("physical_gpu_index")
    if not isinstance(device, str):
        raise ValueError("benchmark GPU identity is invalid")
    _gpu_index(device)
    if type(physical_gpu_index) is not int or physical_gpu_index < 0:
        raise ValueError("benchmark physical GPU identity is invalid")
    experiment = load_config(profile_path)
    _validate_profile(experiment, gpu_index=physical_gpu_index)
    lock_value = payload.get("execution_lock")
    if not isinstance(lock_value, str):
        raise ValueError("benchmark execution lock is missing")
    if _execution_lock_path(Path(lock_value), source=source) != Path(lock_value):
        raise ValueError("benchmark execution lock is not canonical")
    _assert_stopped_source(source, allowed_lock_pid=allowed_lock_pid)
    revision = payload.get("source_revision")
    if not isinstance(revision, dict):
        raise ValueError("benchmark source revision is missing")
    observed_commit, observed_clean = _git_revision()
    if (
        revision.get("commit") != observed_commit
        or revision.get("tree_clean") is not True
        or not observed_clean
        or revision.get("training_root") != str(TRAINING_ROOT)
    ):
        raise ValueError("benchmark source revision changed or is dirty")
    if payload.get("release_manifest") != _release_manifest_artifact(observed_commit):
        raise ValueError("benchmark release manifest changed")
    raw_harness = payload.get("harness_artifacts")
    if not isinstance(raw_harness, list) or len(raw_harness) != len(HARNESS_PATHS):
        raise ValueError("benchmark harness artifact set is invalid")
    observed_harness = [
        str(_verify_artifact(raw, f"harness artifact {index}"))
        for index, raw in enumerate(raw_harness)
    ]
    if observed_harness != [str(path) for path in HARNESS_PATHS]:
        raise ValueError("benchmark harness artifact paths are invalid")
    if payload.get("native_extension") != _native_extension_artifact():
        raise ValueError("benchmark native extension changed")
    if payload.get("ring") != 10 or payload.get("search_workers") != 2:
        raise ValueError("benchmark ring or search-worker contract is invalid")
    if payload.get("statistical_contract") != _statistical_contract(experiment):
        raise ValueError("benchmark statistical contract is invalid")

    interval = payload.get("sample_interval_seconds")
    if isinstance(interval, bool) or not isinstance(interval, int | float):
        raise ValueError("sample interval is invalid")
    _validate_sample_interval(float(interval))
    if payload.get("arms") != _arms():
        raise ValueError("benchmark arms do not match the registered contract")
    schedule = payload.get("schedule")
    if not isinstance(schedule, list):
        raise ValueError("benchmark schedule must be an array")
    _validate_repeats(len(schedule))
    if schedule != _schedule(len(schedule)):
        raise ValueError("benchmark schedule does not match the registered contract")
    if payload.get("warmup_arm_order") != [CONTROL_ARM, TREATMENT_ARM]:
        raise ValueError("benchmark warmup order is invalid")
    if payload.get("deployment_policy") != _deployment_policy():
        raise ValueError("benchmark treatment must remain non-deployable")
    return VerifiedArenaOccupancyPlan(
        path=plan_path,
        payload=payload,
        experiment=experiment,
        selection=selection,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        plan = prepare_arena_occupancy_benchmark(
            source_run_root=arguments.source_run_root,
            profile=arguments.profile,
            candidate_manifest=arguments.candidate_manifest,
            output_dir=arguments.output_dir,
            device=arguments.device,
            physical_gpu_index=arguments.physical_gpu_index,
            execution_lock=arguments.execution_lock,
            repeats=arguments.repeats,
            sample_interval_seconds=arguments.sample_interval_seconds,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
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
    print(json.dumps({"status": "ok", **plan}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
