#!/usr/bin/env python3
"""Fork an inactive training snapshot into an isolated Elo-ablation run root."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from startrain.config import ExperimentConfig, load_config
from startrain.replay_store import ReplayStore
from startrain.runtime import atomic_json, load_run_identity

if __package__:
    from .prepare_elo_ablation import verify_winner_snapshot
else:
    from prepare_elo_ablation import verify_winner_snapshot

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-elo-ablation-branch"
_ROTATED_DIRECTORIES = ("status", "logs", "metrics")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--treatment", required=True)
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


def _is_immutable(relative: Path) -> bool:
    parts = relative.parts
    if len(parts) >= 2 and parts[:2] in (
        ("replay", "shards"),
        ("learner", "checkpoints"),
        ("learner", "manifests"),
    ):
        return True
    return (
        len(parts) >= 3
        and parts[:2] == ("learner", "recovery")
        and relative.suffix == ".pt"
    )


def _copy_function(
    source_root: Path,
    counters: dict[str, int],
):
    def copy(source_value: str, destination_value: str) -> str:
        source = Path(source_value)
        destination = Path(destination_value)
        relative = source.relative_to(source_root)
        size = source.stat().st_size
        if _is_immutable(relative):
            try:
                os.link(source, destination)
                counters["linked_files"] += 1
                counters["linked_bytes"] += size
                return str(destination)
            except OSError as error:
                if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                    raise
        shutil.copy2(source, destination)
        counters["copied_files"] += 1
        counters["copied_bytes"] += size
        return str(destination)

    return copy


def _rotate_branch_runtime(destination: Path) -> None:
    parent = destination / "ablation-parent"
    inherited_parent = destination / ".inherited-ablation-parent"
    if parent.exists() or parent.is_symlink():
        if inherited_parent.exists() or inherited_parent.is_symlink():
            raise FileExistsError(
                "forked ablation contains conflicting inherited parent artifacts"
            )
        parent.rename(inherited_parent)
    parent.mkdir()
    if inherited_parent.exists() or inherited_parent.is_symlink():
        inherited_parent.rename(parent / "ancestor")
    for name in _ROTATED_DIRECTORIES:
        current = destination / name
        if current.exists():
            current.rename(parent / name)
        current.mkdir()
    learner_metrics = destination / "learner" / "metrics.jsonl"
    if learner_metrics.exists():
        learner_metrics.rename(parent / "learner-metrics.jsonl")
    strength = destination / "strength-efficiency.json"
    if strength.exists():
        strength.rename(parent / strength.name)
    (destination / "coordinator.lock").unlink(missing_ok=True)


def _prepare_utd_segment(
    destination: Path,
    *,
    experiment: ExperimentConfig,
    run_id: str,
    generation_family: str,
) -> dict[str, object] | None:
    target = experiment.learner.target_updates_per_new_sample
    if target is None:
        return None
    segment_path = destination / "learner" / "utd-segment.json"
    if segment_path.is_file():
        return _read_json(segment_path)
    recovery = _read_json(destination / "learner" / "recovery.json")
    examples = recovery.get("examples_consumed")
    if isinstance(examples, bool) or not isinstance(examples, int) or examples < 0:
        raise ValueError("recovery pointer lacks a valid UTD examples baseline")
    with ReplayStore(destination / "replay") as store:
        if not store.committed_sample_history_is_complete(
            run_id=run_id,
            generation_family=generation_family,
        ):
            raise ValueError("UTD treatment requires complete committed-sample history")
        committed = store.total_committed_sample_count(
            run_id=run_id,
            generation_family=generation_family,
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "generation_family": generation_family,
        "target_updates_per_new_sample": float(target),
        "baseline_examples_consumed": examples,
        "baseline_committed_replay_samples": committed,
        "created_ns": time.time_ns(),
    }
    atomic_json(segment_path, payload)
    return payload


def _plan_treatment(plan: dict[str, Any], treatment: str) -> dict[str, str]:
    raw = plan.get("treatments")
    if not isinstance(raw, list):
        raise ValueError("ablation plan treatments must be a list")
    matches = [
        item
        for item in raw
        if isinstance(item, dict) and item.get("treatment") == treatment
    ]
    if len(matches) != 1:
        raise ValueError(f"plan does not contain one treatment named {treatment!r}")
    item = matches[0]
    required = ("profile", "profile_sha256", "run_root", "treatment")
    if any(not isinstance(item.get(name), str) or not item[name] for name in required):
        raise ValueError("ablation treatment entry is malformed")
    return {name: str(item[name]) for name in required}


def fork_elo_ablation(
    *,
    source_run_root: Path,
    plan_path: Path,
    treatment: str,
) -> dict[str, object]:
    source = source_run_root.expanduser().resolve()
    plan_file = plan_path.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source run root does not exist: {source}")
    if (source / "coordinator.lock").exists():
        raise RuntimeError("source run root still has a coordinator lock")
    if not plan_file.is_file():
        raise FileNotFoundError(f"ablation plan does not exist: {plan_file}")
    plan = _read_json(plan_file)
    if plan.get("report") != "startrain-elo-ablation-plan":
        raise ValueError("unsupported ablation plan")
    configured_source = plan.get("source_run_root")
    if (
        not isinstance(configured_source, str)
        or Path(configured_source).expanduser().resolve() != source
    ):
        raise ValueError("plan source run root does not match requested source")
    raw_winner_snapshot = plan.get("source_winner_snapshot")
    winner_snapshot = (
        verify_winner_snapshot(source, raw_winner_snapshot)
        if isinstance(raw_winner_snapshot, dict)
        else None
    )
    if raw_winner_snapshot is not None and winner_snapshot is None:
        raise ValueError("plan source winner snapshot is malformed")
    entry = _plan_treatment(plan, treatment)
    profile_path = Path(entry["profile"]).expanduser().resolve()
    if not profile_path.is_file() or _sha256(profile_path) != entry["profile_sha256"]:
        raise ValueError("treatment profile is missing or its digest changed")
    experiment = load_config(profile_path)
    destination = Path(entry["run_root"]).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"treatment run root already exists: {destination}")
    identity = load_run_identity(source / "run.json")
    if experiment.orchestration.run_id != identity.run_id:
        raise ValueError("treatment run ID does not match source run identity")
    if Path(experiment.orchestration.directories.root).resolve() != destination:
        raise ValueError("treatment profile run root does not match plan")

    counters = {
        "linked_files": 0,
        "linked_bytes": 0,
        "copied_files": 0,
        "copied_bytes": 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            source,
            destination,
            symlinks=True,
            copy_function=_copy_function(source, counters),
        )
        _rotate_branch_runtime(destination)
        installed_profile = destination / "profile-elo-ablation.yaml"
        shutil.copy2(profile_path, installed_profile)
        profile_sha256 = _sha256(installed_profile)
        checksum = f"{profile_sha256}  {installed_profile.name}\n"
        installed_profile.with_suffix(".sha256").write_text(
            checksum,
            encoding="utf-8",
        )
        (destination / "profile.sha256").write_text(
            checksum,
            encoding="utf-8",
        )
        utd_segment = _prepare_utd_segment(
            destination,
            experiment=experiment,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )
        champion = _read_json(destination / "learner" / "champion.json")
        if winner_snapshot is not None:
            selected = winner_snapshot["champion"]
            assert isinstance(selected, dict)
            if any(
                champion.get(field) != selected.get(field)
                for field in ("model_identity", "model_step")
            ):
                raise ValueError(
                    "forked champion differs from the verified winner snapshot"
                )
        candidate_path = destination / "learner" / "candidate.json"
        candidate = _read_json(candidate_path) if candidate_path.is_file() else None
        metadata: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "report": REPORT_NAME,
            "treatment": treatment,
            "source_run_root": str(source),
            "source_run_id": identity.run_id,
            "source_generation_family": identity.generation_family,
            "source_created_ns": identity.created_ns,
            "source_winner_snapshot": winner_snapshot,
            "futility_policy": plan.get("futility_policy"),
            "prepared_ns": time.time_ns(),
            "measurement_started_ns": None,
            "measurement_stopped_ns": None,
            "measurement_stop_reason": None,
            "profile": str(installed_profile),
            "profile_sha256": profile_sha256,
            "wall_budget_seconds": plan.get("wall_budget_seconds"),
            "leaf_budget": plan.get("leaf_budget"),
            "guard_rings": plan.get("guard_rings"),
            "guard_floor_elo": plan.get("guard_floor_elo"),
            "anchor": {
                key: champion.get(key)
                for key in ("model_identity", "model_step", "updated_ns")
            },
            "starting_candidate": (
                {
                    key: candidate.get(key)
                    for key in ("model_identity", "model_step", "updated_ns")
                }
                if candidate is not None
                else None
            ),
            "utd_segment": utd_segment,
            "storage": counters,
        }
        atomic_json(destination / "ablation.json", metadata)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return metadata


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        metadata = fork_elo_ablation(
            source_run_root=arguments.source_run_root,
            plan_path=arguments.plan,
            treatment=arguments.treatment,
        )
    except (
        FileExistsError,
        FileNotFoundError,
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
    print(json.dumps({"status": "ok", **metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
