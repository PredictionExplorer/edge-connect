#!/usr/bin/env python3
"""Create one hash-pinned empty architecture-treatment run root."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from startrain.config import load_config
from startrain.model import model_parameter_count

SCHEMA_VERSION = 1
REPORT = "startrain-scratch-architecture-initialization"


class ScratchPreparationError(RuntimeError):
    """The plan cannot safely initialize an empty architecture root."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_scratch_root(plan_path: Path, treatment: str) -> dict[str, object]:
    plan_file = plan_path.expanduser().resolve()
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScratchPreparationError(f"cannot read scratch plan: {exc}") from exc
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != 1
        or plan.get("report") != "startrain-elo-ablation-plan"
        or plan.get("initialization") != "scratch"
    ):
        raise ScratchPreparationError("plan is not a scratch architecture plan")
    raw_treatments = plan.get("treatments")
    if not isinstance(raw_treatments, list):
        raise ScratchPreparationError("scratch plan treatments are missing")
    matches = [
        item
        for item in raw_treatments
        if isinstance(item, dict) and item.get("treatment") == treatment
    ]
    if len(matches) != 1:
        raise ScratchPreparationError("scratch treatment is missing or duplicated")
    entry = matches[0]
    profile_value = entry.get("profile")
    root_value = entry.get("run_root")
    run_id = entry.get("run_id")
    profile_sha256 = entry.get("profile_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (profile_value, root_value, run_id, profile_sha256)
    ):
        raise ScratchPreparationError("scratch treatment paths or identity are invalid")
    profile = Path(str(profile_value)).expanduser().resolve()
    root = Path(str(root_value)).expanduser().resolve()
    if (
        not profile.is_file()
        or profile.is_symlink()
        or _sha256(profile) != profile_sha256
    ):
        raise ScratchPreparationError("scratch profile is missing or changed")
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"scratch run root already exists: {root}")
    experiment = load_config(profile)
    if (
        experiment.orchestration.run_id != run_id
        or Path(experiment.orchestration.directories.root).resolve() != root
    ):
        raise ScratchPreparationError("scratch profile identity does not match plan")

    source_root = Path(str(plan.get("source_run_root"))).expanduser().resolve()
    source_champion = source_root / "learner" / "champion.json"
    champion_evidence = None
    if source_champion.is_file() and not source_champion.is_symlink():
        payload = json.loads(source_champion.read_text(encoding="utf-8"))
        champion_evidence = {
            "path": str(source_champion),
            "sha256": _sha256(source_champion),
            "model_identity": payload.get("model_identity"),
            "model_step": payload.get("model_step"),
        }

    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
        installed_profile = root / "profile-architecture-scratch.yaml"
        shutil.copy2(profile, installed_profile)
        installed_sha256 = _sha256(installed_profile)
        checksum = f"{installed_sha256}  {installed_profile.name}\n"
        installed_profile.with_suffix(".sha256").write_text(
            checksum,
            encoding="utf-8",
        )
        (root / "profile.sha256").write_text(checksum, encoding="utf-8")
        evidence = {
            "schema_version": SCHEMA_VERSION,
            "report": REPORT,
            "status": "prepared",
            "created_ns": time.time_ns(),
            "plan": str(plan_file),
            "plan_sha256": _sha256(plan_file),
            "suite": plan.get("suite"),
            "treatment": treatment,
            "seed": plan.get("seed"),
            "run_id": run_id,
            "run_root": str(root),
            "profile": str(installed_profile),
            "profile_sha256": installed_sha256,
            "source_frozen_baseline": champion_evidence,
            "model_config": experiment.as_dict()["model"],
            "model_parameters": model_parameter_count(experiment.model),
            "initialization": {
                "kind": "random_step_zero",
                "external_replay": False,
                "external_checkpoint": False,
                "partial_model_load": False,
            },
        }
        (root / "scratch-initialization.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        forbidden = (
            root / "run.json",
            root / "replay",
            root / "learner",
            root / "arena",
            root / "coordinator.lock",
        )
        if any(path.exists() or path.is_symlink() for path in forbidden):
            raise ScratchPreparationError(
                "scratch root contains imported training artifacts"
            )
        return evidence
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--treatment", required=True)
    arguments = parser.parse_args(argv)
    try:
        evidence = prepare_scratch_root(arguments.plan, arguments.treatment)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
