from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_elo_ablation import (
    RING10_ATTENTION_TREATMENTS,
    prepare_elo_ablation,
)
from scripts.prepare_scratch_architecture import prepare_scratch_root
from startrain.config import load_config

CONFIGS = Path(__file__).parents[1] / "configs"


def test_prepare_scratch_architecture_root_is_empty_and_pinned(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    profiles = tmp_path / "profiles"
    plan = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
        output_dir=profiles,
        run_root_parent=tmp_path / "runs",
        run_id="source-run",
        source_run_root=source,
        prefix="attention",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=RING10_ATTENTION_TREATMENTS,
        guard_rings=(),
        suite="ring10-attention-reallocation",
    )
    plan_path = profiles / "ablation-plan.json"

    evidence = prepare_scratch_root(plan_path, "ring10-attention-control")
    root = Path(str(evidence["run_root"]))
    profile = Path(str(evidence["profile"]))

    assert plan["initialization"] == "scratch"
    assert evidence["initialization"] == {
        "kind": "random_step_zero",
        "external_replay": False,
        "external_checkpoint": False,
        "partial_model_load": False,
    }
    assert evidence["model_parameters"] == 10_929_399
    assert load_config(profile).orchestration.run_id == evidence["run_id"]
    assert (root / "scratch-initialization.json").is_file()
    assert not (root / "run.json").exists()
    assert not (root / "replay").exists()
    assert not (root / "learner").exists()

    with pytest.raises(FileExistsError):
        prepare_scratch_root(plan_path, "ring10-attention-control")


def test_prepare_scratch_rejects_fork_plan(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    profiles = tmp_path / "profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
        output_dir=profiles,
        run_root_parent=tmp_path / "runs",
        run_id="source-run",
        source_run_root=source,
        prefix="control",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=("ring10-only",),
        guard_rings=(),
    )

    with pytest.raises(RuntimeError, match="not a scratch"):
        prepare_scratch_root(
            profiles / "ablation-plan.json",
            "ring10-only",
        )
