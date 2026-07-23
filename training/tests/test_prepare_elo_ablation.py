from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_elo_ablation import (
    DEFAULT_TREATMENTS,
    SYSTEM_TREATMENTS,
    main,
    prepare_elo_ablation,
)
from startrain.config import load_config

CONFIGS = Path(__file__).parents[1] / "configs"


def _prepare(tmp_path: Path) -> tuple[dict[str, object], Path]:
    source = tmp_path / "source-run"
    source.mkdir()
    output = tmp_path / "profiles"
    manifest = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="pilot",
        seed=23,
        wall_budget_hours=8.0,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35.0,
        treatments=DEFAULT_TREATMENTS,
    )
    return manifest, output


def test_prepare_generates_strict_one_factor_profiles(tmp_path: Path) -> None:
    manifest, output = _prepare(tmp_path)

    assert manifest["report"] == "startrain-elo-ablation-plan"
    assert manifest["guard_rings"] == [4, 6, 8]
    assert manifest["wall_budget_seconds"] == 28_800
    assert [item["treatment"] for item in manifest["treatments"]] == list(
        DEFAULT_TREATMENTS
    )
    persisted = json.loads((output / "ablation-plan.json").read_text())
    assert persisted == manifest

    profiles = {
        name: load_config(output / f"{name}.yaml") for name in DEFAULT_TREATMENTS
    }
    for name, profile in profiles.items():
        assert profile.train.seed == profile.selfplay.seed == 23
        assert profile.orchestration.run_id == "shared-parent-run"
        assert profile.orchestration.directories.root.endswith(f"pilot-{name}-seed23")
        assert profile.arena.per_ring_regression_floor_elo == {
            4: -35.0,
            6: -35.0,
            8: -35.0,
        }

    assert profiles["control"].learner.target_updates_per_new_sample is None
    assert profiles["utd-1"].learner.target_updates_per_new_sample == 1.0
    assert (
        profiles["plateau-keep"].orchestration.plateau.action
        == "reduce_lr_keep_weights"
    )
    freshness = profiles["freshness-mix"]
    assert freshness.learner.selfplay_snapshot_interval_examples == 3_000_000
    assert (
        freshness.orchestration.model_refresh.selfplay_source
        == "candidate_champion_history_mix"
    )
    assert freshness.orchestration.model_refresh.candidate_probability == 0.35
    assert freshness.orchestration.model_refresh.history_probability == 0.15
    assert profiles["ring10-70"].orchestration.ring_mixture.weights_for_step(0) == (
        0.1,
        0.1,
        0.1,
        0.7,
    )
    search = profiles["search-quality"].selfplay
    assert search.full_probability == 0.35
    assert search.full_simulations == 384
    assert search.max_considered_cap == 64


def test_prepare_refuses_overwrite_and_invalid_guard_floor(tmp_path: Path) -> None:
    _prepare(tmp_path)
    source = tmp_path / "source-run"

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_elo_ablation(
            base_config=CONFIGS / "h100-8gpu-throughput.yaml",
            output_dir=tmp_path / "profiles",
            run_root_parent=tmp_path / "runs",
            run_id="shared-parent-run",
            source_run_root=source,
            prefix="pilot",
            seed=23,
            wall_budget_hours=8.0,
            leaf_budget=1,
            guard_floor_elo=-35.0,
            treatments=("control",),
        )

    with pytest.raises(ValueError, match="negative non-inferiority"):
        prepare_elo_ablation(
            base_config=CONFIGS / "h100-8gpu-throughput.yaml",
            output_dir=tmp_path / "other",
            run_root_parent=tmp_path / "runs",
            run_id="shared-parent-run",
            source_run_root=source,
            prefix="pilot",
            seed=23,
            wall_budget_hours=8.0,
            leaf_budget=1,
            guard_floor_elo=0.0,
            treatments=("control",),
        )


def test_prepare_generates_optional_system_screening_profiles(tmp_path: Path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()
    output = tmp_path / "system-profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="system",
        seed=29,
        wall_budget_hours=0.5,
        leaf_budget=100_000_000,
        guard_floor_elo=-35,
        treatments=SYSTEM_TREATMENTS,
    )

    actor_batch = load_config(output / "actor-batch-160.yaml")
    assert actor_batch.orchestration.actor_games_per_batch == 160
    assert {gpu.actor_batch_size for gpu in actor_batch.orchestration.actor_gpus} == {
        160
    }
    actor_lanes = load_config(output / "actor-lanes-3.yaml")
    assert sorted(gpu.actor_lanes for gpu in actor_lanes.orchestration.actor_gpus) == [
        1,
        3,
        3,
        3,
        3,
        3,
        3,
    ]
    assert (
        load_config(output / "learner-batch-768.yaml").train.per_rank_batch_size == 768
    )
    learner_1024 = load_config(output / "learner-batch-1024.yaml")
    assert learner_1024.train.per_rank_batch_size == 1024
    assert learner_1024.learner.target_updates_per_new_sample == 1.0


def test_prepare_cli_reports_missing_source_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--base-config",
            str(CONFIGS / "h100-8gpu-throughput.yaml"),
            "--output-dir",
            str(tmp_path / "profiles"),
            "--run-root-parent",
            str(tmp_path / "runs"),
            "--run-id",
            "run",
            "--source-run-root",
            str(tmp_path / "missing"),
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "source run root does not exist" in payload["error"]
