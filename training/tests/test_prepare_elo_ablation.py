from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_elo_ablation import (
    CLEAN_TREATMENTS,
    DEFAULT_TREATMENTS,
    SYSTEM_TREATMENTS,
    WEIGHTED_TREATMENTS,
    main,
    prepare_elo_ablation,
)
from startrain.config import load_config

CONFIGS = Path(__file__).parents[1] / "configs"


def _winner_snapshot(source: Path) -> dict[str, object]:
    run_path = source / "run.json"
    champion_path = source / "learner" / "champion.json"
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    run = {
        "schema_version": 1,
        "run_id": "shared-parent-run",
        "generation_family": "selected-family",
        "created_ns": 1,
    }
    champion = {
        "model_identity": "selected-champion",
        "model_step": 400_000,
        "updated_ns": 2,
    }
    run_path.write_text(json.dumps(run), encoding="utf-8")
    champion_path.write_text(json.dumps(champion), encoding="utf-8")
    return {
        "schema_version": 1,
        "status": "verified",
        "label": "upstream-winner",
        "run_root": str(source.resolve()),
        "run_identity": {
            key: run[key] for key in ("run_id", "generation_family", "created_ns")
        },
        "run_identity_artifact": {
            "path": str(run_path.resolve()),
            "sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        },
        "champion": champion,
        "champion_pointer_artifact": {
            "path": str(champion_path.resolve()),
            "sha256": hashlib.sha256(champion_path.read_bytes()).hexdigest(),
        },
        "source_anchor": {
            "model_identity": "older",
            "model_step": 300_000,
        },
        "selection": "guarded_chronological_champion_frontier",
    }


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


def test_prepare_generates_isolated_weighted_generalist_matrix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-run"
    source.mkdir()
    output = tmp_path / "weighted-profiles"

    manifest = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "weighted-runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="weighted",
        seed=41,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=WEIGHTED_TREATMENTS,
        guard_rings=(),
    )

    assert manifest["guard_rings"] == []
    assert manifest["promotion_objective"] == "weighted_aggregate"
    assert manifest["per_ring_guarantees"] is False
    profiles = {
        name: load_config(output / f"{name}.yaml") for name in WEIGHTED_TREATMENTS
    }
    for profile in profiles.values():
        assert profile.arena.promotion_pair_ratios == {4: 1, 6: 1, 8: 1, 10: 7}
        assert profile.arena.required_regression_rings == ()
        assert profile.arena.per_ring_regression_floor_elo == {}
        assert profile.arena.weighted_initial_blocks == 15
        assert profile.arena.weighted_continuation_blocks == 10
        assert profile.arena.weighted_max_blocks == 50

    control_weights = load_config(
        CONFIGS / "h100-8gpu-throughput.yaml"
    ).orchestration.ring_mixture.weights_for_step(0)
    assert (
        profiles["weighted-control"].orchestration.ring_mixture.weights_for_step(0)
        == control_weights
    )
    assert profiles["ring10-65-weighted"].orchestration.ring_mixture.weights_for_step(
        0
    ) == (0.1, 0.1, 0.15, 0.65)
    assert profiles["ring10-70-weighted"].orchestration.ring_mixture.weights_for_step(
        0
    ) == (0.1, 0.1, 0.1, 0.7)


def test_prepare_rejects_mixed_promotion_objectives(tmp_path: Path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()

    with pytest.raises(ValueError, match="separate ablation plans"):
        prepare_elo_ablation(
            base_config=CONFIGS / "h100-8gpu-throughput.yaml",
            output_dir=tmp_path / "mixed-profiles",
            run_root_parent=tmp_path / "mixed-runs",
            run_id="shared-parent-run",
            source_run_root=source,
            prefix="mixed",
            seed=41,
            wall_budget_hours=8,
            leaf_budget=2_000_000_000,
            guard_floor_elo=-35,
            treatments=("control", "weighted-control"),
        )


def test_prepare_requires_staged_winner_snapshot_to_match_current_champion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected-source"
    source.mkdir()
    snapshot = _winner_snapshot(source)
    output = tmp_path / "selected-profiles"

    manifest = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "selected-runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="selected",
        seed=37,
        wall_budget_hours=1,
        leaf_budget=10,
        guard_floor_elo=-35,
        treatments=("control",),
        winner_snapshot=snapshot,
    )

    assert manifest["source_winner_snapshot"] == snapshot
    champion_path = source / "learner" / "champion.json"
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    champion["model_identity"] = "newer-champion"
    champion_path.write_text(json.dumps(champion), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact is stale"):
        prepare_elo_ablation(
            base_config=CONFIGS / "h100-8gpu-throughput.yaml",
            output_dir=tmp_path / "stale-profiles",
            run_root_parent=tmp_path / "stale-runs",
            run_id="shared-parent-run",
            source_run_root=source,
            prefix="stale",
            seed=37,
            wall_budget_hours=1,
            leaf_budget=10,
            guard_floor_elo=-35,
            treatments=("control",),
            winner_snapshot=snapshot,
        )


def test_prepare_generates_clean_warmstart_treatments(tmp_path: Path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()
    output = tmp_path / "clean-profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-champion-warmstart.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="clean",
        seed=31,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=("control", *CLEAN_TREATMENTS),
    )

    control = load_config(output / "control.yaml")
    quarter = load_config(output / "lr-quarter.yaml")
    assert quarter.optimizer.adamw_lr == pytest.approx(control.optimizer.adamw_lr / 2)
    assert quarter.optimizer.muon_lr == pytest.approx(control.optimizer.muon_lr / 2)

    fresh = load_config(output / "fresh-source.yaml")
    assert fresh.orchestration.model_refresh.candidate_probability == 0.5
    assert fresh.orchestration.model_refresh.history_probability == 0.15
    assert fresh.learner.selfplay_snapshot_warmup_interval_examples == 1_000_000

    hard = load_config(output / "hard-replay.yaml")
    assert hard.data.shards_per_batch == 4
    assert hard.selfplay.policy_surprise_weight == 0.5
    assert hard.selfplay.policy_surprise_max_weight == 4.0

    combined = load_config(output / "fresh-hard.yaml")
    assert combined.orchestration.model_refresh.selfplay_source == (
        "candidate_champion_history_mix"
    )
    assert combined.data.shards_per_batch == 4
    assert combined.selfplay.policy_surprise_weight == 0.5


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
