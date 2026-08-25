from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.prepare_elo_ablation import (
    CLEAN_TREATMENTS,
    DEFAULT_TREATMENTS,
    RING10_ATTENTION_TREATMENTS,
    RING10_CAPACITY_TREATMENTS,
    RING10_EFFICIENCY_TREATMENTS,
    RING10_LIVE_CADENCE_TREATMENTS,
    RING10_ONLY_TREATMENTS,
    RING10_OPTIMIZATION_TREATMENTS,
    RING10_OPTIMIZER_CALIBRATION_LABELS,
    RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
    RING10_RELATIONAL_TREATMENTS,
    RING10_TRAINING_DYNAMICS_TREATMENTS,
    SYSTEM_TREATMENTS,
    WEIGHTED_TREATMENTS,
    _validate_ring10_optimizer_calibration_transition,
    main,
    prepare_elo_ablation,
    resolve_treatments,
)
from startrain.config import load_config
from startrain.model import model_parameter_count

CONFIGS = Path(__file__).parents[1] / "configs"


def _treatment_records(manifest: dict[str, object]) -> list[dict[str, object]]:
    raw = manifest.get("treatments")
    assert isinstance(raw, list)
    assert all(isinstance(item, dict) for item in raw)
    return raw


def _runtime_optimizer_evidence(profile: Path) -> dict[str, object]:
    source = profile.resolve()
    return {
        "adamw_lr": 7.5e-05,
        "muon_lr": 0.005,
        "source_profile_path": str(source),
        "source_profile_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


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


def _write_live_ring10_profile(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    raw = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict)
    learner = raw["learner"]
    assert isinstance(learner, dict)
    learner.update(
        {
            "candidate_interval_examples": 2_000_000,
            "selfplay_snapshot_interval_examples": None,
            "selfplay_snapshot_warmup_examples": 0,
            "selfplay_snapshot_warmup_interval_examples": None,
            "target_updates_per_new_sample": 1.0,
        }
    )
    profile = tmp_path / "frozen-live-ring10.yaml"
    profile.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return profile, raw


def test_prepare_generates_strict_one_factor_profiles(tmp_path: Path) -> None:
    manifest, output = _prepare(tmp_path)

    assert manifest["report"] == "startrain-elo-ablation-plan"
    assert manifest["training_objective"] == "generalist"
    assert manifest["promotion_objective"] == "ring_10_guarded"
    assert manifest["guard_rings"] == [4, 6, 8]
    assert manifest["wall_budget_seconds"] == 28_800
    assert [item["treatment"] for item in _treatment_records(manifest)] == list(
        DEFAULT_TREATMENTS
    )
    persisted = json.loads((output / "ablation-plan.json").read_text())
    assert persisted == manifest

    profiles = {
        name: load_config(output / f"{name}.yaml") for name in DEFAULT_TREATMENTS
    }
    for name, profile in profiles.items():
        assert profile.orchestration.training_objective == "generalist"
        assert profile.train.seed == profile.selfplay.seed == profile.arena.seed == 23
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

    for size in (128, 160, 192):
        actor_batch = load_config(output / f"actor-batch-{size}.yaml")
        assert actor_batch.orchestration.actor_games_per_batch == size
        assert {
            gpu.actor_batch_size for gpu in actor_batch.orchestration.actor_gpus
        } == {size}
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
    assert manifest["training_objective"] == "generalist"
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


def test_prepare_generates_fail_closed_ring10_only_treatment(tmp_path: Path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()
    output = tmp_path / "ring10-only-profiles"

    manifest = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "ring10-only-runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="ring10-only",
        seed=43,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=RING10_ONLY_TREATMENTS,
        guard_rings=(),
    )

    assert manifest["training_objective"] == "ring10_only"
    assert manifest["promotion_objective"] == "ring_10_only"
    assert manifest["guard_rings"] == []
    assert manifest["per_ring_guarantees"] is False
    assert _treatment_records(manifest)[0]["training_objective"] == "ring10_only"
    profile_path = output / "ring10-only.yaml"
    profile = load_config(profile_path)
    assert profile.game.rings == (4, 6, 8, 10)
    assert profile.orchestration.training_objective == "ring10_only"
    assert profile.selfplay.rings == 10
    assert tuple(
        (stage.from_step, stage.weights)
        for stage in profile.orchestration.ring_mixture.step_weights
    ) == ((0, (0.0, 0.0, 0.0, 1.0)),)
    assert profile.arena.rings == (10,)
    assert profile.arena.required_regression_rings == ()
    assert profile.arena.per_ring_regression_floor_elo == {}
    assert profile.arena.promotion_pair_ratios == {}
    assert profile.arena.weighted_initial_blocks == 0
    assert profile.arena.weighted_continuation_blocks == 0
    assert profile.arena.weighted_max_blocks == 0
    serialized = profile_path.read_text(encoding="utf-8")
    assert "promotion_pair_ratios:" not in serialized
    assert "weighted_initial_blocks:" not in serialized


def test_prepare_generates_ring10_efficiency_suite(tmp_path: Path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()
    output = tmp_path / "ring10-efficiency-profiles"

    manifest = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "ring10-efficiency-runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="ring10-efficiency",
        seed=47,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=RING10_EFFICIENCY_TREATMENTS,
        guard_rings=(),
        suite="ring10-efficiency",
    )

    assert [item["treatment"] for item in _treatment_records(manifest)] == list(
        RING10_EFFICIENCY_TREATMENTS
    )
    assert manifest["training_objective"] == "ring10_only"
    assert manifest["promotion_objective"] == "ring_10_only"
    assert manifest["guard_rings"] == []
    assert manifest["suite"] == "ring10-efficiency"

    control = load_config(output / "ring10-only.yaml")
    learner_slack = load_config(output / "ring10-learner-slack-64.yaml")
    actor_lanes = load_config(output / "ring10-actor-lanes-3.yaml")
    for profile in (control, learner_slack, actor_lanes):
        assert profile.orchestration.training_objective == "ring10_only"
        assert profile.arena.rings == (10,)
        assert profile.arena.required_regression_rings == ()
        assert profile.arena.per_ring_regression_floor_elo == {}

    assert not control.orchestration.allow_colocated_workers
    assert sum(gpu.actor_lanes for gpu in control.orchestration.actor_gpus) == 13

    assert learner_slack.orchestration.allow_colocated_workers
    colocated = [
        gpu for gpu in learner_slack.orchestration.actor_gpus if gpu.gpu_id == 0
    ]
    assert len(colocated) == 1
    assert colocated[0].actor_batch_size == 64
    assert colocated[0].actor_lanes == 1
    assert colocated[0].cpu_affinity == "0-103"
    assert learner_slack.orchestration.promotion.gpu_id == 7
    assert sum(gpu.actor_lanes for gpu in learner_slack.orchestration.actor_gpus) == 14

    assert sorted(gpu.actor_lanes for gpu in actor_lanes.orchestration.actor_gpus) == [
        1,
        3,
        3,
        3,
        3,
        3,
        3,
    ]
    assert all(gpu.gpu_id != 0 for gpu in actor_lanes.orchestration.actor_gpus)


def test_ring10_efficiency_suite_rejects_control_topology_drift(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    raw["orchestration"]["gpus"][1]["actor_lanes"] = 1
    base = tmp_path / "drifted-ring10.yaml"
    base.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="topology drifted"):
        prepare_elo_ablation(
            base_config=base,
            output_dir=tmp_path / "profiles",
            run_root_parent=tmp_path / "runs",
            run_id="shared-parent-run",
            source_run_root=source,
            prefix="ring10-efficiency",
            seed=47,
            wall_budget_hours=8,
            leaf_budget=2_000_000_000,
            guard_floor_elo=-35,
            treatments=RING10_EFFICIENCY_TREATMENTS,
            guard_rings=(),
            suite="ring10-efficiency",
        )
    assert not (tmp_path / "profiles").exists()


def test_prepare_generates_ring10_cadence_and_freshness_suite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-run"
    source.mkdir()
    output = tmp_path / "ring10-optimization-profiles"

    manifest = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "ring10-optimization-runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="ring10-optimization",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=RING10_OPTIMIZATION_TREATMENTS,
        guard_rings=(),
        suite="ring10-optimization",
    )

    assert manifest["suite"] == "ring10-optimization"
    assert manifest["training_objective"] == "ring10_only"
    assert [item["treatment"] for item in _treatment_records(manifest)] == list(
        RING10_OPTIMIZATION_TREATMENTS
    )
    control = load_config(output / "ring10-optimization-control.yaml")
    cadence = load_config(output / "ring10-cadence-5m.yaml")
    freshness = load_config(output / "ring10-freshness-50.yaml")
    for profile in (control, cadence, freshness):
        assert profile.learner.target_updates_per_new_sample == 1.0
        assert profile.orchestration.training_objective == "ring10_only"
        assert profile.arena.rings == (10,)
        assert profile.arena.pairs_per_ring == 50
        assert profile.arena.continuation_pairs_per_ring == 50
        assert sum(gpu.actor_lanes for gpu in profile.orchestration.actor_gpus) == 13
    assert control.learner.candidate_interval_examples == 2_000_000
    assert cadence.learner.candidate_interval_examples == 5_000_000
    assert freshness.learner.candidate_interval_examples == 2_000_000
    assert control.orchestration.model_refresh.selfplay_source == "champion"
    assert cadence.orchestration.model_refresh.selfplay_source == "champion"
    assert (
        freshness.orchestration.model_refresh.selfplay_source
        == "candidate_champion_mix"
    )
    assert freshness.orchestration.model_refresh.candidate_probability == 0.5
    assert freshness.orchestration.model_refresh.history_probability == 0.0
    assert freshness.learner.selfplay_snapshot_interval_examples == 3_000_000


def test_prepare_generates_strict_ring10_optimizer_calibration_suite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    replay = source / "replay"
    replay.mkdir()
    with sqlite3.connect(replay / "manifest.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE shards (id INTEGER PRIMARY KEY, state TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO shards(id, state) VALUES (42, 'ready')")
    runtime_profile = source / "profile-relocated.yaml"
    runtime_profile.write_bytes((CONFIGS / "h100-8gpu-ring10-only.yaml").read_bytes())
    runtime_optimizer = {
        **_runtime_optimizer_evidence(runtime_profile),
        "recovery_checkpoint_sha256": "a" * 64,
    }
    output = tmp_path / "optimizer-calibration"

    manifest = prepare_elo_ablation(
        base_config=runtime_profile,
        output_dir=output,
        run_root_parent=tmp_path / "runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="optimizer-calibration",
        seed=17,
        wall_budget_hours=2,
        leaf_budget=1,
        guard_floor_elo=-35,
        treatments=RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
        guard_rings=(),
        suite="ring10-optimizer-calibration",
        runtime_effective_optimizer=runtime_optimizer,
    )

    assert manifest["suite"] == "ring10-optimizer-calibration"
    assert manifest["source_replay_cutoff"] == 42
    assert manifest["runtime_effective_optimizer"] == runtime_optimizer
    records = _treatment_records(manifest)
    assert [record["treatment"] for record in records] == list(
        RING10_OPTIMIZER_CALIBRATION_TREATMENTS
    )
    assert [record["calibration_label"] for record in records] == [
        RING10_OPTIMIZER_CALIBRATION_LABELS[treatment]
        for treatment in RING10_OPTIMIZER_CALIBRATION_TREATMENTS
    ]
    assert [record["calibration_phase"] for record in records] == [
        "primary",
        "primary",
        "primary",
        "follow_on",
    ]

    control = load_config(output / "ring10-optimizer-runtime-effective-control.yaml")
    clip_two = load_config(output / "ring10-optimizer-clip-norm-2.yaml")
    clip_five = load_config(output / "ring10-optimizer-clip-norm-5.yaml")
    half_lr = load_config(output / "ring10-optimizer-0.5x-effective-lr.yaml")
    assert {
        profile.optimizer.kind for profile in (control, clip_two, clip_five, half_lr)
    } == {"muon_adamw"}
    assert control.train.gradient_clip_norm == 1.0
    assert control.optimizer.adamw_lr == pytest.approx(7.5e-05)
    assert control.optimizer.muon_lr == pytest.approx(0.005)
    assert clip_two.train.gradient_clip_norm == 2.0
    assert clip_five.train.gradient_clip_norm == 5.0
    assert half_lr.train.gradient_clip_norm == 1.0
    assert half_lr.optimizer.adamw_lr == pytest.approx(control.optimizer.adamw_lr * 0.5)
    assert half_lr.optimizer.muon_lr == pytest.approx(control.optimizer.muon_lr * 0.5)
    for profile in (control, clip_two, clip_five, half_lr):
        assert profile.learner.minimum_replay_shard_id_exclusive == 42


def test_ring10_optimizer_calibration_rejects_adamw_and_extra_factors(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    raw["optimizer"]["kind"] = "adamw"
    base = tmp_path / "adamw.yaml"
    base.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="excludes AdamW"):
        prepare_elo_ablation(
            base_config=base,
            output_dir=tmp_path / "profiles",
            run_root_parent=tmp_path / "runs",
            run_id="source-run",
            source_run_root=source,
            prefix="optimizer-calibration",
            seed=17,
            wall_budget_hours=2,
            leaf_budget=1,
            guard_floor_elo=-35,
            treatments=RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
            guard_rings=(),
            suite="ring10-optimizer-calibration",
            runtime_effective_optimizer=_runtime_optimizer_evidence(base),
        )

    with pytest.raises(ValueError, match="exact frozen profile"):
        prepare_elo_ablation(
            base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
            output_dir=tmp_path / "static-profiles",
            run_root_parent=tmp_path / "static-runs",
            run_id="source-run",
            source_run_root=source,
            prefix="optimizer-calibration",
            seed=17,
            wall_budget_hours=2,
            leaf_budget=1,
            guard_floor_elo=-35,
            treatments=RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
            guard_rings=(),
            suite="ring10-optimizer-calibration",
            runtime_effective_optimizer=_runtime_optimizer_evidence(
                CONFIGS / "h100-8gpu-ring10-only.yaml"
            ),
        )

    zero_source = tmp_path / "zero-source"
    zero_replay = zero_source / "replay"
    zero_replay.mkdir(parents=True)
    with sqlite3.connect(zero_replay / "manifest.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE shards (id INTEGER PRIMARY KEY, state TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO shards(id, state) VALUES (0, 'ready')")
    zero_profile = zero_source / "profile-relocated.yaml"
    zero_profile.write_bytes((CONFIGS / "h100-8gpu-ring10-only.yaml").read_bytes())
    with pytest.raises(ValueError, match="positive frozen replay cutoff"):
        prepare_elo_ablation(
            base_config=zero_profile,
            output_dir=tmp_path / "zero-profiles",
            run_root_parent=tmp_path / "zero-runs",
            run_id="source-run",
            source_run_root=zero_source,
            prefix="optimizer-calibration",
            seed=17,
            wall_budget_hours=2,
            leaf_budget=1,
            guard_floor_elo=-35,
            treatments=RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
            guard_rings=(),
            suite="ring10-optimizer-calibration",
            runtime_effective_optimizer=_runtime_optimizer_evidence(zero_profile),
        )

    before = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    after = deepcopy(before)
    after["train"]["gradient_clip_norm"] = 2.0
    after["optimizer"]["weight_decay"] = 0.02
    with pytest.raises(ValueError, match="one-factor contract"):
        _validate_ring10_optimizer_calibration_transition(
            before,
            after,
            "ring10-optimizer-clip-norm-2",
        )


def test_ring10_optimizer_treatments_require_complete_named_suite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(
        ValueError,
        match="complete --suite ring10-optimizer-calibration",
    ):
        prepare_elo_ablation(
            base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
            output_dir=tmp_path / "profiles",
            run_root_parent=tmp_path / "runs",
            run_id="source-run",
            source_run_root=source,
            prefix="optimizer-calibration",
            seed=17,
            wall_budget_hours=2,
            leaf_budget=1,
            guard_floor_elo=-35,
            treatments=("ring10-optimizer-clip-norm-2",),
            guard_rings=(),
        )


def test_prepare_generates_frozen_live_ring10_cadence_suite(
    tmp_path: Path,
) -> None:
    base, frozen = _write_live_ring10_profile(tmp_path)
    source = tmp_path / "source-run"
    source.mkdir()
    output = tmp_path / "ring10-live-cadence-profiles"
    run_root_parent = tmp_path / "ring10-live-cadence-runs"

    manifest = prepare_elo_ablation(
        base_config=base,
        output_dir=output,
        run_root_parent=run_root_parent,
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="ring10-live-cadence",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=RING10_LIVE_CADENCE_TREATMENTS,
        guard_rings=(),
        suite="ring10-live-cadence",
    )

    assert manifest["suite"] == "ring10-live-cadence"
    assert manifest["initialization"] == "fork"
    assert manifest["training_objective"] == "ring10_only"
    assert manifest["promotion_objective"] == "ring_10_only"
    assert manifest["guard_rings"] == []
    assert [item["treatment"] for item in _treatment_records(manifest)] == list(
        RING10_LIVE_CADENCE_TREATMENTS
    )

    control_raw = yaml.safe_load(
        (output / "ring10-live-cadence-control.yaml").read_text(encoding="utf-8")
    )
    treatment_raw = yaml.safe_load(
        (output / "ring10-live-cadence-5m.yaml").read_text(encoding="utf-8")
    )
    expected_control = deepcopy(frozen)
    control_orchestration = expected_control["orchestration"]
    assert isinstance(control_orchestration, dict)
    control_orchestration["run_id"] = "shared-parent-run"
    control_directories = control_orchestration["directories"]
    assert isinstance(control_directories, dict)
    control_directories["root"] = str(
        run_root_parent / "ring10-live-cadence-ring10-live-cadence-control-seed17"
    )
    assert control_raw == expected_control

    expected_treatment = deepcopy(expected_control)
    treatment_orchestration = expected_treatment["orchestration"]
    assert isinstance(treatment_orchestration, dict)
    treatment_directories = treatment_orchestration["directories"]
    assert isinstance(treatment_directories, dict)
    treatment_directories["root"] = str(
        run_root_parent / "ring10-live-cadence-ring10-live-cadence-5m-seed17"
    )
    treatment_learner = expected_treatment["learner"]
    assert isinstance(treatment_learner, dict)
    treatment_learner["candidate_interval_examples"] = 5_000_000
    assert treatment_raw == expected_treatment

    control = load_config(output / "ring10-live-cadence-control.yaml")
    treatment = load_config(output / "ring10-live-cadence-5m.yaml")
    assert control.learner.candidate_interval_examples == 2_000_000
    assert treatment.learner.candidate_interval_examples == 5_000_000
    for profile in (control, treatment):
        assert profile.learner.target_updates_per_new_sample == 1.0
        assert profile.learner.selfplay_snapshot_interval_examples is None
        assert profile.learner.selfplay_snapshot_warmup_examples == 0
        assert profile.learner.selfplay_snapshot_warmup_interval_examples is None
        assert profile.orchestration.model_refresh.selfplay_source == "champion"
        assert profile.arena.rings == (10,)
        assert profile.arena.per_ring_regression_floor_elo == {}


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("missing-candidate-cadence", "explicit learner.candidate_interval_examples"),
        ("already-at-target", "below 5000000"),
        ("slower-than-target", "below 5000000"),
        ("missing-utd", "frozen UTD=1.0"),
        ("different-utd", "frozen UTD=1.0"),
        ("generalist-objective", "requires a ring10_only base profile"),
        ("multi-ring-arena", "requires a single-ring unguarded legacy arena"),
    ],
)
def test_live_ring10_cadence_suite_rejects_ambiguous_base(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    base, raw = _write_live_ring10_profile(tmp_path)
    learner = raw["learner"]
    assert isinstance(learner, dict)
    if mutation == "missing-candidate-cadence":
        learner.pop("candidate_interval_examples")
    elif mutation == "already-at-target":
        learner["candidate_interval_examples"] = 5_000_000
    elif mutation == "slower-than-target":
        learner["candidate_interval_examples"] = 6_000_000
    elif mutation == "missing-utd":
        learner.pop("target_updates_per_new_sample")
    elif mutation == "different-utd":
        learner["target_updates_per_new_sample"] = 0.75
    elif mutation == "generalist-objective":
        orchestration = raw["orchestration"]
        assert isinstance(orchestration, dict)
        orchestration["training_objective"] = "generalist"
    elif mutation == "multi-ring-arena":
        arena = raw["arena"]
        assert isinstance(arena, dict)
        arena["rings"] = [4, 6, 8, 10]
    else:
        raise AssertionError(f"unsupported mutation: {mutation}")
    base.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "profiles"

    with pytest.raises(ValueError, match=expected_error):
        prepare_elo_ablation(
            base_config=base,
            output_dir=output,
            run_root_parent=tmp_path / "runs",
            run_id="shared-parent-run",
            source_run_root=source,
            prefix="ring10-live-cadence",
            seed=17,
            wall_budget_hours=8,
            leaf_budget=2_000_000_000,
            guard_floor_elo=-35,
            treatments=RING10_LIVE_CADENCE_TREATMENTS,
            guard_rings=(),
            suite="ring10-live-cadence",
        )
    assert not output.exists()


def test_live_ring10_cadence_treatments_require_complete_suite(
    tmp_path: Path,
) -> None:
    base, _ = _write_live_ring10_profile(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "profiles"

    with pytest.raises(ValueError, match="complete --suite ring10-live-cadence"):
        prepare_elo_ablation(
            base_config=base,
            output_dir=output,
            run_root_parent=tmp_path / "runs",
            run_id="shared-parent-run",
            source_run_root=source,
            prefix="ring10-live-cadence",
            seed=17,
            wall_budget_hours=8,
            leaf_budget=2_000_000_000,
            guard_floor_elo=-35,
            treatments=RING10_LIVE_CADENCE_TREATMENTS,
            guard_rings=(),
        )
    assert not output.exists()


def test_prepare_generates_ring10_training_dynamics_suite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    replay = source / "replay"
    replay.mkdir()
    with sqlite3.connect(replay / "manifest.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE shards (id INTEGER PRIMARY KEY, state TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO shards(id, state) VALUES (42, 'ready')")
    output = tmp_path / "profiles"
    manifest = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix="ring10-dynamics",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=RING10_TRAINING_DYNAMICS_TREATMENTS,
        guard_rings=(),
        suite="ring10-training-dynamics",
    )

    assert manifest["initialization"] == "fork"
    assert manifest["source_replay_cutoff"] == 42
    control = load_config(output / "ring10-dynamics-control.yaml")
    adamw = load_config(output / "ring10-dynamics-adamw.yaml")
    ema = load_config(output / "ring10-dynamics-ema-1m.yaml")
    freshness = load_config(output / "ring10-dynamics-freshness-50.yaml")
    clinch = load_config(output / "ring10-dynamics-clinch-outcome-only.yaml")
    assert control.optimizer.kind == "muon_adamw"
    assert adamw.optimizer.kind == "adamw"
    assert ema.train.ema_half_life_examples == 1_000_000
    assert freshness.orchestration.model_refresh.selfplay_source == (
        "candidate_champion_mix"
    )
    assert freshness.orchestration.model_refresh.candidate_probability == 0.5
    assert clinch.selfplay.clinch_auxiliary_targets == "outcome_only"
    for name in RING10_TRAINING_DYNAMICS_TREATMENTS:
        assert (
            load_config(
                output / f"{name}.yaml"
            ).learner.minimum_replay_shard_id_exclusive
            == 42
        )


def test_direct_treatments_cannot_bypass_initialization_or_replay_policy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    replay = source / "replay"
    replay.mkdir(parents=True)
    with sqlite3.connect(replay / "manifest.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE shards (id INTEGER PRIMARY KEY, state TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO shards(id, state) VALUES (73, 'ready')")

    architecture = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
        output_dir=tmp_path / "architecture",
        run_root_parent=tmp_path / "architecture-runs",
        run_id="source-run",
        source_run_root=source,
        prefix="architecture",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=("ring10-attention-full-kv",),
        guard_rings=(),
    )
    assert architecture["initialization"] == "scratch"
    assert _treatment_records(architecture)[0]["run_id"] != "source-run"

    dynamics = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
        output_dir=tmp_path / "dynamics",
        run_root_parent=tmp_path / "dynamics-runs",
        run_id="source-run",
        source_run_root=source,
        prefix="dynamics",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=("ring10-dynamics-adamw",),
        guard_rings=(),
    )
    assert dynamics["source_replay_cutoff"] == 73
    profile = load_config(Path(str(_treatment_records(dynamics)[0]["profile"])))
    assert profile.learner.minimum_replay_shard_id_exclusive == 73

    with pytest.raises(ValueError, match="separate ablation plans"):
        prepare_elo_ablation(
            base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
            output_dir=tmp_path / "mixed",
            run_root_parent=tmp_path / "mixed-runs",
            run_id="source-run",
            source_run_root=source,
            prefix="mixed",
            seed=17,
            wall_budget_hours=8,
            leaf_budget=2_000_000_000,
            guard_floor_elo=-35,
            treatments=(
                "ring10-attention-control",
                "ring10-dynamics-control",
            ),
            guard_rings=(),
        )


@pytest.mark.parametrize(
    ("suite", "treatments"),
    [
        ("ring10-attention-reallocation", RING10_ATTENTION_TREATMENTS),
        ("ring10-relational", RING10_RELATIONAL_TREATMENTS),
        ("ring10-capacity", RING10_CAPACITY_TREATMENTS),
    ],
)
def test_prepare_marks_architecture_suites_scratch(
    tmp_path: Path,
    suite: str,
    treatments: tuple[str, ...],
) -> None:
    source = tmp_path / suite / "source"
    source.mkdir(parents=True)
    output = tmp_path / suite / "profiles"
    manifest = prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
        output_dir=output,
        run_root_parent=tmp_path / suite / "runs",
        run_id="shared-parent-run",
        source_run_root=source,
        prefix=suite,
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=treatments,
        guard_rings=(),
        suite=suite,
    )

    assert manifest["initialization"] == "scratch"
    records = _treatment_records(manifest)
    assert [item["treatment"] for item in records] == list(treatments)
    run_ids = {str(item["run_id"]) for item in records}
    assert len(run_ids) == len(treatments)
    assert all(run_id.startswith("scratch-") for run_id in run_ids)
    for item in records:
        assert (
            load_config(Path(str(item["profile"]))).orchestration.run_id
            == item["run_id"]
        )


def test_prepare_architecture_suite_parameter_contracts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    def prepare(
        suite: str,
        treatments: tuple[str, ...],
    ) -> Path:
        output = tmp_path / suite
        prepare_elo_ablation(
            base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
            output_dir=output,
            run_root_parent=tmp_path / f"{suite}-runs",
            run_id="shared-parent-run",
            source_run_root=source,
            prefix=suite,
            seed=17,
            wall_budget_hours=8,
            leaf_budget=2_000_000_000,
            guard_floor_elo=-35,
            treatments=treatments,
            guard_rings=(),
            suite=suite,
        )
        return output

    attention = prepare(
        "ring10-attention-reallocation",
        RING10_ATTENTION_TREATMENTS,
    )
    attention_control = load_config(attention / "ring10-attention-control.yaml")
    attention_full = load_config(attention / "ring10-attention-full-kv.yaml")
    assert model_parameter_count(attention_control.model) == 10_476_983
    assert model_parameter_count(attention_full.model) == 10_476_983

    relational = prepare("ring10-relational", RING10_RELATIONAL_TREATMENTS)
    relational_control = load_config(relational / "ring10-relational-control.yaml")
    local_heavy = load_config(relational / "ring10-relational-local-heavy.yaml")
    source_gated = load_config(relational / "ring10-relational-source-gated.yaml")
    assert model_parameter_count(relational_control.model) == 10_476_983
    assert model_parameter_count(local_heavy.model) == 10_476_953
    assert model_parameter_count(source_gated.model) == 10_476_983

    capacity = prepare("ring10-capacity", RING10_CAPACITY_TREATMENTS)
    capacity_control = load_config(capacity / "ring10-capacity-control.yaml")
    capacity_depth = load_config(capacity / "ring10-capacity-depth-7.yaml")
    capacity_width = load_config(capacity / "ring10-capacity-width-512.yaml")
    assert model_parameter_count(capacity_control.model) == 10_476_983
    assert model_parameter_count(capacity_depth.model) == 14_614_199
    assert model_parameter_count(capacity_width.model) == 18_556_727


def test_prepare_canonicalizes_architecture_before_parameter_validation(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    raw["model"]["rrt_groups"] = 6
    base = tmp_path / "drifted.yaml"
    base.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()

    output = tmp_path / "profiles"
    prepare_elo_ablation(
        base_config=base,
        output_dir=output,
        run_root_parent=tmp_path / "runs",
        run_id="source-run",
        source_run_root=source,
        prefix="attention",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=("ring10-attention-control",),
        guard_rings=(),
    )
    control = load_config(output / "ring10-attention-control.yaml")
    assert control.model.rrt_groups == 5
    assert model_parameter_count(control.model) == 10_476_983


def test_capacity_control_resets_equal_count_attention_drift(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    raw["model"]["kv_heads"] = 12
    raw["model"]["ff_multiplier"] = 2.0
    base = tmp_path / "equal-count-drift.yaml"
    base.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "capacity"

    prepare_elo_ablation(
        base_config=base,
        output_dir=output,
        run_root_parent=tmp_path / "runs",
        run_id="source-run",
        source_run_root=source,
        prefix="capacity",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=RING10_CAPACITY_TREATMENTS,
        guard_rings=(),
        suite="ring10-capacity",
    )

    control = load_config(output / "ring10-capacity-control.yaml")
    depth = load_config(output / "ring10-capacity-depth-7.yaml")
    width = load_config(output / "ring10-capacity-width-512.yaml")
    assert (control.model.kv_heads, control.model.ff_multiplier) == (3, 2.5)
    assert (depth.model.kv_heads, depth.model.ff_multiplier) == (3, 2.5)
    assert (width.model.kv_heads, width.model.ff_multiplier) == (4, 2.5)


def test_resolve_treatments_keeps_suites_fail_closed() -> None:
    assert (
        resolve_treatments(
            suite="ring10-efficiency",
            treatments=None,
        )
        == RING10_EFFICIENCY_TREATMENTS
    )
    assert (
        resolve_treatments(
            suite="ring10-optimization",
            treatments=None,
        )
        == RING10_OPTIMIZATION_TREATMENTS
    )
    assert (
        resolve_treatments(
            suite="ring10-optimizer-calibration",
            treatments=None,
        )
        == RING10_OPTIMIZER_CALIBRATION_TREATMENTS
    )
    assert (
        resolve_treatments(
            suite="ring10-live-cadence",
            treatments=None,
        )
        == RING10_LIVE_CADENCE_TREATMENTS
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_treatments(
            suite="ring10-efficiency",
            treatments=("ring10-only",),
        )
    with pytest.raises(ValueError, match="unknown treatment suite"):
        resolve_treatments(suite="unknown", treatments=None)


def test_prepare_cli_rejects_suite_and_ad_hoc_treatment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    exit_code = main(
        [
            "--base-config",
            str(CONFIGS / "h100-8gpu-ring10-only.yaml"),
            "--output-dir",
            str(tmp_path / "profiles"),
            "--run-root-parent",
            str(tmp_path / "runs"),
            "--run-id",
            "shared-parent-run",
            "--source-run-root",
            str(source),
            "--suite",
            "ring10-efficiency",
            "--treatment",
            "ring10-only",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "--suite cannot be combined" in payload["error"]


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


def test_prepare_rejects_mixed_training_objectives(tmp_path: Path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()

    with pytest.raises(ValueError, match="incompatible training objectives"):
        prepare_elo_ablation(
            base_config=CONFIGS / "h100-8gpu-throughput.yaml",
            output_dir=tmp_path / "mixed-training-profiles",
            run_root_parent=tmp_path / "mixed-training-runs",
            run_id="shared-parent-run",
            source_run_root=source,
            prefix="mixed-training",
            seed=41,
            wall_budget_hours=8,
            leaf_budget=2_000_000_000,
            guard_floor_elo=-35,
            treatments=("control", "ring10-only"),
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
