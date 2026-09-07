from dataclasses import replace
from pathlib import Path

import pytest

from scripts.validate_continuous_profile import validate_continuous_config
from startrain.config import ArenaConfig, ConfigError, RingWeightStage, load_config
from startrain.config_compatibility import compatible_config_epoch_payloads
from startrain.model import model_parameter_count

CONFIGS = Path(__file__).parents[1] / "configs"


def priority_profile():
    return load_config(CONFIGS / "h100-8gpu-largest-board-priority.yaml")


def test_priority_profile_changes_only_objective_and_opted_in_inference():
    before = load_config(CONFIGS / "h100-8gpu-variant-efficiency-stage-b.yaml")
    after = priority_profile()
    validate_continuous_config(before)
    validate_continuous_config(after)
    expected = before.as_dict()
    expected["orchestration"]["training_objective"] = "ring10_priority"
    expected["orchestration"]["ring_mixture"]["step_weights"] = (
        {"from_step": 0, "weights": (0.05, 0.05, 0.05, 0.85)},
    )
    expected["selfplay"]["rings"] = 10
    expected["arena"]["rings"] = (10,)
    expected["arena"]["required_regression_rings"] = ()
    expected["orchestration"]["model_refresh"]["inference"][
        "preserve_broadcast_topology"
    ] = True
    assert after.as_dict() == expected
    assert model_parameter_count(after.model) == 17_402_775
    assert after.learner.target_updates_per_new_sample == 1.5
    assert after.orchestration.promotion.pause_strategy == "suspend"
    for step in (0, 100_000, 10**18):
        assert after.orchestration.ring_mixture.weights_for_step(step) == (
            0.05,
            0.05,
            0.05,
            0.85,
        )
    variants = after.selfplay.variants
    mode_fractions = (
        variants.standard,
        variants.classic,
        variants.pie * variants.pie_classic_share,
        variants.pie * (1 - variants.pie_classic_share),
        variants.handicap * variants.handicap_classic_share,
        variants.handicap * (1 - variants.handicap_classic_share),
    )
    assert mode_fractions == pytest.approx((1 / 6,) * 6)


@pytest.mark.parametrize("rings", [(10,), (4, 10), (6, 8, 10), (4, 6, 8, 10)])
def test_balanced_arena_accepts_supported_ring_subsets(rings):
    assert ArenaConfig(rings=rings, balanced_cells=True).rings == rings


@pytest.mark.parametrize("rings", [(), (10, 4), (10, 10), (5,)])
def test_balanced_arena_rejects_invalid_ring_subsets(rings):
    with pytest.raises(ConfigError, match="sorted unique subset"):
        ArenaConfig(rings=rings, balanced_cells=True)


@pytest.mark.parametrize(
    "stages",
    [
        (),
        (RingWeightStage(1, (0.05, 0.05, 0.05, 0.85)),),
        (RingWeightStage(0, (0.0, 0.05, 0.05, 0.9)),),
        (RingWeightStage(0, (0.1, 0.1, 0.3, 0.5)),),
        (RingWeightStage(0, (0.05, 0.05, 0.05, 0.9)),),
        (
            RingWeightStage(0, (0.05, 0.05, 0.05, 0.85)),
            RingWeightStage(100, (0.25, 0.25, 0.25, 0.25)),
        ),
    ],
)
def test_priority_objective_requires_largest_board_majority_and_small_board_coverage(
    stages,
):
    config = priority_profile()
    with pytest.raises(ConfigError, match="ring-10 majority"):
        replace(
            config,
            orchestration=replace(
                config.orchestration,
                ring_mixture=replace(
                    config.orchestration.ring_mixture, step_weights=stages
                ),
            ),
        )


def test_priority_objective_allows_future_tuning_within_its_declared_objective():
    config = priority_profile()
    mixture = replace(
        config.orchestration.ring_mixture,
        step_weights=(
            RingWeightStage(0, (0.05, 0.1, 0.15, 0.7)),
            RingWeightStage(100, (0.01, 0.01, 0.01, 0.97)),
        ),
    )
    validate_continuous_config(
        replace(
            config, orchestration=replace(config.orchestration, ring_mixture=mixture)
        )
    )


@pytest.mark.parametrize(
    "arena_changes",
    [
        {"rings": (4, 6, 8, 10)},
        {"balanced_cells": False},
        {"required_regression_rings": None},
        {"required_regression_rings": (10,)},
        {"per_ring_regression_floor_elo": {10: -100.0}},
    ],
)
def test_priority_objective_rejects_alternate_promotion_or_guard_schedules(
    arena_changes,
):
    config = priority_profile()
    with pytest.raises(ConfigError, match="balanced ring-10 promotion"):
        replace(config, arena=replace(config.arena, **arena_changes))


@pytest.mark.parametrize("field", ["pie_classic_share", "handicap_classic_share"])
def test_priority_objective_rejects_unequal_modes(field):
    config = priority_profile()
    with pytest.raises(ConfigError, match="all six modes equally"):
        replace(
            config,
            selfplay=replace(
                config.selfplay,
                variants=replace(config.selfplay.variants, **{field: 0.8}),
            ),
        )


def test_priority_objective_requires_replay_to_follow_the_training_mix():
    config = priority_profile()
    for learner in (
        replace(config.learner, use_ring_mixture_curriculum=False),
        replace(config.learner, segment_quotas=None),
        replace(config.learner, segment_quotas={"standard": 1.0}),
    ):
        with pytest.raises(ConfigError, match="ring10_priority"):
            replace(config, learner=learner)
    with pytest.raises(ConfigError, match="ring-stratified"):
        replace(config, data=replace(config.data, ring_stratified=False))
    with pytest.raises(ConfigError, match="ring-10 default"):
        replace(config, selfplay=replace(config.selfplay, rings=4))


def test_generalist_service_cannot_silently_drop_small_board_promotion():
    config = priority_profile()
    config = replace(
        config,
        orchestration=replace(config.orchestration, training_objective="generalist"),
    )
    with pytest.raises(ValueError, match="generalist balanced promotion"):
        validate_continuous_config(config)


def test_compatibility_epochs_never_erase_new_objective_authority():
    config = priority_profile()
    for payload in compatible_config_epoch_payloads(config.as_dict()):
        assert payload["orchestration"]["training_objective"] == "ring10_priority"
        assert payload["arena"]["rings"] == (10,)
        assert payload["arena"]["balanced_cells"] is True
