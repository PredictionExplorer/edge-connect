"""Native end-to-end self-play for every rule variant."""

from __future__ import annotations

from dataclasses import replace
from collections import Counter
from pathlib import Path

import pytest

from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.model import GraphResTNet, ModelConfig, model_parameter_count
from startrain.native import validate_native_module
from startrain.replay import collate_replay_samples
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity
from startrain.selfplay import (
    STANDARD_VARIANT,
    GameVariant,
    SelfPlayActor,
    SelfPlayConfig,
    SelfPlayIdentity,
    VariantMixtureConfig,
)

MODEL_IDENTITY = "sha256-" + "c" * 64


def evaluator() -> GraphInferenceAdapter:
    return GraphInferenceAdapter(
        GraphResTNet(ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)),
        config=InferenceConfig(precision="fp32"),
        model_version=MODEL_IDENTITY,
        model_step=0,
        model_identity=MODEL_IDENTITY,
    )


def test_game_variant_labels_segments_and_parsing() -> None:
    assert STANDARD_VARIANT.label == "double" and STANDARD_VARIANT.segment == "standard"
    classic = GameVariant(mode="classic")
    assert classic.label == "classic" and classic.segment == "classic"
    handicap = GameVariant(mode="double", handicap=5)
    assert handicap.label == "handicap-5-double" and handicap.segment == "handicap"
    pie = GameVariant(mode="classic", pie=True)
    assert pie.label == "pie-classic" and pie.segment == "pie"
    for variant in (STANDARD_VARIANT, classic, handicap, pie):
        assert GameVariant.parse(variant.label) == variant
    with pytest.raises(ValueError, match="pie"):
        GameVariant(mode="double", handicap=2, pie=True)
    with pytest.raises(ValueError, match="handicap"):
        GameVariant(mode="double", handicap=10)
    with pytest.raises(ValueError, match="unknown variant"):
        GameVariant.parse("triple")


def test_variant_mixture_draws_follow_the_fractions() -> None:
    mixture = VariantMixtureConfig(enabled=True)
    counts: dict[str, int] = {}
    handicaps: set[int] = set()
    modes: set[str] = set()
    for seed in range(4000):
        # Spread the unit interval and the secondary draw independently.
        roll = (seed * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        variant = mixture.draw(roll)
        counts[variant.segment] = counts.get(variant.segment, 0) + 1
        if variant.segment == "handicap":
            handicaps.add(variant.handicap)
        if variant.segment == "pie":
            modes.add(variant.mode)
    total = sum(counts.values())
    assert abs(counts["standard"] / total - 0.45) < 0.05
    assert abs(counts["classic"] / total - 0.25) < 0.05
    assert abs(counts["handicap"] / total - 0.20) < 0.05
    assert abs(counts["pie"] / total - 0.10) < 0.05
    assert handicaps == set(range(2, 10))
    assert modes == {"classic", "double"}
    assert VariantMixtureConfig().draw(12345) == STANDARD_VARIANT
    assert mixture.pda_for_handicap(1) == 0
    assert mixture.pda_for_handicap(2) == 1
    assert mixture.pda_for_handicap(5) == 2
    assert mixture.pda_for_handicap(9) == 3
    with pytest.raises(ValueError, match="sum to one"):
        VariantMixtureConfig(enabled=True, standard=0.9)
    with pytest.raises(ValueError, match="handicap_pda"):
        VariantMixtureConfig(handicap_pda=(1, 2))
    with pytest.raises(ValueError, match="segment"):
        VariantMixtureConfig(score_utility_weight_by_segment={"bogus": 0.1})


@pytest.mark.parametrize("share", [0.0, 0.3, 0.5, 1.0])
def test_handicap_modes_are_independent_of_size_and_preserve_legacy_draws(
    share,
) -> None:
    legacy = VariantMixtureConfig(enabled=True)
    mixed = replace(legacy, handicap_classic_share=share)
    counts: Counter[tuple[int, str]] = Counter()
    for seed in range(40_000):
        roll = (seed * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        before, after = legacy.draw(roll), mixed.draw(roll)
        assert mixed.draw(roll) == after
        if before.segment != "handicap":
            assert after == before
            continue
        assert before.mode == "double"
        assert after.handicap == before.handicap
        assert after.segment == "handicap" and not after.pie
        counts[after.handicap, after.mode] += 1
    for handicap in range(2, 10):
        classic = counts[handicap, "classic"]
        total = classic + counts[handicap, "double"]
        assert total > 800
        assert classic / total == pytest.approx(share, abs=0.02)


@pytest.mark.parametrize("share", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_handicap_classic_share_rejects_invalid_values(share) -> None:
    with pytest.raises(ValueError, match="handicap_classic_share"):
        VariantMixtureConfig(handicap_classic_share=share)


def test_playout_budgets_keep_the_doubling_ratio_inside_the_caps() -> None:
    config = SelfPlayConfig(
        rings=6,
        fast_simulations=32,
        full_simulations=256,
        simulation_reference_rings=6,
    )
    assert config.playout_budgets(simulations=32, pda=0) == (32, 32)
    assert config.playout_budgets(simulations=32, pda=1) == (64, 32)
    assert config.playout_budgets(simulations=32, pda=3) == (256, 32)
    assert config.playout_budgets(simulations=256, pda=1) == (256, 128)
    assert config.playout_budgets(simulations=256, pda=3) == (256, 32)
    weighted = replace(
        config,
        score_utility_weight=0.05,
        variants=VariantMixtureConfig(
            enabled=True, score_utility_weight_by_segment={"handicap": 0.25}
        ),
        handicap=4,
    )
    assert weighted.effective_score_utility_weight() == 0.25
    assert config.effective_score_utility_weight() == 0.0
    assert config.with_variant(GameVariant(mode="classic")).variant.label == "classic"


@pytest.mark.native
@pytest.mark.parametrize(
    ("variant", "expected_pda"),
    [
        (GameVariant(mode="classic"), (0, 0)),
        (GameVariant(mode="double", handicap=4), (-2, 2)),
        (GameVariant(mode="classic", handicap=4), (-2, 2)),
        (GameVariant(mode="double", pie=True), (0, 0)),
        (GameVariant(mode="classic", pie=True), (0, 0)),
    ],
)
def test_native_variant_games_complete_with_variant_provenance(
    tmp_path, variant: GameVariant, expected_pda: tuple[int, int]
) -> None:
    native = pytest.importorskip("star_native")
    validate_native_module(native)
    config = replace(
        SelfPlayConfig.cpu_smoke(seed=7),
        batch_size=4,
        games=4,
        full_simulations=4,
        max_considered=4,
        variants=VariantMixtureConfig(enabled=True, swap_dead_zone=0.0),
    ).with_variant(variant)
    identity = RunIdentity(tmp_path / "run.json", "run-variants", "family-variants", 1)
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor-variants")
        actor = SelfPlayActor(
            native,
            evaluator(),
            store,
            config,
            SelfPlayIdentity(
                identity.run_id,
                identity.generation_family,
                "actor-variants",
                generation,
            ),
        )
        summaries = actor.run()
        assert len(summaries) == 4
        assert {summary.variant for summary in summaries} == {variant.label}
        assert {(summary.pda_seat0, summary.pda_seat1) for summary in summaries} == {
            expected_pda
        }
        metrics = actor.metrics_snapshot()
        if variant.pie:
            assert metrics.pie_decisions == 4
            assert metrics.pie_swaps == sum(summary.swapped for summary in summaries)
        else:
            assert metrics.pie_decisions == 0
            assert not any(summary.swapped for summary in summaries)
        assert metrics.asymmetric_games == (4 if variant.handicap >= 2 else 0)
        samples = store.load_recent_samples(
            sample_window=4096,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=0,
            max_model_lag_steps=0,
        )
        assert samples
        assert {sample.variant_label for sample in samples} == {variant.label}
        assert all(sample.history_known for sample in samples)
        assert all(
            f"variant={variant.label}" in sample.search_provenance for sample in samples
        )
        if variant.handicap >= 2:
            openings = [sample for sample in samples if sample.opening]
            assert openings
            assert {sample.moves_left for sample in openings} == set(
                range(1, variant.handicap + 1)
            )
            # Player 0 holds the handicap and sees the negated advantage.
            assert {sample.pda for sample in samples if sample.to_move == 0} == {-2}
            assert {sample.pda for sample in samples if sample.to_move == 1} == {2}
            # The disadvantaged side searched with the smaller budget.
            full = config.simulation_budget(full=True)
            advantaged, disadvantaged = config.playout_budgets(simulations=full, pda=2)
            assert advantaged > disadvantaged
            assert {
                sample.search_provenance.split("simulations=")[1].split(":")[0]
                for sample in samples
                if sample.to_move == 0
            } == {str(disadvantaged)}
            assert {
                sample.search_provenance.split("simulations=")[1].split(":")[0]
                for sample in samples
                if sample.to_move == 1
            } == {str(advantaged)}
        if variant.pie:
            pending = [sample for sample in samples if sample.opening]
            assert pending and all(
                sample.to_position().pie_pending for sample in pending
            )
            responder = [sample for sample in samples if sample.swap_available]
            assert len(responder) == 4
            swapped_games = {
                sample.game_id
                for sample in samples
                if "swap=taken" in sample.search_provenance
            }
            assert len(swapped_games) == sum(summary.swapped for summary in summaries)
            assert any(sample.swapped for sample in samples) == bool(swapped_games)
        recent = store.recent_shards(
            sample_window=4096,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )
        assert {record.variant for record in recent} == {variant.label}
        assert {record.segment for record in recent} == {variant.segment}
        batch = collate_replay_samples(samples[:8])
        assert batch.inputs.global_features.shape[1] == 25


def test_yaml_variant_mixture_flows_into_selfplay_and_learner_quotas(tmp_path) -> None:
    from startrain.config import ConfigError, load_config

    configs = Path(__file__).resolve().parents[1] / "configs"
    source = (configs / "small.yaml").read_text(encoding="utf-8")
    mixed = tmp_path / "mixed.yaml"
    mixed.write_text(
        source.replace(
            "selfplay:\n",
            "selfplay:\n"
            "  variants:\n"
            "    enabled: true\n"
            "    standard: 0.45\n"
            "    classic: 0.25\n"
            "    handicap: 0.2\n"
            "    pie: 0.1\n"
            "    handicap_pda: [1, 1, 2, 2, 2, 3, 3, 3]\n"
            "    pda_magnitudes: [1, 2]\n"
            "    score_utility_weight_by_segment:\n"
            "      handicap: 0.25\n",
            1,
        ),
        encoding="utf-8",
    )
    experiment = load_config(mixed)
    assert experiment.selfplay.variants.enabled
    assert experiment.selfplay.variants.handicap_pda == (1, 1, 2, 2, 2, 3, 3, 3)
    assert experiment.selfplay.variants.score_utility_weight_by_segment == {
        "handicap": 0.25
    }
    assert experiment.learner.segment_quotas == {
        "standard": 0.45,
        "classic": 0.25,
        "handicap": 0.2,
        "pie": 0.1,
    }
    baseline = load_config(configs / "small.yaml")
    assert not baseline.selfplay.variants.enabled
    assert baseline.learner.segment_quotas is None

    narrow = tmp_path / "narrow.yaml"
    narrow.write_text(
        mixed.read_text(encoding="utf-8").replace(
            "game:\n",
            "game:\n  variants:\n    modes: [double]\n    pie_allowed: false\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="family excludes"):
        load_config(narrow)

    mismatched = tmp_path / "mismatched.yaml"
    mismatched.write_text(
        source.replace("selfplay:\n", "selfplay:\n  mode: classic\n", 1),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="standard variant"):
        load_config(mismatched)

    bogus = tmp_path / "bogus.yaml"
    bogus.write_text(
        source.replace(
            "learner:\n", "learner:\n  segment_quotas:\n    bogus: 1.0\n", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown replay segment"):
        load_config(bogus)


def test_variant_stage_profiles_validate_and_migrate(tmp_path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.migrate_continuous_profile import (
        _ALLOWED_PROFILE_PATHS,
        _profile_diffs,
    )
    from scripts.validate_continuous_profile import validate_continuous_config
    from startrain.config import load_config

    configs = Path(__file__).resolve().parents[1] / "configs"
    stage_a = load_config(configs / "h100-8gpu-variant-stage-a.yaml")
    stage_b = load_config(configs / "h100-8gpu-variant-stage-b.yaml")
    assert stage_a.model.relational_bias and stage_a.model.adaln_hidden == 32
    assert stage_a.model.feature_schema_version == 4
    assert (stage_a.model.width, stage_a.model.rrt_groups) == (384, 8)
    assert model_parameter_count(stage_a.model) == 17_402_775
    assert stage_b.model == stage_a.model
    assert stage_a.loss.uses_teacher
    assert not stage_a.selfplay.variants.enabled
    assert stage_a.learner.segment_quotas is None
    assert not stage_a.orchestration.autonomous.enabled
    assert stage_a.orchestration.run_id == "variant-network"
    validate_continuous_config(stage_a)

    assert stage_b.selfplay.variants.enabled
    assert stage_a.selfplay.variants.handicap_classic_share == 0.0
    assert stage_a.arena.segment_handicap_classic_share == 0.0
    assert stage_b.selfplay.variants.handicap_classic_share == 0.5
    assert stage_b.selfplay.variants.pie_classic_share == 0.5
    assert stage_b.arena.segment_handicap_classic_share == 0.5
    for step in (0, 120_001, 150_000, 500_000, 10**9):
        assert stage_b.orchestration.ring_mixture.weights_for_step(step) == (
            0.25,
            0.25,
            0.25,
            0.25,
        )
    assert stage_b.selfplay.variants.segment_fractions == {
        "standard": 1 / 6,
        "classic": 1 / 6,
        "handicap": 1 / 3,
        "pie": 1 / 3,
    }
    category_counts: Counter[tuple[str, str]] = Counter()
    for seed in range(40_000):
        roll = (seed * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        variant = stage_b.selfplay.variants.draw(roll)
        category_counts[variant.segment, variant.mode] += 1
    assert len(category_counts) == 6
    assert all(
        count / 40_000 == pytest.approx(1 / 6, abs=0.005)
        for count in category_counts.values()
    )
    assert stage_b.learner.segment_quotas == stage_b.selfplay.variants.segment_fractions
    assert set(stage_b.arena.segment_pairs_per_ring) == {"classic", "handicap", "pie"}
    assert stage_b.arena.segment_handicap_pda == stage_b.selfplay.variants.handicap_pda
    validate_continuous_config(stage_b)
    legacy = load_config(configs / "h100-8gpu-autonomous.yaml")
    with pytest.raises(ValueError, match="ring-10-weighted"):
        validate_continuous_config(
            replace(
                legacy,
                orchestration=replace(
                    legacy.orchestration,
                    ring_mixture=replace(
                        legacy.orchestration.ring_mixture,
                        step_weights=stage_b.orchestration.ring_mixture.step_weights,
                    ),
                ),
            )
        )

    # Stage A -> Stage B is a legal mid-run migration: only mixture, quota,
    # ring allocation, and arena-segment paths differ.
    differences = list(_profile_diffs(stage_a.as_dict(), stage_b.as_dict()))
    assert differences
    assert [
        ".".join(path)
        for path, _, _ in differences
        if path not in _ALLOWED_PROFILE_PATHS
    ] == []

    # The validator refuses a mixture whose learner quotas or arena guards drift.
    with pytest.raises(ValueError, match="segment_quotas"):
        validate_continuous_config(
            replace(
                stage_b,
                learner=replace(stage_b.learner, segment_quotas={"standard": 1.0}),
            )
        )
    with pytest.raises(ValueError, match="arena segment pairs"):
        validate_continuous_config(
            replace(
                stage_b,
                arena=replace(
                    stage_b.arena,
                    segment_pairs_per_ring={"classic": 5},
                    segment_regression_floor_elo={},
                ),
            )
        )
    with pytest.raises(ValueError, match="arena handicap modes"):
        validate_continuous_config(
            replace(
                stage_b,
                arena=replace(stage_b.arena, segment_handicap_classic_share=0.0),
            )
        )


@pytest.mark.parametrize("mode", ["classic", "double"])
def test_handicap_mixture_and_arena_respect_the_allowed_mode_family(mode) -> None:
    from startrain.config import ConfigError, VariantRulesConfig, load_config

    base = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    family = VariantRulesConfig(modes=(mode,))
    share = float(mode == "classic")
    game = replace(base.game, mode=mode, variants=family)
    mixture = VariantMixtureConfig(
        enabled=True,
        standard=0,
        classic=0,
        handicap=1,
        pie=0,
        handicap_classic_share=share,
    )
    selfplay = replace(base.selfplay, mode=mode, variants=mixture)
    arena = replace(
        base.arena,
        segment_pairs_per_ring={"handicap": 2},
        segment_handicap_classic_share=share,
    )
    supported = replace(base, game=game, selfplay=selfplay, arena=arena)
    assert supported.selfplay.variants.draw(0).mode == mode
    with pytest.raises(ConfigError, match="selfplay.variants handicap"):
        replace(
            supported,
            selfplay=replace(
                selfplay, variants=replace(mixture, handicap_classic_share=0.5)
            ),
        )
    with pytest.raises(ConfigError, match="arena handicap variants"):
        replace(
            supported,
            arena=replace(arena, segment_handicap_classic_share=0.5),
        )
