"""Arena mixture segments: variant-carrying pairs and veto-on-regress floors."""

from __future__ import annotations

import pytest

from startrain.arena import (
    ARENA_RESULT_SCHEMA_VERSION,
    ArenaGame,
    ArenaPair,
    ArenaRunner,
    segment_floor_assessment,
    summarize_arena_pairs,
    summarize_completed_arena_pairs,
)
from startrain.config import ArenaConfig, ConfigError
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.model import GraphResTNet, ModelConfig
from startrain.native import validate_native_module
from startrain.selfplay import GameVariant


def _pairs(
    ring: int,
    count: int,
    outcomes: tuple[int, int],
    *,
    variant: str = "double",
    start: int = 0,
) -> list[ArenaPair]:
    return [
        ArenaPair(
            ring=ring,
            pair=start + index,
            opening_seed=index,
            opening_action=None,
            forced_opening=False,
            outcomes=outcomes,
            variant=variant,
            segment=GameVariant.parse(variant).segment,
        )
        for index in range(count)
    ]


def test_arena_pairs_and_games_validate_variant_metadata() -> None:
    with pytest.raises(ValueError, match="segment disagrees"):
        ArenaPair(
            ring=4,
            pair=0,
            opening_seed=0,
            opening_action=None,
            forced_opening=False,
            outcomes=(1, 1),
            variant="classic",
            segment="standard",
        )
    with pytest.raises(ValueError, match="pie game"):
        ArenaGame(
            ring=4,
            pair=0,
            candidate_player=0,
            opening_seed=0,
            opening_action=None,
            forced_opening=False,
            winner=0,
            outcome=1,
            searched_moves=1,
            swapped=True,
        )
    game = ArenaGame(
        ring=4,
        pair=0,
        candidate_player=0,
        opening_seed=0,
        opening_action=None,
        forced_opening=False,
        winner=0,
        outcome=1,
        searched_moves=1,
        variant="pie-double",
        segment="pie",
        swapped=True,
    )
    assert game.segment == "pie"


def test_arena_config_validates_segments() -> None:
    with pytest.raises(ConfigError, match="segment_pairs_per_ring"):
        ArenaConfig(segment_pairs_per_ring={"standard": 2})
    with pytest.raises(ConfigError, match="segment_pairs_per_ring"):
        ArenaConfig(segment_pairs_per_ring={"classic": 0})
    with pytest.raises(ConfigError, match="segment_regression_floor_elo"):
        ArenaConfig(segment_regression_floor_elo={"classic": -50.0})
    with pytest.raises(ConfigError, match="segment_handicaps"):
        ArenaConfig(segment_handicaps=(1,))
    with pytest.raises(ConfigError, match="swap_dead_zone"):
        ArenaConfig(swap_dead_zone=1.5)


def test_segment_floors_veto_only_proven_regressions() -> None:
    config = ArenaConfig(
        rings=(4,),
        pairs_per_ring=2,
        minimum_pairs_per_ring=2,
        max_pairs_per_ring=200,
        bootstrap_samples=200,
        segment_pairs_per_ring={"classic": 2, "handicap": 2},
        segment_regression_floor_elo={"classic": -50.0},
    )
    standard = _pairs(4, 40, (1, 1))
    # Classic collapses completely: the candidate loses every classic pair.
    classic = _pairs(4, 30, (-1, -1), variant="classic")
    summary = summarize_arena_pairs(standard + classic, config)
    promotion = summary["promotion"]
    assert isinstance(promotion, dict)
    assert promotion["decision"] == "reject_ring_regression"
    assert promotion["regression_source"] == "segment"
    assert promotion["vetoed_decision"] == "promote"
    assert promotion["segment_vetoes"] == ["classic"]
    floors = promotion["segment_floors"]
    assert isinstance(floors, dict)
    assert floors["classic"]["status"] == "regress"
    assert floors["classic"]["floor_elo"] == -50.0
    assert floors["handicap"]["status"] == "continue"
    assert floors["handicap"]["pairs"] == 0
    assert floors["handicap"]["floor_elo"] == config.regression_floor_elo
    assert set(summary["per_segment"]) == {"classic"}
    # The standard-only aggregate is untouched by the segment pairs.
    assert summary["aggregate"]["pairs"] == 40

    # A merely mediocre segment never vetoes.
    balanced = _pairs(4, 30, (1, -1), variant="classic")
    summary = summarize_arena_pairs(standard + balanced, config)
    promotion = summary["promotion"]
    assert isinstance(promotion, dict)
    assert promotion["decision"] == "promote"
    assert promotion["segment_vetoes"] == []
    assert "regression_source" not in promotion

    # Incomplete ring coverage still reports the segment floors and vetoes.
    partial_config = ArenaConfig(
        rings=(4, 6),
        pairs_per_ring=2,
        minimum_pairs_per_ring=2,
        max_pairs_per_ring=200,
        bootstrap_samples=200,
        segment_pairs_per_ring={"classic": 2},
    )
    summary = summarize_completed_arena_pairs(standard + classic, partial_config)
    promotion = summary["promotion"]
    assert isinstance(promotion, dict)
    assert promotion["reason"] == "incomplete_ring_coverage"
    assert promotion["decision"] == "reject_ring_regression"
    assert promotion["regression_source"] == "segment"
    assert promotion["vetoed_decision"] == "continue"

    assessment = segment_floor_assessment({"classic": classic}, config)
    assert assessment["classic"]["vetoes_promotion"] is True
    assert assessment["classic"]["per_ring_pairs"] == {"4": 30}


@pytest.mark.native
def test_native_arena_plays_every_segment_with_variant_provenance() -> None:
    native = pytest.importorskip("star_native")
    validate_native_module(native)
    identity = "sha256-" + "d" * 64
    evaluator = GraphInferenceAdapter(
        GraphResTNet(ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)),
        config=InferenceConfig(precision="fp32"),
        model_version=identity,
        model_step=0,
        model_identity=identity,
    )
    config = ArenaConfig(
        rings=(4,),
        pairs_per_ring=2,
        simulations=2,
        max_considered=2,
        regression_floor_elo=-2_500.0,
        segment_pairs_per_ring={"classic": 2, "handicap": 2, "pie": 2},
        segment_handicaps=(3, 6),
        swap_dead_zone=0.0,
        bootstrap_samples=200,
    )
    result = ArenaRunner(
        native_module=native,
        candidate=evaluator,
        baseline=evaluator,
        config=config,
    ).run()
    assert result["schema_version"] == ARENA_RESULT_SCHEMA_VERSION
    pairs = result["pairs"]
    assert isinstance(pairs, list)
    by_segment: dict[str, int] = {}
    for pair in pairs:
        by_segment[pair["segment"]] = by_segment.get(pair["segment"], 0) + 1
    assert by_segment == {"standard": 2, "classic": 2, "handicap": 2, "pie": 2}
    games = result["games"]
    assert isinstance(games, list)
    assert len(games) == 16
    handicap_games = [game for game in games if game["segment"] == "handicap"]
    assert {game["variant"] for game in handicap_games} <= {
        "handicap-3-double",
        "handicap-6-double",
    }
    assert all(game["pda"] >= 1 for game in handicap_games)
    assert all(game["pda"] == 0 for game in games if game["segment"] != "handicap")
    pie_games = [game for game in games if game["segment"] == "pie"]
    assert {game["variant"] for game in pie_games} <= {"pie-double", "pie-classic"}
    assert all(not game["swapped"] for game in games if game["segment"] != "pie")
    # Every pair is a complete role reversal within one variant.
    for offset in range(0, len(games), 2):
        first, second = games[offset], games[offset + 1]
        assert first["pair"] == second["pair"]
        assert first["variant"] == second["variant"]
        assert {first["candidate_player"], second["candidate_player"]} == {0, 1}
    search = result["search"]
    assert isinstance(search, dict)
    assert search["pie_rule"] is True
    assert search["segments"] == {
        "standard": 2,
        "classic": 2,
        "handicap": 2,
        "pie": 2,
    }
    promotion = result["promotion"]
    assert isinstance(promotion, dict)
    assert set(promotion["segment_floors"]) == {"classic", "handicap", "pie"}
    assert result["aggregate"]["pairs"] == 2
    assert set(result["per_segment"]) == {"classic", "handicap", "pie"}
