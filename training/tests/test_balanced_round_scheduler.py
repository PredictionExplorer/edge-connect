from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import startrain.promotion as promotion_module
from startrain.arena import ArenaPair, summarize_completed_arena_pairs
from startrain.balanced_evaluation import (
    BALANCED_CATEGORIES,
    balanced_opening_seed,
    cell_variant,
    completed_counts_by_ring,
    pair_key,
)
from startrain.config import ArenaConfig, HistoricalEvaluationConfig, load_config
from startrain.promotion import PromotionSupervisor, _balanced_round_plan

from test_promotion import _promotion_wave_case


def _config(**changes) -> ArenaConfig:
    return ArenaConfig(
        balanced_cells=True,
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        continuation_pairs_per_ring=4,
        max_pairs_per_ring=40,
        **changes,
    )


def _pair(config: ArenaConfig, ring: int, name: str, index: int) -> ArenaPair:
    variant = cell_variant(name, index, config)
    return ArenaPair(
        ring=ring,
        pair=index,
        opening_seed=balanced_opening_seed(config.seed, ring, variant, index),
        opening_action=None,
        forced_opening=False,
        outcomes=(1, -1),
        variant=variant.label,
        segment=variant.segment,
    )


def _pairs(config: ArenaConfig, counts: dict[int, int]) -> list[ArenaPair]:
    return [
        _pair(config, ring, name, index)
        for ring, count in counts.items()
        for index in range(count)
        for name in BALANCED_CATEGORIES
    ]


def _pending(config, accumulated, starts, counts):
    finished = {pair_key(pair) for pair in accumulated}
    return [
        _pair(config, ring, name, index)
        for ring in config.rings
        for index in range(starts[ring], starts[ring] + counts[ring])
        for name in BALANCED_CATEGORIES
        if (ring, name, index) not in finished
    ]


@pytest.mark.parametrize("small_count", [0, 1, 2, 3])
def test_frontloaded_boards_wait_for_the_first_complete_round(small_count):
    config = _config()
    accumulated = _pairs(config, {4: 40, 6: 40, 8: small_count, 10: 0})
    original = list(accumulated)

    starts, counts = _balanced_round_plan(accumulated, config)

    assert starts == {4: 40, 6: 40, 8: small_count, 10: 0}
    assert counts == {4: 0, 6: 0, 8: 4 - small_count, 10: 4}
    assert accumulated == original


@pytest.mark.parametrize(
    ("slowest", "target"),
    [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 6),
        (4, 6),
        (6, 8),
        (7, 8),
        (8, 13),
        (9, 13),
        (12, 13),
        (13, 18),
        (17, 18),
        (18, 20),
        (20, 20),
    ],
)
def test_round_targets_stay_fixed_across_partial_initial_and_continuation_waves(
    slowest, target
):
    config = replace(
        _config(),
        pairs_per_ring=3,
        minimum_pairs_per_ring=8,
        continuation_pairs_per_ring=5,
        max_pairs_per_ring=20,
    )
    accumulated = _pairs(config, {4: 20, 6: 20, 8: 20, 10: slowest})

    starts, counts = _balanced_round_plan(accumulated, config)

    assert starts[10] == slowest
    assert counts == {4: 0, 6: 0, 8: 0, 10: target - slowest}


def test_sparse_disjoint_mode_results_request_only_missing_pairs_in_current_round():
    config = _config()
    accumulated = _pairs(config, {4: 4, 6: 4, 8: 4, 10: 0})
    accumulated.extend(
        _pair(config, 10, name, index)
        for mode, name in enumerate(BALANCED_CATEGORIES)
        for index in range(4)
        if index != mode % 4
    )
    before = {pair_key(pair) for pair in accumulated}

    starts, counts = _balanced_round_plan(accumulated, config)
    pending = _pending(config, accumulated, starts, counts)

    assert starts == {4: 4, 6: 4, 8: 4, 10: 0}
    assert counts == {4: 0, 6: 0, 8: 0, 10: 4}
    assert len(pending) == len(BALANCED_CATEGORIES)
    assert not before.intersection(pair_key(pair) for pair in pending)
    assert completed_counts_by_ring(accumulated + pending, config) == {
        ring: 4 for ring in config.rings
    }


def test_completed_modes_are_not_extended_while_other_modes_finish_the_round():
    config = _config(rings=(10,))
    accumulated = [
        _pair(config, 10, BALANCED_CATEGORIES[0], index) for index in range(40)
    ]

    starts, counts = _balanced_round_plan(accumulated, config)
    pending = _pending(config, accumulated, starts, counts)

    assert starts == {10: 0} and counts == {10: 4}
    assert len(pending) == 5 * 4
    assert all(pair_key(pair)[1] != BALANCED_CATEGORIES[0] for pair in pending)


@pytest.mark.parametrize("rings", [(4, 6, 8, 10), (10,)])
def test_repeated_short_sessions_finish_every_round_before_extending_any_cell(rings):
    config = replace(_config(rings=rings), max_pairs_per_ring=12)
    accumulated = []
    counts = {ring: 1 for ring in rings}
    sessions = 0
    while any(counts.values()):
        starts, counts = _balanced_round_plan(accumulated, config)
        pending = _pending(config, accumulated, starts, counts)
        if not pending:
            break
        target = (min(starts.values()) // 4 + 1) * 4
        assert all(pair.pair < target for pair in pending)
        # Repeatedly interrupt inside individual modes and handicap cycles.
        completed = pending[: (1, 5, 7)[sessions % 3]]
        accumulated.extend(completed)
        assert len({pair_key(pair) for pair in accumulated}) == len(accumulated)
        sessions += 1
        assert sessions < 300
    assert sessions > 10
    assert completed_counts_by_ring(accumulated, config) == {ring: 12 for ring in rings}
    assert len(accumulated) == len(rings) * len(BALANCED_CATEGORIES) * 12


def test_candidate_planner_uses_the_same_balanced_round_as_the_shared_helper():
    config = _config()
    supervisor = object.__new__(PromotionSupervisor)
    supervisor.experiment = replace(
        load_config(Path(__file__).parents[1] / "configs" / "small.yaml"),
        arena=config,
    )
    accumulated = _pairs(config, {4: 8, 6: 7, 8: 6, 10: 5})
    assert supervisor._wave_plan(accumulated) == (
        {4: 8, 6: 7, 8: 6, 10: 5},
        {4: 0, 6: 1, 8: 2, 10: 3},
    )


@pytest.mark.parametrize("kind", ["candidate", "historical"])
def test_supervisor_slices_preserve_evidence_and_complete_balanced_rounds(
    tmp_path, monkeypatch, kind
):
    case = _promotion_wave_case(tmp_path, monkeypatch)
    subject = case.supervisor
    subject.experiment = replace(
        case.experiment,
        arena=replace(_config(), max_pairs_per_ring=12),
        orchestration=replace(
            case.experiment.orchestration,
            historical_evaluation=HistoricalEvaluationConfig(
                enabled=True, pairs_per_ring=4, max_pairs_per_ring=12
            ),
        ),
    )
    stopped = False
    requested_targets = []

    class InterruptedArena:
        def __init__(self, **options):
            self.config = options["config"]

        def run(self, *, pair_starts, pair_counts, previous_pairs, **_options):
            nonlocal stopped
            requested_targets.append(
                {
                    ring: pair_starts[ring] + pair_counts[ring]
                    for ring in self.config.rings
                }
            )
            completed = _pending(self.config, previous_pairs, pair_starts, pair_counts)[
                :3
            ]
            assert completed
            stopped = True
            return {
                "candidate": case.candidate.model_identity,
                "baseline": case.champion.model_identity,
                "pairs": [asdict(pair) for pair in completed],
                "games": [
                    {
                        "ring": pair.ring,
                        "variant": pair.variant,
                        "pair": pair.pair,
                        "candidate_player": player,
                        "outcome": pair.outcomes[player],
                    }
                    for pair in completed
                    for player in (0, 1)
                ],
                **summarize_completed_arena_pairs(
                    previous_pairs + completed, self.config
                ),
            }

    monkeypatch.setattr(promotion_module, "ArenaRunner", InterruptedArena)
    path = (
        subject._result_path(case.candidate, case.champion)
        if kind == "candidate"
        else tmp_path / "arena/history.json"
    )
    previous = None
    preserved = {}
    # Three pairs per interrupted session require 32 sessions per full round.
    for session in range(64):
        stopped = False
        if kind == "candidate":
            evaluated, _ = subject._evaluate_candidate_session(
                candidate=case.candidate,
                champion=case.champion,
                previous=previous,
                stop_requested=lambda: stopped,
                progress=None,
                once=True,
            )
        else:
            evaluated = subject._evaluate_historical_waves(
                candidate=case.candidate,
                baseline=case.champion,
                previous=previous,
                result_path=path,
                stop_requested=lambda: stopped,
                progress=None,
                once=True,
                yield_to_candidates=False,
            )
        assert evaluated == 1
        previous = json.loads(path.read_text())
        current = {
            pair_key(pair): asdict(pair)
            for pair in subject._pairs_from_result(previous)
        }
        assert preserved.items() <= current.items()
        preserved = current
        assert len(previous["pairs"]) == (session + 1) * 3
        assert len(previous["games"]) == (session + 1) * 6
        target = 4 if session < 32 else 8
        assert requested_targets[-1] == {ring: target for ring in (4, 6, 8, 10)}
        if kind == "candidate":
            assert previous["evaluation_metrics"]["requested_pairs"] == (
                96 - (session % 32) * 3
            )
    assert completed_counts_by_ring(
        subject._pairs_from_result(previous), subject.experiment.arena
    ) == {ring: 8 for ring in (4, 6, 8, 10)}
