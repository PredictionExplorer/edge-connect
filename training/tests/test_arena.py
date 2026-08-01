from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from startrain.arena import (
    ARENA_RESULT_SCHEMA_VERSION,
    WEIGHTED_OBSERVATION_MODEL,
    ArenaPair,
    ArenaRunner,
    ArenaSearchBudget,
    BinaryResults,
    bounded_confidence_sequence,
    bounded_log_e_value,
    internal_elo_target_assessment,
    pair_confidence_sequence,
    promotion_assessment,
    summarize_arena_pairs,
    summarize_binary_results,
    summarize_completed_arena_pairs,
    summarize_pairs,
    wilson_interval,
)
from startrain.config import ArenaConfig, ConfigError
from startrain.inference import InferenceResponse
from startrain.native import BITBOARD_WORDS


class FakeRequests:
    def __init__(self, states: object) -> None:
        self.tokens = [1]
        self.states = states
        self.legal_offsets = [0, 2]
        self.legal_actions = [0, 1]

    def __len__(self) -> int:
        return 1


class RoleEvaluator:
    def __init__(self, model_version: str, selected_action: int) -> None:
        self.model_version = model_version
        self.selected_action = selected_action
        self.evaluator_calls = 0
        self.evaluator_rows = 0

    def evaluate(self, requests: FakeRequests) -> InferenceResponse:
        self.evaluator_calls += 1
        self.evaluator_rows += len(requests)
        logits = [0.0, 0.0]
        logits[self.selected_action] = 1.0
        return InferenceResponse([1], [0.0], [0, 2], logits)


class FakeStateBatch:
    tied = False

    def __init__(self, rings: int, batch_size: int) -> None:
        assert rings == 4 and batch_size == 1
        self.terminal = False
        self.to_move = 0
        self.winner = -1
        self.search_started = False

    def apply_many(self, indices: list[int], actions: list[int]) -> None:
        assert indices == [0] and len(actions) == 1
        if not self.search_started:
            return
        self.winner = -1 if self.tied else actions[0]
        self.terminal = True

    def data(self) -> object:
        return SimpleNamespace(terminal=[self.terminal], to_move=[self.to_move])

    def score_data(self) -> object:
        return SimpleNamespace(winner=[self.winner])


class FakeSearchBatch:
    def __init__(self, states: FakeStateBatch, **_options: object) -> None:
        self.states = states
        self.states.search_started = True
        self.selected = -1
        self.initialized = False

    def root_requests(self) -> FakeRequests:
        return FakeRequests(self.states.data())

    def initialize_roots(
        self,
        _tokens: list[int],
        _values: list[float],
        _offsets: list[int],
        logits: list[float],
    ) -> None:
        self.selected = max(range(len(logits)), key=logits.__getitem__)
        self.initialized = True

    def is_done(self) -> bool:
        return self.initialized

    def next_requests(self) -> object:
        raise AssertionError("fake search completes at its root")

    def submit(self, *_buffers: object) -> None:
        raise AssertionError("fake search has no leaves")

    def results(self) -> object:
        return SimpleNamespace(terminal=[False], selected_actions=[self.selected])


class FakeNative:
    StateBatch = FakeStateBatch
    SearchBatch = FakeSearchBatch


def arena_config(**overrides: object) -> ArenaConfig:
    values = {
        "rings": (4,),
        "pairs_per_ring": 2,
        "simulations": 1,
        "max_considered": 2,
        "minimum_pairs_per_ring": 2,
        "max_pairs_per_ring": 4,
        "bootstrap_samples": 200,
        "regression_floor_elo": -2_500.0,
    }
    values.update(overrides)
    return ArenaConfig(**values)


def weighted_arena_config(**overrides: object) -> ArenaConfig:
    values = {
        "rings": (4, 6, 8, 10),
        "pairs_per_ring": 2,
        "simulations": 1,
        "max_considered": 2,
        "minimum_pairs_per_ring": 2,
        "max_pairs_per_ring": 200,
        "bootstrap_samples": 200,
        "regression_floor_elo": 0.0,
        "promotion_pair_ratios": {4: 1, 6: 1, 8: 1, 10: 7},
        "required_regression_rings": (),
        "weighted_initial_blocks": 1,
        "weighted_continuation_blocks": 1,
        "weighted_max_blocks": 50,
    }
    values.update(overrides)
    return ArenaConfig(**values)


def scored_pair(ring: int, pair: int, score_rate: float) -> ArenaPair:
    outcomes = {
        0.0: (-1, -1),
        0.5: (1, -1),
        1.0: (1, 1),
    }[score_rate]
    return ArenaPair(ring, pair, pair, 0, True, outcomes)


def weighted_pairs(
    blocks: int,
    *,
    ring_scores: dict[int, float],
) -> dict[int, list[ArenaPair]]:
    ratios = {4: 1, 6: 1, 8: 1, 10: 7}
    return {
        ring: [
            scored_pair(ring, pair, ring_scores[ring])
            for pair in range(blocks * ratios[ring])
        ]
        for ring in ratios
    }


def test_arena_records_only_binary_results() -> None:
    candidate = RoleEvaluator("candidate", selected_action=1)
    baseline = RoleEvaluator("baseline", selected_action=0)
    result = ArenaRunner(
        native_module=FakeNative,
        candidate=candidate,
        baseline=baseline,
        config=arena_config(
            pairs_per_ring=4,
            minimum_pairs_per_ring=4,
            max_pairs_per_ring=4,
            unforced_opening_fraction=0.5,
        ),
    ).run()

    assert result["schema_version"] == ARENA_RESULT_SCHEMA_VERSION
    assert result["aggregate"]["wins"] == 0
    assert result["aggregate"]["losses"] == 8
    assert "draws" not in result["aggregate"]
    assert result["aggregate"]["pair_win_counts"] == {"0": 4, "1": 0, "2": 0}
    assert all(game["outcome"] in (-1, 1) for game in result["games"])
    assert candidate.evaluator_calls == baseline.evaluator_calls
    assert candidate.evaluator_calls > 0
    games = result["games"]
    for index in range(0, len(games), 2):
        first, second = games[index : index + 2]
        assert first["ring"] == second["ring"] == 4
        assert first["pair"] == second["pair"]
        assert first["opening_seed"] == second["opening_seed"]
        assert first["opening_action"] == second["opening_action"]
        assert (first["candidate_player"], second["candidate_player"]) == (0, 1)
    assert any(not pair["forced_opening"] for pair in result["pairs"])
    assert any(pair["forced_opening"] for pair in result["pairs"])
    assert result["search"]["pie_rule"] is False
    assert result["search"]["deterministic"] is True
    assert result["baseline_metadata"]["kind"] == "checkpoint"
    evaluation = result["evaluation_metrics"]
    assert evaluation["candidate_evaluator_calls"] == candidate.evaluator_calls
    assert evaluation["baseline_evaluator_calls"] == baseline.evaluator_calls
    assert evaluation["total_evaluator_rows"] == (
        candidate.evaluator_rows + baseline.evaluator_rows
    )
    assert evaluation["evaluator_rows_per_second"] > 0


def test_arena_stop_returns_only_complete_role_reversed_chunks() -> None:
    stopping = False

    def progress(**details: object) -> None:
        nonlocal stopping
        if details.get("phase") == "arena":
            stopping = True

    result = ArenaRunner(
        native_module=FakeNative,
        candidate=RoleEvaluator("candidate", selected_action=1),
        baseline=RoleEvaluator("baseline", selected_action=0),
        config=arena_config(
            pairs_per_ring=4,
            minimum_pairs_per_ring=4,
            max_pairs_per_ring=4,
            pair_chunk_size=1,
        ),
    ).run(
        progress=progress,
        stop_requested=lambda: stopping,
    )

    assert result["interrupted"] is True
    assert result["evaluation_metrics"]["requested_pairs"] == 4
    assert result["evaluation_metrics"]["completed_pairs"] == 1
    assert [pair["pair"] for pair in result["pairs"]] == [0]
    assert len(result["games"]) == 2
    assert {game["candidate_player"] for game in result["games"]} == {0, 1}
    assert result["promotion"]["decision"] == "continue"


def test_arena_discards_pair_interrupted_during_first_game() -> None:
    stopping = False

    class StoppingEvaluator(RoleEvaluator):
        def evaluate(self, requests: FakeRequests) -> InferenceResponse:
            nonlocal stopping
            response = super().evaluate(requests)
            stopping = True
            return response

    result = ArenaRunner(
        native_module=FakeNative,
        candidate=StoppingEvaluator("candidate", selected_action=1),
        baseline=RoleEvaluator("baseline", selected_action=0),
        config=arena_config(pair_chunk_size=1),
    ).run(stop_requested=lambda: stopping)

    assert result["interrupted"] is True
    assert result["pairs"] == []
    assert result["games"] == []
    assert result["evaluation_metrics"]["completed_pairs"] == 0
    assert result["promotion"]["reason"] == "stopped_before_complete_pair"


def test_arena_rejects_terminal_ties() -> None:
    FakeStateBatch.tied = True
    try:
        with pytest.raises(RuntimeError, match="cannot be tied"):
            ArenaRunner(
                native_module=FakeNative,
                candidate=RoleEvaluator("candidate", 1),
                baseline=RoleEvaluator("baseline", 0),
                config=arena_config(),
            ).run()
    finally:
        FakeStateBatch.tied = False


def test_binary_summary_and_pair_validation() -> None:
    balanced = BinaryResults(wins=50, losses=50)
    lower, upper = wilson_interval(balanced)
    assert lower < 0.5 < upper
    summary = summarize_binary_results(balanced, confidence=0.95)
    assert summary["score_rate"] == 0.5
    assert summary["elo_difference"] == 0.0
    assert "draws" not in summary

    with pytest.raises(ValueError, match="tied"):
        ArenaPair(4, 0, 0, 0, True, (1, 0))
    with pytest.raises(ValueError, match="binary"):
        balanced.record(0)


def test_ring_floor_pass_requires_anytime_valid_evidence() -> None:
    config = arena_config(
        regression_floor_elo=-100.0,
        minimum_pairs_per_ring=20,
        max_pairs_per_ring=40,
    )
    pairs = [ArenaPair(4, pair, pair, 0, True, (1, -1)) for pair in range(20)]

    assessment = promotion_assessment(pairs, {4: pairs}, config)
    ring = assessment["ring_floors"]["4"]

    assert assessment["decision"] == "continue"
    assert ring["status"] == "pass"
    assert ring["passed"] is True
    assert ring["pass_e_value"] >= ring["threshold"]
    assert ring["regression_e_value"] < ring["threshold"]


def test_ring_floor_regression_requires_opposite_one_sided_e_process() -> None:
    config = arena_config(
        beta=0.001,
        regression_floor_elo=0.0,
        minimum_pairs_per_ring=10,
        max_pairs_per_ring=20,
    )
    pairs = [ArenaPair(4, pair, pair, 0, True, (-1, -1)) for pair in range(10)]

    assessment = promotion_assessment(pairs, {4: pairs}, config)
    ring = assessment["ring_floors"]["4"]

    assert assessment["sequential_state"] == "continue"
    assert assessment["decision"] == "reject_ring_regression"
    assert ring["status"] == "regress"
    assert ring["passed"] is False
    assert ring["pass_e_value"] < ring["threshold"]
    assert ring["regression_e_value"] >= ring["threshold"]


def test_ring_floor_uncertainty_at_minimum_continues() -> None:
    config = arena_config(
        regression_floor_elo=-100.0,
        minimum_pairs_per_ring=10,
        max_pairs_per_ring=20,
    )
    pairs = [ArenaPair(4, pair, pair, 0, True, (1, -1)) for pair in range(10)]

    assessment = promotion_assessment(pairs, {4: pairs}, config)
    ring = assessment["ring_floors"]["4"]

    assert assessment["decision"] == "continue"
    assert ring["status"] == "continue"
    assert ring["passed"] is False
    assert ring["pass_e_value"] < ring["threshold"]
    assert ring["regression_e_value"] < ring["threshold"]


def test_promotion_requires_aggregate_alternative_and_every_ring_pass() -> None:
    config = ArenaConfig(
        rings=(4, 6),
        pairs_per_ring=5,
        simulations=1,
        max_considered=2,
        minimum_pairs_per_ring=10,
        max_pairs_per_ring=20,
        bootstrap_samples=200,
        regression_floor_elo=-100.0,
    )
    one_unresolved = {
        4: [ArenaPair(4, pair, pair, 0, True, (1, 1)) for pair in range(10)],
        6: [ArenaPair(6, pair, pair, 0, True, (1, -1)) for pair in range(10)],
    }
    aggregate = [pair for pairs in one_unresolved.values() for pair in pairs]
    unresolved = promotion_assessment(aggregate, one_unresolved, config)

    assert unresolved["sequential_state"] == "accept_alternative"
    assert unresolved["ring_floors"]["4"]["status"] == "pass"
    assert unresolved["ring_floors"]["6"]["status"] == "continue"
    assert unresolved["decision"] == "continue"

    per_ring = {
        ring: [ArenaPair(ring, pair, pair, 0, True, (1, 1)) for pair in range(10)]
        for ring in config.rings
    }
    aggregate = [pair for pairs in per_ring.values() for pair in pairs]
    summary = summarize_pairs(
        aggregate,
        confidence=0.95,
        bootstrap_samples=200,
        seed=1,
    )
    assert summary["pair_win_counts"]["2"] == len(aggregate)
    promoted = promotion_assessment(aggregate, per_ring, config)
    assert promoted["decision"] == "promote"
    assert all(ring["status"] == "pass" for ring in promoted["ring_floors"].values())


def test_unresolved_ring_at_max_pairs_remains_continue_for_supervisor() -> None:
    config = ArenaConfig(
        rings=(4,),
        pairs_per_ring=5,
        simulations=1,
        max_considered=2,
        minimum_pairs_per_ring=10,
        max_pairs_per_ring=10,
        bootstrap_samples=200,
        regression_floor_elo=0.0,
    )
    pairs = [ArenaPair(4, pair, pair, 0, True, (1, -1)) for pair in range(10)]

    assessment = promotion_assessment(pairs, {4: pairs}, config)

    assert len(pairs) == config.max_pairs_per_ring
    assert assessment["ring_floors"]["4"]["status"] == "continue"
    assert assessment["decision"] == "continue"


def test_batched_arena_parallelizes_search_but_serializes_inference() -> None:
    search_barrier: threading.Barrier | None = threading.Barrier(2, timeout=2)
    inference_guard = threading.Lock()
    active_inference = 0
    maximum_active_inference = 0
    inference_threads: set[int] = set()
    search_threads: set[int] = set()

    class BatchRequests:
        def __init__(self, size: int) -> None:
            self.tokens = list(range(size))
            self.states = SimpleNamespace(opening=[False] * size)
            self.legal_offsets = list(range(size + 1))
            self.legal_actions = [0] * size

        def __len__(self) -> int:
            return len(self.tokens)

    class SerializedEvaluator:
        evaluator_calls = 0
        evaluator_rows = 0

        def __init__(self, name: str) -> None:
            self.model_version = name

        def evaluate(self, requests: BatchRequests) -> InferenceResponse:
            nonlocal active_inference, maximum_active_inference
            with inference_guard:
                active_inference += 1
                maximum_active_inference = max(
                    maximum_active_inference, active_inference
                )
                inference_threads.add(threading.get_ident())
            try:
                time.sleep(0.01)
                self.evaluator_calls += 1
                self.evaluator_rows += len(requests)
                return InferenceResponse(
                    tokens=list(requests.tokens),
                    values=[0.0] * len(requests),
                    policy_offsets=list(requests.legal_offsets),
                    policy_logits=[1.0] * len(requests),
                )
            finally:
                with inference_guard:
                    active_inference -= 1

    class BatchStates:
        def __init__(self, rings: int, batch_size: int) -> None:
            assert rings == 4
            self.rings = rings
            self.batch_size = batch_size
            self.terminal = [False] * batch_size
            self.to_move = [0] * batch_size
            self.applied = [0] * batch_size

        @classmethod
        def from_semantic(
            cls,
            rings: int,
            zero_bits: list[int],
            one_bits: list[int],
            to_move: list[int],
            _moves_left: list[int],
            _opening: list[bool],
        ) -> "BatchStates":
            states = cls(rings, len(to_move))
            states.to_move = list(to_move)
            for row in range(len(to_move)):
                start = row * BITBOARD_WORDS
                end = start + BITBOARD_WORDS
                states.applied[row] = int(
                    any(zero_bits[start:end]) or any(one_bits[start:end])
                )
            return states

        def data(self) -> object:
            zero_bits = [0] * (self.batch_size * BITBOARD_WORDS)
            one_bits = [0] * (self.batch_size * BITBOARD_WORDS)
            for row, applied in enumerate(self.applied):
                if applied:
                    zero_bits[row * BITBOARD_WORDS] = 1
            return SimpleNamespace(
                rings=self.rings,
                zero_bits=zero_bits,
                one_bits=one_bits,
                to_move=list(self.to_move),
                moves_left=[1 if value else 2 for value in self.applied],
                opening=[not bool(value) for value in self.applied],
                terminal=list(self.terminal),
            )

        def apply_many(self, rows: list[int], _actions: list[int]) -> None:
            for row in rows:
                self.applied[row] += 1
                if self.applied[row] == 1:
                    self.to_move[row] = 1
                else:
                    self.terminal[row] = True

        def score_data(self) -> object:
            return SimpleNamespace(winner=[0] * self.batch_size)

    class BatchSearch:
        def __init__(self, states: BatchStates, **_options: object) -> None:
            self.size = states.batch_size
            self.initialized = False

        def root_requests(self) -> BatchRequests:
            search_threads.add(threading.get_ident())
            if search_barrier is not None:
                search_barrier.wait()
            return BatchRequests(self.size)

        def initialize_roots(self, *_buffers: object) -> None:
            self.initialized = True

        def is_done(self) -> bool:
            return self.initialized

        def next_requests(self) -> BatchRequests:
            raise AssertionError("batch search completes at the root")

        def submit(self, *_buffers: object) -> None:
            raise AssertionError("batch search has no leaves")

        def results(self) -> object:
            return SimpleNamespace(
                terminal=[False] * self.size,
                selected_actions=[0] * self.size,
            )

    native = SimpleNamespace(StateBatch=BatchStates, SearchBatch=BatchSearch)
    candidate = SerializedEvaluator("candidate")
    baseline = SerializedEvaluator("baseline")
    result = ArenaRunner(
        native_module=native,
        candidate=candidate,
        baseline=baseline,
        config=ArenaConfig(
            rings=(4,),
            pairs_per_ring=2,
            simulations=1,
            max_considered=1,
            minimum_pairs_per_ring=2,
            max_pairs_per_ring=4,
            bootstrap_samples=200,
            regression_floor_elo=-2_500,
            unforced_opening_fraction=0.5,
        ),
    ).run()

    assert result["aggregate"]["wins"] == result["aggregate"]["losses"] == 2
    assert "draws" not in result["aggregate"]
    assert candidate.evaluator_calls == baseline.evaluator_calls == 2
    assert candidate.evaluator_rows == baseline.evaluator_rows == 3
    assert len(search_threads) == 2
    assert maximum_active_inference == 1
    assert len(inference_threads) == 1
    assert result["evaluation_metrics"]["serialized_inference_calls"] == 4
    assert result["search"]["search_workers"] == 2
    assert result["search"]["inference_workers"] == 1

    search_barrier = None
    sequential = ArenaRunner(
        native_module=native,
        candidate=SerializedEvaluator("candidate"),
        baseline=SerializedEvaluator("baseline"),
        config=ArenaConfig(
            rings=(4,),
            pairs_per_ring=2,
            simulations=1,
            max_considered=1,
            minimum_pairs_per_ring=2,
            max_pairs_per_ring=4,
            bootstrap_samples=200,
            regression_floor_elo=-2_500,
            unforced_opening_fraction=0.5,
        ),
        search_workers=1,
    ).run()

    assert result["games"] == sequential["games"]
    assert result["pairs"] == sequential["pairs"]
    assert result["aggregate"] == sequential["aggregate"]
    assert result["promotion"] == sequential["promotion"]


def test_pair_confidence_summary_and_internal_target_use_anytime_bounds() -> None:
    pairs = [ArenaPair(4, pair, pair, 0, True, (1, 1)) for pair in range(50)]
    lower, upper = pair_confidence_sequence(pairs, error_probability=0.025)
    assert lower > 0.5
    assert upper == 1.0
    summary = summarize_pairs(
        pairs,
        confidence=0.95,
        bootstrap_samples=200,
        seed=17,
    )
    assert summary["anytime_confidence_sequence"][0] > 0.5
    assert summary["anytime_elo_interval"][0] > 0

    result = {
        "per_ring": {
            str(ring): {
                "anytime_elo_interval": [lower_elo, 800.0],
                "pairs": 50,
            }
            for ring, lower_elo in ((4, 450.0), (6, 425.0), (8, 399.0), (10, 500.0))
        }
    }
    assessment = internal_elo_target_assessment(
        result,
        rings=(4, 6, 8, 10),
        target_elo=400.0,
    )
    assert assessment["status"] == "not_reached"
    assert assessment["passed"] is False
    assert assessment["per_ring"]["8"]["passed"] is False


def test_arena_summary_requires_every_configured_ring_and_records_budget() -> None:
    config = ArenaConfig(
        rings=(4, 6),
        pairs_per_ring=2,
        simulations=3,
        max_considered=5,
        minimum_pairs_per_ring=2,
        max_pairs_per_ring=4,
        bootstrap_samples=200,
    )
    budget = ArenaSearchBudget.from_config(config)
    assert budget.metadata() == {
        "simulations": 3,
        "max_considered": 5,
        "c_visit": config.c_visit,
        "c_scale": config.c_scale,
    }
    with pytest.raises(ValueError, match="at least one pair per ring"):
        summarize_arena_pairs(
            [ArenaPair(4, 0, 0, 0, True, (1, -1))],
            config,
        )
    with pytest.raises(ValueError, match="positive integer"):
        ArenaSearchBudget(False, 1, 1.0, 1.0)


def test_weighted_macro_block_uses_exact_seven_of_ten_score() -> None:
    config = weighted_arena_config()
    per_ring = {
        4: [scored_pair(4, 0, 0.0)],
        6: [scored_pair(6, 0, 0.0)],
        8: [scored_pair(8, 0, 0.0)],
        10: [scored_pair(10, pair, 1.0) for pair in (6, 0, 5, 1, 4, 2, 3)],
    }
    aggregate = [
        pair for values in reversed(tuple(per_ring.values())) for pair in values
    ]

    summary = summarize_arena_pairs(aggregate, config)
    weighted = summary["weighted_aggregate"]

    assert weighted["observation_model"] == WEIGHTED_OBSERVATION_MODEL
    assert weighted["pair_ratios"] == {"4": 1, "6": 1, "8": 1, "10": 7}
    assert weighted["normalized_weights"] == {
        "4": 0.1,
        "6": 0.1,
        "8": 0.1,
        "10": 0.7,
    }
    assert weighted["complete_blocks"] == 1
    assert weighted["block_scores"] == [0.7]
    assert weighted["score_rate"] == 0.7
    assert weighted["elo_difference"] == pytest.approx(147.19071411783776)
    assert weighted["block_pair_indices"] == [
        {
            "4": [0],
            "6": [0],
            "8": [0],
            "10": [0, 1, 2, 3, 4, 5, 6],
        }
    ]
    assert summary["promotion"]["weighted_aggregate"] == weighted


def test_weighted_macro_blocks_exclude_every_incomplete_pair() -> None:
    config = weighted_arena_config()
    per_ring = {
        ring: [
            scored_pair(ring, 1, 1.0),
            scored_pair(ring, 0, 0.0),
        ]
        for ring in (4, 6, 8)
    }
    per_ring[10] = [
        *[scored_pair(10, pair, 1.0) for pair in range(7)],
        *[scored_pair(10, pair, 0.0) for pair in range(7, 13)],
    ]
    aggregate = [pair for values in per_ring.values() for pair in reversed(values)]

    assessment = promotion_assessment(aggregate, per_ring, config)
    weighted = assessment["weighted_aggregate"]

    assert weighted["complete_blocks"] == 1
    assert weighted["incomplete_pair_counts"] == {
        "4": 1,
        "6": 1,
        "8": 1,
        "10": 6,
    }
    assert weighted["block_scores"] == [0.7]
    assert weighted["score_rate"] == 0.7
    assert assessment["pair_score_rate"] == 0.7


def test_weighted_partial_ring_coverage_persists_incomplete_counts() -> None:
    config = weighted_arena_config()

    summary = summarize_completed_arena_pairs(
        [scored_pair(4, 0, 1.0)],
        config,
    )

    assert summary["promotion"]["decision"] == "continue"
    assert summary["weighted_aggregate"]["complete_blocks"] == 0
    assert summary["weighted_aggregate"]["incomplete_pair_counts"] == {
        "4": 1,
        "6": 0,
        "8": 0,
        "10": 0,
    }


def test_weighted_macro_block_order_is_deterministic_by_pair_index() -> None:
    config = weighted_arena_config()
    per_ring = {
        ring: [
            scored_pair(ring, 1, 1.0),
            scored_pair(ring, 0, 0.0),
        ]
        for ring in (4, 6, 8)
    }
    per_ring[10] = [
        scored_pair(10, pair, 1.0 if pair < 7 else 0.0) for pair in reversed(range(14))
    ]
    aggregate = [pair for values in per_ring.values() for pair in values]
    reversed_per_ring = {
        ring: list(reversed(values)) for ring, values in per_ring.items()
    }

    forward = promotion_assessment(aggregate, per_ring, config)
    backward = promotion_assessment(
        list(reversed(aggregate)),
        reversed_per_ring,
        config,
    )

    assert forward["weighted_aggregate"] == backward["weighted_aggregate"]
    weighted = forward["weighted_aggregate"]
    assert weighted["block_scores"] == [0.7, 0.3]
    assert weighted["block_pair_indices"][0]["10"] == list(range(7))
    assert weighted["block_pair_indices"][1]["10"] == list(range(7, 14))


def test_weighted_macro_blocks_reject_gapped_pair_indices() -> None:
    config = weighted_arena_config()
    per_ring = {
        4: [scored_pair(4, 0, 0.5), scored_pair(4, 2, 0.5)],
        6: [scored_pair(6, 0, 0.5), scored_pair(6, 1, 0.5)],
        8: [scored_pair(8, 0, 0.5), scored_pair(8, 1, 0.5)],
        10: [scored_pair(10, pair, 0.5) for pair in range(14)],
    }
    aggregate = [pair for values in per_ring.values() for pair in values]

    with pytest.raises(ValueError, match="contiguous from zero"):
        promotion_assessment(aggregate, per_ring, config)


def test_weighted_promotion_ignores_nonrequired_small_ring_regression() -> None:
    config = weighted_arena_config(
        minimum_pairs_per_ring=10,
        required_regression_rings=(),
        weighted_initial_blocks=15,
        weighted_continuation_blocks=10,
        weighted_max_blocks=50,
    )
    per_ring = weighted_pairs(
        20,
        ring_scores={4: 0.0, 6: 1.0, 8: 1.0, 10: 1.0},
    )
    aggregate = [pair for values in per_ring.values() for pair in values]

    assessment = promotion_assessment(aggregate, per_ring, config)

    assert assessment["sequential_state"] == "accept_alternative"
    assert assessment["weighted_aggregate"]["score_rate"] == 0.9
    assert assessment["ring_floors"]["4"]["status"] == "regress"
    assert assessment["ring_floors"]["4"]["required_for_promotion"] is False
    assert assessment["decision"] == "promote"


def test_required_regression_ring_retains_legacy_guard_behavior() -> None:
    config = weighted_arena_config(
        minimum_pairs_per_ring=10,
        required_regression_rings=(4,),
        weighted_initial_blocks=15,
        weighted_continuation_blocks=10,
        weighted_max_blocks=50,
    )
    per_ring = weighted_pairs(
        20,
        ring_scores={4: 0.0, 6: 1.0, 8: 1.0, 10: 1.0},
    )
    aggregate = [pair for values in per_ring.values() for pair in values]

    required = promotion_assessment(aggregate, per_ring, config)
    default_all = promotion_assessment(
        aggregate,
        per_ring,
        replace(config, required_regression_rings=None),
    )

    assert required["sequential_state"] == "accept_alternative"
    assert required["ring_floors"]["4"]["required_for_promotion"] is True
    assert required["decision"] == "reject_ring_regression"
    assert default_all["decision"] == "reject_ring_regression"
    assert default_all["weighted_aggregate"]["required_regression_rings"] == [
        4,
        6,
        8,
        10,
    ]


def test_required_regression_rings_control_legacy_pair_path() -> None:
    config = ArenaConfig(
        rings=(4, 6),
        pairs_per_ring=2,
        simulations=1,
        max_considered=2,
        minimum_pairs_per_ring=10,
        max_pairs_per_ring=100,
        bootstrap_samples=200,
        regression_floor_elo=0.0,
    )
    per_ring = {
        4: [scored_pair(4, pair, 0.0) for pair in range(20)],
        6: [scored_pair(6, pair, 1.0) for pair in range(80)],
    }
    aggregate = [pair for values in per_ring.values() for pair in values]

    legacy = promotion_assessment(aggregate, per_ring, config)
    only_six_required = promotion_assessment(
        aggregate,
        per_ring,
        replace(config, required_regression_rings=(6,)),
    )

    assert legacy["sequential_state"] == "accept_alternative"
    assert legacy["ring_floors"]["4"]["status"] == "regress"
    assert legacy["decision"] == "reject_ring_regression"
    assert only_six_required["decision"] == "promote"
    assert "weighted_aggregate" not in only_six_required


def test_empty_ratios_preserve_legacy_arena_result_shape() -> None:
    config = arena_config()
    pairs = [scored_pair(4, pair, 1.0) for pair in range(2)]
    assessment = promotion_assessment(pairs, {4: pairs}, config)
    summary = summarize_arena_pairs(pairs, config)

    assert set(assessment) == {
        "decision",
        "sequential_state",
        "pair_score_rate",
        "confidence_sequence",
        "null_elo",
        "alternative_elo",
        "pair_model",
        "statistical_test",
        "ring_floors",
    }
    assert assessment["pair_model"] == "pair-level-mixture-betting-e-process-v1"
    assert assessment["statistical_test"]["observation_unit"] == (
        "complete-role-reversed-pair"
    )
    assert set(summary) == {"aggregate", "per_ring", "promotion"}


def test_generic_bounded_e_process_matches_pair_specialization() -> None:
    pairs = [scored_pair(4, index, (0.0, 0.5, 1.0)[index % 3]) for index in range(60)]
    observations = [pair.score_rate for pair in pairs]

    generic = bounded_confidence_sequence(
        observations,
        error_probability=0.025,
    )
    paired = pair_confidence_sequence(pairs, error_probability=0.025)
    assert generic == pytest.approx(paired)
    assert bounded_log_e_value(
        [1.0] * 20,
        null_mean=0.5,
        direction="greater",
    ) == pytest.approx(
        bounded_log_e_value(
            [0.0] * 20,
            null_mean=0.5,
            direction="less",
        )
    )

    arbitrary = bounded_confidence_sequence(
        [0.1, 0.3, 0.65, 0.8] * 20,
        error_probability=0.05,
    )
    assert 0.0 <= arbitrary[0] <= arbitrary[1] <= 1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bounded_log_e_value([1.01], null_mean=0.5, direction="greater")


def test_weighted_macro_gate_controls_repeated_look_null_and_has_power() -> None:
    def crossing_count(probability: float, *, seed: int) -> int:
        rng = random.Random(seed)
        crossings = 0
        for _ in range(250):
            observations: list[float] = []
            for blocks in range(1, 51):
                observations.append(
                    sum(rng.random() < probability for _ in range(10)) / 10
                )
                if blocks >= 15 and bounded_log_e_value(
                    observations,
                    null_mean=0.5,
                    direction="greater",
                ) >= math.log(20):
                    crossings += 1
                    break
        return crossings

    null_crossings = crossing_count(0.5, seed=20260851)
    alternative_crossings = crossing_count(0.65, seed=20260866)

    assert null_crossings <= 12
    assert alternative_crossings >= 225


def test_weighted_arena_config_accepts_production_parameters() -> None:
    config = weighted_arena_config(
        weighted_initial_blocks=15,
        weighted_continuation_blocks=10,
        weighted_max_blocks=50,
    )

    assert config.promotion_pair_ratios == {4: 1, 6: 1, 8: 1, 10: 7}
    assert config.required_regression_rings == ()
    assert (
        config.weighted_initial_blocks,
        config.weighted_continuation_blocks,
        config.weighted_max_blocks,
    ) == (15, 10, 50)
    assert replace(
        config, required_regression_rings=(10, 4)
    ).required_regression_rings == (
        10,
        4,
    )
    legacy = arena_config()
    assert legacy.promotion_pair_ratios == {}
    assert (
        legacy.weighted_initial_blocks,
        legacy.weighted_continuation_blocks,
        legacy.weighted_max_blocks,
    ) == (0, 0, 0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"promotion_pair_ratios": {4: 1, 6: 1, 8: 1}},
        {"promotion_pair_ratios": {4: 2, 6: 2, 8: 2, 10: 14}},
        {"promotion_pair_ratios": {4: True, 6: 1, 8: 1, 10: 7}},
        {"weighted_initial_blocks": 0},
        {"weighted_continuation_blocks": 0},
        {"weighted_max_blocks": 14, "weighted_initial_blocks": 15},
        {"promotion_pair_ratios": {}},
        {"required_regression_rings": (4, 4)},
        {"required_regression_rings": (12,)},
        {"weighted_initial_blocks": True},
    ],
)
def test_weighted_arena_config_rejects_incoherent_parameters(
    overrides: dict[str, object],
) -> None:
    values = {
        "rings": (4, 6, 8, 10),
        "promotion_pair_ratios": {4: 1, 6: 1, 8: 1, 10: 7},
        "required_regression_rings": (),
        "weighted_initial_blocks": 15,
        "weighted_continuation_blocks": 10,
        "weighted_max_blocks": 50,
    }
    values.update(overrides)

    with pytest.raises(ConfigError):
        ArenaConfig(**values)
