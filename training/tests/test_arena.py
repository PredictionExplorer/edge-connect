from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from startrain.arena import (
    ARENA_RESULT_SCHEMA_VERSION,
    ArenaPair,
    ArenaRunner,
    ArenaSearchBudget,
    BinaryResults,
    internal_elo_target_assessment,
    pair_confidence_sequence,
    promotion_assessment,
    summarize_arena_pairs,
    summarize_binary_results,
    summarize_pairs,
    wilson_interval,
)
from startrain.config import ArenaConfig
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


def test_pair_promotion_and_per_ring_regression_are_binary() -> None:
    config = ArenaConfig(
        rings=(4, 6, 8, 10),
        pairs_per_ring=5,
        simulations=1,
        max_considered=2,
        minimum_pairs_per_ring=10,
        max_pairs_per_ring=20,
        bootstrap_samples=200,
        regression_floor_elo=-2_500.0,
    )
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
    assert promotion_assessment(aggregate, per_ring, config)["decision"] == "promote"

    regressed = dict(per_ring)
    regressed[10] = [ArenaPair(10, pair, pair, 0, True, (-1, -1)) for pair in range(10)]
    strict = ArenaConfig(
        rings=(4, 6, 8, 10),
        pairs_per_ring=5,
        simulations=1,
        max_considered=2,
        minimum_pairs_per_ring=10,
        max_pairs_per_ring=20,
        bootstrap_samples=200,
        regression_floor_elo=-100.0,
    )
    assessment = promotion_assessment(
        [pair for pairs in regressed.values() for pair in pairs],
        regressed,
        strict,
    )
    assert assessment["decision"] == "reject_ring_regression"
    assert assessment["ring_floors"]["10"]["passed"] is False


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
