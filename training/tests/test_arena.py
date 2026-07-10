from __future__ import annotations

import random
from types import SimpleNamespace

from startrain.arena import (
    ArenaPair,
    ArenaRunner,
    WDL,
    pair_confidence_sequence,
    sprt_assessment,
    summarize_pairs,
    summarize_wdl,
    wilson_interval,
)
from startrain.config import ArenaConfig
from startrain.inference import InferenceResponse


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
        self.calls = 0

    def evaluate(self, requests: FakeRequests) -> InferenceResponse:
        self.calls += 1
        logits = [0.0, 0.0]
        logits[self.selected_action] = 1.0
        return InferenceResponse(
            tokens=[1],
            values=[0.0],
            policy_offsets=[0, 2],
            policy_logits=logits,
        )


class FakeStateBatch:
    def __init__(self, rings: int, batch_size: int) -> None:
        assert 3 <= rings <= 12
        assert batch_size == 1
        self.terminal = False
        self.to_move = 0
        self.applied = 0
        self.winner = -1

    def apply_many(self, indices: list[int], actions: list[int]) -> None:
        assert indices == [0] and len(actions) == 1
        self.applied += 1
        if self.applied == 1:
            self.to_move = 1
        else:
            self.winner = actions[0]
            self.terminal = True

    def data(self) -> object:
        return SimpleNamespace(
            terminal=[self.terminal],
            to_move=[self.to_move],
        )

    def score_data(self) -> object:
        return SimpleNamespace(winner=[self.winner])


class FakeSearchBatch:
    def __init__(self, states: FakeStateBatch, **_options: object) -> None:
        self.states = states
        self.selected = -2
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
        raise AssertionError("fake search completes at root")

    def submit(self, *_buffers: object) -> None:
        raise AssertionError("fake search has no leaves")

    def results(self) -> object:
        return SimpleNamespace(
            terminal=[False],
            selected_actions=[self.selected],
        )


class FakeNative:
    StateBatch = FakeStateBatch
    SearchBatch = FakeSearchBatch


def test_arena_pairs_identical_openings_and_reverses_roles() -> None:
    candidate = RoleEvaluator("candidate", selected_action=1)
    baseline = RoleEvaluator("baseline", selected_action=0)
    config = ArenaConfig(
        pairs_per_ring=2,
        simulations=1,
        max_considered=2,
        regression_floor_elo=-2_500.0,
    )
    result = ArenaRunner(
        native_module=FakeNative,
        candidate=candidate,
        baseline=baseline,
        config=config,
    ).run()
    assert result["aggregate"]["wins"] == 40
    assert result["aggregate"]["losses"] == 0
    assert candidate.calls == baseline.calls
    assert candidate.calls > 0
    games = result["games"]
    for index in range(0, len(games), 2):
        first, second = games[index : index + 2]
        assert first["ring"] == second["ring"]
        assert first["pair"] == second["pair"]
        assert first["opening_seed"] == second["opening_seed"]
        assert first["opening_action"] == second["opening_action"]
        assert (first["candidate_player"], second["candidate_player"]) == (0, 1)
    assert any(not pair["forced_opening"] for pair in result["pairs"])
    assert any(pair["forced_opening"] for pair in result["pairs"])
    assert result["search"]["pie_rule"] is False
    assert result["search"]["deterministic"] is True


def test_pair_level_elo_and_sprt_promotion_with_bootstrap_ring_floors() -> None:
    balanced = WDL(wins=40, draws=20, losses=40)
    lower, upper = wilson_interval(balanced)
    assert lower < 0.5 < upper
    summary = summarize_wdl(balanced, confidence=0.95)
    assert summary["elo_difference"] == 0.0
    assert summary["wilson_elo_interval"][0] < 0
    assert summary["wilson_elo_interval"][1] > 0

    config = ArenaConfig(
        pairs_per_ring=5,
        simulations=1,
        max_considered=2,
        alternative_elo=35.0,
        regression_floor_elo=-2_500.0,
        minimum_pairs_per_ring=10,
    )
    per_ring = {
        ring: [ArenaPair(ring, pair, pair, 0, True, (1, 1)) for pair in range(10)]
        for ring in config.rings
    }
    aggregate = [pair for pairs in per_ring.values() for pair in pairs]
    pair_summary = summarize_pairs(
        aggregate,
        confidence=0.95,
        bootstrap_samples=200,
        seed=1,
    )
    assert pair_summary["pairs"] == 100
    assert pair_summary["pentanomial"]["2"] == 100
    assessment = sprt_assessment(aggregate, per_ring, config)
    assert assessment["sequential_state"] == "accept_alternative"
    assert assessment["decision"] == "promote"

    regressed = dict(per_ring)
    regressed[12] = [ArenaPair(12, pair, pair, 0, True, (-1, -1)) for pair in range(10)]
    assessment = sprt_assessment(
        [pair for pairs in regressed.values() for pair in pairs],
        regressed,
        ArenaConfig(
            pairs_per_ring=5,
            simulations=1,
            max_considered=2,
            regression_floor_elo=-100.0,
            minimum_pairs_per_ring=10,
        ),
    )
    assert assessment["decision"] == "reject_ring_regression"
    assert assessment["ring_floors"]["12"]["passed"] is False


def test_pair_confidence_sequence_controls_null_false_promotions() -> None:
    rng = random.Random(20260710)
    trials = 1_000
    false_promotions = 0
    for trial in range(trials):
        pairs = []
        promoted = False
        for pair_index in range(200):
            outcome = 1 if rng.random() < 0.5 else -1
            pairs.append(
                ArenaPair(
                    3,
                    pair_index,
                    trial * 1_000 + pair_index,
                    0,
                    True,
                    (outcome, outcome),
                )
            )
            if len(pairs) >= 40 and len(pairs) % 10 == 0:
                lower, _ = pair_confidence_sequence(pairs, error_probability=0.05)
                if lower > 0.5:
                    promoted = True
                    break
        false_promotions += int(promoted)
    assert false_promotions / trials <= 0.06
