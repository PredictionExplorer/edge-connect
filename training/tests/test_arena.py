from __future__ import annotations

import math
import random
from pathlib import Path
from types import SimpleNamespace

from startrain.arena import (
    ARENA_RESULT_SCHEMA_VERSION,
    ArenaPair,
    ArenaRunner,
    WDL,
    _pair_mean_exceeds,
    _pair_sequential_state,
    promotion_assessment,
    summarize_pairs,
    summarize_wdl,
    wilson_interval,
)
from startrain.config import ArenaConfig, load_config
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
    assert result["schema_version"] == ARENA_RESULT_SCHEMA_VERSION
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


def test_pair_level_elo_and_e_process_promotion_with_anytime_ring_floors() -> None:
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
    assessment = promotion_assessment(aggregate, per_ring, config)
    assert assessment["sequential_state"] == "accept_alternative"
    assert assessment["decision"] == "promote"
    assert assessment["pair_model"] == "pair-level-mixture-betting-e-process-v1"
    assert assessment["statistical_test"]["name"] == (
        "bounded-mean-mixture-betting-e-process"
    )

    regressed = dict(per_ring)
    regressed[12] = [ArenaPair(12, pair, pair, 0, True, (-1, -1)) for pair in range(10)]
    assessment = promotion_assessment(
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


_REPRESENTATIVE_NULL_PENTANOMIAL = (0.08, 0.17, 0.50, 0.17, 0.08)
# Exponentially tilt the symmetric null distribution to exactly +35 Elo. All
# five pair-score cells remain populated, with variance 0.06126. This models
# role-paired games with substantial draws/role cancellation without assuming
# independence between the two games inside an opening pair.
_REPRESENTATIVE_35_ELO_PENTANOMIAL = (
    0.05202367115614806,
    0.13568689989888738,
    0.4898205401200923,
    0.20440612242441344,
    0.11806276640045883,
)
_PENTANOMIAL_SCORE_RATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _configured_eligible_looks(config: ArenaConfig) -> tuple[int, ...]:
    looks = []
    pairs_per_ring = 0
    while pairs_per_ring < config.max_pairs_per_ring:
        pairs_per_ring = min(
            pairs_per_ring + config.pairs_per_ring,
            config.max_pairs_per_ring,
        )
        if pairs_per_ring >= config.minimum_pairs_per_ring:
            looks.append(pairs_per_ring)
    return tuple(looks)


def _monte_carlo_promotion_rate(
    config: ArenaConfig,
    probabilities: tuple[float, ...],
    *,
    seed: int,
    trials: int,
) -> float:
    rng = random.Random(seed)
    categories = range(len(probabilities))
    eligible_looks = _configured_eligible_looks(config)
    null_score_rate = 1.0 / (1.0 + 10.0 ** (-config.null_elo / 400.0))
    alternative_score_rate = 1.0 / (1.0 + 10.0 ** (-config.alternative_elo / 400.0))
    floor_score_rates = [
        1.0
        / (
            1.0
            + 10.0
            ** (
                -config.per_ring_regression_floor_elo.get(
                    ring, config.regression_floor_elo
                )
                / 400.0
            )
        )
        for ring in config.rings
    ]
    floor_error_probability = (1.0 - config.confidence) / 2.0
    promotions = 0
    for _ in range(trials):
        per_ring = [
            rng.choices(
                categories,
                weights=probabilities,
                k=config.max_pairs_per_ring,
            )
            for _ring in config.rings
        ]
        per_ring_counts = [[0] * len(probabilities) for _ring in config.rings]
        counts = [0] * len(probabilities)
        prior_look = 0
        for look in eligible_looks:
            for ring_counts, ring_categories in zip(
                per_ring_counts, per_ring, strict=True
            ):
                for category in ring_categories[prior_look:look]:
                    ring_counts[category] += 1
                    counts[category] += 1
            prior_look = look
            state, _, _ = _pair_sequential_state(
                counts,
                null_score_rate=null_score_rate,
                alternative_score_rate=alternative_score_rate,
                alpha=config.alpha,
                beta=config.beta,
            )
            floors_pass = all(
                _pair_mean_exceeds(
                    ring_counts,
                    null_score_rate=floor_score_rate,
                    error_probability=floor_error_probability,
                )
                for ring_counts, floor_score_rate in zip(
                    per_ring_counts, floor_score_rates, strict=True
                )
            )
            if state == "accept_alternative" and floors_pass:
                promotions += 1
                break
            if state == "accept_null":
                break
            if not floors_pass:
                break
    return promotions / trials


def test_configured_e_process_operating_characteristics() -> None:
    config_root = Path(__file__).parents[1] / "configs"
    four_gpu = load_config(config_root / "h100-4gpu.yaml").arena
    eight_gpu = load_config(config_root / "h100-8gpu.yaml").arena
    assert four_gpu == eight_gpu
    assert _configured_eligible_looks(four_gpu) == (
        50,
        75,
        100,
        125,
        150,
        175,
        200,
    )
    assert tuple(
        look * len(four_gpu.rings) for look in _configured_eligible_looks(four_gpu)
    ) == (500, 750, 1_000, 1_250, 1_500, 1_750, 2_000)

    null_mean = sum(
        probability * score
        for probability, score in zip(
            _REPRESENTATIVE_NULL_PENTANOMIAL,
            _PENTANOMIAL_SCORE_RATES,
            strict=True,
        )
    )
    alternative_mean = sum(
        probability * score
        for probability, score in zip(
            _REPRESENTATIVE_35_ELO_PENTANOMIAL,
            _PENTANOMIAL_SCORE_RATES,
            strict=True,
        )
    )
    expected_alternative = 1.0 / (1.0 + 10.0 ** (-four_gpu.alternative_elo / 400.0))
    assert abs(null_mean - 0.5) < 1e-12
    assert abs(alternative_mean - expected_alternative) < 1e-12
    maximum_pairs = len(four_gpu.rings) * four_gpu.max_pairs_per_ring
    legacy_allocation = (
        four_gpu.alpha * 6.0 / (math.pi * math.pi * maximum_pairs * maximum_pairs)
    )
    legacy_radius = math.sqrt(math.log(1.0 / legacy_allocation) / (2.0 * maximum_pairs))
    assert alternative_mean - legacy_radius < null_mean

    trials = 2_000
    false_promotion_rate = _monte_carlo_promotion_rate(
        four_gpu,
        _REPRESENTATIVE_NULL_PENTANOMIAL,
        seed=20260710,
        trials=trials,
    )
    promotion_power = _monte_carlo_promotion_rate(
        four_gpu,
        _REPRESENTATIVE_35_ELO_PENTANOMIAL,
        seed=20260711,
        trials=trials,
    )
    # These deterministic Monte Carlo checks exercise both aggregate stopping
    # boundaries and every per-ring anytime regression floor.
    assert false_promotion_rate <= 0.05
    assert promotion_power >= 0.80


def test_configured_ring_floors_allow_representative_non_regression() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "h100-8gpu.yaml").arena
    outcomes = ((-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))
    # Fifty pairs with all five cells represented and score rate 0.55.
    category_counts = (3, 6, 25, 10, 6)
    per_ring = {}
    for ring in config.rings:
        categories = [
            category
            for category, count in enumerate(category_counts)
            for _ in range(count)
        ]
        per_ring[ring] = [
            ArenaPair(ring, pair, pair, 0, True, outcomes[category])
            for pair, category in enumerate(categories)
        ]
    aggregate = [pair for ring in config.rings for pair in per_ring[ring]]
    assessment = promotion_assessment(aggregate, per_ring, config)
    assert assessment["decision"] == "promote"
    assert assessment["confidence_sequence"][0] > 0.5
    evidence = assessment["statistical_test"]["promotion"]
    assert evidence["e_value"] >= evidence["threshold"]
    assert all(
        floor["passed"] is True and floor["anytime_lower_elo"] >= floor["floor_elo"]
        for floor in assessment["ring_floors"].values()
    )
