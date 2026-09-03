"""Deterministic paired arena matches and conservative promotion statistics."""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from statistics import NormalDist
from typing import Any, Literal, Protocol, cast

from .config import ArenaConfig
from .contracts import SEGMENT_STANDARD
from .inference import InferenceResponse, NativeEvalBatchProtocol
from .native import BITBOARD_WORDS
from .selfplay import STANDARD_VARIANT, GameVariant
from .topology import get_topology


ARENA_RESULT_SCHEMA_VERSION = 4
WEIGHTED_OBSERVATION_MODEL = "complete-weighted-macro-block-score-v1"


class ArenaEvaluatorProtocol(Protocol):
    model_version: str

    def evaluate(self, requests: NativeEvalBatchProtocol) -> InferenceResponse: ...


@dataclass(frozen=True, slots=True)
class ArenaSearchBudget:
    """Immutable native search budget for one arena participant."""

    simulations: int
    max_considered: int
    c_visit: float
    c_scale: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.simulations, bool)
            or not isinstance(self.simulations, int)
            or self.simulations <= 0
        ):
            raise ValueError("arena search simulations must be a positive integer")
        if (
            isinstance(self.max_considered, bool)
            or not isinstance(self.max_considered, int)
            or self.max_considered <= 0
        ):
            raise ValueError("arena search max_considered must be a positive integer")
        if not math.isfinite(self.c_visit) or self.c_visit <= 0:
            raise ValueError("arena search c_visit must be finite and positive")
        if not math.isfinite(self.c_scale) or self.c_scale <= 0:
            raise ValueError("arena search c_scale must be finite and positive")

    @classmethod
    def from_config(cls, config: ArenaConfig) -> "ArenaSearchBudget":
        return cls(
            simulations=config.simulations,
            max_considered=config.max_considered,
            c_visit=config.c_visit,
            c_scale=config.c_scale,
        )

    def metadata(self) -> dict[str, int | float]:
        return {
            "simulations": self.simulations,
            "max_considered": self.max_considered,
            "c_visit": self.c_visit,
            "c_scale": self.c_scale,
        }


# A complete role-reversed pair has 0, 1, or 2 candidate wins. The sequential
# test works on corresponding score rates 0, 0.5, and 1.
_PAIR_SCORE_RATES = (0.0, 0.5, 1.0)
_PAIR_BETTING_FRACTIONS = (
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 10.0,
    1.0 / 8.0,
    3.0 / 16.0,
    1.0 / 4.0,
    3.0 / 8.0,
    1.0 / 2.0,
    5.0 / 8.0,
    3.0 / 4.0,
    7.0 / 8.0,
)


@dataclass(slots=True)
class BinaryResults:
    wins: int = 0
    losses: int = 0

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0 for value in (self.wins, self.losses)
        ):
            raise ValueError("binary result counts must be non-negative integers")

    @property
    def games(self) -> int:
        return self.wins + self.losses

    @property
    def score(self) -> float:
        return float(self.wins)

    @property
    def score_rate(self) -> float:
        return self.score / self.games if self.games else 0.5

    def record(self, outcome: int) -> None:
        if outcome == 1:
            self.wins += 1
        elif outcome == -1:
            self.losses += 1
        else:
            raise ValueError("arena outcomes must be binary: -1 or 1")

    def add(self, other: "BinaryResults") -> None:
        self.wins += other.wins
        self.losses += other.losses


@dataclass(frozen=True, slots=True)
class ArenaGame:
    ring: int
    pair: int
    candidate_player: int
    opening_seed: int
    opening_action: int | None
    forced_opening: bool
    winner: int
    outcome: int
    searched_moves: int
    variant: str = STANDARD_VARIANT.label
    segment: str = SEGMENT_STANDARD
    swapped: bool = False
    pda: int = 0

    def __post_init__(self) -> None:
        topology = get_topology(self.ring)
        parsed = GameVariant.parse(self.variant)
        if parsed.segment != self.segment:
            raise ValueError("arena game segment disagrees with its variant")
        if type(self.swapped) is not bool or (self.swapped and not parsed.pie):
            raise ValueError("arena swap metadata requires a pie game")
        if type(self.pda) is not int or not 0 <= self.pda <= 3:
            raise ValueError("arena playout-doubling advantage must be in 0..3")
        if (
            type(self.candidate_player) is not int
            or self.candidate_player not in (0, 1)
            or type(self.winner) is not int
            or self.winner not in (0, 1)
        ):
            raise ValueError("arena players and winner must be binary")
        expected = 1 if self.winner == self.candidate_player else -1
        if type(self.outcome) is not int or self.outcome != expected:
            raise ValueError("arena outcome disagrees with its decisive winner")
        if (
            type(self.pair) is not int
            or self.pair < 0
            or type(self.opening_seed) is not int
            or self.opening_seed < 0
            or type(self.searched_moves) is not int
            or self.searched_moves < 0
        ):
            raise ValueError("arena game counters must be non-negative integers")
        if self.opening_action is not None and not (
            type(self.opening_action) is int and 0 <= self.opening_action < topology.n
        ):
            raise ValueError("arena opening action is invalid")
        if type(self.forced_opening) is not bool or self.forced_opening != (
            self.opening_action is not None
        ):
            raise ValueError("arena opening metadata is inconsistent")


@dataclass(frozen=True, slots=True)
class ArenaPair:
    ring: int
    pair: int
    opening_seed: int
    opening_action: int | None
    forced_opening: bool
    outcomes: tuple[int, int]
    variant: str = STANDARD_VARIANT.label
    segment: str = SEGMENT_STANDARD

    def __post_init__(self) -> None:
        topology = get_topology(self.ring)
        if GameVariant.parse(self.variant).segment != self.segment:
            raise ValueError("arena pair segment disagrees with its variant")
        if any(
            type(outcome) is not int or outcome not in (-1, 1)
            for outcome in self.outcomes
        ):
            raise ValueError("arena pairs cannot contain tied outcomes")
        if (
            type(self.pair) is not int
            or self.pair < 0
            or type(self.opening_seed) is not int
            or self.opening_seed < 0
        ):
            raise ValueError("arena pair counters must be non-negative integers")
        if self.opening_action is not None and not (
            type(self.opening_action) is int and 0 <= self.opening_action < topology.n
        ):
            raise ValueError("arena pair opening action is invalid")
        if type(self.forced_opening) is not bool or self.forced_opening != (
            self.opening_action is not None
        ):
            raise ValueError("arena pair opening metadata is inconsistent")

    @property
    def points(self) -> float:
        return float(sum(value == 1 for value in self.outcomes))

    @property
    def score_rate(self) -> float:
        return self.points / 2.0


def wilson_interval(
    result: BinaryResults, *, confidence: float = 0.95
) -> tuple[float, float]:
    if result.games <= 0:
        raise ValueError("Wilson interval requires at least one game")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    n = float(result.games)
    probability = result.score_rate
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denominator = 1 + z * z / n
    center = (probability + z * z / (2 * n)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1 - probability) / n + z * z / (4 * n * n))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def elo_from_probability(probability: float) -> float:
    clipped = min(1 - 1e-6, max(1e-6, probability))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def summarize_binary_results(
    result: BinaryResults, *, confidence: float
) -> dict[str, object]:
    lower, upper = wilson_interval(result, confidence=confidence)
    return {
        **asdict(result),
        "games": result.games,
        "score_rate": result.score_rate,
        "elo_difference": elo_from_probability(result.score_rate),
        "wilson_score_interval": [lower, upper],
        "wilson_elo_interval": [
            elo_from_probability(lower),
            elo_from_probability(upper),
        ],
    }


def summarize_pairs(
    pairs: Sequence[ArenaPair],
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    if not pairs:
        raise ValueError("paired summary requires at least one pair")
    binary = BinaryResults()
    for pair in pairs:
        for outcome in pair.outcomes:
            binary.record(outcome)
    lower, upper = _paired_bootstrap_interval(
        pairs,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    error_probability = (1.0 - confidence) / 2.0
    anytime_lower, anytime_upper = pair_confidence_sequence(
        pairs,
        error_probability=error_probability,
    )
    counts = _pair_score_counts(pairs)
    return {
        **asdict(binary),
        "games": binary.games,
        "pairs": len(pairs),
        "score_rate": binary.score_rate,
        "elo_difference": elo_from_probability(binary.score_rate),
        "paired_bootstrap_score_interval": [lower, upper],
        "paired_bootstrap_elo_interval": [
            elo_from_probability(lower),
            elo_from_probability(upper),
        ],
        "anytime_confidence_sequence": [anytime_lower, anytime_upper],
        "anytime_elo_interval": [
            elo_from_probability(anytime_lower),
            elo_from_probability(anytime_upper),
        ],
        "anytime_error_probability_per_side": error_probability,
        "pair_win_counts": {
            "0": counts[0],
            "1": counts[1],
            "2": counts[2],
        },
    }


def internal_elo_target_assessment(
    arena_result: Mapping[str, object],
    *,
    rings: Sequence[int],
    target_elo: float,
) -> dict[str, object]:
    """Assess a fixed internal target with paired anytime-valid lower bounds."""

    if not rings or len(set(rings)) != len(rings):
        raise ValueError("internal Elo target rings must be non-empty and unique")
    if not math.isfinite(target_elo):
        raise ValueError("internal Elo target must be finite")
    per_ring = arena_result.get("per_ring")
    if not isinstance(per_ring, Mapping):
        raise ValueError("arena result omitted per-ring summaries")

    assessments: dict[str, object] = {}
    passed = True
    for ring in rings:
        summary = per_ring.get(str(ring))
        if not isinstance(summary, Mapping):
            raise ValueError(f"arena result omitted ring {ring}")
        interval = summary.get("anytime_elo_interval")
        if (
            not isinstance(interval, Sequence)
            or isinstance(interval, str | bytes)
            or len(interval) != 2
            or isinstance(interval[0], bool)
            or not isinstance(interval[0], int | float)
        ):
            raise ValueError(f"arena result has an invalid ring {ring} Elo interval")
        lower = float(interval[0])
        ring_passed = lower >= target_elo
        passed = passed and ring_passed
        assessments[str(ring)] = {
            "lower_elo": lower,
            "target_elo": target_elo,
            "passed": ring_passed,
            "pairs": summary.get("pairs"),
        }
    return {
        "schema_version": 1,
        "status": "passed" if passed else "not_reached",
        "passed": passed,
        "target_elo": target_elo,
        "rings": list(rings),
        "confidence_method": "pair-level-mixture-betting-confidence-sequence-v1",
        "per_ring": assessments,
    }


@dataclass(frozen=True, slots=True)
class _WeightedMacroBlock:
    score_rate: float
    pair_indices: dict[int, tuple[int, ...]]


def _weighted_macro_blocks(
    pairs: Sequence[ArenaPair],
    config: ArenaConfig,
) -> tuple[tuple[_WeightedMacroBlock, ...], dict[int, int]]:
    per_ring: dict[int, list[ArenaPair]] = {ring: [] for ring in config.rings}
    for pair in pairs:
        if pair.ring not in per_ring:
            raise ValueError("arena pair has an unconfigured ring")
        per_ring[pair.ring].append(pair)
    for ring, values in per_ring.items():
        values.sort(key=lambda pair: pair.pair)
        sorted_pair_indices = [pair.pair for pair in values]
        if len(sorted_pair_indices) != len(set(sorted_pair_indices)):
            raise ValueError(f"weighted arena ring {ring} has duplicate pair indices")
        if sorted_pair_indices != list(range(len(sorted_pair_indices))):
            raise ValueError(
                f"weighted arena ring {ring} pair indices must be contiguous from zero"
            )

    complete_blocks = min(
        len(per_ring[ring]) // config.promotion_pair_ratios[ring]
        for ring in config.rings
    )
    pairs_per_block = sum(config.promotion_pair_ratios.values())
    blocks: list[_WeightedMacroBlock] = []
    for block_index in range(complete_blocks):
        selected: list[ArenaPair] = []
        pair_indices: dict[int, tuple[int, ...]] = {}
        for ring in config.rings:
            ratio = config.promotion_pair_ratios[ring]
            start = block_index * ratio
            ring_pairs = per_ring[ring][start : start + ratio]
            selected.extend(ring_pairs)
            pair_indices[ring] = tuple(pair.pair for pair in ring_pairs)
        score_rate = math.fsum(pair.score_rate for pair in selected) / pairs_per_block
        if not 0.0 <= score_rate <= 1.0:
            raise RuntimeError("weighted macro-block score escaped [0, 1]")
        blocks.append(
            _WeightedMacroBlock(
                score_rate=score_rate,
                pair_indices=pair_indices,
            )
        )
    incomplete_pair_counts = {
        ring: len(per_ring[ring]) - complete_blocks * config.promotion_pair_ratios[ring]
        for ring in config.rings
    }
    return tuple(blocks), incomplete_pair_counts


def promotion_assessment(
    aggregate: Sequence[ArenaPair],
    per_ring: Mapping[int, Sequence[ArenaPair]],
    config: ArenaConfig,
) -> dict[str, object]:
    if not aggregate:
        raise ValueError("promotion assessment requires paired games")
    score_rate = sum(pair.score_rate for pair in aggregate) / len(aggregate)
    lower, _ = pair_confidence_sequence(aggregate, error_probability=config.alpha)
    _, upper = pair_confidence_sequence(aggregate, error_probability=config.beta)
    probability_null = _expected_score(config.null_elo)
    probability_alternative = _expected_score(config.alternative_elo)
    counts = _pair_score_counts(aggregate)
    (
        sequential_state,
        promotion_log_e_value,
        rejection_log_e_value,
    ) = _pair_sequential_state(
        counts,
        null_score_rate=probability_null,
        alternative_score_rate=probability_alternative,
        alpha=config.alpha,
        beta=config.beta,
    )
    minimum_ready = all(
        len(per_ring[ring]) >= config.minimum_pairs_per_ring for ring in config.rings
    )
    if not minimum_ready:
        sequential_state = "continue"
    weighted_aggregate: dict[str, object] | None = None
    observation_model = "pair-level-mixture-betting-e-process-v1"
    observation_unit = "complete-role-reversed-pair"
    if config.promotion_pair_ratios:
        blocks, incomplete_pair_counts = _weighted_macro_blocks(aggregate, config)
        block_scores = tuple(block.score_rate for block in blocks)
        if block_scores:
            score_rate = math.fsum(block_scores) / len(block_scores)
            lower, _ = bounded_confidence_sequence(
                block_scores,
                error_probability=config.alpha,
            )
            _, upper = bounded_confidence_sequence(
                block_scores,
                error_probability=config.beta,
            )
            (
                evidence_state,
                promotion_log_e_value,
                rejection_log_e_value,
            ) = _bounded_sequential_state(
                block_scores,
                null_score_rate=probability_null,
                alternative_score_rate=probability_alternative,
                alpha=config.alpha,
                beta=config.beta,
            )
        else:
            score_rate = 0.5
            lower, upper = 0.0, 1.0
            evidence_state = "continue"
            promotion_log_e_value = 0.0
            rejection_log_e_value = 0.0
        minimum_ready = len(blocks) >= config.weighted_initial_blocks
        sequential_state = evidence_state if minimum_ready else "continue"
        observation_model = WEIGHTED_OBSERVATION_MODEL
        observation_unit = "complete-weighted-macro-block"
        ratio_total = sum(config.promotion_pair_ratios.values())
        weighted_score_rate = score_rate if block_scores else None
        weighted_aggregate = {
            "schema_version": 1,
            "observation_model": WEIGHTED_OBSERVATION_MODEL,
            "pair_ratios": {
                str(ring): config.promotion_pair_ratios[ring] for ring in config.rings
            },
            "normalized_weights": {
                str(ring): config.promotion_pair_ratios[ring] / ratio_total
                for ring in config.rings
            },
            "pairs_per_block": ratio_total,
            "complete_blocks": len(blocks),
            "incomplete_pair_counts": {
                str(ring): incomplete_pair_counts[ring] for ring in config.rings
            },
            "block_scores": list(block_scores),
            "block_pair_indices": [
                {str(ring): list(block.pair_indices[ring]) for ring in config.rings}
                for block in blocks
            ],
            "score_rate": weighted_score_rate,
            "elo_difference": (
                elo_from_probability(score_rate) if block_scores else None
            ),
            "anytime_confidence_sequence": [lower, upper],
            "anytime_elo_interval": [
                elo_from_probability(lower),
                elo_from_probability(upper),
            ],
            "anytime_error_probability": {
                "lower": config.alpha,
                "upper": config.beta,
            },
            "promotion_e_value": _reported_e_value(promotion_log_e_value),
            "promotion_log_e_value": promotion_log_e_value,
            "rejection_e_value": _reported_e_value(rejection_log_e_value),
            "rejection_log_e_value": rejection_log_e_value,
            "evidence_state": evidence_state,
            "minimum_ready": minimum_ready,
            "configured_blocks": {
                "initial": config.weighted_initial_blocks,
                "continuation": config.weighted_continuation_blocks,
                "maximum": config.weighted_max_blocks,
            },
        }

    ring_floors: dict[str, object] = {}
    floor_statuses: list[Literal["pass", "regress", "continue"]] = []
    floor_error_probability = (1.0 - config.confidence) / 2.0
    floor_e_value_threshold = 1.0 / floor_error_probability
    floor_log_e_value_threshold = math.log(floor_e_value_threshold)
    for ring in config.rings:
        result = per_ring[ring]
        floor = config.per_ring_regression_floor_elo.get(
            ring, config.regression_floor_elo
        )
        floor_score_rate = _expected_score(floor)
        if len(result) < config.minimum_pairs_per_ring:
            floor_statuses.append("continue")
            ring_floors[str(ring)] = {
                "floor_elo": floor,
                "floor_score_rate": floor_score_rate,
                "paired_bootstrap_lower_elo": None,
                "paired_bootstrap_upper_elo": None,
                "anytime_lower_elo": None,
                "anytime_upper_elo": None,
                "error_probability": floor_error_probability,
                "threshold": floor_e_value_threshold,
                "log_threshold": floor_log_e_value_threshold,
                "pass_e_value": None,
                "pass_log_e_value": None,
                "regression_e_value": None,
                "regression_log_e_value": None,
                "evidence_test": "pair-level-mixture-betting-e-process-v1",
                "pairs": len(result),
                "passed": None,
                "status": "continue",
            }
            continue
        bootstrap_lower_score, bootstrap_upper_score = _paired_bootstrap_interval(
            result,
            confidence=config.confidence,
            samples=config.bootstrap_samples,
            seed=config.seed + ring * 1_000_003,
        )
        anytime_lower_score, anytime_upper_score = pair_confidence_sequence(
            result,
            error_probability=floor_error_probability,
        )
        status, pass_log_e_value, regression_log_e_value = _pair_floor_state(
            _pair_score_counts(result),
            floor_score_rate=floor_score_rate,
            error_probability=floor_error_probability,
        )
        floor_statuses.append(status)
        ring_floors[str(ring)] = {
            "floor_elo": floor,
            "floor_score_rate": floor_score_rate,
            "paired_bootstrap_lower_elo": elo_from_probability(bootstrap_lower_score),
            "paired_bootstrap_upper_elo": elo_from_probability(bootstrap_upper_score),
            "anytime_lower_elo": elo_from_probability(anytime_lower_score),
            "anytime_upper_elo": elo_from_probability(anytime_upper_score),
            "error_probability": floor_error_probability,
            "threshold": floor_e_value_threshold,
            "log_threshold": floor_log_e_value_threshold,
            "pass_e_value": _reported_e_value(pass_log_e_value),
            "pass_log_e_value": pass_log_e_value,
            "regression_e_value": _reported_e_value(regression_log_e_value),
            "regression_log_e_value": regression_log_e_value,
            "evidence_test": "pair-level-mixture-betting-e-process-v1",
            "test": "pair-level-mixture-betting-confidence-sequence-v1",
            "pairs": len(result),
            "passed": status == "pass",
            "status": status,
        }

    configured_required_rings = (
        set(config.rings)
        if config.required_regression_rings is None
        else set(config.required_regression_rings)
    )
    required_regression_rings = tuple(
        ring for ring in config.rings if ring in configured_required_rings
    )
    required_ring_set = set(required_regression_rings)
    required_floor_statuses = [
        status
        for ring, status in zip(config.rings, floor_statuses, strict=True)
        if ring in required_ring_set
    ]
    if weighted_aggregate is not None:
        weighted_aggregate["required_regression_rings"] = list(
            required_regression_rings
        )
        for ring in config.rings:
            floor_summary = cast(dict[str, object], ring_floors[str(ring)])
            floor_summary["required_for_promotion"] = ring in required_ring_set

    if sequential_state == "accept_alternative" and all(
        status == "pass" for status in required_floor_statuses
    ):
        decision = "promote"
    elif sequential_state == "accept_null":
        decision = "reject"
    elif any(status == "regress" for status in required_floor_statuses):
        decision = "reject_ring_regression"
    else:
        decision = "continue"
    assessment: dict[str, object] = {
        "decision": decision,
        "sequential_state": sequential_state,
        "pair_score_rate": score_rate,
        "confidence_sequence": [lower, upper],
        "null_elo": config.null_elo,
        "alternative_elo": config.alternative_elo,
        "pair_model": observation_model,
        "statistical_test": {
            "schema_version": 1,
            "name": "bounded-mean-mixture-betting-e-process",
            "observation_unit": observation_unit,
            "betting_fractions": list(_PAIR_BETTING_FRACTIONS),
            "promotion": {
                "null_score_rate": probability_null,
                "e_value": _reported_e_value(promotion_log_e_value),
                "log_e_value": promotion_log_e_value,
                "threshold": 1.0 / config.alpha,
            },
            "rejection": {
                "null_score_rate": probability_alternative,
                "e_value": _reported_e_value(rejection_log_e_value),
                "log_e_value": rejection_log_e_value,
                "threshold": 1.0 / config.beta,
            },
            "anytime_error_control": "Ville inequality",
        },
        "ring_floors": ring_floors,
    }
    if weighted_aggregate is not None:
        assessment["observation_model"] = WEIGHTED_OBSERVATION_MODEL
        assessment["weighted_aggregate"] = weighted_aggregate
        statistical_test = cast(dict[str, object], assessment["statistical_test"])
        statistical_test["observation_model"] = WEIGHTED_OBSERVATION_MODEL
    return assessment


def pair_confidence_sequence(
    pairs: Sequence[ArenaPair], *, error_probability: float
) -> tuple[float, float]:
    """Invert paired betting e-processes into one-sided anytime-valid bounds.

    A complete role-reversed pair is one bounded observation, so the two games
    may be arbitrarily correlated. For a candidate null mean ``mu`` and fixed
    fraction ``f``, ``prod(1 - f + f * X / mu)`` is a nonnegative
    supermartingale whenever the pair-score conditional mean is at most
    ``mu``. The equal-weight mixture remains an e-process, and Ville's
    inequality makes each returned endpoint valid under continuous monitoring.
    """
    if not pairs or not 0 < error_probability < 1:
        raise ValueError("pair confidence sequence inputs are invalid")
    counts = _pair_score_counts(pairs)
    log_threshold = math.log(1.0 / error_probability)
    epsilon = 1e-12

    if (
        _pair_log_e_value_from_counts(
            counts, null_score_rate=epsilon, direction="greater"
        )
        < log_threshold
    ):
        lower = 0.0
    else:
        low, high = epsilon, 1.0 - epsilon
        for _ in range(52):
            middle = (low + high) / 2.0
            evidence = _pair_log_e_value_from_counts(
                counts,
                null_score_rate=middle,
                direction="greater",
            )
            if evidence >= log_threshold:
                low = middle
            else:
                high = middle
        lower = low

    if (
        _pair_log_e_value_from_counts(
            counts,
            null_score_rate=1.0 - epsilon,
            direction="less",
        )
        < log_threshold
    ):
        upper = 1.0
    else:
        low, high = epsilon, 1.0 - epsilon
        for _ in range(52):
            middle = (low + high) / 2.0
            evidence = _pair_log_e_value_from_counts(
                counts,
                null_score_rate=middle,
                direction="less",
            )
            if evidence >= log_threshold:
                high = middle
            else:
                low = middle
        upper = high
    return lower, upper


def bounded_log_e_value(
    observations: Sequence[float],
    *,
    null_mean: float,
    direction: Literal["greater", "less"],
) -> float:
    """Return mixture-betting log evidence for arbitrary ``[0, 1]`` data."""

    values = _validated_bounded_observations(observations)
    return _bounded_log_e_value(
        values,
        null_mean=null_mean,
        direction=direction,
    )


def bounded_confidence_sequence(
    observations: Sequence[float],
    *,
    error_probability: float,
) -> tuple[float, float]:
    """Invert bounded-mean e-processes into anytime-valid mean bounds."""

    values = _validated_bounded_observations(observations)
    if not 0 < error_probability < 1:
        raise ValueError("bounded confidence sequence error probability is invalid")
    log_threshold = math.log(1.0 / error_probability)
    epsilon = 1e-12

    if (
        _bounded_log_e_value(
            values,
            null_mean=epsilon,
            direction="greater",
        )
        < log_threshold
    ):
        lower = 0.0
    else:
        low, high = epsilon, 1.0 - epsilon
        for _ in range(52):
            middle = (low + high) / 2.0
            evidence = _bounded_log_e_value(
                values,
                null_mean=middle,
                direction="greater",
            )
            if evidence >= log_threshold:
                low = middle
            else:
                high = middle
        lower = low

    if (
        _bounded_log_e_value(
            values,
            null_mean=1.0 - epsilon,
            direction="less",
        )
        < log_threshold
    ):
        upper = 1.0
    else:
        low, high = epsilon, 1.0 - epsilon
        for _ in range(52):
            middle = (low + high) / 2.0
            evidence = _bounded_log_e_value(
                values,
                null_mean=middle,
                direction="less",
            )
            if evidence >= log_threshold:
                high = middle
            else:
                low = middle
        upper = high
    return lower, upper


def _validated_bounded_observations(
    observations: Sequence[float],
) -> tuple[float, ...]:
    values: list[float] = []
    for observation in observations:
        if (
            isinstance(observation, bool)
            or not isinstance(observation, int | float)
            or not math.isfinite(float(observation))
            or not 0.0 <= float(observation) <= 1.0
        ):
            raise ValueError("bounded observations must be finite values in [0, 1]")
        values.append(float(observation))
    if not values:
        raise ValueError("bounded e-process requires at least one observation")
    return tuple(values)


def _bounded_log_e_value(
    observations: Sequence[float],
    *,
    null_mean: float,
    direction: Literal["greater", "less"],
) -> float:
    if (
        isinstance(null_mean, bool)
        or not isinstance(null_mean, int | float)
        or not math.isfinite(float(null_mean))
        or not 0 < null_mean < 1
        or direction not in ("greater", "less")
    ):
        raise ValueError("bounded e-process null and direction are invalid")
    denominator = null_mean if direction == "greater" else 1.0 - null_mean
    transformed = (
        tuple(observations)
        if direction == "greater"
        else tuple(1.0 - observation for observation in observations)
    )
    log_wealths = [
        sum(
            math.log(1.0 - fraction + fraction * observation / denominator)
            for observation in transformed
        )
        for fraction in _PAIR_BETTING_FRACTIONS
    ]
    maximum = max(log_wealths)
    return (
        maximum
        + math.log(sum(math.exp(value - maximum) for value in log_wealths))
        - math.log(len(log_wealths))
    )


def _bounded_sequential_state(
    observations: Sequence[float],
    *,
    null_score_rate: float,
    alternative_score_rate: float,
    alpha: float,
    beta: float,
) -> tuple[str, float, float]:
    if (
        not 0 < null_score_rate < alternative_score_rate < 1
        or not 0 < alpha < 1
        or not 0 < beta < 1
    ):
        raise ValueError("bounded sequential test inputs are invalid")
    promotion_log_e_value = _bounded_log_e_value(
        observations,
        null_mean=null_score_rate,
        direction="greater",
    )
    rejection_log_e_value = _bounded_log_e_value(
        observations,
        null_mean=alternative_score_rate,
        direction="less",
    )
    if promotion_log_e_value >= math.log(1.0 / alpha):
        state = "accept_alternative"
    elif rejection_log_e_value >= math.log(1.0 / beta):
        state = "accept_null"
    else:
        state = "continue"
    return state, promotion_log_e_value, rejection_log_e_value


def _pair_sequential_state(
    counts: Sequence[int],
    *,
    null_score_rate: float,
    alternative_score_rate: float,
    alpha: float,
    beta: float,
) -> tuple[str, float, float]:
    """Return the dual one-sided e-process decision and log evidence."""

    if (
        not 0 < null_score_rate < alternative_score_rate < 1
        or not 0 < alpha < 1
        or not 0 < beta < 1
    ):
        raise ValueError("paired sequential test inputs are invalid")
    promotion_log_e_value = _pair_log_e_value_from_counts(
        counts,
        null_score_rate=null_score_rate,
        direction="greater",
    )
    rejection_log_e_value = _pair_log_e_value_from_counts(
        counts,
        null_score_rate=alternative_score_rate,
        direction="less",
    )
    if promotion_log_e_value >= math.log(1.0 / alpha):
        state = "accept_alternative"
    elif rejection_log_e_value >= math.log(1.0 / beta):
        state = "accept_null"
    else:
        state = "continue"
    return state, promotion_log_e_value, rejection_log_e_value


def _pair_floor_state(
    counts: Sequence[int],
    *,
    floor_score_rate: float,
    error_probability: float,
) -> tuple[Literal["pass", "regress", "continue"], float, float]:
    """Assess both sides of a ring floor with one-sided paired e-processes.

    Pass evidence tests the null that the pair-score mean is at most the floor.
    Regression evidence applies the same construction to ``1 - X`` and tests
    the null that the mean is at least the floor. Each side is anytime-valid
    because its observation unit remains one complete role-reversed pair.
    """

    if not 0 < error_probability < 1:
        raise ValueError("pair floor test error probability is invalid")
    pass_log_e_value = _pair_log_e_value_from_counts(
        counts,
        null_score_rate=floor_score_rate,
        direction="greater",
    )
    regression_log_e_value = _pair_log_e_value_from_counts(
        counts,
        null_score_rate=floor_score_rate,
        direction="less",
    )
    log_threshold = math.log(1.0 / error_probability)
    if pass_log_e_value >= log_threshold:
        status: Literal["pass", "regress", "continue"] = "pass"
    elif regression_log_e_value >= log_threshold:
        status = "regress"
    else:
        status = "continue"
    return status, pass_log_e_value, regression_log_e_value


def _pair_log_e_value_from_counts(
    counts: Sequence[int],
    *,
    null_score_rate: float,
    direction: Literal["greater", "less"],
) -> float:
    if (
        len(counts) != len(_PAIR_SCORE_RATES)
        or any(
            isinstance(count, bool) or int(count) != count or count < 0
            for count in counts
        )
        or sum(counts) <= 0
        or not 0 < null_score_rate < 1
        or direction not in ("greater", "less")
    ):
        raise ValueError("pair e-process inputs are invalid")
    denominator = null_score_rate if direction == "greater" else 1.0 - null_score_rate
    transformed_scores = (
        _PAIR_SCORE_RATES
        if direction == "greater"
        else tuple(1.0 - score for score in _PAIR_SCORE_RATES)
    )
    log_wealths = []
    for fraction in _PAIR_BETTING_FRACTIONS:
        log_wealths.append(
            sum(
                int(count)
                * math.log(1.0 - fraction + fraction * transformed_score / denominator)
                for count, transformed_score in zip(
                    counts, transformed_scores, strict=True
                )
                if count
            )
        )
    maximum = max(log_wealths)
    return (
        maximum
        + math.log(sum(math.exp(value - maximum) for value in log_wealths))
        - math.log(len(log_wealths))
    )


def _reported_e_value(log_e_value: float) -> float:
    # Keep persisted JSON finite while retaining the uncapped log evidence.
    return math.exp(min(log_e_value, math.log(1e300)))


def _pair_score_counts(pairs: Sequence[ArenaPair]) -> tuple[int, ...]:
    counts = [0, 0, 0]
    for pair in pairs:
        counts[int(pair.points)] += 1
    return tuple(counts)


def _paired_bootstrap_interval(
    pairs: Sequence[ArenaPair],
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not pairs:
        raise ValueError("paired bootstrap requires at least one pair")
    rng = random.Random(seed)
    values = [pair.score_rate for pair in pairs]
    estimates = []
    for _ in range(samples):
        estimates.append(
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        )
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    lower = estimates[max(0, int(tail * samples))]
    upper = estimates[min(samples - 1, int((1.0 - tail) * samples))]
    return lower, upper


def _split_segments(
    pairs: Sequence[ArenaPair],
) -> tuple[list[ArenaPair], dict[str, list[ArenaPair]]]:
    standard: list[ArenaPair] = []
    segments: dict[str, list[ArenaPair]] = {}
    for pair in pairs:
        if pair.segment == SEGMENT_STANDARD:
            standard.append(pair)
        else:
            segments.setdefault(pair.segment, []).append(pair)
    return standard, segments


def segment_floor_assessment(
    segments: Mapping[str, Sequence[ArenaPair]],
    config: ArenaConfig,
) -> dict[str, object]:
    """Veto-on-regress floors for the non-standard mixture segments.

    Each segment aggregates its pairs across rings and runs the same
    one-sided paired e-process as the ring floors. A segment can never
    promote a candidate; ``regress`` proves the candidate lost more than the
    segment's floor and vetoes the promotion.
    """

    error_probability = (1.0 - config.confidence) / 2.0
    floors: dict[str, object] = {}
    for segment in sorted(set(config.segment_pairs_per_ring) | set(segments)):
        result = list(segments.get(segment, ()))
        floor = config.segment_regression_floor_elo.get(
            segment, config.regression_floor_elo
        )
        floor_score_rate = _expected_score(floor)
        summary = (
            summarize_pairs(
                result,
                confidence=config.confidence,
                bootstrap_samples=config.bootstrap_samples,
                seed=config.seed + sum(map(ord, segment)) * 1_000_003,
            )
            if result
            else None
        )
        if not result:
            status: Literal["pass", "regress", "continue"] = "continue"
            pass_log_e_value = regression_log_e_value = None
        else:
            status, pass_log_e_value, regression_log_e_value = _pair_floor_state(
                _pair_score_counts(result),
                floor_score_rate=floor_score_rate,
                error_probability=error_probability,
            )
        floors[segment] = {
            "floor_elo": floor,
            "floor_score_rate": floor_score_rate,
            "pairs": len(result),
            "per_ring_pairs": {
                str(ring): sum(pair.ring == ring for pair in result)
                for ring in config.rings
            },
            "summary": summary,
            "error_probability": error_probability,
            "pass_log_e_value": pass_log_e_value,
            "regression_log_e_value": regression_log_e_value,
            "evidence_test": "pair-level-mixture-betting-e-process-v1",
            "status": status,
            "vetoes_promotion": status == "regress",
        }
    return floors


def _apply_segment_vetoes(
    summary: dict[str, object],
    segments: Mapping[str, Sequence[ArenaPair]],
    config: ArenaConfig,
) -> dict[str, object]:
    if not segments and not config.segment_pairs_per_ring:
        return summary
    floors = segment_floor_assessment(segments, config)
    summary["per_segment"] = {
        segment: summarize_pairs(
            values,
            confidence=config.confidence,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed + sum(map(ord, segment)) * 1_000_003,
        )
        for segment, values in sorted(segments.items())
        if values
    }
    promotion = cast(dict[str, object], summary["promotion"])
    promotion["segment_floors"] = floors
    vetoes = [
        segment
        for segment, floor in floors.items()
        if cast(dict[str, object], floor)["status"] == "regress"
    ]
    promotion["segment_vetoes"] = vetoes
    if vetoes and promotion.get("decision") in ("promote", "continue"):
        promotion["decision"] = "reject_ring_regression"
        promotion["regression_source"] = "segment"
        promotion["vetoed_decision"] = (
            "promote"
            if promotion.get("sequential_state") == "accept_alternative"
            else "continue"
        )
    elif promotion.get("decision") == "reject_ring_regression":
        promotion.setdefault("regression_source", "ring")
    return summary


def summarize_arena_pairs(
    pairs: Sequence[ArenaPair], config: ArenaConfig
) -> dict[str, object]:
    if not pairs:
        raise ValueError("arena summary requires pairs")
    standard, segments = _split_segments(pairs)
    if not standard:
        raise ValueError("arena summary requires standard-segment pairs")
    per_ring: dict[int, list[ArenaPair]] = {ring: [] for ring in config.rings}
    for pair in standard:
        if pair.ring not in per_ring:
            raise ValueError("arena pair has an unconfigured ring")
        per_ring[pair.ring].append(pair)
    if any(not values for values in per_ring.values()):
        raise ValueError("arena summary requires at least one pair per ring")
    summary: dict[str, object] = {
        "aggregate": summarize_pairs(
            standard,
            confidence=config.confidence,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed,
        ),
        "per_ring": {
            str(ring): summarize_pairs(
                values,
                confidence=config.confidence,
                bootstrap_samples=config.bootstrap_samples,
                seed=config.seed + ring * 1_000_003,
            )
            for ring, values in per_ring.items()
        },
        "promotion": promotion_assessment(standard, per_ring, config),
    }
    if config.promotion_pair_ratios:
        promotion = cast(dict[str, object], summary["promotion"])
        summary["weighted_aggregate"] = promotion["weighted_aggregate"]
    return _apply_segment_vetoes(summary, segments, config)


def summarize_completed_arena_pairs(
    pairs: Sequence[ArenaPair],
    config: ArenaConfig,
) -> dict[str, object]:
    """Summarize complete pairs without treating partial ring coverage as a decision."""

    if not pairs:
        raise ValueError("arena summary requires pairs")
    standard, segments = _split_segments(pairs)
    present = {pair.ring for pair in standard}
    if standard and present == set(config.rings):
        return summarize_arena_pairs(pairs, config)
    unknown = {pair.ring for pair in pairs} - set(config.rings)
    if unknown:
        raise ValueError("arena pair has an unconfigured ring")
    per_ring = {
        ring: [pair for pair in standard if pair.ring == ring] for ring in config.rings
    }
    if config.promotion_pair_ratios and standard:
        promotion = promotion_assessment(standard, per_ring, config)
        return _apply_segment_vetoes(
            {
                "aggregate": summarize_pairs(
                    standard,
                    confidence=config.confidence,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed,
                ),
                "per_ring": {
                    str(ring): summarize_pairs(
                        values,
                        confidence=config.confidence,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed + ring * 1_000_003,
                    )
                    for ring, values in per_ring.items()
                    if values
                },
                "promotion": promotion,
                "weighted_aggregate": promotion["weighted_aggregate"],
            },
            segments,
            config,
        )
    return _apply_segment_vetoes(
        {
            "aggregate": (
                summarize_pairs(
                    standard,
                    confidence=config.confidence,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed,
                )
                if standard
                else None
            ),
            "per_ring": {
                str(ring): summarize_pairs(
                    values,
                    confidence=config.confidence,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed + ring * 1_000_003,
                )
                for ring, values in per_ring.items()
                if values
            },
            "promotion": {
                "decision": "continue",
                "sequential_state": "continue",
                "reason": "incomplete_ring_coverage",
                "incomplete_rings": [
                    ring for ring, values in per_ring.items() if not values
                ],
                "pair_model": "complete-role-reversed-pairs-only",
            },
        },
        segments,
        config,
    )


def _wave_local_summary_config(
    config: ArenaConfig,
    pair_starts: Mapping[int, int] | None,
) -> ArenaConfig:
    if not config.promotion_pair_ratios or not any(
        int((pair_starts or {}).get(ring, 0)) > 0 for ring in config.rings
    ):
        return config
    # Continuation-wave pair indices are absolute so they can be merged with
    # durable prior waves. A wave-local weighted assessment cannot construct
    # macro blocks starting from zero without those prior pairs; promotion
    # recomputes the authoritative weighted summary after merging.
    return replace(
        config,
        promotion_pair_ratios={},
        required_regression_rings=None,
        weighted_initial_blocks=0,
        weighted_continuation_blocks=0,
        weighted_max_blocks=0,
    )


class ArenaRunner:
    def __init__(
        self,
        *,
        native_module: Any,
        candidate: ArenaEvaluatorProtocol,
        baseline: ArenaEvaluatorProtocol,
        config: ArenaConfig,
        baseline_search: ArenaSearchBudget | None = None,
        baseline_metadata: Mapping[str, object] | None = None,
        search_workers: int = 2,
    ) -> None:
        if search_workers not in (1, 2):
            raise ValueError("arena search_workers must be one or two")
        self.native = native_module
        self.candidate = candidate
        self.baseline = baseline
        self.config = config
        self.candidate_search = ArenaSearchBudget.from_config(config)
        self.baseline_search = baseline_search or self.candidate_search
        self.baseline_metadata = dict(baseline_metadata or {})
        self.search_workers = search_workers
        self._inference_calls = 0
        self._inference_seconds = 0.0
        self._inference_queue_wait_seconds = 0.0

    def run(
        self,
        *,
        progress: Callable[..., None] | None = None,
        pair_starts: Mapping[int, int] | None = None,
        pair_counts: Mapping[int, int] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        started_ns = time.time_ns()
        started = time.perf_counter()
        should_stop = stop_requested or (lambda: False)
        candidate_calls_before = int(getattr(self.candidate, "evaluator_calls", 0))
        candidate_rows_before = int(getattr(self.candidate, "evaluator_rows", 0))
        baseline_calls_before = int(getattr(self.baseline, "evaluator_calls", 0))
        baseline_rows_before = int(getattr(self.baseline, "evaluator_rows", 0))
        games: list[ArenaGame] = []
        pairs: list[ArenaPair] = []
        self._inference_calls = 0
        self._inference_seconds = 0.0
        self._inference_queue_wait_seconds = 0.0
        requested_pairs = sum(
            int((pair_counts or {}).get(ring, self.config.pairs_per_ring))
            for ring in self.config.rings
        )
        interrupted = False
        # Dynamo/Inductor compiled models are not thread-safe. Keep the two
        # GIL-releasing native search groups parallel, but route both models
        # through one stable inference thread for the entire arena run.
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="arena-inference",
        ) as inference_executor:
            for ring in self.config.rings:
                first_pair = int((pair_starts or {}).get(ring, 0))
                pair_count = int(
                    (pair_counts or {}).get(ring, self.config.pairs_per_ring)
                )
                final_pair = first_pair + pair_count
                chunk_size = self.config.pair_chunk_size or max(1, pair_count)
                for chunk_start in range(
                    first_pair,
                    final_pair,
                    chunk_size,
                ):
                    if should_stop():
                        interrupted = True
                        break
                    chunk_stop = min(
                        final_pair,
                        chunk_start + chunk_size,
                    )
                    specifications = self._pair_specifications(
                        ring, range(chunk_start, chunk_stop), STANDARD_VARIANT
                    )
                    completed = self._play_specifications(
                        ring,
                        specifications,
                        STANDARD_VARIANT,
                        games,
                        pairs,
                        progress=progress,
                        inference_executor=inference_executor,
                        stop_requested=should_stop,
                    )
                    if not completed or should_stop():
                        interrupted = True
                        break
                if interrupted:
                    break
                # Non-standard segments play alongside the standard pairs in
                # proportion to their configured counts; every batch stays
                # variant-homogeneous.
                for segment, per_ring in sorted(
                    self.config.segment_pairs_per_ring.items()
                ):
                    if should_stop():
                        interrupted = True
                        break
                    segment_first, segment_final = _segment_pair_range(
                        first_pair,
                        final_pair,
                        pairs_per_ring=self.config.pairs_per_ring,
                        segment_pairs_per_ring=per_ring,
                    )
                    by_variant: dict[GameVariant, list[int]] = {}
                    for pair in range(segment_first, segment_final):
                        variant = _segment_variant(
                            segment,
                            _opening_seed(self.config.seed, ring, pair),
                            self.config,
                        )
                        by_variant.setdefault(variant, []).append(pair)
                    for variant, variant_pairs in sorted(
                        by_variant.items(), key=lambda item: item[0].label
                    ):
                        specifications = self._pair_specifications(
                            ring, variant_pairs, variant
                        )
                        completed = self._play_specifications(
                            ring,
                            specifications,
                            variant,
                            games,
                            pairs,
                            progress=progress,
                            inference_executor=inference_executor,
                            stop_requested=should_stop,
                        )
                        if not completed or should_stop():
                            interrupted = True
                            break
                    if interrupted:
                        break
                if interrupted:
                    break
        summary_config = _wave_local_summary_config(self.config, pair_starts)
        statistical = (
            summarize_completed_arena_pairs(pairs, summary_config)
            if pairs
            else {
                "aggregate": None,
                "per_ring": {},
                "promotion": {
                    "decision": "continue",
                    "sequential_state": "continue",
                    "reason": "stopped_before_complete_pair",
                    "incomplete_rings": list(self.config.rings),
                    "pair_model": "complete-role-reversed-pairs-only",
                },
            }
        )
        candidate_calls = (
            int(getattr(self.candidate, "evaluator_calls", 0)) - candidate_calls_before
        )
        candidate_rows = (
            int(getattr(self.candidate, "evaluator_rows", 0)) - candidate_rows_before
        )
        baseline_calls = (
            int(getattr(self.baseline, "evaluator_calls", 0)) - baseline_calls_before
        )
        baseline_rows = (
            int(getattr(self.baseline, "evaluator_rows", 0)) - baseline_rows_before
        )
        if min(candidate_calls, candidate_rows, baseline_calls, baseline_rows) < 0:
            raise RuntimeError("arena evaluator metrics counters moved backwards")
        elapsed = time.perf_counter() - started
        total_rows = candidate_rows + baseline_rows
        baseline_metadata = dict(self.baseline_metadata)
        baseline_metadata.setdefault("kind", "checkpoint")
        baseline_metadata["identity"] = self.baseline.model_version
        baseline_metadata["search_budget"] = self.baseline_search.metadata()
        baseline_metadata["deterministic"] = True
        baseline_metadata["seed_schedule"] = "arena-runner-v2-pair-chunks"
        return {
            "schema_version": ARENA_RESULT_SCHEMA_VERSION,
            "candidate": self.candidate.model_version,
            "baseline": self.baseline.model_version,
            "baseline_metadata": baseline_metadata,
            "started_ns": started_ns,
            "completed_ns": time.time_ns(),
            "evaluation_metrics": {
                "wall_seconds": elapsed,
                "candidate_evaluator_calls": candidate_calls,
                "candidate_evaluator_rows": candidate_rows,
                "baseline_evaluator_calls": baseline_calls,
                "baseline_evaluator_rows": baseline_rows,
                "total_evaluator_calls": candidate_calls + baseline_calls,
                "total_evaluator_rows": total_rows,
                "evaluator_rows_per_second": (total_rows / elapsed if elapsed else 0.0),
                "serialized_inference_calls": self._inference_calls,
                "serialized_inference_seconds": self._inference_seconds,
                "inference_queue_wait_seconds": (self._inference_queue_wait_seconds),
                "requested_pairs": requested_pairs,
                "completed_pairs": len(pairs),
            },
            "search": {
                "deterministic": True,
                **self.candidate_search.metadata(),
                "pie_rule": "pie" in self.config.segment_pairs_per_ring,
                "segments": {
                    SEGMENT_STANDARD: self.config.pairs_per_ring,
                    **dict(sorted(self.config.segment_pairs_per_ring.items())),
                },
                "segment_handicaps": list(self.config.segment_handicaps),
                "segment_handicap_pda": list(self.config.segment_handicap_pda),
                "swap_dead_zone": self.config.swap_dead_zone,
                "search_workers": self.search_workers,
                "inference_workers": 1,
                "pair_chunk_size": self.config.pair_chunk_size,
                "effective_pair_chunking": (
                    "configured"
                    if self.config.pair_chunk_size is not None
                    else "full_requested_ring_batch"
                ),
            },
            "interrupted": interrupted,
            **statistical,
            "games": [asdict(game) for game in games],
            "pairs": [asdict(pair) for pair in pairs],
        }

    def _pair_specifications(
        self,
        ring: int,
        pair_indices: Sequence[int],
        variant: GameVariant,
    ) -> list[tuple[int, int, int, int | None]]:
        node_count = get_topology(ring).n
        specifications: list[tuple[int, int, int, int | None]] = []
        for pair in pair_indices:
            opening_seed = _opening_seed(self.config.seed, ring, pair)
            forced_opening = _forced_opening(
                opening_seed, self.config.unforced_opening_fraction
            )
            opening_action = opening_seed % node_count if forced_opening else None
            for candidate_player in (0, 1):
                specifications.append(
                    (pair, candidate_player, opening_seed, opening_action)
                )
        return specifications

    def _play_specifications(
        self,
        ring: int,
        specifications: Sequence[tuple[int, int, int, int | None]],
        variant: GameVariant,
        games: list[ArenaGame],
        pairs: list[ArenaPair],
        *,
        progress: Callable[..., None] | None,
        inference_executor: Executor,
        stop_requested: Callable[[], bool],
    ) -> bool:
        """Play one variant-homogeneous batch; return whether it completed."""

        ring_games = self._play_ring_batch(
            ring,
            specifications,
            variant=variant,
            progress=progress,
            inference_executor=inference_executor,
            stop_requested=stop_requested,
        )
        if len(ring_games) % 2:
            raise RuntimeError(
                "arena cancellation produced an incomplete role-reversed pair"
            )
        games.extend(ring_games)
        for offset in range(0, len(ring_games), 2):
            pair_games = ring_games[offset : offset + 2]
            pair = pair_games[0].pair
            if pair_games[1].pair != pair or {
                pair_games[0].candidate_player,
                pair_games[1].candidate_player,
            } != {0, 1}:
                raise RuntimeError("arena batch did not preserve role-reversed pairs")
            pairs.append(
                ArenaPair(
                    ring=ring,
                    pair=pair,
                    opening_seed=pair_games[0].opening_seed,
                    opening_action=pair_games[0].opening_action,
                    forced_opening=pair_games[0].forced_opening,
                    outcomes=(pair_games[0].outcome, pair_games[1].outcome),
                    variant=variant.label,
                    segment=variant.segment,
                )
            )
            if progress is not None:
                progress(
                    phase="arena",
                    ring=ring,
                    pair=pair,
                    variant=variant.label,
                    completed_pairs=len(pairs),
                )
        return len(ring_games) == len(specifications)

    def _pda_seats(self, variant: GameVariant) -> tuple[int, int]:
        """Seat advantages: the second player in a handicap game is advantaged."""

        if variant.handicap < 2:
            return (0, 0)
        advantage = self.config.segment_handicap_pda[variant.handicap - 2]
        return (-advantage, advantage)

    def _budgets(
        self,
        budget: ArenaSearchBudget,
        to_move: Sequence[int],
        pda_seats: tuple[int, int],
    ) -> list[int]:
        return [
            budget.simulations * (2 ** pda_seats[player])
            if pda_seats[player] > 0
            else budget.simulations
            for player in to_move
        ]

    def _play_ring_batch(
        self,
        ring: int,
        specifications: Sequence[tuple[int, int, int, int | None]],
        *,
        variant: GameVariant = STANDARD_VARIANT,
        progress: Callable[..., None] | None,
        inference_executor: Executor,
        stop_requested: Callable[[], bool],
    ) -> list[ArenaGame]:
        importer = getattr(self.native.StateBatch, "from_semantic", None)
        if not callable(importer):
            output: list[ArenaGame] = []
            for offset in range(0, len(specifications), 2):
                if stop_requested():
                    break
                pair_games = []
                for (
                    pair,
                    candidate_player,
                    opening_seed,
                    opening_action,
                ) in specifications[offset : offset + 2]:
                    game = self._play_game(
                        ring=ring,
                        pair=pair,
                        candidate_player=candidate_player,
                        opening_seed=opening_seed,
                        opening_action=opening_action,
                        variant=variant,
                        inference_executor=inference_executor,
                        stop_requested=stop_requested,
                    )
                    if game is None:
                        break
                    pair_games.append(game)
                if len(pair_games) != 2:
                    break
                output.extend(pair_games)
            return output
        states = self.native.StateBatch(
            ring,
            len(specifications),
            mode=variant.mode,
            handicap=variant.handicap,
            pie=variant.pie,
        )
        node_count = get_topology(ring).n
        pda_seats = self._pda_seats(variant)
        forced_rows = [
            index
            for index, specification in enumerate(specifications)
            if specification[3] is not None
        ]
        if forced_rows:
            states.apply_many(
                forced_rows,
                [cast(int, specifications[index][3]) for index in forced_rows],
            )
        searched_moves = [0] * len(specifications)
        swapped = [False] * len(specifications)
        wave = 0
        maximum_moves = node_count + 1
        with ThreadPoolExecutor(
            max_workers=self.search_workers,
            thread_name_prefix="arena-search",
        ) as executor:
            while True:
                if stop_requested():
                    break
                data = states.data()
                active = [
                    row for row, terminal in enumerate(data.terminal) if not terminal
                ]
                if not active:
                    break
                groups: dict[int, list[int]] = {0: [], 1: []}
                for row in active:
                    candidate_player = specifications[row][1]
                    evaluator_index = (
                        0 if int(data.to_move[row]) == candidate_player else 1
                    )
                    groups[evaluator_index].append(row)
                futures = []
                for evaluator_index, rows in groups.items():
                    if not rows:
                        continue
                    subset = self._semantic_subset(data, rows)
                    evaluator = (
                        self.candidate if evaluator_index == 0 else self.baseline
                    )
                    budget = (
                        self.candidate_search
                        if evaluator_index == 0
                        else self.baseline_search
                    )
                    seed = _batch_search_seed(
                        ring,
                        wave,
                        evaluator_index,
                        [specifications[row][2] for row in rows],
                    )
                    futures.append(
                        executor.submit(
                            self._search_group,
                            subset,
                            evaluator,
                            budget,
                            seed,
                            len(rows),
                            rows,
                            inference_executor,
                            stop_requested,
                            self._budgets(
                                budget,
                                [int(data.to_move[row]) for row in rows],
                                pda_seats,
                            ),
                            pda_seats,
                            [bool(data.swap_available[row]) for row in rows],
                            node_count,
                        )
                    )
                search_results = [future.result() for future in futures]
                if any(result is None for result in search_results):
                    break
                for result in search_results:
                    assert result is not None
                    rows, actions = result
                    states.apply_many(rows, actions)
                    for row, action in zip(rows, actions, strict=True):
                        searched_moves[row] += 1
                        if action == node_count:
                            swapped[row] = True
                        if searched_moves[row] > maximum_moves:
                            raise RuntimeError(
                                "batched arena game exceeded the move bound"
                            )
                wave += 1
                if progress is not None and wave % 16 == 0:
                    progress(
                        phase="arena_batch",
                        ring=ring,
                        variant=variant.label,
                        active_games=len(active),
                        wave=wave,
                    )
        terminal = [bool(value) for value in states.data().terminal]
        winners = [int(value) for value in states.score_data().winner]
        output = []
        for offset in range(0, len(specifications), 2):
            if offset + 1 >= len(specifications):
                break
            pair_rows = (offset, offset + 1)
            if not all(terminal[row] for row in pair_rows):
                break
            for row in pair_rows:
                pair, candidate_player, opening_seed, opening_action = specifications[
                    row
                ]
                winner = winners[row]
                if winner not in (0, 1):
                    raise RuntimeError("arena terminal result cannot be tied")
                outcome = 1 if winner == candidate_player else -1
                output.append(
                    ArenaGame(
                        ring=ring,
                        pair=pair,
                        candidate_player=candidate_player,
                        opening_seed=opening_seed,
                        opening_action=opening_action,
                        forced_opening=opening_action is not None,
                        winner=winner,
                        outcome=outcome,
                        searched_moves=searched_moves[row],
                        variant=variant.label,
                        segment=variant.segment,
                        swapped=swapped[row],
                        pda=max(pda_seats),
                    )
                )
        return output

    def _search_group(
        self,
        states: Any,
        evaluator: ArenaEvaluatorProtocol,
        budget: ArenaSearchBudget,
        seed: int,
        row_count: int,
        rows: Sequence[int],
        inference_executor: Executor,
        stop_requested: Callable[[], bool],
        budgets: Sequence[int] | None = None,
        pda_seats: tuple[int, int] = (0, 0),
        swap_available: Sequence[bool] | None = None,
        node_count: int | None = None,
    ) -> tuple[list[int], list[int]] | None:
        if stop_requested():
            return None
        options: dict[str, object] = {}
        if budgets is not None and any(
            value != budget.simulations for value in budgets
        ):
            options["simulations_per_root"] = [int(value) for value in budgets]
        if pda_seats != (0, 0):
            options["pda_by_seat"] = [pda_seats] * row_count
        search = self.native.SearchBatch(
            states,
            simulations=budget.simulations,
            max_considered=budget.max_considered,
            c_visit=budget.c_visit,
            c_scale=budget.c_scale,
            deterministic_seed=seed,
            **options,
        )
        roots = search.root_requests()
        response = self._evaluate_serialized(
            inference_executor,
            evaluator,
            roots,
        )
        if stop_requested():
            return None
        search.initialize_roots(*response.submit_args())
        guard = 0
        guard_limit = (
            max(budgets) if budgets else budget.simulations
        ) * row_count * 4 + 16
        while not search.is_done():
            if stop_requested():
                return None
            guard += 1
            if guard > guard_limit:
                raise RuntimeError("batched arena search failed to make progress")
            requests = search.next_requests()
            if len(requests) == 0:
                continue
            response = self._evaluate_serialized(
                inference_executor,
                evaluator,
                requests,
            )
            if stop_requested():
                return None
            search.submit(*response.submit_args())
        results = search.results()
        actions = [int(value) for value in results.selected_actions]
        if len(actions) != len(rows):
            raise RuntimeError("batched arena search returned the wrong row count")
        if any(action < 0 for action in actions):
            raise RuntimeError("active batched arena row returned invalid placement")
        if swap_available is not None and any(swap_available):
            if node_count is None:
                raise RuntimeError("pie arena rows require the node count")
            root_values = [float(value) for value in results.root_values]
            for index, available in enumerate(swap_available):
                if available and root_values[index] < -self.config.swap_dead_zone:
                    actions[index] = node_count
        return list(rows), actions

    def _evaluate_serialized(
        self,
        inference_executor: Executor,
        evaluator: ArenaEvaluatorProtocol,
        requests: NativeEvalBatchProtocol,
    ) -> InferenceResponse:
        submitted = time.perf_counter()

        def evaluate() -> InferenceResponse:
            started = time.perf_counter()
            self._inference_queue_wait_seconds += started - submitted
            try:
                return evaluator.evaluate(requests)
            finally:
                self._inference_calls += 1
                self._inference_seconds += time.perf_counter() - started

        return inference_executor.submit(evaluate).result()

    def _semantic_subset(self, data: Any, rows: Sequence[int]) -> Any:
        def words(name: str) -> list[int]:
            source = list(getattr(data, name))
            output = []
            for row in rows:
                start = row * BITBOARD_WORDS
                output.extend(source[start : start + BITBOARD_WORDS])
            return output

        def column(name: str, convert: Callable[[Any], Any]) -> list[Any]:
            source = getattr(data, name)
            return [convert(source[row]) for row in rows]

        return self.native.StateBatch.from_semantic(
            int(data.rings),
            words("zero_bits"),
            words("one_bits"),
            column("to_move", int),
            column("moves_left", int),
            column("opening", bool),
            mode=column("mode", int),
            handicap=column("handicap", int),
            pie=column("pie", bool),
            swap_available=column("swap_available", bool),
            swapped=column("swapped", bool),
            current_turn_bits=words("current_turn_bits"),
            previous_turn_bits=words("previous_turn_bits"),
            own_previous_turn_bits=words("own_previous_turn_bits"),
            handicap_bits=words("handicap_bits"),
        )

    def _play_game(
        self,
        *,
        ring: int,
        pair: int,
        candidate_player: int,
        opening_seed: int,
        opening_action: int | None,
        variant: GameVariant = STANDARD_VARIANT,
        inference_executor: Executor,
        stop_requested: Callable[[], bool],
    ) -> ArenaGame | None:
        states = self.native.StateBatch(
            ring, 1, mode=variant.mode, handicap=variant.handicap, pie=variant.pie
        )
        node_count = get_topology(ring).n
        pda_seats = self._pda_seats(variant)
        # Both games in a pair receive the same legal one-stone opening, so
        # role reversal cannot alter it; in a pie game the responder may swap.
        if opening_action is not None:
            states.apply_many([0], [opening_action])
        moves = 0
        swapped = False
        maximum_moves = node_count + 1
        while True:
            if stop_requested():
                return None
            state_data = states.data()
            if bool(state_data.terminal[0]):
                break
            player = int(state_data.to_move[0])
            evaluator = self.candidate if player == candidate_player else self.baseline
            budget = (
                self.candidate_search
                if player == candidate_player
                else self.baseline_search
            )
            result = self._search_group(
                states,
                evaluator,
                budget,
                _search_seed(opening_seed, moves),
                1,
                [0],
                inference_executor,
                stop_requested,
                self._budgets(budget, [player], pda_seats),
                pda_seats,
                [bool(state_data.swap_available[0])],
                node_count,
            )
            if result is None:
                return None
            action = result[1][0]
            if action == node_count:
                swapped = True
            states.apply_many([0], [action])
            moves += 1
            if moves > maximum_moves:
                raise RuntimeError("arena game exceeded the move bound")
        winner = int(states.score_data().winner[0])
        if winner not in (0, 1):
            raise RuntimeError("arena terminal result cannot be tied")
        outcome = 1 if winner == candidate_player else -1
        return ArenaGame(
            ring=ring,
            pair=pair,
            candidate_player=candidate_player,
            opening_seed=opening_seed,
            opening_action=opening_action,
            forced_opening=opening_action is not None,
            winner=winner,
            outcome=outcome,
            searched_moves=moves,
            variant=variant.label,
            segment=variant.segment,
            swapped=swapped,
            pda=max(pda_seats),
        )


def _expected_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def _opening_seed(seed: int, ring: int, pair: int) -> int:
    value = (
        (seed & ((1 << 64) - 1))
        ^ (ring * 0x9E3779B97F4A7C15)
        ^ (pair * 0xD1B54A32D192ED03)
    )
    value ^= value >> 30
    value *= 0xBF58476D1CE4E5B9
    value &= (1 << 64) - 1
    value ^= value >> 27
    value *= 0x94D049BB133111EB
    value &= (1 << 64) - 1
    return value ^ (value >> 31)


def _search_seed(opening_seed: int, move: int) -> int:
    return (opening_seed + (move + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)


def _batch_search_seed(
    ring: int,
    wave: int,
    evaluator_index: int,
    opening_seeds: Sequence[int],
) -> int:
    value = ring ^ (wave << 8) ^ (evaluator_index << 56)
    for seed in opening_seeds:
        value ^= seed
        value = (value * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    return value


def _forced_opening(opening_seed: int, unforced_fraction: float) -> bool:
    threshold = int(unforced_fraction * (1 << 64))
    return opening_seed >= threshold


def _segment_pair_range(
    first_pair: int,
    final_pair: int,
    *,
    pairs_per_ring: int,
    segment_pairs_per_ring: int,
) -> tuple[int, int]:
    """Map a standard pair range onto a segment's proportional pair range.

    Pair indices stay absolute across continuation waves so durable prior
    waves merge without duplicates.
    """

    scale = segment_pairs_per_ring / pairs_per_ring
    start = math.ceil(first_pair * scale)
    stop = math.ceil(final_pair * scale)
    return start, max(start, stop)


def _segment_variant(
    segment: str, opening_seed: int, config: ArenaConfig
) -> GameVariant:
    """Deterministically pick the concrete variant of one segment pair."""

    if segment == "classic":
        return GameVariant(mode="classic")
    if segment == "handicap":
        handicaps = config.segment_handicaps
        return GameVariant(
            mode="double", handicap=handicaps[(opening_seed >> 8) % len(handicaps)]
        )
    if segment == "pie":
        return GameVariant(
            mode="classic" if (opening_seed >> 16) & 1 else "double", pie=True
        )
    raise ValueError(f"unknown arena segment {segment!r}")
