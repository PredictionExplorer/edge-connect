"""Deterministic paired arena matches and conservative promotion statistics."""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Literal, Protocol, cast

from .config import ArenaConfig
from .inference import InferenceResponse, NativeEvalBatchProtocol
from .native import BITBOARD_WORDS
from .topology import get_topology


ARENA_RESULT_SCHEMA_VERSION = 2


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


# A complete role-reversed pair has 0, 0.5, 1, 1.5, or 2 candidate points.
# The sequential test works on the corresponding [0, 1] score rates. Each
# betting fraction is mixed with equal initial capital, so no data-dependent
# tuning or multiplicity correction is needed.
_PAIR_SCORE_RATES = (0.0, 0.25, 0.5, 0.75, 1.0)
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
class WDL:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def score_rate(self) -> float:
        return self.score / self.games if self.games else 0.5

    def record(self, outcome: int) -> None:
        if outcome > 0:
            self.wins += 1
        elif outcome < 0:
            self.losses += 1
        else:
            self.draws += 1

    def add(self, other: "WDL") -> None:
        self.wins += other.wins
        self.draws += other.draws
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


@dataclass(frozen=True, slots=True)
class ArenaPair:
    ring: int
    pair: int
    opening_seed: int
    opening_action: int | None
    forced_opening: bool
    outcomes: tuple[int, int]

    @property
    def points(self) -> float:
        return sum(
            1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in self.outcomes
        )

    @property
    def score_rate(self) -> float:
        return self.points / 2.0


def wilson_interval(result: WDL, *, confidence: float = 0.95) -> tuple[float, float]:
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


def summarize_wdl(result: WDL, *, confidence: float) -> dict[str, object]:
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
    wdl = WDL()
    for pair in pairs:
        for outcome in pair.outcomes:
            wdl.record(outcome)
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
    counts = _pentanomial_counts(pairs)
    return {
        **asdict(wdl),
        "games": wdl.games,
        "pairs": len(pairs),
        "score_rate": wdl.score_rate,
        "elo_difference": elo_from_probability(wdl.score_rate),
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
        "pentanomial": {
            "0": counts[0],
            "0.5": counts[1],
            "1": counts[2],
            "1.5": counts[3],
            "2": counts[4],
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
    counts = _pentanomial_counts(aggregate)
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

    ring_floors: dict[str, object] = {}
    floors_pass = True
    floors_ready = True
    floor_error_probability = (1.0 - config.confidence) / 2.0
    for ring in config.rings:
        result = per_ring[ring]
        floor = config.per_ring_regression_floor_elo.get(
            ring, config.regression_floor_elo
        )
        if len(result) < config.minimum_pairs_per_ring:
            floors_ready = False
            floors_pass = False
            ring_floors[str(ring)] = {
                "floor_elo": floor,
                "paired_bootstrap_lower_elo": None,
                "anytime_lower_elo": None,
                "error_probability": floor_error_probability,
                "pairs": len(result),
                "passed": None,
            }
            continue
        bootstrap_lower_score, _ = _paired_bootstrap_interval(
            result,
            confidence=config.confidence,
            samples=config.bootstrap_samples,
            seed=config.seed + ring * 1_000_003,
        )
        anytime_lower_score, _ = pair_confidence_sequence(
            result,
            error_probability=floor_error_probability,
        )
        floor_score_rate = _expected_score(floor)
        passed = _pair_mean_exceeds(
            _pentanomial_counts(result),
            null_score_rate=floor_score_rate,
            error_probability=floor_error_probability,
        )
        floors_pass = floors_pass and passed
        ring_floors[str(ring)] = {
            "floor_elo": floor,
            "paired_bootstrap_lower_elo": elo_from_probability(bootstrap_lower_score),
            "anytime_lower_elo": elo_from_probability(anytime_lower_score),
            "error_probability": floor_error_probability,
            "test": "pair-level-mixture-betting-confidence-sequence-v1",
            "pairs": len(result),
            "passed": passed,
        }

    if sequential_state == "accept_alternative" and floors_ready and floors_pass:
        decision = "promote"
    elif sequential_state == "accept_null":
        decision = "reject"
    elif floors_ready and not floors_pass:
        decision = "reject_ring_regression"
    else:
        decision = "continue"
    return {
        "decision": decision,
        "sequential_state": sequential_state,
        "pair_score_rate": score_rate,
        "confidence_sequence": [lower, upper],
        "null_elo": config.null_elo,
        "alternative_elo": config.alternative_elo,
        "pair_model": "pair-level-mixture-betting-e-process-v1",
        "statistical_test": {
            "schema_version": 1,
            "name": "bounded-mean-mixture-betting-e-process",
            "observation_unit": "complete-role-reversed-pair",
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


# Retained only as an import-compatibility alias. The implemented method is an
# e-process, not a sequential probability ratio test.
sprt_assessment = promotion_assessment


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
    counts = _pentanomial_counts(pairs)
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


def _pair_mean_exceeds(
    counts: Sequence[int],
    *,
    null_score_rate: float,
    error_probability: float,
) -> bool:
    if not 0 < error_probability < 1:
        raise ValueError("pair mean test error probability is invalid")
    return _pair_log_e_value_from_counts(
        counts,
        null_score_rate=null_score_rate,
        direction="greater",
    ) >= math.log(1.0 / error_probability)


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


def _pentanomial_counts(pairs: Sequence[ArenaPair]) -> tuple[int, ...]:
    counts = [0, 0, 0, 0, 0]
    for pair in pairs:
        counts[int(round(pair.points * 2))] += 1
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


def summarize_arena_pairs(
    pairs: Sequence[ArenaPair], config: ArenaConfig
) -> dict[str, object]:
    if not pairs:
        raise ValueError("arena summary requires pairs")
    per_ring: dict[int, list[ArenaPair]] = {ring: [] for ring in config.rings}
    for pair in pairs:
        if pair.ring not in per_ring:
            raise ValueError("arena pair has an unconfigured ring")
        per_ring[pair.ring].append(pair)
    if any(not values for values in per_ring.values()):
        raise ValueError("arena summary requires at least one pair per ring")
    return {
        "aggregate": summarize_pairs(
            pairs,
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
        "promotion": promotion_assessment(pairs, per_ring, config),
    }


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
    ) -> None:
        self.native = native_module
        self.candidate = candidate
        self.baseline = baseline
        self.config = config
        self.candidate_search = ArenaSearchBudget.from_config(config)
        self.baseline_search = baseline_search or self.candidate_search
        self.baseline_metadata = dict(baseline_metadata or {})

    def run(
        self,
        *,
        progress: Callable[..., None] | None = None,
        pair_starts: Mapping[int, int] | None = None,
        pair_counts: Mapping[int, int] | None = None,
    ) -> dict[str, object]:
        started_ns = time.time_ns()
        started = time.perf_counter()
        candidate_calls_before = int(getattr(self.candidate, "evaluator_calls", 0))
        candidate_rows_before = int(getattr(self.candidate, "evaluator_rows", 0))
        baseline_calls_before = int(getattr(self.baseline, "evaluator_calls", 0))
        baseline_rows_before = int(getattr(self.baseline, "evaluator_rows", 0))
        games: list[ArenaGame] = []
        pairs: list[ArenaPair] = []
        for ring in self.config.rings:
            node_count = get_topology(ring).n
            first_pair = int((pair_starts or {}).get(ring, 0))
            pair_count = int((pair_counts or {}).get(ring, self.config.pairs_per_ring))
            specifications: list[tuple[int, int, int, int | None]] = []
            for pair in range(first_pair, first_pair + pair_count):
                opening_seed = _opening_seed(self.config.seed, ring, pair)
                forced_opening = _forced_opening(
                    opening_seed, self.config.unforced_opening_fraction
                )
                opening_action = opening_seed % node_count if forced_opening else None
                for candidate_player in (0, 1):
                    specifications.append(
                        (
                            pair,
                            candidate_player,
                            opening_seed,
                            opening_action,
                        )
                    )
            ring_games = self._play_ring_batch(ring, specifications, progress=progress)
            games.extend(ring_games)
            for offset in range(0, len(ring_games), 2):
                pair_games = ring_games[offset : offset + 2]
                pair = pair_games[0].pair
                opening_seed = pair_games[0].opening_seed
                opening_action = pair_games[0].opening_action
                forced_opening = pair_games[0].forced_opening
                completed_pair = ArenaPair(
                    ring=ring,
                    pair=pair,
                    opening_seed=opening_seed,
                    opening_action=opening_action,
                    forced_opening=forced_opening,
                    outcomes=(
                        pair_games[0].outcome,
                        pair_games[1].outcome,
                    ),
                )
                pairs.append(completed_pair)
                if progress is not None:
                    progress(
                        phase="arena",
                        ring=ring,
                        pair=pair,
                        completed_pairs=len(pairs),
                    )
        statistical = summarize_arena_pairs(pairs, self.config)
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
        baseline_metadata["seed_schedule"] = "arena-runner-v1"
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
            },
            "search": {
                "deterministic": True,
                **self.candidate_search.metadata(),
                "pie_rule": False,
            },
            **statistical,
            "games": [asdict(game) for game in games],
            "pairs": [asdict(pair) for pair in pairs],
        }

    def _play_ring_batch(
        self,
        ring: int,
        specifications: Sequence[tuple[int, int, int, int | None]],
        *,
        progress: Callable[..., None] | None,
    ) -> list[ArenaGame]:
        importer = getattr(self.native.StateBatch, "from_semantic", None)
        if not callable(importer):
            return [
                self._play_game(
                    ring=ring,
                    pair=pair,
                    candidate_player=candidate_player,
                    opening_seed=opening_seed,
                    opening_action=opening_action,
                )
                for pair, candidate_player, opening_seed, opening_action in specifications
            ]
        states = self.native.StateBatch(ring, len(specifications))
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
        wave = 0
        maximum_moves = 2 * get_topology(ring).n + 2
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="arena-search",
        ) as executor:
            while True:
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
                        )
                    )
                for future in futures:
                    rows, actions = future.result()
                    states.apply_many(rows, actions)
                    for row in rows:
                        searched_moves[row] += 1
                        if searched_moves[row] > maximum_moves:
                            raise RuntimeError(
                                "batched arena game exceeded the move bound"
                            )
                wave += 1
                if progress is not None and wave % 16 == 0:
                    progress(
                        phase="arena_batch",
                        ring=ring,
                        active_games=len(active),
                        wave=wave,
                    )
        winners = [int(value) for value in states.score_data().winner]
        output = []
        for row, specification in enumerate(specifications):
            pair, candidate_player, opening_seed, opening_action = specification
            winner = winners[row]
            outcome = 0 if winner == -1 else 1 if winner == candidate_player else -1
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
    ) -> tuple[list[int], list[int]]:
        search = self.native.SearchBatch(
            states,
            simulations=budget.simulations,
            max_considered=budget.max_considered,
            c_visit=budget.c_visit,
            c_scale=budget.c_scale,
            deterministic_seed=seed,
        )
        roots = search.root_requests()
        response = evaluator.evaluate(roots)
        search.initialize_roots(*response.submit_args())
        guard = 0
        while not search.is_done():
            guard += 1
            if guard > budget.simulations * row_count * 4 + 16:
                raise RuntimeError("batched arena search failed to make progress")
            requests = search.next_requests()
            if len(requests) == 0:
                continue
            response = evaluator.evaluate(requests)
            search.submit(*response.submit_args())
        results = search.results()
        actions = [int(value) for value in results.selected_actions]
        if len(actions) != len(rows):
            raise RuntimeError("batched arena search returned the wrong row count")
        if any(action == -2 for action in actions):
            raise RuntimeError("active batched arena row became terminal")
        return list(rows), actions

    def _semantic_subset(self, data: Any, rows: Sequence[int]) -> Any:
        def words(name: str) -> list[int]:
            source = list(getattr(data, name))
            output = []
            for row in rows:
                start = row * BITBOARD_WORDS
                output.extend(source[start : start + BITBOARD_WORDS])
            return output

        return self.native.StateBatch.from_semantic(
            int(data.rings),
            words("zero_bits"),
            words("one_bits"),
            [int(data.to_move[row]) for row in rows],
            [int(data.moves_left[row]) for row in rows],
            [bool(data.opening[row]) for row in rows],
            [int(data.pass_streak[row]) for row in rows],
        )

    def _play_game(
        self,
        *,
        ring: int,
        pair: int,
        candidate_player: int,
        opening_seed: int,
        opening_action: int | None,
    ) -> ArenaGame:
        states = self.native.StateBatch(ring, 1)
        # Both games in a pair receive the same legal one-stone opening. The
        # native rules expose no swap/pie action, so role reversal cannot alter it.
        if opening_action is not None:
            states.apply_many([0], [opening_action])
        moves = 0
        # A placement can be preceded by a non-terminal pass; two consecutive
        # passes end the game, so this remains a strict finite upper bound.
        maximum_moves = 2 * get_topology(ring).n + 2
        while True:
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
            search = self.native.SearchBatch(
                states,
                simulations=budget.simulations,
                max_considered=budget.max_considered,
                c_visit=budget.c_visit,
                c_scale=budget.c_scale,
                deterministic_seed=_search_seed(opening_seed, moves),
            )
            roots = search.root_requests()
            root_response = evaluator.evaluate(roots)
            search.initialize_roots(*root_response.submit_args())
            guard = 0
            while not search.is_done():
                guard += 1
                if guard > budget.simulations * 4 + 16:
                    raise RuntimeError("arena native search failed to make progress")
                requests = search.next_requests()
                if len(requests) == 0:
                    continue
                response = evaluator.evaluate(requests)
                search.submit(*response.submit_args())
            results = search.results()
            if bool(results.terminal[0]) or int(results.selected_actions[0]) == -2:
                raise RuntimeError("arena search marked an active state terminal")
            states.apply_many([0], [int(results.selected_actions[0])])
            moves += 1
            if moves > maximum_moves:
                raise RuntimeError("arena game exceeded the no-pie move bound")
        winner = int(states.score_data().winner[0])
        outcome = 0 if winner == -1 else (1 if winner == candidate_player else -1)
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
