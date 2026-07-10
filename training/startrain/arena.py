"""Deterministic paired arena matches and conservative promotion statistics."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Protocol

from .config import ArenaConfig
from .inference import InferenceResponse, NativeEvalBatchProtocol
from .native import BITBOARD_WORDS
from .topology import get_topology


class ArenaEvaluatorProtocol(Protocol):
    model_version: str

    def evaluate(self, requests: NativeEvalBatchProtocol) -> InferenceResponse: ...


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
        "pentanomial": {
            "0": counts[0],
            "0.5": counts[1],
            "1": counts[2],
            "1.5": counts[3],
            "2": counts[4],
        },
    }


def promotion_assessment(
    aggregate: Sequence[ArenaPair],
    per_ring: Mapping[int, Sequence[ArenaPair]],
    config: ArenaConfig,
) -> dict[str, object]:
    if not aggregate:
        raise ValueError("SPRT assessment requires paired games")
    score_rate = sum(pair.score_rate for pair in aggregate) / len(aggregate)
    lower, _ = pair_confidence_sequence(aggregate, error_probability=config.alpha)
    _, upper = pair_confidence_sequence(aggregate, error_probability=config.beta)
    probability_null = _expected_score(config.null_elo)
    probability_alternative = _expected_score(config.alternative_elo)
    minimum_ready = all(
        len(per_ring[ring]) >= config.minimum_pairs_per_ring for ring in config.rings
    )
    if minimum_ready and lower > probability_null:
        sprt_state = "accept_alternative"
    elif minimum_ready and upper < probability_alternative:
        sprt_state = "accept_null"
    else:
        sprt_state = "continue"

    ring_floors: dict[str, object] = {}
    floors_pass = True
    floors_ready = True
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
                "pairs": len(result),
                "passed": None,
            }
            continue
        lower_score, _ = _paired_bootstrap_interval(
            result,
            confidence=config.confidence,
            samples=config.bootstrap_samples,
            seed=config.seed + ring * 1_000_003,
        )
        lower_elo = elo_from_probability(lower_score)
        passed = lower_elo >= floor
        floors_pass = floors_pass and passed
        ring_floors[str(ring)] = {
            "floor_elo": floor,
            "paired_bootstrap_lower_elo": lower_elo,
            "pairs": len(result),
            "passed": passed,
        }

    if sprt_state == "accept_alternative" and floors_ready and floors_pass:
        decision = "promote"
    elif sprt_state == "accept_null":
        decision = "reject"
    elif floors_ready and not floors_pass:
        decision = "reject_ring_regression"
    else:
        decision = "continue"
    return {
        "decision": decision,
        "sequential_state": sprt_state,
        "pair_score_rate": score_rate,
        "confidence_sequence": [lower, upper],
        "null_elo": config.null_elo,
        "alternative_elo": config.alternative_elo,
        "pair_model": "pair-level-anytime-hoeffding-confidence-sequence",
        "error_spending": "delta_n=delta*6/(pi^2*n^2)",
        "ring_floors": ring_floors,
    }


sprt_assessment = promotion_assessment


def pair_confidence_sequence(
    pairs: Sequence[ArenaPair], *, error_probability: float
) -> tuple[float, float]:
    """Anytime-valid bounds over independent opening-pair observations.

    A pair contributes its role-reversed average score in ``[0, 1]``. This
    retains arbitrary correlation between the two games. Hoeffding bounds with
    a ``6/(pi² n²)`` error-spending schedule remain valid over repeated looks.
    """

    if not pairs or not 0 < error_probability < 1:
        raise ValueError("pair confidence sequence inputs are invalid")
    count = len(pairs)
    mean = sum(pair.score_rate for pair in pairs) / count
    allocated = error_probability * 6.0 / (math.pi * math.pi * count * count)
    radius = math.sqrt(math.log(1.0 / allocated) / (2.0 * count))
    return max(0.0, mean - radius), min(1.0, mean + radius)


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
        native_module: object,
        candidate: ArenaEvaluatorProtocol,
        baseline: ArenaEvaluatorProtocol,
        config: ArenaConfig,
    ) -> None:
        self.native = native_module
        self.candidate = candidate
        self.baseline = baseline
        self.config = config

    def run(
        self,
        *,
        progress: Callable[..., None] | None = None,
        pair_starts: Mapping[int, int] | None = None,
        pair_counts: Mapping[int, int] | None = None,
    ) -> dict[str, object]:
        started_ns = time.time_ns()
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
        return {
            "schema_version": 1,
            "candidate": self.candidate.model_version,
            "baseline": self.baseline.model_version,
            "started_ns": started_ns,
            "completed_ns": time.time_ns(),
            "search": {
                "deterministic": True,
                "simulations": self.config.simulations,
                "max_considered": self.config.max_considered,
                "c_visit": self.config.c_visit,
                "c_scale": self.config.c_scale,
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
                [int(specifications[index][3]) for index in forced_rows],
            )
        searched_moves = [0] * len(specifications)
        wave = 0
        maximum_moves = 2 * get_topology(ring).n + 2
        while True:
            data = states.data()
            active = [row for row, terminal in enumerate(data.terminal) if not terminal]
            if not active:
                break
            groups: dict[int, list[int]] = {0: [], 1: []}
            for row in active:
                candidate_player = specifications[row][1]
                evaluator_index = 0 if int(data.to_move[row]) == candidate_player else 1
                groups[evaluator_index].append(row)
            for evaluator_index, rows in groups.items():
                if not rows:
                    continue
                subset = self._semantic_subset(data, rows)
                evaluator = self.candidate if evaluator_index == 0 else self.baseline
                search = self.native.SearchBatch(
                    subset,
                    simulations=self.config.simulations,
                    max_considered=self.config.max_considered,
                    c_visit=self.config.c_visit,
                    c_scale=self.config.c_scale,
                    deterministic_seed=_batch_search_seed(
                        ring,
                        wave,
                        evaluator_index,
                        [specifications[row][2] for row in rows],
                    ),
                )
                roots = search.root_requests()
                response = evaluator.evaluate(roots)
                search.initialize_roots(*response.submit_args())
                guard = 0
                while not search.is_done():
                    guard += 1
                    if guard > self.config.simulations * len(rows) * 4 + 16:
                        raise RuntimeError(
                            "batched arena search failed to make progress"
                        )
                    requests = search.next_requests()
                    if len(requests) == 0:
                        continue
                    response = evaluator.evaluate(requests)
                    search.submit(*response.submit_args())
                results = search.results()
                actions = [int(value) for value in results.selected_actions]
                if any(action == -2 for action in actions):
                    raise RuntimeError("active batched arena row became terminal")
                states.apply_many(rows, actions)
                for row in rows:
                    searched_moves[row] += 1
                    if searched_moves[row] > maximum_moves:
                        raise RuntimeError("batched arena game exceeded the move bound")
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

    def _semantic_subset(self, data: object, rows: Sequence[int]) -> object:
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
            search = self.native.SearchBatch(
                states,
                simulations=self.config.simulations,
                max_considered=self.config.max_considered,
                c_visit=self.config.c_visit,
                c_scale=self.config.c_scale,
                deterministic_seed=_search_seed(opening_seed, moves),
            )
            roots = search.root_requests()
            root_response = evaluator.evaluate(roots)
            search.initialize_roots(*root_response.submit_args())
            guard = 0
            while not search.is_done():
                guard += 1
                if guard > self.config.simulations * 4 + 16:
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
