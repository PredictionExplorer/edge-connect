"""Deterministic connected Bradley-Terry ratings for checkpoint arenas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

_ELO_PER_LOGIT = 400.0 / math.log(10.0)


@dataclass(frozen=True, slots=True)
class DecisiveMatch:
    """Aggregate decisive outcomes from one directed arena result."""

    candidate: str
    baseline: str
    wins: int
    losses: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate, str)
            or not self.candidate
            or self.candidate.strip() != self.candidate
            or not isinstance(self.baseline, str)
            or not self.baseline
            or self.baseline.strip() != self.baseline
        ):
            raise ValueError(
                "match identities must be non-empty, whitespace-trimmed strings"
            )
        if self.candidate == self.baseline:
            raise ValueError("a checkpoint cannot be matched against itself")
        if any(
            type(value) is not int or value < 0 for value in (self.wins, self.losses)
        ):
            raise ValueError("decisive result counts must be non-negative integers")
        if self.wins + self.losses <= 0:
            raise ValueError("a decisive match must contain at least one game")


@dataclass(frozen=True, slots=True)
class EloEstimate:
    """One checkpoint's anchored Elo estimate and marginal uncertainty."""

    identity: str
    rating: float
    standard_error: float
    confidence_interval: tuple[float, float]
    decisive_games: int


@dataclass(frozen=True, slots=True)
class BradleyTerryFit:
    """A fit for the graph component containing ``anchor_identity``."""

    anchor_identity: str
    estimates: tuple[EloEstimate, ...]
    components: tuple[tuple[str, ...], ...]
    excluded_identities: tuple[str, ...]
    observation_count: int
    unique_pairing_count: int
    decisive_games: int
    continuity_corrected_pairings: int
    confidence_level: float
    converged: bool
    iterations: int
    log_likelihood: float

    @property
    def connected(self) -> bool:
        return len(self.components) == 1


@dataclass(slots=True)
class _PairTotals:
    first_wins: int = 0
    second_wins: int = 0
    observations: int = 0


def fit_bradley_terry_elo(
    matches: list[DecisiveMatch] | tuple[DecisiveMatch, ...],
    *,
    anchor_identity: str,
    confidence: float = 0.95,
    continuity_correction: float = 0.5,
) -> BradleyTerryFit:
    """Fit an anchored Bradley-Terry model to aggregate decisive outcomes.

    Repeated and oppositely directed records are first reduced to unordered
    checkpoint pairings. The fit uses only the connected component containing
    the fixed anchor, because components without a path to that anchor have no
    identifiable relative Elo. A symmetric continuity correction is applied
    only to one-sided pairings so that separated arena graphs retain finite,
    deterministic estimates. Reported standard errors use the raw (uncorrected)
    observed-information Hessian at the fitted ratings.
    """

    if not isinstance(anchor_identity, str) or not anchor_identity:
        raise ValueError("anchor_identity must be a non-empty string")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")
    if not math.isfinite(continuity_correction) or continuity_correction <= 0.0:
        raise ValueError("continuity_correction must be finite and positive")
    if not matches:
        raise ValueError("Bradley-Terry fit requires at least one decisive match")
    if any(not isinstance(match, DecisiveMatch) for match in matches):
        raise TypeError("matches must contain only DecisiveMatch values")

    totals: dict[tuple[str, str], _PairTotals] = {}
    adjacency: dict[str, set[str]] = {}
    for match in matches:
        first, second = sorted((match.candidate, match.baseline))
        total = totals.setdefault((first, second), _PairTotals())
        if match.candidate == first:
            total.first_wins += match.wins
            total.second_wins += match.losses
        else:
            total.first_wins += match.losses
            total.second_wins += match.wins
        total.observations += 1
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)

    if anchor_identity not in adjacency:
        raise ValueError("anchor_identity is absent from the comparison graph")

    components = _connected_components(adjacency, anchor_identity)
    anchor_component = set(components[0])
    excluded = tuple(identity for component in components[1:] for identity in component)
    pairs = [
        (first, second, total)
        for (first, second), total in sorted(totals.items())
        if first in anchor_component and second in anchor_component
    ]
    identities = sorted(anchor_component)
    variables = [identity for identity in identities if identity != anchor_identity]
    variable_index = {identity: index for index, identity in enumerate(variables)}

    raw_pairs = [
        (
            first,
            second,
            float(total.first_wins),
            float(total.second_wins),
        )
        for first, second, total in pairs
    ]
    adjusted_pairs: list[tuple[str, str, float, float]] = []
    corrected = 0
    for first, second, total in pairs:
        first_wins = float(total.first_wins)
        second_wins = float(total.second_wins)
        if first_wins == 0.0 or second_wins == 0.0:
            first_wins += continuity_correction
            second_wins += continuity_correction
            corrected += 1
        adjusted_pairs.append((first, second, first_wins, second_wins))

    logits = {identity: 0.0 for identity in identities}
    converged = not variables
    iterations = 0
    for iteration in range(1, 201):
        gradient, information = _gradient_and_information(
            adjusted_pairs,
            logits,
            variable_index,
        )
        if not gradient or max(abs(value) for value in gradient) <= 1e-11:
            converged = True
            iterations = iteration - 1
            break
        delta = _solve(information, gradient)
        objective = _log_likelihood(adjusted_pairs, logits)
        scale = 1.0
        accepted = False
        while scale >= 2.0**-40:
            proposal = dict(logits)
            for identity, index in variable_index.items():
                proposal[identity] += scale * delta[index]
            proposed_objective = _log_likelihood(adjusted_pairs, proposal)
            if proposed_objective >= objective - 1e-12:
                logits = proposal
                accepted = True
                break
            scale *= 0.5
        iterations = iteration
        if not accepted:
            break
        if max(abs(scale * value) for value in delta) <= 1e-11:
            converged = True
            break

    _, raw_information = _gradient_and_information(
        raw_pairs,
        logits,
        variable_index,
    )
    covariance = _inverse(raw_information) if variables else []
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    games_by_identity = {identity: 0 for identity in identities}
    for first, second, first_wins, second_wins in raw_pairs:
        games = int(first_wins + second_wins)
        games_by_identity[first] += games
        games_by_identity[second] += games

    estimates = []
    for identity in identities:
        rating = logits[identity] * _ELO_PER_LOGIT
        if identity == anchor_identity:
            rating = 0.0
            standard_error = 0.0
        else:
            variance = covariance[variable_index[identity]][variable_index[identity]]
            standard_error = math.sqrt(max(0.0, variance)) * _ELO_PER_LOGIT
        estimates.append(
            EloEstimate(
                identity=identity,
                rating=rating,
                standard_error=standard_error,
                confidence_interval=(
                    rating - z_value * standard_error,
                    rating + z_value * standard_error,
                ),
                decisive_games=games_by_identity[identity],
            )
        )
    estimates.sort(key=lambda item: (-item.rating, item.identity))

    return BradleyTerryFit(
        anchor_identity=anchor_identity,
        estimates=tuple(estimates),
        components=components,
        excluded_identities=excluded,
        observation_count=sum(total.observations for _, _, total in pairs),
        unique_pairing_count=len(pairs),
        decisive_games=sum(
            total.first_wins + total.second_wins for _, _, total in pairs
        ),
        continuity_corrected_pairings=corrected,
        confidence_level=confidence,
        converged=converged,
        iterations=iterations,
        log_likelihood=_log_likelihood(raw_pairs, logits),
    )


def _connected_components(
    adjacency: dict[str, set[str]],
    anchor_identity: str,
) -> tuple[tuple[str, ...], ...]:
    remaining = set(adjacency)
    components = []
    while remaining:
        start = anchor_identity if anchor_identity in remaining else min(remaining)
        pending = [start]
        component: set[str] = set()
        while pending:
            identity = pending.pop()
            if identity in component:
                continue
            component.add(identity)
            pending.extend(sorted(adjacency[identity] - component, reverse=True))
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    components.sort(
        key=lambda component: (
            0 if anchor_identity in component else 1,
            component,
        )
    )
    return tuple(components)


def _probability(logit_difference: float) -> float:
    if logit_difference >= 0.0:
        exponential = math.exp(-logit_difference)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(logit_difference)
    return exponential / (1.0 + exponential)


def _gradient_and_information(
    pairs: list[tuple[str, str, float, float]],
    logits: dict[str, float],
    variable_index: dict[str, int],
) -> tuple[list[float], list[list[float]]]:
    size = len(variable_index)
    gradient = [0.0] * size
    information = [[0.0] * size for _ in range(size)]
    for first, second, first_wins, second_wins in pairs:
        probability = _probability(logits[first] - logits[second])
        games = first_wins + second_wins
        score = first_wins - games * probability
        weight = games * probability * (1.0 - probability)
        first_index = variable_index.get(first)
        second_index = variable_index.get(second)
        if first_index is not None:
            gradient[first_index] += score
            information[first_index][first_index] += weight
        if second_index is not None:
            gradient[second_index] -= score
            information[second_index][second_index] += weight
        if first_index is not None and second_index is not None:
            information[first_index][second_index] -= weight
            information[second_index][first_index] -= weight
    return gradient, information


def _log_likelihood(
    pairs: list[tuple[str, str, float, float]],
    logits: dict[str, float],
) -> float:
    value = 0.0
    for first, second, first_wins, second_wins in pairs:
        difference = logits[first] - logits[second]
        if difference >= 0.0:
            normalizer = math.log1p(math.exp(-difference))
            log_probability = -normalizer
            log_complement = -difference - normalizer
        else:
            normalizer = math.log1p(math.exp(difference))
            log_probability = difference - normalizer
            log_complement = -normalizer
        value += first_wins * log_probability + second_wins * log_complement
    return value


def _solve(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("linear system dimensions are inconsistent")
    if size == 0:
        return []
    augmented = [
        [float(value) for value in row] + [float(values[index])]
        for index, row in enumerate(matrix)
    ]
    matrix_scale = max(abs(value) for row in matrix for value in row)
    tolerance = max(1e-15, matrix_scale * 1e-13)
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) <= tolerance:
            raise ValueError("observed-information matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for offset in range(column, size + 1):
            augmented[column][offset] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for offset in range(column, size + 1):
                augmented[row][offset] -= factor * augmented[column][offset]
    return [augmented[row][size] for row in range(size)]


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    columns = []
    for column in range(size):
        unit = [0.0] * size
        unit[column] = 1.0
        columns.append(_solve(matrix, unit))
    inverse = [[columns[column][row] for column in range(size)] for row in range(size)]
    for row in range(size):
        for column in range(row + 1, size):
            average = (inverse[row][column] + inverse[column][row]) / 2.0
            inverse[row][column] = average
            inverse[column][row] = average
    return inverse
