"""A fixed equal-cell objective on the configured boards for paired evaluation.

One observation is a complete handicap-severity cycle across every cell. Its
score gives each mode/rule/board cell exactly the same weight, irrespective of
which cells finished first. Seat-reversed games remain one paired observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Sequence

from .selfplay import GameVariant
from .contracts import RULES_SCHEMA_ID, RULES_HASH_WIRE
from .topology import SUPPORTED_RINGS

if TYPE_CHECKING:
    from .arena import ArenaPair
    from .config import ArenaConfig

BALANCED_OBSERVATION_MODEL = "equal-24-cells-complete-severity-cycle-v1"
BALANCED_CATEGORIES = (
    "classic-standard",
    "double-standard",
    "classic-pie",
    "double-pie",
    "classic-handicap",
    "double-handicap",
)
HOEFFDING_LAMBDAS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def balanced_opening_seed(
    seed: int, ring: int, variant: GameVariant, pair_index: int
) -> int:
    material = f"balanced-cell-pair-opening-v1:{seed}:{ring}:{variant.label}:{pair_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def balanced_search_seed(opening_seed: int, candidate_player: int, move: int) -> int:
    """A cell/pair/seat/move stream, independent of cohort membership or order."""
    material = (
        f"balanced-root-search-v2:{opening_seed}:{candidate_player}:{move}".encode()
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def cycle_log_e_value(
    scores: Sequence[float],
    *,
    pairs_per_cycle: int,
    null_mean: float,
    direction: str,
) -> float:
    """Hoeffding mixtures checked only at complete fixed-allocation cycles.

    Pair scores are independent across hashed seed streams, bounded in [0,1],
    and may have different means by cell/severity. A full cycle averages those
    means with fixed weights. Hoeffding's lemma gives exp(lambda*S-lambda²*N/8)
    as an e-process at cycle boundaries; the two games of a pair stay together.
    """
    if (
        pairs_per_cycle <= 0
        or not 0 <= null_mean <= 1
        or direction not in ("greater", "less")
    ):
        raise ValueError("invalid paired cycle e-process parameters")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("cycle scores must be bounded in [0, 1]")
    count = len(scores) * pairs_per_cycle
    centered = pairs_per_cycle * math.fsum(scores) - count * null_mean
    if direction == "less":
        centered = -centered
    logs = [value * centered - value * value * count / 8 for value in HOEFFDING_LAMBDAS]
    maximum = max(logs)
    return (
        maximum
        + math.log(math.fsum(math.exp(value - maximum) for value in logs))
        - math.log(len(logs))
    )


def cycle_confidence_sequence(
    scores: Sequence[float],
    *,
    pairs_per_cycle: int,
    error_probability: float,
) -> tuple[float, float]:
    if not 0 < error_probability < 1:
        raise ValueError("cycle error probability must be in (0, 1)")
    threshold = math.log(1 / error_probability)
    bounds = []
    for direction, endpoint in (("greater", 0.0), ("less", 1.0)):
        if (
            not scores
            or cycle_log_e_value(
                scores,
                pairs_per_cycle=pairs_per_cycle,
                null_mean=endpoint,
                direction=direction,
            )
            < threshold
        ):
            bounds.append(endpoint)
            continue
        low, high = 0.0, 1.0
        for _ in range(52):
            middle = (low + high) / 2
            exceeds = (
                cycle_log_e_value(
                    scores,
                    pairs_per_cycle=pairs_per_cycle,
                    null_mean=middle,
                    direction=direction,
                )
                >= threshold
            )
            if exceeds == (direction == "greater"):
                low = middle
            else:
                high = middle
        bounds.append(low if direction == "greater" else high)
    return bounds[0], bounds[1]


def category(variant: GameVariant) -> str:
    rule = "pie" if variant.pie else "handicap" if variant.handicap > 1 else "standard"
    return f"{variant.mode}-{rule}"


def cell_key(ring: int, variant: GameVariant) -> str:
    return f"r{ring}/{category(variant)}"


def balanced_cells(config: ArenaConfig) -> tuple[str, ...]:
    return tuple(
        f"r{ring}/{name}" for ring in config.rings for name in BALANCED_CATEGORIES
    )


def balanced_observation_model(config: ArenaConfig) -> str:
    """Keep the original all-board contract while naming subsets truthfully."""
    if config.rings == SUPPORTED_RINGS:
        return BALANCED_OBSERVATION_MODEL
    rings = "-".join(str(ring) for ring in config.rings)
    return (
        f"equal-{len(balanced_cells(config))}-cells-rings-{rings}"
        "-complete-severity-cycle-v1"
    )


def cell_variant(name: str, pair_index: int, config: ArenaConfig) -> GameVariant:
    mode, rule = name.split("-", 1)
    severity = config.handicap_severity_cycle
    return GameVariant(
        mode=mode,
        pie=rule == "pie",
        handicap=severity[pair_index % len(severity)] if rule == "handicap" else 1,
    )


def pair_key(pair: ArenaPair) -> tuple[int, str, int]:
    return pair.ring, category(GameVariant.parse(pair.variant)), pair.pair


def evaluation_contract(config: ArenaConfig) -> dict[str, object]:
    """A budget-specific immutable contract; candidate-dependent seeds are separate."""
    contract: dict[str, object] = {
        "schema_version": 1,
        "objective": balanced_observation_model(config),
        "rules_schema": RULES_SCHEMA_ID,
        "rules_hash": RULES_HASH_WIRE,
        "cells": list(balanced_cells(config)),
        "cell_weight": 1 / len(balanced_cells(config)),
        "handicap_severity_cycle": list(config.handicap_severity_cycle),
        "handicap_pda": list(config.segment_handicap_pda),
        "simulations": config.simulations,
        "max_considered": config.max_considered,
        "c_visit": config.c_visit,
        "c_scale": config.c_scale,
        "unforced_opening_fraction": config.unforced_opening_fraction,
        "swap_dead_zone": config.swap_dead_zone,
        "seed_schedule": "sha256-balanced-cell-pair-opening-v1-root-seat-move-v2",
        "native_root_seed": "explicit-seed-statehash-fixed-tree-index-zero",
        "handicap_schedule": "absolute-pair-index-mod-severity-cycle-v1",
        "observation_unit": "all-cells-complete-role-reversed-severity-cycle",
        "statistical_test": "complete-cycle-paired-hoeffding-mixture-v1",
        "hoeffding_lambdas": list(HOEFFDING_LAMBDAS),
        "pair_independence": "independent seed streams across pairs; arbitrary dependence within each seat reversal",
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {**contract, "identity": "sha256-" + hashlib.sha256(encoded).hexdigest()}


def grouped_cell_pairs(
    pairs: Sequence[ArenaPair], config: ArenaConfig
) -> dict[str, list[ArenaPair]]:
    grouped: dict[str, list[ArenaPair]] = {key: [] for key in balanced_cells(config)}
    for pair in pairs:
        variant = GameVariant.parse(pair.variant)
        key = cell_key(pair.ring, variant)
        if key not in grouped:
            raise ValueError("balanced arena pair has an unconfigured cell")
        if variant != cell_variant(category(variant), pair.pair, config):
            raise ValueError("balanced arena pair violates the fixed handicap schedule")
        grouped[key].append(pair)
    for values in grouped.values():
        values.sort(key=lambda pair: pair.pair)
        indices = [pair.pair for pair in values]
        if len(indices) != len(set(indices)):
            raise ValueError("balanced cell pair indices must be unique")
    if len({pair.opening_seed for pair in pairs}) != len(pairs):
        raise ValueError("balanced pairs must use distinct seed streams across cells")
    return grouped


def _contiguous_prefix(values: Sequence[ArenaPair]) -> int:
    for expected, pair in enumerate(values):
        if pair.pair != expected:
            return expected
    return len(values)


def completed_counts_by_ring(
    pairs: Sequence[ArenaPair], config: ArenaConfig
) -> dict[int, int]:
    grouped = grouped_cell_pairs(pairs, config)
    return {
        ring: min(
            _contiguous_prefix(grouped[f"r{ring}/{name}"])
            for name in BALANCED_CATEGORIES
        )
        for ring in config.rings
    }


def _elo(probability: float) -> float | None:
    return (
        400 * math.log10(probability / (1 - probability))
        if 0 < probability < 1
        else None
    )


def summarize_balanced_pairs(
    pairs: Sequence[ArenaPair], config: ArenaConfig
) -> dict[str, object]:
    from .arena import (
        _expected_score,
        _reported_e_value,
    )

    grouped = grouped_cell_pairs(pairs, config)
    cycle_length = len(config.handicap_severity_cycle)
    complete_cycles = min(
        _contiguous_prefix(values) // cycle_length for values in grouped.values()
    )
    included = complete_cycles * cycle_length
    per_cell_scores = {
        key: tuple(
            math.fsum(pair.score_rate for pair in values[start : start + cycle_length])
            / cycle_length
            for start in range(0, included, cycle_length)
        )
        for key, values in grouped.items()
    }
    cycle_scores = tuple(
        math.fsum(scores[index] for scores in per_cell_scores.values()) / len(grouped)
        for index in range(complete_cycles)
    )
    cell_error = (1 - config.confidence) / (2 * len(grouped))
    per_cell: dict[str, object] = {}
    vetoes = []
    floor_score = _expected_score(config.cell_regression_floor_elo)
    for key, scores in per_cell_scores.items():
        mean = math.fsum(scores) / len(scores) if scores else None
        interval = (
            cycle_confidence_sequence(
                scores, pairs_per_cycle=cycle_length, error_probability=cell_error
            )
            if scores
            else (0.0, 1.0)
        )
        regression_log_e = (
            cycle_log_e_value(
                scores,
                pairs_per_cycle=cycle_length,
                null_mean=floor_score,
                direction="less",
            )
            if scores
            else 0.0
        )
        regressed = regression_log_e >= math.log(1 / cell_error)
        if regressed:
            vetoes.append(key)
        per_cell[key] = {
            "pairs": len(grouped[key]),
            "included_pairs": included,
            "complete_cycles": complete_cycles,
            "score_rate": mean,
            "elo_difference": _elo(mean) if mean is not None else None,
            "status": "missing"
            if not grouped[key]
            else "incomplete"
            if not scores
            else "saturated"
            if mean in (0, 1)
            else "measured",
            "anytime_confidence_sequence": list(interval),
            "anytime_elo_interval": [_elo(bound) for bound in interval],
            "error_probability_per_side": cell_error,
            "floor_elo": config.cell_regression_floor_elo,
            "regression_log_e_value": regression_log_e,
            "regression_status": "regress" if regressed else "not_established",
        }
    if cycle_scores:
        mean = math.fsum(cycle_scores) / complete_cycles
        pairs_per_cycle = cycle_length * len(grouped)
        promotion_e = cycle_log_e_value(
            cycle_scores,
            pairs_per_cycle=pairs_per_cycle,
            null_mean=_expected_score(config.null_elo),
            direction="greater",
        )
        rejection_e = cycle_log_e_value(
            cycle_scores,
            pairs_per_cycle=pairs_per_cycle,
            null_mean=_expected_score(config.alternative_elo),
            direction="less",
        )
        state = (
            "accept_alternative"
            if promotion_e >= math.log(1 / config.alpha)
            else "accept_null"
            if rejection_e >= math.log(1 / config.beta)
            else "continue"
        )
        lower, _ = cycle_confidence_sequence(
            cycle_scores,
            pairs_per_cycle=pairs_per_cycle,
            error_probability=config.alpha,
        )
        _, upper = cycle_confidence_sequence(
            cycle_scores, pairs_per_cycle=pairs_per_cycle, error_probability=config.beta
        )
    else:
        mean = None
        state, promotion_e, rejection_e = "continue", 0.0, 0.0
        lower, upper = 0.0, 1.0
    minimum_ready = included >= config.minimum_pairs_per_ring
    decision = "continue"
    if minimum_ready:
        if vetoes:
            decision = "reject_ring_regression"
        elif state == "accept_alternative":
            decision = "promote"
        elif state == "accept_null":
            decision = "reject"
    aggregate = {
        "observation_model": balanced_observation_model(config),
        "status": "missing"
        if not pairs
        else "incomplete"
        if not cycle_scores
        else "saturated"
        if mean in (0, 1)
        else "measured",
        "cells": len(grouped),
        "complete_cycles": complete_cycles,
        "pairs_per_cycle": cycle_length * len(grouped),
        "pairs": included * len(grouped),
        "available_pairs": len(pairs),
        "games": included * len(grouped) * 2,
        "score_rate": mean,
        "elo_difference": _elo(mean) if mean is not None else None,
        "anytime_confidence_sequence": [lower, upper],
        "anytime_elo_interval": [_elo(lower), _elo(upper)],
        "cycle_scores": list(cycle_scores),
        "cell_weights": {key: 1 / len(grouped) for key in grouped},
        "missing_cells": [key for key, values in grouped.items() if not values],
    }
    return {
        "evaluation_contract": evaluation_contract(config),
        "balanced_aggregate": aggregate,
        "aggregate": aggregate,
        "per_cell": per_cell,
        "per_ring": {
            str(ring): {
                "cells": {
                    key: summary
                    for key, summary in per_cell.items()
                    if key.startswith(f"r{ring}/")
                }
            }
            for ring in config.rings
        },
        "promotion": {
            "decision": decision,
            "sequential_state": state,
            "pair_model": balanced_observation_model(config),
            "minimum_ready": minimum_ready,
            "cell_vetoes": vetoes,
            "ring_floors": {},
            "regression_source": "cell" if vetoes else None,
            "confidence_sequence": [lower, upper],
            "statistical_test": {
                "name": "complete-cycle-paired-hoeffding-mixture-e-process",
                "observation_unit": balanced_observation_model(config),
                "promotion": {
                    "log_e_value": promotion_e,
                    "e_value": _reported_e_value(promotion_e),
                    "threshold": 1 / config.alpha,
                },
                "rejection": {
                    "log_e_value": rejection_e,
                    "e_value": _reported_e_value(rejection_e),
                    "threshold": 1 / config.beta,
                },
                "cell_guard_familywise_error": 1 - config.confidence,
            },
        },
    }
