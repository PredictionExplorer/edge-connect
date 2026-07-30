from __future__ import annotations

import pytest

from startrain.autonomous_elo import (
    DecisiveMatch,
    fit_bradley_terry_elo,
)


def _ratings(matches: list[DecisiveMatch], anchor: str) -> dict[str, float]:
    fit = fit_bradley_terry_elo(matches, anchor_identity=anchor)
    return {estimate.identity: estimate.rating for estimate in fit.estimates}


def test_bradley_terry_fit_recovers_known_synthetic_ranking() -> None:
    matches = [
        DecisiveMatch("checkpoint-a", "checkpoint-b", 70, 30),
        DecisiveMatch("checkpoint-b", "checkpoint-c", 70, 30),
        DecisiveMatch("checkpoint-a", "checkpoint-c", 490, 90),
    ]

    fit = fit_bradley_terry_elo(matches, anchor_identity="checkpoint-c")
    estimates = {estimate.identity: estimate for estimate in fit.estimates}
    one_step_elo = 400.0 * 0.3679767853

    assert fit.connected is True
    assert fit.converged is True
    assert estimates["checkpoint-c"].rating == 0
    assert estimates["checkpoint-c"].standard_error == 0
    assert estimates["checkpoint-c"].confidence_interval == (0, 0)
    assert estimates["checkpoint-b"].rating == pytest.approx(one_step_elo)
    assert estimates["checkpoint-a"].rating == pytest.approx(2 * one_step_elo)
    assert estimates["checkpoint-a"].rating > estimates["checkpoint-b"].rating > 0
    assert estimates["checkpoint-a"].standard_error > 0
    assert (
        estimates["checkpoint-a"].confidence_interval[0]
        < estimates["checkpoint-a"].rating
        < estimates["checkpoint-a"].confidence_interval[1]
    )
    contrast = fit.contrast("checkpoint-a", "checkpoint-b")
    assert contrast.difference == pytest.approx(one_step_elo)
    assert (
        0
        < contrast.standard_error
        < (
            estimates["checkpoint-a"].standard_error ** 2
            + estimates["checkpoint-b"].standard_error ** 2
        )
        ** 0.5
    )
    assert (
        contrast.confidence_interval[0]
        < contrast.difference
        < contrast.confidence_interval[1]
    )


def test_reversed_and_repeated_directed_results_are_aggregated() -> None:
    directed = [
        DecisiveMatch("checkpoint-a", "checkpoint-b", 30, 10),
        DecisiveMatch("checkpoint-b", "checkpoint-a", 5, 15),
    ]
    aggregate = [DecisiveMatch("checkpoint-a", "checkpoint-b", 45, 15)]

    directed_fit = fit_bradley_terry_elo(
        directed,
        anchor_identity="checkpoint-b",
    )
    aggregate_fit = fit_bradley_terry_elo(
        aggregate,
        anchor_identity="checkpoint-b",
    )

    assert _ratings(directed, "checkpoint-b") == pytest.approx(
        _ratings(aggregate, "checkpoint-b")
    )
    assert directed_fit.unique_pairing_count == 1
    assert directed_fit.observation_count == 2
    assert aggregate_fit.observation_count == 1
    assert _ratings(directed, "checkpoint-b")["checkpoint-a"] == pytest.approx(
        400.0 * 0.4771212547
    )


def test_disconnected_nodes_are_excluded_from_anchored_fit() -> None:
    fit = fit_bradley_terry_elo(
        [
            DecisiveMatch("checkpoint-a", "checkpoint-b", 60, 40),
            DecisiveMatch("checkpoint-x", "checkpoint-y", 80, 20),
        ],
        anchor_identity="checkpoint-b",
    )

    assert fit.connected is False
    assert fit.components == (
        ("checkpoint-a", "checkpoint-b"),
        ("checkpoint-x", "checkpoint-y"),
    )
    assert fit.excluded_identities == ("checkpoint-x", "checkpoint-y")
    assert {estimate.identity for estimate in fit.estimates} == {
        "checkpoint-a",
        "checkpoint-b",
    }
    assert fit.decisive_games == 100


def test_fit_is_deterministic_across_input_order_and_separated_results() -> None:
    matches = [
        DecisiveMatch("checkpoint-a", "checkpoint-b", 20, 0),
        DecisiveMatch("checkpoint-c", "checkpoint-b", 3, 17),
        DecisiveMatch("checkpoint-a", "checkpoint-c", 14, 6),
    ]

    first = fit_bradley_terry_elo(matches, anchor_identity="checkpoint-b")
    second = fit_bradley_terry_elo(
        list(reversed(matches)),
        anchor_identity="checkpoint-b",
    )

    assert first == second
    assert first.converged is True
    assert first.continuity_corrected_pairings == 1
    assert all(
        estimate.standard_error >= 0
        and estimate.confidence_interval[0]
        <= estimate.rating
        <= estimate.confidence_interval[1]
        for estimate in first.estimates
    )


def test_rating_inputs_are_strictly_validated() -> None:
    with pytest.raises(ValueError, match="at least one decisive match"):
        fit_bradley_terry_elo([], anchor_identity="checkpoint-a")
    with pytest.raises(ValueError, match="at least one game"):
        DecisiveMatch("checkpoint-a", "checkpoint-b", 0, 0)
    with pytest.raises(ValueError, match="against itself"):
        DecisiveMatch("checkpoint-a", "checkpoint-a", 1, 0)
    with pytest.raises(ValueError, match="non-negative integers"):
        DecisiveMatch("checkpoint-a", "checkpoint-b", True, 0)
