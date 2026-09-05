from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from startrain.arena import ArenaPair
from startrain.balanced_evaluation import (
    BALANCED_CATEGORIES,
    cell_variant,
    balanced_opening_seed,
    evaluation_contract,
)
from startrain.balanced_strength import balanced_strength_summary
from startrain.config import ArenaConfig


def measurement(
    candidate: str, baseline: str, timestamp: int, *, saturated=False, simulations=1024
):
    cfg = ArenaConfig(
        balanced_cells=True,
        simulations=simulations,
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
    )
    pairs = []
    for ring in cfg.rings:
        for category in BALANCED_CATEGORIES:
            for index in range(4):
                variant = cell_variant(category, index, cfg)
                pairs.append(
                    asdict(
                        ArenaPair(
                            ring=ring,
                            pair=index,
                            opening_seed=balanced_opening_seed(
                                cfg.seed, ring, variant, index
                            ),
                            opening_action=None,
                            forced_opening=False,
                            outcomes=(1, 1) if saturated or index < 3 else (-1, -1),
                            variant=variant.label,
                            segment=variant.segment,
                        )
                    )
                )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "completed_ns": timestamp,
        "started_ns": timestamp - 1,
        "terminal": True,
        "result_kind": "historical_crossplay",
        "evaluation_contract": evaluation_contract(cfg),
        "pairs": pairs,
    }


def frontier(root, identity):
    (root / "learner").mkdir(exist_ok=True)
    (root / "learner" / "champion.json").write_text(
        json.dumps({"model_identity": identity})
    )


def test_balanced_report_uses_champion_frontier_and_total_wall_time(tmp_path):
    frontier(tmp_path, "champion-2")
    first = measurement("champion-1", "anchor", 100)
    second = measurement("champion-2", "champion-1", 200)
    rejected = measurement("newest-rejected", "champion-2", 300, simulations=256)
    rejected["result_kind"] = "promotion"
    report = balanced_strength_summary(
        tmp_path, [first, second, rejected], wall_seconds=3600 * 10, provisioned_gpus=8
    )
    assert report["frontier_identity"] == "champion-2"
    assert report["anchor_identity"] == "anchor"
    assert report["rating"] == pytest.approx(2 * 400 * 0.47712125471966244)
    assert report["elo_per_wall_hour"] == pytest.approx(report["rating"] / 10)
    assert report["elo_per_provisioned_gpu_hour"] == pytest.approx(
        report["rating"] / 80
    )
    assert len(report["per_cell"]) == 24
    assert report["status"] == "measured"
    assert (
        0
        < report["confidence_interval"][0]
        < report["rating"]
        < report["confidence_interval"][1]
    )
    assert len(report["path"]) == 2
    assert (
        report["excluded_results"][0]["reason"] == "not a separate strength measurement"
    )


@pytest.mark.parametrize("state", ["missing", "disconnected", "saturated"])
def test_balanced_report_does_not_invent_missing_disconnected_or_saturated_elo(
    tmp_path, state
):
    frontier(tmp_path, "champion")
    records = (
        []
        if state == "missing"
        else [
            measurement(
                "other" if state == "disconnected" else "champion",
                "anchor",
                100,
                saturated=state == "saturated",
            )
        ]
    )
    report = balanced_strength_summary(
        tmp_path, records, wall_seconds=3600, provisioned_gpus=8
    )
    assert report["status"] == state
    assert report["rating"] is None and report["elo_per_wall_hour"] is None
    assert not report["available"]


def test_balanced_report_checks_actual_cells_and_contract_instead_of_trusting_summary(
    tmp_path,
):
    frontier(tmp_path, "champion")
    incomplete = measurement("champion", "anchor", 100)
    incomplete["pairs"] = incomplete["pairs"][:-4]
    incomplete["balanced_aggregate"] = {"score_rate": 0.9, "complete_cycles": 100}
    wrong_budget = measurement("champion", "anchor", 101, simulations=256)
    forged_contract = measurement("champion", "anchor", 102)
    forged_contract["evaluation_contract"]["cell_weight"] = 1.0
    report = balanced_strength_summary(
        tmp_path,
        [incomplete, wrong_budget, forged_contract],
        wall_seconds=3600,
        provisioned_gpus=8,
    )
    assert report["status"] == "missing"
    assert len(report["excluded_results"]) == 3


def test_different_balanced_contract_epochs_are_never_connected(tmp_path):
    frontier(tmp_path, "champion-2")
    first = measurement("champion-1", "anchor", 100)
    second = measurement("champion-2", "champion-1", 200)
    contract = second["evaluation_contract"]
    cfg = ArenaConfig(balanced_cells=True, simulations=1024, max_considered=48)
    second["evaluation_contract"] = evaluation_contract(cfg)
    assert contract != second["evaluation_contract"]
    report = balanced_strength_summary(
        tmp_path, [first, second], wall_seconds=3600, provisioned_gpus=8
    )
    assert report["anchor_identity"] == "champion-1"
    assert len(report["path"]) == 1


@pytest.mark.parametrize("outcomes", [(1,), (1, -1, 1)])
def test_balanced_report_rejects_incomplete_or_overfull_seat_reversed_pairs(
    tmp_path, outcomes
):
    frontier(tmp_path, "champion")
    malformed = measurement("champion", "anchor", 100)
    malformed["pairs"][0]["outcomes"] = outcomes
    report = balanced_strength_summary(
        tmp_path, [malformed], wall_seconds=3600, provisioned_gpus=8
    )
    assert report["status"] == "missing"
    assert "exactly two" in report["excluded_results"][0]["reason"]


def test_new_shortcut_measurement_does_not_reuse_the_first_path_edges_alpha(tmp_path):
    frontier(tmp_path, "champion-2")
    records = [
        measurement("champion-1", "anchor", 100),
        measurement("champion-2", "champion-1", 200),
    ]
    original = balanced_strength_summary(
        tmp_path, records, wall_seconds=3600, provisioned_gpus=8
    )
    records.append(measurement("champion-2", "anchor", 300))
    shortcut = balanced_strength_summary(
        tmp_path, records, wall_seconds=3600, provisioned_gpus=8
    )
    assert len(shortcut["path"]) == 1
    assert original["path"][0]["error_probability_per_side"] == 0.05 / 4
    assert shortcut["path"][0]["error_probability_per_side"] == 0.05 / 24
