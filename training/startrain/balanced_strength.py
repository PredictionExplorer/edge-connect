"""Descriptive champion-frontier strength from an immutable balanced ladder."""

from __future__ import annotations

import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from .arena import ArenaPair
from .balanced_evaluation import (
    balanced_cells,
    balanced_observation_model,
    evaluation_contract,
    summarize_balanced_pairs,
    cycle_confidence_sequence,
)
from .config import ArenaConfig
from .checkpoint import load_model_manifest
from .search_options import parse_search_execution


def _elo(score: float) -> float | None:
    return 400 * math.log10(score / (1 - score)) if 0 < score < 1 else None


def _completed_time(result: Mapping[str, Any]) -> int:
    value = result.get("completed_ns")
    return value if type(value) is int else 0


def _validated_measurement(
    result: Mapping[str, Any], simulations: int
) -> dict[str, Any]:
    contract = result.get("evaluation_contract")
    if not isinstance(contract, dict) or contract.get("simulations") != simulations:
        raise ValueError("missing balanced strength-budget contract")
    cells = contract.get("cells")
    if (
        not isinstance(cells, list)
        or not cells
        or any(not isinstance(cell, str) for cell in cells)
    ):
        raise ValueError("balanced contract must name its board/rule cells")
    try:
        rings = tuple(sorted({int(cell.split("/", 1)[0][1:]) for cell in cells}))
    except ValueError as error:
        raise ValueError("balanced contract has invalid board/rule cells") from error
    cfg = ArenaConfig(
        rings=rings,
        balanced_cells=True,
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        simulations=simulations,
        max_considered=contract["max_considered"],
        c_visit=contract["c_visit"],
        c_scale=contract["c_scale"],
        handicap_severity_cycle=tuple(contract["handicap_severity_cycle"]),
        segment_handicap_pda=tuple(contract["handicap_pda"]),
        unforced_opening_fraction=contract["unforced_opening_fraction"],
        swap_dead_zone=contract["swap_dead_zone"],
        search_execution=parse_search_execution(contract.get("search_execution", {})),
    )
    if evaluation_contract(cfg) != contract:
        raise ValueError("balanced evaluation contract identity or fields disagree")
    raw_pairs = result.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("balanced measurement omitted paired outcomes")
    pairs = []
    for raw in raw_pairs:
        if not isinstance(raw, dict):
            raise ValueError("balanced pair is not an object")
        values = dict(raw)
        values["outcomes"] = tuple(values["outcomes"])
        pairs.append(ArenaPair(**values))
    summary = summarize_balanced_pairs(pairs, cfg)
    aggregate = summary["balanced_aggregate"]
    if not isinstance(aggregate, dict) or not aggregate["complete_cycles"]:
        raise ValueError("measurement has no complete equal-cell severity cycle")
    return {
        "result": result,
        "summary": summary,
        "aggregate": aggregate,
        "contract": contract,
    }


def _balanced_strength_summary(
    root: Path,
    results: Sequence[Mapping[str, Any]],
    *,
    wall_seconds: float,
    provisioned_gpus: int,
    strength_simulations: int = 1024,
    evaluation_config: ArenaConfig | None = None,
    anchor_identity: str | None = None,
) -> dict[str, Any]:
    """Keep latest rejected candidates and cheap screens out of the headline.

    The estimate sums paired balanced-score Elo differences along a connected
    measurement path to the persisted champion. It is descriptive and assumes
    additive Elo across that path; it is not a promotion test or absolute Elo.
    """
    if evaluation_config is not None and (
        not evaluation_config.balanced_cells
        or evaluation_config.simulations != strength_simulations
    ):
        raise ValueError("strength evaluation config must match the balanced budget")
    expected_contract = (
        evaluation_contract(evaluation_config)
        if evaluation_config is not None
        else None
    )
    frontier = None
    try:
        pointer = json.loads((root / "learner" / "champion.json").read_text())
        frontier = pointer.get("model_identity") if isinstance(pointer, dict) else None
    except (OSError, ValueError):
        pass
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "missing",
        "available": False,
        "frontier_identity": frontier,
        "frontier_source": "learner/champion.json",
        "anchor_identity": None,
        "rating": None,
        "confidence_interval": [None, None],
        "simulations": strength_simulations,
        "expected_cells": len(balanced_cells(evaluation_config))
        if evaluation_config is not None
        else None,
        "objective": balanced_observation_model(evaluation_config)
        if evaluation_config is not None
        else None,
        "contract_selection": "active_profile"
        if evaluation_config is not None
        else "latest_champion_measurement",
        "method": "sum-of-paired-equal-cell-elo-contrasts-on-connected-champion-path",
        "statistical_role": "descriptive_only",
        "absolute_elo": False,
        "wall_seconds": wall_seconds,
        "provisioned_gpu_hours": provisioned_gpus * wall_seconds / 3600,
        "denominator": "total-provisioned-run-wall-time-including-warmup-pauses-and-rejected-candidates",
        "elo_per_wall_hour": None,
        "elo_per_provisioned_gpu_hour": None,
        "per_cell": {},
        "path": [],
        "excluded_results": [],
    }
    exclusions: list[dict[str, object]] = []
    measurements = []
    for result in sorted(results, key=_completed_time):
        reason = None
        if result.get("result_kind") != "historical_crossplay":
            reason = "not a separate strength measurement"
        elif result.get("terminal") is not True:
            reason = "strength measurement is unfinished"
        else:
            try:
                measurement = _validated_measurement(result, strength_simulations)
                if (
                    expected_contract is not None
                    and measurement["contract"] != expected_contract
                ):
                    raise ValueError(
                        "strength measurement belongs to another evaluation contract"
                    )
                candidate, baseline = result.get("candidate"), result.get("baseline")
                if (
                    not isinstance(candidate, str)
                    or not isinstance(baseline, str)
                    or candidate == baseline
                ):
                    raise ValueError("invalid measurement checkpoint identities")
                measurements.append(measurement)
            except (KeyError, TypeError, ValueError) as error:
                reason = str(error)
        if reason:
            exclusions.append({"path": result.get("_path"), "reason": reason})
    output["excluded_results"] = exclusions
    if expected_contract is not None:
        output["evaluation_contract"] = expected_contract
    if not measurements:
        output["reason"] = (
            f"no complete {strength_simulations}-simulation balanced strength "
            "measurements for the selected evaluation contract"
        )
        return output
    # A changed evaluation contract starts a distinct rating epoch. Never mix
    # search budgets, severity schedules, or rule-cell definitions in one graph.
    selected = next(
        (
            item
            for item in reversed(measurements)
            if item["result"].get("candidate") == frontier
        ),
        measurements[-1],
    )
    contract = selected["contract"]
    measurements = [item for item in measurements if item["contract"] == contract]
    # Allocate error to each immutable chronological measurement, not to its
    # position on a path: new shortcut edges must not reuse an old edge's alpha.
    for ordinal, item in enumerate(measurements):
        item["error_probability_per_side"] = 0.05 / (2 * (ordinal + 1) * (ordinal + 2))
    output["evaluation_contract"] = contract
    output["expected_cells"] = len(contract["cells"])
    output["objective"] = contract["objective"]
    anchor = anchor_identity or measurements[0]["result"]["baseline"]
    output["anchor_identity"] = anchor
    if not isinstance(frontier, str):
        output["reason"] = "persisted champion identity is unavailable"
        return output
    graph: dict[str, list[tuple[str, dict[str, Any], int]]] = {}
    for item in measurements:
        candidate, baseline = item["result"]["candidate"], item["result"]["baseline"]
        graph.setdefault(baseline, []).append((candidate, item, 1))
        graph.setdefault(candidate, []).append((baseline, item, -1))
    pending = deque([(anchor, [])])
    seen = {anchor}
    path = None
    while pending:
        identity, steps = pending.popleft()
        if identity == frontier:
            path = steps
            break
        for neighbor, item, sign in graph.get(identity, []):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append((neighbor, steps + [(item, sign)]))
    if path is None:
        output.update(
            status="disconnected",
            reason="champion has no path to the fixed balanced anchor",
        )
        return output
    rating = 0.0
    low: float | None = 0.0
    high: float | None = 0.0
    saturated = False
    serialized = []
    per_cell: dict[str, dict[str, Any]] = {}
    for item, sign in path:
        aggregate = item["aggregate"]
        score = aggregate["score_rate"]
        difference = _elo(score)
        # The allocation belongs to this measurement even if a new graph edge
        # changes the selected path. Reversal only inverts its paired interval.
        error = item["error_probability_per_side"]
        bounds = cycle_confidence_sequence(
            aggregate["cycle_scores"],
            pairs_per_cycle=aggregate["pairs_per_cycle"],
            error_probability=error,
        )
        elo_bounds = [_elo(bound) for bound in bounds]
        if difference is None:
            saturated = True
        else:
            rating += sign * difference
        edge_low, edge_high = (
            elo_bounds
            if sign == 1
            else [
                -elo_bounds[1] if elo_bounds[1] is not None else None,
                -elo_bounds[0] if elo_bounds[0] is not None else None,
            ]
        )
        low = low + edge_low if low is not None and edge_low is not None else None
        high = high + edge_high if high is not None and edge_high is not None else None
        for key, cell in item["summary"]["per_cell"].items():
            entry = per_cell.setdefault(
                key, {"rating": 0.0, "status": "measured", "edges": []}
            )
            cell_elo = cell["elo_difference"]
            if cell_elo is None or entry["rating"] is None:
                entry["rating"], entry["status"] = None, "saturated"
            else:
                entry["rating"] += sign * cell_elo
            entry["edges"].append(
                {
                    "candidate": item["result"]["candidate"],
                    "baseline": item["result"]["baseline"],
                    "direction": sign,
                    **cell,
                }
            )
        serialized.append(
            {
                "candidate": item["result"]["candidate"],
                "baseline": item["result"]["baseline"],
                "direction": sign,
                "elo_difference": difference,
                "complete_cycles": aggregate["complete_cycles"],
                "score_confidence_interval": list(bounds),
                "error_probability_per_side": error,
                "source": item["result"].get("_path"),
            }
        )
    output.update(
        status="saturated"
        if saturated
        else "uncertain"
        if low is None or high is None or low <= 0 <= high
        else "measured",
        available=not saturated,
        rating=None if saturated else rating,
        confidence_interval=[low, high],
        path=serialized,
        per_cell=per_cell,
        reason="one-sided evidence cannot identify finite Elo" if saturated else None,
    )
    if not saturated and wall_seconds > 0:
        output["elo_per_wall_hour"] = rating / (wall_seconds / 3600)
        output["elo_per_provisioned_gpu_hour"] = rating / (
            provisioned_gpus * wall_seconds / 3600
        )
    return output


def _epoch_candidate_step(root: Path, result: Mapping[str, Any]) -> int:
    explicit = result.get("candidate_step")
    if explicit is not None and (type(explicit) is not int or explicit < 0):
        raise ValueError("epoch candidate step is invalid")
    manifest_path = result.get("candidate_manifest")
    if isinstance(manifest_path, str) and manifest_path:
        path = Path(manifest_path)
        if not path.is_absolute():
            path = root / path
        manifest = load_model_manifest(path)
        if manifest.model_identity != result.get("candidate"):
            raise ValueError("epoch candidate manifest identity disagrees")
        if explicit is not None and explicit != manifest.model_step:
            raise ValueError("epoch candidate step disagrees with its manifest")
        return manifest.model_step
    if type(explicit) is not int or explicit < 0:
        raise ValueError("epoch candidate step is unavailable")
    return explicit


def balanced_strength_summary(
    root: Path,
    results: Sequence[Mapping[str, Any]],
    *,
    wall_seconds: float,
    provisioned_gpus: int,
    strength_simulations: int = 1024,
    evaluation_config: ArenaConfig | None = None,
    observed_until_ns: int | None = None,
) -> dict[str, Any]:
    """Keep historical evidence, but use an explicit origin for current rates.

    A strength-epoch.json marker names the immutable evaluation contract, its
    starting champion, activation time and stopped learner step. Reported epoch
    gain is realized champion-frontier change, not causal release attribution.
    """
    historical = _balanced_strength_summary(
        root,
        results,
        wall_seconds=wall_seconds,
        provisioned_gpus=provisioned_gpus,
        strength_simulations=strength_simulations,
        evaluation_config=evaluation_config,
    )
    lifetime = {
        "lifetime_normalized_elo_per_wall_hour": historical["elo_per_wall_hour"],
        "lifetime_normalized_elo_per_provisioned_gpu_hour": historical[
            "elo_per_provisioned_gpu_hour"
        ],
        "lifetime_wall_seconds": wall_seconds,
        "lifetime_provisioned_gpu_hours": provisioned_gpus * wall_seconds / 3600,
    }
    output = {
        **historical,
        **lifetime,
        "elo_per_wall_hour": None,
        "elo_per_provisioned_gpu_hour": None,
        "denominator": "unavailable-without-valid-explicit-strength-epoch",
        "rate_status": "epoch_not_configured",
        "rating_scope": "historical-contract-anchor",
        "strength_epoch": None,
    }
    marker_path = root / "strength-epoch.json"
    if not marker_path.exists():
        return output
    try:
        if marker_path.is_symlink():
            raise ValueError("strength epoch may not be a symbolic link")
        marker = json.loads(marker_path.read_text())
        if (
            not isinstance(marker, dict)
            or type(marker.get("schema_version")) is not int
            or marker.get("schema_version") != 1
        ):
            raise ValueError("strength epoch schema is invalid")
        started = marker.get("started_ns")
        boundary = marker.get("minimum_candidate_step")
        anchor = marker.get("anchor_identity")
        if (
            type(started) is not int
            or started <= 0
            or type(boundary) is not int
            or boundary < 0
            or not isinstance(anchor, str)
            or not anchor
        ):
            raise ValueError("strength epoch anchor/time/boundary is invalid")
        source = marker.get("source_commit")
        if source is not None and (
            not isinstance(source, str)
            or re.fullmatch(r"[0-9a-f]{7,64}", source) is None
        ):
            raise ValueError("strength epoch source commit is invalid")
        contract = historical.get("evaluation_contract")
        if not isinstance(contract, dict) or marker.get(
            "evaluation_contract_identity"
        ) != contract.get("identity"):
            raise ValueError("strength epoch belongs to another evaluation contract")
        run_started = None
        run_path = root / "run.json"
        if run_path.is_file():
            run = json.loads(run_path.read_text())
            run_started = run.get("created_ns") if isinstance(run, dict) else None
            if (
                type(run_started) is not int
                or run_started <= 0
                or started < run_started
            ):
                raise ValueError("strength epoch predates or lacks a valid run origin")
        if observed_until_ns is None:
            if (
                run_started is None
                or not math.isfinite(wall_seconds)
                or wall_seconds < 0
            ):
                raise ValueError("strength epoch observation end is unavailable")
            observed_until_ns = run_started + round(wall_seconds * 1e9)
        if type(observed_until_ns) is not int or observed_until_ns <= started:
            raise ValueError("strength epoch observation end must follow activation")
    except (OSError, ValueError, TypeError) as error:
        output["rate_status"] = "invalid_epoch"
        output["strength_epoch"] = {"path": str(marker_path), "error": str(error)}
        return output

    eligible = []
    exclusions = []
    for result in results:
        if (
            result.get("result_kind") != "historical_crossplay"
            or result.get("terminal") is not True
        ):
            eligible.append(result)
            continue
        try:
            evaluation_started = result.get("started_ns")
            if (
                type(evaluation_started) is not int
                or evaluation_started < started
                or _completed_time(result) < evaluation_started
                or _completed_time(result) > observed_until_ns
            ):
                raise ValueError("strength evaluation is outside the explicit epoch")
            if _epoch_candidate_step(root, result) <= boundary:
                raise ValueError(
                    "strength candidate predates the epoch learner boundary"
                )
            eligible.append(result)
        except (OSError, ValueError, TypeError) as error:
            exclusions.append({"path": result.get("_path"), "reason": str(error)})
    elapsed = (observed_until_ns - started) / 1e9
    epoch = _balanced_strength_summary(
        root,
        eligible,
        wall_seconds=elapsed,
        provisioned_gpus=provisioned_gpus,
        strength_simulations=strength_simulations,
        evaluation_config=evaluation_config,
        anchor_identity=anchor,
    )
    if historical["frontier_identity"] != anchor and not any(
        result.get("result_kind") == "historical_crossplay"
        and result.get("terminal") is True
        and result.get("candidate") == historical["frontier_identity"]
        for result in eligible
    ):
        epoch.update(
            status="missing",
            available=False,
            rating=None,
            confidence_interval=[None, None],
            elo_per_wall_hour=None,
            elo_per_provisioned_gpu_hour=None,
            reason="frontier has no post-boundary candidate measurement",
        )
    return {
        **epoch,
        **lifetime,
        "anchor_identity": anchor,
        "historical": historical,
        "rating_scope": "explicit-epoch-anchor",
        "rate_status": "available"
        if epoch["elo_per_wall_hour"] is not None
        else "epoch_evidence_unavailable",
        "denominator": "explicit-epoch-wall-time-including-training-evaluation-warmup-pauses-and-rejected-candidates",
        "strength_epoch": {
            **marker,
            "observed_until_ns": observed_until_ns,
            "wall_seconds": elapsed,
            "scope": "realized-champion-frontier-change-not-causal-release-efficiency",
        },
        "excluded_results": [*epoch["excluded_results"], *exclusions],
    }
