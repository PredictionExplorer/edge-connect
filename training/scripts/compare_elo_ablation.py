#!/usr/bin/env python3
"""Rank training treatments by objective-compatible Elo gained per wall hour."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from startrain.arena import bounded_confidence_sequence, elo_from_probability
from startrain.config import load_config

if __package__:
    from .strength_efficiency_report import build_strength_efficiency_report
else:
    from strength_efficiency_report import build_strength_efficiency_report

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-elo-ablation-comparison"
DEFAULT_PROVISIONED_GPUS = 8
DEFAULT_GUARD_RINGS = (4, 6, 8)
DEFAULT_GUARD_FLOOR_ELO = -35.0
CONFIDENCE_LEVEL = 0.95
ONE_SIDED_95_NORMAL_QUANTILE = 1.6448536269514722
_VALID_INTEGRITY_STATUSES = frozenset(
    {"ok", "pass", "passed", "valid", "verified", "healthy"}
)
_CLEAN_TEARDOWN_STATUSES = frozenset(
    {"not_required", "clean", "complete", "completed", "released"}
)

_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TERMINAL_DECISIONS = frozenset(
    {"promote", "reject", "reject_ring_regression", "reject_max_pairs"}
)


@dataclass
class _Treatment:
    label: str
    payload: dict[str, object]
    anchor_identity: str | None
    ranking_score: float | None
    point_score: float | None
    reasons: list[dict[str, str]]
    weighted_ranking_score: float | None = None
    weighted_point_score: float | None = None
    weighted_objective: str | None = None
    training_objective: str | None = "generalist"


def _positive_integer(argument: str) -> int:
    try:
        value = int(argument)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _finite_float(argument: str) -> float:
    try:
        value = float(argument)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="treatment label and run root; repeat for every treatment",
    )
    parser.add_argument(
        "--provisioned-gpus",
        type=_positive_integer,
        default=DEFAULT_PROVISIONED_GPUS,
    )
    parser.add_argument(
        "--guard-ring",
        action="append",
        type=_positive_integer,
        help="ring requiring non-inferiority evidence; defaults to 4, 6, and 8",
    )
    parser.add_argument(
        "--no-guard-rings",
        action="store_true",
        help="disable blocking per-ring floors for a pre-registered aggregate objective",
    )
    parser.add_argument(
        "--guard-floor-elo",
        type=_finite_float,
        default=DEFAULT_GUARD_FLOOR_ELO,
    )
    parser.add_argument("--output", type=Path)
    return parser


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _nonnegative_integer(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _integer(value: object) -> int | None:
    return value if type(value) is int else None


def _positive_timestamp(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _failure(
    path: Path | str,
    error: str,
    *,
    line: int | None = None,
) -> dict[str, object]:
    return {"path": str(path), "line": line, "error": error}


def _normalized_failures(
    failures: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized: dict[tuple[str, int | None, str], dict[str, object]] = {}
    for failure in failures:
        path = str(failure.get("path") or "<unknown>")
        raw_line = failure.get("line")
        line = raw_line if type(raw_line) is int and raw_line > 0 else None
        error = str(
            failure.get("error") or failure.get("reason") or "unspecified parse failure"
        )
        normalized[(path, line, error)] = {
            "path": path,
            "line": line,
            "error": error,
        }
    return [
        normalized[key]
        for key in sorted(
            normalized,
            key=lambda item: (item[0], item[1] or 0, item[2]),
        )
    ]


def _read_json(
    path: Path,
    *,
    failures: list[dict[str, object]],
) -> dict[str, object] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(_failure(path, f"{type(error).__name__}: {error}"))
        return None
    if not isinstance(loaded, dict):
        failures.append(_failure(path, "JSON document is not an object"))
        return None
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(
    path: Path,
    *,
    failures: list[dict[str, object]],
) -> list[tuple[int, dict[str, object]]]:
    if not path.is_file():
        return []
    records: list[tuple[int, dict[str, object]]] = []
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as error:
        failures.append(_failure(path, f"{type(error).__name__}: {error}"))
        return records
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append(
                    _failure(
                        path,
                        f"JSONDecodeError: {error}",
                        line=line_number,
                    )
                )
                continue
            if not isinstance(loaded, dict):
                failures.append(
                    _failure(
                        path,
                        "JSONL record is not an object",
                        line=line_number,
                    )
                )
                continue
            records.append((line_number, loaded))
    return records


def _stats(values: Sequence[float]) -> dict[str, object] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def _arena_results(
    root: Path,
    *,
    failures: list[dict[str, object]],
) -> list[dict[str, object]]:
    arena = root / "arena"
    if not arena.is_dir():
        return []
    results = []
    for path in sorted(arena.glob("*.json")):
        payload = _read_json(path, failures=failures)
        if payload is None:
            continue
        candidate = payload.get("candidate")
        baseline = payload.get("baseline")
        if candidate is None and baseline is None:
            continue
        if (
            not isinstance(candidate, str)
            or not candidate
            or not isinstance(baseline, str)
            or not baseline
        ):
            failures.append(
                _failure(path, "arena candidate and baseline must be non-empty strings")
            )
            continue
        results.append({**payload, "_path": str(path)})
    return results


def _weighted_summary(
    result: Mapping[str, object],
    promotion: Mapping[str, object],
    *,
    path: str,
    failures: list[dict[str, object]],
) -> dict[str, object] | None:
    raw = result.get("weighted_aggregate")
    if raw is None:
        raw = promotion.get("weighted_aggregate")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        failures.append(_failure(path, "weighted_aggregate must be an object"))
        return None

    raw_ratios = raw.get(
        "pair_ratios",
        raw.get("promotion_pair_ratios", raw.get("ratios")),
    )
    ratios: dict[int, int] = {}
    if isinstance(raw_ratios, Mapping):
        for raw_ring, raw_ratio in raw_ratios.items():
            try:
                ring = int(raw_ring)
            except (TypeError, ValueError):
                ratios = {}
                break
            if type(raw_ratio) is not int or raw_ratio <= 0:
                ratios = {}
                break
            ratios[ring] = raw_ratio
    if not ratios:
        failures.append(
            _failure(path, "weighted_aggregate pair ratios are missing or invalid")
        )
        return None

    score_rate = _number(raw.get("score_rate", raw.get("weighted_score_rate")))
    elo_difference = _number(
        raw.get("elo_difference", raw.get("weighted_elo_difference"))
    )
    interval = raw.get("anytime_elo_interval")
    if (
        score_rate is None
        or not 0 <= score_rate <= 1
        or elo_difference is None
        or not isinstance(interval, Sequence)
        or isinstance(interval, str | bytes)
        or len(interval) != 2
        or (lower := _number(interval[0])) is None
        or (upper := _number(interval[1])) is None
        or lower > upper
    ):
        failures.append(
            _failure(path, "weighted_aggregate Elo evidence is missing or invalid")
        )
        return None
    complete_blocks = _nonnegative_integer(
        raw.get("complete_blocks", raw.get("completed_blocks"))
    )
    if complete_blocks is None:
        failures.append(_failure(path, "weighted_aggregate complete_blocks is invalid"))
        return None

    ratio_total = sum(ratios.values())
    serialized_weights = {
        str(ring): ratios[ring] / ratio_total for ring in sorted(ratios)
    }
    error_probability = _mapping(raw.get("anytime_error_probability")) or {}
    objective = json.dumps(
        {
            "pair_ratios": {str(ring): ratios[ring] for ring in sorted(ratios)},
            "normalized_weights": serialized_weights,
            "observation_model": raw.get("observation_model"),
            "null_elo": promotion.get("null_elo"),
            "alternative_elo": promotion.get("alternative_elo"),
            "lower_error_probability": error_probability.get("lower"),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        **dict(raw),
        "pair_ratios": {str(ring): ratios[ring] for ring in sorted(ratios)},
        "normalized_weights": serialized_weights,
        "complete_blocks": complete_blocks,
        "score_rate": score_rate,
        "elo_difference": elo_difference,
        "anytime_elo_interval": [lower, upper],
        "one_sided_lower_elo": lower,
        "objective": objective,
    }


def _per_ring_evidence(
    result: Mapping[str, object],
    *,
    path: str,
    failures: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    raw_per_ring = result.get("per_ring")
    if not isinstance(raw_per_ring, Mapping):
        return {}
    output: dict[str, dict[str, object]] = {}
    for raw_ring, raw_summary in raw_per_ring.items():
        try:
            ring = int(raw_ring)
        except (TypeError, ValueError):
            failures.append(_failure(path, "per-ring evidence has an invalid ring"))
            continue
        if ring <= 0 or not isinstance(raw_summary, Mapping):
            failures.append(_failure(path, f"ring {ring} evidence must be an object"))
            continue
        summary = dict(raw_summary)
        interval = summary.get("anytime_elo_interval")
        if interval is not None and (
            not isinstance(interval, Sequence)
            or isinstance(interval, str | bytes)
            or len(interval) != 2
            or _number(interval[0]) is None
            or _number(interval[1]) is None
        ):
            failures.append(
                _failure(path, f"ring {ring} anytime_elo_interval is invalid")
            )
            continue
        output[str(ring)] = summary
    return output


def _terminal_evaluations(
    results: Sequence[Mapping[str, object]],
    *,
    guard_rings: Sequence[int],
    guard_floor_elo: float,
    failures: list[dict[str, object]],
) -> list[dict[str, object]]:
    evaluations = []
    for result in results:
        promotion = _mapping(result.get("promotion"))
        if promotion is None:
            continue
        path = str(result.get("_path") or "<unknown arena result>")
        terminal = result.get("terminal")
        if type(terminal) is not bool:
            failures.append(
                _failure(path, "arena promotion result has invalid terminal")
            )
            continue
        if not terminal:
            continue
        decision = promotion.get("decision")
        if decision == "superseded":
            continue
        if not isinstance(decision, str) or decision not in _TERMINAL_DECISIONS:
            failures.append(
                _failure(path, "terminal arena promotion decision is invalid")
            )
            continue
        completed_ns = _positive_timestamp(result.get("completed_ns"))
        if completed_ns is None:
            failures.append(
                _failure(path, "terminal arena result has invalid completed_ns")
            )
            continue
        ring_floors = _mapping(promotion.get("ring_floors"))
        weighted = _weighted_summary(
            result,
            promotion,
            path=path,
            failures=failures,
        )
        per_ring = _per_ring_evidence(result, path=path, failures=failures)
        guards = []
        for ring in guard_rings:
            floor = (
                _mapping(ring_floors.get(str(ring)))
                if ring_floors is not None
                else None
            )
            if floor is None:
                guards.append(
                    {
                        "ring": ring,
                        "status": "missing",
                        "source_floor_elo": None,
                        "source_status": None,
                        "anytime_lower_elo": None,
                        "passes_configured_floor": False,
                    }
                )
                continue
            source_floor = _number(floor.get("floor_elo"))
            if floor.get("floor_elo") is not None and source_floor is None:
                failures.append(_failure(path, f"ring {ring} floor_elo must be finite"))
            lower = _number(floor.get("anytime_lower_elo"))
            if floor.get("anytime_lower_elo") is not None and lower is None:
                failures.append(
                    _failure(path, f"ring {ring} anytime_lower_elo must be finite")
                )
            source_status = floor.get("status")
            if source_status is not None and not isinstance(source_status, str):
                failures.append(
                    _failure(path, f"ring {ring} guard status must be a string")
                )
                source_status = None
            passes = lower is not None and lower >= guard_floor_elo
            guards.append(
                {
                    "ring": ring,
                    "status": "pass"
                    if passes
                    else ("fail" if lower is not None else "missing"),
                    "source_floor_elo": source_floor,
                    "source_status": source_status,
                    "anytime_lower_elo": lower,
                    "passes_configured_floor": passes,
                }
            )
        evaluations.append(
            {
                "path": path,
                "candidate": result["candidate"],
                "baseline": result["baseline"],
                "decision": decision,
                "completed_ns": completed_ns,
                "guards": guards,
                "weighted_aggregate": weighted,
                "per_ring": per_ring,
            }
        )
    return sorted(
        evaluations,
        key=lambda item: (
            _positive_timestamp(item.get("completed_ns")) or 0,
            str(item["path"]),
        ),
    )


def _guardrail_summary(
    evaluations: Sequence[Mapping[str, object]],
    *,
    guard_rings: Sequence[int],
    guard_floor_elo: float,
) -> dict[str, object]:
    ring_summaries = []
    for ring in guard_rings:
        observations = []
        for evaluation in evaluations:
            guards = evaluation.get("guards")
            if not isinstance(guards, list):
                continue
            guard = next(
                (
                    item
                    for item in guards
                    if isinstance(item, Mapping) and item.get("ring") == ring
                ),
                None,
            )
            lower = (
                _number(guard.get("anytime_lower_elo"))
                if isinstance(guard, Mapping)
                else None
            )
            observations.append(
                {
                    "path": evaluation.get("path"),
                    "candidate": evaluation.get("candidate"),
                    "decision": evaluation.get("decision"),
                    "completed_ns": evaluation.get("completed_ns"),
                    "source_floor_elo": guard.get("source_floor_elo")
                    if isinstance(guard, Mapping)
                    else None,
                    "source_status": guard.get("source_status")
                    if isinstance(guard, Mapping)
                    else None,
                    "anytime_lower_elo": lower,
                    "passes_configured_floor": (
                        lower is not None and lower >= guard_floor_elo
                    ),
                }
            )
        valid = [
            value
            for observation in observations
            if (value := _number(observation.get("anytime_lower_elo"))) is not None
        ]
        missing_count = len(observations) - len(valid)
        below_count = sum(value < guard_floor_elo for value in valid)
        if not observations or missing_count:
            status = "missing"
        elif below_count:
            status = "fail"
        else:
            status = "pass"
        ring_summaries.append(
            {
                "ring": ring,
                "status": status,
                "observation_count": len(observations),
                "missing_evidence_count": missing_count,
                "below_floor_count": below_count,
                "minimum_anytime_lower_elo": min(valid) if valid else None,
                "latest_anytime_lower_elo": valid[-1] if valid else None,
                "observations": observations,
            }
        )
    rejected = sum(
        evaluation.get("decision") == "reject_ring_regression"
        for evaluation in evaluations
    )
    if not evaluations:
        status = "unavailable"
    elif any(summary["status"] != "pass" for summary in ring_summaries):
        status = "fail"
    else:
        status = "pass"
    return {
        "status": status,
        "configured_rings": list(guard_rings),
        "configured_floor_elo": guard_floor_elo,
        "evidence_field": "promotion.ring_floors.<ring>.anytime_lower_elo",
        "terminal_evaluation_count": len(evaluations),
        "reject_ring_regression_count": rejected,
        "rings": ring_summaries,
        "terminal_evaluations": list(evaluations),
    }


def _record_publication(
    publications: dict[str, set[tuple[int, int | None]]],
    payload: Mapping[str, object],
    *,
    path: Path,
    line: int | None,
    timestamp_field: str,
    failures: list[dict[str, object]],
) -> None:
    identity = payload.get("model_identity")
    published_ns = _positive_timestamp(payload.get(timestamp_field))
    step_value = payload.get("model_step")
    step = _nonnegative_integer(step_value)
    if not isinstance(identity, str) or not identity:
        failures.append(
            _failure(path, "model publication identity is invalid", line=line)
        )
        return
    if published_ns is None:
        failures.append(
            _failure(
                path,
                f"model publication {timestamp_field} is invalid",
                line=line,
            )
        )
        return
    if step_value is not None and step is None:
        failures.append(_failure(path, "model publication step is invalid", line=line))
        return
    publications.setdefault(identity, set()).add((published_ns, step))


def _model_publications(
    root: Path,
    *,
    failures: list[dict[str, object]],
) -> dict[str, dict[str, int | None]]:
    publications: dict[str, set[tuple[int, int | None]]] = {}
    history_path = root / "learner" / "model-history.jsonl"
    for line, record in _read_jsonl(history_path, failures=failures):
        _record_publication(
            publications,
            record,
            path=history_path,
            line=line,
            timestamp_field="published_ns",
            failures=failures,
        )
    metadata_paths = [
        *(root / "learner" / "manifests").glob("manifest-*.json"),
    ]
    for path in sorted(set(metadata_paths)):
        payload = _read_json(path, failures=failures)
        if payload is not None:
            _record_publication(
                publications,
                payload,
                path=path,
                line=None,
                timestamp_field="created_ns",
                failures=failures,
            )
    resolved = {}
    for identity, evidence in sorted(publications.items()):
        timestamps = {item[0] for item in evidence}
        steps = {item[1] for item in evidence if item[1] is not None}
        if len(timestamps) != 1 or len(steps) > 1:
            failures.append(
                _failure(
                    root / "learner",
                    f"conflicting publication evidence for {identity}",
                )
            )
            continue
        resolved[identity] = {
            "published_ns": next(iter(timestamps)),
            "model_step": next(iter(steps)) if steps else None,
        }
    return resolved


def _is_replay_wait(record: Mapping[str, object]) -> bool:
    for name in ("phase", "event", "state", "status"):
        value = record.get(name)
        if not isinstance(value, str):
            continue
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        if "replay_wait" in normalized:
            return True
    return False


def _learner_summary(
    report: Mapping[str, object],
    records: Sequence[tuple[int, Mapping[str, object]]],
    *,
    metrics_path: Path,
    failures: list[dict[str, object]],
) -> dict[str, object]:
    updates = []
    replay_wait_durations = []
    replay_wait_events = 0
    active_waits: dict[str, int] = {}
    for line, record in records:
        if "updates_per_new_sample" in record:
            raw_update = record.get("updates_per_new_sample")
            if raw_update is not None:
                update = _number(raw_update)
                if update is None or update < 0:
                    failures.append(
                        _failure(
                            metrics_path,
                            "updates_per_new_sample must be finite and non-negative",
                            line=line,
                        )
                    )
                else:
                    updates.append(update)

        wait_marker = _is_replay_wait(record)
        explicit_name = next(
            (
                name
                for name in (
                    "replay_wait_seconds",
                    "replay_wait_duration_seconds",
                )
                if name in record
            ),
            None,
        )
        if explicit_name is None and wait_marker:
            explicit_name = next(
                (
                    name
                    for name in ("wait_seconds", "duration_seconds", "elapsed_seconds")
                    if name in record
                ),
                None,
            )
        explicit_wait = (
            _number(record.get(explicit_name)) if explicit_name is not None else None
        )
        if explicit_name is not None and (explicit_wait is None or explicit_wait < 0):
            failures.append(
                _failure(
                    metrics_path,
                    f"{explicit_name} must be finite and non-negative",
                    line=line,
                )
            )
            explicit_wait = None
        raw_worker = record.get("worker")
        worker = raw_worker if isinstance(raw_worker, str) else "learner"
        timestamp_ns = _positive_timestamp(record.get("timestamp_ns"))
        if explicit_wait is not None:
            replay_wait_events += 1
            replay_wait_durations.append(explicit_wait)
            active_waits.pop(worker, None)
        elif wait_marker:
            if worker not in active_waits:
                replay_wait_events += 1
                if timestamp_ns is None:
                    failures.append(
                        _failure(
                            metrics_path,
                            "replay-wait marker requires a positive timestamp_ns",
                            line=line,
                        )
                    )
                else:
                    active_waits[worker] = timestamp_ns
        elif worker in active_waits and timestamp_ns is not None:
            started_ns = active_waits.pop(worker)
            if timestamp_ns < started_ns:
                failures.append(
                    _failure(
                        metrics_path,
                        "replay-wait completion predates its start",
                        line=line,
                    )
                )
            else:
                replay_wait_durations.append(
                    (timestamp_ns - started_ns) / 1_000_000_000
                )

    learner = _mapping(report.get("learner")) or {}
    update_stats = _stats(updates)
    if update_stats is not None:
        update_stats["latest"] = updates[-1]
    return {
        "updates_per_new_sample": {
            "availability": "observed" if updates else "not_recorded",
            "statistics": update_stats,
        },
        "replay_waits": {
            "availability": (
                "observed" if replay_wait_events else "not_recorded_in_jsonl"
            ),
            "events": replay_wait_events,
            "completed_intervals": len(replay_wait_durations),
            "open_intervals": len(active_waits),
            "seconds": _stats(replay_wait_durations),
        },
        "device_duty_fraction": learner.get("device_duty_fraction"),
        "data_wait_seconds": learner.get("data_wait_seconds"),
        "data_wait_fraction": learner.get("data_wait_fraction"),
        "end_to_end_examples_per_second": learner.get("end_to_end_examples_per_second"),
        "measured_wall_seconds": learner.get("measured_wall_seconds"),
    }


def _actor_summary(report: Mapping[str, object]) -> dict[str, object]:
    actors = _mapping(report.get("actors")) or {}
    return {
        "worker_count": actors.get("worker_count"),
        "games": actors.get("games"),
        "samples": actors.get("samples"),
        "evaluator_rows": actors.get("evaluator_rows"),
        "aggregate_games_per_second": actors.get("aggregate_games_per_second"),
        "aggregate_samples_per_second": actors.get("aggregate_samples_per_second"),
        "aggregate_evaluator_rows_per_second": actors.get(
            "aggregate_evaluator_rows_per_second"
        ),
        "samples_per_physical_gpu_second": actors.get(
            "samples_per_physical_gpu_second"
        ),
    }


def _candidate_latency_summary(
    evaluations: Sequence[Mapping[str, object]],
    publications: Mapping[str, Mapping[str, int | None]],
    *,
    failures: list[dict[str, object]],
) -> dict[str, object]:
    records = []
    missing = []
    for evaluation in evaluations:
        candidate = str(evaluation["candidate"])
        publication = publications.get(candidate)
        published_ns = (
            _positive_timestamp(publication.get("published_ns"))
            if publication is not None
            else None
        )
        terminal_ns = _positive_timestamp(evaluation.get("completed_ns"))
        if published_ns is None or terminal_ns is None:
            missing.append(
                {
                    "path": evaluation.get("path"),
                    "candidate": candidate,
                    "reason": "candidate publication timestamp is unavailable"
                    if published_ns is None
                    else "terminal completion timestamp is unavailable",
                }
            )
            continue
        if terminal_ns < published_ns:
            failures.append(
                _failure(
                    str(evaluation.get("path")),
                    "candidate terminal completion predates publication",
                )
            )
            continue
        candidate_step = (
            publication.get("model_step") if publication is not None else None
        )
        records.append(
            {
                "path": evaluation.get("path"),
                "candidate": candidate,
                "candidate_step": candidate_step,
                "decision": evaluation.get("decision"),
                "published_ns": published_ns,
                "terminal_ns": terminal_ns,
                "latency_seconds": (terminal_ns - published_ns) / 1_000_000_000,
            }
        )
    latencies = [
        latency
        for record in records
        if (latency := _number(record.get("latency_seconds"))) is not None
    ]
    if not evaluations:
        status = "unavailable"
    elif missing or len(records) != len(evaluations):
        status = "incomplete"
    else:
        status = "complete"
    return {
        "status": status,
        "terminal_candidate_count": len(evaluations),
        "measured_candidate_count": len(records),
        "seconds": _stats(latencies),
        "records": records,
        "missing": missing,
    }


def _report_anchor(report: Mapping[str, object]) -> dict[str, object]:
    autonomous = _mapping(report.get("autonomous_elo")) or {}
    anchor = _mapping(autonomous.get("anchor")) or {}
    return {
        "identity": anchor.get("identity"),
        "step": anchor.get("step"),
        "rating_elo": anchor.get("rating"),
        "standard_error_elo": 0.0 if isinstance(anchor.get("identity"), str) else None,
        "selection": anchor.get("selection"),
    }


def _normalized_status(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value.lower().replace("-", "_").replace(" ", "_")
    return None


def _first_lifecycle_value(
    sources: Sequence[Mapping[str, object]],
    *names: str,
) -> object:
    for source in sources:
        for name in names:
            if name in source:
                return source[name]
    return None


def _lifecycle_sections(
    metadata: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    lifecycle = _mapping(metadata.get("lifecycle")) or {}
    measurement = (
        _mapping(lifecycle.get("measurement"))
        or _mapping(metadata.get("measurement"))
        or {}
    )
    teardown = (
        _mapping(metadata.get("teardown"))
        or _mapping(metadata.get("teardown_status"))
        or _mapping(lifecycle.get("teardown"))
        or {}
    )
    integrity = (
        _mapping(metadata.get("integrity"))
        or _mapping(metadata.get("integrity_status"))
        or _mapping(lifecycle.get("integrity"))
        or _mapping(teardown.get("integrity"))
        or {}
    )
    return lifecycle, measurement, teardown, integrity


def _integrity_valid(
    integrity: Mapping[str, object],
    *,
    status: str | None,
) -> bool:
    explicit = integrity.get("valid")
    if type(explicit) is bool:
        return explicit
    return status in _VALID_INTEGRITY_STATUSES


def _profile_checksum(
    root: Path,
    *,
    failures: list[dict[str, object]],
) -> tuple[Path | None, str | None]:
    checksum_path = root / "profile.sha256"
    try:
        fields = checksum_path.read_text(encoding="utf-8").strip().split()
    except OSError:
        return None, None
    if (
        len(fields) != 2
        or not re.fullmatch(r"[0-9a-f]{64}", fields[0])
        or Path(fields[1]).name != fields[1]
    ):
        failures.append(_failure(checksum_path, "profile checksum is malformed"))
        return None, None
    return root / fields[1], fields[0]


def _verified_profile_objectives(
    root: Path,
    metadata: Mapping[str, object] | None,
    *,
    failures: list[dict[str, object]],
) -> dict[str, object]:
    checksum_profile, checksum_sha256 = _profile_checksum(root, failures=failures)
    profile_path: Path | None = None
    expected_sha256: str | None = None
    if metadata is not None:
        raw_profile = metadata.get("profile")
        if isinstance(raw_profile, str) and raw_profile:
            profile_path = Path(raw_profile).expanduser().resolve()
            if profile_path.parent != root:
                failures.append(
                    _failure(
                        root / "ablation.json",
                        "ablation profile must be installed directly in the run root",
                    )
                )
                profile_path = None
        else:
            failures.append(
                _failure(root / "ablation.json", "ablation profile path is invalid")
            )
        raw_sha256 = metadata.get("profile_sha256")
        if isinstance(raw_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", raw_sha256):
            expected_sha256 = raw_sha256
        else:
            failures.append(
                _failure(root / "ablation.json", "ablation profile SHA-256 is invalid")
            )
    else:
        profile_path = checksum_profile
        expected_sha256 = checksum_sha256
        if profile_path is None:
            profile_path = next(
                (
                    candidate
                    for candidate in (
                        root / "profile-elo-ablation.yaml",
                        root / "profile.yaml",
                        root / "resolved-config.yaml",
                    )
                    if candidate.is_file()
                ),
                None,
            )
            failures.append(_failure(root, "run profile checksum is missing"))

    if (
        profile_path is not None
        and checksum_profile is not None
        and profile_path != checksum_profile.resolve()
    ):
        failures.append(
            _failure(root / "profile.sha256", "profile checksum names another profile")
        )
    if (
        expected_sha256 is not None
        and checksum_sha256 is not None
        and expected_sha256 != checksum_sha256
    ):
        failures.append(_failure(root / "profile.sha256", "profile checksums disagree"))
    expected_sha256 = expected_sha256 or checksum_sha256
    if profile_path is None or expected_sha256 is None:
        return {
            "path": str(profile_path) if profile_path is not None else None,
            "sha256": expected_sha256,
            "training_objective": None,
            "promotion_objective": None,
        }
    try:
        actual_sha256 = _sha256(profile_path)
    except OSError as error:
        failures.append(
            _failure(profile_path, f"profile could not be read: {type(error).__name__}")
        )
        return {
            "path": str(profile_path),
            "sha256": expected_sha256,
            "training_objective": None,
            "promotion_objective": None,
        }
    if actual_sha256 != expected_sha256:
        failures.append(_failure(profile_path, "profile SHA-256 mismatch"))
    try:
        config = load_config(profile_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        failures.append(
            _failure(
                profile_path, f"profile is invalid: {type(error).__name__}: {error}"
            )
        )
        return {
            "path": str(profile_path),
            "sha256": actual_sha256,
            "training_objective": None,
            "promotion_objective": None,
        }
    training_objective = config.orchestration.training_objective
    promotion_objective = (
        "ring_10_only"
        if training_objective == "ring10_only"
        else (
            "weighted_aggregate"
            if config.arena.promotion_pair_ratios
            else "ring_10_guarded"
        )
    )
    if metadata is not None:
        for name, expected in (
            ("training_objective", training_objective),
            ("promotion_objective", promotion_objective),
        ):
            claimed = metadata.get(name)
            if claimed is not None and claimed != expected:
                failures.append(
                    _failure(
                        root / "ablation.json",
                        f"ablation {name} differs from the frozen profile",
                    )
                )
    return {
        "path": str(profile_path),
        "sha256": actual_sha256,
        "training_objective": training_objective,
        "promotion_objective": promotion_objective,
    }


def _measurement_context(
    root: Path,
    report: Mapping[str, object],
    *,
    failures: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object], str | None]:
    metadata_path = root / "ablation.json"
    if not metadata_path.is_file():
        profile = _verified_profile_objectives(root, None, failures=failures)
        started_ns = _positive_timestamp(report.get("started_ns"))
        stopped_ns = _positive_timestamp(report.get("observed_until_ns"))
        wall_seconds = (
            (stopped_ns - started_ns) / 1_000_000_000
            if started_ns is not None
            and stopped_ns is not None
            and stopped_ns >= started_ns
            else None
        )
        return (
            {
                "source": "strength_efficiency_report",
                "profile": profile,
                "training_objective": profile["training_objective"],
                "promotion_objective": profile["promotion_objective"],
                "status": "complete",
                "started_ns": started_ns,
                "stopped_ns": stopped_ns,
                "cutoff_ns": stopped_ns,
                "wall_seconds": wall_seconds,
                "resource_released_ns": stopped_ns,
                "resource_wall_seconds": wall_seconds,
                "stop_reason": "last_observed_run_timestamp",
                "exit_code": None,
                "outcome": None,
                "attempt_count": None,
                "failure": None,
                "lifecycle_status": None,
                "completion_status": None,
                "warnings": [],
                "teardown_status": "not_recorded",
                "teardown": None,
                "integrity_status": "not_recorded",
                "integrity": None,
                "wall_budget_seconds": None,
                "leaf_budget": None,
            },
            _report_anchor(report),
            None,
        )

    metadata = _read_json(metadata_path, failures=failures)
    if metadata is None:
        return (
            {
                "source": "ablation.json",
                "training_objective": None,
                "promotion_objective": None,
                "status": "invalid",
                "started_ns": None,
                "stopped_ns": None,
                "cutoff_ns": None,
                "wall_seconds": None,
                "resource_released_ns": None,
                "resource_wall_seconds": None,
                "stop_reason": None,
                "exit_code": None,
                "outcome": None,
                "attempt_count": None,
                "failure": "ablation.json could not be parsed",
                "lifecycle_status": None,
                "completion_status": None,
                "warnings": [],
                "teardown_status": None,
                "teardown": None,
                "integrity_status": None,
                "integrity": None,
                "wall_budget_seconds": None,
                "leaf_budget": None,
            },
            {
                "identity": None,
                "step": None,
                "rating_elo": None,
                "standard_error_elo": None,
                "selection": "ablation_metadata",
            },
            "ablation.json could not be parsed",
        )
    if metadata.get("report") != "startrain-elo-ablation-branch":
        failures.append(
            _failure(
                metadata_path,
                "ablation metadata has an unsupported report identifier",
            )
        )
    profile = _verified_profile_objectives(root, metadata, failures=failures)
    training_objective = profile["training_objective"]
    promotion_objective = profile["promotion_objective"]
    raw_anchor = _mapping(metadata.get("anchor"))
    anchor_identity = raw_anchor.get("model_identity") if raw_anchor else None
    anchor_step = (
        _nonnegative_integer(raw_anchor.get("model_step")) if raw_anchor else None
    )
    if not isinstance(anchor_identity, str) or not anchor_identity:
        failures.append(
            _failure(metadata_path, "ablation anchor model_identity is invalid")
        )
        anchor_identity = None
    if raw_anchor is None or anchor_step is None:
        failures.append(
            _failure(metadata_path, "ablation anchor model_step is invalid")
        )

    lifecycle, measurement_section, teardown, integrity = _lifecycle_sections(metadata)
    lifecycle_sources = (measurement_section, metadata, lifecycle)
    raw_started = _first_lifecycle_value(
        lifecycle_sources,
        "measurement_started_ns",
        "started_ns",
    )
    raw_legacy_stopped = metadata.get("measurement_stopped_ns")
    raw_cutoff = _first_lifecycle_value(
        lifecycle_sources,
        "measurement_cutoff_ns",
        "cutoff_ns",
    )
    if raw_cutoff is None:
        raw_cutoff = raw_legacy_stopped
    resource_sources = (metadata, lifecycle, teardown)
    resource_release_recorded = any(
        name in source
        for source in resource_sources
        for name in ("resource_released_ns", "released_ns")
    )
    raw_resource_released = _first_lifecycle_value(
        resource_sources,
        "resource_released_ns",
        "released_ns",
    )
    if raw_resource_released is None and not resource_release_recorded:
        raw_resource_released = raw_legacy_stopped or raw_cutoff
    started_ns = _positive_timestamp(raw_started)
    cutoff_ns = _positive_timestamp(raw_cutoff)
    resource_released_ns = _positive_timestamp(raw_resource_released)
    if raw_started is not None and started_ns is None:
        failures.append(
            _failure(metadata_path, "measurement_started_ns must be positive")
        )
    if raw_cutoff is not None and cutoff_ns is None:
        failures.append(_failure(metadata_path, "measurement cutoff must be positive"))
    if raw_resource_released is not None and resource_released_ns is None:
        failures.append(
            _failure(metadata_path, "resource_released_ns must be positive")
        )
    stop_reason = _first_lifecycle_value(
        lifecycle_sources,
        "measurement_stop_reason",
        "stop_reason",
    )
    if stop_reason is not None and not isinstance(stop_reason, str):
        failures.append(
            _failure(metadata_path, "measurement_stop_reason must be a string")
        )
        stop_reason = None
    raw_exit_code = _first_lifecycle_value(
        lifecycle_sources,
        "measurement_exit_code",
        "exit_code",
    )
    exit_code = _integer(raw_exit_code)
    if raw_exit_code is not None and exit_code is None:
        failures.append(
            _failure(metadata_path, "measurement_exit_code must be an integer")
        )
    raw_status = _first_lifecycle_value(
        lifecycle_sources,
        "measurement_status",
        "status",
    )
    measurement_status = _normalized_status(raw_status)
    if raw_status is not None and measurement_status is None:
        failures.append(_failure(metadata_path, "measurement_status must be a string"))
    elif measurement_status not in {
        None,
        "running",
        "retryable",
        "complete",
        "completed",
        "completed_with_teardown_failure",
        "completed_with_teardown_warning",
        "failed",
    }:
        failures.append(_failure(metadata_path, "measurement_status is not recognized"))
    raw_outcome = _first_lifecycle_value(
        lifecycle_sources,
        "measurement_outcome",
        "outcome",
    )
    outcome = _normalized_status(raw_outcome)
    if raw_outcome is not None and outcome is None:
        failures.append(_failure(metadata_path, "measurement_outcome must be a string"))
    elif outcome not in {
        None,
        "budget_completion",
        "transient_crash",
        "fatal_orchestrator_exit",
        "runner_error",
    }:
        failures.append(
            _failure(metadata_path, "measurement_outcome is not recognized")
        )
    raw_attempt_count = _first_lifecycle_value(
        lifecycle_sources,
        "measurement_attempt_count",
        "attempt_count",
        "attempt",
    )
    attempt_count = _nonnegative_integer(raw_attempt_count)
    if raw_attempt_count is not None and attempt_count is None:
        failures.append(
            _failure(metadata_path, "measurement_attempt_count must be non-negative")
        )
    raw_failure = _first_lifecycle_value(
        lifecycle_sources,
        "measurement_failure",
        "failure",
    )
    measurement_failure = raw_failure if isinstance(raw_failure, str) else None
    if raw_failure is not None and measurement_failure is None:
        failures.append(_failure(metadata_path, "measurement_failure must be a string"))
    completion_status = _normalized_status(
        _first_lifecycle_value(
            lifecycle_sources,
            "measurement_completion_status",
            "completion_status",
        )
    )
    raw_warnings = _first_lifecycle_value(
        lifecycle_sources,
        "measurement_warnings",
        "warnings",
    )
    lifecycle_warnings = (
        [str(item) for item in raw_warnings] if isinstance(raw_warnings, list) else []
    )
    raw_teardown_status = _first_lifecycle_value(
        (metadata, lifecycle),
        "teardown_status",
    )
    if isinstance(raw_teardown_status, Mapping):
        raw_teardown_status = raw_teardown_status.get("status")
    if raw_teardown_status is None:
        raw_teardown_status = teardown.get("status")
    teardown_status = _normalized_status(raw_teardown_status)
    if raw_teardown_status is not None and teardown_status is None:
        failures.append(_failure(metadata_path, "teardown status must be a string"))
    raw_integrity_status = _first_lifecycle_value(
        (metadata, lifecycle, teardown),
        "integrity_status",
    )
    if isinstance(raw_integrity_status, Mapping):
        raw_integrity_status = raw_integrity_status.get("status")
    if raw_integrity_status is None:
        raw_integrity_status = integrity.get("status")
    integrity_status = _normalized_status(raw_integrity_status)
    if raw_integrity_status is not None and integrity_status is None:
        failures.append(_failure(metadata_path, "integrity status must be a string"))
    failure_domain = _normalized_status(
        _first_lifecycle_value(
            lifecycle_sources,
            "failure_domain",
            "domain",
        )
    )
    failure_phase = _normalized_status(
        _first_lifecycle_value(
            lifecycle_sources,
            "failure_phase",
            "phase",
        )
    )
    ordered = (
        started_ns is not None and cutoff_ns is not None and cutoff_ns >= started_ns
    )
    resources_ordered = (
        resource_released_ns is not None
        and cutoff_ns is not None
        and resource_released_ns >= cutoff_ns
    )
    if started_ns is not None and cutoff_ns is not None and cutoff_ns < started_ns:
        failures.append(
            _failure(
                metadata_path,
                "measurement cutoff predates measurement_started_ns",
            )
        )
    if (
        resource_released_ns is not None
        and cutoff_ns is not None
        and resource_released_ns < cutoff_ns
    ):
        failures.append(
            _failure(metadata_path, "resource release predates measurement cutoff")
        )
    has_resilient_lifecycle = raw_status is not None or raw_outcome is not None
    if has_resilient_lifecycle and (attempt_count is None or attempt_count <= 0):
        failures.append(
            _failure(
                metadata_path,
                "resilient measurement lifecycle requires a positive attempt count",
            )
        )
    structured_lifecycle = bool(
        lifecycle
        or measurement_section
        or teardown
        or integrity
        or any(
            name in metadata
            for name in (
                "measurement_cutoff_ns",
                "resource_released_ns",
                "teardown_status",
                "integrity_status",
                "failure_domain",
                "failure_phase",
            )
        )
    )
    teardown_warning = (
        teardown_status is not None and teardown_status not in _CLEAN_TEARDOWN_STATUSES
    )
    integrity_is_valid = _integrity_valid(integrity, status=integrity_status)
    lifecycle_complete = (
        measurement_status
        in {
            "complete",
            "completed",
            "completed_with_teardown_failure",
            "completed_with_teardown_warning",
        }
        and outcome == "budget_completion"
        and stop_reason in {"wall_budget", "leaf_budget"}
        and attempt_count is not None
        and attempt_count > 0
        and completion_status
        in {
            None,
            "complete",
            "completed",
            "complete_with_warning",
            "completed_with_teardown_failure",
            "completed_with_teardown_warning",
        }
    )
    if structured_lifecycle:
        complete = (
            ordered
            and resources_ordered
            and lifecycle_complete
            and (integrity_status is None or integrity_is_valid)
            and (
                not teardown_warning
                or (
                    integrity_is_valid
                    and failure_phase not in {"measurement", "pre_cutoff", "pre_budget"}
                )
            )
        )
    else:
        complete = ordered and (
            (lifecycle_complete and exit_code in {None, 0, -15})
            if has_resilient_lifecycle
            else (
                stop_reason in {"wall_budget", "leaf_budget"} and exit_code in {0, -15}
            )
        )
    if ordered:
        assert started_ns is not None
        assert cutoff_ns is not None
        wall_seconds = (cutoff_ns - started_ns) / 1_000_000_000
    else:
        wall_seconds = None
    resource_wall_seconds = (
        (resource_released_ns - started_ns) / 1_000_000_000
        if started_ns is not None
        and resource_released_ns is not None
        and resource_released_ns >= started_ns
        else None
    )
    if complete:
        incomplete_reason = None
    elif started_ns is None:
        incomplete_reason = "ablation measurement has not started"
    elif cutoff_ns is None:
        incomplete_reason = "ablation measurement has no cutoff"
    elif resource_released_ns is None:
        incomplete_reason = "ablation resources have not been released"
    else:
        incomplete_reason = (
            "ablation measurement is not an eligible budget completion "
            f"(status={measurement_status!r}, outcome={outcome!r}, "
            f"reason={stop_reason!r}, exit={exit_code!r}, "
            f"teardown={teardown_status!r}, integrity={integrity_status!r}, "
            f"failure_phase={failure_phase!r}, failure={measurement_failure!r})"
        )
    return (
        {
            "source": "ablation.json",
            "profile": profile,
            "training_objective": training_objective,
            "promotion_objective": promotion_objective,
            "status": "complete" if complete else "incomplete",
            "started_ns": started_ns,
            "stopped_ns": cutoff_ns,
            "cutoff_ns": cutoff_ns,
            "wall_seconds": wall_seconds,
            "resource_released_ns": resource_released_ns,
            "resource_wall_seconds": resource_wall_seconds,
            "stop_reason": stop_reason,
            "exit_code": exit_code,
            "outcome": outcome,
            "attempt_count": attempt_count,
            "failure": measurement_failure,
            "lifecycle_status": measurement_status,
            "completion_status": completion_status,
            "warnings": lifecycle_warnings,
            "failure_domain": failure_domain,
            "failure_phase": failure_phase,
            "teardown_status": teardown_status
            or ("not_recorded" if not structured_lifecycle else None),
            "teardown": dict(teardown) if teardown else None,
            "integrity_status": integrity_status
            or ("not_recorded" if not structured_lifecycle else None),
            "integrity": dict(integrity) if integrity else None,
            "wall_budget_seconds": metadata.get("wall_budget_seconds"),
            "leaf_budget": metadata.get("leaf_budget"),
        },
        {
            "identity": anchor_identity,
            "step": anchor_step,
            "rating_elo": None,
            "standard_error_elo": None,
            "selection": "ablation_metadata",
        },
        incomplete_reason,
    )


def _endpoint(
    report: Mapping[str, object],
    evaluations: Sequence[Mapping[str, object]],
    publications: Mapping[str, Mapping[str, int | None]],
    *,
    anchor_identity: str | None,
) -> tuple[
    dict[str, object] | None,
    dict[str, object] | None,
    str | None,
]:
    autonomous = _mapping(report.get("autonomous_elo")) or {}
    primary = _mapping(autonomous.get("primary_ring_10")) or {}
    if primary.get("status") != "available":
        return (
            None,
            None,
            str(primary.get("reason") or "ring-10 ladder is unavailable"),
        )
    ladder = primary.get("ladder")
    if not isinstance(ladder, list):
        return None, None, "ring-10 ladder is missing"
    estimates = {
        item.get("identity"): item
        for item in ladder
        if isinstance(item, Mapping) and isinstance(item.get("identity"), str)
    }
    if anchor_identity is None:
        return None, None, "common anchor identity is unavailable"
    anchor_estimate = estimates.get(anchor_identity)
    if anchor_estimate is None:
        return (
            None,
            None,
            f"common anchor {anchor_identity} is absent from ring-10 ladder",
        )
    anchor_rating = _number(anchor_estimate.get("rating"))
    anchor_standard_error = _number(anchor_estimate.get("standard_error"))
    if (
        anchor_rating is None
        or anchor_standard_error is None
        or anchor_standard_error < 0
    ):
        return (
            None,
            None,
            f"common anchor {anchor_identity} has an invalid Elo estimate",
        )
    serialized_anchor = {
        "identity": anchor_identity,
        "step": anchor_estimate.get("step"),
        "rating_elo": anchor_rating,
        "standard_error_elo": anchor_standard_error,
        "selection": "comparison_common_anchor",
    }
    if not evaluations:
        return (
            None,
            serialized_anchor,
            "no terminal candidate evaluation is available",
        )

    def selection_key(evaluation: Mapping[str, object]) -> tuple[bool, int, int, str]:
        candidate = str(evaluation["candidate"])
        estimate = estimates.get(candidate)
        publication = publications.get(candidate)
        step = (
            _nonnegative_integer(estimate.get("step")) if estimate is not None else None
        )
        if step is None and publication is not None:
            step = _nonnegative_integer(publication.get("model_step"))
        return (
            step is not None,
            step if step is not None else -1,
            _positive_timestamp(evaluation.get("completed_ns")) or 0,
            candidate,
        )

    selected = max(evaluations, key=selection_key)
    identity = str(selected["candidate"])
    estimate = estimates.get(identity)
    if estimate is None:
        return (
            None,
            serialized_anchor,
            f"latest terminal candidate {identity} is absent from ring-10 ladder",
        )
    rating = _number(estimate.get("rating"))
    standard_error = _number(estimate.get("standard_error"))
    if rating is None or standard_error is None or standard_error < 0:
        return (
            None,
            serialized_anchor,
            f"latest terminal candidate {identity} has invalid Elo estimate",
        )
    lower = rating - ONE_SIDED_95_NORMAL_QUANTILE * standard_error
    return (
        {
            "identity": identity,
            "step": estimate.get("step"),
            "rating_elo": rating,
            "standard_error_elo": standard_error,
            "one_sided_95_lower_rating_elo": lower,
            "two_sided_95_confidence_interval_elo": estimate.get("confidence_interval"),
            "decisive_games": estimate.get("decisive_games"),
            "terminal_decision": selected.get("decision"),
            "terminal_completed_ns": selected.get("completed_ns"),
            "selection": "latest_terminal_maximum_step",
        },
        serialized_anchor,
        None,
    )


def _champion_frontier(
    report: Mapping[str, object],
    evaluations: Sequence[Mapping[str, object]],
    *,
    anchor_identity: str | None,
) -> tuple[dict[str, object] | None, str | None]:
    autonomous = _mapping(report.get("autonomous_elo")) or {}
    primary = _mapping(autonomous.get("primary_ring_10")) or {}
    if primary.get("status") != "available":
        return None, str(primary.get("reason") or "ring-10 ladder is unavailable")
    ladder = primary.get("ladder")
    if not isinstance(ladder, list):
        return None, "ring-10 ladder is missing"
    estimates = {
        item.get("identity"): item
        for item in ladder
        if isinstance(item, Mapping) and isinstance(item.get("identity"), str)
    }
    if anchor_identity is None:
        return None, "common anchor identity is unavailable"
    if anchor_identity not in estimates:
        return None, f"common anchor {anchor_identity} is absent from ring-10 ladder"

    current = anchor_identity
    promotions = []
    for evaluation in evaluations:
        if evaluation.get("decision") != "promote":
            continue
        baseline = evaluation.get("baseline")
        candidate = evaluation.get("candidate")
        if baseline != current:
            return (
                None,
                "promoted champion chain is not contiguous: "
                f"expected baseline {current!r}, observed {baseline!r}",
            )
        if not isinstance(candidate, str) or candidate not in estimates:
            return (
                None,
                f"promoted champion {candidate!r} is absent from ring-10 ladder",
            )
        promotions.append(
            {
                "from_identity": current,
                "to_identity": candidate,
                "completed_ns": evaluation.get("completed_ns"),
                "path": evaluation.get("path"),
            }
        )
        current = candidate

    estimate = estimates[current]
    rating = _number(estimate.get("rating"))
    standard_error = _number(estimate.get("standard_error"))
    if rating is None or standard_error is None or standard_error < 0:
        return None, f"champion frontier {current} has an invalid Elo estimate"
    return (
        {
            "identity": current,
            "step": estimate.get("step"),
            "rating_elo": rating,
            "standard_error_elo": standard_error,
            "two_sided_95_confidence_interval_elo": estimate.get("confidence_interval"),
            "decisive_games": estimate.get("decisive_games"),
            "promotion_count": len(promotions),
            "promotions": promotions,
            "selection": "chronological_promotions_from_common_anchor",
            "non_promoted_terminal_count": len(evaluations) - len(promotions),
        },
        None,
    )


def _weighted_champion_frontier(
    evaluations: Sequence[Mapping[str, object]],
    *,
    anchor_identity: str | None,
) -> tuple[dict[str, object] | None, str | None, str | None]:
    weighted = [
        summary
        for evaluation in evaluations
        if isinstance(
            summary := evaluation.get("weighted_aggregate"),
            Mapping,
        )
    ]
    if not weighted:
        return None, None, None
    objectives = {
        str(summary.get("objective"))
        for summary in weighted
        if isinstance(summary.get("objective"), str)
    }
    objective = next(iter(objectives)) if len(objectives) == 1 else None
    if objective is None:
        return None, "weighted promotion objective changed within the run", None
    if len(weighted) != len(evaluations):
        return (
            None,
            "weighted promotion data is missing from a terminal evaluation",
            objective,
        )
    if anchor_identity is None:
        return None, "common anchor identity is unavailable", objective

    current = anchor_identity
    point_gain = 0.0
    lower_gain = 0.0
    promotions = []
    promoted_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation.get("decision") == "promote"
    ]
    objective_document = json.loads(objective)
    familywise_error = _number(objective_document.get("lower_error_probability"))
    if familywise_error is None or not 0 < familywise_error < 1:
        return None, "weighted objective has invalid lower error probability", objective
    for promotion_index, evaluation in enumerate(promoted_evaluations):
        baseline = evaluation.get("baseline")
        candidate = evaluation.get("candidate")
        if baseline != current:
            return (
                None,
                "weighted promoted champion chain is not contiguous: "
                f"expected baseline {current!r}, observed {baseline!r}",
                objective,
            )
        summary = evaluation.get("weighted_aggregate")
        assert isinstance(summary, Mapping)
        link_elo = _number(summary.get("elo_difference"))
        raw_scores = summary.get("block_scores")
        if (
            link_elo is None
            or not isinstance(raw_scores, Sequence)
            or isinstance(raw_scores, str | bytes)
            or not raw_scores
            or any(
                isinstance(value, bool) or not isinstance(value, int | float)
                for value in raw_scores
            )
        ):
            return (
                None,
                f"weighted promotion {candidate!r} has invalid Elo evidence",
                objective,
            )
        link_error_probability = familywise_error / (2 ** (promotion_index + 1))
        link_lower_score, _ = bounded_confidence_sequence(
            tuple(float(value) for value in raw_scores),
            error_probability=link_error_probability,
        )
        link_lower = elo_from_probability(link_lower_score)
        if not isinstance(candidate, str) or not candidate:
            return None, "weighted promoted champion identity is invalid", objective
        point_gain += link_elo
        lower_gain += link_lower
        promotions.append(
            {
                "from_identity": current,
                "to_identity": candidate,
                "completed_ns": evaluation.get("completed_ns"),
                "path": evaluation.get("path"),
                "weighted_elo_difference": link_elo,
                "weighted_anytime_lower_elo": link_lower,
                "link_error_probability": link_error_probability,
                "complete_blocks": summary.get("complete_blocks"),
            }
        )
        current = candidate
    return (
        {
            "identity": current,
            "weighted_elo_gained": point_gain,
            "weighted_elo_one_sided_lower_bound": lower_gain,
            "promotion_count": len(promotions),
            "promotions": promotions,
            "selection": "chronological_weighted_promotions_from_common_anchor",
            "non_promoted_terminal_count": len(evaluations) - len(promotions),
            "objective": json.loads(objective),
            "lower_bound_method": (
                "sum_of_geometric_alpha_spending_anytime_lower_bounds"
            ),
            "familywise_error_probability": familywise_error,
            "link_error_spending": "alpha / 2^(promotion_index + 1)",
        },
        None,
        objective,
    )


def _per_ring_diagnostics(
    evaluations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    observed_rings: set[int] = set()
    for evaluation in evaluations:
        per_ring = evaluation.get("per_ring")
        if not isinstance(per_ring, Mapping):
            continue
        observed_rings.update(
            int(ring) for ring in per_ring if str(ring).isdigit() and int(ring) > 0
        )
    rings = sorted(observed_rings)
    summaries = []
    for ring in rings:
        observations = []
        for evaluation in evaluations:
            per_ring = evaluation.get("per_ring")
            if not isinstance(per_ring, Mapping):
                continue
            summary = per_ring.get(str(ring), per_ring.get(ring))
            if not isinstance(summary, Mapping):
                continue
            observations.append(
                {
                    "path": evaluation.get("path"),
                    "candidate": evaluation.get("candidate"),
                    "baseline": evaluation.get("baseline"),
                    "decision": evaluation.get("decision"),
                    "completed_ns": evaluation.get("completed_ns"),
                    "summary": dict(summary),
                }
            )
        summaries.append(
            {
                "ring": ring,
                "observation_count": len(observations),
                "latest": observations[-1] if observations else None,
                "observations": observations,
            }
        )
    return {
        "status": "available" if summaries else "unavailable",
        "rings": summaries,
    }


def _winner_snapshot(
    root: Path,
    *,
    label: str,
    frontier: Mapping[str, object] | None,
    anchor: Mapping[str, object],
    failures: list[dict[str, object]],
) -> tuple[dict[str, object] | None, str | None]:
    if frontier is None:
        return None, "champion frontier is unavailable"
    run_path = root / "run.json"
    champion_path = root / "learner" / "champion.json"
    run = _read_json(run_path, failures=failures)
    champion = _read_json(champion_path, failures=failures)
    if run is None or champion is None:
        return None, "run identity or champion pointer could not be parsed"
    identity = champion.get("model_identity")
    step = _nonnegative_integer(champion.get("model_step"))
    if identity != frontier.get("identity") or step != frontier.get("step"):
        return (
            None,
            "champion pointer does not match the chronological promoted frontier "
            f"(pointer={identity!r}@{step!r}, "
            f"frontier={frontier.get('identity')!r}@{frontier.get('step')!r})",
        )
    run_id = run.get("run_id")
    generation_family = run.get("generation_family")
    created_ns = _positive_timestamp(run.get("created_ns"))
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(generation_family, str)
        or not generation_family
        or created_ns is None
    ):
        return None, "run identity is incomplete"
    return (
        {
            "schema_version": 1,
            "status": "verified",
            "label": label,
            "run_root": str(root),
            "run_identity": {
                "run_id": run_id,
                "generation_family": generation_family,
                "created_ns": created_ns,
            },
            "run_identity_artifact": {
                "path": str(run_path),
                "sha256": _sha256(run_path),
            },
            "champion": {
                "model_identity": identity,
                "model_step": step,
                "updated_ns": champion.get("updated_ns"),
            },
            "champion_pointer_artifact": {
                "path": str(champion_path),
                "sha256": _sha256(champion_path),
            },
            "source_anchor": {
                "model_identity": anchor.get("identity"),
                "model_step": anchor.get("step"),
            },
            "selection": "guarded_chronological_champion_frontier",
        },
        None,
    )


def _add_reason(reasons: list[dict[str, str]], code: str, message: str) -> None:
    reason = {"code": code, "message": message}
    if reason not in reasons:
        reasons.append(reason)


def _empty_payload(
    *,
    label: str,
    root: Path,
    provisioned_gpus: int,
    guard_rings: Sequence[int],
    guard_floor_elo: float,
) -> dict[str, object]:
    return {
        "rank": None,
        "label": label,
        "status": "ineligible",
        "eligible": False,
        "ineligibility_reasons": [],
        "run_root": str(root),
        "run_id": None,
        "generation_family": None,
        "source_report_status": "error",
        "training_objective": None,
        "promotion_objective": None,
        "anchor": {
            "identity": None,
            "step": None,
            "rating_elo": None,
            "standard_error_elo": None,
            "selection": None,
        },
        "endpoint": None,
        "diagnostics": {
            "latest_terminal_endpoint": None,
            "latest_terminal_endpoint_error": None,
        },
        "champion_frontier": None,
        "weighted_champion_frontier": None,
        "verified_winner_snapshot": None,
        "measurement": {
            "source": None,
            "status": "unavailable",
            "started_ns": None,
            "stopped_ns": None,
            "cutoff_ns": None,
            "wall_seconds": None,
            "resource_released_ns": None,
            "resource_wall_seconds": None,
            "stop_reason": None,
            "exit_code": None,
            "outcome": None,
            "attempt_count": None,
            "failure": None,
            "lifecycle_status": None,
            "completion_status": None,
            "warnings": [],
            "failure_domain": None,
            "failure_phase": None,
            "teardown_status": None,
            "teardown": None,
            "integrity_status": None,
            "integrity": None,
            "wall_budget_seconds": None,
            "leaf_budget": None,
        },
        "resource_accounting": {
            "source": None,
            "started_ns": None,
            "measurement_cutoff_ns": None,
            "resource_released_ns": None,
            "measurement_wall_seconds": None,
            "teardown_wall_seconds": None,
            "total_provisioned_wall_seconds": None,
            "total_provisioned_wall_hours": None,
            "provisioned_gpus": provisioned_gpus,
            "total_provisioned_gpu_hours": None,
        },
        "deployment_metric": {
            "name": (
                "guarded_champion_frontier_ring_10_elo_lcb_per_"
                "total_provisioned_wall_hour"
            ),
            "value": None,
            "point_value": None,
            "time_basis": "measurement_started_ns_to_resource_released_ns",
            "selection": "chronological_promotions_only",
        },
        "ring_10_deployment_metric": None,
        "weighted_deployment_metric": {
            "name": (
                "weighted_champion_frontier_elo_lcb_per_total_provisioned_wall_hour"
            ),
            "available": False,
            "value": None,
            "point_value": None,
            "time_basis": "measurement_started_ns_to_resource_released_ns",
            "selection": "chronological_weighted_promotions_only",
        },
        "efficiency": {
            "accounting_basis": None,
            "started_ns": None,
            "stopped_ns": None,
            "wall_seconds": None,
            "wall_hours": None,
            "provisioned_gpus": provisioned_gpus,
            "provisioned_gpu_hours": None,
            "ring_10_elo_gained": None,
            "ring_10_elo_gain_conservative_standard_error": None,
            "ring_10_elo_one_sided_95_lower_bound": None,
            "ring_10_elo_per_wall_hour": None,
            "ring_10_elo_lcb_per_wall_hour": None,
            "ring_10_elo_per_provisioned_gpu_hour": None,
            "ring_10_elo_lcb_per_provisioned_gpu_hour": None,
        },
        "guardrails": {
            "status": "unavailable",
            "configured_rings": list(guard_rings),
            "configured_floor_elo": guard_floor_elo,
            "evidence_field": "promotion.ring_floors.<ring>.anytime_lower_elo",
            "terminal_evaluation_count": 0,
            "reject_ring_regression_count": 0,
            "rings": [],
            "terminal_evaluations": [],
        },
        "per_ring_diagnostics": {
            "status": "unavailable",
            "rings": [],
        },
        "candidate_publish_to_terminal": {
            "status": "unavailable",
            "terminal_candidate_count": 0,
            "measured_candidate_count": 0,
            "seconds": None,
            "records": [],
            "missing": [],
        },
        "learner": {
            "updates_per_new_sample": {
                "availability": "not_recorded",
                "statistics": None,
            },
            "replay_waits": {
                "availability": "not_recorded_in_jsonl",
                "events": 0,
                "completed_intervals": 0,
                "open_intervals": 0,
                "seconds": None,
            },
            "device_duty_fraction": None,
            "data_wait_seconds": None,
            "data_wait_fraction": None,
            "end_to_end_examples_per_second": None,
            "measured_wall_seconds": None,
        },
        "actors": {
            "worker_count": None,
            "games": None,
            "samples": None,
            "evaluator_rows": None,
            "aggregate_games_per_second": None,
            "aggregate_samples_per_second": None,
            "aggregate_evaluator_rows_per_second": None,
            "samples_per_physical_gpu_second": None,
        },
        "parse_failure_count": 0,
        "parse_failures": [],
        "error": None,
    }


def _analyze_treatment(
    label: str,
    run_root: str | Path,
    *,
    provisioned_gpus: int,
    guard_rings: Sequence[int],
    guard_floor_elo: float,
) -> _Treatment:
    root = Path(run_root).expanduser().resolve()
    payload = _empty_payload(
        label=label,
        root=root,
        provisioned_gpus=provisioned_gpus,
        guard_rings=guard_rings,
        guard_floor_elo=guard_floor_elo,
    )
    reasons: list[dict[str, str]] = []
    try:
        report = build_strength_efficiency_report(
            root,
            provisioned_gpus=provisioned_gpus,
        )
    except (OSError, TypeError, ValueError) as error:
        message = f"{type(error).__name__}: {error}"
        payload["error"] = message
        _add_reason(reasons, "report_error", message)
        return _Treatment(label, payload, None, None, None, reasons)

    payload["run_id"] = report.get("run_id")
    payload["generation_family"] = report.get("generation_family")
    payload["source_report_status"] = report.get("status")
    raw_failures = report.get("parse_failures")
    failures = (
        [dict(failure) for failure in raw_failures if isinstance(failure, Mapping)]
        if isinstance(raw_failures, list)
        else []
    )
    measurement, anchor, incomplete_measurement = _measurement_context(
        root,
        report,
        failures=failures,
    )
    payload["measurement"] = measurement
    training_objective = measurement.get("training_objective")
    payload["training_objective"] = training_objective
    payload["promotion_objective"] = measurement.get("promotion_objective")
    payload["anchor"] = anchor
    raw_anchor_identity = anchor.get("identity")
    anchor_identity = (
        raw_anchor_identity if isinstance(raw_anchor_identity, str) else None
    )
    measurement_started_ns = _positive_timestamp(measurement.get("started_ns"))
    measurement_stopped_ns = _positive_timestamp(measurement.get("stopped_ns"))
    scoped_measurement = (
        measurement.get("source") == "ablation.json"
        and measurement_started_ns is not None
        and measurement_stopped_ns is not None
    )

    arena_results = _arena_results(root, failures=failures)
    if scoped_measurement:
        assert measurement_started_ns is not None
        assert measurement_stopped_ns is not None
        arena_results = [
            result
            for result in arena_results
            if (completed_ns := _positive_timestamp(result.get("completed_ns"))) is None
            or measurement_started_ns <= completed_ns <= measurement_stopped_ns
        ]
    evaluations = _terminal_evaluations(
        arena_results,
        guard_rings=guard_rings,
        guard_floor_elo=guard_floor_elo,
        failures=failures,
    )
    if training_objective == "ring10_only":
        for evaluation in evaluations:
            per_ring = _mapping(evaluation.get("per_ring")) or {}
            incompatible_rings = sorted(
                int(ring) for ring in per_ring if int(ring) != 10
            )
            if incompatible_rings:
                failures.append(
                    _failure(
                        str(evaluation.get("path")),
                        "ring10_only arena evidence contains incompatible rings "
                        f"{incompatible_rings}",
                    )
                )
    publications = _model_publications(root, failures=failures)
    metrics_path = root / "learner" / "metrics.jsonl"
    learner_records = _read_jsonl(metrics_path, failures=failures)
    if scoped_measurement:
        assert measurement_started_ns is not None
        assert measurement_stopped_ns is not None
        learner_records = [
            (line, record)
            for line, record in learner_records
            if (timestamp_ns := _positive_timestamp(record.get("timestamp_ns"))) is None
            or measurement_started_ns <= timestamp_ns <= measurement_stopped_ns
        ]
    payload["learner"] = _learner_summary(
        report,
        learner_records,
        metrics_path=metrics_path,
        failures=failures,
    )
    payload["actors"] = _actor_summary(report)
    guardrails = _guardrail_summary(
        evaluations,
        guard_rings=guard_rings,
        guard_floor_elo=guard_floor_elo,
    )
    payload["guardrails"] = guardrails
    payload["per_ring_diagnostics"] = _per_ring_diagnostics(evaluations)
    payload["candidate_publish_to_terminal"] = _candidate_latency_summary(
        evaluations,
        publications,
        failures=failures,
    )
    endpoint, anchor_estimate, endpoint_error = _endpoint(
        report,
        evaluations,
        publications,
        anchor_identity=anchor_identity,
    )
    if anchor_estimate is not None:
        anchor_estimate["selection"] = anchor.get("selection")
        anchor = anchor_estimate
        payload["anchor"] = anchor
    payload["endpoint"] = endpoint
    payload["diagnostics"] = {
        "latest_terminal_endpoint": endpoint,
        "latest_terminal_endpoint_error": endpoint_error,
    }
    frontier, frontier_error = _champion_frontier(
        report,
        evaluations,
        anchor_identity=anchor_identity,
    )
    payload["champion_frontier"] = frontier
    (
        weighted_frontier,
        weighted_frontier_error,
        weighted_objective,
    ) = _weighted_champion_frontier(
        evaluations,
        anchor_identity=anchor_identity,
    )
    payload["weighted_champion_frontier"] = weighted_frontier
    promotion_objective = measurement.get("promotion_objective")
    profile_evidence = measurement.get("profile")
    raw_profile_failure_path = (
        profile_evidence.get("path") if isinstance(profile_evidence, Mapping) else None
    )
    profile_failure_path: Path | str = (
        raw_profile_failure_path
        if isinstance(raw_profile_failure_path, Path | str)
        else root
    )
    if weighted_objective is not None and promotion_objective != "weighted_aggregate":
        failures.append(
            _failure(
                profile_failure_path,
                "arena weighted evidence differs from the frozen profile objective",
            )
        )
    elif (
        promotion_objective == "weighted_aggregate"
        and evaluations
        and weighted_objective is None
    ):
        failures.append(
            _failure(
                profile_failure_path,
                "frozen weighted objective lacks compatible arena evidence",
            )
        )
    winner_snapshot, snapshot_error = _winner_snapshot(
        root,
        label=label,
        frontier=frontier,
        anchor=anchor,
        failures=failures,
    )
    payload["verified_winner_snapshot"] = winner_snapshot

    normalized_failures = _normalized_failures(failures)
    payload["parse_failure_count"] = len(normalized_failures)
    payload["parse_failures"] = normalized_failures
    if report.get("status") != "complete" or normalized_failures:
        first = normalized_failures[0] if normalized_failures else None
        if first is not None:
            line_suffix = f":{first['line']}" if first["line"] is not None else ""
            detail = f"; first: {first['path']}{line_suffix}: {first['error']}"
        else:
            detail = ""
        _add_reason(
            reasons,
            "parse_failure",
            f"{len(normalized_failures)} parse/schema failure(s){detail}",
        )
    if incomplete_measurement is not None:
        _add_reason(
            reasons,
            "incomplete_measurement",
            incomplete_measurement,
        )
    if anchor_estimate is None:
        _add_reason(
            reasons,
            "missing_common_anchor",
            endpoint_error or "common anchor is unavailable in the ring-10 ladder",
        )
    if frontier is None:
        _add_reason(
            reasons,
            "missing_champion_frontier",
            frontier_error or "ring-10 champion frontier is unavailable",
        )
    if weighted_objective is not None and weighted_frontier is None:
        _add_reason(
            reasons,
            "missing_weighted_champion_frontier",
            weighted_frontier_error
            or "weighted champion frontier evidence is unavailable",
        )
    if winner_snapshot is None:
        _add_reason(
            reasons,
            "unverified_winner_snapshot",
            snapshot_error or "winner snapshot could not be verified",
        )
    if (_nonnegative_integer(guardrails.get("reject_ring_regression_count")) or 0) > 0:
        _add_reason(
            reasons,
            "reject_ring_regression",
            "at least one terminal arena decision rejected a ring regression",
        )
    ring_summaries = guardrails["rings"]
    assert isinstance(ring_summaries, list)
    below = [
        ring
        for summary in ring_summaries
        if isinstance(summary, Mapping)
        and (ring := _nonnegative_integer(summary.get("ring"))) is not None
        and (_nonnegative_integer(summary.get("below_floor_count")) or 0) > 0
    ]
    missing = [
        ring
        for summary in ring_summaries
        if isinstance(summary, Mapping)
        and (ring := _nonnegative_integer(summary.get("ring"))) is not None
        and (
            (_nonnegative_integer(summary.get("missing_evidence_count")) or 0) > 0
            or (_nonnegative_integer(summary.get("observation_count")) or 0) == 0
        )
    ]
    if below:
        _add_reason(
            reasons,
            "guard_evidence_below_floor",
            f"ring(s) {below} have anytime lower Elo below {guard_floor_elo}",
        )
    if missing:
        _add_reason(
            reasons,
            "missing_guard_evidence",
            f"ring(s) {missing} lack terminal non-inferiority evidence",
        )

    wall_seconds = _number(measurement.get("wall_seconds"))
    wall_hours = (
        wall_seconds / 3_600.0
        if wall_seconds is not None and wall_seconds > 0
        else None
    )
    if wall_hours is None:
        _add_reason(
            reasons,
            "invalid_wall_time",
            "run has no positive observed wall-clock interval",
        )
    provisioned_gpu_hours = (
        provisioned_gpus * wall_hours if wall_hours is not None else None
    )
    resource_wall_seconds = _number(measurement.get("resource_wall_seconds"))
    resource_wall_hours = (
        resource_wall_seconds / 3_600.0
        if resource_wall_seconds is not None and resource_wall_seconds > 0
        else None
    )
    if resource_wall_hours is None:
        _add_reason(
            reasons,
            "invalid_resource_wall_time",
            "run has no positive start-to-resource-release interval",
        )
    total_provisioned_gpu_hours = (
        provisioned_gpus * resource_wall_hours
        if resource_wall_hours is not None
        else None
    )
    teardown_wall_seconds = (
        resource_wall_seconds - wall_seconds
        if resource_wall_seconds is not None
        and wall_seconds is not None
        and resource_wall_seconds >= wall_seconds
        else None
    )
    payload["resource_accounting"] = {
        "source": measurement.get("source"),
        "started_ns": measurement.get("started_ns"),
        "measurement_cutoff_ns": measurement.get("cutoff_ns"),
        "resource_released_ns": measurement.get("resource_released_ns"),
        "measurement_wall_seconds": wall_seconds,
        "teardown_wall_seconds": teardown_wall_seconds,
        "total_provisioned_wall_seconds": resource_wall_seconds,
        "total_provisioned_wall_hours": resource_wall_hours,
        "provisioned_gpus": provisioned_gpus,
        "total_provisioned_gpu_hours": total_provisioned_gpu_hours,
    }
    anchor_rating = _number(anchor.get("rating_elo"))
    anchor_standard_error = _number(anchor.get("standard_error_elo"))
    rating = _number(endpoint.get("rating_elo")) if endpoint is not None else None
    endpoint_standard_error = (
        _number(endpoint.get("standard_error_elo")) if endpoint is not None else None
    )
    elo_gained = (
        rating - anchor_rating
        if rating is not None and anchor_rating is not None
        else None
    )
    conservative_standard_error = (
        anchor_standard_error + endpoint_standard_error
        if anchor_standard_error is not None and endpoint_standard_error is not None
        else None
    )
    elo_lcb = (
        elo_gained - ONE_SIDED_95_NORMAL_QUANTILE * conservative_standard_error
        if elo_gained is not None and conservative_standard_error is not None
        else None
    )
    endpoint_point_score = (
        elo_gained / wall_hours
        if elo_gained is not None and wall_hours is not None
        else None
    )
    endpoint_ranking_score = (
        elo_lcb / wall_hours if elo_lcb is not None and wall_hours is not None else None
    )
    payload["efficiency"] = {
        "accounting_basis": (
            "ablation.json measurement interval"
            if measurement.get("source") == "ablation.json"
            else "run.created_ns through last observed run timestamp"
        ),
        "started_ns": measurement.get("started_ns"),
        "stopped_ns": measurement.get("stopped_ns"),
        "wall_seconds": wall_seconds,
        "wall_hours": wall_hours,
        "provisioned_gpus": provisioned_gpus,
        "provisioned_gpu_hours": provisioned_gpu_hours,
        "ring_10_elo_gained": elo_gained,
        "ring_10_elo_gain_conservative_standard_error": (conservative_standard_error),
        "ring_10_elo_one_sided_95_lower_bound": elo_lcb,
        "ring_10_elo_per_wall_hour": endpoint_point_score,
        "ring_10_elo_lcb_per_wall_hour": endpoint_ranking_score,
        "ring_10_elo_per_provisioned_gpu_hour": (
            elo_gained / provisioned_gpu_hours
            if elo_gained is not None and provisioned_gpu_hours
            else None
        ),
        "ring_10_elo_lcb_per_provisioned_gpu_hour": (
            elo_lcb / provisioned_gpu_hours
            if elo_lcb is not None and provisioned_gpu_hours
            else None
        ),
    }
    frontier_rating = (
        _number(frontier.get("rating_elo")) if frontier is not None else None
    )
    frontier_standard_error = (
        _number(frontier.get("standard_error_elo")) if frontier is not None else None
    )
    frontier_gain = (
        frontier_rating - anchor_rating
        if frontier_rating is not None and anchor_rating is not None
        else None
    )
    if (
        frontier is not None
        and frontier.get("identity") == anchor.get("identity")
        and frontier_gain is not None
    ):
        frontier_gain_standard_error = 0.0
    else:
        frontier_gain_standard_error = (
            anchor_standard_error + frontier_standard_error
            if anchor_standard_error is not None and frontier_standard_error is not None
            else None
        )
    frontier_lcb = (
        frontier_gain - ONE_SIDED_95_NORMAL_QUANTILE * frontier_gain_standard_error
        if frontier_gain is not None and frontier_gain_standard_error is not None
        else None
    )
    point_score = (
        frontier_gain / resource_wall_hours
        if frontier_gain is not None and resource_wall_hours is not None
        else None
    )
    ranking_score = (
        frontier_lcb / resource_wall_hours
        if frontier_lcb is not None and resource_wall_hours is not None
        else None
    )
    ring_10_deployment_metric = {
        "name": (
            "guarded_champion_frontier_ring_10_elo_lcb_per_total_provisioned_wall_hour"
        ),
        "value": ranking_score,
        "point_value": point_score,
        "champion_frontier_ring_10_elo_gained": frontier_gain,
        "champion_frontier_ring_10_elo_gain_conservative_standard_error": (
            frontier_gain_standard_error
        ),
        "champion_frontier_ring_10_elo_one_sided_95_lower_bound": frontier_lcb,
        "total_provisioned_wall_hours": resource_wall_hours,
        "total_provisioned_gpu_hours": total_provisioned_gpu_hours,
        "guardrail_status": guardrails.get("status"),
        "time_basis": "measurement_started_ns_to_resource_released_ns",
        "selection": "chronological_promotions_only",
    }
    payload["ring_10_deployment_metric"] = ring_10_deployment_metric
    payload["deployment_metric"] = ring_10_deployment_metric
    weighted_gain = (
        _number(weighted_frontier.get("weighted_elo_gained"))
        if weighted_frontier is not None
        else None
    )
    weighted_lower = (
        _number(weighted_frontier.get("weighted_elo_one_sided_lower_bound"))
        if weighted_frontier is not None
        else None
    )
    weighted_point_score = (
        weighted_gain / resource_wall_hours
        if weighted_gain is not None and resource_wall_hours is not None
        else None
    )
    weighted_ranking_score = (
        weighted_lower / resource_wall_hours
        if weighted_lower is not None and resource_wall_hours is not None
        else None
    )
    payload["weighted_deployment_metric"] = {
        "name": ("weighted_champion_frontier_elo_lcb_per_total_provisioned_wall_hour"),
        "available": weighted_frontier is not None,
        "value": weighted_ranking_score,
        "point_value": weighted_point_score,
        "champion_frontier_weighted_elo_gained": weighted_gain,
        "champion_frontier_weighted_elo_one_sided_lower_bound": weighted_lower,
        "total_provisioned_wall_hours": resource_wall_hours,
        "total_provisioned_gpu_hours": total_provisioned_gpu_hours,
        "objective": (
            weighted_frontier.get("objective")
            if weighted_frontier is not None
            else None
        ),
        "lower_bound_method": (
            weighted_frontier.get("lower_bound_method")
            if weighted_frontier is not None
            else None
        ),
        "time_basis": "measurement_started_ns_to_resource_released_ns",
        "selection": "chronological_weighted_promotions_only",
    }
    return _Treatment(
        label,
        payload,
        anchor_identity if anchor_estimate is not None else None,
        ranking_score,
        point_score,
        reasons,
        weighted_ranking_score,
        weighted_point_score,
        weighted_objective,
        training_objective if isinstance(training_objective, str) else None,
    )


def _validate_runs(
    runs: Mapping[str, str | Path],
) -> list[tuple[str, Path]]:
    if len(runs) < 2:
        raise ValueError("at least two distinct --run treatments are required")
    normalized = []
    seen_roots: dict[Path, str] = {}
    for label, raw_root in sorted(runs.items()):
        if not _LABEL_PATTERN.fullmatch(label):
            raise ValueError(
                f"run label {label!r} must match {_LABEL_PATTERN.pattern!r}"
            )
        root = Path(raw_root).expanduser().resolve()
        if root in seen_roots:
            raise ValueError(
                f"run labels {seen_roots[root]!r} and {label!r} use the same root: {root}"
            )
        seen_roots[root] = label
        normalized.append((label, root))
    return normalized


def build_elo_ablation_comparison(
    runs: Mapping[str, str | Path],
    *,
    provisioned_gpus: int = DEFAULT_PROVISIONED_GPUS,
    guard_rings: Sequence[int] = DEFAULT_GUARD_RINGS,
    guard_floor_elo: float = DEFAULT_GUARD_FLOOR_ELO,
    forced_ineligible: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a deterministic guarded comparison for two or more run roots."""
    if type(provisioned_gpus) is not int or provisioned_gpus <= 0:
        raise ValueError("provisioned_gpus must be a positive integer")
    if (
        isinstance(guard_floor_elo, bool)
        or not isinstance(guard_floor_elo, int | float)
        or not math.isfinite(float(guard_floor_elo))
    ):
        raise ValueError("guard_floor_elo must be finite")
    parsed_rings = tuple(sorted(guard_rings))
    if any(type(ring) is not int or ring <= 0 for ring in parsed_rings):
        raise ValueError("guard_rings must contain positive integers")
    if len(set(parsed_rings)) != len(parsed_rings):
        raise ValueError("guard_rings must not contain duplicates")
    normalized_runs = _validate_runs(runs)
    forced = dict(forced_ineligible or {})
    unknown_forced = sorted(set(forced) - {label for label, _root in normalized_runs})
    if unknown_forced:
        raise ValueError(
            f"forced-ineligible labels are not configured runs: {unknown_forced}"
        )
    if any(
        not isinstance(reason, str) or not reason.strip() for reason in forced.values()
    ):
        raise ValueError("forced-ineligible reasons must be non-empty strings")
    treatments = [
        _analyze_treatment(
            label,
            root,
            provisioned_gpus=provisioned_gpus,
            guard_rings=parsed_rings,
            guard_floor_elo=float(guard_floor_elo),
        )
        for label, root in normalized_runs
    ]

    anchors = {treatment.label: treatment.anchor_identity for treatment in treatments}
    distinct = {identity for identity in anchors.values() if identity is not None}
    common_available = len(distinct) == 1 and all(
        identity is not None for identity in anchors.values()
    )
    common_identity = next(iter(distinct)) if common_available else None
    errors = []
    if not common_available:
        by_treatment = ", ".join(
            f"{label}={identity or '<missing>'}"
            for label, identity in sorted(anchors.items())
        )
        message = f"run roots do not expose one common Elo anchor: {by_treatment}"
        errors.append({"code": "missing_common_anchor", "message": message})
        for treatment in treatments:
            _add_reason(treatment.reasons, "missing_common_anchor", message)

    training_objectives = {treatment.training_objective for treatment in treatments}
    compatible_training_objective = (
        next(iter(training_objectives))
        if len(training_objectives) == 1 and None not in training_objectives
        else None
    )
    weighted_objectives = {
        treatment.weighted_objective
        for treatment in treatments
        if treatment.weighted_objective is not None
    }
    if compatible_training_objective is None:
        ranking_objective = "incompatible"
        message = (
            "all treatments must expose the same training objective; "
            "generalist and ring10_only runs cannot be mixed"
        )
        errors.append({"code": "incompatible_training_objectives", "message": message})
        for treatment in treatments:
            _add_reason(
                treatment.reasons,
                "incompatible_training_objectives",
                message,
            )
    elif compatible_training_objective == "ring10_only":
        if weighted_objectives:
            ranking_objective = "incompatible"
            message = "ring10_only runs cannot expose weighted promotion evidence"
            errors.append(
                {"code": "incompatible_promotion_objectives", "message": message}
            )
            for treatment in treatments:
                _add_reason(
                    treatment.reasons,
                    "incompatible_promotion_objectives",
                    message,
                )
        elif parsed_rings:
            ranking_objective = "incompatible"
            message = "ring10_only comparisons cannot configure guard rings"
            errors.append({"code": "objective_guard_mismatch", "message": message})
            for treatment in treatments:
                _add_reason(treatment.reasons, "objective_guard_mismatch", message)
        else:
            ranking_objective = "ring_10_only"
    elif not weighted_objectives:
        ranking_objective = "ring_10_guarded"
    elif len(weighted_objectives) == 1 and all(
        treatment.weighted_objective is not None for treatment in treatments
    ):
        ranking_objective = "weighted_aggregate"
    else:
        ranking_objective = "incompatible"
        message = (
            "all treatments must expose the same pre-registered weighted promotion "
            "objective; weighted and legacy metrics cannot be mixed"
        )
        errors.append({"code": "incompatible_promotion_objectives", "message": message})
        for treatment in treatments:
            _add_reason(
                treatment.reasons,
                "incompatible_promotion_objectives",
                message,
            )

    legacy_metric_name = (
        "guarded_champion_frontier_ring_10_elo_lcb_per_total_provisioned_wall_hour"
    )
    ring10_only_metric_name = (
        "ring10_only_champion_frontier_elo_lcb_per_total_provisioned_wall_hour"
    )
    weighted_metric_name = (
        "weighted_champion_frontier_elo_lcb_per_total_provisioned_wall_hour"
    )
    for treatment in treatments:
        if ranking_objective == "weighted_aggregate":
            treatment.ranking_score = treatment.weighted_ranking_score
            treatment.point_score = treatment.weighted_point_score
            weighted_metric = treatment.payload.get("weighted_deployment_metric")
            if isinstance(weighted_metric, Mapping):
                treatment.payload["deployment_metric"] = weighted_metric
        else:
            ring_metric = treatment.payload.get("ring_10_deployment_metric")
            if isinstance(ring_metric, Mapping):
                if ranking_objective == "ring_10_only":
                    objective_metric = {
                        **ring_metric,
                        "name": ring10_only_metric_name,
                        "objective": "ring10_only",
                    }
                    treatment.payload["ring_10_deployment_metric"] = objective_metric
                    treatment.payload["deployment_metric"] = objective_metric
                else:
                    treatment.payload["deployment_metric"] = ring_metric
        treatment.payload["ranking_objective"] = ranking_objective
        if treatment.ranking_score is None or treatment.point_score is None:
            _add_reason(
                treatment.reasons,
                "unavailable_ranking_metric",
                (
                    "weighted champion-frontier Elo/hour lower bound could not "
                    "be computed"
                    if ranking_objective == "weighted_aggregate"
                    else (
                        "ring10-only champion-frontier Elo/hour lower bound could "
                        "not be computed"
                        if ranking_objective == "ring_10_only"
                        else "guarded champion-frontier ring-10 Elo/hour lower bound "
                        "could not be computed"
                    )
                ),
            )
        if treatment.label in forced:
            _add_reason(
                treatment.reasons,
                "queue_arm_incomplete",
                forced[treatment.label],
            )
        treatment.reasons.sort(key=lambda reason: (reason["code"], reason["message"]))
        eligible = not treatment.reasons
        treatment.payload["eligible"] = eligible
        treatment.payload["status"] = (
            "eligible"
            if eligible
            else ("error" if treatment.payload.get("error") else "ineligible")
        )
        treatment.payload["ineligibility_reasons"] = treatment.reasons

    eligible_treatments = [
        treatment
        for treatment in treatments
        if treatment.payload["eligible"] is True
        and treatment.ranking_score is not None
        and treatment.point_score is not None
    ]

    def ranking_key(treatment: _Treatment) -> tuple[float, float, str]:
        assert treatment.ranking_score is not None
        assert treatment.point_score is not None
        return (
            -treatment.ranking_score,
            -treatment.point_score,
            treatment.label,
        )

    eligible_treatments.sort(key=ranking_key)
    for rank, treatment in enumerate(eligible_treatments, start=1):
        treatment.payload["rank"] = rank
    ineligible_treatments = sorted(
        (
            treatment
            for treatment in treatments
            if treatment.payload["eligible"] is not True
        ),
        key=lambda treatment: treatment.label,
    )
    ordered = [*eligible_treatments, *ineligible_treatments]
    status = "complete" if len(eligible_treatments) == len(treatments) else "incomplete"
    winner_snapshot = (
        eligible_treatments[0].payload.get("verified_winner_snapshot")
        if status == "complete" and eligible_treatments
        else None
    )
    serialized_winner_snapshot = (
        dict(winner_snapshot)
        if isinstance(winner_snapshot, Mapping)
        and winner_snapshot.get("status") == "verified"
        else None
    )
    if (
        serialized_winner_snapshot is not None
        and ranking_objective == "weighted_aggregate"
    ):
        serialized_winner_snapshot["selection"] = (
            "weighted_chronological_champion_frontier"
        )
        serialized_winner_snapshot["ranking_metric"] = weighted_metric_name
    elif serialized_winner_snapshot is not None and ranking_objective == "ring_10_only":
        serialized_winner_snapshot["selection"] = (
            "ring10_only_chronological_champion_frontier"
        )
        serialized_winner_snapshot["ranking_metric"] = ring10_only_metric_name
    selector_verified = serialized_winner_snapshot is not None
    ranking_metric = (
        weighted_metric_name
        if ranking_objective == "weighted_aggregate"
        else (
            ring10_only_metric_name
            if ranking_objective == "ring_10_only"
            else (
                legacy_metric_name if ranking_objective == "ring_10_guarded" else None
            )
        )
    )
    selector = {
        "status": "verified" if selector_verified else "unavailable",
        "ranking_metric": ranking_metric,
        "ranking_objective": ranking_objective,
        "winner_snapshot": serialized_winner_snapshot,
        "reason": (
            None
            if selector_verified
            else "all treatments must be eligible with a verified champion snapshot"
        ),
        "selection": (
            "highest_ranked_chronological_weighted_champion_frontier"
            if ranking_objective == "weighted_aggregate"
            else (
                "highest_ranked_chronological_ring10_only_champion_frontier"
                if ranking_objective == "ring_10_only"
                else "highest_ranked_chronological_champion_frontier"
            )
        ),
        "non_promoted_endpoints_are_diagnostic_only": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": status,
        "ranking_metric": ranking_metric,
        "ranking_objective": ranking_objective,
        "confidence": (
            {
                "level": CONFIDENCE_LEVEL,
                "sidedness": "one-sided-lower",
                "method": (
                    "normal lower bound using the conservative sum of frontier and "
                    "common-anchor Bradley-Terry standard_error"
                ),
                "normal_quantile": ONE_SIDED_95_NORMAL_QUANTILE,
            }
            if ranking_objective != "weighted_aggregate"
            else {
                "level": None,
                "component_level": CONFIDENCE_LEVEL,
                "sidedness": "one-sided-lower",
                "method": (
                    "sum of pre-registered chronological promoted-link anytime "
                    "weighted Elo lower bounds"
                ),
                "normal_quantile": None,
                "note": (
                    "component confidence sequences are anytime-valid; the report "
                    "does not claim a fixed simultaneous level across multiple "
                    "promoted links"
                ),
            }
        ),
        "compute_accounting": {
            "provisioned_gpus": provisioned_gpus,
            "basis": (
                "all provisioned GPUs from measurement start through resource release"
            ),
        },
        "guardrail_configuration": {
            "rings": list(parsed_rings),
            "floor_elo": float(guard_floor_elo),
        },
        "common_anchor": {
            "status": "available" if common_available else "unavailable",
            "identity": common_identity,
            "by_treatment": [
                {"label": label, "identity": identity}
                for label, identity in sorted(anchors.items())
            ],
        },
        "run_count": len(treatments),
        "eligible_count": len(eligible_treatments),
        "selector": selector,
        "errors": errors,
        "treatments": [treatment.payload for treatment in ordered],
    }


def _parse_run_arguments(arguments: Sequence[str]) -> dict[str, Path]:
    if len(arguments) < 2:
        raise ValueError("at least two --run LABEL=PATH entries are required")
    runs = {}
    for argument in arguments:
        if "=" not in argument:
            raise ValueError(
                f"invalid --run {argument!r}; expected a value in LABEL=PATH form"
            )
        label, raw_path = argument.split("=", 1)
        if not label or not raw_path:
            raise ValueError(
                f"invalid --run {argument!r}; label and path must both be non-empty"
            )
        if label in runs:
            raise ValueError(f"duplicate --run label: {label!r}")
        runs[label] = Path(raw_path)
    return runs


def _error_document(error: Exception) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        runs = _parse_run_arguments(arguments.run)
        if arguments.no_guard_rings and arguments.guard_ring is not None:
            raise ValueError("--no-guard-rings cannot be combined with --guard-ring")
        guard_rings = (
            ()
            if arguments.no_guard_rings
            else (
                tuple(arguments.guard_ring)
                if arguments.guard_ring is not None
                else DEFAULT_GUARD_RINGS
            )
        )
        report = build_elo_ablation_comparison(
            runs,
            provisioned_gpus=arguments.provisioned_gpus,
            guard_rings=guard_rings,
            guard_floor_elo=arguments.guard_floor_elo,
        )
    except (OSError, TypeError, ValueError) as error:
        print(json.dumps(_error_document(error), sort_keys=True, allow_nan=False))
        return 2
    serialized = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if arguments.output is not None:
        try:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(serialized, encoding="utf-8")
        except OSError as error:
            print(json.dumps(_error_document(error), sort_keys=True, allow_nan=False))
            return 2
    print(serialized, end="")
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
