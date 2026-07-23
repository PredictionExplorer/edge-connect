#!/usr/bin/env python3
"""Rank training treatments by guarded ring-10 Elo gained per wall hour."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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


def _measurement_context(
    root: Path,
    report: Mapping[str, object],
    *,
    failures: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object], str | None]:
    metadata_path = root / "ablation.json"
    if not metadata_path.is_file():
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
                "status": "complete",
                "started_ns": started_ns,
                "stopped_ns": stopped_ns,
                "wall_seconds": wall_seconds,
                "stop_reason": "last_observed_run_timestamp",
                "exit_code": None,
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
                "status": "invalid",
                "started_ns": None,
                "stopped_ns": None,
                "wall_seconds": None,
                "stop_reason": None,
                "exit_code": None,
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

    raw_started = metadata.get("measurement_started_ns")
    raw_stopped = metadata.get("measurement_stopped_ns")
    started_ns = _positive_timestamp(raw_started)
    stopped_ns = _positive_timestamp(raw_stopped)
    if raw_started is not None and started_ns is None:
        failures.append(
            _failure(metadata_path, "measurement_started_ns must be positive")
        )
    if raw_stopped is not None and stopped_ns is None:
        failures.append(
            _failure(metadata_path, "measurement_stopped_ns must be positive")
        )
    stop_reason = metadata.get("measurement_stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        failures.append(
            _failure(metadata_path, "measurement_stop_reason must be a string")
        )
        stop_reason = None
    raw_exit_code = metadata.get("measurement_exit_code")
    exit_code = _integer(raw_exit_code)
    if raw_exit_code is not None and exit_code is None:
        failures.append(
            _failure(metadata_path, "measurement_exit_code must be an integer")
        )
    ordered = (
        started_ns is not None and stopped_ns is not None and stopped_ns >= started_ns
    )
    if started_ns is not None and stopped_ns is not None and stopped_ns < started_ns:
        failures.append(
            _failure(
                metadata_path,
                "measurement_stopped_ns predates measurement_started_ns",
            )
        )
    complete = (
        ordered
        and stop_reason in {"wall_budget", "leaf_budget"}
        and exit_code in {0, -15}
    )
    if ordered:
        assert started_ns is not None
        assert stopped_ns is not None
        wall_seconds = (stopped_ns - started_ns) / 1_000_000_000
    else:
        wall_seconds = None
    if complete:
        incomplete_reason = None
    elif started_ns is None:
        incomplete_reason = "ablation measurement has not started"
    elif stopped_ns is None:
        incomplete_reason = "ablation measurement has not stopped"
    else:
        incomplete_reason = (
            "ablation measurement did not end at its wall/leaf budget "
            f"with a successful exit (reason={stop_reason!r}, exit={exit_code!r})"
        )
    return (
        {
            "source": "ablation.json",
            "status": "complete" if complete else "incomplete",
            "started_ns": started_ns,
            "stopped_ns": stopped_ns,
            "wall_seconds": wall_seconds,
            "stop_reason": stop_reason,
            "exit_code": exit_code,
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
        "anchor": {
            "identity": None,
            "step": None,
            "rating_elo": None,
            "standard_error_elo": None,
            "selection": None,
        },
        "endpoint": None,
        "measurement": {
            "source": None,
            "status": "unavailable",
            "started_ns": None,
            "stopped_ns": None,
            "wall_seconds": None,
            "stop_reason": None,
            "exit_code": None,
            "wall_budget_seconds": None,
            "leaf_budget": None,
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
    if endpoint is None:
        _add_reason(
            reasons,
            "missing_ring_10_endpoint",
            endpoint_error or "ring-10 terminal endpoint is unavailable",
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
    point_score = (
        elo_gained / wall_hours
        if elo_gained is not None and wall_hours is not None
        else None
    )
    ranking_score = (
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
        "ring_10_elo_per_wall_hour": point_score,
        "ring_10_elo_lcb_per_wall_hour": ranking_score,
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
    if ranking_score is None:
        _add_reason(
            reasons,
            "unavailable_ranking_metric",
            "ring-10 Elo/hour lower bound could not be computed",
        )
    return _Treatment(
        label,
        payload,
        anchor_identity if anchor_estimate is not None else None,
        ranking_score,
        point_score,
        reasons,
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
    if not parsed_rings or any(
        type(ring) is not int or ring <= 0 for ring in parsed_rings
    ):
        raise ValueError("guard_rings must contain positive integers")
    if len(set(parsed_rings)) != len(parsed_rings):
        raise ValueError("guard_rings must not contain duplicates")
    normalized_runs = _validate_runs(runs)
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

    for treatment in treatments:
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
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": status,
        "ranking_metric": "ring_10_elo_lcb_per_wall_hour",
        "confidence": {
            "level": CONFIDENCE_LEVEL,
            "sidedness": "one-sided-lower",
            "method": (
                "normal lower bound using the conservative sum of endpoint and "
                "common-anchor Bradley-Terry standard_error"
            ),
            "normal_quantile": ONE_SIDED_95_NORMAL_QUANTILE,
        },
        "compute_accounting": {
            "provisioned_gpus": provisioned_gpus,
            "basis": "all provisioned GPUs over each full observed wall interval",
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
        guard_rings = (
            tuple(arguments.guard_ring)
            if arguments.guard_ring is not None
            else DEFAULT_GUARD_RINGS
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
