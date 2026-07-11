#!/usr/bin/env python3
"""Join learner, actor, and arena evidence into strength-per-GPU-hour metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-strength-efficiency"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--provisioned-gpus", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _read_jsonl(
    path: Path,
    *,
    failures: list[dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "error": f"JSONDecodeError: {error}",
                    }
                )
                continue
            if not isinstance(payload, dict):
                failures.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "error": "JSONL record is not an object",
                    }
                )
                continue
            records.append(payload)
    return records


def _sum(records: Iterable[Mapping[str, object]], name: str) -> float:
    return sum(
        value for record in records if (value := _number(record.get(name))) is not None
    )


def _timestamp(record: Mapping[str, object]) -> int | None:
    for name in ("timestamp_ns", "completed_ns", "updated_ns"):
        value = record.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _learner_summary(records: list[dict[str, object]]) -> dict[str, object]:
    timed = [record for record in records if _number(record.get("step_seconds"))]
    interval_steps = 0
    examples = 0.0
    wall_seconds = 0.0
    device_seconds = 0.0
    data_wait_seconds = 0.0
    h2d_seconds = 0.0
    window_setup_seconds = 0.0
    legacy_records = 0
    previous_step: int | None = None
    for record in timed:
        current_step_value = _number(record.get("step"))
        current_step = (
            int(current_step_value) if current_step_value is not None else None
        )
        explicit_steps = _number(record.get("metrics_interval_steps"))
        if explicit_steps is not None and explicit_steps > 0:
            steps = int(explicit_steps)
        elif (
            current_step is not None
            and previous_step is not None
            and current_step > previous_step
        ):
            steps = current_step - previous_step
            legacy_records += 1
        else:
            steps = 1
            legacy_records += 1
        if current_step is not None:
            previous_step = current_step
        interval_steps += steps

        step_seconds = _number(record.get("step_seconds")) or 0.0
        interval_wall = _number(record.get("metrics_interval_wall_seconds"))
        wall_seconds += (
            interval_wall if interval_wall is not None else step_seconds * steps
        )

        batch = _number(record.get("global_batch_size"))
        if batch is None:
            examples_per_second = _number(record.get("examples_per_second"))
            if examples_per_second is not None and step_seconds:
                batch = examples_per_second * step_seconds
        if batch is not None:
            examples += batch * steps

        for name, destination in (
            ("device_step_seconds", "device"),
            ("data_wait_seconds", "data_wait"),
            ("h2d_seconds", "h2d"),
            ("window_setup_seconds", "window_setup"),
        ):
            value = _number(record.get(name))
            if value is None:
                continue
            weighted = value * steps
            if destination == "device":
                device_seconds += weighted
            elif destination == "data_wait":
                data_wait_seconds += weighted
            elif destination == "h2d":
                h2d_seconds += weighted
            else:
                window_setup_seconds += weighted
    step_values = [
        int(value)
        for record in records
        if (value := _number(record.get("step"))) is not None
    ]
    return {
        "records": len(records),
        "first_step": min(step_values, default=None),
        "last_step": max(step_values, default=None),
        "measured_steps": interval_steps,
        "measured_examples": int(round(examples)),
        "legacy_metric_records": legacy_records,
        "measured_wall_seconds": wall_seconds or None,
        "end_to_end_examples_per_second": (
            examples / wall_seconds if examples and wall_seconds else None
        ),
        "device_seconds": device_seconds or None,
        "device_duty_fraction": (
            device_seconds / wall_seconds if device_seconds and wall_seconds else None
        ),
        "data_wait_seconds": data_wait_seconds or None,
        "data_wait_fraction": (
            data_wait_seconds / wall_seconds
            if data_wait_seconds and wall_seconds
            else None
        ),
        "h2d_seconds": h2d_seconds or None,
        "window_setup_seconds": window_setup_seconds or None,
    }


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _actor_gpu_id(record: Mapping[str, object]) -> int | None:
    configured = _number(record.get("gpu_id"))
    if configured is not None and configured >= 0 and configured.is_integer():
        return int(configured)
    worker = record.get("worker")
    if not isinstance(worker, str):
        return None
    parts = worker.split("-")
    try:
        index = parts.index("gpu") + 1
        return int(parts[index])
    except (ValueError, IndexError):
        return None


def _actor_interval(record: Mapping[str, object]) -> tuple[int, int] | None:
    completed = record.get("batch_completed_ns", record.get("timestamp_ns"))
    started = record.get("batch_started_ns")
    if (
        isinstance(completed, int)
        and not isinstance(completed, bool)
        and isinstance(started, int)
        and not isinstance(started, bool)
        and completed >= started
    ):
        return started, completed
    elapsed = _number(record.get("elapsed_seconds"))
    if (
        isinstance(completed, int)
        and not isinstance(completed, bool)
        and elapsed is not None
        and elapsed >= 0
    ):
        return completed - int(elapsed * 1_000_000_000), completed
    return None


def _actor_summary(records: list[dict[str, object]]) -> dict[str, object]:
    lane_seconds = _sum(records, "elapsed_seconds")
    workers = sorted(
        {
            str(record["worker"])
            for record in records
            if isinstance(record.get("worker"), str)
        }
    )
    intervals_by_gpu: dict[int, list[tuple[int, int]]] = defaultdict(list)
    all_intervals = []
    for record in records:
        gpu_id = _actor_gpu_id(record)
        interval = _actor_interval(record)
        if gpu_id is not None and interval is not None:
            intervals_by_gpu[gpu_id].append(interval)
            all_intervals.append(interval)
    gpu_seconds = (
        sum(_merge_intervals(intervals) for intervals in intervals_by_gpu.values())
        / 1_000_000_000
    )
    fleet_wall_seconds = _merge_intervals(all_intervals) / 1_000_000_000
    rate_seconds = fleet_wall_seconds or lane_seconds
    policy_targets = _sum(records, "policy_samples")
    policy_weight = _sum(records, "policy_weight_sum")
    policy_weight_count = 0.0
    for record in records:
        count = _number(record.get("policy_weight_count"))
        policy_weight_count += (
            count
            if count is not None
            else (_number(record.get("policy_samples")) or 0.0)
        )
    return {
        "records": len(records),
        "workers": workers,
        "worker_count": len(workers),
        "gpu_ids": sorted(intervals_by_gpu),
        "actor_lane_seconds": lane_seconds,
        "actor_gpu_seconds": gpu_seconds or None,
        "fleet_wall_seconds": fleet_wall_seconds or None,
        "games": int(_sum(records, "games")),
        "samples": int(_sum(records, "samples")),
        "search_simulations": int(_sum(records, "search_simulations")),
        "evaluator_rows": int(_sum(records, "evaluator_rows")),
        "aggregate_games_per_second": (
            _sum(records, "games") / rate_seconds if rate_seconds else None
        ),
        "aggregate_samples_per_second": (
            _sum(records, "samples") / rate_seconds if rate_seconds else None
        ),
        "aggregate_evaluator_rows_per_second": (
            _sum(records, "evaluator_rows") / rate_seconds if rate_seconds else None
        ),
        "samples_per_physical_gpu_second": (
            _sum(records, "samples") / gpu_seconds if gpu_seconds else None
        ),
        "policy_target_count": int(policy_targets),
        "policy_weight_count": int(policy_weight_count),
        "effective_policy_weight": policy_weight or None,
        "mean_policy_weight": (
            policy_weight / policy_weight_count
            if policy_weight and policy_weight_count
            else None
        ),
    }


def _arena_results(
    root: Path,
    *,
    failures: list[dict[str, object]],
) -> list[dict[str, object]]:
    results = []
    arena = root / "arena"
    if not arena.is_dir():
        return results
    for path in sorted(arena.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(
                {"path": str(path), "error": f"{type(error).__name__}: {error}"}
            )
            continue
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("aggregate"), dict)
            and isinstance(payload.get("evaluation_metrics"), dict)
        ):
            results.append({**payload, "_path": str(path)})
    return results


def _arena_summary(
    results: list[dict[str, object]],
    *,
    run_started_ns: int,
    provisioned_gpus: int,
) -> dict[str, object]:
    by_baseline: dict[str, list[dict[str, object]]] = defaultdict(list)
    serialized = []
    for result in results:
        aggregate = result["aggregate"]
        assert isinstance(aggregate, dict)
        baseline = str(result.get("baseline") or "unknown")
        completed_ns = _timestamp(result)
        elo = _number(aggregate.get("elo_difference"))
        interval = aggregate.get("anytime_elo_interval")
        lower = (
            _number(interval[0])
            if isinstance(interval, list) and len(interval) == 2
            else None
        )
        gpu_hours = (
            provisioned_gpus * (completed_ns - run_started_ns) / 3_600_000_000_000
            if completed_ns is not None and completed_ns >= run_started_ns
            else None
        )
        item = {
            "path": result["_path"],
            "candidate": result.get("candidate"),
            "baseline": baseline,
            "baseline_metadata": result.get("baseline_metadata"),
            "completed_ns": completed_ns,
            "elo_difference": elo,
            "anytime_elo_lower": lower,
            "provisioned_gpu_hours": gpu_hours,
            "elo_per_gpu_hour": elo / gpu_hours
            if elo is not None and gpu_hours
            else None,
            "elo_lcb_per_gpu_hour": (
                lower / gpu_hours if lower is not None and gpu_hours else None
            ),
        }
        serialized.append(item)
        by_baseline[baseline].append(item)

    trends = {}
    for baseline, items in sorted(by_baseline.items()):
        ordered = sorted(
            items,
            key=lambda item: _number(item.get("completed_ns")) or 0,
        )
        first, latest = ordered[0], ordered[-1]
        elapsed_gpu_hours = None
        delta_elo = None
        if len(ordered) > 1:
            first_hours = _number(first["provisioned_gpu_hours"])
            latest_hours = _number(latest["provisioned_gpu_hours"])
            first_elo = _number(first["elo_difference"])
            latest_elo = _number(latest["elo_difference"])
            if first_hours is not None and latest_hours is not None:
                elapsed_gpu_hours = latest_hours - first_hours
            if first_elo is not None and latest_elo is not None:
                delta_elo = latest_elo - first_elo
        trends[baseline] = {
            "evaluations": len(ordered),
            "first": first,
            "latest": latest,
            "delta_elo": delta_elo,
            "elapsed_gpu_hours": elapsed_gpu_hours,
            "delta_elo_per_gpu_hour": (
                delta_elo / elapsed_gpu_hours
                if delta_elo is not None and elapsed_gpu_hours
                else None
            ),
        }
    return {
        "results": sorted(
            serialized,
            key=lambda item: _number(item.get("completed_ns")) or 0,
        ),
        "by_baseline": trends,
    }


def build_strength_efficiency_report(
    run_root: str | Path,
    *,
    provisioned_gpus: int = 8,
) -> dict[str, object]:
    root = Path(run_root).expanduser().resolve()
    if provisioned_gpus <= 0:
        raise ValueError("provisioned_gpus must be positive")
    run_path = root / "run.json"
    if not run_path.is_file():
        raise ValueError(f"run identity does not exist: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(run, dict):
        raise ValueError("run identity must be a JSON object")
    started_ns = run.get("created_ns")
    if (
        isinstance(started_ns, bool)
        or not isinstance(started_ns, int)
        or started_ns <= 0
    ):
        raise ValueError("run identity has an invalid created_ns")

    failures: list[dict[str, object]] = []
    learner_records = _read_jsonl(
        root / "learner" / "metrics.jsonl",
        failures=failures,
    )
    actor_records: list[dict[str, object]] = []
    metrics_directory = root / "metrics"
    if metrics_directory.is_dir():
        for path in sorted(metrics_directory.glob("actor-*.jsonl")):
            actor_records.extend(_read_jsonl(path, failures=failures))
    arenas = _arena_results(root, failures=failures)
    observed_timestamps = [
        timestamp
        for record in [*learner_records, *actor_records, *arenas]
        if (timestamp := _timestamp(record)) is not None
    ]
    observed_until_ns = max(observed_timestamps, default=started_ns)
    wall_seconds = max(0.0, (observed_until_ns - started_ns) / 1_000_000_000)
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": "complete" if not failures else "incomplete",
        "run_root": str(root),
        "run_id": run.get("run_id"),
        "generation_family": run.get("generation_family"),
        "started_ns": started_ns,
        "observed_until_ns": observed_until_ns,
        "wall_seconds": wall_seconds,
        "provisioned_gpus": provisioned_gpus,
        "provisioned_gpu_hours": provisioned_gpus * wall_seconds / 3_600.0,
        "learner": _learner_summary(learner_records),
        "actors": _actor_summary(actor_records),
        "arena": _arena_summary(
            arenas,
            run_started_ns=started_ns,
            provisioned_gpus=provisioned_gpus,
        ),
        "parse_failure_count": len(failures),
        "parse_failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = build_strength_efficiency_report(
            arguments.run_root,
            provisioned_gpus=arguments.provisioned_gpus,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "report": REPORT_NAME,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            )
        )
        return 2
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
