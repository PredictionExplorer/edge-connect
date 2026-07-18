#!/usr/bin/env python3
"""Join learner, actor, and arena evidence into strength-per-GPU-hour metrics."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from startrain.autonomous_elo import DecisiveMatch, fit_bradley_terry_elo

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-strength-efficiency"
AUTONOMOUS_ELO_SCHEMA_VERSION = 1
AUTONOMOUS_ELO_CONFIDENCE = 0.95
PRIMARY_ELO_RING = 10
MIGRATION_SCHEMA_VERSION = 1
ARENA_RESULT_KINDS = ("promotion", "crossplay", "historical_crossplay")


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


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


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


def _latest_value(
    records: Iterable[Mapping[str, object]],
    name: str,
) -> object | None:
    materialized = records if isinstance(records, list) else list(records)
    for record in reversed(materialized):
        if name in record:
            return record[name]
    return None


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
    latest_fields = {
        name: _latest_value(records, name)
        for name in (
            "updates_per_new_sample",
            "lifetime_updates_per_new_sample",
            "segment_updates_per_new_sample",
            "utd_segment_target_updates_per_new_sample",
            "utd_segment_baseline_examples_consumed",
            "utd_segment_baseline_committed_replay_samples",
            "loader_workers_effective",
            "window_setup_seconds",
            "window_setup_amortized_seconds",
            "window_batches_allocated",
            "window_batches_consumed",
            "window_batches_consumed_this_spin",
            "window_batches_remaining",
            "window_reuse",
            "window_reuse_spins",
            "window_refresh_reason",
            "utd_wait_spins",
        )
    }
    allocation_records = [
        record for record in records if record.get("event") == "replay_window_allocated"
    ]
    consumption_records = [
        record for record in records if record.get("event") == "replay_window_consumed"
    ]
    refresh_records = [
        record for record in records if record.get("event") == "replay_window_refreshed"
    ]
    has_persistent_window_metrics = any(
        record.get("event")
        in (
            "replay_window_allocated",
            "replay_window_consumed",
            "replay_window_refreshed",
        )
        or any(
            name in record
            for name in (
                "loader_workers_effective",
                "window_setup_amortized_seconds",
                "window_reuse",
                "window_reuse_spins",
            )
        )
        for record in records
    )
    reused_consumption_records = sum(
        record.get("window_reuse") is True for record in consumption_records
    )
    persistent_window = (
        {
            "allocation_records": len(allocation_records),
            "consumption_records": len(consumption_records),
            "refresh_records": len(refresh_records),
            "reused_consumption_records": reused_consumption_records,
            "reuse_fraction": (
                reused_consumption_records / len(consumption_records)
                if consumption_records
                else None
            ),
            "consumed_batches": int(
                _sum(consumption_records, "window_batches_consumed_this_spin")
            ),
            "setup_seconds": _sum(allocation_records, "window_setup_seconds") or None,
            "latest": {
                name: latest_fields[name]
                for name in (
                    "loader_workers_effective",
                    "window_setup_seconds",
                    "window_setup_amortized_seconds",
                    "window_batches_allocated",
                    "window_batches_consumed",
                    "window_batches_consumed_this_spin",
                    "window_batches_remaining",
                    "window_reuse",
                    "window_reuse_spins",
                    "window_refresh_reason",
                    "utd_wait_spins",
                )
            },
        }
        if has_persistent_window_metrics
        else None
    )
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
        "updates_per_new_sample": latest_fields["updates_per_new_sample"],
        "lifetime_updates_per_new_sample": latest_fields[
            "lifetime_updates_per_new_sample"
        ],
        "segment_updates_per_new_sample": latest_fields[
            "segment_updates_per_new_sample"
        ],
        "utd_segment_target_updates_per_new_sample": latest_fields[
            "utd_segment_target_updates_per_new_sample"
        ],
        "utd_segment_baseline_examples_consumed": latest_fields[
            "utd_segment_baseline_examples_consumed"
        ],
        "utd_segment_baseline_committed_replay_samples": latest_fields[
            "utd_segment_baseline_committed_replay_samples"
        ],
        "loader_workers_effective": latest_fields["loader_workers_effective"],
        "latest_window_setup_seconds": latest_fields["window_setup_seconds"],
        "window_setup_amortized_seconds": latest_fields[
            "window_setup_amortized_seconds"
        ],
        "window_batches_allocated": latest_fields["window_batches_allocated"],
        "window_batches_consumed": latest_fields["window_batches_consumed"],
        "window_batches_consumed_this_spin": latest_fields[
            "window_batches_consumed_this_spin"
        ],
        "window_batches_remaining": latest_fields["window_batches_remaining"],
        "window_reuse": latest_fields["window_reuse"],
        "window_reuse_spins": latest_fields["window_reuse_spins"],
        "window_refresh_reason": latest_fields["window_refresh_reason"],
        "utd_wait_spins": latest_fields["utd_wait_spins"],
        "persistent_window": persistent_window,
    }


def _sha256(value: object) -> str | None:
    return (
        value
        if isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        else None
    )


def _migration_field(
    record: Mapping[str, object],
    nested_name: str,
    *names: str,
) -> object | None:
    nested = _mapping(record.get(nested_name))
    for name in names:
        if name in record:
            return record[name]
    for name in names:
        if name in nested:
            return nested[name]
    return None


def _migration_boundary(
    record: Mapping[str, object],
    *,
    line_number: int,
) -> dict[str, object] | None:
    raw_boundary = record.get(
        "boundary",
        record.get("learner_boundary", record.get("utd_boundary")),
    )
    if raw_boundary is not None and not isinstance(raw_boundary, Mapping):
        raise ValueError(
            f"autonomous migration line {line_number} boundary must be an object"
        )
    utd_segment = _mapping(record.get("utd_segment"))
    sources = [
        _mapping(raw_boundary),
        record,
        utd_segment,
    ]
    aliases = {
        "step": ("step", "learner_step"),
        "examples_consumed": (
            "examples_consumed",
            "learner_examples_consumed",
            "baseline_examples_consumed",
        ),
        "committed_replay_samples": (
            "committed_replay_samples",
            "total_replay_samples",
            "replay_samples",
            "baseline_committed_replay_samples",
        ),
    }
    values: dict[str, object | None] = {}
    for normalized, names in aliases.items():
        values[normalized] = next(
            (source[name] for source in sources for name in names if name in source),
            None,
        )
    if all(value is None for value in values.values()):
        return None
    invalid = [
        name
        for name in ("step", "examples_consumed")
        if type(values[name]) is not int or cast(int, values[name]) < 0
    ]
    replay_samples = values["committed_replay_samples"]
    if replay_samples is not None and (
        type(replay_samples) is not int or replay_samples < 0
    ):
        invalid.append("committed_replay_samples")
    if invalid:
        raise ValueError(
            "autonomous migration line "
            f"{line_number} has an incomplete or invalid boundary: "
            + ", ".join(invalid)
        )
    target = next(
        (
            source[name]
            for source in sources
            for name in (
                "target_updates_per_new_sample",
                "utd_target_updates_per_new_sample",
            )
            if name in source
        ),
        None,
    )
    parsed_target = _number(target)
    if target is not None and (parsed_target is None or parsed_target <= 0):
        raise ValueError(
            f"autonomous migration line {line_number} has an invalid UTD target"
        )
    return {
        "step": values["step"],
        "examples_consumed": values["examples_consumed"],
        "committed_replay_samples": values["committed_replay_samples"],
        "target_updates_per_new_sample": parsed_target,
    }


def _migration_summary(
    root: Path,
    *,
    run: Mapping[str, object],
    run_started_ns: int,
    observed_until_ns: int,
    provisioned_gpus: int,
) -> dict[str, object]:
    path = root / "autonomous-migrations.jsonl"
    if not path.is_file():
        return {
            "schema_version": MIGRATION_SCHEMA_VERSION,
            "record_count": 0,
            "boundary_count": 0,
            "ignored_record_count": 0,
            "records": [],
            "boundaries": [],
            "segments": [
                {
                    "index": 0,
                    "started_ns": run_started_ns,
                    "ended_ns": observed_until_ns,
                    "migration_id": None,
                    "config_sha256": None,
                    "start_boundary": None,
                    "wall_seconds": max(
                        0.0,
                        (observed_until_ns - run_started_ns) / 1_000_000_000,
                    ),
                    "provisioned_gpu_hours": max(
                        0.0,
                        provisioned_gpus
                        * (observed_until_ns - run_started_ns)
                        / 3_600_000_000_000,
                    ),
                }
            ],
        }

    active_run_id = run.get("run_id")
    active_family = run.get("generation_family")
    records: list[dict[str, object]] = []
    ignored = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "malformed autonomous migration record "
                    f"at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(
                    "malformed autonomous migration record "
                    f"at {path}:{line_number}: record is not an object"
                )
            record_run_id = payload.get("run_id")
            if record_run_id is not None and record_run_id != active_run_id:
                ignored += 1
                continue
            record_family = payload.get("generation_family")
            if record_family is not None and record_family != active_family:
                raise ValueError(
                    f"autonomous migration line {line_number} generation "
                    "family does not match the active run"
                )
            schema_version = payload.get("schema_version", MIGRATION_SCHEMA_VERSION)
            if (
                type(schema_version) is not int
                or schema_version != MIGRATION_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"autonomous migration line {line_number} has unsupported "
                    f"schema_version={schema_version!r}"
                )
            timestamp_ns = next(
                (
                    payload[name]
                    for name in ("timestamp_ns", "created_ns", "applied_ns")
                    if name in payload
                ),
                None,
            )
            if type(timestamp_ns) is not int or timestamp_ns <= 0:
                raise ValueError(
                    f"autonomous migration line {line_number} has an invalid timestamp"
                )
            from_sha256 = _migration_field(
                payload,
                "source",
                "from_sha256",
                "from_config_sha256",
                "source_config_sha256",
                "previous_config_sha256",
                "old_config_sha256",
                "config_sha256",
            )
            to_sha256 = _migration_field(
                payload,
                "target",
                "to_sha256",
                "to_config_sha256",
                "target_config_sha256",
                "new_config_sha256",
                "config_sha256",
            )
            parsed_from = _sha256(from_sha256)
            parsed_to = _sha256(to_sha256)
            if parsed_from is None or parsed_to is None:
                raise ValueError(
                    f"autonomous migration line {line_number} has invalid "
                    "source/target config hashes"
                )
            boundary = _migration_boundary(payload, line_number=line_number)
            migration_id = payload.get(
                "migration_id",
                f"migration-{timestamp_ns}",
            )
            if not isinstance(migration_id, str) or not migration_id.strip():
                raise ValueError(
                    f"autonomous migration line {line_number} has an invalid "
                    "migration_id"
                )
            reason = payload.get("reason")
            if reason is not None and (
                not isinstance(reason, str) or not reason.strip() or "\n" in reason
            ):
                raise ValueError(
                    f"autonomous migration line {line_number} has an invalid reason"
                )
            records.append(
                {
                    **payload,
                    "schema_version": MIGRATION_SCHEMA_VERSION,
                    "timestamp_ns": timestamp_ns,
                    "from_sha256": parsed_from,
                    "to_sha256": parsed_to,
                    "migration_id": migration_id,
                    "boundary": boundary,
                    "_line": line_number,
                }
            )

    previous: Mapping[str, object] | None = None
    for record in records:
        line_number = record["_line"]
        timestamp_ns = cast(int, record["timestamp_ns"])
        if timestamp_ns < run_started_ns:
            raise ValueError(
                f"autonomous migration line {line_number} predates the active run"
            )
        if previous is not None:
            if timestamp_ns <= cast(int, previous["timestamp_ns"]):
                raise ValueError(
                    "autonomous migration timestamps must be strictly increasing"
                )
            if record["from_sha256"] != previous["to_sha256"]:
                raise ValueError(
                    f"autonomous migration line {line_number} breaks the config hash chain"
                )
            for current_name, previous_name, description in (
                ("from_profile", "to_profile", "profile"),
                ("from_source_commit", "to_source_commit", "source commit"),
            ):
                current_value = record.get(current_name)
                previous_value = previous.get(previous_name)
                if (
                    current_value is not None
                    and previous_value is not None
                    and current_value != previous_value
                ):
                    raise ValueError(
                        f"autonomous migration line {line_number} breaks the "
                        f"{description} chain"
                    )
            boundary = record.get("boundary")
            previous_boundary = previous.get("boundary")
            if isinstance(boundary, Mapping) and isinstance(previous_boundary, Mapping):
                for name in (
                    "step",
                    "examples_consumed",
                    "committed_replay_samples",
                ):
                    value = boundary.get(name)
                    previous_value = previous_boundary.get(name)
                    if (
                        type(value) is int
                        and type(previous_value) is int
                        and value < previous_value
                    ):
                        raise ValueError(
                            f"autonomous migration line {line_number} moves "
                            f"{name} backwards"
                        )
        previous = record

    provenance_path = root / "autonomous-provenance.json"
    if records and provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"cannot validate autonomous migration head against {provenance_path}: "
                f"{error}"
            ) from error
        provenance_hash = (
            _sha256(provenance.get("config_sha256"))
            if isinstance(provenance, Mapping)
            else None
        )
        if provenance_hash is None or provenance_hash != records[-1]["to_sha256"]:
            raise ValueError(
                "autonomous migration chain does not end at provenance config_sha256"
            )

    boundaries = []
    for record in records:
        boundary = record.get("boundary")
        if not isinstance(boundary, Mapping):
            continue
        boundaries.append(
            {
                "migration_id": record["migration_id"],
                "timestamp_ns": record["timestamp_ns"],
                "from_sha256": record["from_sha256"],
                "to_sha256": record["to_sha256"],
                **boundary,
            }
        )
    segment_observed_until_ns = max(
        [
            observed_until_ns,
            *(cast(int, record["timestamp_ns"]) for record in records),
        ]
    )
    segments = []
    for index in range(len(records) + 1):
        prior = records[index - 1] if index else None
        following = records[index] if index < len(records) else None
        started_ns = (
            cast(int, prior["timestamp_ns"]) if prior is not None else run_started_ns
        )
        ended_ns = (
            cast(int, following["timestamp_ns"])
            if following is not None
            else segment_observed_until_ns
        )
        start_boundary = prior.get("boundary") if prior is not None else None
        start_boundary = start_boundary if isinstance(start_boundary, Mapping) else {}
        segments.append(
            {
                "index": index,
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "migration_id": (prior["migration_id"] if prior is not None else None),
                "config_sha256": (
                    prior["to_sha256"]
                    if prior is not None
                    else (records[0]["from_sha256"] if records else None)
                ),
                "start_boundary": (prior["boundary"] if prior is not None else None),
                "started_step": start_boundary.get("step"),
                "started_examples_consumed": start_boundary.get("examples_consumed"),
                "started_committed_replay_samples": start_boundary.get(
                    "committed_replay_samples"
                ),
                "target_updates_per_new_sample": start_boundary.get(
                    "target_updates_per_new_sample"
                ),
                "wall_seconds": max(
                    0.0,
                    (ended_ns - started_ns) / 1_000_000_000,
                ),
                "provisioned_gpu_hours": max(
                    0.0,
                    provisioned_gpus * (ended_ns - started_ns) / 3_600_000_000_000,
                ),
            }
        )
    serialized_records = [
        {key: value for key, value in record.items() if key != "_line"}
        for record in records
    ]
    return {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "record_count": len(records),
        "boundary_count": len(boundaries),
        "ignored_record_count": ignored,
        "records": serialized_records,
        "boundaries": boundaries,
        "segments": segments,
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


def _arena_result_kind(result: Mapping[str, object]) -> str:
    value = result.get("result_kind")
    if value is None:
        return "promotion"
    return value if value in ARENA_RESULT_KINDS else "unknown"


def _arena_result_category(result: Mapping[str, object]) -> str:
    return (
        "crossplay"
        if _arena_result_kind(result) in ("crossplay", "historical_crossplay")
        else _arena_result_kind(result)
    )


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
            and "candidate" in payload
            and "baseline" in payload
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
        aggregate = result.get("aggregate")
        if not isinstance(aggregate, dict):
            continue
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
            "result_kind": _arena_result_kind(result),
            "result_category": _arena_result_category(result),
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
    result_kind_counts = {
        kind: sum(item.get("result_kind") == kind for item in serialized)
        for kind in (*ARENA_RESULT_KINDS, "unknown")
    }
    result_category_counts = {
        kind: sum(item.get("result_category") == kind for item in serialized)
        for kind in ("promotion", "crossplay", "unknown")
    }
    return {
        "results": sorted(
            serialized,
            key=lambda item: _number(item.get("completed_ns")) or 0,
        ),
        "by_baseline": trends,
        "result_kind_counts": result_kind_counts,
        "result_category_counts": result_category_counts,
        "promotion_result_count": result_category_counts["promotion"],
        "crossplay_result_count": result_category_counts["crossplay"],
    }


def _nonnegative_integer(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def _checkpoint_identity(value: object) -> str | None:
    return (
        value
        if isinstance(value, str) and bool(value) and value.strip() == value
        else None
    )


def _content_addressed_identity(identity: str) -> bool:
    digest = identity.removeprefix("sha256-")
    return (
        identity.startswith("sha256-")
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _record_step_evidence(
    evidence: dict[str, set[int]],
    *,
    identity: object,
    step: object,
) -> None:
    parsed_identity = _checkpoint_identity(identity)
    parsed_step = _nonnegative_integer(step)
    if parsed_identity is not None and parsed_step is not None:
        evidence.setdefault(parsed_identity, set()).add(parsed_step)


def _checkpoint_step_evidence(
    root: Path,
    *,
    actor_records: list[dict[str, object]],
    arena_results: list[dict[str, object]],
) -> tuple[dict[str, int], dict[str, int], list[dict[str, object]]]:
    evidence: dict[str, set[int]] = {}
    publication_evidence: dict[str, set[int]] = {}
    metadata_failures: list[dict[str, object]] = []

    def record_publication(identity: object, published_ns: object) -> None:
        parsed_identity = _checkpoint_identity(identity)
        parsed_ns = _nonnegative_integer(published_ns)
        if parsed_identity is not None and parsed_ns is not None and parsed_ns > 0:
            publication_evidence.setdefault(parsed_identity, set()).add(parsed_ns)

    for record in actor_records:
        _record_step_evidence(
            evidence,
            identity=record.get("model_identity", record.get("model_version")),
            step=record.get("model_step"),
        )
    for result in arena_results:
        _record_step_evidence(
            evidence,
            identity=result.get("candidate"),
            step=result.get("candidate_step"),
        )
        _record_step_evidence(
            evidence,
            identity=result.get("baseline"),
            step=result.get("baseline_step", result.get("champion_step")),
        )
        for participant, metadata_name in (
            (result.get("candidate"), "candidate_metadata"),
            (result.get("baseline"), "baseline_metadata"),
        ):
            metadata = result.get(metadata_name)
            if isinstance(metadata, Mapping):
                _record_step_evidence(
                    evidence,
                    identity=metadata.get("identity", participant),
                    step=metadata.get("model_step", metadata.get("step")),
                )
    history_records = _read_jsonl(
        root / "learner" / "model-history.jsonl",
        failures=metadata_failures,
    )
    for record in history_records:
        _record_step_evidence(
            evidence,
            identity=record.get("model_identity"),
            step=record.get("model_step"),
        )
        record_publication(
            record.get("model_identity"),
            record.get("published_ns"),
        )

    metadata_paths = [
        *(root / "learner" / "manifests").glob("manifest-*.json"),
        *(root / "learner").glob("candidate.json"),
        *(root / "learner").glob("champion.json"),
        *(root / "arena").glob("promotion-status.json"),
    ]
    for path in sorted(set(metadata_paths)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            metadata_failures.append(
                {
                    "path": str(path),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if not isinstance(payload, dict):
            metadata_failures.append(
                {"path": str(path), "reason": "checkpoint metadata is not an object"}
            )
            continue
        _record_step_evidence(
            evidence,
            identity=payload.get(
                "model_identity",
                payload.get("candidate_identity"),
            ),
            step=payload.get("model_step", payload.get("candidate_step")),
        )
        record_publication(
            payload.get(
                "model_identity",
                payload.get("candidate_identity"),
            ),
            payload.get("published_ns"),
        )
        _record_step_evidence(
            evidence,
            identity=payload.get("champion_identity"),
            step=payload.get("champion_step"),
        )

    conflicts = [
        {
            "identity": identity,
            "steps": sorted(steps),
            "reason": "conflicting model_step evidence",
        }
        for identity, steps in sorted(evidence.items())
        if len(steps) > 1
    ]
    resolved = {
        identity: next(iter(steps))
        for identity, steps in evidence.items()
        if len(steps) == 1
    }
    publication_conflicts = [
        {
            "identity": identity,
            "published_ns": sorted(values),
            "reason": "conflicting checkpoint publication evidence",
        }
        for identity, values in sorted(publication_evidence.items())
        if len(values) > 1
    ]
    publications = {
        identity: next(iter(values))
        for identity, values in publication_evidence.items()
        if len(values) == 1
    }
    return (
        resolved,
        publications,
        [*metadata_failures, *conflicts, *publication_conflicts],
    )


def _baseline_kind(
    result: Mapping[str, object],
    *,
    checkpoint_steps: Mapping[str, int],
) -> tuple[str, str]:
    metadata = result.get("baseline_metadata")
    baseline = _checkpoint_identity(result.get("baseline"))
    if isinstance(metadata, Mapping):
        kind = metadata.get("kind")
        if isinstance(kind, str) and kind:
            metadata_identity = metadata.get("identity")
            if (
                kind == "checkpoint"
                and metadata_identity is not None
                and metadata_identity != baseline
            ):
                return "unknown", "checkpoint baseline metadata identity disagrees"
            return (
                ("checkpoint", "baseline_metadata")
                if kind == "checkpoint"
                else ("frozen", kind)
            )
        if "kind" in metadata:
            return "unknown", "baseline metadata kind is invalid"
    elif metadata is not None:
        return "unknown", "baseline metadata must be an object"
    candidate = _checkpoint_identity(result.get("candidate"))
    if (
        candidate is not None
        and baseline is not None
        and (
            (candidate in checkpoint_steps and baseline in checkpoint_steps)
            or (
                _content_addressed_identity(candidate)
                and _content_addressed_identity(baseline)
            )
        )
    ):
        return "checkpoint", "identity_evidence"
    return "unknown", "checkpoint baseline was not established"


def _decisive_match(
    result: Mapping[str, object],
    *,
    scope: str,
) -> tuple[DecisiveMatch | None, str | None]:
    candidate = _checkpoint_identity(result.get("candidate"))
    baseline = _checkpoint_identity(result.get("baseline"))
    if candidate is None or baseline is None:
        return None, "candidate and baseline identities must be non-empty strings"
    if scope == "aggregate":
        summary = result.get("aggregate")
    else:
        per_ring = result.get("per_ring")
        summary = (
            per_ring.get(str(PRIMARY_ELO_RING))
            if isinstance(per_ring, Mapping)
            else None
        )
    if not isinstance(summary, Mapping):
        return None, f"{scope} decisive summary is missing"
    wins = _nonnegative_integer(summary.get("wins"))
    losses = _nonnegative_integer(summary.get("losses"))
    if wins is None or losses is None:
        return None, f"{scope} wins/losses must be non-negative integers"
    games = summary.get("games")
    if games is not None and _nonnegative_integer(games) != wins + losses:
        return None, f"{scope} games disagrees with wins plus losses"
    try:
        return DecisiveMatch(candidate, baseline, wins, losses), None
    except ValueError as error:
        return None, str(error)


def _scope_inputs(
    arena_results: list[dict[str, object]],
    *,
    scope: str,
    checkpoint_steps: Mapping[str, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    included = []
    exclusions = []
    for result in arena_results:
        kind, kind_evidence = _baseline_kind(
            result,
            checkpoint_steps=checkpoint_steps,
        )
        if kind != "checkpoint":
            if kind == "unknown":
                exclusions.append(
                    {
                        "path": result.get("_path"),
                        "scope": scope,
                        "candidate": result.get("candidate"),
                        "baseline": result.get("baseline"),
                        "reason": kind_evidence,
                    }
                )
            continue
        match, reason = _decisive_match(result, scope=scope)
        if match is None:
            exclusions.append(
                {
                    "path": result.get("_path"),
                    "scope": scope,
                    "candidate": result.get("candidate"),
                    "baseline": result.get("baseline"),
                    "reason": reason,
                }
            )
            continue
        included.append(
            {
                "match": match,
                "path": result.get("_path"),
                "completed_ns": _timestamp(result),
                "classification": kind_evidence,
            }
        )
    return included, exclusions


def _select_anchor(
    scoped_inputs: list[dict[str, object]],
    *,
    checkpoint_steps: Mapping[str, int],
) -> tuple[str | None, str]:
    identities = {
        identity
        for item in scoped_inputs
        if isinstance((match := item.get("match")), DecisiveMatch)
        for identity in (match.candidate, match.baseline)
    }
    if not identities:
        return None, "unavailable"
    step_zero = sorted(
        identity for identity in identities if checkpoint_steps.get(identity) == 0
    )
    if step_zero:
        return step_zero[0], "step_zero_snapshot"
    stepped = sorted(
        (step, identity)
        for identity in identities
        if (step := checkpoint_steps.get(identity)) is not None
    )
    if stepped:
        return stepped[0][1], "earliest_step_snapshot"
    return min(identities), "lexical_fallback_step_unavailable"


def _graph_components(matches: list[DecisiveMatch]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {}
    for match in matches:
        adjacency.setdefault(match.candidate, set()).add(match.baseline)
        adjacency.setdefault(match.baseline, set()).add(match.candidate)
    remaining = set(adjacency)
    components = []
    while remaining:
        pending = [min(remaining)]
        component: set[str] = set()
        while pending:
            identity = pending.pop()
            if identity in component:
                continue
            component.add(identity)
            pending.extend(sorted(adjacency[identity] - component, reverse=True))
        remaining.difference_update(component)
        components.append(sorted(component))
    return sorted(components)


def _saturated_one_sided_pairings(
    inputs: list[dict[str, object]],
    *,
    included_identities: set[str] | None = None,
) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], dict[str, object]] = {}
    for item in inputs:
        match = item.get("match")
        if not isinstance(match, DecisiveMatch):
            continue
        first, second = sorted((match.candidate, match.baseline))
        if included_identities is not None and (
            first not in included_identities or second not in included_identities
        ):
            continue
        total = totals.setdefault(
            (first, second),
            {
                "first_identity": first,
                "second_identity": second,
                "first_wins": 0,
                "second_wins": 0,
                "result_count": 0,
                "result_paths": [],
            },
        )
        first_wins = match.wins if match.candidate == first else match.losses
        second_wins = match.losses if match.candidate == first else match.wins
        total["first_wins"] = cast(int, total["first_wins"]) + first_wins
        total["second_wins"] = cast(int, total["second_wins"]) + second_wins
        total["result_count"] = cast(int, total["result_count"]) + 1
        path = item.get("path")
        if isinstance(path, str):
            paths = cast(list[str], total["result_paths"])
            paths.append(path)
    saturated = []
    for total in totals.values():
        first_wins = cast(int, total["first_wins"])
        second_wins = cast(int, total["second_wins"])
        if first_wins and second_wins:
            continue
        saturated.append(
            {
                **total,
                "decisive_games": first_wins + second_wins,
                "result_paths": sorted(set(cast(list[str], total["result_paths"]))),
            }
        )
    return sorted(
        saturated,
        key=lambda item: (
            str(item["first_identity"]),
            str(item["second_identity"]),
        ),
    )


def _unavailable_ladder(
    *,
    scope: str,
    anchor_identity: str | None,
    inputs: list[dict[str, object]],
    exclusions: list[dict[str, object]],
    reason: str,
) -> dict[str, object]:
    matches = [
        match
        for item in inputs
        if isinstance((match := item.get("match")), DecisiveMatch)
    ]
    components = _graph_components(matches)
    identities = sorted(
        {identity for component in components for identity in component}
    )
    saturated_pairings = _saturated_one_sided_pairings(inputs)
    return {
        "status": "unavailable",
        "scope": scope,
        "ring": PRIMARY_ELO_RING if scope == "ring_10" else None,
        "reason": reason,
        "anchor_identity": anchor_identity,
        "ladder": [],
        "latest": None,
        "input": {
            "result_count": len(inputs),
            "identity_count": len(identities),
            "decisive_games": sum(match.wins + match.losses for match in matches),
            "saturated_one_sided_pairing_count": len(saturated_pairings),
            "saturated_one_sided_pairings": saturated_pairings,
        },
        "connectedness": {
            "connected": len(components) == 1 and bool(components),
            "component_count": len(components),
            "components": components,
            "excluded_identities": identities
            if anchor_identity is not None and anchor_identity not in identities
            else [],
        },
        "exclusions": exclusions,
    }


def _build_ladder(
    *,
    scope: str,
    inputs: list[dict[str, object]],
    exclusions: list[dict[str, object]],
    anchor_identity: str | None,
    checkpoint_steps: Mapping[str, int],
    checkpoint_publications: Mapping[str, int],
) -> dict[str, object]:
    if anchor_identity is None:
        return _unavailable_ladder(
            scope=scope,
            anchor_identity=None,
            inputs=inputs,
            exclusions=exclusions,
            reason="no valid checkpoint-vs-checkpoint decisive results",
        )
    matches = [
        match
        for item in inputs
        if isinstance((match := item.get("match")), DecisiveMatch)
    ]
    if not matches:
        return _unavailable_ladder(
            scope=scope,
            anchor_identity=anchor_identity,
            inputs=inputs,
            exclusions=exclusions,
            reason=f"no valid {scope} checkpoint-vs-checkpoint decisive results",
        )
    identities = {
        identity for match in matches for identity in (match.candidate, match.baseline)
    }
    if anchor_identity not in identities:
        return _unavailable_ladder(
            scope=scope,
            anchor_identity=anchor_identity,
            inputs=inputs,
            exclusions=exclusions,
            reason="the fixed anchor is absent from this comparison graph",
        )
    try:
        fit = fit_bradley_terry_elo(
            matches,
            anchor_identity=anchor_identity,
            confidence=AUTONOMOUS_ELO_CONFIDENCE,
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return _unavailable_ladder(
            scope=scope,
            anchor_identity=anchor_identity,
            inputs=inputs,
            exclusions=exclusions,
            reason=f"{type(error).__name__}: {error}",
        )

    last_seen: dict[str, int] = {}
    for order, item in enumerate(inputs):
        match = item["match"]
        assert isinstance(match, DecisiveMatch)
        completed_ns = item.get("completed_ns")
        observed = completed_ns if isinstance(completed_ns, int) else order
        for identity in (match.candidate, match.baseline):
            last_seen[identity] = max(last_seen.get(identity, 0), observed)
    ladder = [
        {
            "rank": rank,
            "identity": estimate.identity,
            "step": checkpoint_steps.get(estimate.identity),
            "rating": estimate.rating,
            "standard_error": estimate.standard_error,
            "confidence_interval": list(estimate.confidence_interval),
            "decisive_games": estimate.decisive_games,
            "last_observed_ns": last_seen.get(estimate.identity),
            "published_ns": checkpoint_publications.get(estimate.identity),
        }
        for rank, estimate in enumerate(fit.estimates, start=1)
    ]
    stepped = [item for item in ladder if item["step"] is not None]
    if stepped:
        latest = max(
            stepped,
            key=lambda item: (
                int(item["step"]),
                last_seen.get(str(item["identity"]), 0),
                str(item["identity"]),
            ),
        )
        latest_basis = "maximum_step"
    else:
        latest = max(
            ladder,
            key=lambda item: (
                last_seen.get(str(item["identity"]), 0),
                str(item["identity"]),
            ),
        )
        latest_basis = "last_observed_step_unavailable"
    latest = {**latest, "selection": latest_basis}
    disconnected_results = [
        item.get("path")
        for item in inputs
        if isinstance((match := item.get("match")), DecisiveMatch)
        and (
            match.candidate in fit.excluded_identities
            or match.baseline in fit.excluded_identities
        )
    ]
    fitted_identities = {estimate.identity for estimate in fit.estimates}
    saturated_pairings = _saturated_one_sided_pairings(
        inputs,
        included_identities=fitted_identities,
    )
    ordered_steps = sorted(
        stepped,
        key=lambda item: (int(item["step"]), str(item["identity"])),
    )
    marginal_contrasts = []
    for previous_step, current_step in zip(
        ordered_steps,
        ordered_steps[1:],
        strict=False,
    ):
        contrast = fit.contrast(
            str(current_step["identity"]),
            str(previous_step["identity"]),
        )
        previous_ns = previous_step.get("published_ns")
        current_ns = current_step.get("published_ns")
        elapsed_hours = (
            (current_ns - previous_ns) / 3_600_000_000_000
            if isinstance(previous_ns, int)
            and isinstance(current_ns, int)
            and current_ns > previous_ns
            else None
        )
        marginal_contrasts.append(
            {
                "from_identity": previous_step["identity"],
                "from_step": previous_step["step"],
                "to_identity": current_step["identity"],
                "to_step": current_step["step"],
                "delta_elo": contrast.difference,
                "standard_error": contrast.standard_error,
                "confidence_interval": list(contrast.confidence_interval),
                "elapsed_wall_hours": elapsed_hours,
                "elo_per_wall_hour": (
                    contrast.difference / elapsed_hours if elapsed_hours else None
                ),
                "elo_per_wall_hour_confidence_interval": (
                    [
                        contrast.confidence_interval[0] / elapsed_hours,
                        contrast.confidence_interval[1] / elapsed_hours,
                    ]
                    if elapsed_hours
                    else None
                ),
                "time_basis": "checkpoint_publication",
            }
        )
    return {
        "status": "available",
        "scope": scope,
        "ring": PRIMARY_ELO_RING if scope == "ring_10" else None,
        "anchor_identity": anchor_identity,
        "ladder": ladder,
        "latest": latest,
        "marginal_contrasts": marginal_contrasts,
        "input": {
            "result_count": len(inputs),
            "fitted_result_count": fit.observation_count,
            "unique_pairing_count": fit.unique_pairing_count,
            "decisive_games": fit.decisive_games,
            "continuity_corrected_pairings": (fit.continuity_corrected_pairings),
            "saturated_one_sided_pairing_count": len(saturated_pairings),
            "saturated_one_sided_pairings": saturated_pairings,
        },
        "fit": {
            "converged": fit.converged,
            "iterations": fit.iterations,
            "log_likelihood": fit.log_likelihood,
        },
        "connectedness": {
            "connected": fit.connected,
            "component_count": len(fit.components),
            "anchor_component_size": len(fit.estimates),
            "identity_count": sum(len(component) for component in fit.components),
            "components": [list(component) for component in fit.components],
            "excluded_identities": list(fit.excluded_identities),
            "excluded_result_paths": sorted(
                path for path in disconnected_results if isinstance(path, str)
            ),
        },
        "exclusions": exclusions,
    }


def _frozen_baseline_results(
    arena_results: list[dict[str, object]],
    *,
    checkpoint_steps: Mapping[str, int],
) -> dict[str, object]:
    serialized = []
    for result in arena_results:
        kind, evidence = _baseline_kind(
            result,
            checkpoint_steps=checkpoint_steps,
        )
        if kind != "frozen":
            continue
        aggregate = result.get("aggregate")
        per_ring = result.get("per_ring")
        ring_10 = (
            per_ring.get(str(PRIMARY_ELO_RING))
            if isinstance(per_ring, Mapping)
            else None
        )

        def summary(value: object) -> dict[str, object] | None:
            if not isinstance(value, Mapping):
                return None
            return {
                "wins": value.get("wins"),
                "losses": value.get("losses"),
                "games": value.get("games"),
                "elo_difference": value.get("elo_difference"),
                "anytime_elo_interval": value.get("anytime_elo_interval"),
            }

        serialized.append(
            {
                "path": result.get("_path"),
                "result_kind": _arena_result_kind(result),
                "result_category": _arena_result_category(result),
                "candidate": result.get("candidate"),
                "baseline": result.get("baseline"),
                "kind": evidence,
                "completed_ns": _timestamp(result),
                "aggregate": summary(aggregate),
                "ring_10": summary(ring_10),
            }
        )
    serialized.sort(
        key=lambda item: (
            _number(item.get("completed_ns")) or 0,
            str(item.get("path")),
        )
    )
    return {
        "connected_to_primary": False,
        "result_count": len(serialized),
        "results": serialized,
    }


def _autonomous_elo_summary(
    root: Path,
    *,
    arena_results: list[dict[str, object]],
    actor_records: list[dict[str, object]],
    actor_summary: Mapping[str, object],
    provisioned_gpu_hours: float,
) -> dict[str, object]:
    (
        checkpoint_steps,
        checkpoint_publications,
        metadata_exclusions,
    ) = _checkpoint_step_evidence(
        root,
        actor_records=actor_records,
        arena_results=arena_results,
    )
    aggregate_inputs, aggregate_exclusions = _scope_inputs(
        arena_results,
        scope="aggregate",
        checkpoint_steps=checkpoint_steps,
    )
    ring_inputs, ring_exclusions = _scope_inputs(
        arena_results,
        scope="ring_10",
        checkpoint_steps=checkpoint_steps,
    )
    anchor_inputs = aggregate_inputs or ring_inputs
    anchor_identity, anchor_selection = _select_anchor(
        anchor_inputs,
        checkpoint_steps=checkpoint_steps,
    )
    primary = _build_ladder(
        scope="ring_10",
        inputs=ring_inputs,
        exclusions=ring_exclusions,
        anchor_identity=anchor_identity,
        checkpoint_steps=checkpoint_steps,
        checkpoint_publications=checkpoint_publications,
    )
    aggregate = _build_ladder(
        scope="aggregate",
        inputs=aggregate_inputs,
        exclusions=aggregate_exclusions,
        anchor_identity=anchor_identity,
        checkpoint_steps=checkpoint_steps,
        checkpoint_publications=checkpoint_publications,
    )
    latest_source = None
    latest = None
    if primary.get("status") == "available":
        latest_source = "ring_10"
        latest = primary.get("latest")
    elif aggregate.get("status") == "available":
        latest_source = "aggregate"
        latest = aggregate.get("latest")
    latest_elo = _number(latest.get("rating")) if isinstance(latest, Mapping) else None
    aggregate_latest = (
        aggregate.get("latest") if aggregate.get("status") == "available" else None
    )
    headline = (
        {
            "source": "aggregate",
            "confidence_level": AUTONOMOUS_ELO_CONFIDENCE,
            **aggregate_latest,
        }
        if isinstance(aggregate_latest, dict)
        else None
    )
    headline_elo = (
        _number(headline.get("rating")) if isinstance(headline, Mapping) else None
    )
    leaf_evaluations = (
        _nonnegative_integer(actor_summary.get("evaluator_rows"))
        if actor_records
        else None
    )
    billion_leaf_evaluations = (
        leaf_evaluations / 1_000_000_000 if leaf_evaluations is not None else None
    )
    return {
        "schema_version": AUTONOMOUS_ELO_SCHEMA_VERSION,
        "method": "connected-bradley-terry-elo-v1",
        "fit_estimator": (
            "maximum-likelihood with 0.5 symmetric continuity correction "
            "for one-sided pairings"
        ),
        "uncertainty_method": (
            "marginal and correlated-contrast normal intervals from the raw "
            "observed-information Hessian"
        ),
        "confidence_level": AUTONOMOUS_ELO_CONFIDENCE,
        "primary_ring": PRIMARY_ELO_RING,
        "anchor": {
            "identity": anchor_identity,
            "step": checkpoint_steps.get(anchor_identity)
            if anchor_identity is not None
            else None,
            "rating": 0.0 if anchor_identity is not None else None,
            "selection": anchor_selection,
        },
        "primary_ring_10": primary,
        "aggregate": aggregate,
        "latest": (
            {"source": latest_source, **latest} if isinstance(latest, dict) else None
        ),
        "latest_elo": latest_elo,
        "headline": headline,
        "headline_elo": headline_elo,
        "efficiency": {
            "leaf_evaluations": leaf_evaluations,
            "leaf_evaluation_source": "actors.evaluator_rows"
            if leaf_evaluations is not None
            else None,
            "billion_leaf_evaluations": billion_leaf_evaluations,
            "provisioned_gpu_hours": provisioned_gpu_hours or None,
            "elo_per_billion_leaf_evaluations": (
                latest_elo / billion_leaf_evaluations
                if latest_elo is not None and billion_leaf_evaluations
                else None
            ),
            "elo_per_provisioned_gpu_hour": (
                latest_elo / provisioned_gpu_hours
                if latest_elo is not None and provisioned_gpu_hours
                else None
            ),
            "headline_elo_per_billion_leaf_evaluations": (
                headline_elo / billion_leaf_evaluations
                if headline_elo is not None and billion_leaf_evaluations
                else None
            ),
            "headline_elo_per_provisioned_gpu_hour": (
                headline_elo / provisioned_gpu_hours
                if headline_elo is not None and provisioned_gpu_hours
                else None
            ),
        },
        "saturation": {
            "aggregate": {
                "saturated_one_sided_pairing_count": _nonnegative_integer(
                    _mapping(aggregate.get("input")).get(
                        "saturated_one_sided_pairing_count"
                    )
                ),
                "saturated_one_sided_pairings": _mapping(aggregate.get("input")).get(
                    "saturated_one_sided_pairings", []
                ),
            },
            "primary_ring_10": {
                "saturated_one_sided_pairing_count": _nonnegative_integer(
                    _mapping(primary.get("input")).get(
                        "saturated_one_sided_pairing_count"
                    )
                ),
                "saturated_one_sided_pairings": _mapping(primary.get("input")).get(
                    "saturated_one_sided_pairings", []
                ),
            },
        },
        "frozen_baselines": _frozen_baseline_results(
            arena_results,
            checkpoint_steps=checkpoint_steps,
        ),
        "step_metadata_exclusions": metadata_exclusions,
    }


def _coordinator_summary(records: list[dict[str, object]]) -> dict[str, object]:
    ready_by_token: dict[str, int] = {}
    pause_seconds = 0.0
    completed_leases = 0
    learner_pause_restarts = 0
    hardware_failures = 0
    for record in records:
        event = record.get("event")
        token = record.get("token")
        timestamp = _timestamp(record)
        if event == "pause_lease_ready" and isinstance(token, str) and timestamp:
            ready_by_token[token] = timestamp
        elif (
            event in ("pause_lease_release_requested", "pause_lease_released")
            and isinstance(token, str)
            and timestamp
            and token in ready_by_token
        ):
            pause_seconds += max(
                0.0, (timestamp - ready_by_token.pop(token)) / 1_000_000_000
            )
            completed_leases += 1
        if event == "pause_target_restarted" and record.get("target") == "learner":
            learner_pause_restarts += 1
        if event == "hardware_health_failure":
            hardware_failures += 1
    return {
        "records": len(records),
        "completed_pause_leases": completed_leases,
        "pause_lease_seconds": pause_seconds,
        "learner_pause_restarts": learner_pause_restarts,
        "hardware_health_failures": hardware_failures,
        "open_pause_lease_count": len(ready_by_token),
        "efficiency_denominator_policy": (
            "pause, cooldown, restart, and idle time remain included in total "
            "wall time and provisioned GPU-hours"
        ),
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
    coordinator_records = _read_jsonl(
        root / "metrics" / "coordinator.jsonl",
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
        for record in [*learner_records, *actor_records, *coordinator_records, *arenas]
        if (timestamp := _timestamp(record)) is not None
    ]
    latest_metric_ns = max(observed_timestamps, default=started_ns)
    coordinator_status_path = root / "status" / "coordinator.json"
    coordinator_status = {}
    if coordinator_status_path.is_file():
        try:
            loaded_status = json.loads(
                coordinator_status_path.read_text(encoding="utf-8")
            )
            coordinator_status = (
                loaded_status if isinstance(loaded_status, dict) else {}
            )
        except (OSError, json.JSONDecodeError):
            coordinator_status = {}
    coordinator_timestamp = coordinator_status.get("timestamp_ns")
    if coordinator_status.get("state") == "running":
        observed_until_ns = max(latest_metric_ns, time.time_ns())
        observation_end_source = "report_capture_while_running"
    elif (
        coordinator_status.get("state") == "stopped"
        and isinstance(coordinator_timestamp, int)
        and not isinstance(coordinator_timestamp, bool)
        and coordinator_timestamp > 0
    ):
        observed_until_ns = max(latest_metric_ns, coordinator_timestamp)
        observation_end_source = "coordinator_terminal_timestamp"
    else:
        observed_until_ns = latest_metric_ns
        observation_end_source = "latest_metric"
    wall_seconds = max(0.0, (observed_until_ns - started_ns) / 1_000_000_000)
    provisioned_gpu_hours = provisioned_gpus * wall_seconds / 3_600.0
    learner_summary = _learner_summary(learner_records)
    actor_summary = _actor_summary(actor_records)
    migration_summary = _migration_summary(
        root,
        run=run,
        run_started_ns=started_ns,
        observed_until_ns=observed_until_ns,
        provisioned_gpus=provisioned_gpus,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": "complete" if not failures else "incomplete",
        "run_root": str(root),
        "run_id": run.get("run_id"),
        "generation_family": run.get("generation_family"),
        "started_ns": started_ns,
        "observed_until_ns": observed_until_ns,
        "observation_end_source": observation_end_source,
        "wall_seconds": wall_seconds,
        "provisioned_gpus": provisioned_gpus,
        "provisioned_gpu_hours": provisioned_gpu_hours,
        "migrations": migration_summary,
        "migration_boundaries": migration_summary["boundaries"],
        "migration_segments": migration_summary["segments"],
        "coordinator": _coordinator_summary(coordinator_records),
        "learner": learner_summary,
        "actors": actor_summary,
        "arena": _arena_summary(
            arenas,
            run_started_ns=started_ns,
            provisioned_gpus=provisioned_gpus,
        ),
        "autonomous_elo": _autonomous_elo_summary(
            root,
            arena_results=arenas,
            actor_records=actor_records,
            actor_summary=actor_summary,
            provisioned_gpu_hours=provisioned_gpu_hours,
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
