#!/usr/bin/env python3
"""Read-only operational evidence for a self-play pipeline canary.

Reports never change a service, profile, control file or replay. A valid report
exits successfully; its separate gate is pass, pending or fail. Optional output
must be outside the run directory and may replace only this reporter's artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import time

FORMAT = "startrain.selfplay-pipeline-canary"
MAX_JSON_BYTES = 2 * 1024**2
MAX_TAIL_BYTES = 8 * 1024**2


def _json(path: Path) -> dict:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return {}
        result = json.loads(path.read_text())
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def _number(value):
    return (
        value
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
        else None
    )


def _integer(value):
    return value if type(value) is int and value >= 0 else None


def _tail(path: Path, evidence: dict) -> list[str]:
    try:
        with path.open("rb") as stream:
            size = path.stat().st_size
            start = max(0, size - MAX_TAIL_BYTES)
            stream.seek(start)
            if start:
                stream.readline()
                evidence["truncated_files"].append(str(path))
            return (
                stream.read(MAX_TAIL_BYTES)
                .decode(errors="replace")
                .splitlines(keepends=True)
            )
    except OSError:
        return []


def _rows(path: Path, evidence: dict) -> list[dict]:
    result = []
    for line in _tail(path, evidence):
        if not line.endswith("\n"):
            evidence["partial_jsonl_lines"] += 1
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except ValueError:
            evidence["malformed_jsonl_lines"] += 1
    return result


def _belongs(child: str, parent: str) -> bool:
    return (
        child == parent
        or re.fullmatch(re.escape(parent) + r"-cohort-\d+", child) is not None
    )


def _maximum(rows: list[dict], field: str):
    values = [value for row in rows if (value := _number(row.get(field))) is not None]
    return max(values) if values else None


def _settings(config, gpu):
    pipeline = gpu.actor_pipeline
    refresh = config.orchestration.model_refresh
    return {
        "compatible_work": pipeline.compatible_work
        if pipeline
        else refresh.compatible_cohort_work,
        "stream_completed_games": pipeline.stream_completed_games
        if pipeline
        else config.selfplay.stream_completed_games,
        "rolling_game_slots": pipeline.rolling_game_slots
        if pipeline
        else config.selfplay.rolling_game_slots,
        "seed_contract": pipeline.seed_contract
        if pipeline
        else config.selfplay.seed_contract,
        "cuda_graphs": pipeline.cuda_graphs
        if pipeline
        else refresh.inference.cuda_graphs,
        "games_per_task": pipeline.games_per_task if pipeline else None,
        "slots_per_cohort": gpu.actor_batch_size,
        "cohorts": gpu.actor_cohorts,
        "native_threads": gpu.native_threads or gpu.cpu_threads,
        "blas_threads": gpu.blas_threads or gpu.cpu_threads,
        "cpu_affinity": gpu.cpu_affinity,
        "graph_max_entries_per_model": refresh.inference.cuda_graph_max_entries,
        "graph_max_bytes_aggregate": refresh.inference.cuda_graph_max_bytes,
        "registry_model_capacity": gpu.actor_cohorts + 2
        if gpu.actor_cohorts > 1
        else 1,
        "graph_max_bytes_per_model": max(
            1,
            refresh.inference.cuda_graph_max_bytes
            // (gpu.actor_cohorts + 2 if gpu.actor_cohorts > 1 else 1),
        ),
    }


def build_report(
    run_root: Path,
    profile: Path,
    *,
    gpu: int = 7,
    since_ns: int,
    min_games: int = 16,
    min_graph_replays: int = 100,
    min_refills: int = 1,
    now_ns: int | None = None,
) -> dict:
    from startrain.config import load_config
    from startrain.model import model_parameter_count

    if any(
        type(value) is not int or value < 0
        for value in (gpu, since_ns, min_games, min_graph_replays, min_refills)
    ):
        raise ValueError(
            "GPU, timestamps and minimum counters must be nonnegative integers"
        )
    root = run_root.resolve()
    config = load_config(profile)
    directory = config.orchestration.directories
    status, metrics, logs = (
        root / directory.status,
        root / directory.metrics,
        root / directory.logs,
    )
    now = time.time_ns() if now_ns is None else now_ns
    evidence = {
        "truncated_files": [],
        "partial_jsonl_lines": 0,
        "malformed_jsonl_lines": 0,
    }
    failures, pending = [], []
    coordinator = _json(status / "coordinator.json")
    workers = coordinator.get("workers", {})
    if not isinstance(workers, dict):
        workers = {}
    if coordinator.get("state") != "running" or coordinator.get("draining"):
        pending.append("coordinator is not running normally")
    if coordinator.get("failure") or coordinator.get("hardware_failure_reason"):
        failures.append("coordinator reports a terminal or hardware failure")
    identity = _json(root / "run.json")
    if not identity.get("run_id") or not identity.get("generation_family"):
        pending.append("run identity is unavailable")
    maximum_age = config.orchestration.shutdown.stale_heartbeat_seconds
    coordinator_stamp = _integer(coordinator.get("timestamp_ns"))
    if (
        coordinator_stamp is None
        or not 0 <= (now - coordinator_stamp) / 1e9 <= maximum_age
    ):
        pending.append("coordinator snapshot is unavailable or stale")
    health = []
    learner_step = 0
    for name, worker in workers.items():
        if not isinstance(worker, dict):
            pending.append(f"invalid coordinator worker {name}")
            continue
        if (_number(worker.get("restart_count")) or 0) > 0 or worker.get(
            "failure_reason"
        ):
            failures.append(f"worker {name} has a restart or failure")
        heartbeat_path = Path(str(worker.get("heartbeat", "")))
        if not heartbeat_path.is_absolute():
            heartbeat_path = status / heartbeat_path
        heartbeat = (
            _json(heartbeat_path)
            if heartbeat_path.resolve().is_relative_to(status.resolve())
            else {}
        )
        age = (
            (now - heartbeat.get("heartbeat_ns", 0)) / 1e9
            if _integer(heartbeat.get("heartbeat_ns")) is not None
            else None
        )
        valid = (
            type(worker.get("pid")) is int
            and worker["pid"] > 0
            and heartbeat.get("pid") == worker["pid"]
            and heartbeat.get("worker") == name
            and age is not None
            and 0 <= age <= maximum_age
        )
        if not valid:
            pending.append(f"worker {name} lacks a fresh matching-PID heartbeat")
        elif worker.get("role") == "learner":
            learner_step = int(_integer(heartbeat.get("step")) or 0)
        health.append(
            {
                "worker": name,
                "pid": worker.get("pid"),
                "state": worker.get("state"),
                "restart_count": worker.get("restart_count"),
                "heartbeat_age_seconds": age,
                "heartbeat_valid": valid,
            }
        )
    events = [
        row
        for row in _rows(metrics / "coordinator.jsonl", evidence)
        if (_integer(row.get("timestamp_ns")) or 0) >= since_ns
    ]
    bad_events = [
        row
        for row in events
        if row.get("event")
        in {
            "worker_exited",
            "worker_spawn_failed",
            "worker_restart_exhausted",
            "coordinator_terminal_failure",
        }
        or (_number(row.get("restart_count")) or 0) > 0
    ]
    if bad_events:
        failures.append(
            "coordinator history records a worker failure or restart since the cutoff"
        )
    by_gpu = {}
    for configured in config.orchestration.gpus:
        if configured.role != "actor":
            continue
        settings = _settings(config, configured)
        assigned = [
            (name, worker)
            for name, worker in workers.items()
            if isinstance(worker, dict)
            and worker.get("role") == "actor"
            and worker.get("gpu_ids") == [configured.gpu_id]
        ]
        if configured.gpu_id == gpu and len(assigned) != 1:
            pending.append("canary requires exactly one coordinator actor process")
        process_reports, physical, broker, work = [], {}, {}, {}
        parent_heartbeat = {}
        fallback_reasons = Counter()
        task_records = {}
        for parent, worker in assigned:
            heartbeats = {}
            for path in status.glob(f"{parent}*.json"):
                heartbeat = _json(path)
                name = heartbeat.get("worker")
                if not isinstance(name, str) or not _belongs(name, parent):
                    continue
                stamp = _integer(heartbeat.get("heartbeat_ns"))
                if (
                    heartbeat.get("pid") != worker.get("pid")
                    or stamp is None
                    or stamp < since_ns
                    or not 0 <= (now - stamp) / 1e9 <= maximum_age
                ):
                    continue
                heartbeats[name] = heartbeat
                if heartbeat.get("phase") == "failed":
                    failures.append(f"actor heartbeat {name} reports failure")
            parent_heartbeat = heartbeats.get(parent, {})
            broker = parent_heartbeat.get("inference", {})
            broker = broker if isinstance(broker, dict) else {}
            physical = broker.get("physical_inference", {})
            physical = physical if isinstance(physical, dict) else {}
            work = parent_heartbeat.get("compatible_work", {})
            rows = []
            for path in metrics.glob(f"{parent}*.jsonl"):
                rows.extend(_rows(path, evidence))
            for name, heartbeat in heartbeats.items():
                candidates = [
                    row
                    for row in rows
                    if row.get("worker") == name
                    and row.get("gpu_id") == configured.gpu_id
                    and row.get("run_id") == identity.get("run_id")
                    and row.get("generation_family")
                    == identity.get("generation_family")
                    and _integer(row.get("process_started_ns")) is not None
                    and (_integer(row.get("process_started_ns")) or 0) >= since_ns
                    and _integer(row.get("timestamp_ns")) is not None
                    and (_integer(row.get("timestamp_ns")) or 0) >= since_ns
                ]
                process_start = _integer(heartbeat.get("process_started_ns"))
                if process_start is None:
                    anchors = [
                        row["process_started_ns"]
                        for row in candidates
                        if row.get("generation") == heartbeat.get("generation")
                    ]
                    process_start = max(anchors) if anchors else None
                if process_start is None or process_start < since_ns:
                    continue
                current = [
                    row
                    for row in candidates
                    if row["process_started_ns"] == process_start
                ]
                if not current and "cumulative_games" not in heartbeat:
                    continue
                current_and_heartbeat = [*current, heartbeat]
                finals = {}
                for row in current:
                    generation = _integer(row.get("generation"))
                    if generation is None:
                        continue
                    key = (name, process_start, generation)
                    task_records[key] = row
                    if (
                        row.get("record_kind") != "publication"
                        and _integer(row.get("games")) is not None
                    ):
                        finals[key] = row
                started = sum(
                    int(
                        row.get(
                            "started_games",
                            int(row.get("games", 0)) + int(row.get("dropped_games", 0)),
                        )
                    )
                    for row in finals.values()
                )
                refills = (
                    sum(
                        max(
                            0,
                            int(
                                row.get(
                                    "refilled_games",
                                    int(row.get("started_games", row.get("games", 0)))
                                    - int(configured.actor_batch_size or 0),
                                )
                            ),
                        )
                        for row in finals.values()
                    )
                    if settings["rolling_game_slots"]
                    else 0
                )
                active_key = (name, process_start, heartbeat.get("generation"))
                if active_key not in finals:
                    active_rows = [
                        row
                        for row in current
                        if row.get("generation") == heartbeat.get("generation")
                    ]
                    active_completed = int(
                        _maximum(active_rows, "published_task_games") or 0
                    )
                    # Heartbeat details survive task boundaries. Old task-local
                    # refill counters must never be counted again as new work.
                    first_task = (
                        bool(active_rows)
                        and _maximum(current_and_heartbeat, "cumulative_games")
                        == active_completed
                    )
                    if heartbeat.get("phase") == "selfplay_refill" or first_task:
                        started += int(
                            _integer(heartbeat.get("started_games")) or active_completed
                        )
                        refills += int(_integer(heartbeat.get("refilled_games")) or 0)
                    else:
                        started += active_completed
                        if settings["rolling_game_slots"]:
                            refills += max(
                                0,
                                active_completed
                                - int(configured.actor_batch_size or 0),
                            )
                dropped = sum(
                    int(row.get("dropped_games", 0)) for row in finals.values()
                )
                dropped_decisions = sum(
                    int(row.get("dropped_decisions", 0)) for row in finals.values()
                )
                games = _maximum(current_and_heartbeat, "cumulative_games")
                samples = _maximum(current_and_heartbeat, "cumulative_samples")
                wall = _maximum(current_and_heartbeat, "cumulative_batch_wall_seconds")
                process_reports.append(
                    {
                        "worker": name,
                        "pid": worker["pid"],
                        "process_started_ns": process_start,
                        "durable_games": games,
                        "durable_samples": samples,
                        "started_games_lower_bound": started,
                        "refilled_games_lower_bound": refills,
                        "dropped_games": dropped,
                        "dropped_decisions": dropped_decisions,
                        "completed_tasks": len(finals),
                        "active_phase": heartbeat.get("phase"),
                        "cumulative_task_wall_seconds": wall,
                        "samples_per_task_second": samples / wall
                        if samples is not None and wall
                        else None,
                    }
                )
            for line in _tail(logs / f"{parent}.log", evidence):
                if "CUDA graph inference fallback:" in line:
                    fallback_reasons[
                        line.split("CUDA graph inference fallback:", 1)[1].strip()
                    ] += 1
        mix = {
            key: Counter(str(row.get(key, "unknown")) for row in task_records.values())
            for key in (
                "model_identity",
                "model_role",
                "requested_model_role",
                "ring",
                "variant",
            )
        }
        by_gpu[str(configured.gpu_id)] = {
            "settings": settings,
            "processes": process_reports,
            "durable_games": sum(row["durable_games"] or 0 for row in process_reports),
            "durable_samples": sum(
                row["durable_samples"] or 0 for row in process_reports
            ),
            "refilled_games_lower_bound": sum(
                row["refilled_games_lower_bound"] for row in process_reports
            ),
            "started_games_lower_bound": sum(
                row["started_games_lower_bound"] for row in process_reports
            ),
            "dropped_games": sum(row["dropped_games"] for row in process_reports),
            "dropped_decisions": sum(
                row["dropped_decisions"] for row in process_reports
            ),
            "physical_inference": physical,
            "broker": {
                key: value
                for key, value in broker.items()
                if key != "physical_inference"
            },
            "compatible_work": work,
            "effective_coordinated_work": parent_heartbeat.get(
                "effective_coordinated_work"
            )
            if assigned
            else None,
            "observed_task_mix": {key: dict(value) for key, value in mix.items()},
            "fallback_logs": {
                "scope": "bounded log tail; lines have no PID/timestamp attribution",
                "count": sum(fallback_reasons.values()),
                "reasons": dict(fallback_reasons),
            },
        }
    target = by_gpu.get(str(gpu), {})
    if not target:
        pending.append("canary GPU is not a configured actor")
    if not target.get("processes"):
        pending.append("no current-process durable actor evidence yet")
    if (
        target.get("settings", {}).get("compatible_work")
        and target.get("effective_coordinated_work") is not True
    ):
        pending.append("compatible work is not confirmed by the current heartbeat")
    for key in ("worker_failures", "failed_requests"):
        value = _number(target.get("broker", {}).get(key))
        if value is None:
            pending.append(f"canary broker {key} evidence is unavailable")
        elif value:
            failures.append(f"canary broker {key}={value}")
    validation_failures = _number(
        target.get("physical_inference", {}).get("graph_validation_failures")
    )
    if validation_failures is None:
        pending.append("graph validation evidence is unavailable")
    elif validation_failures:
        failures.append(f"graph validation failures={validation_failures}")
    if target.get("dropped_games") or target.get("dropped_decisions"):
        failures.append("canary reports dropped games or trajectories")
    checks = {
        "durable_games": (target.get("durable_games", 0), min_games),
        "graph_replays": (
            target.get("physical_inference", {}).get("graph_replays", 0),
            min_graph_replays,
        ),
        "refilled_games": (target.get("refilled_games_lower_bound", 0), min_refills),
    }
    for name, (observed, minimum) in checks.items():
        if _number(observed) is None or observed < minimum:
            pending.append(f"minimum {name}: {observed}/{minimum}")
    variants = config.selfplay.variants
    mode_weights = {
        "classic-standard": variants.classic,
        "double-standard": variants.standard,
        "classic-pie": variants.pie * variants.pie_classic_share,
        "double-pie": variants.pie * (1 - variants.pie_classic_share),
        "classic-handicap": variants.handicap * variants.handicap_classic_share,
        "double-handicap": variants.handicap * (1 - variants.handicap_classic_share),
    }
    weights = config.orchestration.ring_mixture.weights_for_step(learner_step)
    expected = {
        "model_parameters": model_parameter_count(config.model),
        "board_rings": list(config.orchestration.ring_mixture.rings),
        "board_weights": weights,
        "weights_at_learner_step": learner_step,
        "mode_weights": mode_weights,
        "promotion_rings": list(config.arena.rings),
        "required_regression_rings": list(config.arena.required_regression_rings or ()),
    }
    if (
        expected["model_parameters"] != 17_402_775
        or tuple(expected["board_rings"]) != (4, 6, 8, 10)
        or tuple(weights or ()) != (0.05, 0.05, 0.05, 0.85)
        or not variants.enabled
        or tuple(config.arena.rings) != (10,)
        or config.arena.required_regression_rings not in ((), None)
        or any(
            not math.isclose(value, 1 / 6, abs_tol=1e-12)
            for value in mode_weights.values()
        )
    ):
        failures.append(
            "profile differs from the approved model or 85/5 six-mode objective"
        )
    again = _json(status / "coordinator.json")
    new_workers = again.get("workers", {})
    new_workers = new_workers if isinstance(new_workers, dict) else {}
    if again.get("coordinator_pid") != coordinator.get("coordinator_pid") or {
        name: value.get("pid")
        for name, value in new_workers.items()
        if isinstance(value, dict)
    } != {
        name: value.get("pid")
        for name, value in workers.items()
        if isinstance(value, dict)
    }:
        pending.append("coordinator identities changed while reading the snapshot")
    return {
        "format": FORMAT,
        "schema_version": 1,
        "timestamp_ns": now,
        "run_root": str(root),
        "profile": str(profile.resolve()),
        "gpu": gpu,
        "since_ns": since_ns,
        "gate": "fail" if failures else "pending" if pending else "pass",
        "failures": sorted(set(failures)),
        "pending": sorted(set(pending)),
        "minimums": {
            key: {"observed": value[0], "required": value[1]}
            for key, value in checks.items()
        },
        "expected": expected,
        "workers": health,
        "coordinator_failure_events": bad_events,
        "canary": target,
        "baseline_gpus": {
            key: value for key, value in by_gpu.items() if key != str(gpu)
        },
        "read_evidence": evidence,
        "comparison_scope": "Descriptive rates only: board, mode, model and task workloads are unmatched. No Elo or speedup conclusion. Incomplete tasks alone are not evidence of a stall.",
    }


def write_report(path: Path, report: dict) -> None:
    from startrain.runtime import atomic_json

    destination = path.resolve()
    root = Path(report["run_root"])
    if (
        destination == root
        or destination.is_relative_to(root)
        or destination == Path(report["profile"])
    ):
        raise ValueError("report output must be outside protected run/profile paths")
    if path.is_symlink():
        raise ValueError("report output cannot be a symlink")
    if path.exists():
        previous = _json(path)
        if (
            previous.get("format") != FORMAT
            or previous.get("run_root") != report["run_root"]
            or previous.get("gpu") != report["gpu"]
        ):
            raise ValueError("existing output is not this canary reporter's artifact")
    atomic_json(destination, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--gpu7", dest="gpu", action="store_const", const=7)
    parser.add_argument("--since-ns", type=int, required=True)
    parser.add_argument("--min-games", type=int, default=16)
    parser.add_argument("--min-graph-replays", type=int, default=100)
    parser.add_argument("--min-refills", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report(
        args.run_root,
        args.profile,
        gpu=args.gpu,
        since_ns=args.since_ns,
        min_games=args.min_games,
        min_graph_replays=args.min_graph_replays,
        min_refills=args.min_refills,
    )
    if args.output is not None:
        write_report(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
