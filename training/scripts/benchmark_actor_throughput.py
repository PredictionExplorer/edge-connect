#!/usr/bin/env python3
"""Compare fixed production self-play search prefixes on an isolated GPU.

Each arm runs identical logical game cohorts, changing only their concurrency,
native thread count and inference row limit. The prefix ends after fully searched
and applied moves. No partial game is called a completed game or training sample.
Exact trace and search-work parity are required before reporting relative speed;
a completed-game production canary is still required to claim training throughput.

Example (GPU isolation is the operator's responsibility):
  PYTHONPATH=. .venv/bin/python scripts/benchmark_actor_throughput.py \
    --config /path/profile.yaml --checkpoint /path/immutable-manifest.json \
    --device cuda:7 --cpu-affinity 40-55 --arms 2:4:256 4:4:256 4:8:512 \
    --tasks 4 --batch-size 128 --plies 8 --ring 10 --variants double \
    --timeout-seconds 900 --output-dir /path/new-report --execute

For coordinated prewarming, launch one arm per harness with --start-barrier and
--ready-marker pointing to new paths. After warmup the atomic ready JSON contains
a token and child PID. The operator releases all warmed arms by atomically writing
{"schema_version":1,"ready_tokens":[each ready marker's token]} to the common
barrier. The per-arm timeout includes loading, warmup, barrier wait and measurement.
Use --load-context isolated only for isolated measurements; that mode also checks
GPU process ownership immediately before and after the measured interval.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import combinations, islice
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Literal, cast
import uuid


class BenchmarkInterrupted(Exception):
    def __init__(self, signum: int, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(f"benchmark interrupted by signal {signum}")
        self.signum = signum
        self.stdout = stdout
        self.stderr = stderr


@contextmanager
def _controller_signals():
    def interrupted(signum, _frame):
        raise BenchmarkInterrupted(signum)

    previous = {
        kind: signal.signal(kind, interrupted)
        for kind in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for kind, handler in previous.items():
            signal.signal(kind, handler)


def _stop_owned_group(
    process: subprocess.Popen[str], *, grace_seconds: float
) -> tuple[str, str]:
    # Keep the leader unreaped until KILL has reached the whole session. This
    # retains the process-group identity while compiler grandchildren drain.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            return process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # A descendant outside the owned group may retain a pipe, but it
            # cannot prevent bounded reaping of the owned child process.
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            process.wait(timeout=2)
            raise RuntimeError("owned arm terminated but output pipes did not close")
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _run_owned(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    grace_seconds: float = 3.0,
) -> subprocess.CompletedProcess[str]:
    if os.name != "posix":
        raise RuntimeError("owned process-group benchmark requires POSIX")
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException as error:
        stdout, stderr = _stop_owned_group(process, grace_seconds=grace_seconds)
        if isinstance(error, subprocess.TimeoutExpired):
            raise subprocess.TimeoutExpired(
                command, timeout, output=stdout, stderr=stderr
            ) from error
        if isinstance(error, BenchmarkInterrupted):
            raise BenchmarkInterrupted(
                error.signum, stdout=stdout, stderr=stderr
            ) from error
        if isinstance(error, KeyboardInterrupt):
            raise BenchmarkInterrupted(
                signal.SIGINT, stdout=stdout, stderr=stderr
            ) from error
        raise


@dataclass(frozen=True)
class Arm:
    cohorts: int
    native_threads: int
    max_batch_rows: int

    @property
    def label(self) -> str:
        return f"{self.cohorts}:{self.native_threads}:{self.max_batch_rows}"


def parse_arm(value: str) -> Arm:
    try:
        result = Arm(*(int(part) for part in value.split(":")))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("arm must be cohorts:threads:rows") from error
    if not (1 <= result.cohorts <= 16 and 1 <= result.native_threads <= 64):
        raise argparse.ArgumentTypeError("cohorts must be 1..16 and threads 1..64")
    if not 1 <= result.max_batch_rows <= 1024:
        raise argparse.ArgumentTypeError("inference rows must be 1..1024")
    return result


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write_ready(path: Path, evidence: dict[str, object]) -> None:
    """Publish a complete marker atomically without replacing an older marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            json.dump(evidence, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _await_start(args, evidence: dict[str, object], *, deadline: float) -> float:
    if args.start_barrier is None:
        return 0.0
    waited = time.monotonic()
    token = str(uuid.uuid4())
    _write_ready(
        args.ready_marker,
        evidence
        | {
            "schema_version": 1,
            "status": "ready",
            "token": token,
            "pid": os.getpid(),
            "ready_ns": time.time_ns(),
            "barrier": str(args.start_barrier.resolve()),
        },
    )
    while time.monotonic() < deadline:
        if args.start_barrier.exists():
            if args.start_barrier.stat().st_size > 65536:
                raise ValueError("start barrier exceeds bounded size")
            barrier = json.loads(args.start_barrier.read_text())
            if not isinstance(barrier, dict) or barrier.get("schema_version") != 1:
                raise ValueError("invalid start barrier schema")
            tokens = barrier.get("ready_tokens")
            if not isinstance(tokens, list) or any(
                not isinstance(item, str) for item in tokens
            ):
                raise ValueError("start barrier must contain ready_tokens")
            if token in tokens:
                return time.monotonic() - waited
        time.sleep(0.1)
    raise TimeoutError("benchmark deadline expired while waiting for start barrier")


def _gpu_ownership(gpu_uuid: str) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                gpu_uuid,
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode:
            return {"verified": False, "error": completed.stderr.strip()}
        owners = []
        for line in completed.stdout.splitlines():
            reported_uuid, pid = (part.strip() for part in line.split(",", 1))
            if reported_uuid != gpu_uuid:
                raise ValueError("nvidia-smi returned another GPU")
            owners.append(int(pid))
        return {
            "verified": owners == [os.getpid()],
            "gpu_uuid": gpu_uuid,
            "owner_pids": owners,
            "observed_ns": time.time_ns(),
        }
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"verified": False, "error": str(error)}


class _NoReplay:
    def append(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("prefix benchmark unexpectedly completed a game")


def _prefix_actor_type():
    # Lazy import leaves the parent harness free of Torch/native initialization.
    from startrain.selfplay import SelfPlayActor

    class PrefixActor(SelfPlayActor):
        def __init__(self, *args: Any, plies: int, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.plies = plies
            self.trace: list[dict[str, Any]] = []
            self.search_decisions = 0
            self.search_simulations = 0

        def _record_decisions(self, *args: Any, **kwargs: Any) -> None:
            super()._record_decisions(*args, **kwargs)
            trajectories, _positions, _state_data, results = args
            active = [not bool(terminal) for terminal in results.terminal]
            budgets = kwargs["budgets"]
            self.search_decisions += sum(active)
            self.search_simulations += sum(
                budget for budget, valid in zip(budgets, active, strict=True) if valid
            )
            self.trace.append(
                {
                    "search_seed": kwargs["search_seed"],
                    "full_search": kwargs["full_search"],
                    "budgets": list(budgets),
                    "swaps": list(kwargs["swaps"]),
                    "results": {
                        field: list(getattr(results, field))
                        for field in (
                            "selected_actions",
                            "terminal",
                            "action_offsets",
                            "actions",
                            "visits",
                            "policy_target",
                            "root_values",
                            "q_values",
                            "priors",
                        )
                    },
                    "replay_prefix": [
                        {
                            "stones": row[-1].position.stones.tolist(),
                            "pda": row[-1].position.pda,
                            "policy": (
                                row[-1].policy.tolist()
                                if row[-1].policy is not None
                                else None
                            ),
                            "policy_weight": row[-1].policy_weight,
                            "policy_surprise": row[-1].policy_surprise,
                            "phase": row[-1].phase,
                        }
                        for row, valid in zip(trajectories, active, strict=True)
                        if valid
                    ],
                }
            )

        def run_prefix(self) -> dict[str, Any]:
            # SelfPlayActor checks this after applying the preceding wave's moves.
            games = self.run(stop_requested=lambda: len(self.trace) >= self.plies)
            metrics = self.metrics_snapshot()
            expected = self.plies * self.config.batch_size
            if (
                games
                or len(self.trace) != self.plies
                or self.search_decisions != expected
            ):
                raise RuntimeError(
                    "self-play did not finish the exact requested prefix"
                )
            if self.persisted_decisions or metrics.completed_decisions:
                raise RuntimeError("prefix benchmark emitted training samples")
            if metrics.dropped_decisions != self.search_decisions:
                raise RuntimeError(
                    "prefix decision accounting differs from native work"
                )
            actions = [
                {key: wave[key] for key in ("search_seed", "budgets", "swaps")}
                | {
                    "results": {
                        key: wave["results"][key]
                        for key in ("selected_actions", "terminal", "actions", "visits")
                    }
                }
                for wave in self.trace
            ]
            return {
                "task_id": self.identity.actor_id,
                "variant": self.config.variant.label,
                "completed_search_waves": len(self.trace),
                "completed_search_decisions": self.search_decisions,
                "search_simulations": self.search_simulations,
                "completed_games": 0,
                "persisted_positions": 0,
                "action_visit_sha256": _digest(actions),
                "replay_trace_sha256": _digest(self.trace),
            }

    return PrefixActor


def run_tasks(
    native,
    evaluator,
    config,
    *,
    broker,
    tasks: int,
    batch_size: int,
    plies: int,
    ring: int,
    variants: list[str],
    cohorts: int,
):
    from startrain.selfplay import GameVariant, SelfPlayIdentity

    actor_type = _prefix_actor_type()

    def task(index: int):
        variant = GameVariant.parse(variants[index % len(variants)])
        task_config = replace(
            config.selfplay, rings=ring, games=batch_size, batch_size=batch_size
        ).with_variant(variant)
        adapter = broker.cohort_adapter(
            evaluator, score_utility_weight=task_config.effective_score_utility_weight()
        )
        actor = actor_type(
            native,
            adapter,
            _NoReplay(),
            task_config,
            SelfPlayIdentity(
                "actor-throughput", "actor-throughput", f"task-{index}", 0
            ),
            plies=plies,
        )
        return actor.run_prefix()

    with ThreadPoolExecutor(
        max_workers=cohorts, thread_name_prefix="prefix-task"
    ) as pool:
        return list(pool.map(task, range(tasks)))


def _warmup_inference(
    native, evaluator, *, ring: int, max_rows: int, variants: list[str]
) -> list[dict[str, object]]:
    from startrain.selfplay import GameVariant

    physical_limit = 1 << (max_rows - 1).bit_length()
    warmed = []
    for variant_label in variants:
        variant = GameVariant.parse(variant_label)
        rows = 1
        while rows <= physical_limit:
            states = native.StateBatch(
                ring,
                rows,
                mode=variant.mode,
                handicap=variant.handicap,
                pie=variant.pie,
            )
            # A unique occupied set prevents cache deduplication in every mode,
            # even when the first three stones all belong to one handicap seat.
            placements = list(islice(combinations(range(states.node_count), 3), rows))
            if len(placements) != rows:
                raise ValueError("board cannot provide enough unique warmup positions")
            for ply in range(3):
                states.apply_many(
                    list(range(rows)), [places[ply] for places in placements]
                )
            request = native.SearchBatch(
                states, simulations=1, max_considered=2, deterministic_seed=7
            ).root_requests()
            evaluator.clear_inference_cache()
            before = evaluator.metrics_snapshot().neural_rows
            evaluator.evaluate(request)
            physical_rows = evaluator.metrics_snapshot().neural_rows - before
            if physical_rows != rows:
                raise RuntimeError(
                    f"warmup expected {rows} physical neural rows, observed {physical_rows}"
                )
            warmed.append({"variant": variant_label, "physical_rows": physical_rows})
            rows *= 2
    evaluator.clear_inference_cache()
    return warmed


def _child(args, arm: Arm) -> dict[str, object]:
    started = time.monotonic()
    from startrain.config import load_config, parse_cpu_affinity

    getattr(os, "sched_setaffinity")(0, set(parse_cpu_affinity(args.cpu_affinity)))
    import torch
    from startrain.actor import ManifestModelProvider
    from startrain.checkpoint import load_model_manifest
    from startrain.inference_batching import BoundedInferenceBroker
    from startrain.inference import GraphInferenceAdapter
    from startrain.model import model_parameter_count
    from startrain.native import load_star_native
    from startrain.runtime import RunIdentity

    torch.set_num_threads(args.blas_threads)
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise ValueError(
            "target-host benchmark requires an explicitly selected CUDA GPU"
        )
    torch.cuda.set_device(args.device)
    config = load_config(args.config)
    if model_parameter_count(config.model) != 17_402_775:
        raise ValueError("benchmark requires the unchanged 17,402,775-parameter model")
    manifest = load_model_manifest(args.checkpoint)
    if manifest.manifest_sha256 != args.manifest_sha256:
        raise ValueError("pinned immutable checkpoint manifest changed")
    native = load_star_native(required=True)
    assert native is not None
    if native.rayon_num_threads() != arm.native_threads:
        raise ValueError("native thread pool does not match requested arm")
    inference = config.orchestration.model_refresh.inference
    # Production partitions its bounded cache among possible resident models.
    inference = replace(
        inference,
        cache_max_entries=inference.cache_max_entries // (arm.cohorts + 2),
        cache_max_bytes=inference.cache_max_bytes // (arm.cohorts + 2),
    )
    config = replace(
        config,
        orchestration=replace(
            config.orchestration,
            model_refresh=replace(
                config.orchestration.model_refresh, inference=inference
            ),
        ),
    )
    provider = ManifestModelProvider(
        config,
        args.checkpoint,
        device=args.device,
        run_identity=RunIdentity(
            Path("unused-benchmark-identity.json"),
            manifest.run_id,
            manifest.generation_family,
            0,
        ),
        expected_role=cast(Literal["champion", "candidate", "direct"], manifest.role),
    )
    evaluator = provider.refresh()
    assert isinstance(evaluator, GraphInferenceAdapter)
    startup_seconds = time.monotonic() - started
    warmup_started = time.monotonic()
    warmed_buckets = _warmup_inference(
        native,
        evaluator,
        ring=args.ring,
        max_rows=arm.max_batch_rows,
        variants=args.variants,
    )
    torch.cuda.synchronize(args.device)
    warmup_seconds = time.monotonic() - warmup_started
    gpu_uuid = str(torch.cuda.get_device_properties(args.device).uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    barrier_wait_seconds = _await_start(
        args,
        {
            "arm": arm.label,
            "device": args.device,
            "gpu_uuid": gpu_uuid,
            "model_identity": manifest.model_identity,
            "manifest_sha256": manifest.manifest_sha256,
            "startup_seconds": startup_seconds,
            "warmup_seconds": warmup_seconds,
            "warmed_physical_buckets": warmed_buckets,
            "load_context": args.load_context,
            "preserve_broadcast_topology": bool(
                getattr(evaluator.config, "preserve_broadcast_topology", False)
            ),
        },
        deadline=started + args.timeout_seconds,
    )
    ownership_before = _gpu_ownership(gpu_uuid)
    if args.load_context == "isolated" and not ownership_before["verified"]:
        raise RuntimeError(f"isolated GPU ownership check failed: {ownership_before}")
    torch.cuda.reset_peak_memory_stats(args.device)
    with BoundedInferenceBroker(
        max_batch_rows=arm.max_batch_rows,
        max_pending_requests=max(inference.max_pending_requests, arm.cohorts),
        max_wait_seconds=inference.max_wait_seconds,
    ) as broker:
        inference_before = asdict(evaluator.metrics_snapshot())
        measured = time.monotonic()
        task_results = run_tasks(
            native,
            evaluator,
            config,
            broker=broker,
            tasks=args.tasks,
            batch_size=args.batch_size,
            plies=args.plies,
            ring=args.ring,
            variants=args.variants,
            cohorts=arm.cohorts,
        )
        torch.cuda.synchronize(args.device)
        elapsed = time.monotonic() - measured
        inference_after = asdict(evaluator.metrics_snapshot())
        metrics = broker.metrics_snapshot()
    ownership_after = _gpu_ownership(gpu_uuid)
    report = {
        "status": "measured",
        "arm": arm.label,
        "startup_seconds": startup_seconds,
        "warmup_seconds": warmup_seconds,
        "warmed_physical_buckets": warmed_buckets,
        "barrier_wait_seconds": barrier_wait_seconds,
        "load_context": args.load_context,
        "gpu_ownership_before": ownership_before,
        "gpu_ownership_after": ownership_after,
        "isolation_verified": args.load_context == "isolated"
        and bool(ownership_before["verified"] and ownership_after["verified"]),
        "isolation_scope": "process ownership snapshots immediately before and after measurement",
        "measured_seconds": elapsed,
        "completed_search_decisions": sum(
            t["completed_search_decisions"] for t in task_results
        ),
        "search_simulations": sum(t["search_simulations"] for t in task_results),
        "completed_games": 0,
        "persisted_positions": 0,
        "tasks": task_results,
        "broker": metrics,
        "inference_delta": {
            name: value - inference_before[name]
            for name, value in inference_after.items()
        },
        "effective_model_cache": {
            "max_entries": evaluator.config.cache_max_entries,
            "max_bytes": evaluator.config.cache_max_bytes,
        },
        "requested_neural_rows": metrics["requested_rows"],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "model_identity": manifest.model_identity,
        "manifest_sha256": manifest.manifest_sha256,
        "parameters": model_parameter_count(config.model),
        "preserve_broadcast_topology": bool(
            getattr(evaluator.config, "preserve_broadcast_topology", False)
        ),
        "selfplay_config": asdict(config.selfplay),
        "native_threads": native.rayon_num_threads(),
        "blas_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "precision": config.train.precision,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cpu_affinity": sorted(getattr(os, "sched_getaffinity")(0)),
    }
    report["search_decisions_per_second"] = (
        report["completed_search_decisions"] / elapsed
    )
    evaluator.close()
    return report


def compare(reference: dict, current: dict) -> dict[str, object]:
    fields = (
        "model_identity",
        "manifest_sha256",
        "selfplay_config",
        "completed_search_decisions",
        "search_simulations",
        "requested_neural_rows",
        "tasks",
    )
    mismatches = [
        field
        for field in fields
        if field not in current
        or field not in reference
        or current[field] != reference[field]
    ]
    valid_times = all(
        isinstance(record.get("measured_seconds"), int | float)
        and math.isfinite(record["measured_seconds"])
        and record["measured_seconds"] > 0
        for record in (reference, current)
    )
    equal = (
        not mismatches
        and valid_times
        and reference.get("status") == current.get("status") == "measured"
    )
    return {
        "exact_work_and_trace_parity": equal,
        "parity_mismatches": mismatches,
        "adoptable_comparison": equal
        and all(
            record.get("load_context") == "isolated"
            and record.get("isolation_verified") is True
            for record in (reference, current)
        ),
        "relative_search_speed": (
            reference["measured_seconds"] / current["measured_seconds"]
            if equal
            else None
        ),
    }


def combine_reports(paths: list[Path]) -> dict[str, object]:
    """Rank independent one-arm reports; the first path is always the baseline.

    Root/operator can call this after coordinating separate GPUs or runtimes.
    No process control occurs here, and shared/unknown ownership never enters
    the adoptable ranking even when its numerical search trace matches.
    """
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("comparison requires a baseline and unique report paths")
    cases = []
    for path in paths:
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("benchmark") != "fixed-selfplay-search-prefix"
        ):
            raise ValueError(f"not a fixed-search benchmark report: {path}")
        records = payload.get("cases")
        if (
            not isinstance(records, list)
            or len(records) != 1
            or not isinstance(records[0], dict)
        ):
            raise ValueError(
                f"independent comparison requires exactly one arm per report: {path}"
            )
        cases.append(records[0] | {"report_path": str(path.resolve())})
    baseline = cases[0]
    evaluated = [case | compare(baseline, case) for case in cases]
    return {
        "benchmark": "fixed-selfplay-search-prefix-comparison",
        "baseline_report": baseline["report_path"],
        "scope": "exact search prefix only; completed-game canary required",
        "ranked_adoptable": sorted(
            (case for case in evaluated if case["adoptable_comparison"]),
            key=lambda case: case["measured_seconds"],
        ),
        "rejected": [case for case in evaluated if not case["adoptable_comparison"]],
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cpu-affinity", required=True)
    parser.add_argument(
        "--arms",
        type=parse_arm,
        nargs="+",
        default=[parse_arm(x) for x in ("2:4:256", "4:4:256", "4:8:512")],
    )
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--ring", type=int, choices=(4, 6, 8, 10), default=10)
    parser.add_argument("--variants", nargs="+", default=["double"])
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--start-barrier", type=Path)
    parser.add_argument("--ready-marker", type=Path)
    parser.add_argument(
        "--load-context",
        choices=("isolated", "shared", "unverified"),
        default="unverified",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    return parser


def _main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    from startrain.config import parse_cpu_affinity
    from startrain.selfplay import GameVariant

    cpus = parse_cpu_affinity(args.cpu_affinity)
    if (args.start_barrier is None) != (args.ready_marker is None):
        parser.error("--start-barrier and --ready-marker must be provided together")
    if args.start_barrier is not None:
        if len(args.arms) != 1:
            parser.error("barrier coordination requires exactly one arm per harness")
        if args.start_barrier.exists() or args.ready_marker.exists():
            parser.error("barrier and ready marker must be new paths")
    if not 1 <= args.plies <= 16 or not 1 <= args.batch_size <= 256:
        parser.error("plies must be 1..16 and batch size 1..256")
    if not 1 <= args.tasks <= 32 or any(a.cohorts > args.tasks for a in args.arms):
        parser.error("tasks must be 1..32 and at least the largest cohort count")
    if args.tasks % len(args.variants):
        parser.error("tasks must be divisible by the number of variants")
    if any(a.max_batch_rows < args.batch_size for a in args.arms):
        parser.error("inference row limit must fit one logical batch")
    if not 1 <= args.blas_threads <= len(cpus) or any(
        a.native_threads > len(cpus) for a in args.arms
    ):
        parser.error("native and BLAS thread budgets must fit reserved CPUs")
    if not math.isfinite(args.timeout_seconds) or not 1 <= args.timeout_seconds <= 3600:
        parser.error("timeout must be finite and in 1..3600 seconds")
    try:
        for variant in args.variants:
            GameVariant.parse(variant)
    except ValueError as error:
        parser.error(str(error))
    if args.child:
        if len(args.arms) != 1:
            parser.error("child requires exactly one arm")
        print(json.dumps(_child(args, args.arms[0]), allow_nan=False))
        return 0
    plan = {
        "benchmark": "fixed-selfplay-search-prefix",
        "arms": [a.label for a in args.arms],
        "tasks": args.tasks,
        "logical_batch_size": args.batch_size,
        "plies": args.plies,
        "ring": args.ring,
        "variants": args.variants,
        "scope": "search prefix only; completed-game canary required",
        "load_context": args.load_context,
    }
    if not args.execute:
        print(json.dumps(plan))
        return 0
    from startrain.checkpoint import load_model_manifest

    manifest = load_model_manifest(args.checkpoint)
    checkpoint = manifest.artifact_manifest or manifest.path
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, arm in enumerate(args.arms):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--config",
            str(args.config.resolve()),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--manifest-sha256",
            manifest.manifest_sha256,
            "--device",
            args.device,
            "--cpu-affinity",
            args.cpu_affinity,
            "--arms",
            arm.label,
            "--blas-threads",
            str(args.blas_threads),
            "--tasks",
            str(args.tasks),
            "--batch-size",
            str(args.batch_size),
            "--plies",
            str(args.plies),
            "--ring",
            str(args.ring),
            "--variants",
            *args.variants,
            "--output-dir",
            str(args.output_dir.resolve()),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--load-context",
            args.load_context,
        ]
        if args.start_barrier is not None:
            command.extend(
                [
                    "--start-barrier",
                    str(args.start_barrier.resolve()),
                    "--ready-marker",
                    str(args.ready_marker.resolve()),
                ]
            )
        environment = dict(
            os.environ,
            RAYON_NUM_THREADS=str(arm.native_threads),
            OMP_NUM_THREADS=str(args.blas_threads),
            MKL_NUM_THREADS=str(args.blas_threads),
            OPENBLAS_NUM_THREADS=str(args.blas_threads),
        )
        try:
            child = _run_owned(
                command,
                env=environment,
                timeout=args.timeout_seconds,
            )
            (args.output_dir / f"arm-{index}.stderr.log").write_text(child.stderr)
            (args.output_dir / f"arm-{index}.stdout.log").write_text(child.stdout)
            if child.returncode:
                case = {
                    "status": "failed",
                    "arm": arm.label,
                    "returncode": child.returncode,
                }
            else:
                case = json.loads(child.stdout.splitlines()[-1])
        except (subprocess.TimeoutExpired, BenchmarkInterrupted) as error:
            case = {
                "status": "interrupted"
                if isinstance(error, BenchmarkInterrupted)
                else "timeout",
                "arm": arm.label,
            }
            if isinstance(error, BenchmarkInterrupted):
                case["signal"] = error.signum
            for stream in ("stdout", "stderr"):
                value = getattr(error, stream) or ""
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                (args.output_dir / f"arm-{index}.{stream}.log").write_text(value)
        if cases:
            case.update(compare(cases[0], case))
        cases.append(case)
        (args.output_dir / "report.json").write_text(
            json.dumps(plan | {"cases": cases}, indent=2)
        )
        print(
            json.dumps(
                {
                    "arm": arm.label,
                    "status": case["status"],
                    "exact_work_and_trace_parity": case.get(
                        "exact_work_and_trace_parity"
                    ),
                }
            ),
            flush=True,
        )
        if case["status"] == "interrupted":
            return 128 + case["signal"]
    return int(
        any(
            c["status"] != "measured" or c.get("exact_work_and_trace_parity") is False
            for c in cases
        )
    )


def main(argv: list[str] | None = None) -> int:
    with _controller_signals():
        try:
            return _main(argv)
        except BenchmarkInterrupted as error:
            return 128 + error.signum


if __name__ == "__main__":
    raise SystemExit(main())
