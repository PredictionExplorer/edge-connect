#!/usr/bin/env python3
"""Bounded frozen-EMA comparison of CUDA graph padding buckets.

The default prints a pinned plan. --execute starts one worker with a hard
deadline. Timings include adapter transfers and output conversion, exclude
native request preparation, and charge graph warmup/compilation separately.
No training profile, model pointer, replay or service is modified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations, islice
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import cast


ARMS = ("existing-graph-buckets", "small-intermediate-graph-buckets")
DEFAULT_ROWS = (3, 5, 9, 17, 33, 65)


def _gpu_snapshot(gpu_uuid):
    """Observe physical GPU ownership and load; errors never imply isolation."""
    observed = {"observed_ns": time.time_ns(), "gpu_uuid": gpu_uuid, "verified": False}
    try:

        def query(fields, kind):
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--id",
                    gpu_uuid,
                    f"--query-{kind}={fields}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            return [
                [item.strip() for item in row.split(",")]
                for row in result.stdout.splitlines()
                if row.strip()
            ]

        gpu = query(
            "uuid,utilization.gpu,memory.used,memory.total,clocks.sm,power.draw,temperature.gpu",
            "gpu",
        )
        if len(gpu) != 1 or len(gpu[0]) != 7 or gpu[0][0] != gpu_uuid:
            raise ValueError(
                "GPU telemetry does not identify the selected physical GPU"
            )
        observed["load"] = dict(
            zip(
                (
                    "utilization_percent",
                    "memory_used_mib",
                    "memory_total_mib",
                    "sm_clock_mhz",
                    "power_watts",
                    "temperature_c",
                ),
                gpu[0][1:],
                strict=True,
            )
        )
        owners = []
        for uuid, pid, memory in query("gpu_uuid,pid,used_gpu_memory", "compute-apps"):
            if uuid != gpu_uuid:
                raise ValueError("process telemetry returned another GPU")
            # Linux start time also distinguishes reuse of a PID during cutover.
            stat = Path(f"/proc/{int(pid)}/stat").read_text()
            start_ticks = int(stat.rsplit(")", 1)[1].split()[19])
            owners.append(
                {"pid": int(pid), "start_ticks": start_ticks, "memory_mib": memory}
            )
        observed.update(
            verified=True, owners=sorted(owners, key=lambda row: row["pid"])
        )
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        observed["error"] = str(error)
    return observed


def _load_assessment(snapshots, *, declared, worker_pid):
    verified = bool(snapshots) and all(row.get("verified") is True for row in snapshots)
    identities = [
        tuple((owner["pid"], owner["start_ticks"]) for owner in row.get("owners", ()))
        for row in snapshots
    ]
    stationary = verified and all(identity == identities[0] for identity in identities)
    isolated = verified and all(
        [owner["pid"] for owner in row.get("owners", ())] == [worker_pid]
        for row in snapshots
    )
    return {
        "declared_context": declared,
        "telemetry_verified": verified,
        "observed_pid_stationarity": stationary,
        "observed_isolation": isolated,
        "comparison_adoptable": declared == "isolated" and stationary and isolated,
        "limitation": "point samples cannot prove stationary utilization or rule out activity between observations",
    }


def _compare_outputs(expected, actual, rows, *, rtol=0.01, atol=0.01):
    import numpy as np

    if expected is None or actual is None:
        raise ValueError("detailed benchmark outputs are missing")
    left, right = expected.response, actual.response
    if (
        len(left.tokens) != rows
        or left.tokens != right.tokens
        or left.policy_offsets != right.policy_offsets
        or len(left.policy_offsets) != rows + 1
        or left.policy_offsets[-1] != len(left.policy_logits)
    ):
        raise ValueError("valid-row tokens or legal policy routing disagree")
    result = {}
    for name, a, b in (
        ("values", left.values, right.values),
        ("policy_logits", left.policy_logits, right.policy_logits),
        *(
            (name, getattr(expected, name), getattr(actual, name))
            for name in (
                "outcome_probabilities",
                "outcome_values",
                "score_expectations",
                "score_probabilities",
            )
        ),
    ):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a.shape != b.shape or (
            name != "policy_logits" and (not a.ndim or a.shape[0] != rows)
        ):
            raise ValueError(f"{name} exposes missing or padded result rows")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"{name} contains nonfinite predictions")
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            raise ValueError(f"{name} predictions exceed numerical tolerance")
        delta = np.abs(a - b)
        result[name] = {
            "shape": list(a.shape),
            "max_absolute_difference": float(delta.max(initial=0)),
            "rms_difference": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
        }
    return result


def _case(
    request,
    adapters,
    *,
    repeats,
    iterations,
    warmups,
    sync,
    snapshot,
    load_context,
    clock=time.perf_counter,
):
    prepared = []
    preparation = []
    for adapter in adapters:
        tick = clock()
        prepared.append(adapter.prepare_requests(request))
        preparation.append(clock() - tick)

    def evaluate(arm, details=False):
        return adapters[arm].evaluate_prepared(
            [prepared[arm]], include_details=[details]
        )[0]

    warm = []
    for arm in (0, 1):
        sync()
        tick = clock()
        for _ in range(warmups):
            evaluate(arm)
        sync()
        warm.append(clock() - tick)
    expected, actual = (evaluate(arm, True)[1] for arm in (0, 1))
    parity = _compare_outputs(expected, actual, len(request))
    before = [adapter.metrics_snapshot() for adapter in adapters]
    snapshots = [snapshot()]
    samples = [[], []]
    sequence = []
    for repeat in range(repeats):
        order = (0, 1) if repeat % 2 == 0 else (1, 0)
        for arm in order:
            sync()
            tick = clock()
            for _ in range(iterations):
                evaluate(arm)
            sync()
            samples[arm].append((clock() - tick) / iterations)
            sequence.append(
                {
                    "repeat": repeat,
                    "arm": ARMS[arm],
                    "seconds_per_call": samples[arm][-1],
                }
            )
            snapshots.append(snapshot())
    after = [adapter.metrics_snapshot() for adapter in adapters]
    differences = [
        asdict(end.delta(start)) for start, end in zip(before, after, strict=True)
    ]
    physical = [adapter._inference_batch_rows(len(request)) for adapter in adapters]
    calls = repeats * iterations
    graphs_valid = all(
        metrics["graph_captures"] == 0
        and metrics["graph_fallbacks"] == 0
        and metrics["graph_validation_failures"] == 0
        and metrics["graph_replays"] == calls
        and metrics["neural_calls"] == calls
        and metrics["neural_rows"] == calls * rows
        and metrics["evaluator_rows"] == calls * len(request)
        for metrics, rows in zip(differences, physical, strict=True)
    )
    load = _load_assessment(snapshots, declared=load_context, worker_pid=os.getpid())
    adoptable = graphs_valid and load["comparison_adoptable"]
    ratios = [old / new for old, new in zip(*samples, strict=True)]
    return {
        "valid_rows": len(request),
        "physical_rows": physical,
        "control_same_bucket": physical[0] == physical[1],
        "preparation_seconds": preparation,
        "warmup_capture_compile_seconds": warm,
        "output_parity": parity,
        "rtol": 0.01,
        "atol": 0.01,
        "latency_seconds": samples,
        "counterbalanced_sequence": sequence,
        "raw_paired_latency_ratios": ratios,
        "paired_speedup_median": statistics.median(ratios) if adoptable else None,
        "paired_speedup_min": min(ratios) if adoptable else None,
        "paired_speedup_max": max(ratios) if adoptable else None,
        "comparison_adoptable": adoptable,
        "graph_replay_only": graphs_valid,
        "warmup_metrics": [asdict(row) for row in before],
        "measured_metrics": differences,
        "load_assessment": load,
        "gpu_observations": snapshots,
    }


def _native_requests(native, ring, rows):
    states = native.StateBatch(ring, rows, mode="double", handicap=1, pie=False)
    positions = list(islice(combinations(range(states.node_count), 3), rows))
    if len(positions) != rows:
        raise ValueError("board has too few distinct legal test positions")
    for ply in range(3):
        states.apply_many(list(range(rows)), [position[ply] for position in positions])
    requests = native.SearchBatch(
        states, simulations=1, max_considered=2, deterministic_seed=7
    ).root_requests()
    if len(requests) != rows:
        raise ValueError("native test positions unexpectedly reached terminal states")
    return requests


def _plan(args):
    from startrain.checkpoint import load_model_manifest
    from startrain.config import load_config
    from startrain import inference, inference_graphs, model, native

    config = load_config(args.config)
    manifest = load_model_manifest(args.checkpoint)
    if not set(args.rings) <= set(config.game.rings):
        raise ValueError(
            "requested boards are outside the frozen model game configuration"
        )
    plan = {
        "schema_version": 1,
        "benchmark": "paired-graph-bucket-padding",
        "model_identity": manifest.model_identity,
        "manifest_sha256": manifest.manifest_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_sha256": {
            module.__name__: hashlib.sha256(
                Path(cast(str, module.__file__)).read_bytes()
            ).hexdigest()
            for module in (inference, inference_graphs, model, native)
        },
        "benchmark_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "precision": config.train.precision,
        "device": args.device,
        "rings": args.rings,
        "valid_rows": args.batch_sizes,
        "position_fixture": "distinct legal three-placement double openings; native search seed 7",
        "repeats": args.repeats,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "compile": args.compile_model,
        "load_context": args.load_context,
        "memory_fraction": args.memory_fraction,
        "graph_cache_bytes_per_arm": args.graph_cache_bytes,
        "compact_inference_gather": config.orchestration.model_refresh.inference.compact_inference_gather,
        "preserve_broadcast_topology": config.orchestration.model_refresh.inference.preserve_broadcast_topology,
        "arms": ARMS,
        "scope": "adapter latency including transfers and CPU output conversion; no Elo or training-throughput claim",
    }
    return plan, config, manifest


def _worker(args, plan, identity, config, manifest):
    import torch
    from startrain.checkpoint import load_ema_checkpoint
    from startrain.inference import GraphInferenceAdapter, InferenceConfig
    from startrain.model import GraphResTNet
    from startrain.native import load_star_native

    started = time.monotonic()
    if not torch.cuda.is_available():
        raise ValueError("graph bucket benchmark requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
    model = GraphResTNet(config.model).eval().to(device)
    load_ema_checkpoint(
        manifest.checkpoint,
        model=model,
        expected_model_config=asdict(config.model),
        expected_game_config=asdict(config.game),
        expected_run_id=manifest.run_id,
        expected_generation_family=manifest.generation_family,
        expected_sha256=manifest.checkpoint_sha256,
        expected_bytes=manifest.checkpoint_bytes,
        map_location=device,
    )
    model.requires_grad_(False)
    model.set_compact_inference_gather(plan["compact_inference_gather"])
    if args.compile_model:
        model = cast(GraphResTNet, torch.compile(model, dynamic=True, fullgraph=True))
    native = load_star_native(required=True)
    if native is None:
        raise ValueError("compiled native engine is unavailable")
    setup_seconds = time.monotonic() - started
    records = []
    for ring in args.rings:
        for rows in args.batch_sizes:
            adapters = []
            try:
                for small in (False, True):
                    adapters.append(
                        GraphInferenceAdapter(
                            model,
                            device=device,
                            model_identity=manifest.model_identity,
                            model_version=manifest.model_version,
                            model_step=manifest.model_step,
                            homogeneous_relational_bias=True,
                            config=InferenceConfig(
                                precision=config.train.precision,
                                cuda_graphs=True,
                                cuda_graph_max_entries=1,
                                cuda_graph_max_bytes=args.graph_cache_bytes,
                                small_batch_graph_buckets=small,
                                compact_inference_gather=plan[
                                    "compact_inference_gather"
                                ],
                                preserve_broadcast_topology=plan[
                                    "preserve_broadcast_topology"
                                ],
                            ),
                        )
                    )
                request = _native_requests(native, ring, rows)
                request_digest = hashlib.sha256()
                for key in request.inference_keys():
                    request_digest.update(len(key).to_bytes(8, "little"))
                    request_digest.update(key)
                record = _case(
                    request,
                    adapters,
                    repeats=args.repeats,
                    iterations=args.iterations,
                    warmups=args.warmups,
                    sync=lambda: torch.cuda.synchronize(device),
                    snapshot=lambda: _gpu_snapshot(gpu_uuid),
                    load_context=args.load_context,
                )
                records.append(
                    {
                        "ring": ring,
                        "native_request_keys_sha256": request_digest.hexdigest(),
                        "resolved_precision": adapters[0].config.precision,
                        **record,
                    }
                )
            finally:
                for adapter in adapters:
                    adapter.close()
                torch.cuda.synchronize(device)
    return {
        "plan": plan,
        "plan_sha256": identity,
        "pid": os.getpid(),
        "gpu_uuid": gpu_uuid,
        "records": records,
        "setup_seconds": setup_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rings", type=int, nargs="+", default=[10])
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(DEFAULT_ROWS)
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument(
        "--load-context",
        choices=("unverified", "shared", "isolated"),
        default="unverified",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.10)
    parser.add_argument("--graph-cache-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pinned-plan", help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if (
        not args.device.startswith("cuda:")
        or not args.device[5:].isdigit()
        or len(set(args.rings)) != len(args.rings)
        or any(ring not in (4, 6, 8, 10) for ring in args.rings)
        or len(set(args.batch_sizes)) != len(args.batch_sizes)
        or not set(args.batch_sizes) <= set(DEFAULT_ROWS)
        or not 2 <= args.repeats <= 12
        or args.repeats % 2
        or not 1 <= args.iterations <= 16
        or not 2 <= args.warmups <= 8
        or not math.isfinite(args.timeout_seconds)
        or not 1 <= args.timeout_seconds <= 900
        or not math.isfinite(args.memory_fraction)
        or not 0.01 <= args.memory_fraction <= 0.25
        or not 1024**2 <= args.graph_cache_bytes <= 4 * 1024**3
    ):
        parser.error(
            "invalid bounded dimensions, counterbalancing, memory limit, CUDA device or deadline"
        )
    if args.worker and (not args.execute or args.pinned_plan is None):
        parser.error("worker requires an executed pinned plan")
    plan, config, manifest = _plan(args)
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if args.pinned_plan is not None and args.pinned_plan != identity:
        raise ValueError("benchmark inputs changed after planning")
    if not args.execute:
        print(json.dumps({**plan, "plan_sha256": identity}, indent=2))
        return 0
    if args.output is None or args.output.exists():
        parser.error("--execute requires a new --output path")
    if not args.worker:
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *arguments,
                "--worker",
                "--pinned-plan",
                identity,
            ],
            env=os.environ
            | dict.fromkeys(
                (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "TORCHINDUCTOR_COMPILE_THREADS",
                    "RAYON_NUM_THREADS",
                ),
                "2",
            ),
            start_new_session=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            output, errors = child.communicate(timeout=args.timeout_seconds)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.communicate()
            raise
        if child.returncode:
            raise RuntimeError("graph bucket benchmark failed: " + errors[-8000:])
        print(output, end="")
        return 0
    report = _worker(args, plan, identity, config, manifest)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plan_sha256": identity,
                "comparisons": len(report["records"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
