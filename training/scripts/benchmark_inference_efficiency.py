#!/usr/bin/env python3
"""Bounded same-checkpoint native-request inference comparison on one GPU.

Run from training/, with an external timeout covering compilation and fixture
generation as well as timed samples, for example:

    PYTHONPATH=. timeout --signal=TERM --kill-after=10s 300s .venv/bin/python \\
      scripts/benchmark_inference_efficiency.py --config /path/profile.yaml \\
      --checkpoint /path/checkpoint.pt --device cuda:7 --compile \\
      --load-context shared --arms baseline bias full

Fresh synthetic positions estimate the cache-miss path. Replayed requests show
the upper bound from perfect reuse, not an expected production cache hit rate.
Each sample evaluates two independent cohorts; all throughput uses their total
rows. Compilation/warmup are reported separately. Shared-GPU measurements are
exploratory; they cannot authorize a performance claim by themselves.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, cast

import torch

from startrain.checkpoint import load_ema_checkpoint, sha256_file
from startrain.config import load_config
from startrain.inference import (
    GraphInferenceAdapter,
    InferenceConfig,
    InferenceResponse,
)
from startrain.inference_batching import BoundedInferenceBroker
from startrain.model import GraphResTNet
from startrain.native import load_star_native
from startrain.topology import SUPPORTED_RINGS, get_topology


def _requests(
    native: Any, *, ring: int, rows: int, seed: int, mode: str, handicap: int, pie: bool
) -> Any:
    random_source = random.Random(seed)
    states = native.StateBatch(ring, rows, mode=mode, handicap=handicap, pie=pie)
    nodes = get_topology(ring).n
    available = [list(range(nodes)) for _ in range(rows)]
    depths = [row % 13 for row in range(rows)]
    for depth in range(max(depths)):
        active = [row for row in range(rows) if depths[row] > depth]
        actions = [
            available[row].pop(random_source.randrange(len(available[row])))
            for row in active
        ]
        states.apply_many(active, actions)
    return native.SearchBatch(
        states, simulations=1, max_considered=2, deterministic_seed=seed
    ).root_requests()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--rings", type=int, nargs="+", default=[4, 10])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("baseline", "pinned", "bias", "cache", "broker", "full"),
        default=["baseline", "full"],
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("fresh", "replayed"),
        default=["fresh", "replayed"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--duration-seconds", type=float, default=45)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument(
        "--load-context",
        choices=("unverified", "shared", "isolated"),
        default="unverified",
    )
    parser.add_argument("--mode", choices=("classic", "double"), default="double")
    parser.add_argument("--handicap", type=int, default=1)
    parser.add_argument("--pie", action="store_true")
    args = parser.parse_args()
    if (
        not 1 <= args.batch_size <= 256
        or not 1 <= args.repeats <= 20
        or not 1 <= args.warmup <= 5
    ):
        parser.error("batch size1..256, repeats1..20, warmup1..5 required")
    if not 0 < args.duration_seconds <= 60:
        parser.error("duration must be in (0,60] seconds")
    if (
        not args.rings
        or any(ring not in SUPPORTED_RINGS for ring in args.rings)
        or len(set(args.rings)) != len(args.rings)
    ):
        parser.error("rings must be unique supported board sizes")
    if not 1 <= args.handicap <= 9 or (args.pie and args.handicap != 1):
        parser.error("invalid handicap/pie combination")
    if (
        not args.arms
        or args.arms[0] != "baseline"
        or len(set(args.arms)) != len(args.arms)
    ):
        parser.error("arms must be unique and begin with baseline for output parity")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this benchmark requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    native = load_star_native(required=True)
    assert native is not None
    config = load_config(args.config)
    cpu_model = GraphResTNet(config.model).eval()
    if cpu_model.parameter_count() != 17_402_775:
        parser.error(
            "benchmark requires the unchanged17,402,775-parameter production model"
        )
    identity = sha256_file(args.checkpoint)
    metadata = load_ema_checkpoint(
        args.checkpoint,
        model=cpu_model,
        map_location="cpu",
        expected_sha256=identity,
        expected_model_config=asdict(config.model),
        expected_game_config=asdict(config.game),
    )
    starting = time.monotonic()
    timed_seconds = 0.0
    records: list[dict[str, object]] = []
    exhausted = False
    for ring in args.rings:
        reference_responses: list[InferenceResponse] | None = None
        fixtures = [
            tuple(
                _requests(
                    native,
                    ring=ring,
                    rows=args.batch_size,
                    seed=913 + ring * 10000 + repeat * 2 + cohort,
                    mode=args.mode,
                    handicap=args.handicap,
                    pie=args.pie,
                )
                for cohort in range(2)
            )
            for repeat in range(args.repeats + 1)
        ]
        for arm in args.arms:
            model = deepcopy(cpu_model).to(device).eval()
            runner = (
                cast(
                    torch.nn.Module,
                    torch.compile(model, dynamic=True, fullgraph=True, mode="default"),
                )
                if args.compile_model
                else model
            )
            cache_enabled = arm in ("cache", "full")
            adapter = GraphInferenceAdapter(
                runner,
                device=device,
                model_identity=identity,
                model_version=identity,
                model_step=int(metadata["step"]),
                homogeneous_relational_bias=arm in ("bias", "full"),
                config=InferenceConfig(
                    precision="bf16",
                    score_utility_weight=0.05,
                    cache_max_entries=10000 if cache_enabled else 0,
                    cache_max_bytes=128 * 1024**2 if cache_enabled else 0,
                    deduplicate=cache_enabled,
                    pinned_transfers=arm in ("pinned", "full"),
                ),
            )
            broker = (
                BoundedInferenceBroker(
                    max_batch_rows=2 * args.batch_size,
                    max_pending_requests=4,
                    max_wait_seconds=0.01,
                )
                if arm in ("broker", "full")
                else None
            )

            def evaluate(
                pair: tuple[Any, ...],
                inference: GraphInferenceAdapter = adapter,
                batching: BoundedInferenceBroker | None = broker,
            ) -> list[InferenceResponse]:
                if batching is None:
                    return [inference.evaluate(request) for request in pair]
                futures = [batching.submit(inference, request) for request in pair]
                output = [future.result(timeout=120) for future in futures]
                if any(
                    not isinstance(response, InferenceResponse) for response in output
                ):
                    raise RuntimeError("unexpected detailed benchmark output")
                return [
                    response
                    for response in output
                    if isinstance(response, InferenceResponse)
                ]

            try:
                warmed = time.monotonic()
                warmed_responses: list[InferenceResponse] = []
                for _ in range(args.warmup):
                    warmed_responses = evaluate(fixtures[0])
                torch.cuda.synchronize(device)
                warmup_seconds = time.monotonic() - warmed
                parity = {
                    "max_abs_policy_difference": 0.0,
                    "max_abs_value_difference": 0.0,
                }
                if arm == "baseline":
                    reference_responses = warmed_responses
                else:
                    assert reference_responses is not None
                    for expected, actual in zip(
                        reference_responses, warmed_responses, strict=True
                    ):
                        if (
                            expected.tokens != actual.tokens
                            or expected.policy_offsets != actual.policy_offsets
                        ):
                            raise RuntimeError(
                                "inference arm changed token/action routing"
                            )
                        for field, key in (
                            ("policy_logits", "max_abs_policy_difference"),
                            ("values", "max_abs_value_difference"),
                        ):
                            expected_tensor = torch.tensor(getattr(expected, field))
                            actual_tensor = torch.tensor(getattr(actual, field))
                            parity[key] = max(
                                parity[key],
                                (expected_tensor - actual_tensor).abs().max().item(),
                            )
                            torch.testing.assert_close(
                                actual_tensor,
                                expected_tensor,
                                atol=0.03 if field == "policy_logits" else 0.002,
                                rtol=0.02,
                            )
                for workload in args.workloads:
                    times = []
                    before = adapter.metrics_snapshot()
                    torch.cuda.reset_peak_memory_stats(device)
                    for repeat in range(args.repeats):
                        if timed_seconds >= args.duration_seconds:
                            exhausted = True
                            break
                        pair = (
                            fixtures[repeat + 1] if workload == "fresh" else fixtures[0]
                        )
                        torch.cuda.synchronize(device)
                        tick = time.monotonic()
                        responses = evaluate(pair)
                        torch.cuda.synchronize(device)
                        elapsed = time.monotonic() - tick
                        timed_seconds += elapsed
                        times.append(elapsed)
                        if any(
                            len(response.tokens) != args.batch_size
                            for response in responses
                        ):
                            raise RuntimeError("benchmark lost response rows")
                    if times:
                        record = {
                            "ring": ring,
                            "arm": arm,
                            "workload": workload,
                            "cohort_rows": args.batch_size,
                            "cohorts_per_sample": 2,
                            "median_seconds": statistics.median(times),
                            "rows_per_second": 2
                            * args.batch_size
                            * len(times)
                            / sum(times),
                            "samples_seconds": times,
                            "compile_and_warmup_seconds": warmup_seconds,
                            "warmup_output_parity": parity,
                            "metrics_delta": asdict(
                                adapter.metrics_snapshot().delta(before)
                            ),
                            "efficiency": adapter.efficiency_snapshot(),
                            "broker": broker.metrics_snapshot()
                            if broker is not None
                            else None,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                                device
                            ),
                        }
                        records.append(record)
                        print(json.dumps(record), flush=True)
                    if exhausted:
                        break
            finally:
                if broker is not None:
                    broker.shutdown()
                adapter.close()
                del evaluate, adapter, runner, model
                gc.collect()
                torch.cuda.empty_cache()
            if exhausted:
                break
        if exhausted:
            break
    print(
        json.dumps(
            {
                "benchmark": "native-request-inference-efficiency",
                "evidence_status": "exploratory"
                if args.load_context != "isolated"
                else "isolated-microbenchmark",
                "checkpoint_sha256": identity,
                "model_parameters": cpu_model.parameter_count(),
                "config_sha256": sha256_file(args.config),
                "compiled": args.compile_model,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "load_context": args.load_context,
                "variant": {
                    "mode": args.mode,
                    "handicap": args.handicap,
                    "pie": args.pie,
                },
                "timed_seconds": timed_seconds,
                "total_wall_seconds": time.monotonic() - starting,
                "budget_exhausted": exhausted,
                "records": records,
                "limitation": "Synthetic fresh/replayed workload; measure actual actor throughput and cache hit rate before adoption.",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
