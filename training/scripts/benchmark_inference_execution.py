#!/usr/bin/env python3
"""Bounded, paired whole-model inference comparisons against one frozen EMA.

The default prints a pinned plan. --execute runs an isolated worker with a hard
deadline. No replay, model pointer, service or training configuration is modified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rings", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pinned-plan", help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if (
        not args.rings
        or any(ring not in (4, 6, 8, 10) for ring in args.rings)
        or len(set(args.rings)) != len(args.rings)
        or not args.batch_sizes
        or len(args.batch_sizes) > 6
        or len(set(args.batch_sizes)) != len(args.batch_sizes)
        or any(not 1 <= batch <= 256 for batch in args.batch_sizes)
        or not 2 <= args.repeats <= 12
        or not 1 <= args.iterations <= 16
        or not 1 <= args.warmups <= 10
        or not math.isfinite(args.timeout_seconds)
        or not 1 <= args.timeout_seconds <= 900
    ):
        parser.error("invalid bounded benchmark dimensions or deadline")
    from startrain.checkpoint import load_model_manifest
    from startrain.config import load_config
    import startrain.model as model_module

    config = load_config(args.config)
    manifest = load_model_manifest(args.checkpoint)
    plan = {
        "schema_version": 1,
        "model_identity": manifest.model_identity,
        "manifest_sha256": manifest.manifest_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model_source_sha256": hashlib.sha256(
            Path(model_module.__file__).read_bytes()
        ).hexdigest(),
        "precision": config.train.precision,
        "device": args.device,
        "rings": args.rings,
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "compile": args.compile_model,
        "arms": ["preserve-gather", "compact-gather"],
        "scope": "paired whole-model latency and numerical comparison; not Elo or training throughput",
    }
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if args.pinned_plan is not None and args.pinned_plan != identity:
        raise ValueError("benchmark inputs changed after planning")
    if not args.execute:
        print(json.dumps({**plan, "plan_sha256": identity}, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("--execute requires a new --output path")
    if not args.worker:
        child_env = os.environ | {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "RAYON_NUM_THREADS": "2",
        }
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *arguments,
                "--worker",
                "--pinned-plan",
                identity,
            ],
            env=child_env,
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
            raise RuntimeError("inference benchmark failed: " + errors[-8000:])
        print(output, end="")
        return

    import torch
    from startrain.checkpoint import load_ema_checkpoint
    from startrain.native import encode_native_feature_data, load_star_native
    from startrain.inference import resolve_precision

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(0.15, device)
        torch.cuda.reset_peak_memory_stats(device)
    native = load_star_native(required=True)
    assert native is not None
    precision = resolve_precision(config.train.precision, device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        precision
    ]
    models = []
    for compact in (False, True):
        model = model_module.GraphResTNet(config.model).eval().to(device)
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
        model.set_compact_inference_gather(compact)
        models.append(
            (
                model,
                torch.compile(model, dynamic=True, fullgraph=True)
                if args.compile_model
                else model,
            )
        )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    records = []
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32),
    ):
        for ring in args.rings:
            biases = [
                model.prepare_inference_relational_bias(ring, dtype=dtype)
                for model, _ in models
            ]
            for size in args.batch_sizes:
                if time.monotonic() - started >= args.timeout_seconds:
                    raise TimeoutError("benchmark deadline expired")
                states = native.StateBatch(ring, size)
                nodes = states.node_count
                for ply in range(8):
                    states.apply_many(
                        list(range(size)),
                        [(row * 37 + ply * 19) % nodes for row in range(size)],
                    )
                inputs = encode_native_feature_data(
                    states.feature_data(), source="native_request"
                ).to(device)

                def forward(arm):
                    return models[arm][1](
                        *inputs.model_args(),
                        homogeneous_ring=ring,
                        inference_relation_bias=biases[arm],
                    )

                warm_started = time.monotonic()
                for arm in (0, 1):
                    for _ in range(args.warmups):
                        forward(arm)
                sync()
                warm_seconds = time.monotonic() - warm_started
                before, after = forward(0), forward(1)
                difference = {}
                for name, expected, actual in zip(
                    before._fields, before, after, strict=True
                ):
                    if not bool(torch.isfinite(actual).all()):
                        raise ValueError("nonfinite benchmark prediction")
                    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
                    difference[name] = float(
                        (actual.float() - expected.float()).abs().max().cpu()
                    )
                samples = [[], []]
                for repeat in range(args.repeats):
                    for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                        sync()
                        tick = time.perf_counter()
                        for _ in range(args.iterations):
                            forward(arm)
                        sync()
                        samples[arm].append(
                            (time.perf_counter() - tick) / args.iterations
                        )
                ratios = [old / new for old, new in zip(*samples, strict=True)]
                records.append(
                    {
                        "ring": ring,
                        "batch_size": size,
                        "warmup_and_compile_seconds": warm_seconds,
                        "latency_seconds": samples,
                        "paired_speedup_median": statistics.median(ratios),
                        "paired_speedup_min": min(ratios),
                        "paired_speedup_max": max(ratios),
                        "max_absolute_output_differences": difference,
                    }
                )
    report = {
        "plan": plan,
        "plan_sha256": identity,
        "records": records,
        "elapsed_seconds": time.monotonic() - started,
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plan_sha256": identity,
                "comparisons": len(records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
