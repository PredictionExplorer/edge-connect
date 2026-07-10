#!/usr/bin/env python3
"""Measure the real Python→native-feature→GPU inference boundary.

This benchmark intentionally includes native-state decoding, schema-v2 feature
construction, host-to-device transfer, model execution, and legal-logit copies.
It is therefore a more useful actor-capacity gate than a model-only microbenchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from startrain.config import load_config
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.model import GraphResTNet
from startrain.native import load_star_native
from startrain.training import maybe_compile_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rings", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--minimum-leaves-per-second",
        type=float,
        default=5_000.0,
        help="exit nonzero when the measured boundary misses this gate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.batch_size <= 0 or arguments.warmup < 0 or arguments.iterations <= 0:
        raise SystemExit(
            "batch-size/iterations must be positive and warmup non-negative"
        )
    device = torch.device(arguments.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("hardware preflight requires a CUDA device")
    torch.cuda.set_device(device)

    experiment = load_config(arguments.config)
    native = load_star_native(required=True)
    assert native is not None
    model = GraphResTNet(experiment.model).to(device).eval()
    inference_model = maybe_compile_model(
        model,
        enabled=experiment.train.compile,
        dynamic=True,
        fullgraph=True,
    )
    evaluator = GraphInferenceAdapter(
        inference_model,
        device=device,
        config=InferenceConfig(
            precision=experiment.train.precision,
            score_utility_weight=experiment.selfplay.score_utility_weight,
            initial_pass_logit_penalty=(experiment.selfplay.initial_pass_logit_penalty),
        ),
        model_version="hardware-preflight",
    )
    states = native.StateBatch(arguments.rings, arguments.batch_size)
    search = native.SearchBatch(
        states,
        simulations=1,
        max_considered=1,
        c_visit=experiment.selfplay.c_visit,
        c_scale=experiment.selfplay.c_scale,
        deterministic_seed=experiment.selfplay.seed,
    )
    requests = search.root_requests()
    if len(requests) != arguments.batch_size:
        raise RuntimeError("native root request count disagrees with benchmark batch")

    for _ in range(arguments.warmup):
        evaluator.evaluate(requests)
    torch.cuda.synchronize(device)

    durations = []
    for _ in range(arguments.iterations):
        started = time.perf_counter()
        evaluator.evaluate(requests)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)

    total_leaves = arguments.batch_size * arguments.iterations
    total_seconds = sum(durations)
    leaves_per_second = total_leaves / total_seconds
    sorted_durations = sorted(durations)
    p95_index = min(len(sorted_durations) - 1, int(0.95 * len(sorted_durations)))
    payload = {
        "schema_version": 1,
        "benchmark": "native-feature-model-inference-boundary",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "rings": arguments.rings,
        "batch_size": arguments.batch_size,
        "iterations": arguments.iterations,
        "model_parameters": model.parameter_count(),
        "feature_path": evaluator.last_feature_path,
        "feature_path_counts": evaluator.feature_path_counts,
        "mean_batch_ms": 1_000.0 * statistics.mean(durations),
        "p95_batch_ms": 1_000.0 * sorted_durations[p95_index],
        "leaf_evaluations_per_second": leaves_per_second,
        "minimum_leaf_evaluations_per_second": (arguments.minimum_leaves_per_second),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "passed": leaves_per_second >= arguments.minimum_leaves_per_second,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
