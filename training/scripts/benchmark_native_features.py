#!/usr/bin/env python3
"""Measure native feature exports with exact Python-oracle verification.

Use --native-only to compare old/new native binaries without timing Python
conversion. Keep all arguments and RAYON_NUM_THREADS fixed between runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import fields

import torch

from startrain.features import EncodedBatch, encode_batch
from startrain.features_v3 import encode_legacy_batch
from startrain.native import (
    encode_native_state_data,
    load_star_native,
    positions_from_native,
)
from startrain.topology import SUPPORTED_RINGS


def _time(operation, iterations: int) -> list[float]:
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - started)
    return durations


def _assert_equal(actual: EncodedBatch, expected: EncodedBatch) -> None:
    for field in fields(EncodedBatch):
        torch.testing.assert_close(
            getattr(actual, field.name),
            getattr(expected, field.name),
            rtol=0,
            atol=0,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rings", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--schema-version", type=int, choices=(3, 4), default=4)
    parser.add_argument("--mode", choices=("classic", "double"), default="double")
    parser.add_argument("--handicap", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--pie", action="store_true")
    parser.add_argument("--occupancy", type=float, default=0.0)
    parser.add_argument("--native-threads", type=int)
    parser.add_argument("--native-only", action="store_true")
    arguments = parser.parse_args()
    if (
        arguments.rings not in SUPPORTED_RINGS
        or arguments.batch_size <= 0
        or arguments.warmup < 0
        or arguments.iterations <= 0
        or not math.isfinite(arguments.occupancy)
        or not 0 <= arguments.occupancy <= 1
        or (arguments.native_threads is not None and arguments.native_threads <= 0)
        or (arguments.pie and arguments.handicap != 1)
    ):
        raise SystemExit("invalid benchmark dimensions")

    native = load_star_native(required=True)
    assert native is not None
    if arguments.native_threads is not None:
        native.configure_rayon_threads(arguments.native_threads)
    states = native.StateBatch(
        arguments.rings,
        arguments.batch_size,
        mode=arguments.mode,
        handicap=arguments.handicap,
        pie=arguments.pie,
    )
    placements = int(states.node_count * arguments.occupancy)
    for row in range(arguments.batch_size):
        states.apply_many(
            [row] * placements,
            [(node + row) % states.node_count for node in range(placements)],
        )

    def native_export():
        return states.feature_data(schema_version=arguments.schema_version)

    def rust_path() -> EncodedBatch:
        return encode_native_state_data(
            states.data(), schema_version=arguments.schema_version
        )

    def python_path() -> EncodedBatch:
        data = states.data()
        encoder = encode_batch if arguments.schema_version == 4 else encode_legacy_batch
        return encoder(positions_from_native(data))

    _assert_equal(rust_path(), python_path())
    operation = native_export if arguments.native_only else rust_path
    for _ in range(arguments.warmup):
        operation()
        if not arguments.native_only:
            python_path()
    rust = _time(operation, arguments.iterations)
    rust_median = statistics.median(rust)
    exported = native_export()
    fingerprint = hashlib.sha256()
    feature_bytes = 0
    for name in (
        "rings",
        "node_features",
        "global_features",
        "node_mask",
        "legal_action_mask",
        "score_components",
        "node_owner",
        "alive_stones",
    ):
        data = bytes(getattr(exported, name))
        fingerprint.update(len(data).to_bytes(8, "little"))
        fingerprint.update(data)
        feature_bytes += len(data)
    result = {
        "schema_version": 1,
        "benchmark": f"native-schema-v{arguments.schema_version}-feature-batch",
        "timed_operation": "native_export"
        if arguments.native_only
        else "state_adapter",
        "feature_schema_version": arguments.schema_version,
        "feature_sha256": fingerprint.hexdigest(),
        "feature_bytes": feature_bytes,
        "rings": arguments.rings,
        "nodes": states.node_count,
        "mode": arguments.mode,
        "handicap": arguments.handicap,
        "pie": arguments.pie,
        "placements": placements,
        "native_threads": native.rayon_num_threads(),
        "batch_size": arguments.batch_size,
        "warmup": arguments.warmup,
        "iterations": arguments.iterations,
        "rust_median_ms": rust_median * 1_000.0,
        "exact_parity": True,
    }
    if not arguments.native_only:
        python_median = statistics.median(_time(python_path, arguments.iterations))
        result.update(
            python_median_ms=python_median * 1_000.0,
            speedup=python_median / rust_median,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
