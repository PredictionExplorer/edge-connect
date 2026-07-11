#!/usr/bin/env python3
"""Compare Rust-native schema-v3 batches with the Python oracle path."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import fields

import torch

from startrain.features import EncodedBatch, encode_batch
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
    arguments = parser.parse_args()
    if (
        arguments.rings not in SUPPORTED_RINGS
        or arguments.batch_size <= 0
        or arguments.warmup < 0
        or arguments.iterations <= 0
    ):
        raise SystemExit("invalid benchmark dimensions")

    native = load_star_native(required=True)
    assert native is not None
    states = native.StateBatch(arguments.rings, arguments.batch_size)

    def rust_path() -> EncodedBatch:
        return encode_native_state_data(states.data())

    def python_path() -> EncodedBatch:
        data = states.data()
        return encode_batch(positions_from_native(data))

    _assert_equal(rust_path(), python_path())
    for _ in range(arguments.warmup):
        rust_path()
        python_path()
    rust = _time(rust_path, arguments.iterations)
    python = _time(python_path, arguments.iterations)
    rust_median = statistics.median(rust)
    python_median = statistics.median(python)
    result = {
        "schema_version": 1,
        "benchmark": "native-schema-v3-feature-batch",
        "rings": arguments.rings,
        "batch_size": arguments.batch_size,
        "iterations": arguments.iterations,
        "rust_median_ms": rust_median * 1_000.0,
        "python_median_ms": python_median * 1_000.0,
        "speedup": python_median / rust_median,
        "exact_parity": True,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
