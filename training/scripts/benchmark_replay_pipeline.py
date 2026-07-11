#!/usr/bin/env python3
"""Benchmark decode-once replay loading and selected-row materialization."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from startrain.replay import decode_replay_shard

SCHEMA_VERSION = 1
BENCHMARK_NAME = "replay-decode-selected-rows"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    return parser


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def benchmark_replay_shard(
    shard_path: str | Path,
    *,
    rows: int,
    repeats: int,
) -> dict[str, object]:
    path = Path(shard_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"replay shard does not exist: {path}")
    if rows <= 0 or repeats <= 0:
        raise ValueError("rows and repeats must be positive")

    decode_seconds: list[float] = []
    materialize_seconds: list[float] = []
    sample_count: int | None = None
    selected_rows: int | None = None
    member_count: int | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        decoded = decode_replay_shard(path)
        decode_seconds.append(time.perf_counter() - started)
        sample_count = len(decoded)
        member_count = len(decoded.arrays) + 1
        selected_rows = min(rows, sample_count)
        indices = tuple(range(selected_rows))
        started = time.perf_counter()
        samples = decoded.samples(indices)
        materialize_seconds.append(time.perf_counter() - started)
        if len(samples) != selected_rows:
            raise RuntimeError("selected-row materialization returned the wrong count")

    assert sample_count is not None
    assert selected_rows is not None
    assert member_count is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "shard": str(path),
        "compressed_bytes": path.stat().st_size,
        "sample_count": sample_count,
        "selected_rows": selected_rows,
        "npz_members_loaded_per_repeat": member_count,
        "decode_seconds": _stats(decode_seconds),
        "selected_row_materialization_seconds": _stats(materialize_seconds),
        "selected_rows_per_second": (
            selected_rows / statistics.median(materialize_seconds)
            if statistics.median(materialize_seconds)
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = benchmark_replay_shard(
            arguments.shard,
            rows=arguments.rows,
            repeats=arguments.repeats,
        )
    except (OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark": BENCHMARK_NAME,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps({**result, "status": "ok"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
