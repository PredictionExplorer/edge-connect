#!/usr/bin/env python3
"""Fail closed when configured NVIDIA GPUs are unsafe for training."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

from startrain.config import load_config
from startrain.device import normalize_device_string
from startrain.hardware_health import query_gpu_health, unhealthy_reasons


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--allow-non-h100",
        action="store_true",
        help="ignore the profile's required GPU model for development hosts",
    )
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    health = config.orchestration.hardware_health
    require_gpu_model = None if arguments.allow_non_h100 else health.require_gpu_model
    worker_device = normalize_device_string(config.orchestration.device)
    promotion_device = normalize_device_string(config.orchestration.promotion.device)
    if not worker_device.startswith("cuda") and not promotion_device.startswith("cuda"):
        report = {
            "schema_version": 1,
            "captured_ns": time.time_ns(),
            "hostname": socket.gethostname(),
            "config": str(arguments.config.resolve()),
            "healthy": True,
            "skipped": f"no CUDA workers (devices: {worker_device},"
            f" {promotion_device}); NVIDIA health gate does not apply",
            "gpus": [],
        }
        if arguments.output is not None:
            _atomic_json(arguments.output, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    expected_indices = sorted(
        {
            *(gpu.gpu_id for gpu in config.orchestration.gpus),
            config.orchestration.promotion.gpu_id,
        }
    )
    try:
        report = query_gpu_health(
            expected_indices=expected_indices,
            require_gpu_model=require_gpu_model,
            fail_on_aggregate_uncorrectable=health.fail_on_aggregate_uncorrectable,
            timeout=arguments.timeout_seconds,
        )
        report.update(
            {
                "captured_ns": time.time_ns(),
                "hostname": socket.gethostname(),
                "config": str(arguments.config.resolve()),
            }
        )
    except (RuntimeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "captured_ns": time.time_ns(),
            "hostname": socket.gethostname(),
            "config": str(arguments.config.resolve()),
            "healthy": False,
            "query_error": f"{type(exc).__name__}: {exc}",
            "expected_indices": expected_indices,
            "gpus": [],
        }

    if arguments.output is not None:
        _atomic_json(arguments.output, report)
    print(json.dumps(report, sort_keys=True))
    if report.get("healthy") is True:
        return 0
    reasons = unhealthy_reasons(report)
    if reasons:
        print("GPU health gate failed: " + "; ".join(reasons), file=sys.stderr)
    else:
        print(
            "GPU health gate failed: "
            + str(report.get("query_error", "unknown error")),
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
