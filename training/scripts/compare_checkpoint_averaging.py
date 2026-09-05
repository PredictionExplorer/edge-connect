#!/usr/bin/env python3
"""Paired raw-versus-EMA diagnostic on one immutable checkpoint; never publishes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Literal, cast

import torch

from scripts.run_lineage_arena import load_candidate
from startrain.arena import ArenaRunner
from startrain.config import ArenaConfig, load_config
from startrain.checkpoint import sha256_file
from startrain.native import validate_native_module
from startrain.runtime import atomic_json


def compare(
    checkpoint: Path,
    *,
    config: ArenaConfig,
    device: torch.device,
    precision: Literal["fp32", "bf16"],
    native_module: object,
) -> dict[str, object]:
    pinned_digest = sha256_file(checkpoint)
    raw, raw_metadata = load_candidate(
        checkpoint, device=device, weights="raw", precision=precision
    )
    ema, ema_metadata = load_candidate(
        checkpoint, device=device, weights="ema", precision=precision
    )
    if (
        raw_metadata["checkpoint_sha256"] != pinned_digest
        or ema_metadata["checkpoint_sha256"] != pinned_digest
        or raw_metadata["step"] != ema_metadata["step"]
        or sha256_file(checkpoint) != pinned_digest
    ):
        raise ValueError(
            "raw and EMA weights must come from the same immutable checkpoint"
        )
    result = ArenaRunner(
        native_module=native_module,
        candidate=raw,
        baseline=ema,
        config=config,
        stable_pair_seeds=True,
        baseline_metadata={"kind": "same_checkpoint_ema", **ema_metadata},
    ).run()
    result.update(
        {
            "result_kind": "checkpoint_averaging_diagnostic",
            "candidate_metadata": raw_metadata,
            "checkpoint_sha256": ema_metadata["checkpoint_sha256"],
            "precision": precision,
            "interpretation": "Positive contrast favors raw weights; this diagnostic does not change EMA or publish a champion.",
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--pairs-per-cell", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("diagnostic output already exists")
        import star_native

        validate_native_module(star_native)
        source = load_config(args.profile)
        arena = replace(
            source.arena,
            balanced_cells=args.balanced,
            promotion_pair_ratios={},
            segment_pairs_per_ring={},
            segment_regression_floor_elo={},
            pairs_per_ring=args.pairs_per_cell,
            minimum_pairs_per_ring=args.pairs_per_cell,
            max_pairs_per_ring=args.pairs_per_cell,
            continuation_pairs_per_ring=None,
            simulations=args.simulations,
        )
        result = compare(
            args.checkpoint,
            config=arena,
            device=torch.device(args.device),
            precision=cast(Literal["fp32", "bf16"], args.precision),
            native_module=star_native,
        )
        atomic_json(args.output, result)
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate": result["candidate"],
                "baseline": result["baseline"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
