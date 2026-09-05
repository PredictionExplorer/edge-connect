#!/usr/bin/env python3
"""Check evidence-based Stage B readiness without changing training state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from startrain.config import load_config
from startrain.runtime import atomic_json
from startrain.transfer_gate import transfer_readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--lineage-result", type=Path)
    parser.add_argument("--minimum-elo-lower", type=float, default=-15.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--purpose",
        choices=("legacy-certification", "curriculum"),
        default="legacy-certification",
        help="Curriculum readiness checks teacher-window turnover; it never certifies legacy playing strength.",
    )
    args = parser.parse_args(argv)
    try:
        report = transfer_readiness(
            args.run_root,
            load_config(args.profile),
            profile_path=args.profile,
            lineage_result=args.lineage_result,
            minimum_elo_lower=args.minimum_elo_lower,
        )
        report["requested_purpose"] = args.purpose
        report["activation_ready"] = (
            report["curriculum_ready"]
            if args.purpose == "curriculum"
            else report["legacy_strength_certified"]
        )
        if args.output is not None:
            target = args.output.resolve()
            root = args.run_root.resolve()
            inputs = {args.profile.resolve()}
            if args.lineage_result is not None:
                inputs.add(args.lineage_result.resolve())
            if target in inputs or (
                target.is_relative_to(root)
                and target != root / "status/transfer-readiness.json"
            ):
                raise ValueError("readiness output cannot replace an input")
            atomic_json(args.output, report)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["activation_ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
