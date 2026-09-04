#!/usr/bin/env python
"""Seed a variant-capable (rules v3) run root from the previous lineage.

The legacy champion becomes a frozen teacher: every position of the legacy
replay store is upgraded to replay schema v5 and labelled with the teacher's
soft policy, outcome, and score-margin distributions, then committed to the
new run's replay store under a fresh run identity. Launch the Stage A profile
against that root afterwards; ``loss.teacher_*`` distils the teacher while
new self-play data phases the transferred shards out.

The legacy store is opened read-only and never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from startrain.lineage import (  # noqa: E402
    LineageTransferError,
    list_legacy_shards,
    load_legacy_teacher,
    new_run_identity,
    resolve_legacy_champion,
    select_recent_legacy_shards,
    transfer_lineage,
    transfer_lineage_parallel,
    write_transfer_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    teacher = parser.add_mutually_exclusive_group(required=True)
    teacher.add_argument(
        "--legacy-checkpoint",
        type=Path,
        help="EMA checkpoint of the previous lineage's champion (the teacher)",
    )
    teacher.add_argument(
        "--legacy-champion",
        type=Path,
        help="previous lineage learner/champion.json pointer resolving to the teacher",
    )
    parser.add_argument(
        "--legacy-replay-root",
        type=Path,
        required=True,
        help="previous lineage replay root containing manifest.sqlite3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new run root; its replay store receives the labelled shards",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-family", required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="keep only the most recent legacy samples up to this count",
    )
    parser.add_argument(
        "--max-samples-per-ring",
        type=int,
        default=None,
        help="keep only the most recent legacy samples of every ring up to this count",
    )
    parser.add_argument(
        "--rings",
        default=None,
        help="comma-separated rings to transfer (default: every ring)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="spawn this many labelling processes, each with its own teacher copy",
    )
    parser.add_argument(
        "--replay-subdirectory",
        default="replay",
        help="replay directory name under the output root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rings = (
            tuple(int(value) for value in str(args.rings).split(","))
            if args.rings
            else None
        )
        checkpoint = (
            args.legacy_checkpoint
            if args.legacy_checkpoint is not None
            else resolve_legacy_champion(args.legacy_champion)
        )
        if args.workers <= 0:
            raise LineageTransferError("--workers must be positive")
        shards = select_recent_legacy_shards(
            list_legacy_shards(args.legacy_replay_root, rings=rings),
            max_samples=args.max_samples,
            max_samples_per_ring=args.max_samples_per_ring,
        )
        if not shards:
            raise LineageTransferError("no legacy shards matched the transfer request")
        args.output_root.mkdir(parents=True, exist_ok=True)
        identity = new_run_identity(
            args.output_root,
            run_id=args.run_id,
            generation_family=args.generation_family,
        )
        if args.workers == 1:
            teacher = load_legacy_teacher(checkpoint, device=torch.device(args.device))
            report = transfer_lineage(
                teacher=teacher,
                legacy_shards=shards,
                replay_root=args.output_root / args.replay_subdirectory,
                identity=identity,
                batch_size=args.batch_size,
                progress=lambda **fields: print(json.dumps(fields), file=sys.stderr),
            )
        else:
            report = transfer_lineage_parallel(
                checkpoint=checkpoint,
                device=args.device,
                legacy_shards=shards,
                replay_root=args.output_root / args.replay_subdirectory,
                identity=identity,
                workers=args.workers,
                batch_size=args.batch_size,
            )
        destination = write_transfer_report(
            args.output_root / "lineage-transfer.json", report
        )
    except (LineageTransferError, ValueError, OSError) as error:
        print(f"lineage transfer failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report": str(destination),
                "committed_shards": report["committed_shards"],
                "committed_samples": report["committed_samples"],
                "teacher": report["teacher"]["identity"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
