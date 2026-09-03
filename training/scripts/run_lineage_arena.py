#!/usr/bin/env python
"""Cross-schema arena: a variant-capable candidate against the legacy champion.

The candidate (rules v3, feature schema v4) and the previous lineage's champion
(rules v2, feature schema v3) play standard Double *Star pairs under one search
budget. The legacy side evaluates through its frozen v3 encoder, so the match
measures exactly how much of the old lineage's strength the new network has
recovered. Only the standard segment is played: the legacy network never saw
classic, handicap, or pie games.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from startrain.arena import ArenaRunner  # noqa: E402
from startrain.checkpoint import (  # noqa: E402
    load_ema_checkpoint,
    normalize_model_config,
    sha256_file,
)
from startrain.config import ArenaConfig, ConfigError  # noqa: E402
from startrain.contracts import (  # noqa: E402
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_SCHEMA_HASH,
    LEGACY_FEATURE_SCHEMA_VERSION,
    LEGACY_RULES_HASH_WIRE,
    RULES_HASH_WIRE,
)
from startrain.inference import GraphInferenceAdapter, InferenceConfig  # noqa: E402
from startrain.lineage import LineageTransferError, load_legacy_teacher  # noqa: E402
from startrain.model import GraphResTNet, ModelConfig  # noqa: E402
from startrain.native import validate_native_module  # noqa: E402
from startrain.runtime import atomic_json  # noqa: E402

RESULT_KIND = "lineage_crossplay"
EVALUATION_MODE = "cross_schema"


class LineageArenaError(RuntimeError):
    pass


def load_candidate(
    checkpoint: Path, *, device: torch.device
) -> tuple[GraphInferenceAdapter, dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise LineageArenaError("candidate checkpoint configuration is missing")
    raw_model = payload["config"].get("model")
    raw_game = payload["config"].get("game")
    if not isinstance(raw_model, dict) or not isinstance(raw_game, dict):
        raise LineageArenaError(
            "candidate checkpoint model/game configuration is missing"
        )
    config = ModelConfig(**normalize_model_config(raw_model))  # type: ignore[arg-type]
    if config.is_legacy:
        raise LineageArenaError("candidate must use the variant-capable feature schema")
    model = GraphResTNet(config)
    metadata = load_ema_checkpoint(
        checkpoint,
        model=model,
        expected_model_config=raw_model,
        expected_game_config=raw_game,
        map_location="cpu",
    )
    model.to(device).eval()
    digest = sha256_file(checkpoint)
    identity = f"sha256-{digest}"
    adapter = GraphInferenceAdapter(
        model,
        device=device,
        config=InferenceConfig(precision="fp32"),
        model_version=identity,
        model_step=int(metadata["step"]),
        model_identity=identity,
    )
    return adapter, {
        "identity": identity,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "step": int(metadata["step"]),
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
        "model_config": {
            name: getattr(config, name) for name in config.__dataclass_fields__
        },
    }


def run_lineage_arena(
    *,
    native_module: object,
    candidate_checkpoint: Path,
    legacy_checkpoint: Path,
    rings: tuple[int, ...],
    pairs_per_ring: int,
    simulations: int,
    max_considered: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    candidate, candidate_metadata = load_candidate(candidate_checkpoint, device=device)
    teacher = load_legacy_teacher(legacy_checkpoint, device=device)
    baseline = GraphInferenceAdapter(
        teacher.model,
        device=device,
        config=InferenceConfig(
            precision="fp32", feature_schema_version=LEGACY_FEATURE_SCHEMA_VERSION
        ),
        model_version=teacher.identity,
        model_step=teacher.step,
        model_identity=teacher.identity,
    )
    try:
        config = ArenaConfig(
            rings=rings,
            pairs_per_ring=pairs_per_ring,
            minimum_pairs_per_ring=pairs_per_ring,
            max_pairs_per_ring=pairs_per_ring,
            simulations=simulations,
            max_considered=max_considered,
            seed=seed,
            bootstrap_samples=2_000,
        )
    except ConfigError as error:
        raise LineageArenaError(str(error)) from error
    started_ns = time.time_ns()
    result = ArenaRunner(
        native_module=native_module,
        candidate=candidate,
        baseline=baseline,
        config=config,
        baseline_metadata={
            "kind": "legacy_champion",
            "rules_hash": LEGACY_RULES_HASH_WIRE,
            "feature_schema_version": LEGACY_FEATURE_SCHEMA_VERSION,
            "feature_schema_hash": f"{LEGACY_FEATURE_SCHEMA_HASH:016x}",
            "checkpoint": str(teacher.checkpoint),
            "checkpoint_sha256": teacher.checkpoint_sha256,
            "step": teacher.step,
            "feature_path": baseline.last_feature_path,
        },
    ).run()
    if baseline.feature_path_counts["python-legacy"] == 0 or any(
        baseline.feature_path_counts[path] for path in ("rust", "python")
    ):
        raise LineageArenaError(
            "legacy baseline did not evaluate through the v3 encoder"
        )
    if candidate.feature_path_counts["python-legacy"]:
        raise LineageArenaError("candidate evaluated through the legacy encoder")
    result["result_kind"] = RESULT_KIND
    result["evaluation_mode"] = EVALUATION_MODE
    result["candidate_metadata"] = candidate_metadata
    result["started_ns"] = started_ns
    result["segment"] = "standard"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rings", default="4,6,8,10")
    parser.add_argument("--pairs-per-ring", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--max-considered", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import star_native

        validate_native_module(star_native)
        result = run_lineage_arena(
            native_module=star_native,
            candidate_checkpoint=args.candidate_checkpoint,
            legacy_checkpoint=args.legacy_checkpoint,
            rings=tuple(int(value) for value in str(args.rings).split(",")),
            pairs_per_ring=args.pairs_per_ring,
            simulations=args.simulations,
            max_considered=args.max_considered,
            seed=args.seed,
            device=torch.device(args.device),
        )
        if args.output.exists():
            raise LineageArenaError(f"output already exists: {args.output}")
        atomic_json(args.output, result)
    except (LineageArenaError, LineageTransferError, ValueError, OSError) as error:
        print(f"lineage arena failed: {error}", file=sys.stderr)
        return 1
    aggregate = result.get("aggregate")
    promotion = result.get("promotion")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate": result["candidate"],
                "baseline": result["baseline"],
                "elo_difference": (
                    aggregate.get("elo_difference")
                    if isinstance(aggregate, dict)
                    else None
                ),
                "decision": (
                    promotion.get("decision") if isinstance(promotion, dict) else None
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
