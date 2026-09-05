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
from dataclasses import asdict
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Literal

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from startrain.arena import ArenaRunner  # noqa: E402
from startrain.checkpoint import (  # noqa: E402
    load_ema_checkpoint,
    load_checkpoint,
    normalize_model_config,
    sha256_file,
)
from startrain.config import ArenaConfig, ConfigError, ExperimentConfig, load_config  # noqa: E402
from startrain.contracts import (  # noqa: E402
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_SCHEMA_HASH,
    LEGACY_FEATURE_SCHEMA_VERSION,
    LEGACY_RULES_HASH_WIRE,
    RULES_HASH_WIRE,
)
from startrain.inference import GraphInferenceAdapter, InferenceConfig  # noqa: E402
from startrain.lineage import (  # noqa: E402
    LineageTransferError,
    load_legacy_teacher,
    resolve_legacy_champion,
)
from startrain.model import GraphResTNet, ModelConfig  # noqa: E402
from startrain.native import validate_native_module  # noqa: E402
from startrain.runtime import atomic_json  # noqa: E402
from startrain.training import maybe_compile_model  # noqa: E402

RESULT_KIND = "lineage_crossplay"
EVALUATION_MODE = "cross_schema"


class LineageArenaError(RuntimeError):
    pass


def load_candidate(
    checkpoint: Path,
    *,
    device: torch.device,
    weights: Literal["ema", "raw"] = "ema",
    precision: Literal["fp32", "bf16"] = "fp32",
    inference_config: InferenceConfig | None = None,
    homogeneous_relational_bias: bool = False,
    compile_model: bool = False,
    compile_dynamic: bool = True,
    compile_mode: str = "default",
) -> tuple[GraphInferenceAdapter, dict[str, object]]:
    if weights not in ("ema", "raw"):
        raise LineageArenaError("candidate weights must be ema or raw")
    digest = sha256_file(checkpoint)
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
    loader = load_ema_checkpoint if weights == "ema" else load_checkpoint
    metadata = loader(
        checkpoint,
        model=model,
        expected_model_config=raw_model,
        expected_game_config=raw_game,
        map_location="cpu",
        expected_sha256=digest,
    )
    model.to(device).eval()
    if sha256_file(checkpoint) != digest:
        raise LineageArenaError("candidate checkpoint changed while loading")
    selected_digest = (
        digest
        if weights == "ema"
        else hashlib.sha256(
            f"startrain-weights:raw:{digest}".encode("ascii")
        ).hexdigest()
    )
    identity = f"sha256-{selected_digest}"
    options = inference_config or InferenceConfig(precision=precision)
    if (
        options.feature_schema_version != FEATURE_SCHEMA_VERSION
        or options.precision != precision
    ):
        raise LineageArenaError(
            "candidate inference schema/precision differs from its explicit contract"
        )
    inference_model = maybe_compile_model(
        model,
        enabled=compile_model,
        dynamic=compile_dynamic,
        fullgraph=True,
        mode=compile_mode,
    )
    adapter = GraphInferenceAdapter(
        inference_model,
        device=device,
        config=options,
        model_version=identity,
        model_step=int(metadata["step"]),
        model_identity=identity,
        homogeneous_relational_bias=homogeneous_relational_bias,
    )
    return adapter, {
        "identity": identity,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "step": int(metadata["step"]),
        "weights": weights,
        "precision": precision,
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
        "model_config": {
            name: getattr(config, name) for name in config.__dataclass_fields__
        },
    }


def _profile_inference_config(
    profile: ExperimentConfig | None, *, schema: int
) -> InferenceConfig:
    if profile is None:
        return InferenceConfig(precision="fp32", feature_schema_version=schema)
    flags = profile.orchestration.model_refresh.inference
    return InferenceConfig(
        precision="fp32",
        feature_schema_version=schema,
        cache_max_entries=flags.cache_max_entries,
        cache_max_bytes=flags.cache_max_bytes,
        deduplicate=flags.deduplicate,
        pinned_transfers=flags.pinned_transfers,
        pinned_buffer_slots=flags.pinned_buffer_slots,
    )


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
    inference_profile: ExperimentConfig | None = None,
) -> dict[str, object]:
    refresh = (
        inference_profile.orchestration.model_refresh
        if inference_profile is not None
        else None
    )
    compile_options = {
        "compile_model": inference_profile.train.compile
        if inference_profile is not None
        else False,
        "compile_dynamic": refresh.inference_compile_dynamic
        if refresh is not None
        else True,
        "compile_mode": refresh.inference_compile_mode
        if refresh is not None
        else "default",
    }
    candidate, candidate_metadata = load_candidate(
        candidate_checkpoint,
        device=device,
        inference_config=_profile_inference_config(
            inference_profile, schema=FEATURE_SCHEMA_VERSION
        ),
        homogeneous_relational_bias=refresh.inference.homogeneous_relational_bias
        if refresh is not None
        else False,
        **compile_options,
    )
    teacher = load_legacy_teacher(legacy_checkpoint, device=device)
    teacher_inference = maybe_compile_model(
        teacher.model,
        enabled=compile_options["compile_model"],
        dynamic=compile_options["compile_dynamic"],
        fullgraph=True,
        mode=compile_options["compile_mode"],
    )
    baseline = GraphInferenceAdapter(
        teacher_inference,
        device=device,
        config=_profile_inference_config(
            inference_profile, schema=LEGACY_FEATURE_SCHEMA_VERSION
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
        stable_pair_seeds=True,
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
    result["inference_runtime"] = {
        "candidate": asdict(candidate.config),
        "baseline": asdict(baseline.config),
        "homogeneous_relational_bias": candidate.homogeneous_relational_bias,
        **compile_options,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    teacher = parser.add_mutually_exclusive_group(required=True)
    teacher.add_argument("--legacy-checkpoint", type=Path)
    teacher.add_argument(
        "--legacy-champion",
        type=Path,
        help="previous lineage learner/champion.json pointer",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rings", default="4,6,8,10")
    parser.add_argument("--pairs-per-ring", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--max-considered", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--inference-profile",
        type=Path,
        help="Optional profile for cache/pinned/bias/compile runtime flags; both sides retain fp32.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output.exists():
            raise LineageArenaError(f"output already exists: {args.output}")
        import star_native

        validate_native_module(star_native)
        legacy_checkpoint = (
            args.legacy_checkpoint
            if args.legacy_checkpoint is not None
            else resolve_legacy_champion(args.legacy_champion)
        )
        result = run_lineage_arena(
            native_module=star_native,
            candidate_checkpoint=args.candidate_checkpoint,
            legacy_checkpoint=legacy_checkpoint,
            rings=tuple(int(value) for value in str(args.rings).split(",")),
            pairs_per_ring=args.pairs_per_ring,
            simulations=args.simulations,
            max_considered=args.max_considered,
            seed=args.seed,
            device=torch.device(args.device),
            inference_profile=load_config(args.inference_profile)
            if args.inference_profile is not None
            else None,
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
