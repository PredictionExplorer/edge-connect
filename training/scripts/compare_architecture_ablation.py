#!/usr/bin/env python3
"""Verify a pinned three-way heterogeneous architecture evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import cast
from dataclasses import dataclass
from pathlib import Path

from startrain.arena import (
    ARENA_RESULT_SCHEMA_VERSION,
    ArenaGame,
    ArenaPair,
    elo_from_probability,
    pair_confidence_sequence,
)
from startrain.checkpoint import (
    ModelManifest,
    VerifiedModelConfig,
    extract_verified_manifest_config,
    load_ema_checkpoint,
    load_model_manifest,
    sha256_file,
    verify_file,
)
from startrain.model import GraphResTNet
from startrain.runtime import atomic_json

SUITE_FORMAT = "startrain.architecture-ablation-suite"
SUITE_SCHEMA_VERSION = 1
EVIDENCE_FORMAT = "startrain.architecture-ablation-evidence"
EVIDENCE_SCHEMA_VERSION = 1
ARCHITECTURE_RESULT_KIND = "architecture_evaluation"
PAIR_ERROR_PROBABILITY = 0.025

MODEL_ROLES = ("control", "treatment", "baseline")
COMPARISON_ROLES: dict[str, tuple[str, str]] = {
    "control_vs_baseline": ("control", "baseline"),
    "treatment_vs_baseline": ("treatment", "baseline"),
    "treatment_vs_control": ("treatment", "control"),
}
_SUITE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True, slots=True)
class ArtifactPin:
    path: Path
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class _ModelEvidence:
    pin: ArtifactPin
    manifest: ModelManifest
    config: VerifiedModelConfig
    parameter_count: int


@dataclass(frozen=True, slots=True)
class _ArenaEvidence:
    pin: ArtifactPin
    candidate_role: str
    baseline_role: str
    pairs: tuple[ArenaPair, ...]
    schedule: tuple[dict[str, object], ...]
    search: Mapping[str, object]
    arena_contract: Mapping[str, object]
    evaluator_contract: Mapping[str, object]
    evaluation_metrics: Mapping[str, object]
    diagnostic_assessment: Mapping[str, object]


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: Sequence[str],
    name: str,
) -> None:
    expected_keys = set(expected)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected_keys)}; "
            f"got {sorted(actual_keys)}"
        )


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _artifact_pin(
    value: object,
    *,
    base: Path,
    name: str,
) -> ArtifactPin:
    payload = _mapping(value, name)
    _exact_keys(payload, ("path", "sha256", "bytes"), name)
    raw_path = payload["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{name} path must be non-empty")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    pin = ArtifactPin(
        path=path.resolve(),
        sha256=_sha256(payload["sha256"], f"{name} sha256"),
        bytes=_positive_int(payload["bytes"], f"{name} bytes"),
    )
    verify_file(
        pin.path,
        expected_sha256=pin.sha256,
        expected_bytes=pin.bytes,
    )
    return pin


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_model(role: str, pin: ArtifactPin) -> _ModelEvidence:
    manifest = load_model_manifest(pin.path)
    artifact = (manifest.artifact_manifest or manifest.path).resolve()
    if manifest.path.resolve() != pin.path or artifact != pin.path:
        raise ValueError(f"{role} model pin must reference an immutable manifest")
    if (
        manifest.role != "direct"
        or manifest.manifest_sha256 != pin.sha256
        or manifest.manifest_bytes != pin.bytes
    ):
        raise ValueError(f"{role} model manifest pin disagrees with its artifact")
    config = extract_verified_manifest_config(manifest, require_ema=True)
    model = GraphResTNet(config.model)
    load_ema_checkpoint(
        manifest.checkpoint,
        model=model,
        expected_model_config=config.model_config,
        expected_game_config=config.game_config,
        expected_run_id=manifest.run_id,
        expected_generation_family=manifest.generation_family,
        expected_sha256=manifest.checkpoint_sha256,
        expected_bytes=manifest.checkpoint_bytes,
    )
    parameter_count = model.parameter_count()
    del model
    return _ModelEvidence(
        pin=pin,
        manifest=manifest,
        config=config,
        parameter_count=parameter_count,
    )


def _manifest_reference(
    artifact: Path,
    value: object,
    name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = artifact.parent / path
    return path.resolve()


def _variant_fields(payload: Mapping[str, object], name: str) -> dict[str, object]:
    """Validate the rules-v3 variant provenance of one arena pair or game."""

    variant = payload["variant"]
    segment = payload["segment"]
    if type(variant) is not str or type(segment) is not str:
        raise ValueError(f"{name} variant and segment must be strings")
    fields: dict[str, object] = {"variant": variant, "segment": segment}
    if "swapped" in payload:
        if type(payload["swapped"]) is not bool:
            raise ValueError(f"{name} swapped must be boolean")
        fields["swapped"] = payload["swapped"]
    if "pda" in payload:
        fields["pda"] = _nonnegative_int(payload["pda"], f"{name} pda")
    return fields


def _arena_pair(value: object, name: str) -> ArenaPair:
    payload = _mapping(value, name)
    _exact_keys(
        payload,
        (
            "ring",
            "pair",
            "opening_seed",
            "opening_action",
            "forced_opening",
            "outcomes",
            "variant",
            "segment",
        ),
        name,
    )
    outcomes = payload["outcomes"]
    if (
        not isinstance(outcomes, (list, tuple))
        or len(outcomes) != 2
        or any(type(outcome) is not int for outcome in outcomes)
    ):
        raise ValueError(f"{name} outcomes must contain exactly two integers")
    opening_action = payload["opening_action"]
    if opening_action is not None and type(opening_action) is not int:
        raise ValueError(f"{name} opening_action must be an integer or null")
    if type(payload["forced_opening"]) is not bool:
        raise ValueError(f"{name} forced_opening must be boolean")
    variant_fields = _variant_fields(payload, name)
    return ArenaPair(
        ring=_positive_int(payload["ring"], f"{name} ring"),
        pair=_nonnegative_int(payload["pair"], f"{name} pair"),
        opening_seed=_nonnegative_int(payload["opening_seed"], f"{name} opening_seed"),
        opening_action=opening_action,
        forced_opening=payload["forced_opening"],
        outcomes=(outcomes[0], outcomes[1]),
        variant=cast(str, variant_fields["variant"]),
        segment=cast(str, variant_fields["segment"]),
    )


def _arena_game(value: object, name: str) -> ArenaGame:
    payload = _mapping(value, name)
    _exact_keys(
        payload,
        (
            "ring",
            "pair",
            "candidate_player",
            "opening_seed",
            "opening_action",
            "forced_opening",
            "winner",
            "outcome",
            "searched_moves",
            "variant",
            "segment",
            "swapped",
            "pda",
        ),
        name,
    )
    opening_action = payload["opening_action"]
    if opening_action is not None and type(opening_action) is not int:
        raise ValueError(f"{name} opening_action must be an integer or null")
    if type(payload["forced_opening"]) is not bool:
        raise ValueError(f"{name} forced_opening must be boolean")
    variant_fields = _variant_fields(payload, name)
    return ArenaGame(
        ring=_positive_int(payload["ring"], f"{name} ring"),
        pair=_nonnegative_int(payload["pair"], f"{name} pair"),
        candidate_player=_nonnegative_int(
            payload["candidate_player"], f"{name} candidate_player"
        ),
        opening_seed=_nonnegative_int(payload["opening_seed"], f"{name} opening_seed"),
        opening_action=opening_action,
        forced_opening=payload["forced_opening"],
        winner=_nonnegative_int(payload["winner"], f"{name} winner"),
        outcome=_nonnegative_int(payload["outcome"], f"{name} outcome")
        if payload["outcome"] != -1
        else -1,
        searched_moves=_nonnegative_int(
            payload["searched_moves"], f"{name} searched_moves"
        ),
        variant=cast(str, variant_fields["variant"]),
        segment=cast(str, variant_fields["segment"]),
        swapped=cast(bool, variant_fields.get("swapped", False)),
        pda=cast(int, variant_fields.get("pda", 0)),
    )


def _complete_role_reversed_pairs(
    payload: Mapping[str, object],
    artifact: Path,
    *,
    expected_pair_counts: Mapping[int, int],
) -> tuple[tuple[ArenaPair, ...], tuple[dict[str, object], ...]]:
    raw_pairs = payload.get("pairs")
    raw_games = payload.get("games")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError(f"architecture arena has no pairs: {artifact}")
    if not isinstance(raw_games, list) or not raw_games:
        raise ValueError(f"architecture arena has no games: {artifact}")
    pairs = tuple(
        _arena_pair(value, f"{artifact} pair[{index}]")
        for index, value in enumerate(raw_pairs)
    )
    games = tuple(
        _arena_game(value, f"{artifact} game[{index}]")
        for index, value in enumerate(raw_games)
    )

    pair_map: dict[tuple[int, int], ArenaPair] = {}
    for pair in pairs:
        key = (pair.ring, pair.pair)
        if key in pair_map:
            raise ValueError(f"architecture arena has duplicate pair {key}: {artifact}")
        pair_map[key] = pair
    game_map: dict[tuple[int, int, int], ArenaGame] = {}
    for game in games:
        key = (game.ring, game.pair, game.candidate_player)
        if key in game_map:
            raise ValueError(f"architecture arena has duplicate game {key}: {artifact}")
        game_map[key] = game
    expected_games = {
        (ring, pair, candidate_player)
        for ring, pair in pair_map
        for candidate_player in (0, 1)
    }
    if set(game_map) != expected_games:
        raise ValueError(
            f"architecture arena lacks complete role-reversed games: {artifact}"
        )

    per_ring: dict[int, list[int]] = {}
    for (ring, pair), completed_pair in pair_map.items():
        per_ring.setdefault(ring, []).append(pair)
        pair_games = (game_map[(ring, pair, 0)], game_map[(ring, pair, 1)])
        if any(
            (
                game.opening_seed,
                game.opening_action,
                game.forced_opening,
            )
            != (
                completed_pair.opening_seed,
                completed_pair.opening_action,
                completed_pair.forced_opening,
            )
            for game in pair_games
        ):
            raise ValueError(
                f"architecture arena pair/game opening mismatch: {artifact}"
            )
        if tuple(game.outcome for game in pair_games) != completed_pair.outcomes:
            raise ValueError(
                f"architecture arena pair/game outcome mismatch: {artifact}"
            )
    if set(per_ring) != set(expected_pair_counts):
        raise ValueError(
            f"architecture arena rings differ from its pinned contract: {artifact}"
        )
    for ring, expected_count in expected_pair_counts.items():
        indices = per_ring[ring]
        if sorted(indices) != list(range(expected_count)):
            raise ValueError(
                f"architecture arena ring {ring} pair indices are not complete"
            )

    metrics = _mapping(payload.get("evaluation_metrics"), "evaluation_metrics")
    requested = _positive_int(metrics.get("requested_pairs"), "requested_pairs")
    completed = _positive_int(metrics.get("completed_pairs"), "completed_pairs")
    expected_total = sum(expected_pair_counts.values())
    if (
        requested != expected_total
        or completed != expected_total
        or len(pairs) != expected_total
    ):
        raise ValueError(
            f"architecture arena did not complete every requested pair: {artifact}"
        )

    ordered = tuple(pair_map[key] for key in sorted(pair_map))
    schedule = tuple(
        {
            "ring": pair.ring,
            "pair": pair.pair,
            "opening_seed": pair.opening_seed,
            "opening_action": pair.opening_action,
            "forced_opening": pair.forced_opening,
        }
        for pair in ordered
    )
    return ordered, schedule


def _arena_contract(
    value: object,
    comparison: str,
) -> tuple[Mapping[str, object], dict[int, int]]:
    contract = _mapping(value, f"{comparison} arena_contract")
    rings = contract.get("rings")
    if (
        not isinstance(rings, (list, tuple))
        or not rings
        or any(type(ring) is not int or ring <= 0 for ring in rings)
        or len(set(rings)) != len(rings)
    ):
        raise ValueError(f"{comparison} arena contract rings are invalid")
    pairs_per_ring = _positive_int(
        contract.get("pairs_per_ring"),
        f"{comparison} pairs_per_ring",
    )
    return contract, {ring: pairs_per_ring for ring in rings}


def _load_arena(
    comparison: str,
    pin: ArtifactPin,
    *,
    candidate_role: str,
    baseline_role: str,
    models: Mapping[str, _ModelEvidence],
) -> _ArenaEvidence:
    payload = _load_json(pin.path, f"{comparison} arena")
    candidate = models[candidate_role]
    baseline = models[baseline_role]
    if (
        payload.get("schema_version") != ARENA_RESULT_SCHEMA_VERSION
        or payload.get("result_kind") != ARCHITECTURE_RESULT_KIND
        or payload.get("evaluation_mode") != "architecture"
    ):
        raise ValueError(f"{comparison} is not a supported architecture arena")
    if (
        payload.get("candidate") != candidate.manifest.model_identity
        or payload.get("baseline") != baseline.manifest.model_identity
    ):
        raise ValueError(f"{comparison} arena identities are incompatible")
    if (
        payload.get("diagnostic_only") is not True
        or payload.get("promotion_authorized") is not False
        or payload.get("adoption_authorized") is not False
    ):
        raise ValueError(f"{comparison} arena is not diagnostic-only")
    if payload.get("interrupted") is not False:
        raise ValueError(f"{comparison} arena was interrupted")
    promotion = _mapping(payload.get("promotion"), f"{comparison} promotion")
    if (
        promotion.get("decision") != "evaluation"
        or promotion.get("authorized") is not False
    ):
        raise ValueError(f"{comparison} arena could authorize promotion")
    assessment = _mapping(
        payload.get("diagnostic_assessment"),
        f"{comparison} diagnostic assessment",
    )

    for side, expected in (("candidate", candidate), ("baseline", baseline)):
        reference = _manifest_reference(
            pin.path,
            payload.get(f"{side}_manifest"),
            f"{comparison} {side}_manifest",
        )
        if (
            reference != expected.pin.path
            or payload.get(f"{side}_manifest_sha256") != expected.pin.sha256
            or payload.get(f"{side}_manifest_bytes") != expected.pin.bytes
        ):
            raise ValueError(f"{comparison} {side} manifest pin is incompatible")

    model_configs = _mapping(
        payload.get("model_configs"), f"{comparison} model_configs"
    )
    _exact_keys(model_configs, ("candidate", "baseline"), "model_configs")
    if _canonical(model_configs["candidate"]) != _canonical(
        candidate.config.model_config
    ) or _canonical(model_configs["baseline"]) != _canonical(
        baseline.config.model_config
    ):
        raise ValueError(f"{comparison} model configurations are incompatible")
    if _canonical(payload.get("evaluation_contract")) != _canonical(
        candidate.config.evaluation_contract
    ):
        raise ValueError(f"{comparison} evaluation contract is incompatible")

    arena_contract, expected_pair_counts = _arena_contract(
        payload.get("arena_contract"),
        comparison,
    )
    evaluator_contract = _mapping(
        payload.get("evaluator_contract"),
        f"{comparison} evaluator_contract",
    )
    if set(evaluator_contract) != {
        "device_type",
        "precision",
        "score_utility_weight",
        "compile_enabled",
        "compile_dynamic",
        "compile_mode",
        "fullgraph",
    }:
        raise ValueError(f"{comparison} evaluator contract fields are incompatible")
    pairs, schedule = _complete_role_reversed_pairs(
        payload,
        pin.path,
        expected_pair_counts=expected_pair_counts,
    )
    evaluation_metrics = _mapping(
        payload.get("evaluation_metrics"),
        f"{comparison} evaluation_metrics",
    )
    search = _mapping(payload.get("search"), f"{comparison} search")
    for key in (
        "simulations",
        "max_considered",
        "c_visit",
        "c_scale",
        "pair_chunk_size",
    ):
        if search.get(key) != arena_contract.get(key):
            raise ValueError(f"{comparison} search differs from its arena contract")
    return _ArenaEvidence(
        pin=pin,
        candidate_role=candidate_role,
        baseline_role=baseline_role,
        pairs=pairs,
        schedule=schedule,
        search=search,
        arena_contract=arena_contract,
        evaluator_contract=evaluator_contract,
        evaluation_metrics=evaluation_metrics,
        diagnostic_assessment=assessment,
    )


def _pair_summary(pairs: Sequence[ArenaPair]) -> dict[str, object]:
    if not pairs:
        raise ValueError("pair summary requires complete pairs")
    wins = sum(outcome == 1 for pair in pairs for outcome in pair.outcomes)
    games = 2 * len(pairs)
    score_rate = wins / games
    lower, upper = pair_confidence_sequence(
        pairs,
        error_probability=PAIR_ERROR_PROBABILITY,
    )
    pair_win_counts = {
        str(points): sum(int(pair.points) == points for pair in pairs)
        for points in (0, 1, 2)
    }
    return {
        "observation_unit": "complete-role-reversed-pair",
        "pairs": len(pairs),
        "games": games,
        "wins": wins,
        "losses": games - wins,
        "score_rate": score_rate,
        "elo_difference": elo_from_probability(score_rate),
        "anytime_error_probability_per_side": PAIR_ERROR_PROBABILITY,
        "anytime_score_interval": [lower, upper],
        "anytime_elo_interval": [
            elo_from_probability(lower),
            elo_from_probability(upper),
        ],
        "pair_win_counts": pair_win_counts,
    }


def _comparison_evidence(arena: _ArenaEvidence) -> dict[str, object]:
    rings = sorted({pair.ring for pair in arena.pairs})
    return {
        "artifact": arena.pin.as_dict(),
        "candidate_role": arena.candidate_role,
        "baseline_role": arena.baseline_role,
        "complete_role_reversed_pairs": True,
        "aggregate": _pair_summary(arena.pairs),
        "per_ring": {
            str(ring): _pair_summary(
                tuple(pair for pair in arena.pairs if pair.ring == ring)
            )
            for ring in rings
        },
        "evaluation_metrics": dict(arena.evaluation_metrics),
        "diagnostic_assessment": dict(arena.diagnostic_assessment),
    }


def build_architecture_ablation_evidence(
    suite_manifest: str | Path,
) -> dict[str, object]:
    suite_path = Path(suite_manifest).resolve()
    payload = _load_json(suite_path, "architecture ablation suite")
    _exact_keys(
        payload,
        ("format", "schema_version", "suite_id", "models", "arenas"),
        "architecture ablation suite",
    )
    if (
        payload.get("format") != SUITE_FORMAT
        or payload.get("schema_version") != SUITE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported architecture ablation suite")
    suite_id = payload.get("suite_id")
    if not isinstance(suite_id, str) or _SUITE_ID.fullmatch(suite_id) is None:
        raise ValueError("architecture ablation suite_id is invalid")

    raw_models = _mapping(payload["models"], "architecture suite models")
    _exact_keys(raw_models, MODEL_ROLES, "architecture suite models")
    model_pins = {
        role: _artifact_pin(
            raw_models[role],
            base=suite_path.parent,
            name=f"{role} model",
        )
        for role in MODEL_ROLES
    }
    models = {role: _load_model(role, model_pins[role]) for role in MODEL_ROLES}
    identities = {model.manifest.model_identity for model in models.values()}
    if len(identities) != len(MODEL_ROLES):
        raise ValueError("control, treatment, and baseline must be distinct models")
    if models["control"].config.model_config == models["treatment"].config.model_config:
        raise ValueError("architecture suite control and treatment are homogeneous")
    common_contract = models["control"].config.evaluation_contract
    if any(
        model.config.evaluation_contract != common_contract for model in models.values()
    ):
        raise ValueError("architecture suite evaluation contracts are incompatible")

    raw_arenas = _mapping(payload["arenas"], "architecture suite arenas")
    _exact_keys(raw_arenas, tuple(COMPARISON_ROLES), "architecture suite arenas")
    arena_pins = {
        comparison: _artifact_pin(
            raw_arenas[comparison],
            base=suite_path.parent,
            name=f"{comparison} arena",
        )
        for comparison in COMPARISON_ROLES
    }
    if len({pin.path for pin in arena_pins.values()}) != len(COMPARISON_ROLES):
        raise ValueError("architecture suite arena artifacts must be distinct")
    arenas = {
        comparison: _load_arena(
            comparison,
            arena_pins[comparison],
            candidate_role=roles[0],
            baseline_role=roles[1],
            models=models,
        )
        for comparison, roles in COMPARISON_ROLES.items()
    }
    schedules = {_canonical(arena.schedule) for arena in arenas.values()}
    if len(schedules) != 1:
        raise ValueError(
            "architecture arenas do not share one complete opening schedule"
        )
    search_contracts = {_canonical(arena.search) for arena in arenas.values()}
    if len(search_contracts) != 1:
        raise ValueError("architecture arenas do not share one search contract")
    arena_contracts = {_canonical(arena.arena_contract) for arena in arenas.values()}
    if len(arena_contracts) != 1:
        raise ValueError("architecture arenas do not share one arena contract")
    evaluator_contracts = {
        _canonical(arena.evaluator_contract) for arena in arenas.values()
    }
    if len(evaluator_contracts) != 1:
        raise ValueError("architecture arenas do not share one evaluator contract")
    schedule = arenas["control_vs_baseline"].schedule
    schedule_sha256 = hashlib.sha256(_canonical(schedule).encode("utf-8")).hexdigest()

    suite_pin = ArtifactPin(
        path=suite_path,
        sha256=sha256_file(suite_path),
        bytes=suite_path.stat().st_size,
    )
    evidence = {
        "format": EVIDENCE_FORMAT,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "suite_id": suite_id,
        "suite_manifest": suite_pin.as_dict(),
        "diagnostic_only": True,
        "production_promotion_authorized": False,
        "adoption_authorized": False,
        "common_frozen_baseline": models["baseline"].manifest.model_identity,
        "direct_treatment_control_crossplay": True,
        "evaluation_contract": common_contract,
        "model_relation": {
            "heterogeneous": True,
            "control_treatment_parameter_count_equal": (
                models["control"].parameter_count == models["treatment"].parameter_count
            ),
        },
        "models": {
            role: {
                "identity": model.manifest.model_identity,
                "manifest": model.pin.as_dict(),
                "checkpoint": {
                    "path": str(model.manifest.checkpoint),
                    "sha256": model.manifest.checkpoint_sha256,
                    "bytes": model.manifest.checkpoint_bytes,
                },
                "model_config": model.config.model_config,
                "parameter_count": model.parameter_count,
            }
            for role, model in models.items()
        },
        "pair_validity": {
            "complete_role_reversed_pairs_only": True,
            "aligned_opening_schedule": True,
            "opening_schedule_sha256": schedule_sha256,
            "pair_count_per_comparison": len(schedule),
            "confidence_method": "pair-level-mixture-betting-confidence-sequence-v1",
            "error_probability_per_side": PAIR_ERROR_PROBABILITY,
        },
        "search_contract": json.loads(next(iter(search_contracts))),
        "arena_contract": json.loads(next(iter(arena_contracts))),
        "evaluator_contract": json.loads(next(iter(evaluator_contracts))),
        "comparisons": {
            comparison: _comparison_evidence(arenas[comparison])
            for comparison in COMPARISON_ROLES
        },
    }
    normalized = json.loads(_canonical(evidence))
    if not isinstance(normalized, dict):
        raise RuntimeError("architecture evidence normalization failed")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-manifest",
        "--suite",
        dest="suite_manifest",
        required=True,
        type=Path,
        help="immutable suite pinning exactly three manifests and three arena results",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    evidence = build_architecture_ablation_evidence(arguments.suite_manifest)
    atomic_json(arguments.output, evidence)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "suite_id": evidence["suite_id"],
                "diagnostic_only": True,
                "production_promotion_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
