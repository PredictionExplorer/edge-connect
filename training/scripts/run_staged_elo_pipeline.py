#!/usr/bin/env python3
"""Advance one Elo stage only from a verified upstream winner snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from startrain.runtime import atomic_json

if __package__:
    from .fork_elo_ablation import fork_elo_ablation
    from .prepare_champion_warm_start import prepare_champion_warm_start
    from .prepare_elo_ablation import (
        GUARD_RINGS,
        WEIGHTED_INITIAL_BLOCKS,
        WEIGHTED_TREATMENTS,
        blocking_guard_rings,
        prepare_elo_ablation,
        validate_futility_policy,
        verify_winner_snapshot,
        winner_snapshot_from_document,
    )
    from .run_elo_ablation_queue import run_ablation_queue
else:
    from fork_elo_ablation import fork_elo_ablation
    from prepare_champion_warm_start import prepare_champion_warm_start
    from prepare_elo_ablation import (
        GUARD_RINGS,
        WEIGHTED_INITIAL_BLOCKS,
        WEIGHTED_TREATMENTS,
        blocking_guard_rings,
        prepare_elo_ablation,
        validate_futility_policy,
        verify_winner_snapshot,
        winner_snapshot_from_document,
    )
    from run_elo_ablation_queue import run_ablation_queue

SCHEMA_VERSION = 1
PIPELINE_REPORT = "startrain-staged-elo-pipeline"
PIPELINE_STATE_REPORT = "startrain-staged-elo-pipeline-state"
FUTILITY_REPORT = "startrain-elo-futility-policy"
FUTILITY_EVALUATION_REPORT = "startrain-elo-futility-evaluation"

QueueRunner = Callable[..., dict[str, object]]
Preparer = Callable[..., dict[str, object]]
Forker = Callable[..., dict[str, object]]
WarmStarter = Callable[..., dict[str, object]]


class StagedEloPipelineError(RuntimeError):
    """Raised when a stage transition is unverified, stale, or malformed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StagedEloPipelineError(
            f"cannot read JSON object {path}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise StagedEloPipelineError(f"{path} must contain a JSON object")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StagedEloPipelineError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise StagedEloPipelineError(f"{name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StagedEloPipelineError(f"{name} must be a non-empty string")
    return value


def build_futility_policy(
    *,
    guard_rings: Sequence[int] = GUARD_RINGS,
    guard_floor_elo: float,
    minimum_decisive_games: int | None = None,
    minimum_control_advantage_elo: float = 0.0,
    control_objective: str = "ring_10",
    enabled: bool = True,
) -> dict[str, object]:
    """Build the deterministic, pre-registered stop-only policy."""
    if control_objective not in {"ring_10", "weighted_aggregate"}:
        raise ValueError("futility control objective is unsupported")
    if minimum_decisive_games is None:
        minimum_decisive_games = (
            WEIGHTED_INITIAL_BLOCKS
            if control_objective == "weighted_aggregate"
            else 200
        )
    control_comparison: dict[str, object] = {
        "minimum_advantage_elo": minimum_control_advantage_elo,
        "arm_upper_field": "anytime_upper_elo",
        "control_lower_field": "anytime_lower_elo",
    }
    if control_objective == "ring_10":
        control_comparison["ring"] = 10
    else:
        control_comparison["objective"] = "weighted_aggregate"
    policy: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report": FUTILITY_REPORT,
        "enabled": enabled,
        "decision_scope": "stop_only",
        "evidence": "anytime_valid_confidence_sequences",
        "application": "pure_evaluator_no_implicit_live_polling",
        "minimum_decisive_games": minimum_decisive_games,
        "guard_regression": {
            "rings": list(guard_rings),
            "floor_elo": guard_floor_elo,
            "arm_upper_field": "anytime_upper_elo",
        },
        "control_comparison": control_comparison,
    }
    return validate_futility_policy(
        policy,
        guard_rings=guard_rings,
        guard_floor_elo=guard_floor_elo,
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def evaluate_futility(
    policy: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    """Apply only pre-registered anytime-valid stop decisions."""
    guard = policy.get("guard_regression")
    if not isinstance(guard, Mapping):
        raise ValueError("futility guard-regression policy is missing")
    rings = guard.get("rings")
    floor = _finite_number(guard.get("floor_elo"))
    if not isinstance(rings, list) or floor is None:
        raise ValueError("futility guard-regression policy is malformed")
    validated = validate_futility_policy(
        policy,
        guard_rings=tuple(rings),
        guard_floor_elo=floor,
    )
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report": FUTILITY_EVALUATION_REPORT,
        "policy": validated,
        "promotion_allowed": False,
    }
    if validated["enabled"] is not True:
        return {**base, "decision": "continue", "reason": "policy_disabled"}
    if (
        evidence.get("anytime_valid") is not True
        or evidence.get("method") != "anytime_valid_confidence_sequence"
    ):
        return {
            **base,
            "decision": "continue",
            "reason": "insufficient_anytime_valid_evidence",
        }
    control = validated["control_comparison"]
    assert isinstance(control, dict)
    objective = str(control.get("objective", "ring_10"))
    objective_evidence = evidence.get(objective)
    games = (
        evidence.get(
            "complete_blocks",
            (
                objective_evidence.get("complete_blocks")
                if isinstance(objective_evidence, Mapping)
                else evidence.get("decisive_games")
            ),
        )
        if objective == "weighted_aggregate"
        else evidence.get("decisive_games")
    )
    raw_minimum_games = validated["minimum_decisive_games"]
    assert type(raw_minimum_games) is int
    minimum_games = raw_minimum_games
    if type(games) is not int or games < minimum_games:
        return {
            **base,
            "decision": "continue",
            "reason": "minimum_decisive_games_not_reached",
        }

    raw_guards = evidence.get("guard_rings")
    if rings and not isinstance(raw_guards, Mapping):
        return {
            **base,
            "decision": "continue",
            "reason": "insufficient_anytime_valid_evidence",
        }
    for ring in rings:
        assert isinstance(raw_guards, Mapping)
        raw_ring = raw_guards.get(str(ring), raw_guards.get(ring))
        if not isinstance(raw_ring, Mapping):
            return {
                **base,
                "decision": "continue",
                "reason": "insufficient_anytime_valid_evidence",
            }
        upper = _finite_number(raw_ring.get("anytime_upper_elo"))
        if upper is None:
            return {
                **base,
                "decision": "continue",
                "reason": "insufficient_anytime_valid_evidence",
            }
        if upper < floor:
            return {
                **base,
                "decision": "stop_for_futility",
                "reason": "definitive_ring_regression",
                "ring": ring,
                "arm_anytime_upper_elo": upper,
                "guard_floor_elo": floor,
            }

    comparison_evidence = objective_evidence
    if not isinstance(comparison_evidence, Mapping):
        return {
            **base,
            "decision": "continue",
            "reason": "insufficient_anytime_valid_evidence",
        }
    arm_upper = _finite_number(comparison_evidence.get("anytime_upper_elo"))
    control_lower = _finite_number(comparison_evidence.get("control_anytime_lower_elo"))
    arm_interval = comparison_evidence.get("anytime_elo_interval")
    if (
        arm_upper is None
        and isinstance(arm_interval, Sequence)
        and not isinstance(arm_interval, str | bytes)
        and len(arm_interval) == 2
    ):
        arm_upper = _finite_number(arm_interval[1])
    control_interval = comparison_evidence.get("control_anytime_elo_interval")
    if (
        control_lower is None
        and isinstance(control_interval, Sequence)
        and not isinstance(control_interval, str | bytes)
        and len(control_interval) == 2
    ):
        control_lower = _finite_number(control_interval[0])
    if arm_upper is None or control_lower is None:
        return {
            **base,
            "decision": "continue",
            "reason": "insufficient_anytime_valid_evidence",
        }
    required_advantage = float(control["minimum_advantage_elo"])
    if arm_upper <= control_lower + required_advantage:
        return {
            **base,
            "decision": "stop_for_futility",
            "reason": "cannot_beat_control",
            "arm_anytime_upper_elo": arm_upper,
            "control_anytime_lower_elo": control_lower,
            "minimum_control_advantage_elo": required_advantage,
            "control_objective": objective,
        }
    return {
        **base,
        "decision": "continue",
        "reason": "futility_not_proven",
        "control_objective": objective,
    }


def _preparation_objective(
    preparation: Mapping[str, object],
) -> tuple[tuple[int, ...], str]:
    treatments = _list(preparation.get("treatments"), "downstream treatments")
    if any(not isinstance(item, str) or not item for item in treatments):
        raise StagedEloPipelineError("downstream treatments must be strings")
    try:
        inferred = blocking_guard_rings(tuple(str(item) for item in treatments))
    except ValueError as error:
        raise StagedEloPipelineError(str(error)) from error
    raw_rings = preparation.get("guard_rings")
    if raw_rings is None:
        guard_rings = inferred
    else:
        configured = _list(raw_rings, "downstream guard rings")
        if any(type(ring) is not int or ring <= 0 for ring in configured):
            raise StagedEloPipelineError(
                "downstream guard rings must contain positive integers"
            )
        if len(set(configured)) != len(configured):
            raise StagedEloPipelineError("downstream guard rings contain duplicates")
        guard_rings = tuple(configured)
        if guard_rings != inferred:
            raise StagedEloPipelineError(
                "downstream guard rings do not match the treatment objective"
            )
    raw_objective = preparation.get("promotion_objective")
    objective = (
        str(raw_objective)
        if raw_objective is not None
        else ("weighted_aggregate" if not guard_rings else "ring_10")
    )
    if objective not in {"ring_10", "weighted_aggregate"}:
        raise StagedEloPipelineError("downstream promotion objective is unsupported")
    if (objective == "weighted_aggregate") != all(
        str(item) in WEIGHTED_TREATMENTS for item in treatments
    ):
        raise StagedEloPipelineError(
            "downstream promotion objective does not match its treatments"
        )
    return guard_rings, objective


def _comparison_path(deployment_manifest: Path) -> Path:
    deployment = _read_json(deployment_manifest)
    queue = _mapping(deployment.get("queue"), "upstream deployment queue")
    return (
        Path(_string(queue.get("comparison_output"), "upstream comparison output"))
        .expanduser()
        .resolve()
    )


def _stage_spec_document(
    *,
    preparation: Mapping[str, object],
    source: Path,
    snapshot: Mapping[str, object],
    futility_policy: Mapping[str, object],
) -> dict[str, object]:
    base_config = (
        Path(_string(preparation.get("base_config"), "downstream base config"))
        .expanduser()
        .resolve()
    )
    if not base_config.is_file():
        raise StagedEloPipelineError(
            f"downstream base config does not exist: {base_config}"
        )
    material = {
        "preparation": dict(preparation),
        "source_run_root": str(source),
        "source_winner_snapshot": dict(snapshot),
        "futility_policy": dict(futility_policy),
        "base_config_sha256": _sha256(base_config),
    }
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        **material,
    }


def _prepared_plan(
    *,
    preparation: Mapping[str, object],
    source: Path,
    snapshot: Mapping[str, object],
    futility_policy: Mapping[str, object],
    guard_rings: Sequence[int],
    preparer: Preparer,
) -> tuple[dict[str, object], Path]:
    output_dir = (
        Path(_string(preparation.get("output_dir"), "downstream output directory"))
        .expanduser()
        .resolve()
    )
    plan_path = output_dir / "ablation-plan.json"
    treatments = _list(preparation.get("treatments"), "downstream treatments")
    if any(not isinstance(item, str) or not item for item in treatments):
        raise StagedEloPipelineError("downstream treatments must be strings")
    stage_spec = _stage_spec_document(
        preparation=preparation,
        source=source,
        snapshot=snapshot,
        futility_policy=futility_policy,
    )
    if output_dir.exists():
        if not plan_path.is_file():
            raise StagedEloPipelineError(
                "downstream output exists without a prepared ablation plan"
            )
        plan = _read_json(plan_path)
        if plan.get("source_winner_snapshot") != dict(snapshot):
            raise StagedEloPipelineError(
                "downstream stage was prepared from a stale winner snapshot"
            )
        if plan.get("futility_policy") != dict(futility_policy):
            raise StagedEloPipelineError(
                "downstream stage futility policy differs from pre-registration"
            )
        if plan.get("staged_pipeline_spec") != stage_spec:
            raise StagedEloPipelineError(
                "downstream stage specification changed after preparation"
            )
        return plan, plan_path

    try:
        plan = preparer(
            base_config=Path(
                _string(preparation.get("base_config"), "downstream base config")
            ),
            output_dir=output_dir,
            run_root_parent=Path(
                _string(
                    preparation.get("run_root_parent"),
                    "downstream run-root parent",
                )
            ),
            run_id=_string(preparation.get("run_id"), "downstream run ID"),
            source_run_root=source,
            prefix=_string(preparation.get("prefix"), "downstream prefix"),
            seed=preparation.get("seed"),
            wall_budget_hours=preparation.get("wall_budget_hours"),
            leaf_budget=preparation.get("leaf_budget"),
            guard_floor_elo=preparation.get("guard_floor_elo"),
            treatments=tuple(treatments),
            winner_snapshot=snapshot,
            futility_policy=futility_policy,
            guard_rings=guard_rings,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise StagedEloPipelineError(
            f"downstream preparation failed: {type(error).__name__}: {error}"
        ) from error
    plan["staged_pipeline_spec"] = stage_spec
    atomic_json(plan_path, plan)
    return plan, plan_path


def advance_staged_elo_pipeline(
    pipeline_path: Path,
    *,
    stage_index: int,
    queue_runner: QueueRunner = run_ablation_queue,
    preparer: Preparer = prepare_elo_ablation,
    forker: Forker = fork_elo_ablation,
    warm_starter: WarmStarter = prepare_champion_warm_start,
) -> dict[str, object]:
    """Run one upstream queue and prepare/fork exactly one downstream stage."""
    path = pipeline_path.expanduser().resolve()
    pipeline = _read_json(path)
    if (
        pipeline.get("schema_version") != SCHEMA_VERSION
        or pipeline.get("report") != PIPELINE_REPORT
    ):
        raise StagedEloPipelineError("unsupported staged Elo pipeline")
    stages = _list(pipeline.get("stages"), "pipeline stages")
    if (
        type(stage_index) is not int
        or stage_index < 0
        or stage_index + 1 >= len(stages)
    ):
        raise StagedEloPipelineError("stage index has no downstream transition")
    upstream = _mapping(stages[stage_index], "upstream stage")
    downstream = _mapping(stages[stage_index + 1], "downstream stage")
    upstream_name = _string(upstream.get("name"), "upstream stage name")
    downstream_name = _string(downstream.get("name"), "downstream stage name")
    deployment_manifest = (
        Path(
            _string(
                upstream.get("deployment_manifest"),
                "upstream deployment manifest",
            )
        )
        .expanduser()
        .resolve()
    )

    queue_state = queue_runner(deployment_manifest)
    finalization = _mapping(queue_state.get("finalization"), "queue finalization")
    if (
        queue_state.get("queue_status") != "completed"
        or finalization.get("status") != "completed"
        or finalization.get("comparison_status") != "complete"
    ):
        raise StagedEloPipelineError(
            "upstream stage is not a fully finalized eligible comparison"
        )
    comparison_path = _comparison_path(deployment_manifest)
    comparison = _read_json(comparison_path)
    try:
        snapshot = winner_snapshot_from_document(comparison)
        source = Path(_string(snapshot.get("run_root"), "winner run root")).resolve()
        snapshot = verify_winner_snapshot(source, snapshot)
    except (OSError, TypeError, ValueError) as error:
        raise StagedEloPipelineError(
            f"upstream winner snapshot is not verified: {error}"
        ) from error

    preparation = _mapping(downstream.get("prepare"), "downstream preparation")
    champion = _mapping(snapshot.get("champion"), "winner champion")
    selected_identity = _string(
        champion.get("model_identity"),
        "winner champion identity",
    )
    expected_anchor = preparation.get("expected_anchor_identity")
    if expected_anchor is not None and expected_anchor != selected_identity:
        raise StagedEloPipelineError(
            "downstream stage declares a stale anchor identity "
            f"{expected_anchor!r}; selected winner is {selected_identity!r}"
        )
    configured_source = preparation.get("source_run_root")
    if configured_source is not None and (
        not isinstance(configured_source, str)
        or Path(configured_source).expanduser().resolve() != source
    ):
        raise StagedEloPipelineError(
            "downstream stage declares a stale source run root"
        )
    run_identity = _mapping(snapshot.get("run_identity"), "winner run identity")
    if preparation.get("run_id") != run_identity.get("run_id"):
        raise StagedEloPipelineError(
            "downstream run ID differs from the selected winner identity"
        )
    guard_floor = preparation.get("guard_floor_elo")
    if isinstance(guard_floor, bool) or not isinstance(guard_floor, int | float):
        raise StagedEloPipelineError("downstream guard floor must be numeric")
    guard_rings, control_objective = _preparation_objective(preparation)
    raw_policy = downstream.get("futility_policy")
    if raw_policy is None:
        policy = build_futility_policy(
            guard_rings=guard_rings,
            guard_floor_elo=float(guard_floor),
            control_objective=control_objective,
        )
    elif isinstance(raw_policy, Mapping):
        try:
            policy = validate_futility_policy(
                raw_policy,
                guard_rings=guard_rings,
                guard_floor_elo=float(guard_floor),
            )
        except ValueError as error:
            raise StagedEloPipelineError(str(error)) from error
    else:
        raise StagedEloPipelineError("downstream futility policy must be an object")
    control_comparison = _mapping(
        policy.get("control_comparison"),
        "downstream futility control comparison",
    )
    policy_objective = str(control_comparison.get("objective", "ring_10"))
    if policy_objective != control_objective:
        raise StagedEloPipelineError(
            "downstream futility objective differs from the treatment objective"
        )

    verify_winner_snapshot(source, snapshot)
    plan, plan_path = _prepared_plan(
        preparation=preparation,
        source=source,
        snapshot=snapshot,
        futility_policy=policy,
        guard_rings=guard_rings,
        preparer=preparer,
    )
    raw_treatments = _list(plan.get("treatments"), "prepared treatments")
    forked = []
    for raw_treatment in raw_treatments:
        treatment = _mapping(raw_treatment, "prepared treatment")
        label = _string(treatment.get("treatment"), "prepared treatment label")
        run_root = (
            Path(_string(treatment.get("run_root"), f"{label} run root"))
            .expanduser()
            .resolve()
        )
        verify_winner_snapshot(source, snapshot)
        metadata_path = run_root / "ablation.json"
        if run_root.exists():
            if not metadata_path.is_file():
                raise StagedEloPipelineError(
                    f"{label} run root exists without ablation metadata"
                )
            metadata = _read_json(metadata_path)
        else:
            try:
                metadata = forker(
                    source_run_root=source,
                    plan_path=plan_path,
                    treatment=label,
                )
            except (
                FileExistsError,
                FileNotFoundError,
                OSError,
                RuntimeError,
                ValueError,
            ) as error:
                raise StagedEloPipelineError(
                    f"{label} fork failed: {type(error).__name__}: {error}"
                ) from error
        anchor = _mapping(metadata.get("anchor"), f"{label} fork anchor")
        if (
            anchor.get("model_identity") != selected_identity
            or anchor.get("model_step") != champion.get("model_step")
            or metadata.get("source_winner_snapshot") != snapshot
        ):
            raise StagedEloPipelineError(
                f"{label} fork does not preserve the verified winner anchor"
            )
        profile_path = run_root / "profile-elo-ablation.yaml"
        try:
            warm_start = warm_starter(
                run_root,
                profile_path,
                apply=True,
                replace_existing=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise StagedEloPipelineError(
                f"{label} champion warm-start failed: {type(error).__name__}: {error}"
            ) from error
        marker = _read_json(run_root / "learner" / "champion-warm-start.json")
        recovery = _read_json(run_root / "learner" / "recovery.json")
        cutover = _read_json(run_root / "learner" / "resume-cutover.json")
        if (
            marker.get("source_model_identity") != selected_identity
            or marker.get("source_model_step") != champion.get("model_step")
            or recovery.get("step") != champion.get("model_step")
            or cutover.get("step") != champion.get("model_step")
        ):
            raise StagedEloPipelineError(
                f"{label} did not rebase recovery state to the verified winner"
            )
        metadata["starting_candidate"] = {
            key: champion.get(key)
            for key in ("model_identity", "model_step", "updated_ns")
        }
        metadata["staged_warm_start"] = warm_start
        atomic_json(metadata_path, metadata)
        forked.append(
            {
                "treatment": label,
                "run_root": str(run_root),
                "metadata": str(metadata_path),
                "warm_start": warm_start,
            }
        )

    state_path = (
        Path(_string(pipeline.get("state_path"), "pipeline state path"))
        .expanduser()
        .resolve()
    )
    state: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report": PIPELINE_STATE_REPORT,
        "status": "downstream_prepared",
        "pipeline": str(path),
        "pipeline_sha256": _sha256(path),
        "transition": {
            "stage_index": stage_index,
            "from": upstream_name,
            "to": downstream_name,
        },
        "upstream": {
            "deployment_manifest": str(deployment_manifest),
            "comparison": str(comparison_path),
            "comparison_sha256": _sha256(comparison_path),
            "winner_snapshot": snapshot,
        },
        "downstream": {
            "plan": str(plan_path),
            "plan_sha256": _sha256(plan_path),
            "futility_policy": policy,
            "forks": forked,
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(state_path, state)
    return state


def _error_document(error: Exception) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report": PIPELINE_STATE_REPORT,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        state = advance_staged_elo_pipeline(
            arguments.pipeline,
            stage_index=arguments.stage_index,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(json.dumps(_error_document(error), sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(state, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
