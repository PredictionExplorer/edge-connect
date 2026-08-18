#!/usr/bin/env python3
"""Advance one Elo stage only from a verified upstream winner snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from startrain.runtime import atomic_json

if __package__:
    from .compare_elo_ablation_seeds import (
        REQUIRED_SEEDS,
        PinnedComparison,
        build_cross_seed_comparison,
    )
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
    from .run_elo_ablation_queue import (
        exclusive_execution_lock,
        exclusive_queue_lock,
        run_ablation_queue,
    )
else:
    from compare_elo_ablation_seeds import (
        REQUIRED_SEEDS,
        PinnedComparison,
        build_cross_seed_comparison,
    )
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
    from run_elo_ablation_queue import (
        exclusive_execution_lock,
        exclusive_queue_lock,
        run_ablation_queue,
    )

SCHEMA_VERSION = 1
PIPELINE_REPORT = "startrain-staged-elo-pipeline"
PIPELINE_STATE_REPORT = "startrain-staged-elo-pipeline-state"
FUTILITY_REPORT = "startrain-elo-futility-policy"
FUTILITY_EVALUATION_REPORT = "startrain-elo-futility-evaluation"
CONFIRMATION_CAMPAIGN_REPORT = "startrain-elo-confirmation-campaign"
CONFIRMATION_CAMPAIGN_STATE_REPORT = "startrain-elo-confirmation-campaign-state"
CONFIRMATION_HOLD_REPORT = "startrain-elo-seed-boundary-hold"

QueueRunner = Callable[..., dict[str, object]]
CrossSeedBuilder = Callable[..., dict[str, object]]
Preparer = Callable[..., dict[str, object]]
Forker = Callable[..., dict[str, object]]
WarmStarter = Callable[..., dict[str, object]]


class StagedEloPipelineError(RuntimeError):
    """Raised when a stage transition is unverified, stale, or malformed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pipeline", type=Path)
    source.add_argument("--confirmation-campaign", type=Path)
    parser.add_argument("--stage-index", type=int)
    parser.add_argument(
        "--seed-boundary-action",
        choices=("run", "request", "release", "status"),
        default="run",
    )
    parser.add_argument("--hold-reason")
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


def _read_json_with_digest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        loaded = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise StagedEloPipelineError(
            f"cannot read JSON object {path}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise StagedEloPipelineError(f"{path} must contain a JSON object")
    return loaded, hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable_json(path: Path, document: Mapping[str, object]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp")
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


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


def _confirmation_hold_path(
    campaign: Mapping[str, object],
) -> Path | None:
    raw = campaign.get("seed_boundary_hold_path")
    if raw is None:
        return None
    return Path(_string(raw, "seed boundary hold path")).expanduser().resolve()


def _read_confirmation_hold(
    path: Path,
    *,
    campaign_sha256: str,
) -> tuple[dict[str, Any], str] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise StagedEloPipelineError("seed boundary hold path is unsafe")
    document, digest = _read_json_with_digest(path)
    if set(document) != {
        "schema_version",
        "report",
        "status",
        "campaign_sha256",
        "created_ns",
        "reason",
    }:
        raise StagedEloPipelineError("seed boundary hold fields are incompatible")
    created_ns = document.get("created_ns")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("report") != CONFIRMATION_HOLD_REPORT
        or document.get("status") != "requested"
        or document.get("campaign_sha256") != campaign_sha256
        or isinstance(created_ns, bool)
        or not isinstance(created_ns, int)
        or created_ns <= 0
        or not isinstance(document.get("reason"), str)
        or not document["reason"]
    ):
        raise StagedEloPipelineError("seed boundary hold is invalid")
    return document, digest


def _effective_hold_boundary(
    state_path: Path,
    *,
    campaign_sha256: str,
) -> int | None:
    if not state_path.is_file() or state_path.is_symlink():
        return None
    state = _read_json(state_path)
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("report") != CONFIRMATION_CAMPAIGN_STATE_REPORT
        or state.get("campaign_sha256") != campaign_sha256
    ):
        raise StagedEloPipelineError(
            "confirmation campaign state does not match hold request"
        )
    seeds = _list(state.get("seeds"), "confirmation campaign seed state")
    active: list[int] = []
    completed: list[int] = []
    for record in seeds:
        if not isinstance(record, Mapping):
            continue
        seed = record.get("seed")
        if type(seed) is not int:
            continue
        if record.get("status") in {"launch_committed", "running"}:
            active.append(seed)
        elif record.get("status") == "completed":
            completed.append(seed)
    if active:
        return max(active)
    return max(completed) if completed else None


def _manage_confirmation_hold_locked(
    campaign_path: Path,
    *,
    action: str,
    reason: str | None = None,
    expected_campaign_sha256: str,
    expected_hold_path: Path,
) -> dict[str, object]:
    path = campaign_path.expanduser().resolve()
    campaign, campaign_sha256 = _read_json_with_digest(path)
    if campaign_sha256 != expected_campaign_sha256:
        raise StagedEloPipelineError(
            "confirmation campaign changed before hold lock acquisition"
        )
    if (
        campaign.get("schema_version") != SCHEMA_VERSION
        or campaign.get("report") != CONFIRMATION_CAMPAIGN_REPORT
    ):
        raise StagedEloPipelineError("unsupported confirmation campaign")
    hold_path = _confirmation_hold_path(campaign)
    if hold_path is None:
        raise StagedEloPipelineError(
            "confirmation campaign has no seed boundary hold path"
        )
    if hold_path != expected_hold_path:
        raise StagedEloPipelineError(
            "seed boundary hold path changed before lock acquisition"
        )
    state_path = (
        Path(_string(campaign.get("state_path"), "campaign state path"))
        .expanduser()
        .resolve()
    )
    if hold_path.parent != state_path.parent or hold_path == state_path:
        raise StagedEloPipelineError(
            "seed boundary hold must be a distinct file beside campaign state"
        )
    existing = _read_confirmation_hold(
        hold_path,
        campaign_sha256=campaign_sha256,
    )
    if action == "request":
        if reason is None or not reason.strip() or reason.strip() != reason:
            raise StagedEloPipelineError(
                "requesting a seed boundary hold requires a normalized reason"
            )
        if existing is not None and existing[0].get("reason") != reason:
            raise StagedEloPipelineError(
                "existing seed boundary hold has a different reason"
            )
        if existing is None:
            hold_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(
                hold_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "report": CONFIRMATION_HOLD_REPORT,
                    "status": "requested",
                    "campaign_sha256": campaign_sha256,
                    "created_ns": time.time_ns(),
                    "reason": reason,
                },
            )
            existing = _read_confirmation_hold(
                hold_path,
                campaign_sha256=campaign_sha256,
            )
        assert existing is not None
        document, digest = existing
        return {
            "status": "requested",
            "path": str(hold_path),
            "sha256": digest,
            "hold": document,
            "effective_after_seed": _effective_hold_boundary(
                state_path,
                campaign_sha256=campaign_sha256,
            ),
        }
    if reason is not None:
        raise StagedEloPipelineError("--hold-reason is valid only with request")
    if action == "release":
        if existing is None:
            return {"status": "not_requested", "path": str(hold_path)}
        document, digest = existing
        hold_path.unlink()
        descriptor = os.open(hold_path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {
            "status": "released",
            "path": str(hold_path),
            "released_sha256": digest,
            "hold": document,
        }
    if action == "status":
        if existing is None:
            return {"status": "not_requested", "path": str(hold_path)}
        document, digest = existing
        return {
            "status": "requested",
            "path": str(hold_path),
            "sha256": digest,
            "hold": document,
            "effective_after_seed": _effective_hold_boundary(
                state_path,
                campaign_sha256=campaign_sha256,
            ),
        }
    raise StagedEloPipelineError(f"unsupported seed boundary action: {action}")


def manage_confirmation_hold(
    campaign_path: Path,
    *,
    action: str,
    reason: str | None = None,
) -> dict[str, object]:
    path = campaign_path.expanduser().resolve()
    campaign, campaign_sha256 = _read_json_with_digest(path)
    if (
        campaign.get("schema_version") != SCHEMA_VERSION
        or campaign.get("report") != CONFIRMATION_CAMPAIGN_REPORT
    ):
        raise StagedEloPipelineError("unsupported confirmation campaign")
    hold_path = _confirmation_hold_path(campaign)
    if hold_path is None:
        raise StagedEloPipelineError(
            "confirmation campaign has no seed boundary hold path"
        )
    lock = exclusive_queue_lock(hold_path)
    lock.__enter__()
    try:
        return _manage_confirmation_hold_locked(
            path,
            action=action,
            reason=reason,
            expected_campaign_sha256=campaign_sha256,
            expected_hold_path=hold_path,
        )
    finally:
        lock.__exit__(None, None, None)


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


def _sha256_pin(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise StagedEloPipelineError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _campaign_seed_specs(
    campaign: Mapping[str, object],
) -> tuple[list[dict[str, object]], Path, set[Path]]:
    raw_seeds = _list(campaign.get("seeds"), "confirmation campaign seeds")
    specs = []
    observed_seeds = []
    execution_locks: set[Path] = set()
    protected_paths: list[Path] = []
    for index, raw_seed in enumerate(raw_seeds):
        seed_spec = _mapping(raw_seed, f"confirmation seed {index}")
        seed = seed_spec.get("seed")
        if type(seed) is not int:
            raise StagedEloPipelineError("confirmation seed must be an integer")
        manifest_path = (
            Path(
                _string(
                    seed_spec.get("deployment_manifest"),
                    f"seed {seed} deployment manifest",
                )
            )
            .expanduser()
            .resolve()
        )
        manifest_sha256 = _sha256_pin(
            seed_spec.get("deployment_sha256"),
            f"seed {seed} deployment SHA-256",
        )
        if not manifest_path.is_file():
            raise StagedEloPipelineError(
                f"seed {seed} deployment manifest is missing or changed"
            )
        manifest, observed_manifest_sha256 = _read_json_with_digest(manifest_path)
        if observed_manifest_sha256 != manifest_sha256:
            raise StagedEloPipelineError(
                f"seed {seed} deployment manifest is missing or changed"
            )
        queue = _mapping(manifest.get("queue"), f"seed {seed} deployment queue")
        if queue.get("seed") != seed:
            raise StagedEloPipelineError(
                f"seed {seed} deployment queue declares a different seed"
            )
        execution_lock = (
            Path(
                _string(
                    queue.get("execution_lock_path"),
                    f"seed {seed} execution lock",
                )
            )
            .expanduser()
            .resolve()
        )
        comparison_output = (
            Path(
                _string(
                    queue.get("comparison_output"),
                    f"seed {seed} comparison output",
                )
            )
            .expanduser()
            .resolve()
        )
        queue_state_path = (
            Path(
                _string(
                    queue.get("state_path"),
                    f"seed {seed} queue state",
                )
            )
            .expanduser()
            .resolve()
        )
        handoff_output = (
            Path(
                _string(
                    queue.get("continuity_handoff_output"),
                    f"seed {seed} continuity handoff output",
                )
            )
            .expanduser()
            .resolve()
        )
        execution_locks.add(execution_lock)
        protected_paths.extend(
            (
                manifest_path,
                queue_state_path,
                comparison_output,
                handoff_output,
            )
        )
        observed_seeds.append(seed)
        specs.append(
            {
                "seed": seed,
                "deployment_manifest": manifest_path,
                "deployment_sha256": manifest_sha256,
                "comparison_output": comparison_output,
                "execution_lock": execution_lock,
                "queue_state_path": queue_state_path,
                "handoff_output": handoff_output,
            }
        )
    if tuple(sorted(observed_seeds)) != REQUIRED_SEEDS or len(
        set(observed_seeds)
    ) != len(observed_seeds):
        raise StagedEloPipelineError(
            f"confirmation campaign seeds must be exactly {list(REQUIRED_SEEDS)}"
        )
    if len(execution_locks) != 1:
        raise StagedEloPipelineError(
            "all confirmation seeds must share one host execution lock"
        )
    if len(set(protected_paths)) != len(protected_paths):
        raise StagedEloPipelineError(
            "confirmation seed manifests, states, comparisons, and handoffs "
            "must use distinct paths"
        )

    def seed_order(spec: Mapping[str, object]) -> int:
        raw_seed = spec.get("seed")
        assert type(raw_seed) is int
        return raw_seed

    specs.sort(key=seed_order)
    return specs, next(iter(execution_locks)), set(protected_paths)


def _campaign_state(
    *,
    state_path: Path,
    campaign_path: Path,
    campaign_sha256: str,
) -> dict[str, object]:
    if state_path.is_file():
        state = _read_json(state_path)
        if (
            state.get("schema_version") != SCHEMA_VERSION
            or state.get("report") != CONFIRMATION_CAMPAIGN_STATE_REPORT
            or state.get("campaign") != str(campaign_path)
            or state.get("campaign_sha256") != campaign_sha256
        ):
            raise StagedEloPipelineError(
                "confirmation campaign state does not match its immutable campaign"
            )
        if not isinstance(state.get("seeds"), list):
            raise StagedEloPipelineError("confirmation campaign seed state is invalid")
        return state
    return {
        "schema_version": SCHEMA_VERSION,
        "report": CONFIRMATION_CAMPAIGN_STATE_REPORT,
        "status": "pending",
        "campaign": str(campaign_path),
        "campaign_sha256": campaign_sha256,
        "seeds": [],
        "cross_seed": None,
        "automatic_adoption_authorized": False,
    }


def _campaign_seed_record(
    state: Mapping[str, object],
    seed: int,
) -> dict[str, object] | None:
    records = state.get("seeds")
    if not isinstance(records, list):
        return None
    return next(
        (
            dict(record)
            for record in records
            if isinstance(record, Mapping) and record.get("seed") == seed
        ),
        None,
    )


def _upsert_campaign_seed(
    state: dict[str, object],
    record: Mapping[str, object],
) -> None:
    records = _list(state.get("seeds"), "confirmation campaign seed state")
    seed = record.get("seed")
    state["seeds"] = [
        *[
            existing
            for existing in records
            if not isinstance(existing, Mapping) or existing.get("seed") != seed
        ],
        dict(record),
    ]
    cast_records = state["seeds"]
    assert isinstance(cast_records, list)

    def seed_order(existing: object) -> int:
        raw_seed = existing.get("seed") if isinstance(existing, Mapping) else None
        return raw_seed if type(raw_seed) is int else -1

    cast_records.sort(key=seed_order)


def _remove_campaign_seed(state: dict[str, object], seed: int) -> None:
    records = _list(state.get("seeds"), "confirmation campaign seed state")
    state["seeds"] = [
        record
        for record in records
        if not isinstance(record, Mapping) or record.get("seed") != seed
    ]


def _pause_campaign_at_boundary(
    *,
    state: dict[str, object],
    state_path: Path,
    hold_path: Path,
    campaign_sha256: str,
    previous_seed: int,
    next_seed: int,
) -> bool:
    hold = _read_confirmation_hold(
        hold_path,
        campaign_sha256=campaign_sha256,
    )
    if hold is None:
        return False
    previous_record = _campaign_seed_record(state, previous_seed)
    if previous_record is None or previous_record.get("status") != "completed":
        raise StagedEloPipelineError(
            "seed boundary hold encountered before previous seed completed"
        )
    hold_document, hold_sha256 = hold
    next_record = _campaign_seed_record(state, next_seed)
    launch_committed_ns = (
        next_record.get("launch_committed_ns") if next_record is not None else None
    )
    created_ns = hold_document["created_ns"]
    assert isinstance(created_ns, int)
    if type(launch_committed_ns) is int and launch_committed_ns < created_ns:
        return False
    if next_record is not None and next_record.get("status") == "launch_committed":
        _remove_campaign_seed(state, next_seed)
    state.update(
        {
            "status": "paused",
            "paused_after_seed": previous_seed,
            "next_seed": next_seed,
            "paused_ns": time.time_ns(),
            "seed_boundary_hold": {
                "path": str(hold_path),
                "sha256": hold_sha256,
                "reason": hold_document["reason"],
            },
            "operator_resume_required": True,
            "continuity_fallback_expected": True,
            "automatic_adoption_authorized": False,
        }
    )
    state.pop("operator_execution_required", None)
    atomic_json(state_path, state)
    return True


def _verified_completed_queue(
    queue_state: Mapping[str, object],
    *,
    seed: int,
    handoff_output: Path,
    manifest_path: Path,
) -> None:
    finalization = _mapping(
        queue_state.get("finalization"),
        f"seed {seed} queue finalization",
    )
    if (
        queue_state.get("queue_status") != "completed"
        or finalization.get("status") != "completed"
        or finalization.get("comparison_status") != "complete"
    ):
        raise StagedEloPipelineError(
            f"seed {seed} queue did not complete with a verified comparison"
        )
    arms = _list(queue_state.get("arms"), f"seed {seed} queue arms")
    if not arms:
        raise StagedEloPipelineError(f"seed {seed} queue has no completed arms")
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise StagedEloPipelineError(f"seed {seed} queue arm is invalid")
        cutoff_ns = arm.get("measurement_cutoff_ns")
        released_ns = arm.get("resource_released_ns")
        teardown = arm.get("teardown")
        completion_status = arm.get("completion_status")
        clean_completion = (
            completion_status == "complete" and arm.get("teardown_status") == "clean"
        )
        verified_warning = (
            completion_status == "complete_with_warning"
            and arm.get("failure_phase") == "post_cutoff"
            and arm.get("integrity_status")
            in {"ok", "pass", "passed", "valid", "verified", "healthy"}
            and arm.get("teardown_status") not in {"resources_not_released", None}
        )
        if (
            arm.get("status") != "completed"
            or not (clean_completion or verified_warning)
            or type(cutoff_ns) is not int
            or cutoff_ns <= 0
            or type(released_ns) is not int
            or released_ns < cutoff_ns
            or not isinstance(teardown, Mapping)
            or teardown.get("process_group_released") is not True
            or teardown.get("resource_released_ns") != released_ns
        ):
            raise StagedEloPipelineError(
                f"seed {seed} queue did not prove ordered arm resource release"
            )
    handoff = _mapping(
        queue_state.get("continuity_handoff"),
        f"seed {seed} continuity handoff",
    )
    if (
        handoff.get("requested") is not True
        or handoff.get("status") != "requested"
        or handoff.get("action") != "request_fallback"
        or handoff.get("requested_action") != "reconcile_training_continuity"
        or handoff.get("reason") != "queue_completed"
        or Path(str(handoff.get("path"))).expanduser().resolve() != handoff_output
        or not handoff_output.is_file()
    ):
        raise StagedEloPipelineError(
            f"seed {seed} queue did not preserve verified LKG fallback"
        )
    persisted_handoff = _read_json(handoff_output)
    source = persisted_handoff.get("source")
    if (
        persisted_handoff.get("schema_version") != SCHEMA_VERSION
        or persisted_handoff.get("report") != "startrain-continuity-handoff-request"
        or persisted_handoff.get("requested") is not True
        or persisted_handoff.get("status") != "requested"
        or persisted_handoff.get("action") != "request_fallback"
        or persisted_handoff.get("requested_action") != "reconcile_training_continuity"
        or persisted_handoff.get("reason") != "queue_completed"
        or not isinstance(source, Mapping)
        or source.get("kind") != "elo_ablation_queue"
        or Path(str(source.get("manifest"))).expanduser().resolve()
        != manifest_path.expanduser().resolve()
        or source.get("queue_status") != "completed"
    ):
        raise StagedEloPipelineError(
            f"seed {seed} durable continuity fallback is invalid"
        )


def run_confirmation_campaign(
    campaign_path: Path,
    *,
    queue_runner: QueueRunner = run_ablation_queue,
    cross_seed_builder: CrossSeedBuilder = build_cross_seed_comparison,
) -> dict[str, object]:
    """Run three pinned seed queues under one host lock, then compare them."""
    path = campaign_path.expanduser().resolve()
    campaign, campaign_sha256 = _read_json_with_digest(path)
    if (
        campaign.get("schema_version") != SCHEMA_VERSION
        or campaign.get("report") != CONFIRMATION_CAMPAIGN_REPORT
    ):
        raise StagedEloPipelineError("unsupported confirmation campaign")
    state_path = (
        Path(_string(campaign.get("state_path"), "campaign state path"))
        .expanduser()
        .resolve()
    )
    hold_path = _confirmation_hold_path(campaign)
    if hold_path is not None and (
        hold_path.parent != state_path.parent or hold_path == state_path
    ):
        raise StagedEloPipelineError(
            "seed boundary hold must be a distinct file beside campaign state"
        )
    cross_seed = _mapping(campaign.get("cross_seed"), "campaign cross-seed plan")
    policy_path = (
        Path(_string(cross_seed.get("policy"), "campaign adoption policy"))
        .expanduser()
        .resolve()
    )
    policy_sha256 = _sha256_pin(
        cross_seed.get("policy_sha256"),
        "campaign adoption policy SHA-256",
    )
    output_path = (
        Path(_string(cross_seed.get("output"), "campaign cross-seed output"))
        .expanduser()
        .resolve()
    )
    if not policy_path.is_file() or _sha256(policy_path) != policy_sha256:
        raise StagedEloPipelineError("campaign adoption policy is missing or changed")
    specs, execution_lock_path, protected_paths = _campaign_seed_specs(campaign)
    all_paths = [
        *protected_paths,
        path,
        state_path,
        policy_path,
        output_path,
        execution_lock_path,
        *([hold_path] if hold_path is not None else []),
    ]
    if len(set(all_paths)) != len(all_paths):
        raise StagedEloPipelineError(
            "campaign, state, policy, lock, manifests, queue states, comparisons, "
            "handoffs, and cross-seed output must use distinct paths"
        )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_state_lock = exclusive_queue_lock(state_path)
    campaign_state_lock.__enter__()
    state: dict[str, object] | None = None
    pinned_comparisons: list[PinnedComparison] = []
    try:
        with exclusive_execution_lock(execution_lock_path) as execution_lock_lease:
            locked_campaign, locked_campaign_sha256 = _read_json_with_digest(path)
            if locked_campaign_sha256 != campaign_sha256 or locked_campaign != campaign:
                raise StagedEloPipelineError(
                    "confirmation campaign changed before lock acquisition"
                )
            if _sha256(policy_path) != policy_sha256:
                raise StagedEloPipelineError(
                    "campaign adoption policy changed before execution"
                )
            state = _campaign_state(
                state_path=state_path,
                campaign_path=path,
                campaign_sha256=campaign_sha256,
            )
            previous_cross_seed = state.get("cross_seed")
            state["status"] = "running"
            for stale in (
                "error",
                "continuity_fallback_expected",
                "operator_execution_required",
            ):
                state.pop(stale, None)
            atomic_json(state_path, state)
            for index, spec in enumerate(specs):
                raw_seed = spec["seed"]
                assert type(raw_seed) is int
                seed = raw_seed
                manifest_path = spec["deployment_manifest"]
                comparison_output = spec["comparison_output"]
                queue_state_path = spec["queue_state_path"]
                handoff_output = spec["handoff_output"]
                assert isinstance(manifest_path, Path)
                assert isinstance(comparison_output, Path)
                assert isinstance(queue_state_path, Path)
                assert isinstance(handoff_output, Path)
                manifest_sha256 = str(spec["deployment_sha256"])
                existing = _campaign_seed_record(state, seed)
                if existing is not None and existing.get("status") == "completed":
                    comparison_sha256 = _sha256_pin(
                        existing.get("comparison_sha256"),
                        f"seed {seed} comparison SHA-256",
                    )
                    if (
                        existing.get("deployment_sha256") != manifest_sha256
                        or not comparison_output.is_file()
                        or _sha256(comparison_output) != comparison_sha256
                    ):
                        raise StagedEloPipelineError(
                            f"seed {seed} completed campaign evidence changed"
                        )
                    if not queue_state_path.is_file():
                        raise StagedEloPipelineError(
                            f"seed {seed} completed queue state is missing"
                        )
                    _verified_completed_queue(
                        _read_json(queue_state_path),
                        seed=seed,
                        handoff_output=handoff_output,
                        manifest_path=manifest_path,
                    )
                    pinned_comparisons.append(
                        PinnedComparison(seed, comparison_output, comparison_sha256)
                    )
                    continue
                if _sha256(manifest_path) != manifest_sha256:
                    raise StagedEloPipelineError(
                        f"seed {seed} deployment manifest changed before execution"
                    )
                existing_launch_ns = (
                    existing.get("launch_committed_ns")
                    if existing is not None
                    and existing.get("status") == "launch_committed"
                    else None
                )
                launch_committed_ns = (
                    existing_launch_ns
                    if type(existing_launch_ns) is int
                    else time.time_ns()
                )
                for stale in (
                    "operator_resume_required",
                    "paused_after_seed",
                    "next_seed",
                    "paused_ns",
                    "seed_boundary_hold",
                ):
                    state.pop(stale, None)
                _upsert_campaign_seed(
                    state,
                    {
                        "seed": seed,
                        "status": "launch_committed",
                        "launch_committed_ns": launch_committed_ns,
                        "deployment_manifest": str(manifest_path),
                        "deployment_sha256": manifest_sha256,
                        "comparison": str(comparison_output),
                    },
                )
                atomic_json(state_path, state)
                if index > 0 and hold_path is not None:
                    previous_raw_seed = specs[index - 1]["seed"]
                    assert type(previous_raw_seed) is int
                    if _pause_campaign_at_boundary(
                        state=state,
                        state_path=state_path,
                        hold_path=hold_path,
                        campaign_sha256=campaign_sha256,
                        previous_seed=previous_raw_seed,
                        next_seed=seed,
                    ):
                        return state
                queue_state = queue_runner(
                    manifest_path,
                    execution_lock_lease=execution_lock_lease,
                    expected_manifest_sha256=manifest_sha256,
                )
                _verified_completed_queue(
                    queue_state,
                    seed=seed,
                    handoff_output=handoff_output,
                    manifest_path=manifest_path,
                )
                if not queue_state_path.is_file():
                    raise StagedEloPipelineError(
                        f"seed {seed} durable queue state is missing"
                    )
                _verified_completed_queue(
                    _read_json(queue_state_path),
                    seed=seed,
                    handoff_output=handoff_output,
                    manifest_path=manifest_path,
                )
                if not comparison_output.is_file():
                    raise StagedEloPipelineError(
                        f"seed {seed} comparison output is missing"
                    )
                comparison_sha256 = _sha256(comparison_output)
                pinned_comparisons.append(
                    PinnedComparison(seed, comparison_output, comparison_sha256)
                )
                _upsert_campaign_seed(
                    state,
                    {
                        "seed": seed,
                        "status": "completed",
                        "deployment_manifest": str(manifest_path),
                        "deployment_sha256": manifest_sha256,
                        "comparison": str(comparison_output),
                        "comparison_sha256": comparison_sha256,
                        "queue_status": queue_state.get("queue_status"),
                        "resource_release_verified": True,
                        "continuity_fallback_preserved": True,
                    },
                )
                atomic_json(state_path, state)
            report = cross_seed_builder(
                pinned_comparisons,
                policy_path=policy_path,
                policy_sha256=policy_sha256,
            )
            report_semantic_sha256 = hashlib.sha256(
                json.dumps(
                    report,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            previous_cross = (
                previous_cross_seed
                if isinstance(previous_cross_seed, Mapping)
                else None
            )
            if output_path.is_file():
                previous_status = (
                    previous_cross.get("status") if previous_cross is not None else None
                )
                previous_digest = (
                    previous_cross.get("sha256") if previous_cross is not None else None
                )
                previous_semantic_sha256 = (
                    previous_cross.get("report_semantic_sha256")
                    if previous_cross is not None
                    else None
                )
                pinned_completed = (
                    previous_status in {"eligible", "ineligible"}
                    and isinstance(previous_digest, str)
                    and _sha256(output_path) == previous_digest
                )
                recoverable_publish = (
                    previous_status == "publishing"
                    and previous_semantic_sha256 == report_semantic_sha256
                )
                if (
                    not (pinned_completed or recoverable_publish)
                    or _read_json(output_path) != report
                ):
                    raise StagedEloPipelineError(
                        "existing cross-seed output is unpinned or differs from evidence"
                    )
            else:
                state["cross_seed"] = {
                    "output": str(output_path),
                    "status": "publishing",
                    "report_semantic_sha256": report_semantic_sha256,
                }
                atomic_json(state_path, state)
                _write_immutable_json(output_path, report)
            output_sha256 = _sha256(output_path)
            state.update(
                {
                    "status": "completed",
                    "cross_seed": {
                        "output": str(output_path),
                        "sha256": output_sha256,
                        "status": report.get("status"),
                        "eligible": report.get("eligible"),
                        "report_semantic_sha256": report_semantic_sha256,
                    },
                    "automatic_adoption_authorized": False,
                    "operator_execution_required": True,
                }
            )
            state.pop("continuity_fallback_expected", None)
            atomic_json(state_path, state)
            return state
    except Exception as error:
        if state is not None:
            state.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "continuity_fallback_expected": True,
                    "automatic_adoption_authorized": False,
                }
            )
            state.pop("operator_execution_required", None)
            atomic_json(state_path, state)
        raise
    finally:
        campaign_state_lock.__exit__(None, None, None)


def _error_document(
    error: Exception,
    *,
    report: str = PIPELINE_STATE_REPORT,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report": report,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.confirmation_campaign is not None:
            if arguments.stage_index is not None:
                raise StagedEloPipelineError(
                    "--stage-index cannot be used with --confirmation-campaign"
                )
            if arguments.seed_boundary_action == "run":
                if arguments.hold_reason is not None:
                    raise StagedEloPipelineError(
                        "--hold-reason requires --seed-boundary-action request"
                    )
                state = run_confirmation_campaign(arguments.confirmation_campaign)
            else:
                state = manage_confirmation_hold(
                    arguments.confirmation_campaign,
                    action=arguments.seed_boundary_action,
                    reason=arguments.hold_reason,
                )
        else:
            if (
                arguments.seed_boundary_action != "run"
                or arguments.hold_reason is not None
            ):
                raise StagedEloPipelineError(
                    "seed boundary actions require --confirmation-campaign"
                )
            if arguments.stage_index is None:
                raise StagedEloPipelineError(
                    "--stage-index is required with --pipeline"
                )
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
        error_report = (
            CONFIRMATION_CAMPAIGN_STATE_REPORT
            if arguments.confirmation_campaign is not None
            else PIPELINE_STATE_REPORT
        )
        print(
            json.dumps(
                _error_document(error, report=error_report),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    print(json.dumps(state, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
