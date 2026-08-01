from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_staged_elo_pipeline import (
    StagedEloPipelineError,
    advance_staged_elo_pipeline,
    build_futility_policy,
    evaluate_futility,
)

CONFIGS = Path(__file__).parents[1] / "configs"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _selected_source(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "selected"
    run_path = source / "run.json"
    champion_path = source / "learner" / "champion.json"
    run = {
        "schema_version": 1,
        "run_id": "shared-run",
        "generation_family": "selected-family",
        "created_ns": 1,
    }
    champion = {
        "model_identity": "selected-champion",
        "model_step": 367_907,
        "updated_ns": 2,
    }
    _write_json(run_path, run)
    _write_json(champion_path, champion)
    _write_json(
        source / "learner" / "candidate.json",
        {
            "model_identity": "selected-candidate",
            "model_step": 368_000,
            "updated_ns": 3,
        },
    )
    snapshot: dict[str, object] = {
        "schema_version": 1,
        "status": "verified",
        "label": "upstream-winner",
        "run_root": str(source.resolve()),
        "run_identity": {
            key: run[key] for key in ("run_id", "generation_family", "created_ns")
        },
        "run_identity_artifact": {
            "path": str(run_path.resolve()),
            "sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        },
        "champion": champion,
        "champion_pointer_artifact": {
            "path": str(champion_path.resolve()),
            "sha256": hashlib.sha256(champion_path.read_bytes()).hexdigest(),
        },
        "source_anchor": {
            "model_identity": "previous-champion",
            "model_step": 364_000,
        },
        "selection": "guarded_chronological_champion_frontier",
    }
    return source, snapshot


def _pipeline(
    tmp_path: Path,
    *,
    selector_status: str = "verified",
    expected_anchor: str = "selected-champion",
    weighted: bool = False,
) -> tuple[Path, Path, Path]:
    _source, snapshot = _selected_source(tmp_path)
    comparison = tmp_path / "upstream-comparison.json"
    _write_json(
        comparison,
        {
            "schema_version": 1,
            "report": "startrain-elo-ablation-comparison",
            "status": "complete",
            "selector": {
                "status": selector_status,
                "winner_snapshot": snapshot if selector_status == "verified" else None,
            },
        },
    )
    deployment = tmp_path / "upstream-deployment.json"
    _write_json(
        deployment,
        {
            "schema_version": 1,
            "report": "startrain-elo-ablation-deployment",
            "queue": {"comparison_output": str(comparison)},
        },
    )
    downstream_output = tmp_path / "downstream-profiles"
    pipeline = tmp_path / "pipeline.json"
    _write_json(
        pipeline,
        {
            "schema_version": 1,
            "report": "startrain-staged-elo-pipeline",
            "state_path": str(tmp_path / "pipeline-state.json"),
            "stages": [
                {
                    "name": "learning-rate",
                    "deployment_manifest": str(deployment),
                },
                {
                    "name": "replay",
                    "prepare": {
                        "base_config": str(CONFIGS / "h100-8gpu-throughput.yaml"),
                        "output_dir": str(downstream_output),
                        "run_root_parent": str(tmp_path / "downstream-runs"),
                        "run_id": "shared-run",
                        "prefix": "replay",
                        "seed": 47,
                        "wall_budget_hours": 1,
                        "leaf_budget": 10,
                        "guard_floor_elo": -35,
                        "treatments": ["weighted-control" if weighted else "control"],
                        "expected_anchor_identity": expected_anchor,
                        **(
                            {
                                "guard_rings": [],
                                "promotion_objective": "weighted_aggregate",
                            }
                            if weighted
                            else {}
                        ),
                    },
                },
            ],
        },
    )
    return pipeline, downstream_output, comparison


def _completed_queue(_manifest: Path) -> dict[str, object]:
    return {
        "queue_status": "completed",
        "finalization": {
            "status": "completed",
            "comparison_status": "complete",
        },
    }


def _winner_warm_start(
    run_root: Path,
    _profile: Path,
    *,
    apply: bool,
    replace_existing: bool,
) -> dict[str, object]:
    assert apply is True
    assert replace_existing is True
    champion = json.loads(
        (run_root / "learner" / "champion.json").read_text(encoding="utf-8")
    )
    marker = {
        "format": "startrain.champion-warm-start",
        "schema_version": 1,
        "status": "active",
        "source_model_identity": champion["model_identity"],
        "source_model_step": champion["model_step"],
    }
    _write_json(run_root / "learner" / "champion-warm-start.json", marker)
    _write_json(
        run_root / "learner" / "recovery.json",
        {"step": champion["model_step"]},
    )
    _write_json(
        run_root / "learner" / "resume-cutover.json",
        {"step": champion["model_step"]},
    )
    return {"status": "ok", "mode": "apply", "warm_start": marker}


def test_staged_pipeline_prepares_and_forks_only_verified_winner(
    tmp_path: Path,
) -> None:
    pipeline, output, _comparison = _pipeline(tmp_path)

    state = advance_staged_elo_pipeline(
        pipeline,
        stage_index=0,
        queue_runner=_completed_queue,
        warm_starter=_winner_warm_start,
    )
    plan = json.loads((output / "ablation-plan.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (
            tmp_path / "downstream-runs" / "replay-control-seed47" / "ablation.json"
        ).read_text(encoding="utf-8")
    )

    assert state["status"] == "downstream_prepared"
    assert plan["source_winner_snapshot"]["champion"]["model_step"] == 367_907
    assert plan["futility_policy"]["decision_scope"] == "stop_only"
    assert plan["futility_policy"]["evidence"] == ("anytime_valid_confidence_sequences")
    assert metadata["anchor"]["model_identity"] == "selected-champion"
    assert metadata["source_winner_snapshot"] == plan["source_winner_snapshot"]
    assert metadata["starting_candidate"]["model_step"] == 367_907
    assert metadata["staged_warm_start"]["status"] == "ok"
    assert (
        json.loads(
            (
                tmp_path
                / "downstream-runs"
                / "replay-control-seed47"
                / "learner"
                / "recovery.json"
            ).read_text(encoding="utf-8")
        )["step"]
        == 367_907
    )


def test_staged_pipeline_rejects_stale_anchor_before_preparation(
    tmp_path: Path,
) -> None:
    pipeline, output, _comparison = _pipeline(
        tmp_path,
        expected_anchor="stale-champion",
    )

    with pytest.raises(StagedEloPipelineError, match="stale anchor identity"):
        advance_staged_elo_pipeline(
            pipeline,
            stage_index=0,
            queue_runner=_completed_queue,
        )

    assert not output.exists()


def test_staged_pipeline_prepares_weighted_objective_without_guard_rings(
    tmp_path: Path,
) -> None:
    pipeline, output, _comparison = _pipeline(tmp_path, weighted=True)

    state = advance_staged_elo_pipeline(
        pipeline,
        stage_index=0,
        queue_runner=_completed_queue,
        warm_starter=_winner_warm_start,
    )
    plan = json.loads((output / "ablation-plan.json").read_text(encoding="utf-8"))

    assert state["status"] == "downstream_prepared"
    assert plan["guard_rings"] == []
    assert plan["promotion_objective"] == "weighted_aggregate"
    assert plan["futility_policy"]["guard_regression"]["rings"] == []
    assert plan["futility_policy"]["control_comparison"]["objective"] == (
        "weighted_aggregate"
    )
    assert plan["futility_policy"]["minimum_decisive_games"] == 15


def test_staged_pipeline_rejects_unverified_selector(tmp_path: Path) -> None:
    pipeline, output, _comparison = _pipeline(
        tmp_path,
        selector_status="unavailable",
    )

    with pytest.raises(StagedEloPipelineError, match="not verified"):
        advance_staged_elo_pipeline(
            pipeline,
            stage_index=0,
            queue_runner=_completed_queue,
        )

    assert not output.exists()


def test_staged_pipeline_rejects_changed_reused_stage_spec(tmp_path: Path) -> None:
    pipeline, _output, _comparison = _pipeline(tmp_path)
    advance_staged_elo_pipeline(
        pipeline,
        stage_index=0,
        queue_runner=_completed_queue,
        warm_starter=_winner_warm_start,
    )
    document = json.loads(pipeline.read_text(encoding="utf-8"))
    document["stages"][1]["prepare"]["leaf_budget"] = 11
    _write_json(pipeline, document)

    with pytest.raises(StagedEloPipelineError, match="specification changed"):
        advance_staged_elo_pipeline(
            pipeline,
            stage_index=0,
            queue_runner=_completed_queue,
            warm_starter=_winner_warm_start,
        )


def test_futility_evaluator_is_stop_only_and_requires_anytime_evidence() -> None:
    policy = build_futility_policy(guard_floor_elo=-35, minimum_decisive_games=20)
    base_evidence = {
        "anytime_valid": True,
        "method": "anytime_valid_confidence_sequence",
        "decisive_games": 20,
        "guard_rings": {
            "4": {"anytime_upper_elo": 10},
            "6": {"anytime_upper_elo": 10},
            "8": {"anytime_upper_elo": 10},
        },
        "ring_10": {
            "anytime_upper_elo": 20,
            "control_anytime_lower_elo": 0,
        },
    }

    keep = evaluate_futility(policy, base_evidence)
    regression = evaluate_futility(
        policy,
        {
            **base_evidence,
            "guard_rings": {
                **base_evidence["guard_rings"],
                "6": {"anytime_upper_elo": -36},
            },
        },
    )
    no_anytime = evaluate_futility(
        policy,
        {**base_evidence, "anytime_valid": False},
    )
    fixed_time = evaluate_futility(
        policy,
        {**base_evidence, "method": "fixed_time_interval"},
    )
    cannot_beat = evaluate_futility(
        policy,
        {
            **base_evidence,
            "ring_10": {
                "anytime_upper_elo": -1,
                "control_anytime_lower_elo": 0,
            },
        },
    )

    assert keep["decision"] == "continue"
    assert regression["decision"] == "stop_for_futility"
    assert regression["reason"] == "definitive_ring_regression"
    assert cannot_beat["reason"] == "cannot_beat_control"
    assert no_anytime["decision"] == "continue"
    assert fixed_time["decision"] == "continue"
    assert all(
        result["promotion_allowed"] is False
        for result in (keep, regression, no_anytime, fixed_time, cannot_beat)
    )


def test_weighted_futility_accepts_empty_guards_and_aggregate_evidence() -> None:
    policy = build_futility_policy(
        guard_rings=(),
        guard_floor_elo=-35,
        minimum_decisive_games=15,
        control_objective="weighted_aggregate",
    )
    evidence = {
        "anytime_valid": True,
        "method": "anytime_valid_confidence_sequence",
        "weighted_aggregate": {
            "complete_blocks": 15,
            "anytime_elo_interval": [-20, -1],
            "control_anytime_elo_interval": [0, 20],
        },
    }

    result = evaluate_futility(policy, evidence)

    assert policy["guard_regression"]["rings"] == []
    assert result["decision"] == "stop_for_futility"
    assert result["reason"] == "cannot_beat_control"
    assert result["control_objective"] == "weighted_aggregate"
    assert result["promotion_allowed"] is False
