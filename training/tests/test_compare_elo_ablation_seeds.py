from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import pytest
import yaml
from startrain.arena import ARENA_RESULT_SCHEMA_VERSION
from startrain.arena import bounded_confidence_sequence, elo_from_probability

from scripts.compare_elo_ablation import (
    PAIR_VALID_ERROR_PROBABILITY_PER_SIDE,
    PAIR_VALID_LOWER_BOUND_METHOD,
)
from scripts.compare_elo_ablation_seeds import (
    LCB_GATE_METHOD,
    PER_SEED_REPORT,
    POLICY_REPORT,
    RANKING_METRIC,
    REQUIRED_CANARY_GATES,
    CrossSeedComparisonError,
    PinnedComparison,
    build_cross_seed_comparison,
)
from scripts.prepare_ablation_adoption import (
    INELIGIBLE_REPORT,
    PLAN_REPORT,
    prepare_ablation_adoption,
)

SOURCE_COMMIT = "a" * 40
HOUR_NS = 3_600_000_000_000
CONFIGS = Path(__file__).parents[1] / "configs"
COMMON_ANCHOR = "anchor-common"


@dataclass
class Evidence:
    policy: Path
    policy_sha256: str
    comparisons: list[PinnedComparison]
    roots: dict[int, dict[str, Path]]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(
    root: Path,
    *,
    seed: int,
    label: str,
    anchor: str,
) -> dict[str, object]:
    run_path = root / "run.json"
    champion_path = root / "learner" / "champion.json"
    run = {
        "schema_version": 1,
        "run_id": f"run-seed{seed}-{label}",
        "generation_family": f"family-seed{seed}",
        "created_ns": seed * HOUR_NS,
    }
    champion = {
        "schema_version": 2,
        "role": "champion",
        "model_identity": f"champion-seed{seed}-{label}",
        "model_step": 100,
        "updated_ns": seed * HOUR_NS + 1,
    }
    _write_json(run_path, run)
    _write_json(champion_path, champion)
    return {
        "schema_version": 1,
        "status": "verified",
        "label": label,
        "run_root": str(root.resolve()),
        "run_identity": {
            key: run[key] for key in ("run_id", "generation_family", "created_ns")
        },
        "run_identity_artifact": {
            "path": str(run_path.resolve()),
            "sha256": _sha256(run_path),
        },
        "champion": {
            key: champion[key] for key in ("model_identity", "model_step", "updated_ns")
        },
        "champion_pointer_artifact": {
            "path": str(champion_path.resolve()),
            "sha256": _sha256(champion_path),
        },
        "source_anchor": {"model_identity": anchor, "model_step": 0},
        "selection": "guarded_chronological_champion_frontier",
    }


def _treatment(
    root: Path,
    *,
    seed: int,
    label: str,
    anchor: str,
    point_score: float,
    standard_error_score: float,
) -> dict[str, object]:
    snapshot = _snapshot(root, seed=seed, label=label, anchor=anchor)
    champion = snapshot["champion"]
    assert isinstance(champion, dict)
    hours = 10.0
    target_gain = point_score * hours
    standard_error = standard_error_score * hours
    pair_count = max(50, round(25 / (standard_error_score**2)))
    target_score_rate = 1.0 / (1.0 + 10.0 ** (-target_gain / 400.0))
    wins = round(target_score_rate * 2 * pair_count)
    two_win_pairs = max(0, wins - pair_count)
    one_win_pairs = wins - 2 * two_win_pairs
    zero_win_pairs = pair_count - one_win_pairs - two_win_pairs
    pair_scores = [0.0] * zero_win_pairs + [0.5] * one_win_pairs + [1.0] * two_win_pairs
    score_rate = math.fsum(pair_scores) / pair_count
    gain = elo_from_probability(score_rate)
    link_error_probability = PAIR_VALID_ERROR_PROBABILITY_PER_SIDE / 2
    lower_score, upper_score = bounded_confidence_sequence(
        pair_scores,
        error_probability=link_error_probability,
    )
    lower_gain = elo_from_probability(lower_score)
    upper_gain = elo_from_probability(upper_score)
    started_ns = seed * 100 * HOUR_NS
    cutoff_ns = started_ns + 9 * HOUR_NS
    released_ns = started_ns + 10 * HOUR_NS
    completed_ns = started_ns + HOUR_NS
    raw_pairs = []
    for outcomes, count in (
        ((-1, -1), zero_win_pairs),
        ((1, -1), one_win_pairs),
        ((1, 1), two_win_pairs),
    ):
        for _ in range(count):
            pair = len(raw_pairs)
            raw_pairs.append(
                {
                    "ring": 10,
                    "pair": pair,
                    "opening_seed": pair,
                    "opening_action": None,
                    "forced_opening": False,
                    "outcomes": list(outcomes),
                }
            )
    summary = {
        "pairs": pair_count,
        "games": 2 * pair_count,
        "wins": wins,
        "losses": 2 * pair_count - wins,
        "pair_win_counts": {
            "0": zero_win_pairs,
            "1": one_win_pairs,
            "2": two_win_pairs,
        },
        "score_rate": score_rate,
        "elo_difference": gain,
    }
    artifact_path = root / "arena" / "candidate-vs-anchor.json"
    _write_json(
        artifact_path,
        {
            "schema_version": ARENA_RESULT_SCHEMA_VERSION,
            "candidate": champion["model_identity"],
            "baseline": anchor,
            "completed_ns": completed_ns,
            "terminal": True,
            "aggregate": summary,
            "per_ring": {"10": summary},
            "promotion": {"decision": "promote"},
            "pairs": raw_pairs,
        },
    )
    artifact = {
        "path": str(artifact_path.resolve()),
        "bytes": artifact_path.stat().st_size,
        "sha256": _sha256(artifact_path),
    }
    attempt = {
        "attempt_index": 0,
        "baseline": anchor,
        "candidate": champion["model_identity"],
        "decision": "promote",
        "completed_ns": completed_ns,
        "arena_artifact": artifact,
    }
    pair_valid_promotion = {
        "from_identity": anchor,
        "to_identity": champion["model_identity"],
        "completed_ns": completed_ns,
        "path": str(artifact_path.resolve()),
        "ring_10_elo_difference": gain,
        "ring_10_anytime_lower_elo": lower_gain,
        "ring_10_anytime_upper_elo": upper_gain,
        "link_error_probability_per_side": link_error_probability,
        "complete_pairs": pair_count,
        "pair_score_counts": {
            "0": zero_win_pairs,
            "1": one_win_pairs,
            "2": two_win_pairs,
        },
        "terminal_attempt_index": 0,
        "arena_artifact": artifact,
    }
    frontier = {
        "identity": champion["model_identity"],
        "step": champion["model_step"],
        "rating_elo": gain,
        "standard_error_elo": standard_error,
        "pair_valid": True,
        "pair_observation_unit": "complete-role-reversed-pair",
        "pair_valid_elo_gained": gain,
        "pair_valid_elo_one_sided_lower_bound": lower_gain,
        "pair_valid_elo_one_sided_upper_bound": upper_gain,
        "pair_valid_promotions": [pair_valid_promotion],
        "pair_valid_attempts": [attempt],
        "pair_valid_attempt_count": 1,
        "pair_valid_source": {
            "run_root": str(root.resolve()),
            "arena_root": str((root / "arena").resolve()),
            "anchor_identity": anchor,
            "measurement_started_ns": started_ns,
            "measurement_stopped_ns": cutoff_ns,
        },
        "lower_bound_method": PAIR_VALID_LOWER_BOUND_METHOD,
        "familywise_error_probability_per_side": (
            PAIR_VALID_ERROR_PROBABILITY_PER_SIDE
        ),
        "link_error_spending": (
            "error_probability_per_side / 2^(terminal_attempt_index + 1)"
        ),
        "promotion_count": 1,
        "promotions": [],
        "selection": "chronological_promotions_from_common_anchor",
        "non_promoted_terminal_count": 0,
    }
    return {
        "rank": 1 if label == "candidate" else 2,
        "label": label,
        "status": "eligible",
        "eligible": True,
        "run_root": str(root.resolve()),
        "training_objective": "ring10_only",
        "promotion_objective": "ring_10_only",
        "ranking_objective": "ring_10_only",
        "anchor": {
            "identity": anchor,
            "step": 0,
            "rating_elo": 0.0,
            "standard_error_elo": 0.0,
            "selection": "ablation_metadata",
        },
        "champion_frontier": frontier,
        "verified_winner_snapshot": snapshot,
        "measurement": {
            "source": "ablation.json",
            "status": "complete",
            "started_ns": started_ns,
            "stopped_ns": cutoff_ns,
            "cutoff_ns": cutoff_ns,
            "resource_released_ns": released_ns,
            "resource_wall_seconds": 10 * 3_600,
            "stop_reason": "wall_budget",
            "outcome": "budget_completion",
        },
        "resource_accounting": {
            "source": "ablation.json",
            "started_ns": started_ns,
            "measurement_cutoff_ns": cutoff_ns,
            "resource_released_ns": released_ns,
            "measurement_wall_seconds": 9 * 3_600,
            "teardown_wall_seconds": 3_600,
            "total_provisioned_wall_seconds": 10 * 3_600,
            "total_provisioned_wall_hours": hours,
            "provisioned_gpus": 8,
            "total_provisioned_gpu_hours": 80,
        },
        "deployment_metric": {
            "name": RANKING_METRIC,
            "objective": "ring10_only",
            "value": lower_gain / hours,
            "point_value": gain / hours,
            "champion_frontier_ring_10_elo_gained": gain,
            "champion_frontier_ring_10_elo_gain_conservative_standard_error": (None),
            "champion_frontier_ring_10_elo_one_sided_95_lower_bound": lower_gain,
            "champion_frontier_ring_10_elo_one_sided_95_upper_bound": upper_gain,
            "pair_valid": True,
            "pair_observation_unit": "complete-role-reversed-pair",
            "lower_bound_method": PAIR_VALID_LOWER_BOUND_METHOD,
            "familywise_error_probability_per_side": (
                PAIR_VALID_ERROR_PROBABILITY_PER_SIDE
            ),
            "total_provisioned_wall_hours": hours,
            "total_provisioned_gpu_hours": 80,
            "time_basis": "measurement_started_ns_to_resource_released_ns",
            "selection": "chronological_promotions_only",
        },
        "ineligibility_reasons": [],
        "parse_failure_count": 0,
        "parse_failures": [],
    }


def _comparison(
    tmp_path: Path,
    *,
    seed: int,
    control_point: float,
    candidate_point: float,
    control_standard_error: float,
    candidate_standard_error: float,
) -> tuple[Path, dict[str, Path]]:
    anchor = COMMON_ANCHOR
    roots = {
        label: tmp_path / f"seed{seed}" / label for label in ("control", "candidate")
    }
    treatments = [
        _treatment(
            roots["candidate"],
            seed=seed,
            label="candidate",
            anchor=anchor,
            point_score=candidate_point,
            standard_error_score=candidate_standard_error,
        ),
        _treatment(
            roots["control"],
            seed=seed,
            label="control",
            anchor=anchor,
            point_score=control_point,
            standard_error_score=control_standard_error,
        ),
    ]
    candidate_snapshot = treatments[0]["verified_winner_snapshot"]
    path = tmp_path / f"seed{seed}" / "comparison.json"
    plan_path = tmp_path / f"seed{seed}" / "plan.json"
    profile_template = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    profile_artifacts = {}
    for label, root in roots.items():
        profile = json.loads(json.dumps(profile_template))
        profile["train"]["seed"] = seed
        profile["selfplay"]["seed"] = seed
        profile["arena"]["seed"] = seed
        profile["orchestration"]["directories"]["root"] = str(root.resolve())
        profile_path = root / "profile-elo-ablation.yaml"
        profile_path.write_text(
            yaml.safe_dump(profile, sort_keys=False),
            encoding="utf-8",
        )
        profile_artifacts[label] = {
            "path": str(profile_path.resolve()),
            "sha256": _sha256(profile_path),
        }
    _write_json(
        plan_path,
        {
            "seed": seed,
            "treatments": [
                {
                    "treatment": label,
                    "run_root": str(roots[label].resolve()),
                    "profile": profile_artifacts[label]["path"],
                    "profile_sha256": profile_artifacts[label]["sha256"],
                }
                for label in ("control", "candidate")
            ],
        },
    )
    manifest_path = tmp_path / f"seed{seed}" / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "report": "startrain-elo-ablation-deployment",
            "source": {"commit": SOURCE_COMMIT},
            "plan": {
                "path": str(plan_path.resolve()),
                "sha256": _sha256(plan_path),
            },
            "profiles": [
                {
                    "treatment": label,
                    "seed_contract": {
                        "train_seed": seed,
                        "selfplay_seed": seed,
                        "arena_seed": seed,
                    },
                    "profile": profile_artifacts[label],
                    "plan_profile": profile_artifacts[label],
                }
                for label in ("control", "candidate")
            ],
            "queue": {
                "seed": seed,
                "comparison_output": str(path.resolve()),
            },
        },
    )
    report = {
        "schema_version": 1,
        "report": PER_SEED_REPORT,
        "status": "complete",
        "ranking_metric": RANKING_METRIC,
        "ranking_objective": "ring_10_only",
        "confidence": {
            "level": 0.95,
            "sidedness": "two-sided-familywise",
            "method": PAIR_VALID_LOWER_BOUND_METHOD,
            "normal_quantile": None,
            "observation_unit": "complete-role-reversed-pair",
            "familywise_error_probability_per_side": (
                PAIR_VALID_ERROR_PROBABILITY_PER_SIDE
            ),
            "link_error_spending": (
                "error_probability_per_side / 2^(terminal_attempt_index + 1)"
            ),
        },
        "compute_accounting": {
            "provisioned_gpus": 8,
            "basis": "all provisioned GPUs through release",
        },
        "guardrail_configuration": {"rings": [], "floor_elo": -35.0},
        "common_anchor": {
            "status": "available",
            "identity": anchor,
            "by_treatment": [
                {"label": "candidate", "identity": anchor},
                {"label": "control", "identity": anchor},
            ],
        },
        "run_count": 2,
        "eligible_count": 2,
        "selector": {
            "status": "verified",
            "ranking_metric": RANKING_METRIC,
            "ranking_objective": "ring_10_only",
            "winner_snapshot": candidate_snapshot,
            "selection": (
                "highest_ranked_chronological_ring10_only_pair_valid_champion_frontier"
            ),
            "non_promoted_endpoints_are_diagnostic_only": True,
        },
        "errors": [],
        "treatments": treatments,
        "queue": {
            "seed": seed,
            "source_commit": SOURCE_COMMIT,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "state_path": str((tmp_path / f"seed{seed}" / "queue.json").resolve()),
            "queue_status": "completed",
            "arms": [],
        },
    }
    _write_json(path, report)
    return path, roots


def _policy(tmp_path: Path) -> Path:
    future_profile = tmp_path / "profiles" / "future.yaml"
    previous_profile = tmp_path / "profiles" / "previous.yaml"
    previous_root = tmp_path / "previous-lkg"
    previous_root.mkdir(parents=True)
    future_root = tmp_path / "fresh-canary"
    profile_template = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    future_raw = json.loads(json.dumps(profile_template))
    future_raw["orchestration"]["directories"]["root"] = str(future_root.resolve())
    previous_raw = json.loads(json.dumps(profile_template))
    previous_raw["orchestration"]["directories"]["root"] = str(previous_root.resolve())
    future_profile.parent.mkdir(parents=True)
    future_profile.write_text(
        yaml.safe_dump(future_raw, sort_keys=False),
        encoding="utf-8",
    )
    previous_profile.write_text(
        yaml.safe_dump(previous_raw, sort_keys=False),
        encoding="utf-8",
    )
    policy = {
        "schema_version": 1,
        "report": POLICY_REPORT,
        "source_commit": SOURCE_COMMIT,
        "seeds": [17, 18, 19],
        "control_treatment": "control",
        "candidate_treatment": "candidate",
        "common_objective": "ring10_only",
        "common_anchor": {"identity": COMMON_ANCHOR, "step": 0},
        "minimum_median_point_elo_per_hour_improvement": 0.20,
        "cross_seed_lcb_gate": {
            "method": LCB_GATE_METHOD,
            "minimum_advantage_elo_per_hour": 0.0,
            "require_strictly_positive": True,
        },
        "screening_eliminations": [],
        "canary": {
            "duration_hours": 24,
            "operator_hold_path": str((tmp_path / "operator-hold").resolve()),
            "continuity_fallback_required": True,
            "future_fresh_winner": {
                "run_root": str(future_root.resolve()),
                "profile": {
                    "path": str(future_profile.resolve()),
                    "sha256": _sha256(future_profile),
                },
            },
            "previous_lkg": {
                "run_root": str(previous_root.resolve()),
                "profile": {
                    "path": str(previous_profile.resolve()),
                    "sha256": _sha256(previous_profile),
                },
            },
            "required_gates": sorted(REQUIRED_CANARY_GATES),
        },
    }
    path = tmp_path / "adoption-policy.json"
    _write_json(path, policy)
    return path


def _evidence(
    tmp_path: Path,
    *,
    control_point: float = 10.0,
    candidate_point: float = 13.0,
    control_standard_error: float = 0.10,
    candidate_standard_error: float = 0.10,
) -> Evidence:
    policy = _policy(tmp_path)
    comparisons = []
    roots = {}
    for seed in (17, 18, 19):
        path, seed_roots = _comparison(
            tmp_path,
            seed=seed,
            control_point=control_point,
            candidate_point=candidate_point,
            control_standard_error=control_standard_error,
            candidate_standard_error=candidate_standard_error,
        )
        comparisons.append(PinnedComparison(seed, path, _sha256(path)))
        roots[seed] = seed_roots
    return Evidence(policy, _sha256(policy), comparisons, roots)


def _rewrite_comparison(
    evidence: Evidence,
    seed: int,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    comparison = next(item for item in evidence.comparisons if item.seed == seed)
    payload = json.loads(comparison.path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(comparison.path, payload)
    evidence.comparisons = [
        PinnedComparison(item.seed, item.path, _sha256(item.path))
        if item.seed == seed
        else item
        for item in evidence.comparisons
    ]


def _build(evidence: Evidence) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        build_cross_seed_comparison(
            evidence.comparisons,
            policy_path=evidence.policy,
            policy_sha256=evidence.policy_sha256,
        ),
    )


def _write_cross_seed_report(
    tmp_path: Path,
    report: dict[str, object],
    *,
    name: str = "cross-seed.json",
) -> tuple[Path, str]:
    path = tmp_path / name
    _write_json(path, report)
    return path, _sha256(path)


def test_valid_three_seed_candidate_is_eligible(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)

    report = _build(evidence)

    assert report["status"] == "eligible"
    assert report["eligible"] is True
    assert [record["seed"] for record in report["per_seed"]] == [17, 18, 19]
    aggregate = report["aggregate"]
    first_seed = report["per_seed"][0]
    control_point = first_seed["control"]["point_elo_per_total_provisioned_wall_hour"]
    candidate_point = first_seed["candidate"][
        "point_elo_per_total_provisioned_wall_hour"
    ]
    expected_relative = (candidate_point - control_point) / control_point
    assert aggregate["median_relative_point_elo_per_hour_improvement"] == pytest.approx(
        expected_relative
    )
    assert expected_relative > 0.29
    assert aggregate["minimum_per_seed_one_sided_lcb_advantage_elo_per_hour"] > 0
    assert aggregate["cross_seed_confidence"]["level"] is None
    assert aggregate["cross_seed_confidence"]["pooled_interval"] is False
    first_advantage = first_seed["advantage"]
    assert first_advantage["one_sided_lower_bound_elo_per_hour"] == pytest.approx(
        first_seed["candidate"][
            "primary_ring10_only_champion_frontier_lcb_per_total_provisioned_wall_hour"
        ]
        - first_seed["control"]["upper_elo_per_total_provisioned_wall_hour"]
    )
    assert first_advantage["method"] == (
        "candidate_pair_valid_lower_bound_minus_control_pair_valid_upper_bound"
    )
    assert first_advantage["familywise_error_probability"] == pytest.approx(0.05)
    assert report["selector"]["status"] == "verified"
    assert report["selector"]["source_seed"] == 18
    assert report["selector"]["latest_terminal_candidates_ranked"] is False


def test_missing_duplicate_and_single_seed_inputs_are_rejected(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)

    with pytest.raises(CrossSeedComparisonError, match="exactly three"):
        build_cross_seed_comparison(
            evidence.comparisons[:1],
            policy_path=evidence.policy,
            policy_sha256=evidence.policy_sha256,
        )
    with pytest.raises(CrossSeedComparisonError, match="duplicate comparison seed"):
        build_cross_seed_comparison(
            [
                evidence.comparisons[0],
                evidence.comparisons[0],
                evidence.comparisons[2],
            ],
            policy_path=evidence.policy,
            policy_sha256=evidence.policy_sha256,
        )


def test_tampered_comparison_and_policy_are_rejected(tmp_path: Path) -> None:
    comparison_evidence = _evidence(tmp_path / "comparison")
    comparison_evidence.comparisons[0].path.write_text(
        comparison_evidence.comparisons[0].path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(CrossSeedComparisonError, match="digest mismatch"):
        _build(comparison_evidence)

    policy_evidence = _evidence(tmp_path / "policy")
    policy_evidence.policy.write_text(
        policy_evidence.policy.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(CrossSeedComparisonError, match="digest mismatch"):
        _build(policy_evidence)

    malformed_policy_evidence = _evidence(tmp_path / "malformed-policy")
    malformed_policy = json.loads(
        malformed_policy_evidence.policy.read_text(encoding="utf-8")
    )
    malformed_policy["screening_eliminations"] = [{"label": "actor-lanes-3"}]
    _write_json(malformed_policy_evidence.policy, malformed_policy)
    malformed_policy_evidence.policy_sha256 = _sha256(malformed_policy_evidence.policy)
    with pytest.raises(CrossSeedComparisonError, match="exactly label and reason"):
        _build(malformed_policy_evidence)


def test_seed_claim_is_bound_to_pinned_deployment_plan(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    comparison = evidence.comparisons[0]
    report = json.loads(comparison.path.read_text(encoding="utf-8"))
    manifest_path = Path(report["queue"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_path = Path(manifest["plan"]["path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["seed"] = 18
    _write_json(plan_path, plan)
    manifest["plan"]["sha256"] = _sha256(plan_path)
    _write_json(manifest_path, manifest)
    report["queue"]["manifest_sha256"] = _sha256(manifest_path)
    _write_json(comparison.path, report)
    evidence.comparisons[0] = PinnedComparison(
        comparison.seed,
        comparison.path,
        _sha256(comparison.path),
    )

    result = _build(evidence)

    assert result["eligible"] is False
    assert any(
        "pinned ablation plan seed disagrees" in error["message"]
        for error in result["errors"]
    )


def test_policy_pins_one_common_anchor_identity_and_step(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    policy = json.loads(evidence.policy.read_text(encoding="utf-8"))
    policy["common_anchor"] = {"identity": "different-anchor", "step": 0}
    _write_json(evidence.policy, policy)
    evidence.policy_sha256 = _sha256(evidence.policy)

    result = _build(evidence)

    assert result["eligible"] is False
    assert all(record["status"] == "invalid" for record in result["per_seed"])
    assert any(
        "anchor identity differs" in error["message"] for error in result["errors"]
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["treatments"][1].update({"label": "other-control"}),
        lambda report: report["treatments"][0].update(
            {"training_objective": "generalist"}
        ),
        lambda report: report["treatments"][0]["anchor"].update(
            {"identity": "different-anchor"}
        ),
        lambda report: report["compute_accounting"].update({"provisioned_gpus": 4}),
    ],
    ids=("control-label", "objective", "anchor", "topology"),
)
def test_inconsistent_confirmation_fails_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    evidence = _evidence(tmp_path)
    _rewrite_comparison(evidence, 18, mutate)

    report = _build(evidence)

    assert report["status"] == "ineligible"
    assert report["eligible"] is False
    assert report["selector"]["status"] == "unavailable"
    assert report["errors"]


def test_below_twenty_percent_improvement_is_ineligible(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, candidate_point=11.9)

    report = _build(evidence)

    gate = report["gates"]["minimum_median_point_improvement"]
    assert gate["observed"] == pytest.approx(0.19, abs=0.003)
    assert gate["passed"] is False
    assert report["eligible"] is False


def test_nonpositive_lcb_advantage_is_ineligible(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path,
        candidate_point=12.1,
        control_standard_error=1.0,
        candidate_standard_error=1.0,
    )

    report = _build(evidence)

    assert report["gates"]["minimum_median_point_improvement"]["passed"] is True
    lcb_gate = report["gates"]["positive_one_sided_lcb_advantage"]
    assert lcb_gate["observed"] <= 0
    assert lcb_gate["passed"] is False
    assert report["eligible"] is False


def test_self_declared_pair_valid_frontier_without_attempt_evidence_is_ineligible(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)

    def strip_attempt_evidence(report: dict[str, Any]) -> None:
        candidate = report["treatments"][0]["champion_frontier"]
        candidate["pair_valid_attempts"] = []
        candidate["pair_valid_attempt_count"] = 0

    _rewrite_comparison(evidence, 18, strip_attempt_evidence)

    report = _build(evidence)

    assert report["eligible"] is False
    assert "pair-valid frontier verification failed" in report["errors"][0]["message"]


def test_pair_valid_source_must_match_treatment_measurement_context(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)

    def narrow_source_window(report: dict[str, Any]) -> None:
        source = report["treatments"][0]["champion_frontier"]["pair_valid_source"]
        source["measurement_started_ns"] += 1

    _rewrite_comparison(evidence, 18, narrow_source_window)

    report = _build(evidence)

    assert report["eligible"] is False
    assert "differs from validated treatment context" in report["errors"][0]["message"]


def test_unverified_candidate_snapshot_is_ineligible(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)

    def unverify(report: dict[str, Any]) -> None:
        report["treatments"][0]["verified_winner_snapshot"]["status"] = "unverified"

    _rewrite_comparison(evidence, 18, unverify)

    report = _build(evidence)

    assert report["eligible"] is False
    assert "winner snapshot is not verified" in report["errors"][0]["message"]


def test_adoption_plan_is_immutable_and_references_fresh_canary(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    report_path, report_sha256 = _write_cross_seed_report(tmp_path, _build(evidence))
    output = tmp_path / "adoption-plan.json"

    plan = cast(
        dict[str, Any],
        prepare_ablation_adoption(
            report_path=report_path,
            report_sha256=report_sha256,
            policy_path=evidence.policy,
            policy_sha256=evidence.policy_sha256,
            output_path=output,
        ),
    )

    assert plan["report"] == PLAN_REPORT
    assert plan["eligible"] is True
    assert plan["source_commit"] == SOURCE_COMMIT
    assert plan["policy"]["sha256"] == evidence.policy_sha256
    assert plan["cross_seed_report"]["sha256"] == report_sha256
    assert plan["canary"]["duration_hours"] == 24
    assert plan["canary"]["targets"]["future_fresh_winner"]["root_must_be_fresh"]
    assert plan["canary"]["targets"]["previous_lkg"]["run_root"].endswith(
        "previous-lkg"
    )
    assert plan["safety"]["run_roots_mutated"] is False
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_ablation_adoption(
            report_path=report_path,
            report_sha256=report_sha256,
            policy_path=evidence.policy,
            policy_sha256=evidence.policy_sha256,
            output_path=output,
        )


def test_adoption_refuses_ineligible_and_stale_evidence(tmp_path: Path) -> None:
    ineligible_evidence = _evidence(tmp_path / "ineligible", candidate_point=11.9)
    ineligible_path, ineligible_sha256 = _write_cross_seed_report(
        tmp_path,
        _build(ineligible_evidence),
        name="ineligible-cross-seed.json",
    )
    ineligible = cast(
        dict[str, Any],
        prepare_ablation_adoption(
            report_path=ineligible_path,
            report_sha256=ineligible_sha256,
            policy_path=ineligible_evidence.policy,
            policy_sha256=ineligible_evidence.policy_sha256,
            output_path=tmp_path / "ineligible-decision.json",
        ),
    )

    assert ineligible["report"] == INELIGIBLE_REPORT
    assert ineligible["eligible"] is False
    assert ineligible["safety"]["adoption_plan_written"] is False

    stale_evidence = _evidence(tmp_path / "stale")
    stale_report_path, stale_report_sha256 = _write_cross_seed_report(
        tmp_path,
        _build(stale_evidence),
        name="stale-cross-seed.json",
    )
    champion_path = stale_evidence.roots[18]["candidate"] / "learner" / "champion.json"
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    champion["model_step"] = 101
    _write_json(champion_path, champion)

    stale = cast(
        dict[str, Any],
        prepare_ablation_adoption(
            report_path=stale_report_path,
            report_sha256=stale_report_sha256,
            policy_path=stale_evidence.policy,
            policy_sha256=stale_evidence.policy_sha256,
            output_path=tmp_path / "stale-decision.json",
        ),
    )

    assert stale["report"] == INELIGIBLE_REPORT
    assert stale["eligible"] is False
    assert any(reason["code"] == "stale_winner_snapshot" for reason in stale["reasons"])
    assert stale["safety"]["run_roots_mutated"] is False


def test_adoption_rejects_canary_overlap_and_protected_output(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    policy = json.loads(evidence.policy.read_text(encoding="utf-8"))
    future = policy["canary"]["future_fresh_winner"]
    overlapping_root = evidence.roots[17]["candidate"] / "fresh-canary"
    profile_path = Path(future["profile"]["path"])
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["orchestration"]["directories"]["root"] = str(overlapping_root.resolve())
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    future["run_root"] = str(overlapping_root.resolve())
    future["profile"]["sha256"] = _sha256(profile_path)
    _write_json(evidence.policy, policy)
    evidence.policy_sha256 = _sha256(evidence.policy)
    report_path, report_sha256 = _write_cross_seed_report(tmp_path, _build(evidence))

    decision = cast(
        dict[str, Any],
        prepare_ablation_adoption(
            report_path=report_path,
            report_sha256=report_sha256,
            policy_path=evidence.policy,
            policy_sha256=evidence.policy_sha256,
            output_path=tmp_path / "overlap-decision.json",
        ),
    )

    assert decision["eligible"] is False
    assert any(
        reason["code"] == "canary_treatment_root_overlap"
        for reason in decision["reasons"]
    )

    safe_evidence = _evidence(tmp_path / "protected-output")
    safe_report_path, safe_report_sha256 = _write_cross_seed_report(
        tmp_path,
        _build(safe_evidence),
        name="protected-output-cross-seed.json",
    )
    safe_policy = json.loads(safe_evidence.policy.read_text(encoding="utf-8"))
    protected_root = Path(safe_policy["canary"]["future_fresh_winner"]["run_root"])
    output = protected_root / "adoption-plan.json"

    with pytest.raises(ValueError, match="outside canary"):
        prepare_ablation_adoption(
            report_path=safe_report_path,
            report_sha256=safe_report_sha256,
            policy_path=safe_evidence.policy,
            policy_sha256=safe_evidence.policy_sha256,
            output_path=output,
        )
    assert not protected_root.exists()
