from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_elo_ablation import (
    DEFAULT_GUARD_FLOOR_ELO,
    DEFAULT_GUARD_RINGS,
    ONE_SIDED_95_NORMAL_QUANTILE,
    REPORT_NAME,
    build_elo_ablation_comparison,
    main,
)

HOUR_NS = 3_600_000_000_000


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _run_fixture(
    parent: Path,
    label: str,
    *,
    anchor: str = "checkpoint-common-anchor",
    ring_wins: int = 70,
    ring_losses: int = 30,
    decision: str = "promote",
    guard_lowers: dict[int, float] | None = None,
    omitted_guard_rings: tuple[int, ...] = (),
) -> Path:
    root = parent / label
    root.mkdir()
    started_ns = 1_000_000_000_000
    candidate = f"checkpoint-{label}"
    published_ns = started_ns + HOUR_NS
    terminal_ns = started_ns + 2 * HOUR_NS
    observed_until_ns = started_ns + 8 * HOUR_NS
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": f"run-{label}",
                "generation_family": "ablation-fixture",
                "created_ns": started_ns,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        root / "learner" / "model-history.jsonl",
        [
            {
                "schema_version": 1,
                "model_identity": anchor,
                "model_step": 0,
                "published_ns": started_ns,
            },
            {
                "schema_version": 1,
                "model_identity": candidate,
                "model_step": 10,
                "published_ns": published_ns,
            },
        ],
    )
    for role, identity, step in (
        ("candidate", candidate, 10),
        ("champion", anchor, 0),
    ):
        (root / "learner" / f"{role}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "role": role,
                    "model_identity": identity,
                    "model_step": step,
                    "updated_ns": published_ns,
                }
            ),
            encoding="utf-8",
        )
    _write_jsonl(
        root / "learner" / "metrics.jsonl",
        [
            {
                "schema_version": 1,
                "timestamp_ns": started_ns + HOUR_NS // 2,
                "worker": "learner",
                "step": 10,
                "metrics_interval_steps": 10,
                "global_batch_size": 64,
                "step_seconds": 0.5,
                "metrics_interval_wall_seconds": 5.0,
                "device_step_seconds": 0.25,
                "data_wait_seconds": 0.05,
                "updates_per_new_sample": 0.9,
            },
            {
                "schema_version": 1,
                "timestamp_ns": started_ns + 3 * HOUR_NS,
                "worker": "learner",
                "phase": "replay_wait",
            },
            {
                "schema_version": 1,
                "timestamp_ns": started_ns + 3 * HOUR_NS + 30_000_000_000,
                "worker": "learner",
                "phase": "training",
                "updates_per_new_sample": 1.1,
            },
        ],
    )
    _write_jsonl(
        root / "metrics" / "actor-gpu-1-lane-0.jsonl",
        [
            {
                "schema_version": 1,
                "timestamp_ns": observed_until_ns,
                "batch_started_ns": started_ns,
                "batch_completed_ns": observed_until_ns,
                "worker": "actor-gpu-1-lane-0",
                "gpu_id": 1,
                "model_identity": candidate,
                "model_step": 10,
                "elapsed_seconds": 8 * 3_600,
                "games": 800,
                "samples": 8_000,
                "search_simulations": 80_000,
                "evaluator_rows": 800_000,
            }
        ],
    )
    lowers = guard_lowers or {4: -10.0, 6: -8.0, 8: -5.0}
    ring_floors = {
        str(ring): {
            "floor_elo": DEFAULT_GUARD_FLOOR_ELO,
            "anytime_lower_elo": lowers[ring],
            "status": "pass" if lowers[ring] >= DEFAULT_GUARD_FLOOR_ELO else "regress",
        }
        for ring in DEFAULT_GUARD_RINGS
        if ring not in omitted_guard_rings
    }
    per_ring = {
        str(ring): {
            "wins": ring_wins,
            "losses": ring_losses,
            "games": ring_wins + ring_losses,
            "elo_difference": 0.0,
            "anytime_elo_interval": [-100.0, 100.0],
        }
        for ring in (*DEFAULT_GUARD_RINGS, 10)
    }
    arena = root / "arena"
    arena.mkdir()
    (arena / "candidate-vs-anchor.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "candidate": candidate,
                "baseline": anchor,
                "baseline_metadata": {
                    "kind": "checkpoint",
                    "identity": anchor,
                },
                "started_ns": terminal_ns - 1_000_000_000,
                "completed_ns": terminal_ns,
                "terminal": True,
                "aggregate": {
                    "wins": ring_wins,
                    "losses": ring_losses,
                    "games": ring_wins + ring_losses,
                    "elo_difference": 0.0,
                    "anytime_elo_interval": [-100.0, 100.0],
                },
                "per_ring": per_ring,
                "promotion": {
                    "decision": decision,
                    "ring_floors": ring_floors,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def _install_ablation_metadata(
    root: Path,
    *,
    source_anchor: str = "checkpoint-common-anchor",
    frozen_anchor: str = "checkpoint-frozen-anchor",
    complete: bool = True,
) -> None:
    run_path = root / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    original_started_ns = int(run["created_ns"])
    run["created_ns"] = max(1, original_started_ns - HOUR_NS // 2)
    run_path.write_text(json.dumps(run), encoding="utf-8")
    candidate = f"checkpoint-{root.name}"
    frozen_published_ns = original_started_ns + HOUR_NS // 4
    history_path = root / "learner" / "model-history.jsonl"
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_identity": frozen_anchor,
                    "model_step": 5,
                    "published_ns": frozen_published_ns,
                },
                sort_keys=True,
            )
            + "\n"
        )
    (root / "learner" / "champion.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "role": "champion",
                "model_identity": frozen_anchor,
                "model_step": 5,
                "updated_ns": frozen_published_ns,
            }
        ),
        encoding="utf-8",
    )

    new_result_path = root / "arena" / "candidate-vs-anchor.json"
    new_result = json.loads(new_result_path.read_text(encoding="utf-8"))
    new_result["baseline"] = frozen_anchor
    new_result["baseline_metadata"]["identity"] = frozen_anchor
    new_result_path.write_text(
        json.dumps(new_result, sort_keys=True),
        encoding="utf-8",
    )
    old_result = json.loads(json.dumps(new_result))
    old_result.update(
        {
            "candidate": frozen_anchor,
            "baseline": source_anchor,
            "baseline_metadata": {
                "kind": "checkpoint",
                "identity": source_anchor,
            },
            "started_ns": original_started_ns + HOUR_NS // 2 - 1_000_000_000,
            "completed_ns": original_started_ns + HOUR_NS // 2,
        }
    )
    (root / "arena" / "frozen-vs-source.json").write_text(
        json.dumps(old_result, sort_keys=True),
        encoding="utf-8",
    )

    measurement_started_ns = original_started_ns + 3 * HOUR_NS // 4
    measurement_stopped_ns = measurement_started_ns + 8 * HOUR_NS
    (root / "ablation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report": "startrain-elo-ablation-branch",
                "treatment": root.name,
                "source_run_id": run["run_id"],
                "source_generation_family": run["generation_family"],
                "source_created_ns": run["created_ns"],
                "measurement_started_ns": measurement_started_ns,
                "measurement_stopped_ns": (
                    measurement_stopped_ns if complete else None
                ),
                "measurement_stop_reason": "wall_budget" if complete else None,
                "measurement_exit_code": 0 if complete else None,
                "wall_budget_seconds": 8 * 3_600,
                "leaf_budget": 2_000_000_000,
                "guard_rings": [4, 6, 8],
                "guard_floor_elo": -35,
                "anchor": {
                    "model_identity": frozen_anchor,
                    "model_step": 5,
                    "updated_ns": frozen_published_ns,
                },
                "starting_candidate": {
                    "model_identity": candidate,
                    "model_step": 10,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _by_label(report: dict[str, object]) -> dict[str, dict[str, object]]:
    treatments = report["treatments"]
    assert isinstance(treatments, list)
    return {
        str(treatment["label"]): treatment
        for treatment in treatments
        if isinstance(treatment, dict)
    }


def _reason_codes(treatment: dict[str, object]) -> set[str]:
    reasons = treatment["ineligibility_reasons"]
    assert isinstance(reasons, list)
    return {str(reason["code"]) for reason in reasons if isinstance(reason, dict)}


def test_comparison_ranks_common_anchor_elo_lcb_and_retains_operations_metrics(
    tmp_path: Path,
) -> None:
    stronger = _run_fixture(tmp_path, "stronger", ring_wins=72, ring_losses=28)
    weaker = _run_fixture(tmp_path, "weaker", ring_wins=58, ring_losses=42)

    report = build_elo_ablation_comparison({"weaker": weaker, "stronger": stronger})
    repeated = build_elo_ablation_comparison({"stronger": stronger, "weaker": weaker})

    assert report == repeated
    assert report["report"] == REPORT_NAME
    assert report["status"] == "complete"
    assert report["common_anchor"] == {
        "status": "available",
        "identity": "checkpoint-common-anchor",
        "by_treatment": [
            {"label": "stronger", "identity": "checkpoint-common-anchor"},
            {"label": "weaker", "identity": "checkpoint-common-anchor"},
        ],
    }
    treatments = report["treatments"]
    assert isinstance(treatments, list)
    assert [treatment["label"] for treatment in treatments] == [
        "stronger",
        "weaker",
    ]
    stronger_result = treatments[0]
    assert stronger_result["rank"] == 1
    endpoint = stronger_result["endpoint"]
    efficiency = stronger_result["efficiency"]
    assert isinstance(endpoint, dict)
    assert isinstance(efficiency, dict)
    expected_lower = endpoint["rating_elo"] - (
        ONE_SIDED_95_NORMAL_QUANTILE * endpoint["standard_error_elo"]
    )
    assert endpoint["one_sided_95_lower_rating_elo"] == pytest.approx(expected_lower)
    assert efficiency["wall_hours"] == 8
    assert efficiency["provisioned_gpu_hours"] == 64
    assert efficiency["ring_10_elo_lcb_per_wall_hour"] == pytest.approx(
        expected_lower / 8
    )
    assert efficiency["ring_10_elo_lcb_per_provisioned_gpu_hour"] == pytest.approx(
        expected_lower / 64
    )

    latency = stronger_result["candidate_publish_to_terminal"]
    assert isinstance(latency, dict)
    assert latency["status"] == "complete"
    assert latency["seconds"]["median"] == 3_600
    guardrails = stronger_result["guardrails"]
    assert isinstance(guardrails, dict)
    assert guardrails["status"] == "pass"
    assert [ring["ring"] for ring in guardrails["rings"]] == [4, 6, 8]
    learner = stronger_result["learner"]
    assert isinstance(learner, dict)
    assert learner["updates_per_new_sample"]["statistics"]["latest"] == 1.1
    assert learner["replay_waits"]["seconds"]["mean"] == 30
    assert learner["device_duty_fraction"] == 0.5
    actors = stronger_result["actors"]
    assert isinstance(actors, dict)
    assert actors["aggregate_samples_per_second"] == pytest.approx(8_000 / 28_800)
    assert actors["aggregate_evaluator_rows_per_second"] == pytest.approx(
        800_000 / 28_800
    )


def test_ablation_metadata_defines_common_anchor_and_fixed_budget_accounting(
    tmp_path: Path,
) -> None:
    stronger = _run_fixture(
        tmp_path,
        "stronger-ablation",
        ring_wins=72,
        ring_losses=28,
    )
    weaker = _run_fixture(
        tmp_path,
        "weaker-ablation",
        ring_wins=58,
        ring_losses=42,
    )
    _install_ablation_metadata(stronger)
    _install_ablation_metadata(weaker)

    report = build_elo_ablation_comparison({"stronger": stronger, "weaker": weaker})
    stronger_result = _by_label(report)["stronger"]
    anchor = stronger_result["anchor"]
    endpoint = stronger_result["endpoint"]
    efficiency = stronger_result["efficiency"]

    assert report["status"] == "complete"
    assert report["common_anchor"]["identity"] == "checkpoint-frozen-anchor"
    assert stronger_result["measurement"]["source"] == "ablation.json"
    assert stronger_result["measurement"]["status"] == "complete"
    assert stronger_result["guardrails"]["terminal_evaluation_count"] == 1
    assert isinstance(anchor, dict)
    assert isinstance(endpoint, dict)
    assert isinstance(efficiency, dict)
    assert anchor["selection"] == "ablation_metadata"
    assert anchor["rating_elo"] > 0
    assert endpoint["rating_elo"] > anchor["rating_elo"]
    assert efficiency["wall_hours"] == 8
    assert efficiency["provisioned_gpu_hours"] == 64
    assert efficiency["ring_10_elo_gained"] == pytest.approx(
        endpoint["rating_elo"] - anchor["rating_elo"]
    )
    conservative_standard_error = (
        endpoint["standard_error_elo"] + anchor["standard_error_elo"]
    )
    assert efficiency["ring_10_elo_gain_conservative_standard_error"] == pytest.approx(
        conservative_standard_error
    )
    assert efficiency["ring_10_elo_one_sided_95_lower_bound"] == pytest.approx(
        efficiency["ring_10_elo_gained"]
        - ONE_SIDED_95_NORMAL_QUANTILE * conservative_standard_error
    )


def test_incomplete_fixed_budget_ablation_is_ineligible(tmp_path: Path) -> None:
    control = _run_fixture(tmp_path, "control")
    incomplete = _run_fixture(tmp_path, "incomplete")
    _install_ablation_metadata(control)
    _install_ablation_metadata(incomplete, complete=False)

    report = build_elo_ablation_comparison(
        {"control": control, "incomplete": incomplete}
    )
    incomplete_result = _by_label(report)["incomplete"]

    assert report["status"] == "incomplete"
    assert incomplete_result["measurement"]["status"] == "incomplete"
    assert incomplete_result["eligible"] is False
    assert "incomplete_measurement" in _reason_codes(incomplete_result)


def test_comparison_marks_parse_failures_ineligible(tmp_path: Path) -> None:
    valid = _run_fixture(tmp_path, "valid")
    broken = _run_fixture(tmp_path, "broken")
    metrics = broken / "learner" / "metrics.jsonl"
    with metrics.open("a", encoding="utf-8") as stream:
        stream.write("{not json}\n")

    report = build_elo_ablation_comparison({"valid": valid, "broken": broken})
    treatments = _by_label(report)

    assert report["status"] == "incomplete"
    assert treatments["valid"]["eligible"] is True
    assert treatments["broken"]["eligible"] is False
    assert "parse_failure" in _reason_codes(treatments["broken"])
    assert treatments["broken"]["parse_failure_count"] >= 1
    assert "metrics.jsonl" in treatments["broken"]["parse_failures"][0]["path"]


def test_comparison_requires_one_common_anchor(tmp_path: Path) -> None:
    control = _run_fixture(tmp_path, "control", anchor="anchor-control")
    treatment = _run_fixture(tmp_path, "treatment", anchor="anchor-treatment")

    report = build_elo_ablation_comparison({"control": control, "treatment": treatment})

    assert report["status"] == "incomplete"
    assert report["eligible_count"] == 0
    assert report["common_anchor"]["status"] == "unavailable"
    assert report["errors"][0]["code"] == "missing_common_anchor"
    assert all(
        "missing_common_anchor" in _reason_codes(result)
        for result in _by_label(report).values()
    )


def test_reject_ring_regression_is_ineligible_even_if_custom_floor_passes(
    tmp_path: Path,
) -> None:
    control = _run_fixture(tmp_path, "control")
    rejected = _run_fixture(
        tmp_path,
        "rejected",
        decision="reject_ring_regression",
    )

    report = build_elo_ablation_comparison(
        {"control": control, "rejected": rejected},
        guard_floor_elo=-100,
    )
    rejected_result = _by_label(report)["rejected"]

    assert rejected_result["guardrails"]["status"] == "pass"
    assert rejected_result["eligible"] is False
    assert "reject_ring_regression" in _reason_codes(rejected_result)


def test_guard_floor_and_missing_evidence_gate_treatments(tmp_path: Path) -> None:
    control = _run_fixture(tmp_path, "control")
    below = _run_fixture(
        tmp_path,
        "below",
        guard_lowers={4: -36.0, 6: -8.0, 8: -5.0},
    )
    missing = _run_fixture(
        tmp_path,
        "missing",
        omitted_guard_rings=(6,),
    )

    report = build_elo_ablation_comparison(
        {"control": control, "below": below, "missing": missing}
    )
    treatments = _by_label(report)

    assert "guard_evidence_below_floor" in _reason_codes(treatments["below"])
    assert "missing_guard_evidence" in _reason_codes(treatments["missing"])
    assert treatments["below"]["guardrails"]["configured_floor_elo"] == -35

    relaxed = build_elo_ablation_comparison(
        {"control": control, "below": below},
        guard_floor_elo=-40,
    )
    assert relaxed["status"] == "complete"
    assert all(result["eligible"] for result in _by_label(relaxed).values())


def test_cli_writes_exact_deterministic_json_and_honors_repeated_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    control = _run_fixture(tmp_path, "control")
    treatment = _run_fixture(
        tmp_path,
        "treatment",
        guard_lowers={4: -36.0, 6: -50.0, 8: -36.0},
    )
    output_path = tmp_path / "reports" / "comparison.json"

    exit_code = main(
        [
            "--run",
            f"treatment={treatment}",
            "--run",
            f"control={control}",
            "--provisioned-gpus",
            "8",
            "--guard-ring",
            "8",
            "--guard-ring",
            "4",
            "--guard-floor-elo",
            "-40",
            "--output",
            str(output_path),
        ]
    )
    stdout = capsys.readouterr().out
    written = output_path.read_text(encoding="utf-8")
    payload = json.loads(stdout)

    assert exit_code == 0
    assert stdout == written
    assert payload["guardrail_configuration"] == {
        "rings": [4, 8],
        "floor_elo": -40.0,
    }
    assert payload["compute_accounting"]["provisioned_gpus"] == 8
    assert [item["rank"] for item in payload["treatments"]] == [1, 2]


def test_cli_returns_incomplete_status_for_an_ineligible_treatment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    control = _run_fixture(tmp_path, "control")
    below = _run_fixture(
        tmp_path,
        "below",
        guard_lowers={4: -36.0, 6: -8.0, 8: -5.0},
    )

    assert (
        main(
            [
                "--run",
                f"control={control}",
                "--run",
                f"below={below}",
            ]
        )
        == 3
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "incomplete"
    assert _by_label(payload)["below"]["eligible"] is False


@pytest.mark.parametrize(
    ("arguments", "error_fragment"),
    [
        (["--run", "only=/tmp/only"], "at least two"),
        (
            ["--run", "control=/tmp/control", "--run", "malformed"],
            "LABEL=PATH",
        ),
        (
            [
                "--run",
                "same=/tmp/control",
                "--run",
                "same=/tmp/treatment",
            ],
            "duplicate",
        ),
    ],
)
def test_cli_reports_actionable_run_argument_errors(
    arguments: list[str],
    error_fragment: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(arguments) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert error_fragment in payload["error"]


def test_programmatic_schema_validation_is_strict(tmp_path: Path) -> None:
    control = _run_fixture(tmp_path, "control")
    treatment = _run_fixture(tmp_path, "treatment")
    runs = {"control": control, "treatment": treatment}

    with pytest.raises(ValueError, match="duplicates"):
        build_elo_ablation_comparison(runs, guard_rings=(4, 4))
    with pytest.raises(ValueError, match="finite"):
        build_elo_ablation_comparison(runs, guard_floor_elo=float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        build_elo_ablation_comparison(runs, provisioned_gpus=True)
