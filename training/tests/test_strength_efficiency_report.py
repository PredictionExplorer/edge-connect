from __future__ import annotations

import json

import pytest

from scripts.strength_efficiency_report import (
    REPORT_NAME,
    _actor_summary,
    _learner_summary,
    build_strength_efficiency_report,
    main,
)


def _write_jsonl(path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _arena_result(*, completed_ns: int, elo: float, lower: float) -> dict:
    return {
        "schema_version": 2,
        "candidate": f"candidate-{completed_ns}",
        "baseline": "frozen-shallow-v2",
        "baseline_metadata": {"kind": "shallow-search"},
        "started_ns": completed_ns - 100,
        "completed_ns": completed_ns,
        "evaluation_metrics": {"wall_seconds": 1.0},
        "aggregate": {
            "elo_difference": elo,
            "anytime_elo_interval": [lower, elo + 20],
        },
    }


def test_report_joins_wall_throughput_policy_weight_and_arena_strength(
    tmp_path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-report",
                "generation_family": "family-report",
                "created_ns": 1_000_000_000,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        root / "learner" / "metrics.jsonl",
        [
            {
                "timestamp_ns": 2_000_000_000,
                "step": 2,
                "metrics_interval_steps": 2,
                "global_batch_size": 10,
                "step_seconds": 1,
                "metrics_interval_wall_seconds": 2,
                "device_step_seconds": 0.5,
                "data_wait_seconds": 0.25,
                "h2d_seconds": 0.1,
            },
            {
                "timestamp_ns": 3_000_000_000,
                "step": 4,
                "metrics_interval_steps": 2,
                "global_batch_size": 10,
                "step_seconds": 1,
                "metrics_interval_wall_seconds": 2,
                "device_step_seconds": 0.5,
                "data_wait_seconds": 0.25,
                "h2d_seconds": 0.1,
            },
        ],
    )
    _write_jsonl(
        root / "metrics" / "actor-gpu-1-lane-0.jsonl",
        [
            {
                "timestamp_ns": 3_500_000_000,
                "worker": "actor-gpu-1-lane-0",
                "elapsed_seconds": 4,
                "games": 8,
                "samples": 80,
                "search_simulations": 800,
                "evaluator_rows": 400,
                "policy_samples": 20,
                "policy_weight_sum": 5,
            }
        ],
    )
    arena = root / "arena"
    arena.mkdir()
    (arena / "first.json").write_text(
        json.dumps(_arena_result(completed_ns=4_000_000_000, elo=10, lower=-5)),
        encoding="utf-8",
    )
    (arena / "latest.json").write_text(
        json.dumps(_arena_result(completed_ns=6_000_000_000, elo=30, lower=5)),
        encoding="utf-8",
    )

    report = build_strength_efficiency_report(root, provisioned_gpus=8)

    assert report["report"] == REPORT_NAME
    assert report["status"] == "complete"
    assert report["wall_seconds"] == 5
    assert report["provisioned_gpu_hours"] == pytest.approx(8 * 5 / 3600)
    learner = report["learner"]
    assert learner["measured_examples"] == 40
    assert learner["end_to_end_examples_per_second"] == 10
    assert learner["device_duty_fraction"] == 0.5
    assert learner["data_wait_fraction"] == 0.25
    assert learner["h2d_seconds"] == pytest.approx(0.4)
    actors = report["actors"]
    assert actors["worker_count"] == 1
    assert actors["aggregate_samples_per_second"] == 20
    assert actors["mean_policy_weight"] == 0.25
    trend = report["arena"]["by_baseline"]["frozen-shallow-v2"]
    assert trend["evaluations"] == 2
    assert trend["delta_elo"] == 20
    assert trend["delta_elo_per_gpu_hour"] > 0


def test_report_surfaces_jsonl_parse_failures_and_cli_exit_status(
    tmp_path, capsys
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-incomplete",
                "generation_family": "family-incomplete",
                "created_ns": 1,
            }
        ),
        encoding="utf-8",
    )
    metrics = root / "learner" / "metrics.jsonl"
    metrics.parent.mkdir()
    metrics.write_text("{bad json}\n", encoding="utf-8")

    report = build_strength_efficiency_report(root)

    assert report["status"] == "incomplete"
    assert report["parse_failure_count"] == 1
    assert main(["--run-root", str(root)]) == 3
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "incomplete"


def test_actor_rates_merge_overlapping_lanes_on_one_physical_gpu() -> None:
    records = [
        {
            "worker": f"actor-gpu-1-lane-{lane}",
            "gpu_id": 1,
            "batch_started_ns": 1_000_000_000,
            "batch_completed_ns": 11_000_000_000,
            "elapsed_seconds": 10,
            "games": 10,
            "samples": 100,
            "evaluator_rows": 1_000,
            "policy_samples": 20,
            "policy_weight_count": 25,
            "policy_weight_sum": 5,
        }
        for lane in range(2)
    ]

    summary = _actor_summary(records)

    assert summary["actor_lane_seconds"] == 20
    assert summary["actor_gpu_seconds"] == 10
    assert summary["fleet_wall_seconds"] == 10
    assert summary["aggregate_samples_per_second"] == 20
    assert summary["samples_per_physical_gpu_second"] == 20
    assert summary["mean_policy_weight"] == 0.2


def test_actor_policy_weight_count_falls_back_per_legacy_record() -> None:
    summary = _actor_summary(
        [
            {
                "worker": "actor-gpu-1-lane-0",
                "policy_samples": 20,
                "policy_weight_count": 25,
                "policy_weight_sum": 5,
            },
            {
                "worker": "actor-gpu-1",
                "policy_samples": 10,
                "policy_weight_sum": 5,
            },
        ]
    )

    assert summary["policy_weight_count"] == 35
    assert summary["mean_policy_weight"] == pytest.approx(10 / 35)


def test_learner_summary_falls_back_for_legacy_and_mixed_metrics() -> None:
    summary = _learner_summary(
        [
            {
                "timestamp_ns": 1,
                "step": 10,
                "step_seconds": 0.5,
                "examples_per_second": 100,
            },
            {
                "timestamp_ns": 2,
                "step": 20,
                "step_seconds": 0.5,
                "examples_per_second": 100,
            },
            {
                "timestamp_ns": 3,
                "step": 22,
                "metrics_interval_steps": 2,
                "global_batch_size": 64,
                "step_seconds": 0.75,
                "metrics_interval_wall_seconds": 1.5,
                "device_step_seconds": 0.25,
            },
        ]
    )

    assert summary["measured_steps"] == 13
    assert summary["measured_examples"] == 678
    assert summary["measured_wall_seconds"] == 7
    assert summary["legacy_metric_records"] == 2


def test_legacy_learner_step_deltas_reset_when_model_step_rewinds() -> None:
    summary = _learner_summary(
        [
            {"timestamp_ns": 1, "step": 20, "step_seconds": 1},
            {"timestamp_ns": 2, "step": 5, "step_seconds": 1},
            {"timestamp_ns": 3, "step": 7, "step_seconds": 1},
        ]
    )

    assert summary["measured_steps"] == 4
    assert summary["measured_wall_seconds"] == 4


def test_report_rejects_missing_identity(tmp_path, capsys) -> None:
    assert main(["--run-root", str(tmp_path)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "error"
