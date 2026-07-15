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


def _checkpoint_arena_result(
    *,
    candidate: str,
    baseline: str,
    completed_ns: int,
    wins: int,
    losses: int,
    ring_wins: int,
    ring_losses: int,
) -> dict:
    return {
        "schema_version": 3,
        "candidate": candidate,
        "baseline": baseline,
        "baseline_metadata": {
            "kind": "checkpoint",
            "identity": baseline,
        },
        "started_ns": completed_ns - 100,
        "completed_ns": completed_ns,
        "evaluation_metrics": {"wall_seconds": 1.0},
        "aggregate": {
            "wins": wins,
            "losses": losses,
            "games": wins + losses,
            "elo_difference": 0.0,
            "anytime_elo_interval": [-100.0, 100.0],
        },
        "per_ring": {
            "10": {
                "wins": ring_wins,
                "losses": ring_losses,
                "games": ring_wins + ring_losses,
                "elo_difference": 0.0,
                "anytime_elo_interval": [-100.0, 100.0],
            }
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


def test_report_builds_autonomous_checkpoint_ladders_and_efficiency(
    tmp_path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-autonomous-elo",
                "generation_family": "family-autonomous-elo",
                "created_ns": 1_000_000_000,
            }
        ),
        encoding="utf-8",
    )
    identities = {
        "anchor": "checkpoint-anchor",
        "middle": "checkpoint-middle",
        "latest": "checkpoint-latest",
        "x": "checkpoint-x",
        "y": "checkpoint-y",
    }
    manifests = root / "learner" / "manifests"
    manifests.mkdir(parents=True)
    for step, identity in enumerate(identities.values()):
        (manifests / f"manifest-{step}.json").write_text(
            json.dumps({"model_identity": identity, "model_step": step}),
            encoding="utf-8",
        )
    (manifests / "manifest-0.json").unlink()
    _write_jsonl(
        root / "learner" / "model-history.jsonl",
        [
            {
                "schema_version": 1,
                "model_identity": identities["anchor"],
                "model_step": 0,
            }
        ],
    )
    _write_jsonl(
        root / "metrics" / "actor-gpu-0-lane-0.jsonl",
        [
            {
                "timestamp_ns": 5_000_000_000,
                "worker": "actor-gpu-0-lane-0",
                "gpu_id": 0,
                "model_identity": identities["latest"],
                "model_step": 2,
                "evaluator_rows": 2_000_000_000,
            }
        ],
    )
    arena = root / "arena"
    arena.mkdir()
    results = {
        "middle-vs-anchor.json": _checkpoint_arena_result(
            candidate=identities["middle"],
            baseline=identities["anchor"],
            completed_ns=3_000_000_000,
            wins=70,
            losses=30,
            ring_wins=65,
            ring_losses=35,
        ),
        "latest-vs-middle.json": _checkpoint_arena_result(
            candidate=identities["latest"],
            baseline=identities["middle"],
            completed_ns=4_000_000_000,
            wins=70,
            losses=30,
            ring_wins=60,
            ring_losses=40,
        ),
        "disconnected.json": _checkpoint_arena_result(
            candidate=identities["y"],
            baseline=identities["x"],
            completed_ns=4_500_000_000,
            wins=80,
            losses=20,
            ring_wins=75,
            ring_losses=25,
        ),
        "empty.json": _checkpoint_arena_result(
            candidate=identities["latest"],
            baseline=identities["anchor"],
            completed_ns=4_750_000_000,
            wins=0,
            losses=0,
            ring_wins=0,
            ring_losses=0,
        ),
        "frozen.json": {
            **_checkpoint_arena_result(
                candidate=identities["latest"],
                baseline="frozen-shallow-v2",
                completed_ns=5_000_000_000,
                wins=90,
                losses=10,
                ring_wins=85,
                ring_losses=15,
            ),
            "baseline_metadata": {
                "kind": "shallow-search",
                "identity": "frozen-shallow-v2",
            },
        },
    }
    for name, payload in results.items():
        (arena / name).write_text(json.dumps(payload), encoding="utf-8")

    report = build_strength_efficiency_report(root, provisioned_gpus=4)
    repeated = build_strength_efficiency_report(root, provisioned_gpus=4)
    autonomous = report["autonomous_elo"]
    primary = autonomous["primary_ring_10"]
    aggregate = autonomous["aggregate"]

    assert report == repeated
    assert autonomous["anchor"] == {
        "identity": identities["anchor"],
        "step": 0,
        "rating": 0.0,
        "selection": "step_zero_snapshot",
    }
    assert primary["status"] == aggregate["status"] == "available"
    assert {item["identity"] for item in primary["ladder"]} == {
        identities["anchor"],
        identities["middle"],
        identities["latest"],
    }
    assert primary["latest"]["identity"] == identities["latest"]
    assert primary["latest"]["step"] == 2
    assert primary["latest"]["rating"] > 0
    assert primary["connectedness"]["connected"] is False
    assert primary["connectedness"]["excluded_identities"] == [
        identities["x"],
        identities["y"],
    ]
    assert any(
        "at least one game" in str(exclusion["reason"])
        for exclusion in primary["exclusions"]
    )
    assert autonomous["latest_elo"] == primary["latest"]["rating"]
    efficiency = autonomous["efficiency"]
    assert efficiency["leaf_evaluations"] == 2_000_000_000
    assert efficiency["elo_per_billion_leaf_evaluations"] == pytest.approx(
        autonomous["latest_elo"] / 2
    )
    assert efficiency["elo_per_provisioned_gpu_hour"] > 0
    frozen = autonomous["frozen_baselines"]
    assert frozen["connected_to_primary"] is False
    assert frozen["result_count"] == 1
    assert frozen["results"][0]["baseline"] == "frozen-shallow-v2"
    assert all(item["identity"] != "frozen-shallow-v2" for item in primary["ladder"])
