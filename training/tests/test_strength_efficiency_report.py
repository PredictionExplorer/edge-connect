from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import strength_efficiency_report as report_module
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
    _write_jsonl(
        root / "metrics" / "coordinator.jsonl",
        [
            {
                "timestamp_ns": 2_000_000_000,
                "event": "pause_lease_ready",
                "token": "lease-one",
            },
            {
                "timestamp_ns": 4_000_000_000,
                "event": "pause_lease_released",
                "token": "lease-one",
            },
            {
                "timestamp_ns": 4_100_000_000,
                "event": "pause_target_restarted",
                "target": "learner",
            },
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
    assert report["coordinator"]["pause_lease_seconds"] == 2
    assert report["coordinator"]["learner_pause_restarts"] == 1
    assert "remain included" in report["coordinator"]["efficiency_denominator_policy"]
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
    metrics.write_text("\n{bad json}\n\n", encoding="utf-8")

    report = build_strength_efficiency_report(root)

    assert report["status"] == "incomplete"
    assert report["parse_failure_count"] == 1
    assert main(["--run-root", str(root)]) == 3
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "incomplete"


def test_report_cli_publishes_atomically_and_supports_quiet_mode(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "generation_family": "family",
                "created_ns": 1,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    writes = []
    real_atomic_json = report_module.atomic_json

    def recording_atomic_json(path, payload):
        writes.append(Path(path))
        real_atomic_json(path, payload)

    monkeypatch.setattr(report_module, "atomic_json", recording_atomic_json)

    assert (
        main(
            [
                "--run-root",
                str(root),
                "--output",
                str(output),
                "--quiet",
            ]
        )
        == 0
    )
    assert writes == [output]
    assert json.loads(output.read_text())["status"] == "complete"
    assert capsys.readouterr().out == ""
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_report_uses_coordinator_terminal_timestamp_for_idle_wall_time(
    tmp_path,
) -> None:
    root = tmp_path / "run-terminal"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-terminal",
                "generation_family": "family-terminal",
                "created_ns": 1_000_000_000,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        root / "learner" / "metrics.jsonl",
        [{"timestamp_ns": 2_000_000_000, "step": 1}],
    )
    status = root / "status"
    status.mkdir()
    (status / "coordinator.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "stopped",
                "timestamp_ns": 4_000_000_000,
            }
        ),
        encoding="utf-8",
    )

    report = build_strength_efficiency_report(root, provisioned_gpus=8)

    assert report["observed_until_ns"] == 4_000_000_000
    assert report["observation_end_source"] == "coordinator_terminal_timestamp"
    assert report["wall_seconds"] == 3
    assert report["provisioned_gpu_hours"] == pytest.approx(24 / 3600)


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
            json.dumps(
                {
                    "model_identity": identity,
                    "model_step": step,
                    "published_ns": (step + 1) * 1_000_000_000,
                }
            ),
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
                "published_ns": 1_000_000_000,
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
    assert len(primary["marginal_contrasts"]) == 2
    assert primary["marginal_contrasts"][-1]["to_step"] == 2
    assert primary["marginal_contrasts"][-1]["time_basis"] == "checkpoint_publication"
    assert primary["marginal_contrasts"][-1]["elapsed_wall_hours"] == pytest.approx(
        1 / 3600
    )
    assert (
        primary["marginal_contrasts"][-1]["confidence_interval"][0]
        < primary["marginal_contrasts"][-1]["delta_elo"]
        < primary["marginal_contrasts"][-1]["confidence_interval"][1]
    )
    assert primary["connectedness"]["connected"] is False
    assert primary["connectedness"]["excluded_identities"] == [
        identities["x"],
        identities["y"],
    ]
    assert any(
        "at least one game" in str(exclusion["reason"])
        for exclusion in primary["exclusions"]
    )
    assert autonomous["latest"]["source"] == "ring_10"
    assert autonomous["latest_elo"] == primary["latest"]["rating"]
    assert autonomous["headline"]["source"] == "aggregate"
    assert (
        autonomous["headline"]["confidence_interval"]
        == aggregate["latest"]["confidence_interval"]
    )
    assert autonomous["headline_elo"] == aggregate["latest"]["rating"]
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


def test_learner_summary_exposes_segment_utd_and_persistent_window_metrics() -> None:
    summary = _learner_summary(
        [
            {
                "timestamp_ns": 1,
                "event": "replay_window_allocated",
                "step": 10,
                "window_batches_allocated": 100,
                "window_batches_consumed": 0,
                "loader_workers_effective": 8,
                "window_setup_seconds": 4.0,
                "window_setup_amortized_seconds": 0.04,
            },
            {
                "timestamp_ns": 2,
                "step": 12,
                "step_seconds": 1.0,
                "metrics_interval_steps": 2,
                "global_batch_size": 64,
                "lifetime_updates_per_new_sample": 1.04,
                "updates_per_new_sample": 1.04,
                "segment_updates_per_new_sample": 1.2,
                "utd_segment_target_updates_per_new_sample": 1.25,
                "utd_segment_baseline_examples_consumed": 1_000,
                "utd_segment_baseline_committed_replay_samples": 800,
                "loader_workers_effective": 8,
                "window_batches_allocated": 100,
                "window_batches_consumed": 2,
                "window_reuse": False,
                "window_reuse_spins": 0,
                "window_setup_amortized_seconds": 0.04,
            },
            {
                "timestamp_ns": 3,
                "event": "replay_loader_pool_rebound",
                "step": 14,
                "loader_lifecycle": "process",
                "loader_workers_effective": 8,
                "loader_pool_starts": 1,
                "loader_pool_rebinds": 2,
                "loader_pool_shutdowns": 0,
                "loader_worker_pids": [101, 102],
            },
            {
                "timestamp_ns": 4,
                "event": "replay_window_consumed",
                "step": 15,
                "window_batches_allocated": 100,
                "window_batches_consumed": 5,
                "window_batches_consumed_this_spin": 3,
                "window_batches_remaining": 95,
                "window_reuse": True,
                "window_reuse_spins": 1,
                "loader_workers_effective": 8,
            },
        ]
    )

    assert summary["updates_per_new_sample"] == 1.04
    assert summary["lifetime_updates_per_new_sample"] == 1.04
    assert summary["segment_updates_per_new_sample"] == 1.2
    assert summary["utd_segment_target_updates_per_new_sample"] == 1.25
    assert summary["loader_workers_effective"] == 8
    assert summary["loader_lifecycle"] == "process"
    assert summary["loader_pool_starts"] == 1
    assert summary["loader_pool_rebinds"] == 2
    assert summary["loader_pool_shutdowns"] == 0
    assert summary["loader_worker_pids"] == [101, 102]
    assert summary["window_reuse"] is True
    assert summary["window_reuse_spins"] == 1
    persistent = summary["persistent_window"]
    assert persistent["allocation_records"] == 1
    assert persistent["consumption_records"] == 1
    assert persistent["reused_consumption_records"] == 1
    assert persistent["reuse_fraction"] == 1.0
    assert persistent["consumed_batches"] == 3
    assert persistent["setup_seconds"] == 4.0


def test_report_classifies_crossplay_and_legacy_arena_results(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-result-kinds",
                "generation_family": "family-result-kinds",
                "created_ns": 1_000_000_000,
            }
        ),
        encoding="utf-8",
    )
    arena = root / "arena"
    arena.mkdir()
    legacy = _arena_result(completed_ns=2_000_000_000, elo=10, lower=-10)
    crossplay = {
        **_arena_result(completed_ns=3_000_000_000, elo=20, lower=0),
        "result_kind": "historical_crossplay",
    }
    (arena / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")
    (arena / "crossplay.json").write_text(json.dumps(crossplay), encoding="utf-8")

    report = build_strength_efficiency_report(root)

    assert report["arena"]["result_kind_counts"] == {
        "promotion": 1,
        "crossplay": 0,
        "historical_crossplay": 1,
        "unknown": 0,
    }
    assert report["arena"]["result_category_counts"] == {
        "promotion": 1,
        "crossplay": 1,
        "unknown": 0,
    }
    assert [item["result_kind"] for item in report["arena"]["results"]] == [
        "promotion",
        "historical_crossplay",
    ]


def test_report_follows_forked_arena_ancestry_and_conclusiveness(tmp_path) -> None:
    """Forked runs keep one connected ladder and separate inconclusive results."""

    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-fork",
                "generation_family": "family-fork",
                "created_ns": 1_000_000_000,
            }
        ),
        encoding="utf-8",
    )

    def promotion_result(
        *,
        completed_ns: int,
        decision: str,
        conclusive: bool | None,
    ) -> dict:
        result = {
            **_arena_result(completed_ns=completed_ns, elo=5.0, lower=-30.0),
            "result_kind": "promotion",
            "terminal": True,
            "promotion": {"decision": decision},
        }
        if conclusive is not None:
            result["conclusive"] = conclusive
        return result

    current = root / "arena"
    current.mkdir()
    parent = root / "ablation-parent" / "arena"
    parent.mkdir(parents=True)
    grandparent = root / "ablation-parent" / "ancestor" / "arena"
    grandparent.mkdir(parents=True)
    (grandparent / "oldest.json").write_text(
        json.dumps(
            promotion_result(
                completed_ns=2_000_000_000, decision="promote", conclusive=None
            )
        ),
        encoding="utf-8",
    )
    (parent / "capped.json").write_text(
        json.dumps(
            promotion_result(
                completed_ns=3_000_000_000,
                decision="reject_max_pairs",
                conclusive=None,
            )
        ),
        encoding="utf-8",
    )
    (parent / "superseded.json").write_text(
        json.dumps(
            promotion_result(
                completed_ns=3_500_000_000, decision="superseded", conclusive=None
            )
        ),
        encoding="utf-8",
    )
    (current / "recorded.json").write_text(
        json.dumps(
            promotion_result(
                completed_ns=4_000_000_000, decision="reject", conclusive=True
            )
        ),
        encoding="utf-8",
    )
    (current / "recorded-capped.json").write_text(
        json.dumps(
            promotion_result(
                completed_ns=5_000_000_000,
                decision="reject_max_pairs",
                conclusive=False,
            )
        ),
        encoding="utf-8",
    )

    report = build_strength_efficiency_report(root)
    arena = report["arena"]
    results = arena["results"]
    assert [item["ancestry_depth"] for item in results] == [2, 1, 1, 0, 0]
    assert [item["conclusive"] for item in results] == [True, False, False, True, False]
    assert arena["ancestor_result_count"] == 3
    assert arena["terminal_evaluation_count"] == 4
    assert arena["inconclusive_evaluation_count"] == 2
    assert arena["inconclusive_evaluation_fraction"] == pytest.approx(0.5)
    assert report_module._ancestor_arena_directories(root) == [
        (1, parent),
        (2, grandparent),
    ]
    assert report_module._ancestor_arena_directories(tmp_path / "missing") == []


def test_ladder_is_fitted_at_the_anchor_search_budget_only(tmp_path) -> None:
    """A cheaper promotion gate must not be mixed into the historical ladder."""

    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-budget",
                "generation_family": "family-budget",
                "created_ns": 1_000_000_000,
            }
        ),
        encoding="utf-8",
    )
    arena = root / "arena"
    arena.mkdir()
    anchor = "sha256-" + "a" * 64
    second = "sha256-" + "b" * 64
    third = "sha256-" + "c" * 64

    def budgeted(result: dict, simulations: int, max_considered: int) -> dict:
        result["search"] = {
            "simulations": simulations,
            "max_considered": max_considered,
        }
        result["baseline_metadata"]["search_budget"] = {
            "simulations": simulations,
            "max_considered": max_considered,
        }
        return result

    (arena / "promotion-1024.json").write_text(
        json.dumps(
            budgeted(
                _checkpoint_arena_result(
                    candidate=second,
                    baseline=anchor,
                    completed_ns=2_000_000_000,
                    wins=120,
                    losses=80,
                    ring_wins=120,
                    ring_losses=80,
                ),
                1024,
                32,
            )
        ),
        encoding="utf-8",
    )
    (arena / "gate-256.json").write_text(
        json.dumps(
            budgeted(
                _checkpoint_arena_result(
                    candidate=third,
                    baseline=second,
                    completed_ns=3_000_000_000,
                    wins=300,
                    losses=100,
                    ring_wins=300,
                    ring_losses=100,
                ),
                256,
                32,
            )
        ),
        encoding="utf-8",
    )
    (arena / "measurement-1024.json").write_text(
        json.dumps(
            {
                **budgeted(
                    _checkpoint_arena_result(
                        candidate=third,
                        baseline=second,
                        completed_ns=4_000_000_000,
                        wins=110,
                        losses=90,
                        ring_wins=110,
                        ring_losses=90,
                    ),
                    1024,
                    32,
                ),
                "result_kind": "historical_crossplay",
                "crossplay_kind": "measurement",
            }
        ),
        encoding="utf-8",
    )

    report = build_strength_efficiency_report(root)
    elo = report["autonomous_elo"]
    assert elo["search_budget"]["ladder"] == {"simulations": 1024, "max_considered": 32}
    assert elo["search_budget"]["decisive_games_by_budget"] == {
        "1024x32": 400,
        "256x32": 400,
    }
    ladder = elo["primary_ring_10"]
    assert ladder["status"] == "available"
    assert ladder["input"]["decisive_games"] == 400
    excluded = [
        item for item in ladder["exclusions"] if "search budget" in item["reason"]
    ]
    assert len(excluded) == 1
    assert excluded[0]["path"].endswith("gate-256.json")
    assert excluded[0]["reason"] == (
        "search budget 256x32 differs from ladder budget 1024x32"
    )
    ratings = {entry["identity"]: entry["rating"] for entry in ladder["ladder"]}
    assert ratings[anchor] == 0.0
    assert ratings[second] > 0.0
    assert ratings[third] > ratings[second]
    # The 256-simulation result would have implied a far larger gap.
    assert ratings[third] - ratings[second] < 100.0


def test_learning_rate_governance_summary_reconstructs_legacy_compounding() -> None:
    from scripts.strength_efficiency_report import _learning_rate_governance_summary

    legacy_records = [
        {"step": 10, "learning_rates": [6.1e-4, 9e-6]},
        {
            "event": "plateau_reset",
            "champion_identity": "sha256-a",
            "learning_rate_scale": 0.5,
            "learning_rates": [3.05e-4, 4.5e-6],
        },
        {
            "event": "plateau_reset",
            "champion_identity": "sha256-a",
            "learning_rate_scale": 0.5,
            "learning_rates": [3.05e-4, 4.5e-6],
        },
        {
            "event": "plateau_reset",
            "champion_identity": "sha256-b",
            "learning_rate_scale": 0.5,
            "learning_rates": [1.5e-4, 2.25e-6],
        },
        {"step": 20, "learning_rates": [1.5e-4, 2.25e-6]},
    ]
    legacy = _learning_rate_governance_summary(legacy_records)
    assert legacy["status"] == "legacy"
    assert legacy["legacy_reset_count"] == 3
    assert legacy["legacy_reset_champions"] == 2
    assert legacy["legacy_compounded_scale"] == pytest.approx(0.25)
    assert legacy["collapse_warning"] is True
    assert legacy["latest_effective_learning_rates"] == [1.5e-4, 2.25e-6]

    governed_records = [
        {"event": "learning_rate_governor_legacy", "step": 0},
        {
            "step": 10,
            "learning_rates": [3e-4, 4.5e-6],
            "learning_rate_multiplier": 1.0,
            "reference_learning_rates": [3e-4, 4.5e-6],
        },
        {
            "event": "plateau_recovery",
            "learning_rate_reduced": True,
            "learning_rate_multiplier": 0.5,
            "reference_learning_rates": [3e-4, 4.5e-6],
        },
        {
            "event": "plateau_recovery",
            "learning_rate_reduced": False,
            "learning_rate_multiplier": 0.5,
        },
        {"event": "plateau_scale_restored", "learning_rate_multiplier": 1.0},
        {
            "step": 20,
            "learning_rates": [2.9e-4, 4.4e-6],
            "learning_rate_multiplier": 1.0,
            "reference_learning_rates": [3e-4, 4.5e-6],
        },
    ]
    governed = _learning_rate_governance_summary(governed_records)
    assert governed["status"] == "governed"
    assert governed["latest_multiplier"] == 1.0
    assert governed["minimum_multiplier_observed"] == 0.5
    assert governed["reductions"] == 1
    assert governed["lag_releases"] == 1
    assert governed["restorations"] == 1
    assert governed["legacy_reference_events"] == 1
    assert governed["legacy_reset_count"] == 0
    assert governed["legacy_compounded_scale"] == 1.0
    assert governed["collapse_warning"] is False
    assert governed["latest_reference_learning_rates"] == [3e-4, 4.5e-6]
    assert _learning_rate_governance_summary([])["status"] == "unknown"


def test_report_identifies_saturated_one_sided_pairings(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-saturated",
                "generation_family": "family-saturated",
                "created_ns": 1_000_000_000,
            }
        ),
        encoding="utf-8",
    )
    manifests = root / "learner" / "manifests"
    manifests.mkdir(parents=True)
    for step, identity in enumerate(("checkpoint-anchor", "checkpoint-latest")):
        (manifests / f"manifest-{step}.json").write_text(
            json.dumps({"model_identity": identity, "model_step": step}),
            encoding="utf-8",
        )
    arena = root / "arena"
    arena.mkdir()
    (arena / "saturated.json").write_text(
        json.dumps(
            _checkpoint_arena_result(
                candidate="checkpoint-latest",
                baseline="checkpoint-anchor",
                completed_ns=2_000_000_000,
                wins=100,
                losses=0,
                ring_wins=60,
                ring_losses=40,
            )
        ),
        encoding="utf-8",
    )

    report = build_strength_efficiency_report(root)
    aggregate = report["autonomous_elo"]["aggregate"]

    assert aggregate["input"]["continuity_corrected_pairings"] == 1
    assert aggregate["input"]["saturated_one_sided_pairing_count"] == 1
    pairing = aggregate["input"]["saturated_one_sided_pairings"][0]
    assert pairing["decisive_games"] == 100
    assert {pairing["first_wins"], pairing["second_wins"]} == {0, 100}
    assert (
        report["autonomous_elo"]["saturation"]["aggregate"][
            "saturated_one_sided_pairing_count"
        ]
        == 1
    )


def test_report_exposes_validated_migration_boundaries_and_segments(
    tmp_path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-migrated",
                "generation_family": "family-migrated",
                "created_ns": 1_000_000_000,
            }
        ),
        encoding="utf-8",
    )
    before = "a" * 64
    middle = "b" * 64
    after = "c" * 64
    _write_jsonl(
        root / "autonomous-migrations.jsonl",
        [
            {
                "schema_version": 1,
                "timestamp_ns": 2_000_000_000,
                "run_id": "run-migrated",
                "generation_family": "family-migrated",
                "from_config_sha256": before,
                "to_config_sha256": middle,
                "from_profile": "profile.yaml",
                "to_profile": "profile-cadence-v2.yaml",
                "learner_step": 80,
                "examples_consumed": 8_000,
                "reason": "legacy-cadence",
            },
            {
                "schema_version": 1,
                "timestamp_ns": 3_000_000_000,
                "run_id": "run-migrated",
                "generation_family": "family-migrated",
                "from_config_sha256": middle,
                "to_config_sha256": after,
                "from_profile": "profile-cadence-v2.yaml",
                "to_profile": "profile-elo-v3.yaml",
                "learner_step": 100,
                "examples_consumed": 10_000,
                "committed_replay_samples": 8_000,
                "target_updates_per_new_sample": 1.25,
                "reason": "prospective-utd",
            },
        ],
    )

    report = build_strength_efficiency_report(root)
    migrations = report["migrations"]

    assert migrations["record_count"] == 2
    assert migrations["boundary_count"] == 2
    assert migrations["boundaries"] == [
        {
            "migration_id": "migration-2000000000",
            "timestamp_ns": 2_000_000_000,
            "from_sha256": before,
            "to_sha256": middle,
            "step": 80,
            "examples_consumed": 8_000,
            "committed_replay_samples": None,
            "target_updates_per_new_sample": None,
        },
        {
            "migration_id": "migration-3000000000",
            "timestamp_ns": 3_000_000_000,
            "from_sha256": middle,
            "to_sha256": after,
            "step": 100,
            "examples_consumed": 10_000,
            "committed_replay_samples": 8_000,
            "target_updates_per_new_sample": 1.25,
        },
    ]
    assert len(migrations["segments"]) == 3
    assert migrations["segments"][-1]["config_sha256"] == after
    assert report["migration_boundaries"] == migrations["boundaries"]
    assert report["migration_segments"] == migrations["segments"]


def test_report_rejects_malformed_active_migration_record(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-malformed-migration",
                "generation_family": "family-malformed-migration",
                "created_ns": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        root / "autonomous-migrations.jsonl",
        [
            {
                "schema_version": 1,
                "timestamp_ns": 2,
                "run_id": "run-malformed-migration",
                "generation_family": "family-malformed-migration",
                "from_sha256": "a" * 64,
            }
        ],
    )

    with pytest.raises(ValueError, match="source/target config hashes"):
        build_strength_efficiency_report(root)


def test_report_ignores_migration_records_for_another_run(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": "active-run",
                "generation_family": "active-family",
                "created_ns": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        root / "autonomous-migrations.jsonl",
        [{"run_id": "different-run", "malformed": True}],
    )

    report = build_strength_efficiency_report(root)

    assert report["migrations"]["record_count"] == 0
    assert report["migrations"]["ignored_record_count"] == 1
