from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.diagnose_learning_dynamics import build_report


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_learning_dynamics_report_summarizes_metrics_and_arena(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _json(
        root / "run.json",
        {
            "schema_version": 1,
            "run_id": "run-1",
            "generation_family": "family-1",
            "created_ns": 1,
        },
    )
    (root / "profile.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"width": 384},
                "optimizer": {"kind": "muon_adamw"},
                "train": {"per_rank_batch_size": 512, "ema_decay": 0.9999},
                "learner": {
                    "candidate_interval": 10_000,
                    "candidate_interval_examples": 2_000_000,
                    "target_updates_per_new_sample": 1.0,
                    "max_replay_lag_steps": 60_000,
                },
                "orchestration": {"training_objective": "ring10_only"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _jsonl(
        root / "learner" / "metrics.jsonl",
        [
            {
                "step": 1,
                "examples_per_second": 1000,
                "gradient_norm": 2.0,
                "gradient_clipped": True,
                "losses": {
                    "total": 3.0,
                    "policy": 2.0,
                    "clinch_score_margin": 0.0,
                    "clinch_score_margin_available": 0,
                },
            },
            {
                "step": 2,
                "examples_per_second": 1200,
                "gradient_norm": 1.0,
                "gradient_clipped": False,
                "losses": {
                    "total": 2.0,
                    "policy": 1.0,
                    "clinch_score_margin": 2.0,
                    "clinch_score_margin_available": 1,
                },
            },
        ],
    )
    _jsonl(
        root / "metrics" / "actor-gpu-1.jsonl",
        [
            {"model_role": "champion", "samples": 100},
            {"model_role": "candidate", "samples": 40},
        ],
    )
    _jsonl(
        root / "learner" / "model-history.jsonl",
        [
            {"model_step": 100, "examples_consumed": 1000},
            {"model_step": 200, "examples_consumed": 3000},
        ],
    )
    _json(
        root / "learner" / "champion.json",
        {"model_step": 100, "model_identity": "sha256-champion"},
    )
    _json(root / "learner" / "recovery.json", {"step": 200})
    _json(
        root / "arena" / "sha256-result.json",
        {
            "terminal": True,
            "candidate": "candidate",
            "baseline": "champion",
            "completed_ns": 10,
            "games": [{}, {}],
            "promotion": {
                "decision": "reject",
                "pair_score_rate": 0.4,
                "confidence_sequence": [0.2, 0.6],
            },
        },
    )

    report = build_report([("control", root)])
    run = report["runs"][0]

    assert run["learner_metrics"]["examples_per_second"]["mean"] == 1100
    assert run["learner_metrics"]["gradient_clipped"]["mean"] == 0.5
    assert run["learner_metrics"]["losses.clinch_score_margin"]["count"] == 1
    assert run["learner_metrics"]["losses.clinch_score_margin"]["mean"] == 2.0
    assert run["selfplay_source_role_rows"] == {"candidate": 1, "champion": 1}
    assert run["selfplay_source_samples"] == {"candidate": 40.0, "champion": 100.0}
    assert run["publication_step_deltas"] == [100]
    assert run["publication_example_deltas"] == [2000]
    assert run["terminal_decisions"] == {"reject": 1}
    assert run["terminal_evaluations"][0]["pair_score_rate"] == 0.4
