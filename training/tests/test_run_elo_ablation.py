from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.fork_elo_ablation import fork_elo_ablation
from scripts.prepare_elo_ablation import prepare_elo_ablation
from scripts.run_elo_ablation import EvaluatorRows, main, run_elo_ablation

CONFIGS = Path(__file__).parents[1] / "configs"


def test_evaluator_rows_incrementally_reads_actor_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    first = metrics / "actor-gpu-1.jsonl"
    first.write_text(
        json.dumps({"evaluator_rows": 10}) + "\n",
        encoding="utf-8",
    )
    tracker = EvaluatorRows(metrics)

    assert tracker.refresh() == 10
    assert tracker.refresh() == 10
    with first.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"evaluator_rows": 15}) + "\n")
    (metrics / "actor-gpu-2.jsonl").write_text(
        json.dumps({"evaluator_rows": 20}) + "\n",
        encoding="utf-8",
    )

    assert tracker.refresh() == 45
    with first.open("a", encoding="utf-8") as stream:
        stream.write('{"evaluator_rows": 5')
    assert tracker.refresh() == 45
    with first.open("a", encoding="utf-8") as stream:
        stream.write("}\n")
    assert tracker.refresh() == 50


def _forked_run(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    (source / "learner").mkdir(parents=True)
    (source / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "shared-run",
                "generation_family": "shared-family",
                "created_ns": 1,
            }
        ),
        encoding="utf-8",
    )
    for name, identity, step in (
        ("champion.json", "champion", 1),
        ("candidate.json", "candidate", 2),
    ):
        (source / "learner" / name).write_text(
            json.dumps(
                {
                    "model_identity": identity,
                    "model_step": step,
                    "updated_ns": step,
                }
            ),
            encoding="utf-8",
        )
    profiles = tmp_path / "profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=profiles,
        run_root_parent=tmp_path / "runs",
        run_id="shared-run",
        source_run_root=source,
        prefix="pilot",
        seed=17,
        wall_budget_hours=1,
        leaf_budget=100,
        guard_floor_elo=-35,
        treatments=("control",),
    )
    fork_elo_ablation(
        source_run_root=source,
        plan_path=profiles / "ablation-plan.json",
        treatment="control",
    )
    root = tmp_path / "runs" / "pilot-control-seed17"
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["wall_budget_seconds"] = 0.05
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return root, root / "profile-elo-ablation.yaml"


def test_runner_stops_at_wall_budget_and_records_lifecycle(tmp_path: Path) -> None:
    root, profile = _forked_run(tmp_path)
    orchestrator = tmp_path / "fake-orchestrator"
    orchestrator.write_text(
        """#!/usr/bin/env python3
import signal
import time

stopping = False
def stop(_signal, _frame):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
while not stopping:
    time.sleep(0.01)
""",
        encoding="utf-8",
    )
    os.chmod(orchestrator, 0o755)

    report = run_elo_ablation(
        config_path=profile,
        orchestrator=str(orchestrator),
        poll_seconds=0.01,
    )

    assert report["status"] == "complete"
    assert report["stop_reason"] == "wall_budget"
    metadata = json.loads((root / "ablation.json").read_text())
    assert metadata["measurement_started_ns"] > 0
    assert metadata["measurement_stopped_ns"] >= metadata["measurement_started_ns"]
    assert metadata["measurement_stop_reason"] == "wall_budget"
    assert metadata["measurement_exit_code"] in (0, -15)


def test_runner_cli_rejects_second_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["measurement_started_ns"] = 10
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    exit_code = main(["--config", str(profile)])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "already started" in payload["error"]
