from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

import scripts.run_elo_ablation as run_module
from scripts.fork_elo_ablation import fork_elo_ablation
from scripts.prepare_elo_ablation import prepare_elo_ablation
from scripts.run_elo_ablation import (
    BUDGET_COMPLETION,
    FATAL_ORCHESTRATOR_EXIT,
    TRANSIENT_CRASH,
    EvaluatorRows,
    main,
    run_elo_ablation,
)

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.fixture(autouse=True)
def _successful_state_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_module,
        "run_state_preflight",
        lambda _root, _profile, *, apply: {
            "status": "ok",
            "mode": "apply" if apply else "dry-run",
        },
    )


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
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    report = run_elo_ablation(
        config_path=profile,
        orchestrator=str(orchestrator),
        poll_seconds=0.01,
    )

    assert report["status"] == "complete"
    assert report["outcome"] == BUDGET_COMPLETION
    assert report["stop_reason"] == "wall_budget"
    metadata = json.loads((root / "ablation.json").read_text())
    assert metadata["measurement_started_ns"] > 0
    assert metadata["measurement_stopped_ns"] >= metadata["measurement_started_ns"]
    assert metadata["measurement_stop_reason"] == "wall_budget"
    assert metadata["measurement_exit_code"] in (0, -15)
    assert metadata["measurement_status"] == "complete"
    assert metadata["measurement_outcome"] == BUDGET_COMPLETION
    assert metadata["measurement_attempt_count"] == 1
    assert metadata["measurement_attempts"][0]["outcome"] == BUDGET_COMPLETION
    assert metadata["state_preflight"] == {"status": "ok", "mode": "apply"}
    assert signal.getsignal(signal.SIGINT) == previous_sigint
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm


def test_runner_resumes_after_transient_signal_crash(tmp_path: Path) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["wall_budget_seconds"] = 0.3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    orchestrator = tmp_path / "crash-once-orchestrator"
    orchestrator.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import signal
import sys
import time

profile = Path(sys.argv[sys.argv.index("--config") + 1])
marker = profile.parent / "transient-crash-observed"
if not marker.exists():
    marker.write_text("crashed", encoding="utf-8")
    os.kill(os.getpid(), signal.SIGKILL)

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

    first = run_elo_ablation(
        config_path=profile,
        orchestrator=str(orchestrator),
        poll_seconds=0.01,
    )
    first_started_ns = first["started_ns"]
    second = run_elo_ablation(
        config_path=profile,
        orchestrator=str(orchestrator),
        poll_seconds=0.01,
    )

    assert first["status"] == "retryable"
    assert first["outcome"] == TRANSIENT_CRASH
    assert first["exit_code"] == -signal.SIGKILL
    assert second["status"] == "complete"
    assert second["outcome"] == BUDGET_COMPLETION
    assert second["started_ns"] == first_started_ns
    persisted = json.loads(metadata_path.read_text())
    assert persisted["measurement_attempt_count"] == 2
    assert [attempt["outcome"] for attempt in persisted["measurement_attempts"]] == [
        TRANSIENT_CRASH,
        BUDGET_COMPLETION,
    ]


def test_runner_recovers_unfinalized_attempt_without_resetting_wall_clock(
    tmp_path: Path,
) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    started_ns = time.time_ns() - 1_000_000_000
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "wall_budget_seconds": 0.01,
            "measurement_started_ns": started_ns,
            "measurement_status": "running",
            "measurement_outcome": None,
            "measurement_attempt_count": 1,
            "measurement_attempts": [
                {
                    "attempt": 1,
                    "started_ns": started_ns,
                    "stopped_ns": None,
                    "outcome": "running",
                    "status": "running",
                }
            ],
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    orchestrator = tmp_path / "unused-orchestrator"
    orchestrator.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(orchestrator, 0o755)

    report = run_elo_ablation(
        config_path=profile,
        orchestrator=str(orchestrator),
        poll_seconds=0.01,
    )
    persisted = json.loads(metadata_path.read_text())

    assert report["status"] == "complete"
    assert report["outcome"] == BUDGET_COMPLETION
    assert report["started_ns"] == started_ns
    assert report["exit_code"] is None
    assert persisted["measurement_attempt_count"] == 2
    assert persisted["measurement_attempts"][0]["status"] == "retryable"
    assert (
        persisted["measurement_attempts"][0]["stop_reason"] == "runner_restart_recovery"
    )


def test_runner_records_and_refuses_fatal_orchestrator_exit(tmp_path: Path) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["wall_budget_seconds"] = 5
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    orchestrator = tmp_path / "fatal-orchestrator"
    orchestrator.write_text(
        "#!/bin/sh\nexit 78\n",
        encoding="utf-8",
    )
    os.chmod(orchestrator, 0o755)

    report = run_elo_ablation(
        config_path=profile,
        orchestrator=str(orchestrator),
        poll_seconds=0.01,
    )

    assert report["status"] == "failed"
    assert report["outcome"] == FATAL_ORCHESTRATOR_EXIT
    assert report["exit_code"] == 78
    metadata = json.loads(metadata_path.read_text())
    assert metadata["measurement_status"] == "failed"
    assert metadata["measurement_outcome"] == FATAL_ORCHESTRATOR_EXIT
    with pytest.raises(RuntimeError, match="fatal failure"):
        run_elo_ablation(
            config_path=profile,
            orchestrator=str(orchestrator),
            poll_seconds=0.01,
        )


def test_runner_marks_explicit_transient_orchestrator_exit_retryable(
    tmp_path: Path,
) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["wall_budget_seconds"] = 5
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    orchestrator = tmp_path / "transient-orchestrator"
    orchestrator.write_text(
        "#!/bin/sh\nexit 75\n",
        encoding="utf-8",
    )
    os.chmod(orchestrator, 0o755)

    report = run_elo_ablation(
        config_path=profile,
        orchestrator=str(orchestrator),
        poll_seconds=0.01,
    )

    assert report["status"] == "retryable"
    assert report["outcome"] == TRANSIENT_CRASH
    assert report["exit_code"] == 75
    assert "transient failure" in str(report["failure"])


def test_runner_refuses_live_but_allows_stale_coordinator_lock(
    tmp_path: Path,
) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["wall_budget_seconds"] = 5
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    orchestrator = tmp_path / "fatal-orchestrator"
    orchestrator.write_text("#!/bin/sh\nexit 78\n", encoding="utf-8")
    os.chmod(orchestrator, 0o755)
    lock = root / "coordinator.lock"
    lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="live coordinator lock"):
        run_elo_ablation(
            config_path=profile,
            orchestrator=str(orchestrator),
            poll_seconds=0.01,
        )

    lock.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    report = run_elo_ablation(
        config_path=profile,
        orchestrator=str(orchestrator),
        poll_seconds=0.01,
    )
    assert report["outcome"] == FATAL_ORCHESTRATOR_EXIT


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
