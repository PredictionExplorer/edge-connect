from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

import scripts.run_elo_ablation as run_module
from scripts.fork_elo_ablation import fork_elo_ablation
from scripts.prepare_elo_ablation import prepare_elo_ablation
from scripts.run_elo_ablation import (
    BUDGET_COMPLETION,
    FATAL_ORCHESTRATOR_EXIT,
    FATAL_WORKER_EXIT_CODE,
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
    monkeypatch.setattr(run_module, "restore_if_corrupt", lambda _root: None)


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


def test_runner_restores_replay_before_state_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    started_ns = time.time_ns() - 2_000_000_000
    metadata.update(
        {
            "wall_budget_seconds": 0.01,
            "measurement_started_ns": started_ns,
            "measurement_status": "retryable",
            "measurement_outcome": TRANSIENT_CRASH,
            "measurement_attempts": [],
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    restored = root / "recovery" / "replay-manifest" / "manifest-restored.sqlite3"
    events: list[str] = []

    def restore(run_root: Path) -> Path:
        assert run_root == root
        events.append("restore")
        return restored

    def preflight(_root: Path, _profile: Path, *, apply: bool):
        assert apply is True
        events.append("preflight")
        return {"status": "ok", "mode": "apply"}

    monkeypatch.setattr(run_module, "restore_if_corrupt", restore)
    monkeypatch.setattr(run_module, "run_state_preflight", preflight)

    report = run_elo_ablation(
        config_path=profile,
        orchestrator=sys.executable,
        poll_seconds=0.01,
    )

    assert report["status"] == "complete"
    assert events == ["restore", "preflight"]
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert persisted["replay_restore"]["latest"]["status"] == "restored"
    assert persisted["replay_restore"]["latest"]["restored_from"] == str(restored)
    assert len(persisted["replay_restore"]["attempts"]) == 1


def test_runner_fails_closed_when_replay_cannot_be_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, profile = _forked_run(tmp_path)
    preflight_called = False

    def failed_restore(_run_root: Path) -> None:
        raise RuntimeError("corrupt replay has no valid backup")

    def preflight(_root: Path, _profile: Path, *, apply: bool):
        nonlocal preflight_called
        del apply
        preflight_called = True
        return {"status": "ok"}

    monkeypatch.setattr(run_module, "restore_if_corrupt", failed_restore)
    monkeypatch.setattr(run_module, "run_state_preflight", preflight)

    with pytest.raises(RuntimeError, match="no valid backup"):
        run_elo_ablation(
            config_path=profile,
            orchestrator=sys.executable,
            poll_seconds=0.01,
        )
    assert preflight_called is False


def test_runner_persists_restore_evidence_before_preflight_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile = _forked_run(tmp_path)
    restored = root / "recovery" / "replay-manifest" / "manifest-restored.sqlite3"
    monkeypatch.setattr(run_module, "restore_if_corrupt", lambda _root: restored)

    def failed_preflight(_root: Path, _profile: Path, *, apply: bool):
        assert apply is True
        raise RuntimeError("state preflight failed")

    monkeypatch.setattr(run_module, "run_state_preflight", failed_preflight)

    with pytest.raises(RuntimeError, match="state preflight failed"):
        run_elo_ablation(
            config_path=profile,
            orchestrator=sys.executable,
            poll_seconds=0.01,
        )
    persisted = json.loads((root / "ablation.json").read_text(encoding="utf-8"))
    assert persisted["replay_restore"]["latest"]["status"] == "restored"
    assert persisted["replay_restore"]["latest"]["restored_from"] == str(restored)


def test_terminal_measurement_is_rejected_before_replay_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "measurement_started_ns": 1,
            "measurement_status": "complete",
            "measurement_outcome": BUDGET_COMPLETION,
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    restore_called = False

    def restore(_root: Path) -> None:
        nonlocal restore_called
        restore_called = True

    monkeypatch.setattr(run_module, "restore_if_corrupt", restore)

    with pytest.raises(RuntimeError, match="already completed"):
        run_elo_ablation(
            config_path=profile,
            orchestrator=sys.executable,
            poll_seconds=0.01,
        )
    assert restore_called is False
    assert "replay_restore" not in json.loads(metadata_path.read_text(encoding="utf-8"))


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
    assert (
        metadata["measurement_stopped_ns"]
        == metadata["measurement_cutoff_ns"]
        <= metadata["resource_released_ns"]
    )
    assert metadata["measurement_stop_reason"] == "wall_budget"
    assert metadata["measurement_exit_code"] is None
    assert metadata["measurement_resource_exit_code"] in (0, -signal.SIGTERM)
    assert metadata["measurement_status"] == "complete"
    assert metadata["measurement_completion_status"] == "complete"
    assert metadata["measurement_outcome"] == BUDGET_COMPLETION
    assert metadata["measurement_teardown_status"] == "clean"
    assert metadata["measurement_teardown"]["status"] == "graceful"
    assert metadata["measurement_teardown"]["term_target"] == "coordinator_pid"
    assert metadata["measurement_teardown"]["kill_sent"] is False
    assert metadata["measurement_attempt_count"] == 1
    assert metadata["measurement_attempts"][0]["outcome"] == BUDGET_COMPLETION
    assert metadata["state_preflight"] == {"status": "ok", "mode": "apply"}
    assert metadata["replay_restore"]["latest"]["status"] == "uninitialized"
    assert signal.getsignal(signal.SIGINT) == previous_sigint
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm


def test_terminate_signals_leader_then_escalates_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 123
        returncode: int | None = None
        waits = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            self.waits += 1
            if self.waits == 1:
                raise run_module.subprocess.TimeoutExpired("orchestrator", timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    leader_signals: list[tuple[int, int]] = []
    group_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        run_module.os,
        "kill",
        lambda pid, sent: leader_signals.append((pid, sent)),
    )
    monkeypatch.setattr(
        run_module.os,
        "killpg",
        lambda pgid, sent: group_signals.append((pgid, sent)),
    )
    monkeypatch.setattr(run_module, "_process_group_exists", lambda _pgid: False)

    teardown = run_module._terminate(
        FakeProcess(),
        terminate_grace_seconds=0.1,
        kill_grace_seconds=0.1,
    )

    assert leader_signals == [(123, signal.SIGTERM)]
    assert group_signals == [(123, signal.SIGKILL)]
    assert teardown["status"] == "forced"
    assert teardown["clean"] is False
    assert teardown["resource_released_ns"] is not None


def test_terminate_reaps_live_descendants_after_leader_already_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitedProcess:
        pid = 456
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float) -> int:
            del timeout
            return self.returncode

    group_signals: list[tuple[int, int]] = []
    probes = iter((True, False))
    monkeypatch.setattr(
        run_module,
        "_process_group_exists",
        lambda _pgid: next(probes),
    )
    monkeypatch.setattr(
        run_module.os,
        "killpg",
        lambda pgid, sent: group_signals.append((pgid, sent)),
    )

    teardown = run_module._terminate(
        ExitedProcess(),
        terminate_grace_seconds=0.0,
        kill_grace_seconds=0.1,
    )

    assert group_signals == [(456, signal.SIGKILL)]
    assert teardown["status"] == "forced"
    assert teardown["clean"] is False
    assert teardown["process_group_released"] is True
    assert teardown["resource_released_ns"] is not None


def test_budget_completion_exposes_post_cutoff_teardown_failure(
    tmp_path: Path,
) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["wall_budget_seconds"] = 1.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    orchestrator = tmp_path / "fatal-on-term-orchestrator"
    orchestrator.write_text(
        """#!/usr/bin/env python3
import signal
import time

def stop(_signal, _frame):
    raise SystemExit(78)

signal.signal(signal.SIGTERM, stop)
while True:
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
    assert report["outcome"] == BUDGET_COMPLETION
    assert report["completion_status"] == "complete_with_warning"
    assert report["exit_code"] is None
    assert report["resource_exit_code"] == FATAL_WORKER_EXIT_CODE
    assert report["measurement_cutoff_ns"] <= report["resource_released_ns"]
    assert report["teardown_status"] == "unexpected_exit"
    assert report["teardown"]["status"] == "unexpected_exit"
    assert report["integrity_status"] == "valid"
    assert report["integrity"]["valid"] is True
    assert report["warnings"]
    metadata = json.loads((root / "ablation.json").read_text(encoding="utf-8"))
    assert metadata["measurement_status"] == "complete"
    assert metadata["measurement_completion_status"] == "complete_with_warning"
    assert metadata["measurement_exit_code"] is None
    assert metadata["measurement_resource_exit_code"] == FATAL_WORKER_EXIT_CODE
    assert metadata["teardown_status"] == "unexpected_exit"
    assert metadata["integrity_status"] == "valid"


def test_post_cutoff_warning_requires_valid_state_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["wall_budget_seconds"] = 1.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def preflight(_root: Path, _profile: Path, *, apply: bool):
        if apply:
            return {"status": "ok", "mode": "apply"}
        raise RuntimeError("replay manifest integrity failed")

    monkeypatch.setattr(run_module, "run_state_preflight", preflight)
    orchestrator = tmp_path / "fatal-on-term-orchestrator"
    orchestrator.write_text(
        """#!/usr/bin/env python3
import signal
import time

def stop(_signal, _frame):
    raise SystemExit(78)

signal.signal(signal.SIGTERM, stop)
while True:
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

    assert report["status"] == "failed"
    assert report["outcome"] == BUDGET_COMPLETION
    assert report["completion_status"] == "failed"
    assert report["failure_phase"] == "post_cutoff"
    assert report["failure_domain"] == "state_integrity"
    assert report["integrity_status"] == "failed"
    assert "replay manifest integrity failed" in str(report["failure"])


def test_pre_cutoff_terminal_failure_cannot_be_reclassified_as_teardown(
    tmp_path: Path,
) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["wall_budget_seconds"] = 1.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    fatal_path = root / "status" / "fatal.json"
    orchestrator = tmp_path / "pre-cutoff-fatal-on-term-orchestrator"
    orchestrator.write_text(
        f"""#!/usr/bin/env python3
import json
from pathlib import Path
import signal
import time

fatal = Path({str(fatal_path)!r})
fatal.parent.mkdir(parents=True, exist_ok=True)
fatal.write_text(json.dumps({{
    "schema_version": 1,
    "timestamp_ns": time.time_ns(),
    "terminal_reason": "fatal_worker_failure",
    "failure_class": "fatal",
    "reason": "failure occurred before the budget boundary",
}}))

def stop(_signal, _frame):
    raise SystemExit(78)

signal.signal(signal.SIGTERM, stop)
while True:
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

    assert report["status"] == "failed"
    assert report["outcome"] == BUDGET_COMPLETION
    assert report["failure_phase"] == "pre_cutoff"
    assert report["failure_domain"] == "orchestrator"
    assert report["integrity_status"] == "failed"
    terminal = report["integrity"]["terminal_failure"]
    assert terminal["phase"] == "pre_cutoff"
    assert terminal["failure"]["terminal_reason"] == "fatal_worker_failure"


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
    restore_history = persisted["replay_restore"]["attempts"]
    assert [attempt["attempt"] for attempt in restore_history] == [1, 2]
    assert persisted["replay_restore"]["latest"] == restore_history[-1]


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
    assert report["failure_domain"] == "run"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["measurement_status"] == "failed"
    assert metadata["measurement_outcome"] == FATAL_ORCHESTRATOR_EXIT
    with pytest.raises(RuntimeError, match="fatal failure"):
        run_elo_ablation(
            config_path=profile,
            orchestrator=str(orchestrator),
            poll_seconds=0.01,
        )


def test_hardware_fatal_is_classified_as_host_global(tmp_path: Path) -> None:
    root, profile = _forked_run(tmp_path)
    metadata_path = root / "ablation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["wall_budget_seconds"] = 5
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    fatal_path = root / "status" / "fatal.json"
    orchestrator = tmp_path / "hardware-fatal-orchestrator"
    orchestrator.write_text(
        f"""#!/usr/bin/env python3
import json
from pathlib import Path
import time

path = Path({str(fatal_path)!r})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({{
    "schema_version": 1,
    "timestamp_ns": time.time_ns(),
    "terminal_reason": "hardware_health_failure",
    "failure_class": "fatal",
    "reason": "uncorrectable ECC",
}}))
raise SystemExit(78)
""",
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
    assert report["failure_domain"] == "host"


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
