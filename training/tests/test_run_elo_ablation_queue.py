from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scripts.fork_elo_ablation import fork_elo_ablation
from scripts.prepare_elo_ablation import prepare_elo_ablation
from scripts.run_elo_ablation import (
    BUDGET_COMPLETION,
    FATAL_ORCHESTRATOR_EXIT,
    TRANSIENT_CRASH,
)
from scripts.run_elo_ablation_queue import (
    AblationQueueError,
    QueueBusyError,
    exclusive_execution_lock,
    exclusive_queue_lock,
    finalize_ablation_queue,
    generate_deployment_manifest,
    run_ablation_queue,
    verify_deployment_manifest,
)

CONFIGS = Path(__file__).parents[1] / "configs"
DEPLOY = Path(__file__).parents[1] / "deploy"
SOURCE_COMMIT = "a" * 40


@dataclass(frozen=True)
class DeploymentFixture:
    manifest: Path
    state: Path
    comparison: Path
    source: Path
    installed_profile: Path
    script: Path
    unit: Path
    environment: Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _deployment(
    tmp_path: Path,
    *,
    continue_after_fatal: bool = False,
    max_transient_retries: int = 2,
    warm_start: bool = False,
) -> DeploymentFixture:
    source = tmp_path / "seed"
    _write_json(
        source / "run.json",
        {
            "schema_version": 1,
            "run_id": "shared-run",
            "generation_family": "shared-family",
            "created_ns": 1,
        },
    )
    _write_json(
        source / "learner" / "champion.json",
        {
            "model_identity": "seed-champion",
            "model_step": 100,
            "updated_ns": 10,
        },
    )
    _write_json(
        source / "learner" / "candidate.json",
        {
            "model_identity": "seed-candidate",
            "model_step": 110,
            "updated_ns": 11,
        },
    )
    profiles = tmp_path / "profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=profiles,
        run_root_parent=tmp_path / "runs",
        run_id="shared-run",
        source_run_root=source,
        prefix="queue",
        seed=17,
        wall_budget_hours=1,
        leaf_budget=100,
        guard_floor_elo=-35,
        treatments=("control", "plateau-keep"),
    )
    plan = profiles / "ablation-plan.json"
    for treatment in ("control", "plateau-keep"):
        fork_elo_ablation(
            source_run_root=source,
            plan_path=plan,
            treatment=treatment,
        )
        if warm_start:
            root = tmp_path / "runs" / f"queue-{treatment}-seed17"
            checkpoint = root / "learner" / "recovery" / "warm-start.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"warm-start-checkpoint")
            _write_json(
                root / "learner" / "champion-warm-start.json",
                {
                    "format": "startrain.champion-warm-start",
                    "schema_version": 1,
                    "status": "active",
                    "source_model_identity": "seed-champion",
                    "absolute_model_step": 100,
                    "checkpoint": "recovery/warm-start.pt",
                },
            )

    training = tmp_path / "deployed-training"
    scripts = training / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "run_elo_ablation_queue.py",
        "run_elo_ablation.py",
        "compare_elo_ablation.py",
        "preflight_run_state.py",
    ):
        (scripts / name).write_text(f"# deployed {name}\n", encoding="utf-8")
    queue_unit = tmp_path / "systemd" / "ablation-queue.service"
    finalize_unit = tmp_path / "systemd" / "ablation-finalize.service"
    queue_unit.parent.mkdir()
    queue_unit.write_text("[Service]\nExecStart=queue\n", encoding="utf-8")
    finalize_unit.write_text("[Service]\nExecStart=finalize\n", encoding="utf-8")
    environment = tmp_path / "edgeconnect-ablation.env"
    environment.write_text("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n", encoding="utf-8")
    state = tmp_path / "state" / "queue.json"
    comparison = tmp_path / "reports" / "comparison.json"
    manifest = tmp_path / "deployment" / "manifest.json"
    generate_deployment_manifest(
        plan_path=plan,
        output_path=manifest,
        training_dir=training,
        queue_unit=queue_unit,
        finalize_unit=finalize_unit,
        environment_file=environment,
        state_path=state,
        comparison_output=comparison,
        source_commit=SOURCE_COMMIT,
        poll_seconds=0.01,
        max_transient_retries=max_transient_retries,
        retry_delay_seconds=0,
        continue_after_fatal=continue_after_fatal,
    )
    return DeploymentFixture(
        manifest=manifest,
        state=state,
        comparison=comparison,
        source=source,
        installed_profile=(
            tmp_path / "runs" / "queue-control-seed17" / "profile-elo-ablation.yaml"
        ),
        script=scripts / "run_elo_ablation.py",
        unit=queue_unit,
        environment=environment,
    )


def _report(
    outcome: str,
    *,
    status: str,
    stop_reason: str,
    exit_code: int | None,
    failure: str | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "outcome": outcome,
        "stop_reason": stop_reason,
        "exit_code": exit_code,
        "stopped_ns": 100,
        "failure": failure,
    }


def _complete_report() -> dict[str, object]:
    return _report(
        BUDGET_COMPLETION,
        status="complete",
        stop_reason="wall_budget",
        exit_code=-15,
    )


def _fatal_report() -> dict[str, object]:
    return _report(
        FATAL_ORCHESTRATOR_EXIT,
        status="failed",
        stop_reason="process_exit",
        exit_code=78,
        failure="orchestrator exited before budget with code 78",
    )


def _transient_report() -> dict[str, object]:
    return _report(
        TRANSIENT_CRASH,
        status="retryable",
        stop_reason="process_exit",
        exit_code=-9,
        failure="orchestrator was terminated by signal 9",
    )


def _arms(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_arms = state["arms"]
    assert isinstance(raw_arms, list)
    return {str(arm["treatment"]): arm for arm in raw_arms if isinstance(arm, dict)}


def _comparison_treatments(path: Path) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(treatment["label"]): treatment
        for treatment in report["treatments"]
        if isinstance(treatment, dict)
    }


def test_manifest_pins_revision_and_all_launch_inputs(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)

    report = verify_deployment_manifest(
        deployment.manifest,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    manifest = json.loads(deployment.manifest.read_text(encoding="utf-8"))

    assert report["status"] == "verified"
    assert report["artifact_count"] >= 13
    assert manifest["source"]["commit"] == SOURCE_COMMIT
    assert len(manifest["profiles"]) == 2
    assert {unit["name"] for unit in manifest["units"]} == {"queue", "finalize"}
    assert manifest["seed_snapshot"]["run_identity"]["run_id"] == "shared-run"
    assert manifest["environment"]["sha256"]

    with pytest.raises(AblationQueueError, match="mixed-revision"):
        verify_deployment_manifest(
            deployment.manifest,
            current_source_commit="b" * 40,
            source_tree_clean=True,
        )
    with pytest.raises(AblationQueueError, match="not clean"):
        verify_deployment_manifest(
            deployment.manifest,
            current_source_commit=SOURCE_COMMIT,
            source_tree_clean=False,
        )


def test_manifest_pins_active_champion_warm_start_artifacts(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, warm_start=True)

    assert (
        verify_deployment_manifest(
            deployment.manifest,
            current_source_commit=SOURCE_COMMIT,
            source_tree_clean=True,
        )["status"]
        == "verified"
    )

    checkpoint = (
        tmp_path
        / "runs"
        / "queue-control-seed17"
        / "learner"
        / "recovery"
        / "warm-start.pt"
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(AblationQueueError, match="digest mismatch"):
        verify_deployment_manifest(
            deployment.manifest,
            current_source_commit=SOURCE_COMMIT,
            source_tree_clean=True,
        )


def test_systemd_queue_triggers_finalizer_for_both_outcomes() -> None:
    queue = (DEPLOY / "edgeconnect-startrain-ablation-queue.service.example").read_text(
        encoding="utf-8"
    )
    finalizer = (
        DEPLOY / "edgeconnect-startrain-ablation-finalize.service.example"
    ).read_text(encoding="utf-8")

    assert "OnSuccess=@FINALIZE_UNIT@" in queue
    assert "OnFailure=@FINALIZE_UNIT@" in queue
    assert "run_elo_ablation_queue.py verify" in queue
    assert "RestartPreventExitStatus=2 3" in queue
    assert "run_elo_ablation_queue.py finalize" in finalizer


@pytest.mark.parametrize(
    "artifact_name",
    ["seed", "profile", "script", "unit", "environment"],
)
def test_manifest_refuses_changed_artifact(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    deployment = _deployment(tmp_path)
    artifact = {
        "seed": deployment.source / "run.json",
        "profile": deployment.installed_profile,
        "script": deployment.script,
        "unit": deployment.unit,
        "environment": deployment.environment,
    }[artifact_name]
    with artifact.open("ab") as stream:
        stream.write(b"\nchanged")

    with pytest.raises(AblationQueueError, match="digest mismatch"):
        verify_deployment_manifest(
            deployment.manifest,
            current_source_commit=SOURCE_COMMIT,
            source_tree_clean=True,
        )


def test_manifest_allows_lifecycle_updates_but_freezes_arm_identity(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    metadata_path = deployment.installed_profile.parent / "ablation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "measurement_status": "running",
            "measurement_outcome": None,
            "measurement_attempt_count": 1,
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert (
        verify_deployment_manifest(
            deployment.manifest,
            current_source_commit=SOURCE_COMMIT,
            source_tree_clean=True,
        )["status"]
        == "verified"
    )

    metadata["anchor"]["model_identity"] = "different-anchor"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(AblationQueueError, match="immutable ablation metadata"):
        verify_deployment_manifest(
            deployment.manifest,
            current_source_commit=SOURCE_COMMIT,
            source_tree_clean=True,
        )


def test_queue_runs_exclusively_persists_completion_and_finalizes(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    calls: list[str] = []

    def runner(
        *,
        config_path: Path,
        orchestrator: str,
        poll_seconds: float,
    ) -> dict[str, object]:
        del orchestrator, poll_seconds
        calls.append(config_path.parent.name)
        return _complete_report()

    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=runner,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )

    assert calls == ["queue-control-seed17", "queue-plateau-keep-seed17"]
    assert state["queue_status"] == "completed"
    assert {arm["status"] for arm in _arms(state).values()} == {"completed"}
    assert deployment.comparison.is_file()
    persisted = json.loads(deployment.state.read_text(encoding="utf-8"))
    assert persisted["finalization"]["status"] == "completed"
    assert persisted["finalization"]["comparison_status"] == "incomplete"
    with exclusive_queue_lock(deployment.state):
        with pytest.raises(QueueBusyError, match="another ablation queue"):
            with exclusive_queue_lock(deployment.state):
                pass
    manifest = json.loads(deployment.manifest.read_text(encoding="utf-8"))
    execution_lock = Path(manifest["queue"]["execution_lock_path"])
    with exclusive_execution_lock(execution_lock):
        with pytest.raises(QueueBusyError, match="another ablation execution"):
            with exclusive_execution_lock(execution_lock):
                pass


def test_queue_retries_transient_crash_then_resumes_same_arm(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)
    calls = 0

    def runner(
        *,
        config_path: Path,
        orchestrator: str,
        poll_seconds: float,
    ) -> dict[str, object]:
        nonlocal calls
        del config_path, orchestrator, poll_seconds
        calls += 1
        return _transient_report() if calls == 1 else _complete_report()

    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=runner,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    control = _arms(state)["control"]

    assert state["queue_status"] == "completed"
    assert calls == 3
    assert control["attempts"] == 2
    assert control["transient_failures"] == 1
    assert control["status"] == "completed"


def test_exhausted_transient_retry_budget_stops_queue(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, max_transient_retries=1)
    calls = 0

    def runner(
        *,
        config_path: Path,
        orchestrator: str,
        poll_seconds: float,
    ) -> dict[str, object]:
        nonlocal calls
        del config_path, orchestrator, poll_seconds
        calls += 1
        return _transient_report()

    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=runner,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    arms = _arms(state)

    assert calls == 2
    assert state["queue_status"] == "failed"
    assert arms["control"]["status"] == "failed"
    assert "retry budget exhausted" in arms["control"]["failure"]
    assert arms["plateau-keep"]["status"] == "pending"


def test_signal_interruption_leaves_current_arm_pending_for_restart(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    interrupted = _report(
        TRANSIENT_CRASH,
        status="retryable",
        stop_reason="signal_15",
        exit_code=-15,
        failure="runner interrupted by signal_15",
    )

    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=lambda **_kwargs: interrupted,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )

    assert state["queue_status"] == "pending"
    assert _arms(state)["control"]["status"] == "pending"
    assert _arms(state)["control"]["attempts"] == 1


def test_fatal_arm_stops_default_queue_and_remains_ineligible(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    calls = 0

    def runner(
        *,
        config_path: Path,
        orchestrator: str,
        poll_seconds: float,
    ) -> dict[str, object]:
        nonlocal calls
        del config_path, orchestrator, poll_seconds
        calls += 1
        return _fatal_report()

    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=runner,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    arms = _arms(state)
    treatments = _comparison_treatments(deployment.comparison)

    assert state["queue_status"] == "failed"
    assert calls == 1
    assert arms["control"]["status"] == "failed"
    assert arms["plateau-keep"]["status"] == "pending"
    for treatment in treatments.values():
        codes = {reason["code"] for reason in treatment["ineligibility_reasons"]}
        assert "queue_arm_incomplete" in codes


def test_explicit_continue_policy_runs_after_fatal_but_queue_stays_failed(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path, continue_after_fatal=True)
    calls = 0

    def runner(
        *,
        config_path: Path,
        orchestrator: str,
        poll_seconds: float,
    ) -> dict[str, object]:
        nonlocal calls
        del config_path, orchestrator, poll_seconds
        calls += 1
        return _fatal_report() if calls == 1 else _complete_report()

    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=runner,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    arms = _arms(state)

    assert calls == 2
    assert state["queue_status"] == "failed"
    assert arms["control"]["status"] == "failed"
    assert arms["plateau-keep"]["status"] == "completed"


def test_stale_running_arm_is_resumed_after_queue_process_crash(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)

    def crashing_runner(
        *,
        config_path: Path,
        orchestrator: str,
        poll_seconds: float,
    ) -> dict[str, object]:
        del config_path, orchestrator, poll_seconds
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_ablation_queue(
            deployment.manifest,
            arm_runner=crashing_runner,
            current_source_commit=SOURCE_COMMIT,
            source_tree_clean=True,
        )
    crashed = json.loads(deployment.state.read_text(encoding="utf-8"))
    assert _arms(crashed)["control"]["status"] == "running"

    resumed = run_ablation_queue(
        deployment.manifest,
        arm_runner=lambda **_kwargs: _complete_report(),
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )

    assert resumed["queue_status"] == "completed"
    assert _arms(resumed)["control"]["attempts"] == 2


def test_manifest_failure_still_writes_incomplete_final_comparison(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    deployment.script.write_text("# mixed revision\n", encoding="utf-8")

    with pytest.raises(AblationQueueError, match="digest mismatch"):
        run_ablation_queue(
            deployment.manifest,
            arm_runner=lambda **_kwargs: _complete_report(),
            current_source_commit=SOURCE_COMMIT,
            source_tree_clean=True,
        )

    state = json.loads(deployment.state.read_text(encoding="utf-8"))
    assert state["queue_status"] == "failed"
    assert state["finalization"]["status"] == "completed"
    assert deployment.comparison.is_file()
    assert {arm["status"] for arm in _arms(state).values()} == {"pending"}


def test_finalizer_is_idempotent(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)
    run_ablation_queue(
        deployment.manifest,
        arm_runner=lambda **_kwargs: _complete_report(),
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    before = json.loads(deployment.state.read_text(encoding="utf-8"))

    report = finalize_ablation_queue(deployment.manifest)
    after = json.loads(deployment.state.read_text(encoding="utf-8"))

    assert report["status"] == "incomplete"
    assert after["finalization"]["status"] == "completed"
    assert after["finalization"]["attempts"] == (before["finalization"]["attempts"] + 1)
