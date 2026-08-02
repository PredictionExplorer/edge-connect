from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import scripts.run_elo_ablation_queue as queue_module
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
    handoff: Path
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
    replay_backup: bool = False,
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
        "run_staged_elo_pipeline.py",
        "run_elo_ablation.py",
        "compare_elo_ablation.py",
        "preflight_run_state.py",
        "replay_manifest_backup.py",
    ):
        (scripts / name).write_text(f"# deployed {name}\n", encoding="utf-8")
    queue_unit = tmp_path / "systemd" / "ablation-queue.service"
    finalize_unit = tmp_path / "systemd" / "ablation-finalize.service"
    queue_unit.parent.mkdir()
    queue_unit.write_text("[Service]\nExecStart=queue\n", encoding="utf-8")
    finalize_unit.write_text("[Service]\nExecStart=finalize\n", encoding="utf-8")
    state = tmp_path / "state" / "queue.json"
    backup_service_unit = tmp_path / "systemd" / "ablation-backup.service"
    backup_timer_unit = tmp_path / "systemd" / "ablation-backup.timer"
    backup_service_unit.write_text(
        "[Service]\n"
        f"ExecStart=/test/python {scripts / 'replay_manifest_backup.py'} "
        "backup-active-arm "
        f"--queue-state {state} --interval-seconds 3600 --retain 3 "
        f"--max-total-bytes {20 * 1024 * 1024 * 1024}\n",
        encoding="utf-8",
    )
    backup_timer_unit.write_text(
        f"[Timer]\nOnUnitActiveSec=3600s\nUnit={backup_service_unit.name}\n",
        encoding="utf-8",
    )
    environment = tmp_path / "edgeconnect-ablation.env"
    environment.write_text("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n", encoding="utf-8")
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
        replay_backup_service_unit=(backup_service_unit if replay_backup else None),
        replay_backup_timer_unit=(backup_timer_unit if replay_backup else None),
        replay_backup_interval_seconds=3_600 if replay_backup else None,
    )
    return DeploymentFixture(
        manifest=manifest,
        state=state,
        comparison=comparison,
        handoff=state.with_name("continuity-handoff-request.json"),
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


def _isolated_fatal_report() -> dict[str, object]:
    return {
        **_fatal_report(),
        "failure_domain": "arm",
        "failure_phase": "pre_cutoff",
        "measurement_cutoff_ns": 90,
        "resource_released_ns": 100,
        "teardown_status": "clean",
    }


def _complete_with_teardown_warning_report() -> dict[str, object]:
    return {
        **_complete_report(),
        "status": "complete",
        "completion_status": "complete_with_warning",
        "measurement_cutoff_ns": 90,
        "resource_released_ns": 100,
        "failure_phase": "post_cutoff",
        "teardown_status": "unexpected_exit",
        "teardown": {"status": "unexpected_exit", "failure": "kill escalation"},
        "integrity_status": "valid",
        "integrity": {"status": "valid", "valid": True},
        "failure": "post-cutoff kill escalation",
    }


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


def test_manifest_optionally_pins_replay_backup_policy_and_units(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path, replay_backup=True)

    report = verify_deployment_manifest(
        deployment.manifest,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    manifest = json.loads(deployment.manifest.read_text(encoding="utf-8"))

    assert report["status"] == "verified"
    assert manifest["queue"]["replay_backup"] == {
        "enabled": True,
        "interval_seconds": 3_600.0,
        "retain": 3,
        "max_total_bytes": 20 * 1024 * 1024 * 1024,
    }
    assert {script["name"] for script in manifest["scripts"]} == {
        "run_elo_ablation_queue",
        "run_staged_elo_pipeline",
        "run_elo_ablation",
        "compare_elo_ablation",
        "preflight_run_state",
        "replay_manifest_backup",
    }
    assert {unit["name"] for unit in manifest["units"]} == {
        "queue",
        "finalize",
        "replay_backup_service",
        "replay_backup_timer",
    }
    backup_script = next(
        script
        for script in manifest["scripts"]
        if script["name"] == "replay_manifest_backup"
    )
    Path(backup_script["path"]).write_text(
        "# changed backup script\n", encoding="utf-8"
    )
    with pytest.raises(AblationQueueError, match="digest mismatch"):
        verify_deployment_manifest(
            deployment.manifest,
            current_source_commit=SOURCE_COMMIT,
            source_tree_clean=True,
        )


def test_manifest_rejects_replay_backup_unit_policy_drift(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, replay_backup=True)
    manifest = json.loads(deployment.manifest.read_text(encoding="utf-8"))
    units = {unit["name"]: Path(unit["path"]) for unit in manifest["units"]}
    service = units["replay_backup_service"]
    service.write_text(
        service.read_text(encoding="utf-8").replace(
            "--interval-seconds 3600",
            "--interval-seconds 7200",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AblationQueueError, match="service interval differs"):
        generate_deployment_manifest(
            plan_path=Path(manifest["plan"]["path"]),
            output_path=tmp_path / "deployment" / "drifted.json",
            training_dir=Path(manifest["source"]["training_dir"]),
            queue_unit=units["queue"],
            finalize_unit=units["finalize"],
            replay_backup_service_unit=service,
            replay_backup_timer_unit=units["replay_backup_timer"],
            replay_backup_interval_seconds=3_600,
            environment_file=Path(manifest["environment"]["path"]),
            state_path=Path(manifest["queue"]["state_path"]),
            comparison_output=Path(manifest["queue"]["comparison_output"]),
            source_commit=SOURCE_COMMIT,
        )


def test_manifest_accepts_empty_guard_rings_for_weighted_objective(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    manifest = json.loads(deployment.manifest.read_text(encoding="utf-8"))
    manifest["queue"]["comparison"]["guard_rings"] = []
    deployment.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_deployment_manifest(
        deployment.manifest,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )

    assert report["status"] == "verified"


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
    backup_service = (
        DEPLOY / "edgeconnect-startrain-ablation-replay-backup.service.example"
    ).read_text(encoding="utf-8")
    backup_timer = (
        DEPLOY / "edgeconnect-startrain-ablation-replay-backup.timer.example"
    ).read_text(encoding="utf-8")
    assert "backup-active-arm" in backup_service
    assert "--queue-state @QUEUE_STATE@" in backup_service
    assert "Persistent=true" in backup_timer
    assert "edgeconnect-startrain-@QUEUE_ID@-replay-backup.service" in backup_timer


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
    handoff = json.loads(deployment.handoff.read_text(encoding="utf-8"))
    assert handoff["report"] == "startrain-continuity-handoff-request"
    assert handoff["requested"] is True
    assert handoff["action"] == "request_fallback"
    assert handoff["requested_action"] == "reconcile_training_continuity"
    assert handoff["source"]["queue_status"] == "completed"
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


def test_queue_records_verified_final_replay_backups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path, replay_backup=True)
    backup_roots: list[Path] = []

    def backup(
        run_root: Path,
        *,
        retain: int,
        max_total_bytes: int,
    ) -> tuple[Path, dict[str, object]]:
        assert retain == 3
        assert max_total_bytes == 20 * 1024 * 1024 * 1024
        backup_roots.append(run_root)
        destination = (
            run_root
            / "recovery"
            / "replay-manifest"
            / f"manifest-{len(backup_roots)}.sqlite3"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"verified-backup")
        pointer = {
            "schema_version": 1,
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": "a" * 64,
            "created_ns": len(backup_roots),
        }
        _write_json(destination.parent / "latest.json", pointer)
        return destination, {**pointer, "path": str(destination)}

    monkeypatch.setattr(queue_module, "create_backup_with_evidence", backup)
    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=lambda **_kwargs: _complete_report(),
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )

    assert len(backup_roots) == 2
    assert state["queue_status"] == "completed"
    for arm in _arms(state).values():
        backup_state = arm["replay_backup"]
        assert backup_state["enabled"] is True
        assert backup_state["latest"]["status"] == "completed"
        assert backup_state["latest"]["sha256"] == "a" * 64
        assert len(backup_state["attempts"]) == 1
    comparison = json.loads(deployment.comparison.read_text(encoding="utf-8"))
    assert all(
        arm["replay_backup"]["latest"]["status"] == "completed"
        for arm in comparison["queue"]["arms"]
    )


@pytest.mark.parametrize(
    "backup_error",
    (
        RuntimeError("backup disk unavailable"),
        sqlite3.OperationalError("backup database is locked"),
    ),
)
def test_final_replay_backup_failure_does_not_rewrite_arm_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backup_error: Exception,
) -> None:
    deployment = _deployment(tmp_path, replay_backup=True)

    def fail_backup(
        *_args: object, **_kwargs: object
    ) -> tuple[Path, dict[str, object]]:
        raise backup_error

    monkeypatch.setattr(queue_module, "create_backup_with_evidence", fail_backup)
    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=lambda **_kwargs: _complete_report(),
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )

    assert state["queue_status"] == "completed"
    for arm in _arms(state).values():
        assert arm["status"] == "completed"
        assert arm["replay_backup"]["latest"]["status"] == "failed"
        assert str(backup_error) in arm["replay_backup"]["latest"]["error"]


def test_terminal_arm_reconciles_missing_final_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "terminal-arm"
    _write_json(run_root / "ablation.json", {"replay_restore": {"status": "verified"}})
    destination = run_root / "recovery" / "replay-manifest" / "manifest-final.sqlite3"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"backup")

    monkeypatch.setattr(
        queue_module,
        "create_backup_with_evidence",
        lambda *_args, **_kwargs: (
            destination,
            {
                "bytes": 6,
                "sha256": "c" * 64,
                "created_ns": 1,
            },
        ),
    )
    state = {
        "arms": [
            {
                "status": "completed",
                "run_root": str(run_root),
                "replay_restore": None,
                "replay_backup": None,
            }
        ]
    }

    queue_module._ensure_terminal_replay_backups(
        state,
        policy={
            "enabled": True,
            "interval_seconds": 3_600.0,
            "retain": 3,
            "max_total_bytes": 20 * 1024 * 1024 * 1024,
        },
    )
    arm = state["arms"][0]
    assert arm["replay_restore"] == {"status": "verified"}
    assert arm["replay_backup"]["latest"]["status"] == "completed"


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


def test_replay_backup_history_is_preserved_across_transient_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path, replay_backup=True)
    runner_calls = 0
    backup_calls = 0

    def runner(**_kwargs: object) -> dict[str, object]:
        nonlocal runner_calls
        runner_calls += 1
        return _transient_report() if runner_calls == 1 else _complete_report()

    def backup(run_root: Path, **_kwargs: object) -> tuple[Path, dict[str, object]]:
        nonlocal backup_calls
        backup_calls += 1
        destination = (
            run_root
            / "recovery"
            / "replay-manifest"
            / f"manifest-{backup_calls}.sqlite3"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"backup")
        pointer = {
            "schema_version": 1,
            "path": destination.name,
            "bytes": 6,
            "sha256": "b" * 64,
            "created_ns": backup_calls,
        }
        _write_json(destination.parent / "latest.json", pointer)
        return destination, {**pointer, "path": str(destination)}

    monkeypatch.setattr(queue_module, "create_backup_with_evidence", backup)
    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=runner,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    control = _arms(state)["control"]

    assert state["queue_status"] == "completed"
    assert runner_calls == backup_calls == 3
    assert len(control["replay_backup"]["attempts"]) == 2
    assert control["replay_backup"]["latest"]["status"] == "completed"


def test_budget_completion_with_verified_teardown_warning_is_completed(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    calls = 0

    def runner(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return (
            _complete_with_teardown_warning_report()
            if calls == 1
            else _complete_report()
        )

    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=runner,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    control = _arms(state)["control"]

    assert state["queue_status"] == "completed"
    assert control["status"] == "completed"
    assert control["completion_status"] == "complete_with_warning"
    assert control["measurement_cutoff_ns"] == 90
    assert control["resource_released_ns"] == 100
    assert control["teardown_status"] == "unexpected_exit"
    assert control["integrity_status"] == "valid"


def test_unreleased_resources_block_queue_advancement_and_continuity_handoff(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    report = {
        **_complete_with_teardown_warning_report(),
        "resource_released_ns": None,
        "failure_domain": "process_cleanup",
        "failure": "process group release was not confirmed",
    }

    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=lambda **_kwargs: report,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    arms = _arms(state)
    handoff = json.loads(deployment.handoff.read_text(encoding="utf-8"))

    assert state["queue_status"] == "failed"
    assert arms["control"]["status"] == "failed"
    assert arms["plateau-keep"]["status"] == "pending"
    assert handoff["status"] == "blocked"
    assert handoff["requested"] is False
    assert handoff["reason"] == "resources_not_released"
    assert handoff["unreleased_arms"][0]["treatment"] == "control"


def test_structured_isolated_fatal_arm_is_quarantined_and_queue_continues(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    calls = 0

    def runner(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _isolated_fatal_report() if calls == 1 else _complete_report()

    state = run_ablation_queue(
        deployment.manifest,
        arm_runner=runner,
        current_source_commit=SOURCE_COMMIT,
        source_tree_clean=True,
    )
    arms = _arms(state)
    handoff = json.loads(deployment.handoff.read_text(encoding="utf-8"))

    assert calls == 2
    assert state["queue_status"] == "failed"
    assert arms["control"]["status"] == "quarantined"
    assert arms["control"]["quarantine"]["isolated"] is True
    assert arms["control"]["quarantine"]["failure_domain"] == "arm"
    assert arms["plateau-keep"]["status"] == "completed"
    assert state["finalization"]["status"] == "completed"
    assert handoff["reason"] == "queue_completed_with_quarantined_arms"
    assert handoff["quarantined_arms"][0]["treatment"] == "control"


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
