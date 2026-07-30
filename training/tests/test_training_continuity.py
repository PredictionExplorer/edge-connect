from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import monitor_run
from startrain import continuity


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _run(tmp_path: Path, name: str, *, created_ns: int) -> tuple[Path, Path]:
    root = (tmp_path / name).resolve()
    root.mkdir()
    run_id = f"{name}-run"
    _write_json(
        root / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": f"{name}-family",
            "created_ns": created_ns,
        },
    )
    profile = root / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "orchestration": {
                    "run_id": run_id,
                    "directories": {"root": str(root)},
                }
            }
        ),
        encoding="utf-8",
    )
    return root, profile


@dataclass
class Fixture:
    manifest: Path
    state_root: Path
    execution_lock: Path
    hardware_report: Path
    primary_root: Path
    primary_profile: Path
    fallback_root: Path
    fallback_profile: Path
    runtime_manifest: Path
    runtime_orchestrator: Path
    queue_state: Path | None
    now_ns: int


def _fixture(
    tmp_path: Path,
    *,
    queue_artifact: bool = False,
    alert_command: list[str] | None = None,
) -> Fixture:
    now_ns = 2_000_000_000_000
    primary_root, primary_profile = _run(tmp_path, "primary", created_ns=now_ns - 10)
    fallback_root, fallback_profile = _run(tmp_path, "fallback", created_ns=now_ns - 20)
    primary_hashes = continuity.workload_fingerprints(primary_profile, primary_root)
    fallback_hashes = continuity.workload_fingerprints(fallback_profile, fallback_root)
    state_root = (tmp_path / "host-state").resolve()
    execution_lock = (tmp_path / "host-gpu.lock").resolve()
    hardware_report = (state_root / "hardware-health.json").resolve()
    queue_state = (tmp_path / "queue-state.json").resolve() if queue_artifact else None
    release_root = (tmp_path / "release").resolve()
    runtime_training = release_root / "training"
    source_files = {}
    for relative in (
        "startrain/orchestration.py",
        "startrain/continuity.py",
        "scripts/reconcile_training_continuity.py",
    ):
        path = runtime_training / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
        source_files[f"training/{relative}"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    runtime_orchestrator = runtime_training / ".venv" / "bin" / "startrain-orchestrate"
    runtime_orchestrator.parent.mkdir(parents=True)
    runtime_orchestrator.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runtime_orchestrator.chmod(0o755)
    primary_unit = (tmp_path / "edgeconnect-primary.service").resolve()
    fallback_unit = (tmp_path / "edgeconnect-fallback.service").resolve()
    primary_unit.write_text("[Service]\nExecStart=/bin/true\n", encoding="utf-8")
    fallback_unit.write_text("[Service]\nExecStart=/bin/true\n", encoding="utf-8")
    runtime_manifest = (release_root / "release-manifest.json").resolve()
    runtime_manifest.write_text(
        json.dumps(
            {
                "report": "edgeconnect-immutable-release",
                "schema_version": 1,
                "source_files": source_files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runtime_sha256 = hashlib.sha256(runtime_manifest.read_bytes()).hexdigest()
    orchestrator_sha256 = hashlib.sha256(runtime_orchestrator.read_bytes()).hexdigest()
    raw: dict[str, Any] = {
        "format": continuity.MANIFEST_FORMAT,
        "schema_version": 1,
        "state_root": str(state_root),
        "locks": {
            "transition": str((state_root / "transition.lock").resolve()),
            "execution": str(execution_lock),
        },
        "hardware": {
            "report_path": str(hardware_report),
            "max_age_seconds": 300,
            "probe_workload": "fallback-lkg",
        },
        "primary": "primary",
        "workloads": [
            {
                "id": "primary",
                "role": "primary",
                "unit": "edgeconnect-primary.service",
                "profile": {
                    "path": str(primary_profile),
                    "sha256": primary_hashes["profile_sha256"],
                },
                "run_root": {
                    "path": str(primary_root),
                    "sha256": primary_hashes["run_root_sha256"],
                },
                "runtime": {
                    "manifest": str(runtime_manifest),
                    "sha256": runtime_sha256,
                    "training_dir": str(runtime_training),
                    "orchestrator": str(runtime_orchestrator),
                    "orchestrator_sha256": orchestrator_sha256,
                    "unit_path": str(primary_unit),
                    "unit_sha256": hashlib.sha256(
                        primary_unit.read_bytes()
                    ).hexdigest(),
                },
            },
            {
                "id": "fallback-lkg",
                "role": "fallback",
                "unit": "edgeconnect-fallback.service",
                "profile": {
                    "path": str(fallback_profile),
                    "sha256": fallback_hashes["profile_sha256"],
                },
                "run_root": {
                    "path": str(fallback_root),
                    "sha256": fallback_hashes["run_root_sha256"],
                },
                "runtime": {
                    "manifest": str(runtime_manifest),
                    "sha256": runtime_sha256,
                    "training_dir": str(runtime_training),
                    "orchestrator": str(runtime_orchestrator),
                    "orchestrator_sha256": orchestrator_sha256,
                    "unit_path": str(fallback_unit),
                    "unit_sha256": hashlib.sha256(
                        fallback_unit.read_bytes()
                    ).hexdigest(),
                },
                "last_known_good": {
                    "verified_ns": now_ns - 100,
                    "priority": 0,
                },
            },
        ],
        "failure_artifacts": (
            [{"type": "queue_state", "path": str(queue_state)}]
            if queue_state is not None
            else []
        ),
        "policy": {
            "automatic_start": True,
            "maximum_start_attempts": 3,
            "start_retry_seconds": 60,
            "operator_hold_path": str((state_root / "operator-hold").resolve()),
        },
        "alerts": {"timeout_seconds": 1},
    }
    if alert_command is not None:
        raw["alerts"]["command"] = alert_command
    manifest = (tmp_path / "continuity.json").resolve()
    _write_json(manifest, raw)
    _write_json(
        hardware_report,
        {
            "schema_version": 1,
            "captured_ns": now_ns - 1,
            "config": str(fallback_profile),
            "healthy": True,
            "gpus": [{"index": 0, "healthy": True, "reasons": []}],
        },
    )
    return Fixture(
        manifest=manifest,
        state_root=state_root,
        execution_lock=execution_lock,
        hardware_report=hardware_report,
        primary_root=primary_root,
        primary_profile=primary_profile,
        fallback_root=fallback_root,
        fallback_profile=fallback_profile,
        runtime_manifest=runtime_manifest,
        runtime_orchestrator=runtime_orchestrator,
        queue_state=queue_state,
        now_ns=now_ns,
    )


class FakeUnitManager:
    def __init__(self, *, activate_on_start: bool = True) -> None:
        self.activate_on_start = activate_on_start
        self.starts: list[str] = []
        self.stops: list[str] = []
        self.statuses: dict[str, continuity.UnitStatus] = {}

    def status(self, unit: str) -> continuity.UnitStatus:
        return self.statuses.get(unit, continuity.UnitStatus(unit, "inactive"))

    def start(self, unit: str) -> None:
        self.starts.append(unit)
        if self.activate_on_start:
            self.statuses[unit] = continuity.UnitStatus(
                unit,
                "active",
                sub_state="running",
                main_pid=123,
            )

    def stop(self, unit: str) -> None:
        self.stops.append(unit)
        self.statuses[unit] = continuity.UnitStatus(
            unit,
            "deactivating",
            sub_state="stop-sigterm",
            main_pid=123,
        )


def _fatal(fixture: Fixture, **overrides: object) -> None:
    payload: dict[str, object] = {
        "format": "startrain.coordinator-fatal",
        "schema_version": 1,
        "timestamp_ns": fixture.now_ns - 2,
        "terminal_reason": "restart_budget_exhausted",
        "failure_class": "transient",
        "reason": "actor restart budget exhausted",
        "coordinator_exit_code": 1,
    }
    payload.update(overrides)
    _write_json(fixture.primary_root / "status" / "fatal.json", payload)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_software_failure_quarantines_metadata_and_starts_verified_lkg(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _fatal(fixture)
    before = _tree_bytes(fixture.primary_root)
    units = FakeUnitManager()

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )

    assert state["phase"] == "active_fallback"
    assert state["active_workload_id"] == "fallback-lkg"
    assert state["selected_lkg_workload_id"] == "fallback-lkg"
    assert state["fallback_attempts"] == 1
    assert units.starts == ["edgeconnect-fallback.service"]
    assert _tree_bytes(fixture.primary_root) == before
    records = list((fixture.state_root / "quarantine").glob("*.json"))
    assert len(records) == 1
    quarantine = json.loads(records[0].read_text(encoding="utf-8"))
    assert quarantine["metadata_only"] is True
    assert quarantine["artifacts_preserved"] is True
    assert quarantine["workload"]["id"] == "primary"
    assert not (fixture.primary_root / "quarantine").exists()
    assert (
        stat.S_IMODE((fixture.state_root / "continuity-state.json").stat().st_mode)
        == 0o644
    )
    assert (
        stat.S_IMODE((fixture.state_root / "continuity-manifest.json").stat().st_mode)
        == 0o644
    )
    assert stat.S_IMODE(fixture.execution_lock.stat().st_mode) == 0o660


def test_reconcile_is_idempotent_after_fallback_start(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _fatal(fixture)
    units = FakeUnitManager()
    continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )
    continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns + 1,
        unit_manager=units,
    )
    state_before = json.loads(
        (fixture.state_root / "continuity-state.json").read_text(encoding="utf-8")
    )
    quarantine_before = list((fixture.state_root / "quarantine").glob("*.json"))
    alerts_before = list((fixture.state_root / "alerts").glob("*.json"))

    continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns + 2,
        unit_manager=units,
    )

    assert units.starts == ["edgeconnect-fallback.service"]
    assert list((fixture.state_root / "quarantine").glob("*.json")) == quarantine_before
    assert list((fixture.state_root / "alerts").glob("*.json")) == alerts_before
    state_after = json.loads(
        (fixture.state_root / "continuity-state.json").read_text(encoding="utf-8")
    )
    assert state_after["revision"] == state_before["revision"] + 1
    assert state_after["last_reconciled_ns"] == fixture.now_ns + 2
    for key in (
        "active_workload_id",
        "desired_workload_id",
        "quarantined_workloads",
        "quarantine_records",
        "handled_failure_ids",
    ):
        assert state_after[key] == state_before[key]


def test_start_retries_are_bounded_before_lkg_fallback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    units = FakeUnitManager(activate_on_start=False)
    interval_ns = 61_000_000_000

    for attempt in range(3):
        state = continuity.reconcile_training_continuity(
            fixture.manifest,
            now_ns=fixture.now_ns + attempt * interval_ns,
            unit_manager=units,
        )
        assert state["phase"] == "starting_primary"

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns + 3 * interval_ns,
        unit_manager=units,
    )

    assert units.starts == [
        "edgeconnect-primary.service",
        "edgeconnect-primary.service",
        "edgeconnect-primary.service",
        "edgeconnect-fallback.service",
    ]
    assert state["phase"] == "starting_fallback"
    assert state["selected_lkg_workload_id"] == "fallback-lkg"
    assert "primary" in state["quarantined_workloads"]


def test_real_hardware_unsafe_blocks_fallback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _fatal(fixture, terminal_reason="fatal_worker_failure")
    _write_json(
        fixture.hardware_report,
        {
            "schema_version": 1,
            "captured_ns": fixture.now_ns - 1,
            "config": str(fixture.fallback_profile),
            "healthy": False,
            "missing_indices": [],
            "gpus": [
                {
                    "index": 0,
                    "healthy": False,
                    "reasons": ["volatile_uncorrectable_ecc"],
                }
            ],
        },
    )
    units = FakeUnitManager()

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )

    assert state["phase"] == "blocked_hardware_unsafe"
    assert state["selected_lkg_workload_id"] == "fallback-lkg"
    assert state["hardware"]["status"] == "unsafe"
    assert units.starts == []
    persisted = json.loads(
        (fixture.state_root / "continuity-state.json").read_text(encoding="utf-8")
    )

    continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns + 1,
        unit_manager=units,
    )

    refreshed = json.loads(
        (fixture.state_root / "continuity-state.json").read_text(encoding="utf-8")
    )
    assert refreshed["revision"] == persisted["revision"] + 1
    assert refreshed["last_reconciled_ns"] == fixture.now_ns + 1
    assert refreshed["phase"] == persisted["phase"]
    assert refreshed["blocked_reason"] == persisted["blocked_reason"]


def test_verified_unsafe_hardware_stops_an_active_registered_workload(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    units = FakeUnitManager()
    started = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )
    assert started["phase"] == "active_primary"

    _write_json(
        fixture.hardware_report,
        {
            "schema_version": 1,
            "captured_ns": fixture.now_ns,
            "config": str(fixture.fallback_profile),
            "healthy": False,
            "missing_indices": [],
            "gpus": [
                {
                    "index": 0,
                    "healthy": False,
                    "reasons": ["volatile_uncorrectable_ecc"],
                }
            ],
        },
    )
    stopped = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns + 1,
        unit_manager=units,
    )

    assert stopped["phase"] == "stopping_hardware_unsafe"
    assert units.stops == ["edgeconnect-primary.service"]
    assert stopped["last_transition"]["kind"] == "hardware_unsafe_stop_requested"


def test_active_unit_is_not_productive_until_learner_and_actor_progress(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    status_root = fixture.primary_root / "status"
    learner_heartbeat = status_root / "learner.heartbeat.json"
    actor_heartbeat = status_root / "actor.heartbeat.json"
    _write_json(
        learner_heartbeat,
        {
            "heartbeat_ns": fixture.now_ns,
            "phase": "initializing",
            "pid": 101,
        },
    )
    _write_json(
        actor_heartbeat,
        {
            "heartbeat_ns": fixture.now_ns,
            "phase": "selfplay",
            "pid": 102,
            "progress": 1,
        },
    )
    _write_json(
        status_root / "coordinator.json",
        {
            "state": "running",
            "workers": {
                "learner": {
                    "role": "learner",
                    "state": "running",
                    "heartbeat": str(learner_heartbeat),
                },
                "actor": {
                    "role": "actor",
                    "state": "running",
                    "heartbeat": str(actor_heartbeat),
                },
            },
        },
    )
    units = FakeUnitManager()
    units.statuses["edgeconnect-primary.service"] = continuity.UnitStatus(
        "edgeconnect-primary.service",
        "active",
        sub_state="running",
        main_pid=100,
    )

    starting = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )
    assert starting["phase"] == "starting_primary"
    assert starting["productive_idle_since_ns"] == fixture.now_ns
    assert starting["execution"]["productivity"]["productive"] is False

    _write_json(
        learner_heartbeat,
        {
            "heartbeat_ns": fixture.now_ns + 1,
            "phase": "training",
            "pid": 101,
            "progress": 10,
        },
    )
    productive = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns + 1,
        unit_manager=units,
    )
    assert productive["phase"] == "active_primary"
    assert productive["productive_idle_since_ns"] is None
    assert productive["execution"]["productivity"]["productive"] is True


def test_unknown_hardware_query_blocks_fallback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _fatal(fixture)
    _write_json(
        fixture.hardware_report,
        {
            "schema_version": 1,
            "captured_ns": fixture.now_ns - 1,
            "config": str(fixture.fallback_profile),
            "healthy": False,
            "query_error": "nvidia-smi timed out",
            "gpus": [],
        },
    )
    units = FakeUnitManager()

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )

    assert state["phase"] == "blocked_hardware_unavailable"
    assert state["hardware"]["status"] == "unavailable"
    assert units.starts == []


def test_two_live_coordinator_locks_block_split_brain(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    for root in (fixture.primary_root, fixture.fallback_root):
        _write_json(
            root / "coordinator.lock",
            {"pid": os.getpid(), "created_ns": fixture.now_ns},
        )
    units = FakeUnitManager()

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )

    assert state["phase"] == "blocked_split_brain"
    assert state["blocked_reason"]["workload_ids"] == [
        "fallback-lkg",
        "primary",
    ]
    assert units.starts == []


def test_quarantined_primary_cannot_displace_selected_fallback(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _fatal(fixture)
    units = FakeUnitManager()
    continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )
    units.statuses["edgeconnect-fallback.service"] = continuity.UnitStatus(
        "edgeconnect-fallback.service",
        "inactive",
    )
    units.statuses["edgeconnect-primary.service"] = continuity.UnitStatus(
        "edgeconnect-primary.service",
        "active",
        main_pid=999,
    )

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns + 1,
        unit_manager=units,
    )

    assert state["phase"] == "blocked_unexpected_active"
    assert state["active_workload_id"] == "primary"
    assert state["desired_workload_id"] == "fallback-lkg"
    assert state["blocked_reason"]["quarantined"] is True


def test_host_execution_lock_is_nonblocking(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    units = FakeUnitManager()

    with continuity.nonblocking_host_lock(
        fixture.execution_lock,
        owner="test workload",
    ):
        state = continuity.reconcile_training_continuity(
            fixture.manifest,
            now_ns=fixture.now_ns,
            unit_manager=units,
        )

    assert state["phase"] == "busy_external"
    assert units.starts == []


def test_host_transition_lock_is_nonblocking(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = continuity.load_continuity_manifest(fixture.manifest)
    units = FakeUnitManager()

    with continuity.nonblocking_host_lock(
        manifest.transition_lock_path,
        owner="test transition",
    ):
        state = continuity.reconcile_training_continuity(
            fixture.manifest,
            now_ns=fixture.now_ns,
            unit_manager=units,
        )

    assert state["phase"] == "transition_busy"
    assert units.starts == []
    assert not (fixture.state_root / "continuity-state.json").exists()


def test_corrupt_manifest_fails_before_service_control(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.manifest.write_text("{partial", encoding="utf-8")
    units = FakeUnitManager()

    with pytest.raises(continuity.ContinuityManifestError):
        continuity.reconcile_training_continuity(
            fixture.manifest,
            now_ns=fixture.now_ns,
            unit_manager=units,
        )

    assert units.starts == []
    assert not (fixture.state_root / "continuity-state.json").exists()


def test_pinned_manifest_cannot_be_replaced_in_place(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    units = FakeUnitManager()
    continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )
    raw = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    raw["policy"]["automatic_start"] = False
    _write_json(fixture.manifest, raw)

    with pytest.raises(continuity.ContinuityStateError):
        continuity.reconcile_training_continuity(
            fixture.manifest,
            now_ns=fixture.now_ns + 1,
            unit_manager=units,
        )


def test_corrupt_host_state_fails_closed_with_durable_alert(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    units = FakeUnitManager()
    continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )
    (fixture.state_root / "continuity-state.json").write_text(
        "{partial",
        encoding="utf-8",
    )

    with pytest.raises(continuity.ContinuityStateError):
        continuity.reconcile_training_continuity(
            fixture.manifest,
            now_ns=fixture.now_ns + 1,
            unit_manager=units,
        )

    alerts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (fixture.state_root / "alerts").glob("*.json")
        if not path.name.endswith(".delivery.json")
    ]
    assert any(alert["kind"] == "continuity_state_blocked" for alert in alerts)
    assert units.starts == ["edgeconnect-primary.service"]


def test_corrupt_lkg_hash_blocks_instead_of_starting_it(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _fatal(fixture)
    fixture.fallback_profile.write_text(
        fixture.fallback_profile.read_text(encoding="utf-8") + "# drift\n",
        encoding="utf-8",
    )
    units = FakeUnitManager()

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )

    assert state["phase"] == "blocked_no_verified_lkg"
    assert "fallback-lkg" in state["blocked_reason"]["rejected"]
    assert units.starts == []


def test_runtime_manifest_drift_blocks_primary_start(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.runtime_manifest.write_text(
        '{"report":"tampered-release","schema_version":1}',
        encoding="utf-8",
    )
    units = FakeUnitManager()

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )

    assert state["phase"] == "blocked_no_verified_lkg"
    assert set(state["quarantined_workloads"]) == {"primary"}
    assert "runtime manifest hash does not match" in str(
        state["last_failure"]["reason"]
    )
    assert units.starts == []


def test_failed_queue_artifact_requests_fallback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, queue_artifact=True)
    assert fixture.queue_state is not None
    _write_json(
        fixture.queue_state,
        {
            "schema_version": 1,
            "queue_status": "failed",
            "queue_error": "fatal arm stopped the queue",
            "updated_ns": fixture.now_ns - 1,
            "arms": [
                {
                    "treatment": "fresh-hard",
                    "status": "quarantined",
                    "last_outcome": "fatal_crash",
                    "last_stopped_ns": fixture.now_ns - 2,
                    "failure": "coordinator exited 78",
                    "failure_domain": "software",
                }
            ],
        },
    )
    units = FakeUnitManager()

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )

    assert state["phase"] == "active_fallback"
    assert state["last_failure"]["source_type"] == "queue_state"
    assert units.starts == ["edgeconnect-fallback.service"]


def test_completed_queue_handoff_resumes_lkg_without_quarantine(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    handoff = (tmp_path / "continuity-handoff-request.json").resolve()
    raw = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    raw["failure_artifacts"] = [{"type": "continuity_handoff", "path": str(handoff)}]
    _write_json(fixture.manifest, raw)
    _write_json(
        handoff,
        {
            "schema_version": 1,
            "report": "startrain-continuity-handoff-request",
            "status": "requested",
            "requested_action": "reconcile_training_continuity",
            "requested_ns": fixture.now_ns - 1,
            "reason": "queue_completed",
            "source": {
                "kind": "elo_ablation_queue",
                "queue_status": "completed",
            },
            "requires_safe_workload": True,
        },
    )
    units = FakeUnitManager()

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )

    assert state["phase"] == "active_fallback"
    assert state["last_failure"] is None
    assert state["last_handoff"]["domain"] == "handoff"
    assert state["quarantine_records"] == []
    assert units.starts == ["edgeconnect-fallback.service"]
    request = json.loads(handoff.read_text(encoding="utf-8"))
    request["requested_ns"] = fixture.now_ns + 1
    _write_json(handoff, request)
    manifest = continuity.load_continuity_manifest(fixture.manifest)
    assert (
        continuity.detect_failure(
            manifest,
            "primary",
            handled_failure_ids={state["last_handoff"]["failure_id"]},
        )
        is None
    )


def test_alert_delivery_failure_never_gates_recovery(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, alert_command=["/does/not/exist"])
    _fatal(fixture)
    units = FakeUnitManager()

    def fail_delivery(*_args, **_kwargs):
        raise OSError("notification transport unavailable")

    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
        alert_runner=fail_delivery,
    )

    assert state["phase"] == "active_fallback"
    assert units.starts == ["edgeconnect-fallback.service"]
    deliveries = list((fixture.state_root / "alerts").glob("*.delivery.json"))
    assert deliveries
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "failed"
        for path in deliveries
    )


def test_locked_wrapper_refuses_live_coordinator(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_json(
        fixture.primary_root / "coordinator.lock",
        {"pid": os.getpid(), "created_ns": fixture.now_ns},
    )
    called = False

    def fake_exec(_file: str, _args, _environment):
        nonlocal called
        called = True

    with pytest.raises(continuity.ContinuitySplitBrainError):
        continuity.run_locked_workload(
            fixture.manifest,
            "fallback-lkg",
            orchestrator=fixture.runtime_orchestrator,
            exec_fn=fake_exec,
            now_ns=fixture.now_ns,
        )

    assert called is False


def test_locked_wrapper_execs_only_state_selected_workload(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    raw["policy"]["automatic_start"] = False
    _write_json(fixture.manifest, raw)
    units = FakeUnitManager()
    state = continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )
    assert state["desired_workload_id"] == "primary"
    captured: dict[str, object] = {}

    def fake_exec(file: str, args: list[str], environment: dict[str, str]) -> str:
        captured.update({"file": file, "args": args, "environment": environment})
        return "exec-called"

    fixture.state_root.chmod(0o555)
    try:
        result = continuity.run_locked_workload(
            fixture.manifest,
            "primary",
            orchestrator=fixture.runtime_orchestrator,
            exec_fn=fake_exec,
            now_ns=fixture.now_ns,
        )
    finally:
        fixture.state_root.chmod(0o755)

    assert result == "exec-called"
    assert captured["args"] == [
        str(fixture.runtime_orchestrator),
        "--config",
        str(fixture.primary_profile),
    ]
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["STARTRAIN_CONTINUITY_WORKLOAD_ID"] == "primary"


def test_monitor_surfaces_structured_continuity_state(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _fatal(fixture)
    units = FakeUnitManager()
    continuity.reconcile_training_continuity(
        fixture.manifest,
        now_ns=fixture.now_ns,
        unit_manager=units,
    )

    status = monitor_run._continuity_status(
        fixture.state_root / "continuity-state.json",
        now_ns=fixture.now_ns + 5_000_000_000,
    )

    assert status["valid"] is True
    assert status["phase"] == "active_fallback"
    assert status["active_workload_id"] == "fallback-lkg"
    assert status["fallback_attempts"] == 1
    assert status["productive_idle_seconds"] is None
    assert status["reconciliation_age_seconds"] == pytest.approx(5.0)
