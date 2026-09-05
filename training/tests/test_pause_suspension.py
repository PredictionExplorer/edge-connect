from __future__ import annotations

import json
import os
import signal
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

import startrain.orchestration as orchestration
from startrain.orchestration import Coordinator, RunDirectories, build_worker_specs
from startrain.runtime import RunIdentity, atomic_json
from test_orchestration import (
    FakeClock,
    FakeProcess,
    pause_shared_experiment,
    write_pause_request,
)


def suspension_case(tmp_path, monkeypatch):
    experiment = pause_shared_experiment(tmp_path)
    control = experiment.orchestration
    experiment = replace(
        experiment,
        orchestration=replace(
            control,
            gpus=tuple(
                replace(gpu, actor_cohorts=2)
                if gpu.role == "actor" and gpu.gpu_id == control.promotion.gpu_id
                else gpu
                for gpu in control.gpus
            ),
            model_refresh=replace(
                control.model_refresh,
                inference=replace(
                    control.model_refresh.inference,
                    shared_batching=True,
                    max_batch_rows=1024,
                ),
            ),
            promotion=replace(control.promotion, pause_strategy="suspend"),
        ),
    )
    directories = RunDirectories.from_experiment(experiment)
    directories.create()
    specs = build_worker_specs(
        experiment, config_path=tmp_path / "profile.yaml", directories=directories
    )
    clock = FakeClock()
    subject = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        clock=clock,
        sleep=clock.sleep,
    )
    for worker in subject.workers.values():
        worker.process = FakeProcess(exit_immediately=False)
        worker.process_group_id = worker.process.pid
        worker.state = "running"
    target, owner = subject.pause_target, subject.pause_owner
    assert target is not None and owner is not None
    signals = []
    original_signal = orchestration._signal_process

    def signal_process(process, sig):
        signals.append((process.pid, sig))
        original_signal(process, sig)

    monkeypatch.setattr(orchestration, "_signal_process", signal_process)

    def forbidden_signal(*_args):
        pytest.fail("cooperative pause must not send OS process-group signals")

    monkeypatch.setattr(orchestration.os, "killpg", forbidden_signal)
    write_pause_request(
        directories.gpu_pause, token="suspend-lease-token", owner_pid=owner.process.pid
    )
    return SimpleNamespace(
        subject=subject,
        target=target,
        owner=owner,
        signals=signals,
        clock=clock,
        directories=directories,
    )


def heartbeat(case, **changes):
    lease = case.subject.pause_lease
    now = time.time_ns()
    payload = {
        "schema_version": 1,
        "pid": case.target.process.pid,
        "phase": "arena_gpu_pause",
        "lease_token": lease.token,
        "lease_owner_pid": lease.owner_pid,
        "lease_requested_ns": lease.requested_ns,
        "actor_quiescent": True,
        "inference_idle": True,
        "cuda_synchronized": True,
        "parked_cohorts": 2,
        "live_cohorts": 2,
        "progress_ns": now,
        "heartbeat_ns": now,
    }
    payload.update(changes)
    atomic_json(case.target.spec.heartbeat_path, payload)


def acknowledge(case):
    case.subject._reconcile_pause_lease(0.0)
    assert case.target.state == "running"
    waiting = json.loads(case.subject.pause_ack_path.read_text())
    assert waiting["state"] == "waiting"
    assert waiting["target_pid"] == case.target.process.pid
    assert case.signals == []
    heartbeat(case)
    case.subject._reconcile_pause_lease(0.01)
    assert case.target.state == "pause_suspended"
    assert case.target.live
    ack = json.loads(case.subject.pause_ack_path.read_text())
    assert ack["state"] == "ready" and ack["pause_strategy"] == "suspend"
    assert ack["target_suspended"] is True


def release(case, state="released", now=10.0):
    write_pause_request(
        case.directories.gpu_pause,
        token="suspend-lease-token",
        owner_pid=case.owner.process.pid,
        state=state,
    )
    case.subject._reconcile_pause_lease(now)
    assert case.subject.pause_lease.state == "resuming"


def confirm(case, now=10.02):
    token = case.subject.pause_lease.token
    heartbeat(
        case,
        phase="shared_cohorts",
        last_resumed_lease_token=token,
        actor_quiescent=False,
        inference_idle=False,
        cuda_synchronized=False,
        parked_cohorts=0,
    )
    case.subject._reconcile_pause_lease(now)
    assert case.subject.pause_lease is None


def test_suspend_requires_actor_proof_then_resumes_same_pid_with_health_grace(
    tmp_path, monkeypatch
):
    case = suspension_case(tmp_path, monkeypatch)
    process = case.target.process
    acknowledge(case)
    assert not case.subject._monitor_worker(case.target, 5_000.0)
    release(case)
    assert case.target.process is process and case.target.live
    assert case.signals == []
    confirm(case)
    assert case.target.state == "running" and case.target.restart_count == 0
    checked = []
    monkeypatch.setattr(
        case.subject, "_heartbeat_failure", lambda *_args: checked.append(True)
    )
    case.subject._monitor_worker(case.target, 10.03)
    assert not checked
    case.subject._monitor_worker(case.target, case.target.health_grace_until + 0.01)
    assert checked == [True]


@pytest.mark.parametrize(
    "change",
    [
        {"parked_cohorts": 0, "live_cohorts": 0},
        {"parked_cohorts": 1},
        {"inference_idle": False},
        {"cuda_synchronized": False},
        {"lease_token": "wrong-lease-token"},
        {"lease_owner_pid": 999},
        {"progress_ns": 1},
    ],
)
def test_incomplete_or_foreign_quiescence_never_allows_arena_allocation(
    tmp_path, monkeypatch, change
):
    case = suspension_case(tmp_path, monkeypatch)
    case.subject._reconcile_pause_lease(0.0)
    heartbeat(case, **change)
    case.subject._reconcile_pause_lease(0.01)
    assert json.loads(case.subject.pause_ack_path.read_text())["state"] == "waiting"
    assert case.signals == []
    case.subject._reconcile_pause_lease(1.0)
    assert case.subject.pause_failed
    assert json.loads(case.subject.pause_ack_path.read_text())["state"] == "failed"


def test_cancel_before_quiescence_keeps_release_ack_until_actor_confirms(
    tmp_path, monkeypatch
):
    case = suspension_case(tmp_path, monkeypatch)
    case.subject._reconcile_pause_lease(0.0)
    release(case, "cancelled")
    assert json.loads(case.subject.pause_ack_path.read_text())["state"] == "recovered"
    confirm(case)
    assert case.signals == [] and case.target.live


def test_next_request_cannot_overwrite_unobserved_release_ack(tmp_path, monkeypatch):
    case = suspension_case(tmp_path, monkeypatch)
    acknowledge(case)
    release(case)
    write_pause_request(
        case.directories.gpu_pause,
        token="next-suspend-token",
        owner_pid=case.owner.process.pid,
    )
    case.subject._reconcile_pause_lease(10.01)
    assert (
        json.loads(case.subject.pause_ack_path.read_text())["token"]
        == "suspend-lease-token"
    )
    confirm(case)
    case.subject._reconcile_pause_lease(10.03)
    ack = json.loads(case.subject.pause_ack_path.read_text())
    assert ack["token"] == "next-suspend-token" and ack["state"] == "waiting"


def test_missing_request_holds_actor_until_owner_is_reaped(tmp_path, monkeypatch):
    case = suspension_case(tmp_path, monkeypatch)
    acknowledge(case)
    process = case.target.process
    case.owner.process.exit_on_terminate = False
    case.directories.gpu_pause.unlink()
    case.subject._reconcile_pause_lease(1.0)
    assert case.target.state == "pause_suspended"
    assert json.loads(case.subject.pause_ack_path.read_text())["state"] == "ready"
    assert case.signals == [(case.owner.process.pid, signal.SIGTERM)]
    case.owner.process.returncode = -15
    case.subject._monitor_worker(case.owner, 1.01)
    assert case.subject.pause_lease.state == "resuming"
    assert json.loads(case.subject.pause_ack_path.read_text())["state"] == "recovered"
    confirm(case, now=1.02)
    assert case.target.process is process and process.terminate_calls == 0


@pytest.mark.parametrize("shutdown", ["stop", "drain", "quiescence_timeout"])
def test_shutdown_uses_real_stop_without_os_freezing(tmp_path, monkeypatch, shutdown):
    case = suspension_case(tmp_path, monkeypatch)
    process = case.target.process
    if shutdown == "quiescence_timeout":
        case.subject._reconcile_pause_lease(0.0)
        case.subject._reconcile_pause_lease(1.0)
        assert case.subject.pause_failed
        case.subject._stop_all()
    else:
        acknowledge(case)
        if shutdown == "stop":
            case.subject.stopping = True
            case.subject._stop_all()
        else:
            case.subject._begin_drain(1.0)
    assert [sig for pid, sig in case.signals if pid == process.pid] == [signal.SIGTERM]
    assert process.returncode == -signal.SIGTERM and process.kill_calls == 0
    if shutdown != "drain":
        assert (
            json.loads(case.subject.pause_ack_path.read_text())["state"] == "draining"
        )


def test_final_drain_does_not_treat_actor_shutdown_as_quiescence_escape(
    tmp_path, monkeypatch
):
    case = suspension_case(tmp_path, monkeypatch)
    acknowledge(case)
    case.target.process.exit_on_terminate = False
    case.subject._begin_drain(0.02)
    heartbeat(case, actor_quiescent=False)
    assert case.target.state == "pause_terminating"
    case.subject._reconcile_pause_lease(0.025)
    assert case.owner.live and case.owner.process.terminate_calls == 0
    assert not case.subject.pause_failed


def test_status_exposes_paused_live_pid_and_bounded_resume_grace(tmp_path, monkeypatch):
    case = suspension_case(tmp_path, monkeypatch)
    acknowledge(case)
    pid = case.target.process.pid
    case.subject._write_status()
    status = json.loads(case.subject.status_path.read_text())
    worker = status["workers"][case.target.spec.name]
    assert worker["state"] == "paused" and worker["pid"] == pid
    assert status["pause_sharing"]["target_suspended"] is True
    release(case)
    case.clock.value = 10.0
    case.subject._write_status()
    status = json.loads(case.subject.status_path.read_text())
    worker = status["workers"][case.target.spec.name]
    assert worker["state"] == "running" and worker["pid"] == pid
    remaining = worker["heartbeat_grace_until_ns"] - status["timestamp_ns"]
    assert (
        0
        < remaining
        <= int(
            case.subject.experiment.orchestration.shutdown.stale_heartbeat_seconds
            * 1_000_000_000
        )
    )


def test_suspend_request_path_is_only_forwarded_to_shared_actor_target(
    tmp_path, monkeypatch
):
    case = suspension_case(tmp_path, monkeypatch)
    for worker in case.subject.workers.values():
        if worker.spec.role == "actor":
            assert ("--gpu-pause" in worker.spec.command) == (worker is case.target)
    assert "--gpu-pause" in case.owner.spec.command


@pytest.mark.parametrize("phase", ["waiting", "ready", "resuming"])
def test_unexpected_cooperative_actor_exit_fails_closed(tmp_path, monkeypatch, phase):
    case = suspension_case(tmp_path, monkeypatch)
    if phase == "waiting":
        case.subject._reconcile_pause_lease(0.0)
    else:
        acknowledge(case)
        if phase == "resuming":
            release(case)
    case.target.process.returncode = 7
    assert case.subject._monitor_worker(case.target, 10.01)
    assert case.subject.pause_failed
    assert case.subject.pause_lease.target_reaped
    assert json.loads(case.subject.pause_ack_path.read_text())["state"] == "failed"


def test_final_drain_reaps_actor_already_resuming_without_leaving_lease_stuck(
    tmp_path, monkeypatch
):
    case = suspension_case(tmp_path, monkeypatch)
    acknowledge(case)
    release(case)
    case.subject._begin_drain(10.01)
    case.subject._monitor_worker(case.target, 10.02)
    assert case.target.process is None
    assert case.target.state == "drained"
    assert case.subject.pause_lease is None
    assert not case.subject.pause_failed


def test_real_actor_gate_and_coordinator_preserve_release_before_next_token(
    tmp_path, monkeypatch
):
    from startrain.actor_pause import ActorPauseGate

    case = suspension_case(tmp_path, monkeypatch)
    case.target.process.pid = os.getpid()
    control = case.subject.experiment.orchestration
    case.subject.experiment = replace(
        case.subject.experiment,
        orchestration=replace(
            control,
            shutdown=replace(
                control.shutdown,
                stale_heartbeat_seconds=3.0,
                stall_timeout_seconds=10.0,
            ),
            promotion=replace(
                control.promotion,
                pause_ready_timeout_seconds=2.0,
                pause_release_timeout_seconds=2.0,
            ),
        ),
    )
    done = threading.Event()
    identifiers = ["cohort-0", "cohort-1"]
    gate = ActorPauseGate(
        request_path=case.directories.gpu_pause,
        gpu_id=7,
        worker_name=case.target.spec.name,
        run_identity=RunIdentity(
            case.directories.run_identity, "run-test", "family-test", 1
        ),
        cohort_ids=identifiers,
        stop_requested=done.is_set,
        inference_idle=lambda: True,
        synchronize=lambda: None,
        stale_seconds=3.0,
    )
    moves = {identity: 0 for identity in identifiers}

    def producer(identity):
        while not done.is_set():
            gate.checkpoint(identity)
            moves[identity] += 1
            time.sleep(0.002)

    threads = [
        threading.Thread(target=producer, args=(identity,)) for identity in identifiers
    ]
    for thread in threads:
        thread.start()
    try:
        case.subject._reconcile_pause_lease(0.0)
        deadline = time.monotonic() + 2
        while case.subject.pause_lease.state != "ready":
            details = gate.poll()
            if details is not None:
                heartbeat(case, **details)
            case.subject._reconcile_pause_lease(0.01)
            assert time.monotonic() < deadline
            time.sleep(0.002)
        parked_moves = dict(moves)
        time.sleep(0.02)
        assert moves == parked_moves
        release(case)
        write_pause_request(
            case.directories.gpu_pause,
            token="next-suspend-token",
            owner_pid=case.owner.process.pid,
        )
        deadline = time.monotonic() + 2
        while case.subject.pause_lease is not None:
            assert (
                json.loads(case.subject.pause_ack_path.read_text())["token"]
                == "suspend-lease-token"
            )
            details = gate.poll()
            if details is not None:
                heartbeat(case, **details)
            case.subject._reconcile_pause_lease(10.01)
            assert time.monotonic() < deadline
            time.sleep(0.002)
        assert all(moves[identity] > parked_moves[identity] for identity in identifiers)
        case.subject._reconcile_pause_lease(10.02)
        assert (
            json.loads(case.subject.pause_ack_path.read_text())["token"]
            == "next-suspend-token"
        )
        assert case.target.process.pid == os.getpid()
        assert case.signals == []
    finally:
        done.set()
        gate.close()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()
