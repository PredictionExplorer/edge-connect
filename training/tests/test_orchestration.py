from __future__ import annotations

import json
import signal
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import pytest

import startrain.orchestration as orchestration_module
from startrain.actor import ActorSupervisor, RingMixtureScheduler
from startrain.checkpoint import (
    ExponentialMovingAverage,
    latest_checkpoint,
    load_ema_checkpoint,
    load_model_manifest,
)
from startrain.config import (
    CurriculumStage,
    GameConfig,
    ConfigError,
    DistributedConfig,
    GPUWorkerConfig,
    RestartPolicyConfig,
    RingMixtureConfig,
    RingWeightStage,
    RunDirectoryConfig,
    SchedulerConfig,
    ShutdownConfig,
    load_config,
)
from startrain.learner import ImmutableModelPublisher
from startrain.model import GraphResTNet, ModelConfig
from startrain.optim import OptimizerConfig, build_optimizer
from startrain.orchestration import (
    Coordinator,
    RunDirectories,
    build_worker_specs,
    ensure_autonomous_provenance,
    gpu_pause_ack_path,
    validate_autonomous_run_root,
)
from startrain.runtime import (
    RunIdentity,
    atomic_json,
    load_or_create_run_identity,
)
from startrain.training import build_scheduler


CONFIGS = Path(__file__).parents[1] / "configs"
DEPLOY = Path(__file__).parents[1] / "deploy"


def test_finite_and_continuous_systemd_restart_policies_are_distinct() -> None:
    finite = (DEPLOY / "edgeconnect-startrain.service.example").read_text()
    continuous = (
        DEPLOY / "edgeconnect-startrain-continuous.service.example"
    ).read_text()
    assert "Restart=on-failure" in finite
    assert "Restart=always" not in finite
    assert "Restart=always" in continuous
    assert "validate_continuous_profile.py" in continuous
    assert "WatchdogSignal=SIGTERM" in finite
    assert "WatchdogSignal=SIGTERM" in continuous
    report_service = (
        DEPLOY / "edgeconnect-startrain-report.service.example"
    ).read_text()
    report_timer = (DEPLOY / "edgeconnect-startrain-report.timer.example").read_text()
    assert "strength_efficiency_report.py" in report_service
    assert "@PROVISIONED_GPUS@" in report_service
    assert "OnUnitActiveSec=15min" in report_timer
    assert "edgeconnect-startrain-@RUN_ID@-report.service" in report_timer


def test_graceful_stop_signals_only_worker_leader_before_group_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePopen:
        pid = 123

    leader_signals: list[tuple[int, int]] = []
    group_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(orchestration_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        orchestration_module.os,
        "kill",
        lambda pid, sent: leader_signals.append((pid, sent)),
    )
    monkeypatch.setattr(orchestration_module.os, "getpgid", lambda pid: pid + 1)
    monkeypatch.setattr(
        orchestration_module.os,
        "killpg",
        lambda pgid, sent: group_signals.append((pgid, sent)),
    )

    process = FakePopen()
    orchestration_module._signal_process(process, signal.SIGTERM)
    orchestration_module._signal_process(process, signal.SIGKILL)

    assert leader_signals == [(123, signal.SIGTERM)]
    assert group_signals == [(124, signal.SIGKILL)]


def test_h100_layouts_assign_one_learner_and_every_actor_gpu() -> None:
    eight = load_config(CONFIGS / "h100-8gpu.yaml")
    four = load_config(CONFIGS / "h100-4gpu.yaml")
    optimized = load_config(CONFIGS / "h100-8gpu-optimized.yaml")
    assert [gpu.gpu_id for gpu in eight.orchestration.learner_gpus] == [0]
    assert [gpu.gpu_id for gpu in eight.orchestration.actor_gpus] == list(range(1, 7))
    assert [gpu.gpu_id for gpu in four.orchestration.learner_gpus] == [0]
    assert [gpu.gpu_id for gpu in four.orchestration.actor_gpus] == [1, 2]
    assert [gpu.gpu_id for gpu in optimized.orchestration.actor_gpus] == list(
        range(1, 8)
    )
    assert optimized.orchestration.promotion.gpu_id == 7

    directories = RunDirectories.from_experiment(four)
    specs = build_worker_specs(
        four,
        config_path=CONFIGS / "h100-4gpu.yaml",
        directories=directories,
        python_executable="/test/python",
        base_environment={"PATH": "/test"},
    )
    assert len(specs) == 4
    assert specs[0].role == "learner"
    assert specs[0].environment["CUDA_VISIBLE_DEVICES"] == "0"
    for spec, gpu_id in zip(specs[1:3], (1, 2), strict=True):
        assert spec.role == "actor"
        assert spec.gpu_ids == (gpu_id,)
        assert spec.environment["CUDA_VISIBLE_DEVICES"] == str(gpu_id)
        assert spec.environment["RAYON_NUM_THREADS"] == str(spec.cpu_threads)
        assert spec.environment["OMP_NUM_THREADS"] == str(spec.cpu_threads)
    assert specs[-1].role == "arena"
    with pytest.raises(ConfigError, match="overlap"):
        replace(
            four.orchestration,
            promotion=replace(
                four.orchestration.promotion,
                gpu_id=0,
                pause_sharing_mode=False,
            ),
        )
    shared = replace(
        four.orchestration,
        promotion=replace(
            four.orchestration.promotion,
            gpu_id=0,
            pause_sharing_mode=True,
        ),
    )
    assert shared.promotion.pause_sharing_mode is True
    learner_shared_experiment = replace(four, orchestration=shared)
    learner_shared_specs = build_worker_specs(
        learner_shared_experiment,
        config_path=CONFIGS / "h100-4gpu.yaml",
        directories=RunDirectories.from_experiment(learner_shared_experiment),
        python_executable="/test/python",
        base_environment={},
    )
    assert "--gpu-pause" in learner_shared_specs[0].command
    actor_shared = replace(
        four.orchestration,
        promotion=replace(
            four.orchestration.promotion,
            gpu_id=1,
            pause_sharing_mode=True,
        ),
    )
    assert actor_shared.promotion.gpu_id == 1
    with pytest.raises(ConfigError, match="overlap requires"):
        replace(
            four.orchestration,
            promotion=replace(
                four.orchestration.promotion,
                gpu_id=1,
                pause_sharing_mode=False,
            ),
        )
    with pytest.raises(ConfigError, match="requires exactly one"):
        replace(
            four.orchestration,
            promotion=replace(
                four.orchestration.promotion,
                gpu_id=9,
                pause_sharing_mode=True,
            ),
        )

    optimized_directories = RunDirectories.from_experiment(optimized)
    optimized_specs = build_worker_specs(
        optimized,
        config_path=CONFIGS / "h100-8gpu-optimized.yaml",
        directories=optimized_directories,
        python_executable="/test/python",
        base_environment={},
    )
    learner, *_, actor_seven, arena = optimized_specs
    assert learner.role == "learner"
    assert "--gpu-pause" not in learner.command
    assert actor_seven.name == "actor-gpu-7"
    assert actor_seven.environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert arena.role == "arena"
    assert arena.environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert "--gpu-pause" in arena.command


def test_autonomous_run_provenance_rejects_imports_and_profile_drift(
    tmp_path,
) -> None:
    configured = load_config(CONFIGS / "h100-8gpu-autonomous.yaml")
    configured = replace(
        configured,
        orchestration=replace(
            configured.orchestration,
            run_id="autonomous-test",
            directories=replace(
                configured.orchestration.directories,
                root=str(tmp_path / "autonomous"),
            ),
        ),
    )
    directories = RunDirectories.from_experiment(configured)
    directories.create()
    (directories.learner / "candidate.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="imported artifacts"):
        validate_autonomous_run_root(configured, directories)
    (directories.learner / "candidate.json").unlink()

    validate_autonomous_run_root(configured, directories)
    identity = load_or_create_run_identity(
        directories.run_identity,
        requested_run_id="autonomous-test",
    )
    ensure_autonomous_provenance(configured, directories, identity)
    payload = json.loads(directories.autonomous_provenance.read_text(encoding="utf-8"))
    assert payload["mode"] == "random-init-selfplay-only"
    assert payload["external_weights"] is False
    assert payload["external_replay"] is False
    assert payload["external_positions"] is False
    ensure_autonomous_provenance(configured, directories, identity)

    drifted = replace(configured, train=replace(configured.train, seed=99))
    with pytest.raises(ValueError, match="frozen run profile"):
        ensure_autonomous_provenance(drifted, directories, identity)


def test_actor_lanes_expand_worker_specs_with_distinct_identity_and_affinity(
    tmp_path,
) -> None:
    experiment = load_config(CONFIGS / "h100-8gpu-optimized.yaml")
    gpus = tuple(
        replace(
            gpu,
            actor_lanes=2,
            cpu_affinity="0-3,8",
        )
        if gpu.gpu_id == 1
        else gpu
        for gpu in experiment.orchestration.gpus
    )
    experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            gpus=gpus,
            directories=RunDirectoryConfig(root=str(tmp_path / "lanes")),
        ),
    )
    directories = RunDirectories.from_experiment(experiment)

    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-8gpu-optimized.yaml",
        directories=directories,
        python_executable="/test/python",
        base_environment={},
    )
    lanes = [spec for spec in specs if spec.name.startswith("actor-gpu-1-lane-")]

    assert [spec.name for spec in lanes] == [
        "actor-gpu-1-lane-0",
        "actor-gpu-1-lane-1",
    ]
    assert [spec.command[spec.command.index("--lane-id") + 1] for spec in lanes] == [
        "0",
        "1",
    ]
    assert all(spec.cpu_affinity == (0, 1, 2, 3, 8) for spec in lanes)
    assert all(
        spec.environment["STARTRAIN_CPU_AFFINITY"] == "0,1,2,3,8" for spec in lanes
    )
    shared = next(gpu for gpu in gpus if gpu.gpu_id == 7)
    with pytest.raises(ConfigError, match="exactly one actor lane"):
        replace(
            experiment.orchestration,
            gpus=tuple(
                replace(gpu, actor_lanes=2) if gpu is shared else gpu for gpu in gpus
            ),
        )


def test_ring_scheduler_curriculum_then_favors_deficits() -> None:
    experiment = load_config(CONFIGS / "h100-8gpu.yaml")
    scheduler = RingMixtureScheduler(experiment.orchestration.ring_mixture, seed=11)
    empty = {ring: 0 for ring in (4, 6, 8, 10)}
    assert {scheduler.choose(empty) for _ in range(100)} == {4}

    mature = {ring: 400_000 for ring in (4, 6, 8, 10)}
    mature[10] = 0
    selections = [scheduler.choose(mature) for _ in range(2_000)]
    assert set(selections) == {4, 6, 8, 10}
    assert selections.count(10) > selections.count(4) * 2

    weighted = RingMixtureScheduler(
        RingMixtureConfig(
            step_weights=(RingWeightStage(1_000_000, (0.1, 0.1, 0.1, 0.7)),)
        ),
        seed=11,
    )
    post_million = [
        weighted.choose(mature, learner_step=1_000_000) for _ in range(20_000)
    ]
    ring_ten_fraction = post_million.count(10) / len(post_million)
    assert 0.68 <= ring_ten_fraction <= 0.72


def test_ring_mixture_stage_selection_uses_aggregate_sample_boundaries() -> None:
    mixture = RingMixtureConfig(
        curriculum=(
            CurriculumStage(until_samples=100, rings=(4,)),
            CurriculumStage(until_samples=500, rings=(4, 6)),
        )
    )
    assert mixture.active_rings(0) == (4,)
    assert mixture.active_rings(99) == (4,)
    assert mixture.active_rings(100) == (4, 6)
    assert mixture.active_rings(499) == (4, 6)
    assert mixture.active_rings(500) == (4, 6, 8, 10)


def test_actor_scheduling_step_uses_fresh_learner_heartbeat_with_fallback(
    tmp_path,
) -> None:
    actor = object.__new__(ActorSupervisor)
    actor.learner_heartbeat_path = tmp_path / "learner.heartbeat.json"
    actor.experiment = SimpleNamespace(
        orchestration=SimpleNamespace(
            shutdown=SimpleNamespace(stale_heartbeat_seconds=60.0)
        )
    )
    actor.learner_heartbeat_path.write_text(
        json.dumps({"step": 1_000_000}),
        encoding="utf-8",
    )
    assert actor._read_learner_scheduling_step(fallback_step=990_000) == (
        1_000_000,
        "learner_heartbeat",
    )
    actor.learner_heartbeat_path.write_text("{broken", encoding="utf-8")
    assert actor._read_learner_scheduling_step(fallback_step=990_000) == (
        990_000,
        "candidate_manifest",
    )


def test_explicit_ddp_builds_one_torchrun_job_and_partitions_batches(
    tmp_path,
) -> None:
    experiment = load_config(CONFIGS / "h100-4gpu.yaml")
    orchestration = replace(
        experiment.orchestration,
        gpus=(
            GPUWorkerConfig(0, "learner", 8),
            GPUWorkerConfig(2, "learner", 8),
            GPUWorkerConfig(5, "actor", 4, 32),
        ),
        distributed=DistributedConfig(enabled=True, backend="nccl"),
        directories=RunDirectoryConfig(root=str(tmp_path / "ddp-run")),
    )
    experiment = replace(experiment, orchestration=orchestration)
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-4gpu.yaml",
        directories=directories,
        python_executable="/test/python",
        base_environment={},
    )
    assert len(specs) == 3
    learner = specs[0]
    assert learner.gpu_ids == (0, 2)
    assert learner.environment["CUDA_VISIBLE_DEVICES"] == "0,2"
    assert "torch.distributed.run" in learner.command
    assert "--nproc-per-node=2" in learner.command
    assert learner.command.count("--distributed-backend") == 1
    with pytest.raises(ConfigError, match="shared CPU affinity"):
        replace(
            orchestration,
            gpus=(
                GPUWorkerConfig(0, "learner", 8, cpu_affinity="0-7"),
                GPUWorkerConfig(2, "learner", 8),
                GPUWorkerConfig(5, "actor", 4, 32),
            ),
        )


class FakeProcess:
    next_pid = 10_000

    def __init__(
        self,
        *,
        exit_immediately: bool,
        exit_on_terminate: bool = True,
        exit_on_kill: bool = True,
    ) -> None:
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode: int | None = None
        self.exit_immediately = exit_immediately
        self.exit_on_terminate = exit_on_terminate
        self.exit_on_kill = exit_on_kill
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        if self.returncode is None and self.exit_immediately:
            self.returncode = 7
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.exit_on_terminate:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        if self.exit_on_kill:
            self.returncode = -9


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_coordinator_applies_configured_cpu_affinity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = load_config(CONFIGS / "h100-4gpu.yaml")
    gpus = tuple(
        replace(gpu, cpu_affinity="4-6") if gpu.gpu_id == 1 else gpu
        for gpu in experiment.orchestration.gpus
    )
    experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            gpus=gpus,
            directories=RunDirectoryConfig(root=str(tmp_path / "affinity")),
        ),
    )
    directories = RunDirectories.from_experiment(experiment)
    directories.create()
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-4gpu.yaml",
        directories=directories,
        base_environment={},
    )
    process = FakeProcess(exit_immediately=False)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        orchestration_module.shutil,
        "which",
        lambda executable, **_kwargs: f"/usr/bin/{executable}",
    )

    def process_factory(command, **_kwargs):
        commands.append(command)
        return process

    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
    )
    worker = coordinator.workers["actor-gpu-1"]

    coordinator._start(worker)

    assert commands == [
        [
            "/usr/bin/taskset",
            "--cpu-list",
            "4,5,6",
            *worker.spec.command,
        ]
    ]
    assert worker.live
    worker.process = None
    if worker.log_stream is not None:
        worker.log_stream.close()
        worker.log_stream = None


def pause_shared_experiment(tmp_path: Path):
    experiment = load_config(CONFIGS / "h100-8gpu-optimized.yaml")
    return replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            directories=RunDirectoryConfig(root=str(tmp_path / "pause-run")),
            restart=RestartPolicyConfig(
                max_restarts=2,
                initial_backoff_seconds=0.01,
                maximum_backoff_seconds=0.02,
                stable_reset_seconds=100.0,
            ),
            shutdown=ShutdownConfig(
                monitor_interval_seconds=0.01,
                heartbeat_interval_seconds=0.02,
                stale_heartbeat_seconds=0.1,
                stall_timeout_seconds=1.0,
                terminate_grace_seconds=0.02,
                kill_grace_seconds=0.01,
            ),
            promotion=replace(
                experiment.orchestration.promotion,
                pause_ready_timeout_seconds=0.1,
                pause_release_timeout_seconds=0.1,
            ),
        ),
    )


def write_pause_request(
    path: Path,
    *,
    token: str,
    owner_pid: int,
    state: str = "requested",
    requested_ns: int | None = None,
    heartbeat_ns: int | None = None,
    gpu_id: int = 7,
) -> None:
    requested = requested_ns or time.time_ns()
    atomic_json(
        path,
        {
            "schema_version": 1,
            "protocol": "coordinator-pause-v1",
            "token": token,
            "pid": owner_pid,
            "gpu_id": gpu_id,
            "candidate_identity": "candidate-test",
            "state": state,
            "requested_ns": requested,
            "heartbeat_ns": heartbeat_ns or time.time_ns(),
        },
    )


def worker_name(command: list[str]) -> str:
    if "train" in command:
        return "learner"
    if "actor" in command:
        gpu_index = command.index("--gpu-id") + 1
        return f"actor-gpu-{command[gpu_index]}"
    if "promote" in command:
        return "arena-promotion"
    raise AssertionError(f"unknown worker command: {command}")


def coordinator_events(directories: RunDirectories) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (directories.metrics / "coordinator.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_coordinator_restarts_failed_actor_without_duplicate(tmp_path) -> None:
    experiment = load_config(CONFIGS / "h100-4gpu.yaml")
    orchestration = replace(
        experiment.orchestration,
        directories=RunDirectoryConfig(root=str(tmp_path / "run")),
        restart=RestartPolicyConfig(
            max_restarts=2,
            initial_backoff_seconds=0.01,
            maximum_backoff_seconds=0.02,
            stable_reset_seconds=100.0,
        ),
        shutdown=ShutdownConfig(
            monitor_interval_seconds=0.01,
            heartbeat_interval_seconds=1.0,
            stale_heartbeat_seconds=100.0,
            terminate_grace_seconds=0.01,
            kill_grace_seconds=0.01,
        ),
    )
    experiment = replace(experiment, orchestration=orchestration)
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-4gpu.yaml",
        directories=directories,
        base_environment={},
    )
    launches: list[dict[str, Any]] = []
    actor_one_launches = 0

    def process_factory(command: list[str], **options: Any) -> FakeProcess:
        nonlocal actor_one_launches
        gpu_one = (
            "--gpu-id" in command and command[command.index("--gpu-id") + 1] == "1"
        )
        immediate = gpu_one and actor_one_launches == 0
        if gpu_one:
            actor_one_launches += 1
        launches.append({"command": command, **options})
        return FakeProcess(exit_immediately=immediate)

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=lambda: False, max_monitor_cycles=3) == 0
    assert actor_one_launches == 2
    assert len(launches) == len(specs) + 1
    events = [
        json.loads(line)
        for line in (directories.metrics / "coordinator.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert (
        sum(
            event["event"] == "worker_started" and event["worker"] == "actor-gpu-1"
            for event in events
        )
        == 2
    )
    assert any(
        event["event"] == "worker_exited"
        and event["worker"] == "actor-gpu-1"
        and event["restart_in_seconds"] == 0.01
        for event in events
    )
    status = json.loads(
        (directories.status / "coordinator.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "stopped"
    assert not (directories.root / "coordinator.lock").exists()


def test_pause_lease_reaps_actor_before_ready_and_restarts_once(tmp_path) -> None:
    experiment = pause_shared_experiment(tmp_path)
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-8gpu-optimized.yaml",
        directories=directories,
        base_environment={},
    )
    launches: dict[str, list[FakeProcess]] = {}

    def process_factory(command: list[str], **_options: Any) -> FakeProcess:
        name = worker_name(command)
        process = FakeProcess(exit_immediately=False)
        launches.setdefault(name, []).append(process)
        return process

    token = "normal-pause-token"
    phase = "request"
    post_release_cycles = 0

    def stop_requested() -> bool:
        nonlocal phase, post_release_cycles
        if "arena-promotion" not in launches:
            return False
        if phase == "request":
            write_pause_request(
                directories.gpu_pause,
                token=token,
                owner_pid=launches["arena-promotion"][0].pid,
            )
            phase = "ready"
            return False
        if phase == "ready":
            ack_path = gpu_pause_ack_path(directories.gpu_pause)
            if ack_path.is_file():
                acknowledgement = json.loads(ack_path.read_text(encoding="utf-8"))
                if acknowledgement["state"] == "ready":
                    assert launches["actor-gpu-7"][0].returncode == -15
                    request = json.loads(
                        directories.gpu_pause.read_text(encoding="utf-8")
                    )
                    request["state"] = "released"
                    request["heartbeat_ns"] = time.time_ns()
                    atomic_json(directories.gpu_pause, request)
                    phase = "released"
            return False
        if phase == "released" and len(launches["actor-gpu-7"]) == 2:
            post_release_cycles += 1
            return post_release_cycles >= 3
        return False

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=stop_requested, max_monitor_cycles=10) == 0
    assert len(launches["actor-gpu-7"]) == 2
    assert coordinator.workers["actor-gpu-7"].restart_count == 0
    acknowledgement = json.loads(
        gpu_pause_ack_path(directories.gpu_pause).read_text(encoding="utf-8")
    )
    assert acknowledgement["token"] == token
    assert acknowledgement["state"] == "released"
    assert json.loads(directories.gpu_pause.read_text())["state"] == "released"
    event_names = [event["event"] for event in coordinator_events(directories)]
    ordered = [
        "pause_lease_requested",
        "pause_target_reaped",
        "pause_lease_ready",
        "pause_lease_release_requested",
        "pause_target_restarted",
    ]
    assert [event_names.index(name) for name in ordered] == sorted(
        event_names.index(name) for name in ordered
    )


def test_pause_lease_never_acknowledges_live_actor(tmp_path) -> None:
    experiment = pause_shared_experiment(tmp_path)
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-8gpu-optimized.yaml",
        directories=directories,
        base_environment={},
    )
    launches: dict[str, list[FakeProcess]] = {}

    def process_factory(command: list[str], **_options: Any) -> FakeProcess:
        name = worker_name(command)
        process = FakeProcess(
            exit_immediately=False,
            exit_on_terminate=name != "actor-gpu-7",
            exit_on_kill=name != "actor-gpu-7",
        )
        launches.setdefault(name, []).append(process)
        return process

    requested = False

    def stop_requested() -> bool:
        nonlocal requested
        if "arena-promotion" in launches and not requested:
            requested = True
            write_pause_request(
                directories.gpu_pause,
                token="live-actor-token",
                owner_pid=launches["arena-promotion"][0].pid,
            )
        return False

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=stop_requested, max_monitor_cycles=1) == 0
    acknowledgement = json.loads(
        gpu_pause_ack_path(directories.gpu_pause).read_text(encoding="utf-8")
    )
    assert acknowledgement["state"] == "stopping"
    assert launches["actor-gpu-7"][0].returncode is None
    assert not any(
        event["event"] == "pause_lease_ready"
        for event in coordinator_events(directories)
    )


def test_pause_lease_hard_kills_after_grace_then_restarts_actor(tmp_path) -> None:
    experiment = pause_shared_experiment(tmp_path)
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-8gpu-optimized.yaml",
        directories=directories,
        base_environment={},
    )
    launches: dict[str, list[FakeProcess]] = {}

    def process_factory(command: list[str], **_options: Any) -> FakeProcess:
        name = worker_name(command)
        first_shared_actor = name == "actor-gpu-7" and name not in launches
        process = FakeProcess(
            exit_immediately=False,
            exit_on_terminate=not first_shared_actor,
            exit_on_kill=True,
        )
        launches.setdefault(name, []).append(process)
        return process

    phase = "request"

    def stop_requested() -> bool:
        nonlocal phase
        if "arena-promotion" not in launches:
            return False
        if phase == "request":
            write_pause_request(
                directories.gpu_pause,
                token="hard-kill-token",
                owner_pid=launches["arena-promotion"][0].pid,
            )
            phase = "ready"
            return False
        ack_path = gpu_pause_ack_path(directories.gpu_pause)
        if phase == "ready" and ack_path.is_file():
            acknowledgement = json.loads(ack_path.read_text())
            if acknowledgement["state"] == "ready":
                assert launches["actor-gpu-7"][0].returncode == -9
                request = json.loads(directories.gpu_pause.read_text())
                request["state"] = "released"
                request["heartbeat_ns"] = time.time_ns()
                atomic_json(directories.gpu_pause, request)
                phase = "released"
        return phase == "released" and len(launches["actor-gpu-7"]) == 2

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=stop_requested, max_monitor_cycles=20) == 0
    first_actor = launches["actor-gpu-7"][0]
    assert first_actor.terminate_calls == 1
    assert first_actor.kill_calls == 1
    assert len(launches["actor-gpu-7"]) == 2


def test_pause_lease_fails_closed_when_actor_ignores_sigkill(tmp_path) -> None:
    experiment = pause_shared_experiment(tmp_path)
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-8gpu-optimized.yaml",
        directories=directories,
        base_environment={},
    )
    launches: dict[str, list[FakeProcess]] = {}

    def process_factory(command: list[str], **_options: Any) -> FakeProcess:
        name = worker_name(command)
        stubborn = name == "actor-gpu-7"
        process = FakeProcess(
            exit_immediately=False,
            exit_on_terminate=not stubborn,
            exit_on_kill=not stubborn,
        )
        launches.setdefault(name, []).append(process)
        return process

    requested = False

    def stop_requested() -> bool:
        nonlocal requested
        if "arena-promotion" in launches and not requested:
            requested = True
            write_pause_request(
                directories.gpu_pause,
                token="fail-closed-token",
                owner_pid=launches["arena-promotion"][0].pid,
            )
        return False

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=stop_requested) == 1
    assert len(launches["actor-gpu-7"]) == 1
    acknowledgement = json.loads(
        gpu_pause_ack_path(directories.gpu_pause).read_text(encoding="utf-8")
    )
    assert acknowledgement["state"] == "failed"
    assert "SIGKILL" in acknowledgement["reason"]
    assert not any(
        event["event"] == "pause_lease_ready"
        for event in coordinator_events(directories)
    )


def test_pause_lease_fails_closed_when_actor_restart_cannot_spawn(tmp_path) -> None:
    experiment = pause_shared_experiment(tmp_path)
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-8gpu-optimized.yaml",
        directories=directories,
        base_environment={},
    )
    launches: dict[str, list[FakeProcess]] = {}

    def process_factory(command: list[str], **_options: Any) -> FakeProcess:
        name = worker_name(command)
        if name == "actor-gpu-7" and name in launches:
            raise OSError("injected pause restart failure")
        process = FakeProcess(exit_immediately=False)
        launches.setdefault(name, []).append(process)
        return process

    phase = "request"

    def stop_requested() -> bool:
        nonlocal phase
        if "arena-promotion" not in launches:
            return False
        if phase == "request":
            write_pause_request(
                directories.gpu_pause,
                token="spawn-failure-token",
                owner_pid=launches["arena-promotion"][0].pid,
            )
            phase = "ready"
            return False
        ack_path = gpu_pause_ack_path(directories.gpu_pause)
        if phase == "ready" and ack_path.is_file():
            acknowledgement = json.loads(ack_path.read_text())
            if acknowledgement["state"] == "ready":
                request = json.loads(directories.gpu_pause.read_text())
                request["state"] = "released"
                request["heartbeat_ns"] = time.time_ns()
                atomic_json(directories.gpu_pause, request)
                phase = "released"
        return False

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=stop_requested) == 1
    assert len(launches["actor-gpu-7"]) == 1
    assert coordinator.workers["actor-gpu-7"].restart_count == 0
    acknowledgement = json.loads(
        gpu_pause_ack_path(directories.gpu_pause).read_text(encoding="utf-8")
    )
    assert acknowledgement["state"] == "failed"
    assert "injected pause restart failure" in acknowledgement["reason"]


@pytest.mark.parametrize("failure", ["crash", "stale"])
def test_arena_failure_or_stale_lease_restores_actor(tmp_path, failure: str) -> None:
    experiment = pause_shared_experiment(tmp_path)
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-8gpu-optimized.yaml",
        directories=directories,
        base_environment={},
    )
    launches: dict[str, list[FakeProcess]] = {}

    def process_factory(command: list[str], **_options: Any) -> FakeProcess:
        name = worker_name(command)
        process = FakeProcess(exit_immediately=False)
        launches.setdefault(name, []).append(process)
        return process

    token = f"{failure}-recovery-token"
    requested_ns = time.time_ns() - 1_000_000_000
    phase = "request"

    def stop_requested() -> bool:
        nonlocal phase
        if "arena-promotion" not in launches:
            return False
        if phase == "request":
            write_pause_request(
                directories.gpu_pause,
                token=token,
                owner_pid=launches["arena-promotion"][0].pid,
                requested_ns=requested_ns,
            )
            phase = "ready"
            return False
        ack_path = gpu_pause_ack_path(directories.gpu_pause)
        if phase == "ready" and ack_path.is_file():
            acknowledgement = json.loads(ack_path.read_text())
            if acknowledgement["state"] == "ready":
                if failure == "crash":
                    launches["arena-promotion"][0].returncode = 17
                else:
                    write_pause_request(
                        directories.gpu_pause,
                        token=token,
                        owner_pid=launches["arena-promotion"][0].pid,
                        state="active",
                        requested_ns=requested_ns,
                        heartbeat_ns=requested_ns,
                    )
                phase = "recovering"
            return False
        if phase == "recovering" and ack_path.is_file():
            acknowledgement = json.loads(ack_path.read_text())
            return (
                acknowledgement["state"] == "recovered"
                and len(launches["actor-gpu-7"]) == 2
            )
        return False

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=stop_requested, max_monitor_cycles=20) == 0
    assert len(launches["actor-gpu-7"]) == 2
    acknowledgement = json.loads(
        gpu_pause_ack_path(directories.gpu_pause).read_text(encoding="utf-8")
    )
    assert acknowledgement["state"] == "recovered"
    event_names = [event["event"] for event in coordinator_events(directories)]
    assert "pause_owner_reaped" in event_names
    assert "pause_target_restarted" in event_names


def test_learner_pause_sharing_waits_for_fresh_progress_ack(tmp_path) -> None:
    experiment = load_config(CONFIGS / "h100-4gpu.yaml")
    experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            directories=RunDirectoryConfig(root=str(tmp_path / "learner-share")),
            shutdown=ShutdownConfig(
                monitor_interval_seconds=0.01,
                heartbeat_interval_seconds=0.02,
                stale_heartbeat_seconds=0.1,
                stall_timeout_seconds=1.0,
                terminate_grace_seconds=0.02,
                kill_grace_seconds=0.01,
            ),
            promotion=replace(
                experiment.orchestration.promotion,
                gpu_id=0,
                pause_sharing_mode=True,
                pause_ready_timeout_seconds=0.1,
                pause_release_timeout_seconds=0.1,
            ),
        ),
    )
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-4gpu.yaml",
        directories=directories,
        base_environment={},
    )
    launches: dict[str, list[FakeProcess]] = {}

    def process_factory(command: list[str], **_options: Any) -> FakeProcess:
        name = worker_name(command)
        process = FakeProcess(exit_immediately=False)
        launches.setdefault(name, []).append(process)
        return process

    token = "learner-share-token"
    phase = "request"
    observed_without_termination = False

    def stop_requested() -> bool:
        nonlocal observed_without_termination, phase
        if "arena-promotion" not in launches:
            return False
        if phase == "request":
            write_pause_request(
                directories.gpu_pause,
                token=token,
                owner_pid=launches["arena-promotion"][0].pid,
                gpu_id=0,
            )
            phase = "heartbeat"
            return False
        if phase == "heartbeat":
            request = json.loads(directories.gpu_pause.read_text())
            atomic_json(
                directories.status / "learner.heartbeat.json",
                {
                    "schema_version": 1,
                    "worker": "learner",
                    "pid": launches["learner"][0].pid,
                    "heartbeat_ns": time.time_ns(),
                    "phase": "arena_gpu_pause",
                    "progress": 1,
                    "progress_ns": max(time.time_ns(), request["requested_ns"]),
                },
            )
            phase = "ready"
            return False
        acknowledgement = json.loads(
            gpu_pause_ack_path(directories.gpu_pause).read_text(encoding="utf-8")
        )
        if phase == "ready" and acknowledgement["state"] == "ready":
            assert launches["learner"][0].terminate_calls == 0
            request = json.loads(directories.gpu_pause.read_text())
            request["state"] = "released"
            request["heartbeat_ns"] = time.time_ns()
            atomic_json(directories.gpu_pause, request)
            phase = "released"
            return False
        if phase == "released" and acknowledgement["state"] == "released":
            observed_without_termination = launches["learner"][0].terminate_calls == 0
            return True
        return False

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=stop_requested, max_monitor_cycles=10) == 0
    assert observed_without_termination is True
    assert len(launches["learner"]) == 1
    assert not directories.gpu_pause.exists()


def test_final_drain_suppresses_pause_actor_restart(tmp_path) -> None:
    experiment = pause_shared_experiment(tmp_path)
    directories = RunDirectories.from_experiment(experiment)
    directories.create()
    identity = load_or_create_run_identity(directories.run_identity)
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-8gpu-optimized.yaml",
        directories=directories,
        base_environment={},
    )
    launches: dict[str, list[FakeProcess]] = {}

    def process_factory(command: list[str], **_options: Any) -> FakeProcess:
        name = worker_name(command)
        process = FakeProcess(exit_immediately=False)
        launches.setdefault(name, []).append(process)
        return process

    token = "final-drain-token"
    final_identity = "sha256-" + "d" * 64
    phase = "request"

    def stop_requested() -> bool:
        nonlocal phase
        if "arena-promotion" not in launches:
            return False
        if phase == "request":
            write_pause_request(
                directories.gpu_pause,
                token=token,
                owner_pid=launches["arena-promotion"][0].pid,
            )
            phase = "ready"
            return False
        acknowledgement = json.loads(
            gpu_pause_ack_path(directories.gpu_pause).read_text(encoding="utf-8")
        )
        if phase == "ready" and acknowledgement["state"] == "ready":
            atomic_json(
                directories.learner / "learner-complete.json",
                {
                    "schema_version": 1,
                    "run_id": identity.run_id,
                    "generation_family": identity.generation_family,
                    "candidate_identity": final_identity,
                    "candidate_step": 100,
                    "completed_ns": time.time_ns(),
                },
            )
            launches["learner"][0].returncode = 0
            phase = "learner_exiting"
            return False
        if phase == "learner_exiting":
            assert coordinator.draining is True
            request = json.loads(directories.gpu_pause.read_text())
            request["state"] = "released"
            request["heartbeat_ns"] = time.time_ns()
            atomic_json(directories.gpu_pause, request)
            atomic_json(
                directories.arena / "promotion-status.json",
                {
                    "schema_version": 1,
                    "candidate_identity": final_identity,
                    "terminal": True,
                    "decision": "reject",
                },
            )
            phase = "released"
        return False

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=stop_requested, max_monitor_cycles=10) == 0
    assert len(launches["actor-gpu-7"]) == 1
    acknowledgement = json.loads(
        gpu_pause_ack_path(directories.gpu_pause).read_text(encoding="utf-8")
    )
    assert acknowledgement["state"] == "draining"
    assert coordinator.workers["actor-gpu-7"].failure_reason is None
    assert not any(
        event["event"] == "pause_target_restarted"
        for event in coordinator_events(directories)
    )


def test_coordinator_drains_final_candidate_before_stopping(tmp_path) -> None:
    experiment = load_config(CONFIGS / "h100-4gpu.yaml")
    experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            directories=RunDirectoryConfig(root=str(tmp_path / "drain-run")),
            shutdown=ShutdownConfig(
                monitor_interval_seconds=0.01,
                heartbeat_interval_seconds=1.0,
                stale_heartbeat_seconds=100.0,
                stall_timeout_seconds=200.0,
                terminate_grace_seconds=0.01,
                kill_grace_seconds=0.01,
            ),
        ),
    )
    directories = RunDirectories.from_experiment(experiment)
    directories.create()
    identity = load_or_create_run_identity(directories.run_identity)
    final_identity = "sha256-" + "f" * 64
    atomic_json(
        directories.learner / "learner-complete.json",
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "candidate_identity": final_identity,
            "candidate_step": 100,
            "completed_ns": 1,
        },
    )
    atomic_json(
        directories.arena / "promotion-status.json",
        {
            "schema_version": 1,
            "candidate_identity": final_identity,
            "terminal": True,
            "decision": "reject",
        },
    )
    specs = build_worker_specs(
        experiment,
        config_path=CONFIGS / "h100-4gpu.yaml",
        directories=directories,
        base_environment={},
    )
    processes: list[tuple[list[str], FakeProcess]] = []

    def process_factory(command: list[str], **_options: Any) -> FakeProcess:
        process = FakeProcess(exit_immediately=False)
        if "train" in command:
            process.returncode = 0
        processes.append((command, process))
        return process

    clock = FakeClock()
    coordinator = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
        process_factory=process_factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert coordinator.run(stop_requested=lambda: False) == 0
    actor_processes = [process for command, process in processes if "actor" in command]
    assert actor_processes
    assert all(process.returncode == -15 for process in actor_processes)
    events = (directories.metrics / "coordinator.jsonl").read_text(encoding="utf-8")
    assert "final_drain_started" in events


def test_evaluation_loader_requires_and_applies_ema_with_config_checks(
    tmp_path,
) -> None:
    model_config = ModelConfig(
        width=8,
        rrt_groups=1,
        attention_heads=2,
        kv_heads=1,
    )
    game_config = GameConfig()
    model = GraphResTNet(model_config)
    ema = ExponentialMovingAverage(model, decay=0.9)
    for value in ema.shadow.values():
        value.fill_(0.125)
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(warmup_steps=0, total_steps=4)
    )
    identity = RunIdentity(tmp_path / "run.json", "run-test", "family-test", 1)
    publisher = ImmutableModelPublisher(tmp_path, identity)
    published = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=3,
        epoch=0,
        config={
            "model": asdict(model_config),
            "game": asdict(game_config),
        },
    )
    checkpoint = published.checkpoint
    assert latest_checkpoint(tmp_path) == checkpoint
    restored = GraphResTNet(model_config)
    metadata = load_ema_checkpoint(
        checkpoint,
        model=restored,
        expected_model_config=asdict(model_config),
        expected_game_config=asdict(game_config),
        expected_run_id=identity.run_id,
        expected_generation_family=identity.generation_family,
        expected_sha256=published.checkpoint_sha256,
        expected_bytes=published.checkpoint_bytes,
    )
    assert metadata["step"] == 3
    for value in restored.state_dict().values():
        if value.is_floating_point():
            torch.testing.assert_close(value, torch.full_like(value, 0.125))

    manifest_path = tmp_path / "candidate.json"
    loaded = load_model_manifest(manifest_path)
    assert loaded.checkpoint == checkpoint
    assert loaded.role == "candidate"
    assert loaded.model_identity.startswith("sha256-")
    relocated_root = tmp_path.parent / f"{tmp_path.name}-relocated"
    shutil.copytree(tmp_path, relocated_root)
    relocated = load_model_manifest(relocated_root / "candidate.json")
    assert relocated.checkpoint.parent == relocated_root / "checkpoints"
    assert relocated.model_identity == loaded.model_identity

    incompatible = replace(model_config, width=16)
    try:
        load_ema_checkpoint(
            checkpoint,
            model=GraphResTNet(model_config),
            expected_model_config=asdict(incompatible),
            expected_game_config=asdict(game_config),
            expected_run_id=identity.run_id,
            expected_generation_family=identity.generation_family,
            expected_sha256=published.checkpoint_sha256,
            expected_bytes=published.checkpoint_bytes,
        )
    except ValueError as error:
        assert "model/feature configuration" in str(error)
    else:
        raise AssertionError("incompatible model configuration was accepted")
    with checkpoint.open("ab") as stream:
        stream.write(b"tamper")
    try:
        load_model_manifest(manifest_path)
    except ValueError as error:
        assert "byte length" in str(error) or "SHA-256" in str(error)
    else:
        raise AssertionError("tampered checkpoint was accepted")
