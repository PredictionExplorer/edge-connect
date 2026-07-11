from __future__ import annotations

import json
import signal
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import pytest

import startrain.orchestration as orchestration_module
from startrain.actor import RingMixtureScheduler
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
)
from startrain.runtime import (
    RunIdentity,
    atomic_json,
    load_or_create_run_identity,
)
from startrain.training import build_scheduler


CONFIGS = Path(__file__).parents[1] / "configs"


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
    assert [gpu.gpu_id for gpu in eight.orchestration.learner_gpus] == [0]
    assert [gpu.gpu_id for gpu in eight.orchestration.actor_gpus] == list(range(1, 7))
    assert [gpu.gpu_id for gpu in four.orchestration.learner_gpus] == [0]
    assert [gpu.gpu_id for gpu in four.orchestration.actor_gpus] == [1, 2]

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
    with pytest.raises(ConfigError, match="overlaps"):
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
    with pytest.raises(ConfigError, match="self-play actor"):
        replace(
            four.orchestration,
            promotion=replace(
                four.orchestration.promotion,
                gpu_id=1,
                pause_sharing_mode=True,
            ),
        )


def test_ring_scheduler_curriculum_then_favors_deficits() -> None:
    experiment = load_config(CONFIGS / "h100-8gpu.yaml")
    scheduler = RingMixtureScheduler(experiment.orchestration.ring_mixture, seed=11)
    empty = {ring: 0 for ring in range(3, 13)}
    assert {scheduler.choose(empty) for _ in range(100)} <= {3, 4}

    mature = {ring: 200_000 for ring in range(3, 13)}
    mature[12] = 0
    draws = [scheduler.choose(mature) for _ in range(2_000)]
    assert set(draws) == set(range(3, 13))
    assert draws.count(12) > draws.count(3) * 2


def test_ring_mixture_stage_selection_uses_aggregate_sample_boundaries() -> None:
    mixture = RingMixtureConfig(
        curriculum=(
            CurriculumStage(until_samples=100, rings=(3, 4)),
            CurriculumStage(until_samples=500, rings=(3, 4, 5, 6)),
        )
    )
    assert mixture.active_rings(0) == (3, 4)
    assert mixture.active_rings(99) == (3, 4)
    assert mixture.active_rings(100) == (3, 4, 5, 6)
    assert mixture.active_rings(499) == (3, 4, 5, 6)
    assert mixture.active_rings(500) == tuple(range(3, 13))


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


class FakeProcess:
    next_pid = 10_000

    def __init__(self, *, exit_immediately: bool) -> None:
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode: int | None = None
        self.exit_immediately = exit_immediately

    def poll(self) -> int | None:
        if self.returncode is None and self.exit_immediately:
            self.returncode = 7
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


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
