"""CUDA-free single-host process coordinator for learner and actor jobs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from .config import ExperimentConfig, load_config
from .runtime import (
    SignalLatch,
    append_jsonl,
    atomic_json,
    load_or_create_run_identity,
)


class ProcessProtocol(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


ProcessFactory = Callable[..., ProcessProtocol]


@dataclass(frozen=True, slots=True)
class RunDirectories:
    root: Path
    replay: Path
    learner: Path
    logs: Path
    status: Path
    metrics: Path
    arena: Path
    run_identity: Path
    gpu_pause: Path

    @classmethod
    def from_experiment(cls, experiment: ExperimentConfig) -> "RunDirectories":
        configured = experiment.orchestration.directories
        root = Path(configured.root).expanduser().resolve()
        return cls(
            root=root,
            replay=root / configured.replay,
            learner=root / configured.learner,
            logs=root / configured.logs,
            status=root / configured.status,
            metrics=root / configured.metrics,
            arena=root / "arena",
            run_identity=root / "run.json",
            gpu_pause=root / configured.status / "arena-gpu-pause.json",
        )

    def create(self) -> None:
        for path in (
            self.root,
            self.replay,
            self.learner,
            self.logs,
            self.status,
            self.metrics,
            self.arena,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    name: str
    role: str
    gpu_ids: tuple[int, ...]
    cpu_threads: int
    command: tuple[str, ...]
    environment: Mapping[str, str]
    heartbeat_path: Path
    metrics_path: Path
    log_path: Path


@dataclass(slots=True)
class ManagedWorker:
    spec: WorkerSpec
    process: ProcessProtocol | None = None
    log_stream: TextIO | None = None
    started_at: float = 0.0
    restart_count: int = 0
    next_start_at: float = 0.0
    state: str = "pending"
    last_exit_code: int | None = None
    failure_reason: str | None = None
    termination_deadline: float = 0.0
    last_progress: object | None = None
    last_progress_at: float = 0.0

    @property
    def live(self) -> bool:
        return self.process is not None and self.process.poll() is None


class CoordinatorLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "created_ns": time.time_ns()})
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                if self._owner_is_live():
                    raise RuntimeError(
                        f"another coordinator owns run directory {self.path.parent}"
                    ) from None
                self.path.unlink(missing_ok=True)
                continue
            try:
                os.write(descriptor, payload.encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.acquired = True
            return
        raise RuntimeError("could not acquire coordinator lock")

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def _owner_is_live(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def build_worker_specs(
    experiment: ExperimentConfig,
    *,
    config_path: str | Path,
    directories: RunDirectories,
    python_executable: str = sys.executable,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[WorkerSpec, ...]:
    orchestration = experiment.orchestration
    if not orchestration.enabled:
        raise ValueError("orchestration is disabled in this configuration")
    environment = dict(os.environ if base_environment is None else base_environment)
    config = str(Path(config_path).resolve())
    candidate_manifest = directories.learner / "candidate.json"
    champion_manifest = directories.learner / "champion.json"
    learner_gpus = orchestration.learner_gpus
    learner_threads = learner_gpus[0].cpu_threads
    learner_environment = _worker_environment(
        environment,
        gpu_ids=tuple(gpu.gpu_id for gpu in learner_gpus),
        cpu_threads=learner_threads,
    )
    train_arguments = (
        "--config",
        config,
        "--replay-store",
        str(directories.replay),
        "--output",
        str(directories.learner),
        "--device",
        "cuda",
        "--heartbeat",
        str(directories.status / "learner.heartbeat.json"),
        "--run-identity",
        str(directories.run_identity),
        "--promotion-status",
        str(directories.arena / "promotion-status.json"),
    )
    if experiment.learner.resume_latest:
        train_arguments += ("--resume-latest",)
    if orchestration.promotion.pause_sharing_mode:
        train_arguments += ("--gpu-pause", str(directories.gpu_pause))
    if orchestration.distributed.enabled:
        learner_environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        command = (
            python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={len(learner_gpus)}",
            "-m",
            "startrain.cli",
            "train",
            *train_arguments,
            "--distributed-backend",
            orchestration.distributed.backend,
        )
    else:
        command = (
            python_executable,
            "-m",
            "startrain.cli",
            "train",
            *train_arguments,
        )
    specs = [
        WorkerSpec(
            name="learner",
            role="learner",
            gpu_ids=tuple(gpu.gpu_id for gpu in learner_gpus),
            cpu_threads=learner_threads,
            command=command,
            environment=learner_environment,
            heartbeat_path=directories.status / "learner.heartbeat.json",
            metrics_path=directories.learner / "metrics.jsonl",
            log_path=directories.logs / "learner.log",
        )
    ]
    for gpu in orchestration.actor_gpus:
        name = f"actor-gpu-{gpu.gpu_id}"
        specs.append(
            WorkerSpec(
                name=name,
                role="actor",
                gpu_ids=(gpu.gpu_id,),
                cpu_threads=gpu.cpu_threads,
                command=(
                    python_executable,
                    "-m",
                    "startrain.cli",
                    "actor",
                    "--config",
                    config,
                    "--gpu-id",
                    str(gpu.gpu_id),
                    "--replay-store",
                    str(directories.replay),
                    "--manifest",
                    str(champion_manifest),
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--run-identity",
                    str(directories.run_identity),
                    "--heartbeat",
                    str(directories.status / f"{name}.heartbeat.json"),
                    "--metrics",
                    str(directories.metrics / f"{name}.jsonl"),
                    "--device",
                    "cuda",
                ),
                environment=_worker_environment(
                    environment,
                    gpu_ids=(gpu.gpu_id,),
                    cpu_threads=gpu.cpu_threads,
                ),
                heartbeat_path=directories.status / f"{name}.heartbeat.json",
                metrics_path=directories.metrics / f"{name}.jsonl",
                log_path=directories.logs / f"{name}.log",
            )
        )
    promotion = orchestration.promotion
    if promotion.enabled:
        name = "arena-promotion"
        specs.append(
            WorkerSpec(
                name=name,
                role="arena",
                gpu_ids=(promotion.gpu_id,),
                cpu_threads=promotion.cpu_threads,
                command=(
                    python_executable,
                    "-m",
                    "startrain.cli",
                    "promote",
                    "--config",
                    config,
                    "--run-identity",
                    str(directories.run_identity),
                    "--candidate",
                    str(candidate_manifest),
                    "--champion",
                    str(champion_manifest),
                    "--results",
                    str(directories.arena),
                    "--heartbeat",
                    str(directories.status / f"{name}.heartbeat.json"),
                    "--device",
                    promotion.device,
                    *(
                        ("--gpu-pause", str(directories.gpu_pause))
                        if promotion.pause_sharing_mode
                        else ()
                    ),
                ),
                environment=_worker_environment(
                    environment,
                    gpu_ids=(promotion.gpu_id,),
                    cpu_threads=promotion.cpu_threads,
                ),
                heartbeat_path=directories.status / f"{name}.heartbeat.json",
                metrics_path=directories.metrics / f"{name}.jsonl",
                log_path=directories.logs / f"{name}.log",
            )
        )
    return tuple(specs)


def _worker_environment(
    base: Mapping[str, str],
    *,
    gpu_ids: Sequence[int],
    cpu_threads: int,
) -> dict[str, str]:
    output = dict(base)
    output.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_ids),
            "RAYON_NUM_THREADS": str(cpu_threads),
            "OMP_NUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return output


class Coordinator:
    def __init__(
        self,
        *,
        experiment: ExperimentConfig,
        specs: Sequence[WorkerSpec],
        directories: RunDirectories,
        process_factory: ProcessFactory = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not specs or sum(spec.role == "learner" for spec in specs) != 1:
            raise ValueError("coordinator requires exactly one learner job")
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("worker names must be unique")
        self.experiment = experiment
        self.directories = directories
        self.process_factory = process_factory
        self.clock = clock
        self.sleep = sleep
        self.workers = {spec.name: ManagedWorker(spec) for spec in specs}
        self.lock = CoordinatorLock(directories.root / "coordinator.lock")
        self.metrics_path = directories.metrics / "coordinator.jsonl"
        self.status_path = directories.status / "coordinator.json"
        self.stopping = False
        self.draining = False
        self.drain_deadline = 0.0

    def run(
        self,
        *,
        stop_requested: Callable[[], bool],
        max_monitor_cycles: int | None = None,
    ) -> int:
        self.directories.create()
        self.lock.acquire()
        exit_code = 0
        cycles = 0
        try:
            load_or_create_run_identity(
                self.directories.run_identity,
                requested_run_id=self.experiment.orchestration.run_id,
            )
            if stop_requested():
                return 0
            for worker in self.workers.values():
                self._start_or_schedule(worker)
            while not stop_requested():
                now = self.clock()
                exhausted = False
                for worker in self.workers.values():
                    health_failure = (
                        self._heartbeat_failure(worker, now)
                        if worker.state == "running"
                        else None
                    )
                    if health_failure is not None:
                        assert worker.process is not None
                        _signal_process(worker.process, signal.SIGTERM)
                        worker.state = "terminating"
                        worker.failure_reason = health_failure
                        worker.termination_deadline = (
                            now
                            + self.experiment.orchestration.shutdown.terminate_grace_seconds
                        )
                    code = worker.process.poll() if worker.process is not None else None
                    if code is not None and worker.state in (
                        "running",
                        "terminating",
                        "killing",
                        "draining",
                    ):
                        if (
                            worker.spec.role == "learner"
                            and code == 0
                            and worker.state == "running"
                        ):
                            if self._learner_completion_identity() is not None:
                                worker.state = "completed"
                                self._event("worker_completed", worker, exit_code=code)
                                self._begin_drain(now)
                                continue
                            self._schedule_restart(worker, code=1, now=now)
                            worker.failure_reason = (
                                "learner exited without completion marker"
                            )
                            continue
                        if worker.state == "draining":
                            worker.state = "drained"
                            worker.last_exit_code = code
                            if worker.log_stream is not None:
                                worker.log_stream.close()
                                worker.log_stream = None
                            continue
                        self._schedule_restart(worker, code=code, now=now)
                    elif (
                        worker.state == "terminating"
                        and now >= worker.termination_deadline
                    ):
                        assert worker.process is not None
                        _signal_process(worker.process, signal.SIGKILL)
                        worker.state = "killing"
                        worker.termination_deadline = (
                            now
                            + self.experiment.orchestration.shutdown.kill_grace_seconds
                        )
                    elif (
                        worker.state == "killing" and now >= worker.termination_deadline
                    ):
                        worker.state = "exhausted"
                        worker.failure_reason = "worker ignored SIGKILL"
                        exhausted = True
                    if worker.state == "backoff" and now >= worker.next_start_at:
                        if (
                            worker.restart_count
                            > self.experiment.orchestration.restart.max_restarts
                        ):
                            worker.state = "exhausted"
                            exhausted = True
                        else:
                            self._start_or_schedule(worker)
                self._write_status()
                if self.draining and self._drain_complete():
                    exit_code = 0
                    break
                if self.draining and now >= self.drain_deadline:
                    exit_code = 1
                    self._event_drain_timeout()
                    break
                if exhausted:
                    exit_code = 1
                    break
                cycles += 1
                if max_monitor_cycles is not None and cycles >= max_monitor_cycles:
                    break
                self.sleep(
                    self.experiment.orchestration.shutdown.monitor_interval_seconds
                )
            return exit_code
        finally:
            self.stopping = True
            self._stop_all()
            self._write_status(final=True)
            self.lock.release()

    def _start(self, worker: ManagedWorker) -> None:
        if worker.live:
            raise RuntimeError(f"refusing to duplicate live worker {worker.spec.name}")
        if worker.log_stream is not None:
            worker.log_stream.close()
        worker.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        worker.log_stream = worker.spec.log_path.open(
            "a", encoding="utf-8", buffering=1
        )
        worker.process = self.process_factory(
            list(worker.spec.command),
            stdin=subprocess.DEVNULL,
            stdout=worker.log_stream,
            stderr=subprocess.STDOUT,
            env=dict(worker.spec.environment),
            close_fds=True,
            start_new_session=True,
        )
        worker.started_at = self.clock()
        worker.state = "running"
        worker.last_exit_code = None
        worker.failure_reason = None
        worker.termination_deadline = 0.0
        worker.last_progress = None
        worker.last_progress_at = worker.started_at
        self._event("worker_started", worker)

    def _start_or_schedule(self, worker: ManagedWorker) -> None:
        try:
            self._start(worker)
        except OSError as error:
            if worker.log_stream is not None:
                worker.log_stream.close()
                worker.log_stream = None
            worker.process = None
            worker.restart_count += 1
            worker.state = "backoff"
            worker.failure_reason = f"spawn failed: {error}"
            backoff = self._restart_backoff(worker.restart_count)
            worker.next_start_at = self.clock() + backoff
            self._event(
                "worker_spawn_failed",
                worker,
                error=str(error),
                restart_in_seconds=backoff,
            )

    def _schedule_restart(
        self, worker: ManagedWorker, *, code: int, now: float
    ) -> None:
        runtime = max(0.0, now - worker.started_at)
        restart = self.experiment.orchestration.restart
        if runtime >= restart.stable_reset_seconds:
            worker.restart_count = 0
        worker.restart_count += 1
        worker.last_exit_code = code
        worker.state = "backoff"
        worker.process = None
        if worker.log_stream is not None:
            worker.log_stream.close()
            worker.log_stream = None
        backoff = self._restart_backoff(worker.restart_count)
        worker.next_start_at = now + backoff
        self._event(
            "worker_exited",
            worker,
            exit_code=code,
            restart_in_seconds=backoff,
        )

    def _restart_backoff(self, restart_count: int) -> float:
        restart = self.experiment.orchestration.restart
        return min(
            restart.maximum_backoff_seconds,
            restart.initial_backoff_seconds * (2 ** (restart_count - 1)),
        )

    def _heartbeat_failure(self, worker: ManagedWorker, now: float) -> str | None:
        if not worker.live:
            return None
        threshold = self.experiment.orchestration.shutdown.stale_heartbeat_seconds
        if now - worker.started_at <= threshold:
            return None
        try:
            modified_age = time.time() - worker.spec.heartbeat_path.stat().st_mtime
        except FileNotFoundError:
            return "missing heartbeat"
        if modified_age > threshold:
            return "stale heartbeat"
        try:
            payload = json.loads(worker.spec.heartbeat_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "invalid heartbeat"
        progress = payload.get("progress")
        if progress != worker.last_progress:
            worker.last_progress = progress
            worker.last_progress_at = now
            return None
        stall = self.experiment.orchestration.shutdown.stall_timeout_seconds
        if now - worker.last_progress_at > stall:
            return "main loop stalled"
        return None

    def _learner_completion_identity(self) -> str | None:
        path = self.directories.learner / "learner-complete.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        run_identity = load_or_create_run_identity(
            self.directories.run_identity,
            requested_run_id=self.experiment.orchestration.run_id,
        )
        if (
            not isinstance(payload, dict)
            or payload.get("run_id") != run_identity.run_id
            or payload.get("generation_family") != run_identity.generation_family
        ):
            return None
        identity = payload.get("candidate_identity")
        return identity if isinstance(identity, str) and identity else None

    def _begin_drain(self, now: float) -> None:
        if self.draining:
            return
        self.draining = True
        self.drain_deadline = (
            now + self.experiment.orchestration.promotion.final_drain_timeout_seconds
        )
        for worker in self.workers.values():
            if worker.spec.role == "actor" and worker.live:
                assert worker.process is not None
                _signal_process(worker.process, signal.SIGTERM)
                worker.state = "draining"
        append_jsonl(
            self.metrics_path,
            {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "event": "final_drain_started",
                "candidate_identity": self._learner_completion_identity(),
            },
            durable=True,
        )

    def _drain_complete(self) -> bool:
        final_identity = self._learner_completion_identity()
        if final_identity is None:
            return False
        status_path = self.directories.arena / "promotion-status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            isinstance(status, dict)
            and status.get("candidate_identity") == final_identity
            and status.get("terminal") is True
            and status.get("decision")
            in (
                "promote",
                "reject",
                "reject_ring_regression",
                "reject_max_pairs",
            )
        )

    def _event_drain_timeout(self) -> None:
        append_jsonl(
            self.metrics_path,
            {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "event": "final_drain_timeout",
                "candidate_identity": self._learner_completion_identity(),
            },
            durable=True,
        )

    def _stop_all(self) -> None:
        live = [worker for worker in self.workers.values() if worker.live]
        for worker in live:
            assert worker.process is not None
            _signal_process(worker.process, signal.SIGTERM)
            worker.state = "stopping"
        self._wait_for_exit(
            live,
            timeout=self.experiment.orchestration.shutdown.terminate_grace_seconds,
        )
        remaining = [worker for worker in live if worker.live]
        for worker in remaining:
            assert worker.process is not None
            _signal_process(worker.process, signal.SIGKILL)
            worker.state = "killing"
        self._wait_for_exit(
            remaining,
            timeout=self.experiment.orchestration.shutdown.kill_grace_seconds,
        )
        for worker in self.workers.values():
            if worker.process is not None:
                worker.last_exit_code = worker.process.poll()
            if worker.state not in ("completed", "exhausted"):
                worker.state = "stopped" if not worker.live else "unkillable"
            if worker.log_stream is not None:
                worker.log_stream.close()
                worker.log_stream = None

    def _wait_for_exit(
        self, workers: Sequence[ManagedWorker], *, timeout: float
    ) -> None:
        deadline = self.clock() + timeout
        while any(worker.live for worker in workers) and self.clock() < deadline:
            self.sleep(min(0.1, max(0.0, deadline - self.clock())))

    def _event(self, event: str, worker: ManagedWorker, **details: object) -> None:
        append_jsonl(
            self.metrics_path,
            {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "event": event,
                "worker": worker.spec.name,
                "role": worker.spec.role,
                "pid": worker.process.pid if worker.process is not None else None,
                "restart_count": worker.restart_count,
                **details,
            },
            durable=True,
        )

    def _write_status(self, *, final: bool = False) -> None:
        atomic_json(
            self.status_path,
            {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "coordinator_pid": os.getpid(),
                "state": "stopped" if final else "running",
                "workers": {
                    name: {
                        "role": worker.spec.role,
                        "gpu_ids": list(worker.spec.gpu_ids),
                        "pid": (
                            worker.process.pid
                            if worker.process is not None and worker.live
                            else None
                        ),
                        "state": worker.state,
                        "restart_count": worker.restart_count,
                        "last_exit_code": worker.last_exit_code,
                        "failure_reason": worker.failure_reason,
                        "last_progress": worker.last_progress,
                        "heartbeat": str(worker.spec.heartbeat_path),
                    }
                    for name, worker in sorted(self.workers.items())
                },
            },
        )


def _signal_process(process: ProcessProtocol, signal_number: int) -> None:
    if isinstance(process, subprocess.Popen):
        try:
            if signal_number == signal.SIGTERM:
                # Let the worker's SignalLatch unwind first. Signaling the whole
                # group here kills learner DataLoader children underneath the
                # iterator and turns an otherwise clean stop into a restartable
                # RuntimeError. The hard-stop path below still kills the group.
                os.kill(process.pid, signal_number)
            else:
                os.killpg(os.getpgid(process.pid), signal_number)
        except ProcessLookupError:
            return
    elif signal_number == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def orchestrate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Coordinate one learner and one actor per inference GPU"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=sys.executable)
    arguments = parser.parse_args(argv)

    experiment = load_config(arguments.config)
    directories = RunDirectories.from_experiment(experiment)
    specs = build_worker_specs(
        experiment,
        config_path=arguments.config,
        directories=directories,
        python_executable=arguments.python,
    )
    stop = SignalLatch()
    stop.install()
    exit_code = Coordinator(
        experiment=experiment,
        specs=specs,
        directories=directories,
    ).run(stop_requested=stop.is_set)
    if exit_code:
        raise SystemExit(exit_code)
