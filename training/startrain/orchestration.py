"""CUDA-free single-host process coordinator for learner and actor jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from .config import ExperimentConfig, load_config, parse_cpu_affinity
from .runtime import (
    RunIdentity,
    SignalLatch,
    SystemdNotifier,
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


def gpu_pause_ack_path(request_path: str | Path) -> Path:
    """Return the coordinator-owned acknowledgement path for a pause request."""

    request = Path(request_path)
    return request.with_name(f"{request.stem}.ack{request.suffix}")


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

    @property
    def autonomous_provenance(self) -> Path:
        return self.root / "autonomous-provenance.json"


def _autonomous_config_sha256(experiment: ExperimentConfig) -> str:
    encoded = json.dumps(
        experiment.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _autonomous_artifacts(directories: RunDirectories) -> tuple[Path, ...]:
    candidates = (
        directories.replay / "manifest.sqlite3",
        directories.learner / "candidate.json",
        directories.learner / "champion.json",
        directories.learner / "recovery.json",
        directories.learner / "resume-cutover.json",
    )
    discovered = []
    for root, patterns in (
        (directories.replay / "shards", ("*.npz",)),
        (directories.learner / "checkpoints", ("*.pt",)),
        (directories.learner / "manifests", ("*.json",)),
        (directories.arena, ("*.json",)),
    ):
        for pattern in patterns:
            discovered.extend(root.glob(pattern))
    return (*candidates, *discovered)


def validate_autonomous_run_root(
    experiment: ExperimentConfig,
    directories: RunDirectories,
) -> None:
    """Reject imported artifacts before creating a scratch run identity."""

    if not experiment.orchestration.autonomous.enabled:
        return
    if directories.run_identity.exists():
        return
    if directories.autonomous_provenance.exists():
        raise ValueError("autonomous provenance exists without a durable run identity")
    imported = [path for path in _autonomous_artifacts(directories) if path.exists()]
    if imported:
        rendered = ", ".join(str(path) for path in sorted(imported))
        raise ValueError(
            f"autonomous scratch run contains imported artifacts: {rendered}"
        )


def ensure_autonomous_provenance(
    experiment: ExperimentConfig,
    directories: RunDirectories,
    identity: RunIdentity,
) -> None:
    if not experiment.orchestration.autonomous.enabled:
        return
    expected = {
        "schema_version": 1,
        "mode": "random-init-selfplay-only",
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "train_seed": experiment.train.seed,
        "elo_anchor_step": experiment.orchestration.autonomous.elo_anchor_step,
        "external_weights": False,
        "external_replay": False,
        "external_positions": False,
        "config_sha256": _autonomous_config_sha256(experiment),
    }
    path = directories.autonomous_provenance
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read autonomous provenance: {exc}") from exc
        if payload != expected:
            raise ValueError(
                "autonomous provenance disagrees with the frozen run profile"
            )
        return
    if any(path.exists() for path in _autonomous_artifacts(directories)):
        raise ValueError("autonomous provenance is missing for an initialized run")
    atomic_json(path, expected)


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
    cpu_affinity: tuple[int, ...] | None = None


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


@dataclass(slots=True)
class PauseLease:
    token: str
    owner_pid: int
    requested_ns: int
    heartbeat_ns: int
    target_name: str
    target_role: str
    state: str
    request_state: str
    target_stop_requested: bool = False
    target_reaped: bool = False
    release_requested: bool = False
    owner_stop_requested: bool = False
    owner_reaped: bool = False
    failure_reason: str | None = None


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
    selfplay_manifest = (
        directories.learner / "selfplay" / "candidate.json"
        if experiment.learner.selfplay_snapshot_interval_examples is not None
        else candidate_manifest
    )
    champion_manifest = directories.learner / "champion.json"
    learner_gpus = orchestration.learner_gpus
    learner_threads = learner_gpus[0].cpu_threads
    learner_affinity_specs = {
        gpu.cpu_affinity for gpu in learner_gpus if gpu.cpu_affinity is not None
    }
    learner_affinity = (
        parse_cpu_affinity(next(iter(learner_affinity_specs)))
        if len(learner_affinity_specs) == 1
        else None
    )
    learner_environment = _worker_environment(
        environment,
        gpu_ids=tuple(gpu.gpu_id for gpu in learner_gpus),
        cpu_threads=learner_threads,
        cpu_affinity=learner_affinity,
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
    learner_pause_sharing = (
        orchestration.promotion.pause_sharing_mode
        and orchestration.promotion.gpu_id
        in {gpu.gpu_id for gpu in orchestration.learner_gpus}
    )
    if learner_pause_sharing:
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
            cpu_affinity=learner_affinity,
        )
    ]
    for gpu in orchestration.actor_gpus:
        affinity = (
            parse_cpu_affinity(gpu.cpu_affinity)
            if gpu.cpu_affinity is not None
            else None
        )
        for lane_id in range(gpu.actor_lanes):
            name = (
                f"actor-gpu-{gpu.gpu_id}"
                if gpu.actor_lanes == 1
                else f"actor-gpu-{gpu.gpu_id}-lane-{lane_id}"
            )
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
                        "--lane-id",
                        str(lane_id),
                        "--replay-store",
                        str(directories.replay),
                        "--manifest",
                        str(champion_manifest),
                        "--candidate-manifest",
                        str(selfplay_manifest),
                        "--run-identity",
                        str(directories.run_identity),
                        "--heartbeat",
                        str(directories.status / f"{name}.heartbeat.json"),
                        "--learner-heartbeat",
                        str(directories.status / "learner.heartbeat.json"),
                        "--metrics",
                        str(directories.metrics / f"{name}.jsonl"),
                        "--device",
                        "cuda",
                    ),
                    environment=_worker_environment(
                        environment,
                        gpu_ids=(gpu.gpu_id,),
                        cpu_threads=gpu.cpu_threads,
                        cpu_affinity=affinity,
                    ),
                    heartbeat_path=directories.status / f"{name}.heartbeat.json",
                    metrics_path=directories.metrics / f"{name}.jsonl",
                    log_path=directories.logs / f"{name}.log",
                    cpu_affinity=affinity,
                )
            )
    promotion = orchestration.promotion
    if promotion.enabled:
        name = "arena-promotion"
        promotion_gpu = next(
            (gpu for gpu in orchestration.gpus if gpu.gpu_id == promotion.gpu_id),
            None,
        )
        promotion_affinity = (
            parse_cpu_affinity(promotion_gpu.cpu_affinity)
            if promotion_gpu is not None and promotion_gpu.cpu_affinity is not None
            else None
        )
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
                    cpu_affinity=promotion_affinity,
                ),
                heartbeat_path=directories.status / f"{name}.heartbeat.json",
                metrics_path=directories.metrics / f"{name}.jsonl",
                log_path=directories.logs / f"{name}.log",
                cpu_affinity=promotion_affinity,
            )
        )
    return tuple(specs)


def _worker_environment(
    base: Mapping[str, str],
    *,
    gpu_ids: Sequence[int],
    cpu_threads: int,
    cpu_affinity: Sequence[int] | None = None,
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
    if cpu_affinity:
        output["STARTRAIN_CPU_AFFINITY"] = ",".join(str(cpu) for cpu in cpu_affinity)
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
        self.pause_request_path = directories.gpu_pause
        self.pause_ack_path = gpu_pause_ack_path(directories.gpu_pause)
        self.pause_lease: PauseLease | None = None
        self.pause_failed = False
        self.pause_target: ManagedWorker | None = None
        self.pause_owner: ManagedWorker | None = None
        if experiment.orchestration.promotion.pause_sharing_mode:
            shared_gpu = experiment.orchestration.promotion.gpu_id
            targets = [
                worker
                for worker in self.workers.values()
                if worker.spec.role in ("learner", "actor")
                and shared_gpu in worker.spec.gpu_ids
            ]
            owners = [
                worker
                for worker in self.workers.values()
                if worker.spec.role == "arena" and shared_gpu in worker.spec.gpu_ids
            ]
            if len(targets) != 1 or len(owners) != 1:
                raise ValueError(
                    "pause sharing requires one shared worker and one arena owner"
                )
            self.pause_target = targets[0]
            self.pause_owner = owners[0]

    def run(
        self,
        *,
        stop_requested: Callable[[], bool],
        max_monitor_cycles: int | None = None,
    ) -> int:
        notifier = SystemdNotifier()
        self.directories.create()
        validate_autonomous_run_root(self.experiment, self.directories)
        self.lock.acquire()
        exit_code = 0
        cycles = 0
        try:
            identity = load_or_create_run_identity(
                self.directories.run_identity,
                requested_run_id=self.experiment.orchestration.run_id,
            )
            ensure_autonomous_provenance(
                self.experiment,
                self.directories,
                identity,
            )
            if stop_requested():
                return 0
            for worker in self.workers.values():
                self._start_or_schedule(worker)
            notifier.ready("StarTrain coordinator is running")
            while not stop_requested():
                now = self.clock()
                self._reconcile_pause_lease(now)
                exhausted = self.pause_failed
                for worker in self.workers.values():
                    exhausted = self._monitor_worker(worker, now) or exhausted
                exhausted = self.pause_failed or exhausted
                self._write_status()
                notifier.watchdog(
                    "StarTrain coordinator healthy"
                    if not exhausted
                    else "StarTrain coordinator restarting after worker failure"
                )
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
            notifier.stopping("StarTrain coordinator is stopping")
            self.stopping = True
            self._stop_all()
            self._write_status(final=True)
            self.lock.release()

    def _monitor_worker(self, worker: ManagedWorker, now: float) -> bool:
        pause_learner_ready = (
            self.pause_lease is not None
            and worker is self.pause_target
            and worker.spec.role == "learner"
            and self.pause_lease.state == "ready"
        )
        health_failure = (
            self._heartbeat_failure(worker, now)
            if worker.state == "running" and not pause_learner_ready
            else None
        )
        if health_failure is not None:
            assert worker.process is not None
            if worker is self.pause_owner and self.pause_lease is not None:
                self._begin_pause_owner_recovery(
                    f"arena health failure: {health_failure}", now
                )
            else:
                _signal_process(worker.process, signal.SIGTERM)
                worker.state = "terminating"
                worker.failure_reason = health_failure
                worker.termination_deadline = (
                    now + self.experiment.orchestration.shutdown.terminate_grace_seconds
                )

        code = worker.process.poll() if worker.process is not None else None
        lease = self.pause_lease
        if (
            code is not None
            and lease is not None
            and worker is self.pause_target
            and (worker.state.startswith("pause_") or worker.spec.role == "learner")
        ):
            self._pause_target_exited(worker, code=code, now=now)
            return self.pause_failed
        if (
            code is not None
            and lease is not None
            and worker is self.pause_owner
            and worker.process is not None
            and worker.process.pid == lease.owner_pid
        ):
            self._pause_owner_exited(worker, code=code, now=now)
            return self.pause_failed

        if code is not None and worker.state in (
            "running",
            "terminating",
            "killing",
            "draining",
            "lease_terminating",
            "lease_killing",
        ):
            if (
                worker.spec.role == "learner"
                and code == 0
                and worker.state == "running"
            ):
                if self._learner_completion_identity() is not None:
                    worker.state = "completed"
                    worker.last_exit_code = code
                    self._close_worker_process(worker)
                    self._event("worker_completed", worker, exit_code=code)
                    self._begin_drain(now)
                    return False
                self._schedule_restart(worker, code=1, now=now)
                worker.failure_reason = "learner exited without completion marker"
                return False
            if worker.state == "draining" or (
                self.draining and worker.spec.role == "actor"
            ):
                worker.state = "drained"
                worker.last_exit_code = code
                self._close_worker_process(worker)
                return False
            self._schedule_restart(worker, code=code, now=now)
        elif worker.state in ("pause_terminating", "lease_terminating") and (
            now >= worker.termination_deadline
        ):
            assert worker.process is not None
            _signal_process(worker.process, signal.SIGKILL)
            worker.state = (
                "pause_killing"
                if worker.state == "pause_terminating"
                else "lease_killing"
            )
            worker.termination_deadline = (
                now + self.experiment.orchestration.shutdown.kill_grace_seconds
            )
            self._pause_event(
                "pause_hard_kill_requested",
                worker=worker.spec.name,
                reason=worker.failure_reason,
            )
        elif worker.state in ("pause_killing", "lease_killing") and (
            now >= worker.termination_deadline
        ):
            worker.state = "exhausted"
            worker.failure_reason = "pause participant ignored SIGKILL"
            self._fail_pause_lease(worker.failure_reason)
            return True
        elif worker.state == "terminating" and now >= worker.termination_deadline:
            assert worker.process is not None
            _signal_process(worker.process, signal.SIGKILL)
            worker.state = "killing"
            worker.termination_deadline = (
                now + self.experiment.orchestration.shutdown.kill_grace_seconds
            )
        elif worker.state == "killing" and now >= worker.termination_deadline:
            worker.state = "exhausted"
            worker.failure_reason = "worker ignored SIGKILL"
            return True

        if worker.state == "backoff" and now >= worker.next_start_at:
            if self.draining and worker.spec.role == "actor":
                worker.state = "drained"
            elif worker is self.pause_owner and self.pause_lease is not None:
                return False
            elif (
                worker.restart_count
                > self.experiment.orchestration.restart.max_restarts
            ):
                worker.state = "exhausted"
                return True
            else:
                self._start_or_schedule(worker)
        return False

    def _reconcile_pause_lease(self, now: float) -> None:
        if self.pause_target is None or self.pause_owner is None:
            return
        payload = self._read_pause_request()
        lease = self.pause_lease
        if lease is None:
            if payload is None:
                return
            parsed = self._parse_pause_request(payload)
            if parsed is None:
                self.pause_request_path.unlink(missing_ok=True)
                self._pause_event("pause_request_invalid")
                return
            if parsed.request_state != "requested":
                return
            if self._pause_request_stale(parsed):
                self._write_pause_ack(
                    parsed,
                    state="failed",
                    reason="pause request heartbeat is stale",
                )
                self._pause_event(
                    "pause_request_rejected",
                    token=parsed.token,
                    reason="stale heartbeat",
                )
                return
            owner = self.pause_owner
            if (
                owner.process is None
                or owner.process.pid != parsed.owner_pid
                or not owner.live
            ):
                self._write_pause_ack(
                    parsed,
                    state="failed",
                    reason="pause request owner is not the supervised arena",
                )
                self._pause_event(
                    "pause_request_rejected",
                    token=parsed.token,
                    reason="owner mismatch",
                )
                return
            self.pause_lease = parsed
            self._begin_pause_lease(parsed, now)
            return

        if payload is None:
            self._begin_pause_owner_recovery("pause request disappeared", now)
            return
        parsed = self._parse_pause_request(payload)
        if parsed is None or parsed.token != lease.token:
            self._begin_pause_owner_recovery("pause lease token changed", now)
            return
        lease.heartbeat_ns = parsed.heartbeat_ns
        lease.request_state = parsed.request_state
        if parsed.request_state in ("released", "cancelled"):
            if not lease.release_requested:
                lease.release_requested = True
                lease.failure_reason = (
                    "request cancelled before ready"
                    if parsed.request_state == "cancelled"
                    else None
                )
                self._pause_event(
                    "pause_lease_release_requested",
                    token=lease.token,
                    request_state=parsed.request_state,
                )
            self._finish_pause_release(now)
            return
        if self._pause_request_stale(parsed):
            self._begin_pause_owner_recovery("pause lease heartbeat is stale", now)
            return
        if (
            self.pause_owner.process is None
            or self.pause_owner.process.pid != lease.owner_pid
            or not self.pause_owner.live
        ):
            self._begin_pause_owner_recovery("pause lease owner exited", now)
            return
        if lease.target_role == "learner" and lease.state == "waiting":
            self._maybe_ack_learner_pause(lease)

    def _read_pause_request(self) -> dict[str, object] | None:
        try:
            with self.pause_request_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _parse_pause_request(self, payload: Mapping[str, object]) -> PauseLease | None:
        token = payload.get("token")
        owner_pid = payload.get("pid")
        requested_ns = payload.get("requested_ns")
        heartbeat_ns = payload.get("heartbeat_ns")
        state = payload.get("state")
        promotion = self.experiment.orchestration.promotion
        valid = (
            payload.get("schema_version") == 1
            and payload.get("protocol") == "coordinator-pause-v1"
            and isinstance(token, str)
            and 8 <= len(token) <= 128
            and isinstance(owner_pid, int)
            and not isinstance(owner_pid, bool)
            and owner_pid > 0
            and isinstance(requested_ns, int)
            and not isinstance(requested_ns, bool)
            and requested_ns > 0
            and isinstance(heartbeat_ns, int)
            and not isinstance(heartbeat_ns, bool)
            and heartbeat_ns >= requested_ns
            and state in ("requested", "active", "released", "cancelled")
            and payload.get("gpu_id") == promotion.gpu_id
        )
        if not valid:
            return None
        assert isinstance(token, str)
        assert isinstance(owner_pid, int)
        assert isinstance(requested_ns, int)
        assert isinstance(heartbeat_ns, int)
        assert isinstance(state, str)
        assert self.pause_target is not None
        return PauseLease(
            token=token,
            owner_pid=owner_pid,
            requested_ns=requested_ns,
            heartbeat_ns=heartbeat_ns,
            target_name=self.pause_target.spec.name,
            target_role=self.pause_target.spec.role,
            state="requested",
            request_state=state,
        )

    def _pause_request_stale(self, lease: PauseLease) -> bool:
        stale_ns = int(
            self.experiment.orchestration.shutdown.stale_heartbeat_seconds
            * 1_000_000_000
        )
        return time.time_ns() - lease.heartbeat_ns > stale_ns

    def _begin_pause_lease(self, lease: PauseLease, now: float) -> None:
        assert self.pause_target is not None
        target = self.pause_target
        self._pause_event(
            "pause_lease_requested",
            token=lease.token,
            owner_pid=lease.owner_pid,
            target=target.spec.name,
            gpu_id=self.experiment.orchestration.promotion.gpu_id,
        )
        if target.spec.role == "learner" and target.live:
            lease.state = "waiting"
            self._maybe_ack_learner_pause(lease)
            return
        if target.live:
            assert target.process is not None
            _signal_process(target.process, signal.SIGTERM)
            target.state = "pause_terminating"
            target.failure_reason = f"GPU pause lease {lease.token}"
            target.termination_deadline = (
                now + self.experiment.orchestration.shutdown.terminate_grace_seconds
            )
            lease.state = "stopping"
            lease.target_stop_requested = True
            self._pause_event(
                "pause_target_stop_requested",
                token=lease.token,
                target=target.spec.name,
                pid=target.process.pid,
            )
            return

        code = target.process.poll() if target.process is not None else None
        if target.process is not None and code is None:
            self._fail_pause_lease("shared worker state is inconsistent")
            return
        if code is not None:
            target.last_exit_code = code
        self._close_worker_process(target)
        target.state = "paused"
        lease.target_reaped = True
        self._mark_pause_ready(lease)

    def _maybe_ack_learner_pause(self, lease: PauseLease) -> None:
        assert self.pause_target is not None
        target = self.pause_target
        if not target.live:
            lease.target_reaped = True
            target.state = "paused"
            self._close_worker_process(target)
            self._mark_pause_ready(lease)
            return
        try:
            with target.spec.heartbeat_path.open("r", encoding="utf-8") as stream:
                heartbeat = json.load(stream)
        except (OSError, json.JSONDecodeError):
            return
        progress_ns = heartbeat.get("progress_ns") if isinstance(heartbeat, dict) else 0
        if (
            isinstance(heartbeat, dict)
            and heartbeat.get("phase") == "arena_gpu_pause"
            and isinstance(progress_ns, int)
            and not isinstance(progress_ns, bool)
            and progress_ns >= lease.requested_ns
        ):
            self._mark_pause_ready(lease)

    def _mark_pause_ready(self, lease: PauseLease) -> None:
        if lease.state == "ready" or lease.release_requested:
            return
        if lease.target_role == "actor" and not lease.target_reaped:
            raise RuntimeError("actor pause cannot be acknowledged before process exit")
        lease.state = "ready"
        self._write_pause_ack(lease, state="ready")
        self._pause_event(
            "pause_lease_ready",
            token=lease.token,
            target=lease.target_name,
            target_role=lease.target_role,
        )

    def _pause_target_exited(
        self, worker: ManagedWorker, *, code: int, now: float
    ) -> None:
        assert self.pause_lease is not None
        lease = self.pause_lease
        worker.last_exit_code = code
        self._close_worker_process(worker)
        lease.target_reaped = True
        learner_completed = (
            worker.spec.role == "learner"
            and code == 0
            and self._learner_completion_identity() is not None
        )
        worker.state = "completed" if learner_completed else "paused"
        self._pause_event(
            "pause_target_reaped",
            token=lease.token,
            target=worker.spec.name,
            exit_code=code,
            restart_count=worker.restart_count,
        )
        if learner_completed:
            self._event("worker_completed", worker, exit_code=code)
            self._begin_drain(now)
        if lease.release_requested or lease.owner_stop_requested:
            self._finish_pause_release(now)
        else:
            self._mark_pause_ready(lease)

    def _begin_pause_owner_recovery(self, reason: str, now: float) -> None:
        lease = self.pause_lease
        if lease is None or lease.owner_stop_requested:
            return
        lease.release_requested = True
        lease.owner_stop_requested = True
        lease.failure_reason = reason
        self._pause_event(
            "pause_lease_stale",
            token=lease.token,
            reason=reason,
        )
        assert self.pause_owner is not None
        owner = self.pause_owner
        if (
            owner.process is not None
            and owner.process.pid == lease.owner_pid
            and owner.process.poll() is None
        ):
            _signal_process(owner.process, signal.SIGTERM)
            owner.state = "lease_terminating"
            owner.failure_reason = reason
            owner.termination_deadline = (
                now + self.experiment.orchestration.shutdown.terminate_grace_seconds
            )
            self._pause_event(
                "pause_owner_stop_requested",
                token=lease.token,
                pid=owner.process.pid,
                reason=reason,
            )
            return
        if (
            owner.process is not None
            and owner.process.pid == lease.owner_pid
            and owner.process.poll() is not None
        ):
            code = owner.process.returncode
            assert code is not None
            self._pause_owner_exited(owner, code=code, now=now)
            return
        lease.owner_reaped = True
        self._finish_pause_release(now)

    def _pause_owner_exited(
        self, worker: ManagedWorker, *, code: int, now: float
    ) -> None:
        assert self.pause_lease is not None
        lease = self.pause_lease
        lease.owner_reaped = True
        lease.owner_stop_requested = True
        lease.release_requested = True
        if lease.failure_reason is None:
            lease.failure_reason = f"arena owner exited with code {code}"
        self._pause_event(
            "pause_owner_reaped",
            token=lease.token,
            exit_code=code,
            reason=lease.failure_reason,
        )
        self._schedule_restart(worker, code=code, now=now)
        self._finish_pause_release(now)

    def _finish_pause_release(self, now: float) -> None:
        lease = self.pause_lease
        if lease is None or not lease.release_requested:
            return
        if lease.owner_stop_requested and not lease.owner_reaped:
            return
        assert self.pause_target is not None
        target = self.pause_target
        if target.spec.role == "learner" and target.live:
            self.pause_request_path.unlink(missing_ok=True)
            state = "draining" if self.draining else "released"
            self._write_pause_ack(
                lease,
                state=state,
                reason=lease.failure_reason,
            )
            self._pause_event(
                "pause_lease_released",
                token=lease.token,
                target=target.spec.name,
                restart=False,
                outcome=state,
            )
            self.pause_lease = None
            return
        if not lease.target_reaped:
            return
        if self.draining or self.stopping:
            target.state = "drained"
            if lease.failure_reason is None:
                target.failure_reason = None
            if target.spec.role == "learner":
                self.pause_request_path.unlink(missing_ok=True)
            self._write_pause_ack(
                lease,
                state="draining",
                reason=lease.failure_reason,
            )
            self._pause_event(
                "pause_lease_released",
                token=lease.token,
                target=target.spec.name,
                restart=False,
                outcome="draining",
            )
            self.pause_lease = None
            return
        try:
            self._start(target)
        except OSError as error:
            self._close_worker_process(target)
            target.state = "exhausted"
            target.failure_reason = f"pause restart failed: {error}"
            self._fail_pause_lease(target.failure_reason)
            return
        if target.spec.role == "learner":
            self.pause_request_path.unlink(missing_ok=True)
        state = "recovered" if lease.failure_reason is not None else "released"
        self._write_pause_ack(
            lease,
            state=state,
            reason=lease.failure_reason,
        )
        self._pause_event(
            "pause_target_restarted",
            token=lease.token,
            target=target.spec.name,
            pid=target.process.pid if target.process is not None else None,
            restart_count=target.restart_count,
            outcome=state,
        )
        self.pause_lease = None

    def _fail_pause_lease(self, reason: str) -> None:
        lease = self.pause_lease
        self.pause_failed = True
        if lease is not None:
            lease.failure_reason = reason
            self._write_pause_ack(lease, state="failed", reason=reason)
            self._pause_event(
                "pause_lease_failed",
                token=lease.token,
                reason=reason,
            )

    def _write_pause_ack(
        self,
        lease: PauseLease,
        *,
        state: str,
        reason: str | None = None,
    ) -> None:
        atomic_json(
            self.pause_ack_path,
            {
                "schema_version": 1,
                "protocol": "coordinator-pause-v1",
                "token": lease.token,
                "state": state,
                "gpu_id": self.experiment.orchestration.promotion.gpu_id,
                "target_worker": lease.target_name,
                "target_role": lease.target_role,
                "coordinator_pid": os.getpid(),
                "ack_ns": time.time_ns(),
                "reason": reason,
            },
        )

    def _pause_event(self, event: str, **details: object) -> None:
        append_jsonl(
            self.metrics_path,
            {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "event": event,
                **details,
            },
            durable=True,
        )

    @staticmethod
    def _close_worker_process(worker: ManagedWorker) -> None:
        worker.process = None
        if worker.log_stream is not None:
            worker.log_stream.close()
            worker.log_stream = None

    def _start(self, worker: ManagedWorker) -> None:
        if worker.live:
            raise RuntimeError(f"refusing to duplicate live worker {worker.spec.name}")
        if worker.log_stream is not None:
            worker.log_stream.close()
        worker.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        worker.log_stream = worker.spec.log_path.open(
            "a", encoding="utf-8", buffering=1
        )
        command = list(worker.spec.command)
        if worker.spec.cpu_affinity is not None:
            taskset = shutil.which(
                "taskset",
                path=worker.spec.environment.get("PATH"),
            )
            if taskset is None:
                raise OSError("configured CPU affinity requires the taskset executable")
            cpu_list = ",".join(str(cpu) for cpu in worker.spec.cpu_affinity)
            command = [taskset, "--cpu-list", cpu_list, *command]
        process = self.process_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=worker.log_stream,
            stderr=subprocess.STDOUT,
            env=dict(worker.spec.environment),
            close_fds=True,
            start_new_session=True,
        )
        worker.process = process
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
        if self.draining and worker.spec.role == "actor":
            worker.last_exit_code = code
            worker.state = "drained"
            self._close_worker_process(worker)
            self._event("worker_drained", worker, exit_code=code)
            return
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
            if worker.spec.role != "actor":
                continue
            if worker is self.pause_target and self.pause_lease is not None:
                if worker.live and not worker.state.startswith("pause_"):
                    assert worker.process is not None
                    _signal_process(worker.process, signal.SIGTERM)
                    worker.state = "pause_terminating"
                    worker.termination_deadline = (
                        now
                        + self.experiment.orchestration.shutdown.terminate_grace_seconds
                    )
                    self.pause_lease.target_stop_requested = True
                continue
            if worker.live:
                assert worker.process is not None
                _signal_process(worker.process, signal.SIGTERM)
                worker.state = "draining"
            elif worker.state in ("backoff", "pending"):
                worker.state = "drained"
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
        if self.pause_lease is not None:
            self._write_pause_ack(
                self.pause_lease,
                state="failed" if self.pause_failed else "stopping",
                reason=(
                    self.pause_lease.failure_reason
                    if self.pause_failed
                    else "coordinator shutdown takes precedence over pause restart"
                ),
            )
            self._pause_event(
                "pause_lease_shutdown",
                token=self.pause_lease.token,
            )
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
                "draining": self.draining,
                "pause_sharing": (
                    {
                        "token": self.pause_lease.token,
                        "state": self.pause_lease.state,
                        "request_state": self.pause_lease.request_state,
                        "target_worker": self.pause_lease.target_name,
                        "target_role": self.pause_lease.target_role,
                        "owner_pid": self.pause_lease.owner_pid,
                        "heartbeat_ns": self.pause_lease.heartbeat_ns,
                        "release_requested": self.pause_lease.release_requested,
                        "failure_reason": self.pause_lease.failure_reason,
                    }
                    if self.pause_lease is not None
                    else None
                ),
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
