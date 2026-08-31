"""Autocast/compile-aware single-step training utilities."""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from .checkpoint import ExponentialMovingAverage
from .config import SchedulerConfig
from .contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from .device import resolve_compile
from .features import EncodedBatch
from .losses import LossWeights, compute_losses
from .optim import (
    OptimizerGroupDiagnostics,
    OptimizerStepDiagnostics,
    capture_optimizer_diagnostic_snapshot,
    finalize_optimizer_step_diagnostics,
)
from .replay import ReplayBatch

COMPILE_CACHE_SCHEMA_VERSION = 1
_COMPILE_CACHE_ENVIRONMENT = (
    "HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "TRITON_OVERRIDE_DIR",
    "XDG_CACHE_HOME",
    "CUDA_CACHE_PATH",
)
_UNSAFE_COMPILE_ENVIRONMENT = (
    "TRITON_CACHE_MANAGER",
    "TRITON_REMOTE_CACHE_BACKEND",
    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_CACHE",
    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO",
    "TORCH_COMPILE_JOB_ID",
    "TORCH_COMPILE_FORCE_DISABLE_CACHES",
    "CUDA_CACHE_DISABLE",
)
_COMPILE_CONTROL_PREFIXES = (
    "TORCHINDUCTOR_",
    "TORCH_COMPILE_",
    "TORCH_DYNAMO_",
    "TRITON_",
    "CUDA_CACHE_",
)


class NonFiniteTrainingError(FloatingPointError):
    """Fatal non-finite step carrying counters for durable reporting."""

    def __init__(
        self,
        *,
        nonfinite_loss_count: int,
        nonfinite_gradient_count: int,
    ) -> None:
        super().__init__(
            "non-finite training loss or gradient norm on at least one rank"
        )
        self.nonfinite_loss_count = nonfinite_loss_count
        self.nonfinite_gradient_count = nonfinite_gradient_count


@dataclass(frozen=True, slots=True)
class CompileCacheProvenance:
    """Pinned, per-invocation compile-cache identity."""

    root: Path
    environment: dict[str, str]
    runtime: dict[str, str | None]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": COMPILE_CACHE_SCHEMA_VERSION,
            "layout": "startrain-isolated-compile-cache-v1",
            "root": str(self.root),
            "owner_marker": str(self.root / "cache-owner.json"),
            "environment": dict(sorted(self.environment.items())),
            "required_unset_environment": list(_UNSAFE_COMPILE_ENVIRONMENT),
            "rejected_environment_prefixes": list(_COMPILE_CONTROL_PREFIXES),
            "runtime": dict(sorted(self.runtime.items())),
        }


def _secure_cache_directory(path: Path) -> None:
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise ValueError(f"compile cache path is not a regular directory: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"compile cache path became unsafe: {path}")
    path.chmod(0o700)


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"compile cache path contains a symlink: {component}")


def configure_isolated_compile_cache(
    output_dir: str | Path,
) -> CompileCacheProvenance:
    """Configure writable Inductor/Triton caches below one arm output."""

    source = Path(output_dir).expanduser()
    _reject_symlink_components(source)
    inherited = {
        name: value
        for name, value in os.environ.items()
        if value
        and name not in _COMPILE_CACHE_ENVIRONMENT
        and (
            name in _UNSAFE_COMPILE_ENVIRONMENT
            or name.startswith(_COMPILE_CONTROL_PREFIXES)
        )
    }
    if inherited:
        raise ValueError(
            "unsupported inherited compiler controls: " + ", ".join(sorted(inherited))
        )
    output = source.resolve(strict=False)
    cache_root = output / "compile-cache" / f"v{COMPILE_CACHE_SCHEMA_VERSION}"
    try:
        cache_root.relative_to(output)
    except ValueError as error:  # pragma: no cover - defensive path invariant
        raise ValueError("compile cache escaped its arm output") from error

    directories = {
        "HOME": cache_root / "home",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "inductor",
        "TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR": cache_root / "inductor-autotune",
        "TRITON_HOME": cache_root / "triton-home",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "TRITON_DUMP_DIR": cache_root / "triton-dump",
        "TRITON_OVERRIDE_DIR": cache_root / "triton-override",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "CUDA_CACHE_PATH": cache_root / "cuda",
    }
    for path in (output, output / "compile-cache", cache_root, *directories.values()):
        _secure_cache_directory(path)

    sentinel = cache_root / f".write-test-{os.getpid()}-{time.time_ns()}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            sentinel,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.write(descriptor, b"startrain-compile-cache-v1\n")
        os.fsync(descriptor)
    except OSError as error:
        raise ValueError(
            f"compile cache is not writable: {cache_root}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        sentinel.unlink(missing_ok=True)

    environment = {name: str(path) for name, path in directories.items()}
    os.environ.update(environment)
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    return CompileCacheProvenance(
        root=cache_root,
        environment=environment,
        runtime={
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda,
            "triton_version": triton_version,
        },
    )


@contextmanager
def isolated_compile_cache(
    output_dir: str | Path,
) -> Iterator[CompileCacheProvenance]:
    """Temporarily bind all compile caches to one arm-owned directory."""

    previous = {name: os.environ.get(name) for name in _COMPILE_CACHE_ENVIRONMENT}
    try:
        provenance = configure_isolated_compile_cache(output_dir)
        yield provenance
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True, slots=True)
class HostTrainStepMetrics:
    """Host metrics with finite-safe clipping diagnostics.

    ``gradient_norm`` remains the pre-clip global norm. Clip severity is the
    fraction removed (``1 - coefficient``), while clip ratio is the pre-clip
    norm divided by the configured threshold.
    """

    losses: dict[str, float]
    gradient_norm: float
    gradient_clipped: bool
    nonfinite_loss_count: int
    nonfinite_gradient_count: int
    learning_rates: tuple[float, ...]
    optimizer_groups: tuple[OptimizerGroupDiagnostics, ...]
    scheduler: "SchedulerDiagnostics | None"
    ema: "EMADiagnostics | None"
    gradient_post_clip_norm: float | None = None
    gradient_clip_threshold: float | None = None
    gradient_clip_coefficient: float | None = None
    gradient_clip_severity: float | None = None
    gradient_clip_ratio: float | None = None

    @property
    def gradient_pre_clip_norm(self) -> float:
        """Backward-compatible pre-clip global norm."""

        return self.gradient_norm


@dataclass(frozen=True, slots=True)
class SchedulerDiagnostics:
    age_steps: int
    segment: str
    segment_step: int
    segment_length_steps: int | None
    segment_position: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "age_steps": self.age_steps,
            "segment": self.segment,
            "segment_step": self.segment_step,
            "segment_length_steps": self.segment_length_steps,
            "segment_position": self.segment_position,
        }


@dataclass(frozen=True, slots=True)
class EMADiagnostics:
    decay: float
    num_updates: int
    raw_norm: float
    ema_norm: float
    distance_norm: float
    relative_distance: float | None
    effective_turnover: float

    def as_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "raw_norm": self.raw_norm,
            "ema_norm": self.ema_norm,
            "distance_norm": self.distance_norm,
            "relative_distance": self.relative_distance,
            "effective_turnover": self.effective_turnover,
        }


@dataclass(frozen=True, slots=True)
class _EMADiagnosticTensors:
    decay: float
    num_updates: int
    raw_norm: Tensor
    ema_norm: Tensor
    distance_norm: Tensor

    def to_host(self) -> EMADiagnostics:
        raw_norm, ema_norm, distance_norm = (
            torch.stack(
                (
                    self.raw_norm.detach().float(),
                    self.ema_norm.detach().float(),
                    self.distance_norm.detach().float(),
                )
            )
            .cpu()
            .tolist()
        )
        return EMADiagnostics(
            decay=self.decay,
            num_updates=self.num_updates,
            raw_norm=float(raw_norm),
            ema_norm=float(ema_norm),
            distance_norm=float(distance_norm),
            relative_distance=(
                float(distance_norm / raw_norm) if raw_norm > 0 else None
            ),
            effective_turnover=ema_effective_turnover(self.decay, self.num_updates),
        )


@dataclass(frozen=True, slots=True)
class IntervalTrainMetrics:
    steps: int
    gradient_clipped_steps: int
    gradient_clipping_frequency: float
    nonfinite_loss_count: int
    nonfinite_gradient_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "gradient_clipped_steps": self.gradient_clipped_steps,
            "gradient_clipping_frequency": self.gradient_clipping_frequency,
            "nonfinite_loss_count": self.nonfinite_loss_count,
            "nonfinite_gradient_count": self.nonfinite_gradient_count,
        }


class TrainMetricAccumulator:
    """Accumulate cheap device scalars without synchronizing each train step."""

    def __init__(self) -> None:
        self.steps = 0
        self._counts: Tensor | None = None

    def update(self, result: "TrainStepResult") -> None:
        values = torch.stack(
            (
                result.gradient_clipped_tensor.detach(),
                result.nonfinite_loss_count_tensor.detach().bool(),
                result.nonfinite_gradient_count_tensor.detach().bool(),
            )
        ).to(dtype=torch.int64)
        if self._counts is None:
            self._counts = values
        else:
            self._counts.add_(values)
        self.steps += 1

    def to_host(self) -> IntervalTrainMetrics:
        if self.steps == 0:
            return IntervalTrainMetrics(0, 0, 0.0, 0, 0)
        assert self._counts is not None
        clipped, nonfinite_loss, nonfinite_gradient = self._counts.cpu().tolist()
        return IntervalTrainMetrics(
            steps=self.steps,
            gradient_clipped_steps=int(clipped),
            gradient_clipping_frequency=float(clipped / self.steps),
            nonfinite_loss_count=int(nonfinite_loss),
            nonfinite_gradient_count=int(nonfinite_gradient),
        )

    def reset(self) -> None:
        self.steps = 0
        self._counts = None


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    """Device-resident step results, synchronized only when metrics are requested."""

    loss_tensors: dict[str, torch.Tensor]
    gradient_norm_tensor: torch.Tensor
    gradient_clipped_tensor: torch.Tensor
    nonfinite_loss_count_tensor: torch.Tensor
    nonfinite_gradient_count_tensor: torch.Tensor
    learning_rates: tuple[float, ...]
    optimizer_diagnostics: OptimizerStepDiagnostics | None = None
    scheduler_diagnostics: SchedulerDiagnostics | None = None
    ema_diagnostics: _EMADiagnosticTensors | None = None
    gradient_clip_threshold: float | None = None

    def to_host(self) -> HostTrainStepMetrics:
        names = tuple(self.loss_tensors)
        tensors = [
            *(self.loss_tensors[name].detach().float() for name in names),
            self.gradient_norm_tensor.detach().float(),
            self.gradient_clipped_tensor.detach().float(),
            self.nonfinite_loss_count_tensor.detach().float(),
            self.nonfinite_gradient_count_tensor.detach().float(),
        ]
        post_clip_index: int | None = None
        coefficient_index: int | None = None
        if self.gradient_clip_threshold is not None:
            # Match torch.nn.utils.clip_grad_norm_ exactly. Deriving the norm avoids
            # a second reduction over gradients and cannot perturb the optimizer.
            coefficient = torch.clamp(
                self.gradient_clip_threshold
                / (self.gradient_norm_tensor.detach() + 1e-6),
                max=1.0,
            )
            post_clip_index = len(tensors)
            tensors.append((self.gradient_norm_tensor.detach() * coefficient).float())
            coefficient_index = len(tensors)
            tensors.append(coefficient.float())
        host_values = torch.stack(tensors).cpu().tolist()
        loss_count = len(names)
        gradient_norm = float(host_values[loss_count])
        threshold = (
            float(self.gradient_clip_threshold)
            if self.gradient_clip_threshold is not None
            else None
        )
        if threshold is not None and (not math.isfinite(threshold) or threshold <= 0):
            threshold = None
        post_clip_norm = (
            float(host_values[post_clip_index]) if post_clip_index is not None else None
        )
        if post_clip_norm is not None and (
            not math.isfinite(post_clip_norm) or post_clip_norm < 0
        ):
            post_clip_norm = None
        clip_coefficient = (
            float(host_values[coefficient_index])
            if coefficient_index is not None
            else None
        )
        if clip_coefficient is not None and (
            not math.isfinite(clip_coefficient) or not 0 <= clip_coefficient <= 1
        ):
            clip_coefficient = None
        clip_severity = 1.0 - clip_coefficient if clip_coefficient is not None else None
        clip_ratio = (
            gradient_norm / threshold
            if threshold is not None
            and math.isfinite(gradient_norm)
            and gradient_norm >= 0
            else None
        )
        if clip_ratio is not None and not math.isfinite(clip_ratio):
            clip_ratio = None
        return HostTrainStepMetrics(
            losses=dict(zip(names, host_values[:loss_count], strict=True)),
            gradient_norm=gradient_norm,
            gradient_clipped=bool(host_values[loss_count + 1]),
            nonfinite_loss_count=int(host_values[loss_count + 2]),
            nonfinite_gradient_count=int(host_values[loss_count + 3]),
            learning_rates=self.learning_rates,
            optimizer_groups=(
                self.optimizer_diagnostics.to_host()
                if self.optimizer_diagnostics is not None
                else ()
            ),
            scheduler=self.scheduler_diagnostics,
            ema=(
                self.ema_diagnostics.to_host()
                if self.ema_diagnostics is not None
                else None
            ),
            gradient_post_clip_norm=post_clip_norm,
            gradient_clip_threshold=threshold,
            gradient_clip_coefficient=clip_coefficient,
            gradient_clip_severity=clip_severity,
            gradient_clip_ratio=clip_ratio,
        )

    @property
    def losses(self) -> dict[str, float]:
        return self.to_host().losses

    @property
    def gradient_norm(self) -> float:
        return self.to_host().gradient_norm

    @property
    def gradient_pre_clip_norm(self) -> float:
        return self.gradient_norm


class DeviceBatchPrefetcher(Iterator[ReplayBatch]):
    """Move pinned batches on a dedicated CUDA stream ahead of training."""

    def __init__(
        self,
        batches: Iterable[ReplayBatch],
        *,
        device: torch.device | str,
        enabled: bool = True,
    ) -> None:
        self._batches = iter(batches)
        self.device = torch.device(device)
        self._enabled = enabled and self.device.type == "cuda"
        self._suspended = False
        self._closed = False
        self._stream = torch.cuda.Stream(device=self.device) if self._enabled else None
        self._consumed_copy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._next_copy_event: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        self._topology_cache: dict[
            tuple[int, int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._next_batch: ReplayBatch | None = None
        self._next_source: ReplayBatch | None = None
        if self._stream is not None:
            self._preload()

    def __iter__(self) -> "DeviceBatchPrefetcher":
        return self

    def __next__(self) -> ReplayBatch:
        if getattr(self, "_closed", False):
            raise RuntimeError("device prefetcher is closed")
        if getattr(self, "_suspended", False):
            raise RuntimeError("device prefetcher is suspended")
        if self._stream is None:
            source = next(self._batches)
            return source.to(self.device)
        if self._next_batch is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self._stream)
        batch = self._next_batch
        batch.record_stream(current_stream)
        if self._next_copy_event is not None:
            self._consumed_copy_events.append(self._next_copy_event)
        self._next_batch = None
        self._next_source = None
        self._next_copy_event = None
        self._preload()
        return batch

    def _preload(self) -> None:
        assert self._stream is not None
        try:
            source = next(self._batches)
        except StopIteration:
            self._next_batch = None
            self._next_source = None
            self._next_copy_event = None
            return
        self._stage_source(source)

    def _stage_source(self, source: ReplayBatch) -> None:
        assert self._stream is not None
        self._next_source = source
        with torch.cuda.stream(self._stream):
            started = torch.cuda.Event(enable_timing=True)
            completed = torch.cuda.Event(enable_timing=True)
            started.record(self._stream)
            self._next_batch = self._to_device(source)
            completed.record(self._stream)
            self._next_copy_event = (started, completed)

    @property
    def suspended(self) -> bool:
        return self._suspended

    def suspend_device(self) -> None:
        """Release all CUDA-owned prefetch state without advancing the iterator."""

        if self._closed or self._suspended:
            return
        if self._stream is not None:
            self._stream.synchronize()
        self._next_batch = None
        self._next_copy_event = None
        self._consumed_copy_events.clear()
        self._topology_cache.clear()
        self._stream = None
        self._suspended = True

    def resume_device(self) -> None:
        """Restore CUDA prefetch state for the exact retained CPU source batch."""

        if self._closed:
            raise RuntimeError("cannot resume a closed device prefetcher")
        if not self._suspended:
            return
        self._suspended = False
        if not self._enabled:
            return
        self._stream = torch.cuda.Stream(device=self.device)
        source = self._next_source
        if source is None:
            self._preload()
        else:
            self._stage_source(source)

    def close(
        self,
        *,
        strict: bool = True,
        shutdown_workers: bool = True,
    ) -> BaseException | None:
        """Release CUDA state and stop the exact iterator owned by this prefetcher."""

        if self._closed:
            return None
        failure: BaseException | None = None
        try:
            if self._stream is not None:
                self._stream.synchronize()
        except BaseException as exc:
            failure = exc
        self._next_batch = None
        self._next_source = None
        self._next_copy_event = None
        self._consumed_copy_events.clear()
        self._topology_cache.clear()
        self._stream = None
        if shutdown_workers:
            try:
                shutdown = getattr(self._batches, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        self._closed = True
        self._suspended = False
        if strict and failure is not None:
            raise failure
        return failure

    def pop_copy_events(self) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
        """Transfer ownership of events for batches already yielded."""

        events, self._consumed_copy_events = self._consumed_copy_events, []
        return events

    def pop_copy_seconds(self) -> float:
        """Return yielded-batch copy time after the caller synchronizes."""

        events = self.pop_copy_events()
        return sum(start.elapsed_time(end) for start, end in events) / 1_000.0

    def _to_device(self, source: ReplayBatch) -> ReplayBatch:
        inputs = source.inputs
        ring = int(inputs.rings[0])
        if not bool((inputs.rings == ring).all()):
            return source.to(self.device, non_blocking=True)
        batch_size = inputs.batch_size
        key = (
            ring,
            batch_size,
            inputs.max_nodes,
            int(inputs.neighbor_index.shape[-1]),
        )
        topology = self._topology_cache.get(key)
        if topology is None:
            topology = (
                inputs.neighbor_index[0]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .contiguous(),
                inputs.neighbor_mask[0]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .contiguous(),
                inputs.neighbor_edge_type[0]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .contiguous(),
                inputs.node_mask[0]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .contiguous(),
                inputs.rings.to(self.device, non_blocking=True),
            )
            self._topology_cache[key] = topology
        (
            neighbor_index,
            neighbor_mask,
            neighbor_edge_type,
            node_mask,
            rings,
        ) = topology
        encoded = EncodedBatch(
            node_features=inputs.node_features.to(self.device, non_blocking=True),
            global_features=inputs.global_features.to(self.device, non_blocking=True),
            neighbor_index=neighbor_index,
            neighbor_mask=neighbor_mask,
            neighbor_edge_type=neighbor_edge_type,
            node_mask=node_mask,
            legal_action_mask=inputs.legal_action_mask.to(
                self.device, non_blocking=True
            ),
            rings=rings,
        )
        return ReplayBatch(
            inputs=encoded,
            targets=source.targets.to(self.device, non_blocking=True),
            feature_path=source.feature_path,
        )


def unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "module", None)
        if not isinstance(wrapped, nn.Module):
            wrapped = getattr(current, "_orig_mod", None)
        if not isinstance(wrapped, nn.Module):
            return current
        current = wrapped
    raise RuntimeError("cyclic model wrapper chain")


def maybe_compile_model(
    model: nn.Module,
    *,
    enabled: bool | str,
    dynamic: bool = True,
    fullgraph: bool = True,
    backend: str | None = None,
    mode: str | None = None,
    recompile_limit: int | None = None,
    isolate_recompiles: bool = False,
) -> nn.Module:
    if enabled == "auto":
        enabled = resolve_compile("auto", next(model.parameters()).device)
    elif not isinstance(enabled, bool):
        raise ValueError("compile enabled must be a boolean or 'auto'")
    if not enabled:
        return model
    if recompile_limit is not None and recompile_limit <= 0:
        raise ValueError("compile recompile_limit must be positive")
    if mode not in (None, "default", "reduce-overhead", "max-autotune"):
        raise ValueError("compile mode is invalid")
    options: dict[str, Any] = {
        "dynamic": dynamic,
        "fullgraph": fullgraph,
        "recompile_limit": recompile_limit,
        "isolate_recompiles": isolate_recompiles,
    }
    if backend is not None:
        options["backend"] = backend
    if mode not in (None, "default"):
        options["mode"] = mode
    return cast(
        nn.Module,
        torch.compile(model, **options),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to ``min_lr_ratio``."""

    def multiplier(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    setattr(
        scheduler,
        "_startrain_scheduler_config",
        {
            "warmup_steps": config.warmup_steps,
            "total_steps": config.total_steps,
            "min_lr_ratio": config.min_lr_ratio,
        },
    )
    return scheduler


def scheduler_diagnostics(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> SchedulerDiagnostics:
    """Describe scheduler age and position within its configured segment."""

    age = max(0, int(scheduler.last_epoch))
    raw_config = getattr(scheduler, "_startrain_scheduler_config", None)
    if not isinstance(raw_config, dict):
        return SchedulerDiagnostics(age, "unknown", age, None, None)
    try:
        config = SchedulerConfig(**raw_config)
    except (TypeError, ValueError):
        return SchedulerDiagnostics(age, "unknown", age, None, None)
    if config.warmup_steps and age < config.warmup_steps:
        length = config.warmup_steps
        return SchedulerDiagnostics(
            age_steps=age,
            segment="warmup",
            segment_step=age,
            segment_length_steps=length,
            segment_position=age / length,
        )
    if age < config.total_steps:
        length = config.total_steps - config.warmup_steps
        step = max(0, age - config.warmup_steps)
        return SchedulerDiagnostics(
            age_steps=age,
            segment="cosine",
            segment_step=step,
            segment_length_steps=length,
            segment_position=min(1.0, step / max(1, length)),
        )
    return SchedulerDiagnostics(
        age_steps=age,
        segment="floor",
        segment_step=max(0, age - config.total_steps),
        segment_length_steps=None,
        segment_position=1.0,
    )


def ema_effective_turnover(decay: float, updates: int) -> float:
    """Return the fraction of EMA weight replaced over ``updates`` updates."""

    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("EMA updates must be a non-negative integer")
    if updates == 0:
        return 0.0
    if decay == 0:
        return 1.0
    return -math.expm1(updates * math.log(decay))


@torch.no_grad()
def _collect_ema_diagnostics(
    model: nn.Module,
    ema: ExponentialMovingAverage,
) -> _EMADiagnosticTensors:
    model_state = model.state_dict()
    if not ema.shadow:
        raise ValueError("EMA state is empty")
    first_average = next(iter(ema.shadow.values()))
    raw_squared = torch.zeros((), device=first_average.device, dtype=torch.float32)
    ema_squared = torch.zeros_like(raw_squared)
    distance_squared = torch.zeros_like(raw_squared)
    for name, average in ema.shadow.items():
        source = model_state.get(name)
        if source is None or source.shape != average.shape:
            raise ValueError("model state does not match EMA state")
        raw = source.detach().to(device=average.device, dtype=torch.float32)
        average_float = average.detach().float()
        raw_squared.add_(raw.square().sum())
        ema_squared.add_(average_float.square().sum())
        distance_squared.add_((raw - average_float).square().sum())
    return _EMADiagnosticTensors(
        decay=ema.decay,
        num_updates=ema.num_updates,
        raw_norm=raw_squared.sqrt(),
        ema_norm=ema_squared.sqrt(),
        distance_norm=distance_squared.sqrt(),
    )


def train_step(
    model: nn.Module,
    batch: ReplayBatch,
    optimizer: torch.optim.Optimizer,
    *,
    loss_weights: LossWeights = LossWeights(),
    precision: str = "fp32",
    gradient_clip_norm: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ExponentialMovingAverage | None = None,
    trusted_batch: bool = False,
    collect_diagnostics: bool = False,
) -> TrainStepResult:
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    original_model = unwrap_model(model)
    parameter = next(original_model.parameters())
    device = parameter.device
    batch = batch.to(device, non_blocking=device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)

    autocast_enabled = precision == "bf16"
    if autocast_enabled and device.type not in ("cpu", "cuda"):
        raise ValueError(f"BF16 autocast is unsupported on {device.type}")
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        output = model(*batch.inputs.model_args())
        losses = compute_losses(
            output,
            batch.targets,
            legal_action_mask=batch.inputs.legal_action_mask,
            node_mask=batch.inputs.node_mask,
            score_margin_min=SCORE_MARGIN_MIN,
            score_margin_max=SCORE_MARGIN_MAX,
            weights=loss_weights,
            validate_targets=not trusted_batch,
            include_diagnostics=collect_diagnostics,
        )
    total = losses["total"]
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        original_model.parameters(), gradient_clip_norm, error_if_nonfinite=False
    )
    gradient_clipped = gradient_norm > gradient_clip_norm
    loss_is_finite = torch.isfinite(total).all()
    gradient_is_finite = torch.isfinite(gradient_norm).all()
    nonfinite_loss_count = (~loss_is_finite).to(dtype=torch.int64)
    nonfinite_gradient_count = (~gradient_is_finite).to(dtype=torch.int64)
    finite = torch.stack((loss_is_finite, gradient_is_finite)).to(dtype=torch.int32)
    nonfinite_counts = torch.stack((nonfinite_loss_count, nonfinite_gradient_count))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(
            nonfinite_counts,
            op=torch.distributed.ReduceOp.SUM,
        )
    if not bool(finite.all()):
        raise NonFiniteTrainingError(
            nonfinite_loss_count=int(nonfinite_counts[0].item()),
            nonfinite_gradient_count=int(nonfinite_counts[1].item()),
        )
    optimizer_snapshot = (
        capture_optimizer_diagnostic_snapshot(optimizer)
        if collect_diagnostics
        else None
    )
    optimizer.step()
    optimizer_diagnostics = (
        finalize_optimizer_step_diagnostics(optimizer, optimizer_snapshot)
        if optimizer_snapshot is not None
        else None
    )
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update(original_model)
    ema_diagnostics = (
        _collect_ema_diagnostics(original_model, ema)
        if collect_diagnostics and ema is not None
        else None
    )
    return TrainStepResult(
        loss_tensors=losses,
        gradient_norm_tensor=gradient_norm,
        gradient_clipped_tensor=gradient_clipped,
        nonfinite_loss_count_tensor=nonfinite_loss_count,
        nonfinite_gradient_count_tensor=nonfinite_gradient_count,
        learning_rates=tuple(float(group["lr"]) for group in optimizer.param_groups),
        optimizer_diagnostics=optimizer_diagnostics,
        scheduler_diagnostics=(
            scheduler_diagnostics(scheduler) if scheduler is not None else None
        ),
        ema_diagnostics=ema_diagnostics,
        gradient_clip_threshold=float(gradient_clip_norm),
    )
