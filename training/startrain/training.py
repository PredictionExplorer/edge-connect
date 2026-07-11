"""Autocast/compile-aware single-step training utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .checkpoint import ExponentialMovingAverage
from .config import SchedulerConfig
from .contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from .features import EncodedBatch
from .losses import LossWeights, compute_losses
from .replay import ReplayBatch


@dataclass(frozen=True, slots=True)
class HostTrainStepMetrics:
    losses: dict[str, float]
    gradient_norm: float
    learning_rates: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    """Device-resident step results, synchronized only when metrics are requested."""

    loss_tensors: dict[str, torch.Tensor]
    gradient_norm_tensor: torch.Tensor
    learning_rates: tuple[float, ...]

    def to_host(self) -> HostTrainStepMetrics:
        names = tuple(self.loss_tensors)
        values = torch.stack(
            (
                *(self.loss_tensors[name].detach().float() for name in names),
                self.gradient_norm_tensor.detach().float(),
            )
        )
        host_values = values.cpu().tolist()
        return HostTrainStepMetrics(
            losses=dict(zip(names, host_values[:-1], strict=True)),
            gradient_norm=float(host_values[-1]),
            learning_rates=self.learning_rates,
        )

    @property
    def losses(self) -> dict[str, float]:
        return self.to_host().losses

    @property
    def gradient_norm(self) -> float:
        return self.to_host().gradient_norm


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
        self._stream = (
            torch.cuda.Stream(device=self.device)
            if enabled and self.device.type == "cuda"
            else None
        )
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
        self._next_source = source
        with torch.cuda.stream(self._stream):
            started = torch.cuda.Event(enable_timing=True)
            completed = torch.cuda.Event(enable_timing=True)
            started.record(self._stream)
            self._next_batch = self._to_device(source)
            completed.record(self._stream)
            self._next_copy_event = (started, completed)

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
    enabled: bool,
    dynamic: bool = True,
    fullgraph: bool = True,
    backend: str | None = None,
    recompile_limit: int | None = None,
    isolate_recompiles: bool = False,
) -> nn.Module:
    if not enabled:
        return model
    if recompile_limit is not None and recompile_limit <= 0:
        raise ValueError("compile recompile_limit must be positive")
    options = {
        "dynamic": dynamic,
        "fullgraph": fullgraph,
        "recompile_limit": recompile_limit,
        "isolate_recompiles": isolate_recompiles,
    }
    if backend is not None:
        options["backend"] = backend
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

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


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
        )
    total = losses["total"]
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        original_model.parameters(), gradient_clip_norm, error_if_nonfinite=False
    )
    finite = torch.stack(
        (torch.isfinite(total).all(), torch.isfinite(gradient_norm).all())
    ).to(dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
    if not bool(finite.all()):
        raise FloatingPointError(
            "non-finite training loss or gradient norm on at least one rank"
        )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update(original_model)
    return TrainStepResult(
        loss_tensors=losses,
        gradient_norm_tensor=gradient_norm,
        learning_rates=tuple(float(group["lr"]) for group in optimizer.param_groups),
    )
