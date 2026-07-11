"""Autocast/compile-aware single-step training utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .checkpoint import ExponentialMovingAverage
from .config import SchedulerConfig
from .contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from .losses import LossWeights, compute_losses
from .replay import ReplayBatch


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    losses: dict[str, float]
    gradient_norm: float
    learning_rates: tuple[float, ...]


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
) -> TrainStepResult:
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    original_model = unwrap_model(model)
    parameter = next(original_model.parameters())
    device = parameter.device
    batch = batch.to(device)
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
        )
    total = losses["total"]
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("non-finite training loss")
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        original_model.parameters(), gradient_clip_norm, error_if_nonfinite=True
    )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update(original_model)
    return TrainStepResult(
        losses={name: float(value.detach()) for name, value in losses.items()},
        gradient_norm=float(gradient_norm),
        learning_rates=tuple(float(group["lr"]) for group in optimizer.param_groups),
    )
