"""Muon + AdamW parameter routing with a safe native AdamW fallback."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    kind: Literal["muon_adamw", "adamw"] = "muon_adamw"
    adamw_lr: float = 3e-4
    muon_lr: float = 2e-2
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    min_muon_elements: int = 64
    fallback_to_adamw: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("muon_adamw", "adamw"):
            raise ValueError("kind must be 'muon_adamw' or 'adamw'")
        if self.adamw_lr <= 0 or self.muon_lr <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.weight_decay:
            raise ValueError("weight_decay must be non-negative")
        if not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("AdamW betas must be in [0, 1)")
        if not 0 <= self.muon_momentum < 1:
            raise ValueError("muon_momentum must be in [0, 1)")
        if self.muon_ns_steps < 1:
            raise ValueError("muon_ns_steps must be positive")


def _zeroth_power_newton_schulz(gradient: Tensor, steps: int) -> Tensor:
    """Approximate the polar factor in float32 using Muon's quintic iteration."""

    if gradient.ndim != 2:
        raise ValueError("Muon updates require matrix parameters")
    update = gradient.float()
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.transpose(0, 1)
    update = update / update.norm().clamp_min(1e-7)
    coefficient_a, coefficient_b, coefficient_c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        covariance = update @ update.transpose(0, 1)
        polynomial = coefficient_b * covariance + coefficient_c * (
            covariance @ covariance
        )
        update = coefficient_a * update + polynomial @ update
    if transposed:
        update = update.transpose(0, 1)
    return update


class MuonAdamW(torch.optim.Optimizer):
    """Muon for hidden matrices and AdamW for all remaining parameters."""

    def __init__(
        self,
        *,
        muon_params: list[nn.Parameter],
        adamw_decay_params: list[nn.Parameter],
        adamw_no_decay_params: list[nn.Parameter],
        config: OptimizerConfig,
    ) -> None:
        if not muon_params:
            raise ValueError("MuonAdamW requires at least one Muon parameter")
        parameter_groups: list[dict[str, object]] = [
            {
                "params": muon_params,
                "algorithm": "muon",
                "lr": config.muon_lr,
                "weight_decay": config.weight_decay,
                "momentum": config.muon_momentum,
                "nesterov": config.muon_nesterov,
                "ns_steps": config.muon_ns_steps,
            }
        ]
        if adamw_decay_params:
            parameter_groups.append(
                {
                    "params": adamw_decay_params,
                    "algorithm": "adamw",
                    "lr": config.adamw_lr,
                    "weight_decay": config.weight_decay,
                    "betas": config.betas,
                    "eps": config.eps,
                }
            )
        if adamw_no_decay_params:
            parameter_groups.append(
                {
                    "params": adamw_no_decay_params,
                    "algorithm": "adamw",
                    "lr": config.adamw_lr,
                    "weight_decay": 0.0,
                    "betas": config.betas,
                    "eps": config.eps,
                }
            )
        super().__init__(parameter_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def, override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group: dict) -> None:
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        momentum = float(group["momentum"])
        nesterov = bool(group["nesterov"])
        steps = int(group["ns_steps"])
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            if parameter.ndim != 2:
                raise RuntimeError("a non-matrix parameter was routed to Muon")
            state = self.state[parameter]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(
                    parameter, dtype=torch.float32
                )
            gradient = parameter.grad.detach().float()
            buffer = state["momentum_buffer"]
            buffer.mul_(momentum).add_(gradient)
            update_source = gradient.add(buffer, alpha=momentum) if nesterov else buffer
            update = _zeroth_power_newton_schulz(update_source, steps)
            aspect_scale = math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
            if weight_decay:
                parameter.mul_(1.0 - learning_rate * weight_decay)
            parameter.add_(
                update.to(dtype=parameter.dtype),
                alpha=-learning_rate * aspect_scale,
            )

    def _step_adamw(self, group: dict) -> None:
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        beta1, beta2 = group["betas"]
        epsilon = float(group["eps"])
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["step"] += 1
            gradient = parameter.grad.detach().float()
            average = state["exp_avg"]
            square_average = state["exp_avg_sq"]
            average.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            square_average.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            denominator = square_average.sqrt().div_(math.sqrt(bias_correction2))
            denominator.add_(epsilon)
            if weight_decay:
                parameter.mul_(1.0 - learning_rate * weight_decay)
            update = (average / denominator).to(dtype=parameter.dtype)
            parameter.add_(update, alpha=-learning_rate / bias_correction1)


def _native_adamw(
    decay_parameters: list[nn.Parameter],
    no_decay_parameters: list[nn.Parameter],
    config: OptimizerConfig,
) -> torch.optim.AdamW:
    groups = []
    if decay_parameters:
        groups.append(
            {
                "params": decay_parameters,
                "weight_decay": config.weight_decay,
            }
        )
    if no_decay_parameters:
        groups.append({"params": no_decay_parameters, "weight_decay": 0.0})
    return torch.optim.AdamW(
        groups,
        lr=config.adamw_lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=0.0,
    )


def split_decay_parameters(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    """Return decay/no-decay groups without name-only norm heuristics."""

    parameter_modules: dict[int, nn.Module] = {}
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            parameter_modules[id(parameter)] = module
    decay: list[tuple[str, nn.Parameter]] = []
    no_decay: list[tuple[str, nn.Parameter]] = []
    norm_types = (nn.RMSNorm, nn.LayerNorm, nn.GroupNorm, nn.Embedding)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        module = parameter_modules.get(id(parameter))
        exclude = (
            parameter.ndim < 2
            or isinstance(module, norm_types)
            or name == "global_token"
            or "edge_embedding" in name
        )
        (no_decay if exclude else decay).append((name, parameter))
    return decay, no_decay


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig = OptimizerConfig(),
) -> torch.optim.Optimizer:
    """Select Muon+AdamW, falling back safely when no matrix is eligible."""

    decay_named, no_decay_named = split_decay_parameters(model)
    named_parameters = decay_named + no_decay_named
    all_parameters = [parameter for _, parameter in named_parameters]
    if not all_parameters:
        raise ValueError("model has no trainable parameters")
    if config.kind == "adamw":
        return _native_adamw(
            [parameter for _, parameter in decay_named],
            [parameter for _, parameter in no_decay_named],
            config,
        )

    adamw_name_fragments = (
        "norm",
        "bias",
        "node_projection",
        "global_projection",
        "policy",
        "head",
        "embedding",
    )
    muon_params: list[nn.Parameter] = []
    adamw_decay_params: list[nn.Parameter] = []
    for name, parameter in decay_named:
        use_muon = (
            parameter.ndim == 2
            and parameter.numel() >= config.min_muon_elements
            and not any(fragment in name.lower() for fragment in adamw_name_fragments)
        )
        (muon_params if use_muon else adamw_decay_params).append(parameter)
    adamw_no_decay_params = [parameter for _, parameter in no_decay_named]

    if not muon_params:
        if not config.fallback_to_adamw:
            raise ValueError("no parameters are eligible for Muon")
        warnings.warn(
            "no matrix parameters were eligible for Muon; using AdamW",
            RuntimeWarning,
            stacklevel=2,
        )
        return _native_adamw(
            [parameter for _, parameter in decay_named],
            adamw_no_decay_params,
            config,
        )
    try:
        return MuonAdamW(
            muon_params=muon_params,
            adamw_decay_params=adamw_decay_params,
            adamw_no_decay_params=adamw_no_decay_params,
            config=config,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        if not config.fallback_to_adamw:
            raise
        warnings.warn(
            f"Muon initialization failed ({exc}); using AdamW",
            RuntimeWarning,
            stacklevel=2,
        )
        return _native_adamw(
            [parameter for _, parameter in decay_named],
            adamw_no_decay_params,
            config,
        )
