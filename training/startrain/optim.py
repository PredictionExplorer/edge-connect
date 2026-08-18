"""Muon + AdamW parameter routing with a safe native AdamW fallback."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class OptimizerRoutingGroup:
    """Stable, initialization-independent metadata for one parameter route."""

    name: str
    algorithm: str
    weight_decay: float
    parameter_tensors: int
    parameter_elements: int
    parameter_names: tuple[str, ...]

    def as_dict(self, *, include_parameter_names: bool = False) -> dict[str, object]:
        values: dict[str, object] = {
            "name": self.name,
            "algorithm": self.algorithm,
            "weight_decay": self.weight_decay,
            "parameter_tensors": self.parameter_tensors,
            "parameter_elements": self.parameter_elements,
        }
        if include_parameter_names:
            values["parameter_names"] = list(self.parameter_names)
        return values


@dataclass(frozen=True, slots=True)
class OptimizerRoutingMetadata:
    """Auditable routing identity attached to optimizers built by this module."""

    schema_version: int
    requested_kind: str
    implementation: str
    fallback_used: bool
    optimizer_config: OptimizerConfig
    routing_hash: str
    groups: tuple[OptimizerRoutingGroup, ...]

    @property
    def parameter_tensors(self) -> int:
        return sum(group.parameter_tensors for group in self.groups)

    @property
    def parameter_elements(self) -> int:
        return sum(group.parameter_elements for group in self.groups)

    def as_dict(self, *, include_parameter_names: bool = False) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "requested_kind": self.requested_kind,
            "implementation": self.implementation,
            "fallback_used": self.fallback_used,
            "optimizer_config": asdict(self.optimizer_config),
            "routing_hash": self.routing_hash,
            "parameter_tensors": self.parameter_tensors,
            "parameter_elements": self.parameter_elements,
            "groups": [
                group.as_dict(include_parameter_names=include_parameter_names)
                for group in self.groups
            ],
        }


@dataclass(frozen=True, slots=True)
class OptimizerGroupDiagnostics:
    """Observed norms for one optimizer group and one completed update."""

    group_index: int
    name: str
    algorithm: str
    parameter_tensors: int
    parameter_elements: int
    configured_learning_rate: float
    weight_norm: float
    gradient_norm: float
    update_norm: float
    effective_learning_rate: float | None
    update_to_weight_ratio: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "group_index": self.group_index,
            "name": self.name,
            "algorithm": self.algorithm,
            "parameter_tensors": self.parameter_tensors,
            "parameter_elements": self.parameter_elements,
            "configured_learning_rate": self.configured_learning_rate,
            "weight_norm": self.weight_norm,
            "gradient_norm": self.gradient_norm,
            "update_norm": self.update_norm,
            "effective_learning_rate": self.effective_learning_rate,
            "update_to_weight_ratio": self.update_to_weight_ratio,
        }


@dataclass(frozen=True, slots=True)
class _OptimizerGroupDiagnosticTensors:
    group_index: int
    name: str
    algorithm: str
    parameter_tensors: int
    parameter_elements: int
    configured_learning_rate: float
    weight_norm: Tensor
    gradient_norm: Tensor
    update_norm: Tensor


@dataclass(frozen=True, slots=True)
class OptimizerStepDiagnostics:
    """Device-resident diagnostics, synchronized only when converted to host."""

    groups: tuple[_OptimizerGroupDiagnosticTensors, ...]

    def to_host(self) -> tuple[OptimizerGroupDiagnostics, ...]:
        output: list[OptimizerGroupDiagnostics] = []
        for group in self.groups:
            weight_norm, gradient_norm, update_norm = (
                group.weight_norm.detach().float(),
                group.gradient_norm.detach().float(),
                group.update_norm.detach().float(),
            )
            values = (
                torch.stack((weight_norm, gradient_norm, update_norm)).cpu().tolist()
            )
            observed_effective_lr = (
                float(values[2] / values[1]) if values[1] > 0 else None
            )
            update_to_weight = float(values[2] / values[0]) if values[0] > 0 else None
            output.append(
                OptimizerGroupDiagnostics(
                    group_index=group.group_index,
                    name=group.name,
                    algorithm=group.algorithm,
                    parameter_tensors=group.parameter_tensors,
                    parameter_elements=group.parameter_elements,
                    configured_learning_rate=group.configured_learning_rate,
                    weight_norm=float(values[0]),
                    gradient_norm=float(values[1]),
                    update_norm=float(values[2]),
                    effective_learning_rate=observed_effective_lr,
                    update_to_weight_ratio=update_to_weight,
                )
            )
        return tuple(output)


@dataclass(frozen=True, slots=True)
class _OptimizerGroupSnapshot:
    descriptor: OptimizerRoutingGroup
    parameters: tuple[nn.Parameter, ...]
    values_before: tuple[Tensor, ...]
    configured_learning_rate: float
    weight_norm: Tensor
    gradient_norm: Tensor


@dataclass(frozen=True, slots=True)
class OptimizerDiagnosticSnapshot:
    """Pre-step values needed to observe an optimizer's actual update."""

    optimizer_id: int
    groups: tuple[_OptimizerGroupSnapshot, ...]


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
    all_parameters = [*decay_parameters, *no_decay_parameters]
    fused = bool(all_parameters) and all(
        parameter.is_cuda for parameter in all_parameters
    )
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
        fused=fused,
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


def _routing_metadata(
    *,
    requested_kind: str,
    implementation: str,
    fallback_used: bool,
    config: OptimizerConfig,
    groups: list[tuple[str, str, float, list[tuple[str, nn.Parameter]]]],
) -> OptimizerRoutingMetadata:
    canonical_groups: list[dict[str, object]] = []
    routing_groups: list[OptimizerRoutingGroup] = []
    for name, algorithm, weight_decay, parameters in groups:
        canonical_parameters = [
            {
                "name": parameter_name,
                "shape": list(parameter.shape),
                "elements": parameter.numel(),
            }
            for parameter_name, parameter in parameters
        ]
        canonical_groups.append(
            {
                "name": name,
                "algorithm": algorithm,
                "weight_decay": float(weight_decay),
                "parameters": canonical_parameters,
            }
        )
        routing_groups.append(
            OptimizerRoutingGroup(
                name=name,
                algorithm=algorithm,
                weight_decay=float(weight_decay),
                parameter_tensors=len(parameters),
                parameter_elements=sum(
                    parameter.numel() for _, parameter in parameters
                ),
                parameter_names=tuple(
                    parameter_name for parameter_name, _ in parameters
                ),
            )
        )
    canonical = {
        "schema_version": 1,
        "requested_kind": requested_kind,
        "implementation": implementation,
        "fallback_used": fallback_used,
        "optimizer_config": asdict(config),
        "groups": canonical_groups,
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return OptimizerRoutingMetadata(
        schema_version=1,
        requested_kind=requested_kind,
        implementation=implementation,
        fallback_used=fallback_used,
        optimizer_config=config,
        routing_hash=f"sha256-{digest}",
        groups=tuple(routing_groups),
    )


def _attach_routing_metadata(
    optimizer: torch.optim.Optimizer,
    metadata: OptimizerRoutingMetadata,
) -> torch.optim.Optimizer:
    if len(metadata.groups) != len(optimizer.param_groups):
        raise RuntimeError("optimizer routing metadata does not match parameter groups")
    setattr(optimizer, "_startrain_routing_metadata", metadata)
    return optimizer


def optimizer_routing_metadata(
    optimizer: torch.optim.Optimizer,
) -> OptimizerRoutingMetadata:
    """Return the immutable routing identity recorded by ``build_optimizer``."""

    metadata = getattr(optimizer, "_startrain_routing_metadata", None)
    if not isinstance(metadata, OptimizerRoutingMetadata):
        raise ValueError("optimizer has no StarTrain routing metadata")
    if len(metadata.groups) != len(optimizer.param_groups):
        raise ValueError("optimizer parameter groups changed after routing")
    return metadata


def optimizer_checkpoint_contract(
    optimizer: torch.optim.Optimizer,
) -> dict[str, object] | None:
    """Return a strict primitive-only optimizer identity for checkpoints."""

    try:
        metadata = optimizer_routing_metadata(optimizer)
    except ValueError:
        return None
    return metadata.as_dict(include_parameter_names=True)


def _runtime_routing_groups(
    optimizer: torch.optim.Optimizer,
) -> tuple[OptimizerRoutingGroup, ...]:
    metadata = getattr(optimizer, "_startrain_routing_metadata", None)
    if isinstance(metadata, OptimizerRoutingMetadata):
        if len(metadata.groups) != len(optimizer.param_groups):
            raise ValueError("optimizer parameter groups changed after routing")
        return metadata.groups
    groups: list[OptimizerRoutingGroup] = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = tuple(group["params"])
        groups.append(
            OptimizerRoutingGroup(
                name=f"group_{index}",
                algorithm=str(
                    group.get("algorithm", optimizer.__class__.__name__.lower())
                ),
                weight_decay=float(group.get("weight_decay", 0.0)),
                parameter_tensors=len(parameters),
                parameter_elements=sum(parameter.numel() for parameter in parameters),
                parameter_names=(),
            )
        )
    return tuple(groups)


def _squared_norm(tensors: list[Tensor], *, like: Tensor) -> Tensor:
    total = torch.zeros((), device=like.device, dtype=torch.float32)
    for tensor in tensors:
        values = tensor.detach()
        if values.is_sparse:
            values = values.coalesce().values()
        total.add_(values.float().square().sum())
    return total


@torch.no_grad()
def capture_optimizer_diagnostic_snapshot(
    optimizer: torch.optim.Optimizer,
) -> OptimizerDiagnosticSnapshot:
    """Capture pre-step values for sampled, implementation-neutral diagnostics."""

    descriptors = _runtime_routing_groups(optimizer)
    snapshots: list[_OptimizerGroupSnapshot] = []
    for index, (group, descriptor) in enumerate(
        zip(optimizer.param_groups, descriptors, strict=True)
    ):
        parameters = tuple(group["params"])
        if not parameters:
            raise ValueError(f"optimizer group {index} is empty")
        first = parameters[0]
        snapshots.append(
            _OptimizerGroupSnapshot(
                descriptor=descriptor,
                parameters=parameters,
                values_before=tuple(
                    parameter.detach().clone(memory_format=torch.preserve_format)
                    for parameter in parameters
                ),
                configured_learning_rate=float(group["lr"]),
                weight_norm=_squared_norm(list(parameters), like=first).sqrt(),
                gradient_norm=_squared_norm(
                    [
                        parameter.grad
                        for parameter in parameters
                        if parameter.grad is not None
                    ],
                    like=first,
                ).sqrt(),
            )
        )
    return OptimizerDiagnosticSnapshot(id(optimizer), tuple(snapshots))


@torch.no_grad()
def finalize_optimizer_step_diagnostics(
    optimizer: torch.optim.Optimizer,
    snapshot: OptimizerDiagnosticSnapshot,
) -> OptimizerStepDiagnostics:
    """Measure the actual parameter delta after an arbitrary optimizer step."""

    if snapshot.optimizer_id != id(optimizer):
        raise ValueError("optimizer diagnostic snapshot belongs to another optimizer")
    if len(snapshot.groups) != len(optimizer.param_groups):
        raise ValueError("optimizer groups changed during diagnostic sampling")
    diagnostics: list[_OptimizerGroupDiagnosticTensors] = []
    for index, (group, before) in enumerate(
        zip(optimizer.param_groups, snapshot.groups, strict=True)
    ):
        parameters = tuple(group["params"])
        if len(parameters) != len(before.parameters) or any(
            actual is not expected
            for actual, expected in zip(parameters, before.parameters, strict=True)
        ):
            raise ValueError("optimizer parameters changed during diagnostic sampling")
        update_squared = _squared_norm(
            [
                parameter.detach().float() - previous.detach().float()
                for parameter, previous in zip(
                    parameters, before.values_before, strict=True
                )
            ],
            like=parameters[0],
        )
        descriptor = before.descriptor
        diagnostics.append(
            _OptimizerGroupDiagnosticTensors(
                group_index=index,
                name=descriptor.name,
                algorithm=descriptor.algorithm,
                parameter_tensors=descriptor.parameter_tensors,
                parameter_elements=descriptor.parameter_elements,
                configured_learning_rate=before.configured_learning_rate,
                weight_norm=before.weight_norm,
                gradient_norm=before.gradient_norm,
                update_norm=update_squared.sqrt(),
            )
        )
    return OptimizerStepDiagnostics(tuple(diagnostics))


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
        optimizer = _native_adamw(
            [parameter for _, parameter in decay_named],
            [parameter for _, parameter in no_decay_named],
            config,
        )
        return _attach_routing_metadata(
            optimizer,
            _routing_metadata(
                requested_kind=config.kind,
                implementation="torch_adamw",
                fallback_used=False,
                config=config,
                groups=[
                    *(
                        [("adamw_decay", "adamw", config.weight_decay, decay_named)]
                        if decay_named
                        else []
                    ),
                    *(
                        [("adamw_no_decay", "adamw", 0.0, no_decay_named)]
                        if no_decay_named
                        else []
                    ),
                ],
            ),
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
    muon_named: list[tuple[str, nn.Parameter]] = []
    adamw_decay_named: list[tuple[str, nn.Parameter]] = []
    for name, parameter in decay_named:
        use_muon = (
            parameter.ndim == 2
            and parameter.numel() >= config.min_muon_elements
            and not any(fragment in name.lower() for fragment in adamw_name_fragments)
        )
        (muon_named if use_muon else adamw_decay_named).append((name, parameter))
    muon_params = [parameter for _, parameter in muon_named]
    adamw_decay_params = [parameter for _, parameter in adamw_decay_named]
    adamw_no_decay_params = [parameter for _, parameter in no_decay_named]

    if not muon_params:
        if not config.fallback_to_adamw:
            raise ValueError("no parameters are eligible for Muon")
        warnings.warn(
            "no matrix parameters were eligible for Muon; using AdamW",
            RuntimeWarning,
            stacklevel=2,
        )
        optimizer = _native_adamw(
            [parameter for _, parameter in decay_named],
            adamw_no_decay_params,
            config,
        )
        return _attach_routing_metadata(
            optimizer,
            _routing_metadata(
                requested_kind=config.kind,
                implementation="torch_adamw",
                fallback_used=True,
                config=config,
                groups=[
                    *(
                        [("adamw_decay", "adamw", config.weight_decay, decay_named)]
                        if decay_named
                        else []
                    ),
                    *(
                        [("adamw_no_decay", "adamw", 0.0, no_decay_named)]
                        if no_decay_named
                        else []
                    ),
                ],
            ),
        )
    try:
        optimizer = MuonAdamW(
            muon_params=muon_params,
            adamw_decay_params=adamw_decay_params,
            adamw_no_decay_params=adamw_no_decay_params,
            config=config,
        )
        return _attach_routing_metadata(
            optimizer,
            _routing_metadata(
                requested_kind=config.kind,
                implementation="muon_adamw",
                fallback_used=False,
                config=config,
                groups=[
                    ("muon", "muon", config.weight_decay, muon_named),
                    *(
                        [
                            (
                                "adamw_decay",
                                "adamw",
                                config.weight_decay,
                                adamw_decay_named,
                            )
                        ]
                        if adamw_decay_named
                        else []
                    ),
                    *(
                        [("adamw_no_decay", "adamw", 0.0, no_decay_named)]
                        if no_decay_named
                        else []
                    ),
                ],
            ),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        if not config.fallback_to_adamw:
            raise
        warnings.warn(
            f"Muon initialization failed ({exc}); using AdamW",
            RuntimeWarning,
            stacklevel=2,
        )
        optimizer = _native_adamw(
            [parameter for _, parameter in decay_named],
            adamw_no_decay_params,
            config,
        )
        return _attach_routing_metadata(
            optimizer,
            _routing_metadata(
                requested_kind=config.kind,
                implementation="torch_adamw",
                fallback_used=True,
                config=config,
                groups=[
                    *(
                        [("adamw_decay", "adamw", config.weight_decay, decay_named)]
                        if decay_named
                        else []
                    ),
                    *(
                        [("adamw_no_decay", "adamw", 0.0, no_decay_named)]
                        if no_decay_named
                        else []
                    ),
                ],
            ),
        )
