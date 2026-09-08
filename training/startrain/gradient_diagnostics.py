"""Sampled gradient attribution without changing gradients or optimizer state.

Collect after backward and before clipping. Tensor reductions stay on their
existing device until ``to_host`` is requested at a diagnostic boundary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .features import GLOBAL_FEATURE_NAMES
from .optim import OptimizerRoutingMetadata
from .replay import ReplayBatch


_HEADS = ("policy", "outcome", "score_margin", "ownership", "alive", "soft_policy")
_SIX_MODES = (
    "classic",
    "double",
    "pie-classic",
    "pie-double",
    "handicap-classic",
    "handicap-double",
)


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _norm(value: Tensor) -> Tensor:
    detached = value.detach()
    if detached.is_sparse:
        detached = detached.coalesce().values()
    return torch.linalg.vector_norm(detached.float())


def _norms(values: Sequence[Tensor]) -> list[Tensor]:
    """Use grouped reductions for the usual dense, same-device parameters."""

    result: list[Tensor | None] = [None] * len(values)
    groups: dict[tuple[torch.device, torch.dtype], list[int]] = {}
    for index, value in enumerate(values):
        if value.is_sparse:
            result[index] = _norm(value)
        else:
            groups.setdefault((value.device, value.dtype), []).append(index)
    for indices in groups.values():
        measured = torch._foreach_norm(  # pyright: ignore[reportPrivateImportUsage]
            [values[index].detach() for index in indices]
        )
        for index, norm in zip(indices, measured, strict=True):
            result[index] = norm
    assert all(value is not None for value in result)
    return [value for value in result if value is not None]


def _variant(label: str) -> tuple[str, str, int]:
    if label in ("classic", "double"):
        return label, label, 1
    if label in ("pie-classic", "pie-double"):
        return label, label.removeprefix("pie-"), 1
    parts = label.split("-")
    if (
        len(parts) == 3
        and parts[0] == "handicap"
        and parts[1] in tuple(str(value) for value in range(2, 10))
        and parts[2] in ("classic", "double")
    ):
        return f"handicap-{parts[2]}", parts[2], int(parts[1])
    raise ValueError(f"unknown diagnostic variant label: {label!r}")


@dataclass(frozen=True, slots=True)
class _Parameter:
    name: str
    shape: tuple[int, ...]
    elements: int
    group_index: int | None
    group_name: str | None
    algorithm: str | None
    gradient_present: bool


@dataclass(frozen=True, slots=True)
class _BatchSnapshot:
    rows: int
    rings: Tensor
    variant_features: Tensor | None
    labels: tuple[str, ...] | None
    availability: Tensor
    weight_statistics: Tensor

    def to_host(self) -> dict[str, object]:
        rings = [int(value) for value in self.rings.cpu().tolist()]
        modes: Counter[str] = Counter()
        six_modes: Counter[str] = Counter({name: 0 for name in _SIX_MODES})
        severity: Counter[str] = Counter()
        unknown: Counter[str] = Counter()
        ring_modes: dict[str, Counter[str]] = {}
        if self.labels is not None:
            decoded = [_variant(label) for label in self.labels]
            for ring, (six, mode, handicap) in zip(rings, decoded, strict=True):
                modes[mode] += 1
                six_modes[six] += 1
                severity[str(handicap)] += 1
                ring_modes.setdefault(str(ring), Counter())[six] += 1
            source = "original_replay_variant"
        elif self.variant_features is not None:
            source = "encoded_v4_partial"
            for ring, row in zip(
                rings, self.variant_features.cpu().tolist(), strict=True
            ):
                turn, handicap, pending, swap = [float(value) for value in row]
                if not all(math.isfinite(value) for value in row):
                    unknown["invalid_features"] += 1
                    continue
                turn_size = round(turn * 2)
                handicap_size = round(handicap * 9)
                if (
                    turn_size not in (1, 2)
                    or not 1 <= handicap_size <= 9
                    or abs(turn * 2 - turn_size) > 0.02
                    or abs(handicap * 9 - handicap_size) > 0.04
                ):
                    unknown["invalid_features"] += 1
                    continue
                mode = "classic" if turn_size == 1 else "double"
                modes[mode] += 1
                severity[str(handicap_size)] += 1
                if handicap_size > 1:
                    six = f"handicap-{mode}"
                elif pending > 0.5 or swap > 0.5:
                    six = f"pie-{mode}"
                else:
                    unknown[f"standard_or_resolved_pie-{mode}"] += 1
                    continue
                six_modes[six] += 1
                ring_modes.setdefault(str(ring), Counter())[six] += 1
        else:
            source = "unavailable"
            unknown["unsupported_feature_schema"] = self.rows
        availability = [int(value) for value in self.availability.cpu().tolist()]
        statistics = [float(value) for value in self.weight_statistics.cpu().tolist()]
        return {
            "rows": self.rows,
            "rings": dict(sorted(Counter(str(value) for value in rings).items())),
            "modes": dict(sorted(modes.items())),
            "six_modes": dict(six_modes),
            "six_mode_unknown": dict(sorted(unknown.items())),
            "six_mode_counts_exact": self.labels is not None,
            "variant_source": source,
            "handicap_severity": dict(sorted(severity.items())),
            "ring_six_modes": {
                ring: dict(counts) for ring, counts in sorted(ring_modes.items())
            },
            "label_availability": dict(zip(_HEADS, availability[:6], strict=True)),
            "teacher_samples": availability[6],
            "clinch_samples": availability[7],
            "positive_weight_samples": availability[8],
            "positive_policy_weight_samples": availability[9],
            "sample_weight_sum": _finite(statistics[0]),
            "policy_sample_weight_sum": _finite(statistics[1]),
            "teacher_sample_weight_sum": _finite(statistics[2]),
        }


def _batch_snapshot(
    batch: ReplayBatch, variant_labels: Sequence[str] | None
) -> _BatchSnapshot:
    inputs, targets = batch.inputs, batch.targets
    rows = inputs.batch_size
    if inputs.rings.shape != (rows,):
        raise ValueError("diagnostic rings must have one value per sample")
    labels = variant_labels if variant_labels is not None else batch.variant_labels
    if labels is not None:
        if len(labels) != rows:
            raise ValueError("diagnostic variant labels must match batch size")
        labels = tuple(labels)
        for label in labels:
            if not isinstance(label, str):
                raise ValueError("diagnostic variant labels must be strings")
            _variant(label)
    masks: list[Tensor] = []
    for name in _HEADS:
        mask = getattr(targets, f"{name}_mask")
        if mask.shape != (rows,):
            raise ValueError(f"diagnostic {name} mask must match batch size")
        masks.append(mask.detach().bool())
    for name in ("teacher_mask", "clinch_mask"):
        mask = getattr(targets, name)
        if mask is not None and mask.shape != (rows,):
            raise ValueError(f"diagnostic {name} must match batch size")
        masks.append(
            mask.detach().bool() if mask is not None else torch.zeros_like(masks[0])
        )
    weight = targets.sample_weight
    policy_weight = targets.policy_weight
    for name, value in (("sample_weight", weight), ("policy_weight", policy_weight)):
        if value is not None and value.shape != (rows,):
            raise ValueError(f"diagnostic {name} must match batch size")
    weight = (
        weight.detach().float()
        if weight is not None
        else torch.ones_like(masks[0], dtype=torch.float32)
    )
    policy_weight = (
        policy_weight.detach().float()
        if policy_weight is not None
        else torch.ones_like(weight)
    )
    policy_sample_weight = weight * policy_weight
    availability = torch.stack(
        [mask.sum() for mask in masks]
        + [(weight > 0).sum(), (policy_sample_weight > 0).sum()]
    )
    weight_statistics = torch.stack(
        (weight.sum(), policy_sample_weight.sum(), (weight * masks[6]).sum())
    )
    variant_features = None
    if inputs.global_features.shape == (rows, len(GLOBAL_FEATURE_NAMES)):
        columns = [
            GLOBAL_FEATURE_NAMES.index(name)
            for name in (
                "turn_size_fraction",
                "handicap_fraction",
                "pie_pending",
                "swap_available",
            )
        ]
        variant_features = inputs.global_features.detach()[:, columns].clone()
    return _BatchSnapshot(
        rows,
        inputs.rings.detach().clone(),
        variant_features,
        labels,
        availability,
        weight_statistics,
    )


@dataclass(frozen=True, slots=True)
class GradientDiagnostics:
    """An owned sampled snapshot; converting it never rereads live gradients."""

    parameters: tuple[_Parameter, ...]
    norms: Tensor
    total_norm: Tensor
    batch: _BatchSnapshot
    top_n: int
    reused_gradient_norms: int

    def to_host(self) -> dict[str, object]:
        norms = self.norms.cpu().tolist()
        total = float(self.total_norm.cpu())
        squared_total = total * total if math.isfinite(total) else None
        observed = [
            (parameter, float(values[0]), float(values[1]))
            for parameter, values in zip(self.parameters, norms, strict=True)
            if parameter.gradient_present
        ]
        observed.sort(
            key=lambda item: (
                0 if not math.isfinite(item[1]) else 1,
                -item[1] if math.isfinite(item[1]) else 0,
                item[0].name,
            )
        )
        contributors = []
        for parameter, gradient, weight in observed[: self.top_n]:
            share = (
                gradient * gradient / squared_total
                if squared_total is not None
                and squared_total > 0
                and math.isfinite(gradient)
                else None
            )
            if squared_total == 0 and gradient == 0:
                share = 0.0
            contributors.append(
                {
                    "name": parameter.name,
                    "shape": list(parameter.shape),
                    "elements": parameter.elements,
                    "optimizer_group_index": parameter.group_index,
                    "optimizer_group": parameter.group_name,
                    "algorithm": parameter.algorithm,
                    "pre_clip_gradient_norm": _finite(gradient),
                    "gradient_squared_share": _finite(share)
                    if share is not None
                    else None,
                    "parameter_norm": _finite(weight),
                    "gradient_to_parameter_ratio": _finite(gradient / weight)
                    if weight > 0 and math.isfinite(gradient) and math.isfinite(weight)
                    else None,
                    "gradient_finite": math.isfinite(gradient),
                }
            )
        return {
            "schema_version": 1,
            "gradient_stage": "before_clipping",
            "pre_clip_global_norm": _finite(total),
            "global_norm_finite": math.isfinite(total),
            "parameter_tensors": len(self.parameters),
            "gradient_present_tensors": len(observed),
            "missing_gradient_tensors": len(self.parameters) - len(observed),
            "zero_gradient_tensors": sum(value == 0 for _, value, _ in observed),
            "nonfinite_gradient_tensors": sum(
                not math.isfinite(value) for _, value, _ in observed
            ),
            "reused_gradient_norms": self.reused_gradient_norms,
            "top_n": self.top_n,
            "parameters": contributors,
            "batch": self.batch.to_host(),
        }


@torch.no_grad()
def collect_gradient_diagnostics(
    model: nn.Module,
    batch: ReplayBatch,
    optimizer: torch.optim.Optimizer,
    *,
    parameter_norms: Mapping[int, Tensor] | None = None,
    total_norm: Tensor | None = None,
    top_n: int = 12,
    variant_labels: Sequence[str] | None = None,
) -> GradientDiagnostics:
    """Snapshot named pre-clip gradients, routes, and the actual sampled batch.

    ``parameter_norms`` is keyed by ``id(parameter)`` and contains pre-clip
    gradient norms, as returned by ``GradientClipper.measure``. Partial maps
    are allowed; omitted norms are measured here. No clipping is performed.
    """

    if type(top_n) is not int or not 1 <= top_n <= 128:
        raise ValueError("diagnostic top_n must be an integer in [1, 128]")
    named = tuple(model.named_parameters())
    if not named:
        raise ValueError("gradient diagnostics require model parameters")
    device = named[0][1].device
    if any(parameter.device != device for _, parameter in named):
        raise ValueError("gradient diagnostics require one model device")
    metadata = getattr(optimizer, "_startrain_routing_metadata", None)
    if isinstance(metadata, OptimizerRoutingMetadata) and len(metadata.groups) != len(
        optimizer.param_groups
    ):
        raise ValueError("optimizer groups disagree with routing metadata")
    routes: dict[int, tuple[int, str, str]] = {}
    for index, group in enumerate(optimizer.param_groups):
        descriptor = (
            metadata.groups[index]
            if isinstance(metadata, OptimizerRoutingMetadata)
            else None
        )
        route = (
            index,
            descriptor.name if descriptor else f"group_{index}",
            descriptor.algorithm
            if descriptor
            else str(group.get("algorithm", type(optimizer).__name__.lower())),
        )
        for parameter in group["params"]:
            if id(parameter) in routes:
                raise ValueError("a parameter occurs in multiple optimizer routes")
            routes[id(parameter)] = route
    source_norms = parameter_norms or {}
    gradients: list[Tensor] = []
    missing: list[int] = []
    reused = 0
    descriptors = []
    for index, (name, parameter) in enumerate(named):
        route = routes.get(id(parameter), (None, None, None))
        descriptors.append(
            _Parameter(
                name,
                tuple(parameter.shape),
                parameter.numel(),
                *route,
                parameter.grad is not None,
            )
        )
        norm = source_norms.get(id(parameter)) if parameter.grad is not None else None
        if norm is not None:
            if (
                not isinstance(norm, Tensor)
                or norm.numel() != 1
                or norm.device != device
            ):
                raise ValueError(
                    "reused gradient norms must be same-device scalar tensors"
                )
            gradients.append(norm.detach().reshape(()))
            reused += 1
        else:
            gradients.append(torch.zeros((), device=device))
            if parameter.grad is not None:
                missing.append(index)
    measured = _norms([named[index][1].grad for index in missing])  # type: ignore[list-item]
    for index, norm in zip(missing, measured, strict=True):
        gradients[index] = norm
    gradient_vector = torch.stack(gradients).float()
    weight_vector = torch.stack(_norms([parameter for _, parameter in named])).float()
    if total_norm is None:
        total_norm = torch.linalg.vector_norm(gradient_vector)
    elif (
        not isinstance(total_norm, Tensor)
        or total_norm.numel() != 1
        or total_norm.device != device
    ):
        raise ValueError("total gradient norm must be a same-device scalar tensor")
    assert isinstance(total_norm, Tensor)
    return GradientDiagnostics(
        tuple(descriptors),
        torch.stack((gradient_vector, weight_vector), dim=1),
        total_norm.detach().reshape(()).clone(),
        _batch_snapshot(batch, variant_labels),
        top_n,
        reused,
    )
