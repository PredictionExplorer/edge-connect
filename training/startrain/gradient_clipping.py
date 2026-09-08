"""Global clipping and resumable tensor-wise AdaGC gradient control.

AdaGC follows Algorithm 1 of https://arxiv.org/html/2502.11034v2: globally
clipped warmup observations initialize a running minimum, then the previous
EMA bounds each tensor and is updated from its *clipped* norm. Zero or missing
gradients are treated as inactive observations so temporarily unused tensors
cannot become permanently locked at a zero threshold. A tensor first active
after warmup bootstraps from a globally clipped observation.

This changes optimizer inputs; improved convergence on another workload is
not a guarantee for mixed board sizes or rule modes. The default multiplier
allows only 4% above the previous EMA. A large global spike during warmup can
therefore leave low minima for otherwise ordinary tensors; whether that tight
threshold helps a mixed workload must be measured, not silently compensated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
from torch import nn
from torch.utils._foreach_utils import (
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

GRADIENT_CLIPPING_STATE_FORMAT = "startrain.gradient-clipping"
GRADIENT_CLIPPING_STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class GradientClippingConfig:
    mode: Literal["global", "adagc"] = "global"
    beta: float = 0.99
    multiplier: float = 1.04
    warmup_steps: int = 100

    def __post_init__(self) -> None:
        if self.mode not in ("global", "adagc"):
            raise ValueError("gradient clipping mode must be global or adagc")
        for name in ("beta", "multiplier"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"gradient clipping {name} must be finite")
            object.__setattr__(self, name, float(value))
        if not 0 <= self.beta < 1:
            raise ValueError("gradient clipping beta must be in [0, 1)")
        if self.multiplier < 1:
            raise ValueError("gradient clipping multiplier must be at least 1")
        if type(self.warmup_steps) is not int or self.warmup_steps < 1:
            raise ValueError(
                "gradient clipping warmup_steps must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class GradientNorms:
    total_norm: torch.Tensor
    parameter_norms: dict[int, torch.Tensor]
    named_norms: dict[str, torch.Tensor]
    _owner: object = field(repr=False)
    _step: int = field(repr=False)
    _gradients: dict[str, torch.Tensor] = field(repr=False)
    _versions: dict[str, int] = field(repr=False)


@dataclass(frozen=True, slots=True)
class GradientClipResult:
    pre_clip_norm: torch.Tensor
    post_clip_norm: torch.Tensor
    coefficients: dict[str, torch.Tensor]
    clipped: dict[str, torch.Tensor]
    mode: str
    warmup: bool
    steps: int
    parameter_clip_fraction: torch.Tensor
    minimum_coefficient: torch.Tensor
    median_coefficient: torch.Tensor


class GradientClipper:
    """Bound to stable named parameters; checkpoints contain only safe tensors.

    ``measure`` does not mutate gradients or controller state. ``apply_`` must
    run before the optimizer and after any distributed finite-loss check. The
    optional ``finite_checked`` flag skips one aggregate host synchronization
    only when the caller has already validated this measurement collectively.
    Gradient tensors must remain unchanged between measurement and clipping.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        config: GradientClippingConfig = GradientClippingConfig(),
        max_norm: float = 1.0,
    ) -> None:
        if not isinstance(config, GradientClippingConfig):
            raise TypeError("config must be a GradientClippingConfig")
        if (
            isinstance(max_norm, bool)
            or not isinstance(max_norm, int | float)
            or not math.isfinite(max_norm)
            or max_norm <= 0
        ):
            raise ValueError("max_norm must be finite and positive")
        self.config = config
        self.max_norm = float(max_norm)
        self._parameters: dict[str, nn.Parameter] = {}
        identities: set[int] = set()
        for name, parameter in named_parameters:
            if not isinstance(name, str) or not name or name in self._parameters:
                raise ValueError("gradient clipping parameter names must be unique")
            if not isinstance(parameter, nn.Parameter) or id(parameter) in identities:
                raise ValueError(
                    "gradient clipping parameters must be unique Parameters"
                )
            identities.add(id(parameter))
            self._parameters[name] = parameter
        self._manifest = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in self._parameters.items()
        ]
        self._history: dict[str, torch.Tensor] = {}
        self._owner = object()
        self.updates = 0

    @property
    def steps(self) -> int:
        return self.updates

    @staticmethod
    def _history_dtype(parameter: nn.Parameter) -> torch.dtype:
        return (
            torch.float64
            if parameter.dtype in (torch.float64, torch.complex128)
            else torch.float32
        )

    @torch.no_grad()
    def measure(self) -> GradientNorms:
        gradients: dict[str, torch.Tensor] = {}
        for specification, (name, parameter) in zip(
            self._manifest, self._parameters.items(), strict=True
        ):
            if (
                list(parameter.shape) != specification["shape"]
                or str(parameter.dtype) != specification["dtype"]
            ):
                raise ValueError("gradient clipping parameter shape or dtype changed")
            if parameter.grad is not None:
                if parameter.grad.layout != torch.strided:
                    raise ValueError("gradient clipping requires dense gradients")
                gradients[name] = parameter.grad
        names = list(gradients)
        values = list(gradients.values())
        named_norms: dict[str, torch.Tensor] = {}
        norms = []
        # Preserve PyTorch get_total_norm's device/dtype grouping and reduction
        # order, including mixed-dtype inputs. This keeps global clipping exact.
        tensor_values: list[torch.Tensor | None] = list(values)
        groups = (
            _group_tensors_by_device_and_dtype([tensor_values], with_indices=True)
            if tensor_values
            else {}
        )
        for (device, _), ([tensors], indices) in groups.items():
            dense = [tensor for tensor in tensors if tensor is not None]
            local_norms = (
                torch._foreach_norm(dense, 2.0)  # pyright: ignore[reportPrivateImportUsage]
                if _has_foreach_support(dense, device)
                else [torch.linalg.vector_norm(tensor, 2.0) for tensor in dense]
            )
            norms.extend(local_norms)
            for index, norm in zip(indices, local_norms, strict=True):
                named_norms[names[index]] = norm
        total = (
            torch.linalg.vector_norm(
                torch.stack([norm.to(values[0].device) for norm in norms]), 2.0
            )
            if values
            else torch.zeros(
                (),
                device=(
                    next(iter(self._parameters.values())).device
                    if self._parameters
                    else "cpu"
                ),
            )
        )
        return GradientNorms(
            total_norm=total,
            parameter_norms={
                id(self._parameters[name]): norm for name, norm in named_norms.items()
            },
            named_norms=named_norms,
            _owner=self._owner,
            _step=self.updates,
            _gradients=gradients,
            _versions={name: gradient._version for name, gradient in gradients.items()},
        )

    @torch.no_grad()
    def apply_(
        self, measurement: GradientNorms, *, finite_checked: bool = False
    ) -> GradientClipResult:
        if type(finite_checked) is not bool:
            raise TypeError("finite_checked must be boolean")
        if measurement._owner is not self._owner or measurement._step != self.updates:
            raise ValueError(
                "gradient measurement is stale or belongs to another clipper"
            )
        for name, parameter in self._parameters.items():
            gradient = measurement._gradients.get(name)
            if parameter.grad is not gradient or (
                gradient is not None
                and gradient._version != measurement._versions[name]
            ):
                raise ValueError("gradients changed after measurement")
        if not finite_checked and not bool(torch.isfinite(measurement.total_norm)):
            raise ValueError(
                "nonfinite gradient norm; gradients and clipping history unchanged"
            )
        warmup = self.config.mode == "adagc" and self.updates < self.config.warmup_steps
        global_coefficient = torch.clamp(
            self.max_norm / (measurement.total_norm + 1e-6), max=1.0
        )
        names = list(measurement._gradients)
        gradients = list(measurement._gradients.values())
        coefficients: dict[str, torch.Tensor] = {}
        history_groups = []
        tensor_gradients: list[torch.Tensor | None] = list(gradients)
        groups = (
            _group_tensors_by_device_and_dtype([tensor_gradients], with_indices=True)
            if tensor_gradients
            else {}
        )
        for (device, _), ([tensors], indices) in groups.items():
            dense = [tensor for tensor in tensors if tensor is not None]
            local_names = [names[index] for index in indices]
            dtype = self._history_dtype(self._parameters[local_names[0]])
            norms = torch.stack(
                [measurement.named_norms[name].to(dtype=dtype) for name in local_names]
            )
            history = torch.stack(
                [
                    self._history[name].to(device=device, dtype=dtype)
                    if name in self._history
                    else torch.zeros((), device=device, dtype=dtype)
                    for name in local_names
                ]
            )
            if self.config.mode == "global" or warmup:
                local_coefficients = global_coefficient.to(device).expand(
                    len(local_names)
                )
            else:
                safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
                adaptive = torch.clamp(
                    self.config.multiplier * history / safe_norms, max=1.0
                )
                local_coefficients = torch.where(
                    norms > 0,
                    torch.where(history > 0, adaptive, global_coefficient.to(device)),
                    torch.ones_like(norms),
                )
            scalars = list(local_coefficients.unbind())
            coefficients.update(zip(local_names, scalars, strict=True))
            history_groups.append((local_names, history, dtype))
            if self.config.mode == "adagc" and not warmup:
                if _has_foreach_support(dense, device):
                    torch._foreach_mul_(dense, scalars)  # pyright: ignore[reportPrivateImportUsage]
                else:
                    for gradient, coefficient in zip(dense, scalars, strict=True):
                        gradient.mul_(coefficient)
        if self.config.mode == "global" or warmup:
            torch.nn.utils.clip_grads_with_norm_(
                list(self._parameters.values()), self.max_norm, measurement.total_norm
            )
        post = self.measure()
        next_history: dict[str, torch.Tensor] = {}
        if self.config.mode == "adagc":
            for local_names, history, dtype in history_groups:
                norms = torch.stack(
                    [post.named_norms[name].to(dtype=dtype) for name in local_names]
                )
                updated = (
                    torch.minimum(history, norms)
                    if warmup
                    else self.config.beta * history + (1.0 - self.config.beta) * norms
                )
                updated = torch.where(
                    norms > 0, torch.where(history > 0, updated, norms), history
                )
                next_history.update(zip(local_names, updated.unbind(), strict=True))
        self._commit_history(next_history)
        self.updates += 1
        clipped = {
            name: measurement.named_norms[name] > post.named_norms[name]
            for name in names
        }
        device = measurement.total_norm.device
        coefficient_values = (
            torch.stack(
                [
                    value.to(device=device, dtype=torch.float32)
                    for value in coefficients.values()
                ]
            )
            if coefficients
            else torch.ones(1, device=device)
        )
        fraction = (
            torch.stack(
                [
                    value.to(device=device, dtype=torch.float32)
                    for value in clipped.values()
                ]
            ).mean()
            if clipped
            else torch.zeros((), device=device)
        )
        return GradientClipResult(
            pre_clip_norm=measurement.total_norm,
            post_clip_norm=post.total_norm,
            coefficients=coefficients,
            clipped=clipped,
            mode=self.config.mode,
            warmup=warmup,
            steps=self.updates,
            parameter_clip_fraction=fraction,
            minimum_coefficient=coefficient_values.min(),
            median_coefficient=coefficient_values.median(),
        )

    def _commit_history(self, updated: Mapping[str, torch.Tensor]) -> None:
        """Keep one persistent scalar per tensor, even with changing activity.

        Retaining slices of a new active-tensor vector every step can keep many
        old backing vectors alive when tensors stop participating. Persistent
        destinations and batched copies keep history storage linear in the
        parameter count, with no per-parameter host synchronization.
        """

        groups: dict[tuple[torch.device, torch.dtype], list[str]] = {}
        for name, value in updated.items():
            groups.setdefault((value.device, value.dtype), []).append(name)
        for (device, dtype), names in groups.items():
            unallocated = [
                name
                for name in names
                if name not in self._history
                or self._history[name].device != device
                or self._history[name].dtype != dtype
            ]
            if unallocated:
                storage = torch.empty(len(unallocated), device=device, dtype=dtype)
                self._history.update(zip(unallocated, storage.unbind(), strict=True))
            destinations = [self._history[name] for name in names]
            sources = [updated[name] for name in names]
            if _has_foreach_support(destinations, device):
                torch._foreach_copy_(destinations, sources)  # pyright: ignore[reportPrivateImportUsage]
            else:
                for destination, source in zip(destinations, sources, strict=True):
                    destination.copy_(source)

    def state_dict(self) -> dict[str, Any]:
        # A complete norm map distinguishes an inactive tensor (zero) from a
        # missing/corrupted history entry without guessing during restoration.
        norms = (
            {
                name: (
                    self._history[name].detach().to("cpu").clone()
                    if name in self._history
                    else torch.zeros((), dtype=self._history_dtype(parameter))
                )
                for name, parameter in self._parameters.items()
            }
            if self.config.mode == "adagc"
            else {}
        )
        return {
            "format": GRADIENT_CLIPPING_STATE_FORMAT,
            "version": GRADIENT_CLIPPING_STATE_VERSION,
            "config": asdict(self.config),
            "max_norm": self.max_norm,
            "updates": self.updates,
            "parameters": [
                dict(row, shape=list(row["shape"])) for row in self._manifest
            ],
            "ema_norms": norms,
        }

    def reset(self) -> None:
        """Explicitly discard history and begin a fresh warmup."""

        self._history = {}
        self.updates = 0
        self._owner = object()

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate complete checkpoint authority without changing live state."""

        self._validated_state_dict(state)

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        updates, history = self._validated_state_dict(state)
        self._history = history
        self.updates = updates
        self._owner = object()

    @torch.no_grad()
    def _validated_state_dict(
        self, state: Mapping[str, Any]
    ) -> tuple[int, dict[str, torch.Tensor]]:
        required = {
            "format",
            "version",
            "config",
            "max_norm",
            "updates",
            "parameters",
            "ema_norms",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("gradient clipping state fields are invalid")
        if (
            state["format"] != GRADIENT_CLIPPING_STATE_FORMAT
            or type(state["version"]) is not int
            or state["version"] != GRADIENT_CLIPPING_STATE_VERSION
        ):
            raise ValueError("gradient clipping state format or version is invalid")
        config = state["config"]
        if not isinstance(config, dict) or set(config) != set(asdict(self.config)):
            raise ValueError("gradient clipping state configuration is invalid")
        loaded_config = GradientClippingConfig(**config)
        if (
            loaded_config != self.config
            or type(state["max_norm"]) is not float
            or state["max_norm"] != self.max_norm
        ):
            raise ValueError("gradient clipping state hyperparameters differ")
        updates = state["updates"]
        if type(updates) is not int or updates < 0:
            raise ValueError(
                "gradient clipping state updates must be nonnegative integer"
            )
        manifest = state["parameters"]
        if not isinstance(manifest, list) or len(manifest) != len(self._manifest):
            raise ValueError("gradient clipping state parameter manifest differs")
        for expected, actual in zip(self._manifest, manifest, strict=True):
            if (
                not isinstance(actual, dict)
                or set(actual) != {"name", "shape", "dtype"}
                or not isinstance(actual["shape"], list)
                or any(type(value) is not int or value < 0 for value in actual["shape"])
                or actual != expected
            ):
                raise ValueError(
                    "gradient clipping state parameter names, shapes or dtypes differ"
                )
        norms = state["ema_norms"]
        expected_names = set(self._parameters) if self.config.mode == "adagc" else set()
        if not isinstance(norms, dict) or set(norms) != expected_names:
            raise ValueError("gradient clipping state norm names differ")
        history: dict[str, torch.Tensor] = {}
        for name, norm in norms.items():
            parameter = self._parameters[name]
            dtype = self._history_dtype(parameter)
            if (
                not isinstance(norm, torch.Tensor)
                or norm.layout != torch.strided
                or norm.ndim != 0
                or norm.dtype != dtype
                or not bool(torch.isfinite(norm))
                or bool(norm < 0)
                or (updates == 0 and bool(norm != 0))
            ):
                raise ValueError(
                    "gradient clipping state EMA norms must be finite nonnegative scalars"
                )
            history[name] = (
                norm.detach().to(device=parameter.device, dtype=dtype).clone()
            )
        # Validation and device copies complete before changing live authority.
        return updates, history
