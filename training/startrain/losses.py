"""Masked multi-head losses with equal per-sample spatial weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from .contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from .model import StarModelOutput


@dataclass(frozen=True, slots=True)
class LossWeights:
    policy: float = 1.0
    outcome: float = 1.0
    score_margin: float = 0.25
    ownership: float = 0.25
    alive: float = 0.1
    soft_policy: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.policy,
            self.outcome,
            self.score_margin,
            self.ownership,
            self.alive,
            self.soft_policy,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("loss weights must be finite and non-negative")
        if not any(value > 0 for value in values):
            raise ValueError("at least one loss weight must be positive")


@dataclass(frozen=True, slots=True)
class TrainingTargets:
    policy: Tensor
    outcome: Tensor
    score_margin: Tensor
    ownership: Tensor
    alive: Tensor
    soft_policy: Tensor
    policy_mask: Tensor
    outcome_mask: Tensor
    score_margin_mask: Tensor
    ownership_mask: Tensor
    alive_mask: Tensor
    soft_policy_mask: Tensor
    sample_weight: Tensor | None = None
    policy_weight: Tensor | None = None

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "TrainingTargets":
        return TrainingTargets(
            policy=self.policy.to(device, non_blocking=non_blocking),
            outcome=self.outcome.to(device, non_blocking=non_blocking),
            score_margin=self.score_margin.to(device, non_blocking=non_blocking),
            ownership=self.ownership.to(device, non_blocking=non_blocking),
            alive=self.alive.to(device, non_blocking=non_blocking),
            soft_policy=self.soft_policy.to(device, non_blocking=non_blocking),
            policy_mask=self.policy_mask.to(device, non_blocking=non_blocking),
            outcome_mask=self.outcome_mask.to(device, non_blocking=non_blocking),
            score_margin_mask=self.score_margin_mask.to(
                device, non_blocking=non_blocking
            ),
            ownership_mask=self.ownership_mask.to(device, non_blocking=non_blocking),
            alive_mask=self.alive_mask.to(device, non_blocking=non_blocking),
            soft_policy_mask=self.soft_policy_mask.to(
                device, non_blocking=non_blocking
            ),
            sample_weight=(
                self.sample_weight.to(device, non_blocking=non_blocking)
                if self.sample_weight is not None
                else None
            ),
            policy_weight=(
                self.policy_weight.to(device, non_blocking=non_blocking)
                if self.policy_weight is not None
                else None
            ),
        )

    def pin_memory(self) -> "TrainingTargets":
        return TrainingTargets(
            policy=self.policy.pin_memory(),
            outcome=self.outcome.pin_memory(),
            score_margin=self.score_margin.pin_memory(),
            ownership=self.ownership.pin_memory(),
            alive=self.alive.pin_memory(),
            soft_policy=self.soft_policy.pin_memory(),
            policy_mask=self.policy_mask.pin_memory(),
            outcome_mask=self.outcome_mask.pin_memory(),
            score_margin_mask=self.score_margin_mask.pin_memory(),
            ownership_mask=self.ownership_mask.pin_memory(),
            alive_mask=self.alive_mask.pin_memory(),
            soft_policy_mask=self.soft_policy_mask.pin_memory(),
            sample_weight=(
                self.sample_weight.pin_memory()
                if self.sample_weight is not None
                else None
            ),
            policy_weight=(
                self.policy_weight.pin_memory()
                if self.policy_weight is not None
                else None
            ),
        )

    def record_stream(self, stream: torch.Stream) -> None:
        tensors = (
            self.policy,
            self.outcome,
            self.score_margin,
            self.ownership,
            self.alive,
            self.soft_policy,
            self.policy_mask,
            self.outcome_mask,
            self.score_margin_mask,
            self.ownership_mask,
            self.alive_mask,
            self.soft_policy_mask,
        )
        for tensor in tensors:
            tensor.record_stream(stream)
        if self.sample_weight is not None:
            self.sample_weight.record_stream(stream)
        if self.policy_weight is not None:
            self.policy_weight.record_stream(stream)


def _require_tensor(condition: Tensor, message: str) -> None:
    if not bool(condition.all()):
        raise ValueError(message)


def _weighted_mean(values: Tensor, valid: Tensor, weights: Tensor) -> Tensor:
    effective = valid.to(dtype=values.dtype) * weights.to(dtype=values.dtype)
    numerator = (values * effective).sum()
    denominator = effective.sum()
    tiny = torch.finfo(values.dtype).tiny
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(tiny),
        numerator * 0.0,
    )


def _per_sample_masked_mean(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    valid_float = valid.to(dtype=values.dtype)
    counts = valid_float.sum(dim=1)
    per_sample = (values * valid_float).sum(dim=1) / counts.clamp_min(1.0)
    return per_sample, counts > 0


def _soft_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    legal_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    legal_targets = targets.to(dtype=logits.dtype) * legal_mask.to(dtype=logits.dtype)
    mass = legal_targets.sum(dim=-1)
    valid = mass > 0
    normalized = legal_targets / mass.unsqueeze(-1).clamp_min(
        torch.finfo(logits.dtype).tiny
    )
    masked_logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    log_probabilities = functional.log_softmax(masked_logits.float(), dim=-1)
    losses = -(normalized.float() * log_probabilities).sum(dim=-1)
    return losses, valid


def _validate_shapes(
    output: StarModelOutput,
    targets: TrainingTargets,
    *,
    legal_action_mask: Tensor,
    node_mask: Tensor,
    margin_bins: int,
) -> None:
    if legal_action_mask.ndim != 2 or node_mask.ndim != 2:
        raise ValueError("legal and node masks must be rank-two tensors")
    if legal_action_mask.dtype != torch.bool or node_mask.dtype != torch.bool:
        raise ValueError("legal and node masks must be boolean")
    batch_size, actions = legal_action_mask.shape
    node_batch, nodes = node_mask.shape
    if batch_size != node_batch:
        raise ValueError("legal and node mask batch dimensions disagree")
    expected_outputs = (
        (output.policy_logits, (batch_size, actions), "policy logits"),
        (output.outcome_logits, (batch_size, 2), "outcome logits"),
        (
            output.score_margin_logits,
            (batch_size, margin_bins),
            "score-margin logits",
        ),
        (output.ownership_logits, (batch_size, nodes, 3), "ownership logits"),
        (output.alive_logits, (batch_size, nodes), "alive logits"),
        (output.soft_policy_logits, (batch_size, actions), "soft-policy logits"),
    )
    for tensor, shape, name in expected_outputs:
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    expected_targets = (
        (targets.policy, (batch_size, actions), "policy target"),
        (targets.outcome, (batch_size,), "outcome target"),
        (targets.score_margin, (batch_size,), "score-margin target"),
        (targets.ownership, (batch_size, nodes), "ownership target"),
        (targets.alive, (batch_size, nodes), "alive target"),
        (targets.soft_policy, (batch_size, actions), "soft-policy target"),
        (targets.policy_mask, (batch_size,), "policy mask"),
        (targets.outcome_mask, (batch_size,), "outcome mask"),
        (targets.score_margin_mask, (batch_size,), "score-margin mask"),
        (targets.ownership_mask, (batch_size,), "ownership mask"),
        (targets.alive_mask, (batch_size,), "alive mask"),
        (targets.soft_policy_mask, (batch_size,), "soft-policy mask"),
    )
    for tensor, shape, name in expected_targets:
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for weights, name in (
        (targets.sample_weight, "sample weights"),
        (targets.policy_weight, "policy weights"),
    ):
        if weights is not None and weights.shape != (batch_size,):
            raise ValueError(f"{name} must have shape ({batch_size},)")


def compute_losses(
    output: StarModelOutput,
    targets: TrainingTargets,
    *,
    legal_action_mask: Tensor,
    node_mask: Tensor,
    score_margin_min: int = SCORE_MARGIN_MIN,
    score_margin_max: int = SCORE_MARGIN_MAX,
    weights: LossWeights = LossWeights(),
    validate_targets: bool = True,
) -> dict[str, Tensor]:
    """Compute losses using explicit availability masks.

    Score margin ``-100`` is a real label. Missing margins are represented only
    by ``score_margin_mask=False`` and never by a colliding sentinel.
    """

    margin_bins = score_margin_max - score_margin_min + 1
    _validate_shapes(
        output,
        targets,
        legal_action_mask=legal_action_mask,
        node_mask=node_mask,
        margin_bins=margin_bins,
    )
    batch_size = output.policy_logits.shape[0]
    sample_weight = (
        targets.sample_weight
        if targets.sample_weight is not None
        else torch.ones(batch_size, device=output.policy_logits.device)
    )
    if validate_targets:
        _require_tensor(
            (sample_weight >= 0) & torch.isfinite(sample_weight),
            "sample weights must be finite and non-negative",
        )
    policy_weight = (
        targets.policy_weight
        if targets.policy_weight is not None
        else torch.ones_like(sample_weight)
    )
    if validate_targets:
        _require_tensor(
            (policy_weight >= 0) & torch.isfinite(policy_weight),
            "policy weights must be finite and non-negative",
        )
    policy_sample_weight = sample_weight * policy_weight

    policy_values, policy_has_mass = _soft_cross_entropy(
        output.policy_logits, targets.policy, legal_action_mask
    )
    policy_valid = policy_has_mass & targets.policy_mask.bool()
    policy_loss = _weighted_mean(policy_values, policy_valid, policy_sample_weight)

    outcome_mask = targets.outcome_mask.bool()
    if validate_targets:
        _require_tensor(
            (targets.outcome[outcome_mask] >= 0) & (targets.outcome[outcome_mask] <= 1),
            "available outcome labels must be loss=0 or win=1",
        )
    safe_outcome = torch.where(
        outcome_mask, targets.outcome, torch.zeros_like(targets.outcome)
    )
    outcome_values = functional.cross_entropy(
        output.outcome_logits.float(), safe_outcome.long(), reduction="none"
    )
    outcome_loss = _weighted_mean(outcome_values, outcome_mask, sample_weight)

    margin_mask = targets.score_margin_mask.bool()
    available_margin = targets.score_margin[margin_mask]
    if validate_targets:
        _require_tensor(
            (available_margin >= score_margin_min)
            & (available_margin <= score_margin_max),
            f"available score margins must be in "
            f"[{score_margin_min}, {score_margin_max}]",
        )
    safe_margin = torch.where(
        margin_mask,
        targets.score_margin,
        torch.full_like(targets.score_margin, score_margin_min),
    )
    margin_classes = safe_margin.long() - score_margin_min
    margin_values = functional.cross_entropy(
        output.score_margin_logits.float(), margin_classes, reduction="none"
    )
    score_margin_loss = _weighted_mean(margin_values, margin_mask, sample_weight)

    ownership_sample_mask = targets.ownership_mask.bool()
    ownership_valid = (
        node_mask & ownership_sample_mask.unsqueeze(1) & (targets.ownership != -100)
    )
    available_ownership = targets.ownership[ownership_valid]
    if validate_targets:
        _require_tensor(
            (available_ownership >= 0) & (available_ownership <= 2),
            "available ownership labels must be in 0..2",
        )
    safe_ownership = torch.where(
        ownership_valid,
        targets.ownership,
        torch.full_like(targets.ownership, -100),
    )
    ownership_values = functional.cross_entropy(
        output.ownership_logits.float().transpose(1, 2),
        safe_ownership.long(),
        reduction="none",
        ignore_index=-100,
    )
    ownership_per_sample, ownership_has_nodes = _per_sample_masked_mean(
        ownership_values, ownership_valid
    )
    ownership_loss = _weighted_mean(
        ownership_per_sample,
        ownership_has_nodes & ownership_sample_mask,
        sample_weight,
    )

    alive_sample_mask = targets.alive_mask.bool()
    alive_valid = node_mask & alive_sample_mask.unsqueeze(1) & (targets.alive >= 0)
    available_alive = targets.alive[alive_valid]
    if validate_targets:
        _require_tensor(
            (available_alive >= 0) & (available_alive <= 1),
            "available alive labels must be in [0, 1]",
        )
    alive_target = targets.alive.float().clamp(0, 1)
    alive_values = functional.binary_cross_entropy_with_logits(
        output.alive_logits.float(), alive_target, reduction="none"
    )
    alive_per_sample, alive_has_nodes = _per_sample_masked_mean(
        alive_values, alive_valid
    )
    alive_loss = _weighted_mean(
        alive_per_sample,
        alive_has_nodes & alive_sample_mask,
        sample_weight,
    )

    soft_values, soft_has_mass = _soft_cross_entropy(
        output.soft_policy_logits, targets.soft_policy, legal_action_mask
    )
    soft_valid = soft_has_mass & targets.soft_policy_mask.bool()
    soft_policy_loss = _weighted_mean(
        soft_values,
        soft_valid,
        policy_sample_weight,
    )

    total = (
        weights.policy * policy_loss
        + weights.outcome * outcome_loss
        + weights.score_margin * score_margin_loss
        + weights.ownership * ownership_loss
        + weights.alive * alive_loss
        + weights.soft_policy * soft_policy_loss
    )
    return {
        "total": total,
        "policy": policy_loss,
        "outcome": outcome_loss,
        "score_margin": score_margin_loss,
        "ownership": ownership_loss,
        "alive": alive_loss,
        "soft_policy": soft_policy_loss,
    }
