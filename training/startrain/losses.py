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
    wdl: float = 1.0
    score_margin: float = 0.25
    ownership: float = 0.25
    alive: float = 0.1
    soft_policy: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.policy,
            self.wdl,
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
    wdl: Tensor
    score_margin: Tensor
    ownership: Tensor
    alive: Tensor
    soft_policy: Tensor
    policy_mask: Tensor
    wdl_mask: Tensor
    score_margin_mask: Tensor
    ownership_mask: Tensor
    alive_mask: Tensor
    soft_policy_mask: Tensor
    sample_weight: Tensor | None = None

    def to(self, device: torch.device | str) -> "TrainingTargets":
        return TrainingTargets(
            policy=self.policy.to(device),
            wdl=self.wdl.to(device),
            score_margin=self.score_margin.to(device),
            ownership=self.ownership.to(device),
            alive=self.alive.to(device),
            soft_policy=self.soft_policy.to(device),
            policy_mask=self.policy_mask.to(device),
            wdl_mask=self.wdl_mask.to(device),
            score_margin_mask=self.score_margin_mask.to(device),
            ownership_mask=self.ownership_mask.to(device),
            alive_mask=self.alive_mask.to(device),
            soft_policy_mask=self.soft_policy_mask.to(device),
            sample_weight=(
                self.sample_weight.to(device)
                if self.sample_weight is not None
                else None
            ),
        )


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


def compute_losses(
    output: StarModelOutput,
    targets: TrainingTargets,
    *,
    legal_action_mask: Tensor,
    node_mask: Tensor,
    score_margin_min: int = SCORE_MARGIN_MIN,
    score_margin_max: int = SCORE_MARGIN_MAX,
    weights: LossWeights = LossWeights(),
) -> dict[str, Tensor]:
    """Compute losses using explicit availability masks.

    Score margin ``-100`` is a real label. Missing margins are represented only
    by ``score_margin_mask=False`` and never by a colliding sentinel.
    """

    batch_size = output.policy_logits.shape[0]
    sample_weight = (
        targets.sample_weight
        if targets.sample_weight is not None
        else torch.ones(batch_size, device=output.policy_logits.device)
    )
    if bool((sample_weight < 0).any()) or not bool(torch.isfinite(sample_weight).all()):
        raise ValueError("sample weights must be finite and non-negative")

    policy_values, policy_has_mass = _soft_cross_entropy(
        output.policy_logits, targets.policy, legal_action_mask
    )
    policy_valid = policy_has_mass & targets.policy_mask.bool()
    policy_loss = _weighted_mean(policy_values, policy_valid, sample_weight)

    wdl_mask = targets.wdl_mask.bool()
    if bool(((targets.wdl[wdl_mask] < 0) | (targets.wdl[wdl_mask] > 2)).any()):
        raise ValueError("available WDL labels must be in 0..2")
    safe_wdl = torch.where(wdl_mask, targets.wdl, torch.zeros_like(targets.wdl))
    wdl_values = functional.cross_entropy(
        output.wdl_logits.float(), safe_wdl.long(), reduction="none"
    )
    wdl_loss = _weighted_mean(wdl_values, wdl_mask, sample_weight)

    margin_mask = targets.score_margin_mask.bool()
    available_margin = targets.score_margin[margin_mask]
    if bool(
        (
            (available_margin < score_margin_min)
            | (available_margin > score_margin_max)
        ).any()
    ):
        raise ValueError(
            f"available score margins must be in "
            f"[{score_margin_min}, {score_margin_max}]"
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
    if bool(((available_ownership < 0) | (available_ownership > 2)).any()):
        raise ValueError("available ownership labels must be in 0..2")
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
    if bool(((available_alive < 0) | (available_alive > 1)).any()):
        raise ValueError("available alive labels must be in [0, 1]")
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
    soft_policy_loss = _weighted_mean(soft_values, soft_valid, sample_weight)

    total = (
        weights.policy * policy_loss
        + weights.wdl * wdl_loss
        + weights.score_margin * score_margin_loss
        + weights.ownership * ownership_loss
        + weights.alive * alive_loss
        + weights.soft_policy * soft_policy_loss
    )
    return {
        "total": total,
        "policy": policy_loss,
        "wdl": wdl_loss,
        "score_margin": score_margin_loss,
        "ownership": ownership_loss,
        "alive": alive_loss,
        "soft_policy": soft_policy_loss,
    }
