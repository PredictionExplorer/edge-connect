"""Non-compounding learning-rate governance for plateau recovery.

The learner schedule is defined by the profile's optimizer rates (the
*reference*). Plateau recovery may temporarily run below that reference through
a single absolute ``multiplier`` that is floored and restored on the next
promotion. Because the multiplier is stored next to the reference instead of
being folded into the scheduler's base rates, checkpoints saved during a
reduced segment cannot silently lower the rates of every later champion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

LEARNING_RATE_GOVERNOR_KEY = "learning_rate_governor"
LEARNING_RATE_GOVERNOR_SCHEMA_VERSION = 1


def _validated_rates(values: Sequence[float]) -> tuple[float, ...]:
    rates = tuple(float(value) for value in values)
    if not rates or any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0
        for value in rates
    ):
        raise ValueError("learning-rate reference must contain finite positive rates")
    return rates


def reduced_multiplier(scale: float, floor: float, *, current: float = 1.0) -> float:
    """Return the multiplier the next recovery stage may apply.

    Within one champion segment successive recoveries descend by ``scale`` from
    the ``current`` multiplier, so a plateau anneals in stages instead of
    stopping at a single cut. The descent is bounded by ``floor`` and the
    multiplier is restored on promotion, so nothing carries across champions and
    the effective rate can never decay geometrically toward zero.
    """

    if not 0 < scale <= 1 or not 0 < floor <= 1 or not 0 < current <= 1:
        raise ValueError("learning-rate scale, floor, and multiplier must be in (0, 1]")
    return max(float(floor), float(current) * float(scale))


@dataclass(frozen=True, slots=True)
class LearningRateGovernorState:
    """Reference schedule rates plus the single active recovery multiplier."""

    reference_base_lrs: tuple[float, ...]
    multiplier: float = 1.0
    scaled_champion_identity: str | None = None
    legacy_reference: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_base_lrs",
            _validated_rates(self.reference_base_lrs),
        )
        if not 0 < self.multiplier <= 1:
            raise ValueError("learning-rate multiplier must be in (0, 1]")

    @property
    def effective_base_lrs(self) -> tuple[float, ...]:
        return tuple(rate * self.multiplier for rate in self.reference_base_lrs)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": LEARNING_RATE_GOVERNOR_SCHEMA_VERSION,
            "reference_base_lrs": list(self.reference_base_lrs),
            "multiplier": float(self.multiplier),
            "scaled_champion_identity": self.scaled_champion_identity,
            "legacy_reference": bool(self.legacy_reference),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "LearningRateGovernorState":
        rates = payload.get("reference_base_lrs")
        multiplier = payload.get("multiplier", 1.0)
        champion = payload.get("scaled_champion_identity")
        legacy = payload.get("legacy_reference", False)
        if (
            not isinstance(rates, Sequence)
            or isinstance(rates, (str, bytes))
            or isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or (champion is not None and not isinstance(champion, str))
            or type(legacy) is not bool
        ):
            raise ValueError("learning-rate governor state is invalid")
        return cls(
            reference_base_lrs=tuple(float(value) for value in rates),
            multiplier=float(multiplier),
            scaled_champion_identity=champion,
            legacy_reference=legacy,
        )

    @classmethod
    def from_scheduler(
        cls,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        legacy_reference: bool = False,
    ) -> "LearningRateGovernorState":
        """Treat the scheduler's current base rates as the reference schedule."""

        return cls(
            reference_base_lrs=tuple(float(rate) for rate in scheduler.base_lrs),
            multiplier=1.0,
            legacy_reference=legacy_reference,
        )

    def with_multiplier(
        self,
        multiplier: float,
        *,
        scaled_champion_identity: str | None,
    ) -> "LearningRateGovernorState":
        return LearningRateGovernorState(
            reference_base_lrs=self.reference_base_lrs,
            multiplier=multiplier,
            scaled_champion_identity=scaled_champion_identity,
            legacy_reference=self.legacy_reference,
        )

    def restored(self) -> "LearningRateGovernorState":
        return LearningRateGovernorState(
            reference_base_lrs=self.reference_base_lrs,
            multiplier=1.0,
            scaled_champion_identity=None,
            legacy_reference=self.legacy_reference,
        )


def apply_governor(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: LearningRateGovernorState,
) -> tuple[float, ...]:
    """Make the optimizer and scheduler consistent with ``state``.

    Base and initial rates become ``reference * multiplier``; the live group
    rates are recomputed through the scheduler position so the change applies
    from the next step without disturbing warmup or cosine progress. Returns the
    live per-group rates.
    """

    groups = optimizer.param_groups
    if len(groups) != len(state.reference_base_lrs):
        raise ValueError("learning-rate governor does not match optimizer groups")
    effective = state.effective_base_lrs
    lambdas = getattr(scheduler, "lr_lambdas", None)
    last_epoch = max(0, int(getattr(scheduler, "last_epoch", 0)))
    live: list[float] = []
    for index, (group, base) in enumerate(zip(groups, effective, strict=True)):
        factor = 1.0
        if isinstance(lambdas, Sequence) and index < len(lambdas):
            factor = float(lambdas[index](last_epoch))
        rate = base * factor
        group["lr"] = rate
        if "initial_lr" in group:
            group["initial_lr"] = base
        live.append(rate)
    scheduler.base_lrs = list(effective)
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = list(live)
    return tuple(live)


def governor_from_checkpoint_extra(
    extra: object,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> LearningRateGovernorState:
    """Recover governor state from checkpoint metadata.

    Checkpoints written before the governor existed carry only scaled scheduler
    rates. Those rates are adopted verbatim as the reference so a legacy resume
    never jumps the effective learning rate.
    """

    payload = (
        extra.get(LEARNING_RATE_GOVERNOR_KEY) if isinstance(extra, Mapping) else None
    )
    if isinstance(payload, Mapping):
        state = LearningRateGovernorState.from_mapping(payload)
        if len(state.reference_base_lrs) != len(scheduler.base_lrs):
            raise ValueError("checkpoint learning-rate governor does not match groups")
        return state
    return LearningRateGovernorState.from_scheduler(scheduler, legacy_reference=True)
