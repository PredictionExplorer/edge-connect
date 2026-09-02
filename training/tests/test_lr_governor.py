from __future__ import annotations

import math

import pytest
import torch

from startrain.config import SchedulerConfig
from startrain.lr_governor import (
    LEARNING_RATE_GOVERNOR_KEY,
    LearningRateGovernorState,
    apply_governor,
    governor_from_checkpoint_extra,
    reduced_multiplier,
)
from startrain.training import build_scheduler


def _optimizer_and_scheduler(
    *,
    rates: tuple[float, ...] = (1.0, 0.1),
    warmup_steps: int = 10,
    total_steps: int = 110,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(()))], "lr": rate} for rate in rates
    ]
    optimizer = torch.optim.SGD(groups)
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.1,
        ),
    )
    return optimizer, scheduler


def test_reduced_multiplier_is_absolute_and_floored() -> None:
    assert reduced_multiplier(0.5, 0.25) == 0.5
    assert reduced_multiplier(0.1, 0.25) == 0.25
    assert reduced_multiplier(1.0, 0.25) == 1.0
    with pytest.raises(ValueError):
        reduced_multiplier(0.0, 0.25)
    with pytest.raises(ValueError):
        reduced_multiplier(0.5, 1.5)


def test_state_validates_and_round_trips_through_checkpoint_metadata() -> None:
    state = LearningRateGovernorState(
        reference_base_lrs=(3e-4, 4.5e-6, 4.5e-6),
        multiplier=0.5,
        scaled_champion_identity="sha256-champion",
    )
    payload = state.as_dict()
    assert payload["schema_version"] == 1
    assert payload["multiplier"] == 0.5
    assert payload["legacy_reference"] is False
    restored = LearningRateGovernorState.from_mapping(payload)
    assert restored == state
    assert restored.effective_base_lrs == pytest.approx((1.5e-4, 2.25e-6, 2.25e-6))
    assert restored.restored().multiplier == 1.0
    assert restored.restored().scaled_champion_identity is None
    assert restored.with_multiplier(
        0.25, scaled_champion_identity="sha256-other"
    ).effective_base_lrs == pytest.approx((7.5e-5, 1.125e-6, 1.125e-6))

    with pytest.raises(ValueError):
        LearningRateGovernorState(reference_base_lrs=())
    with pytest.raises(ValueError):
        LearningRateGovernorState(reference_base_lrs=(0.0,))
    with pytest.raises(ValueError):
        LearningRateGovernorState(reference_base_lrs=(math.inf,))
    with pytest.raises(ValueError):
        LearningRateGovernorState(reference_base_lrs=(1.0,), multiplier=0.0)
    with pytest.raises(ValueError):
        LearningRateGovernorState(reference_base_lrs=(1.0,), multiplier=1.5)
    for payload in (
        {"reference_base_lrs": "1.0"},
        {"reference_base_lrs": [1.0], "multiplier": True},
        {"reference_base_lrs": [1.0], "scaled_champion_identity": 7},
        {"reference_base_lrs": [1.0], "legacy_reference": "no"},
    ):
        with pytest.raises(ValueError):
            LearningRateGovernorState.from_mapping(payload)


def test_apply_governor_preserves_scheduler_position() -> None:
    optimizer, scheduler = _optimizer_and_scheduler()
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    warmup_factor = scheduler.lr_lambdas[0](scheduler.last_epoch)
    assert 0 < warmup_factor < 1
    state = LearningRateGovernorState.from_scheduler(scheduler).with_multiplier(
        0.5, scaled_champion_identity=None
    )

    live = apply_governor(optimizer, scheduler, state)

    assert scheduler.base_lrs == pytest.approx([0.5, 0.05])
    assert [group["initial_lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.5, 0.05]
    )
    assert live == pytest.approx((0.5 * warmup_factor, 0.05 * warmup_factor))
    assert scheduler.get_last_lr() == pytest.approx(list(live))
    optimizer.step()
    scheduler.step()
    next_factor = scheduler.lr_lambdas[0](scheduler.last_epoch)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5 * next_factor)

    mismatched = LearningRateGovernorState(reference_base_lrs=(1.0,))
    with pytest.raises(ValueError, match="does not match optimizer groups"):
        apply_governor(optimizer, scheduler, mismatched)


def test_checkpoint_extra_recovers_recorded_or_legacy_reference() -> None:
    optimizer, scheduler = _optimizer_and_scheduler(rates=(0.25, 0.025))
    recorded = governor_from_checkpoint_extra(
        {
            LEARNING_RATE_GOVERNOR_KEY: {
                "reference_base_lrs": [1.0, 0.1],
                "multiplier": 0.25,
                "scaled_champion_identity": "sha256-a",
                "legacy_reference": False,
            }
        },
        scheduler,
    )
    assert recorded.reference_base_lrs == (1.0, 0.1)
    assert recorded.multiplier == 0.25
    assert not recorded.legacy_reference

    legacy = governor_from_checkpoint_extra({"run_id": "run"}, scheduler)
    assert legacy.legacy_reference
    assert legacy.reference_base_lrs == pytest.approx((0.25, 0.025))
    assert legacy.multiplier == 1.0
    assert governor_from_checkpoint_extra(None, scheduler) == legacy

    with pytest.raises(ValueError, match="does not match groups"):
        governor_from_checkpoint_extra(
            {LEARNING_RATE_GOVERNOR_KEY: {"reference_base_lrs": [1.0]}},
            scheduler,
        )
