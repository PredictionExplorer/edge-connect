#!/usr/bin/env python3
"""Reject a profile that is unsafe for an always-restarting continuous unit."""

from __future__ import annotations

import argparse

from startrain.config import load_config


def validate_continuous_config(config) -> None:
    if not config.learner.unlimited:
        raise ValueError("continuous service requires learner.unlimited: true")
    if config.learner.recovery_interval_steps is None:
        raise ValueError("continuous service requires recovery_interval_steps")
    if config.learner.steps != 1_000_000:
        raise ValueError("continuous service requires learner.steps: 1000000")
    if config.train.scheduler.total_steps != 1_000_000:
        raise ValueError("continuous service requires scheduler.total_steps: 1000000")
    mixture = config.orchestration.ring_mixture
    weights = mixture.weights_for_step(1_000_000)
    tail_weights = mixture.weights_for_step(10**18)
    expected_weights = (0.1, 0.1, 0.1, 0.7)
    if any(
        configured is None
        or any(
            abs(actual - expected) > 1e-9
            for actual, expected in zip(configured, expected_weights, strict=True)
        )
        for configured in (weights, tail_weights)
    ):
        raise ValueError(
            "continuous service requires step-1000000 weights [0.1, 0.1, 0.1, 0.7]"
        )
    ring_ten_weights = mixture.weights_for_step(360_000)
    expected_stages = (
        (360_000, (0.15, 0.15, 0.15, 0.55)),
        (1_000_000, (0.1, 0.1, 0.1, 0.7)),
    )
    actual_stages = tuple(
        (stage.from_step, stage.weights) for stage in mixture.step_weights
    )
    if ring_ten_weights != (0.15, 0.15, 0.15, 0.55) or actual_stages != expected_stages:
        raise ValueError(
            "continuous service requires step-360000 weights [0.15, 0.15, 0.15, 0.55]"
        )
    selfplay = config.selfplay
    if (
        selfplay.max_considered != 16
        or selfplay.simulation_reference_rings != 6
        or selfplay.max_considered_ring_exponent != 1.0
        or selfplay.max_considered_cap != 32
        or not selfplay.record_fast_policy_targets
        or abs(selfplay.fast_policy_weight - 0.25) > 1e-9
    ):
        raise ValueError(
            "continuous service requires ring-scaled search and weighted fast targets"
        )
    plateau = config.orchestration.plateau
    if (
        not plateau.enabled
        or plateau.action != "reset_from_champion"
        or plateau.consecutive_terminal_rejections > 2
        or plateau.reset_learning_rate_scale > 0.5
    ):
        raise ValueError("continuous service requires bounded lower-LR champion resets")
    retention = config.orchestration.retention
    if not retention.enabled or retention.dry_run or retention.recovery_dry_run:
        raise ValueError("continuous service requires bounded active retention")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    try:
        validate_continuous_config(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "continuous profile validated:",
        config.orchestration.run_id,
        config.orchestration.directories.root,
    )


if __name__ == "__main__":
    main()
