#!/usr/bin/env python3
"""Reject a profile that is unsafe for an always-restarting continuous unit."""

from __future__ import annotations

import argparse

from startrain.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if not config.learner.unlimited:
        raise SystemExit("continuous service requires learner.unlimited: true")
    if config.learner.recovery_interval_steps is None:
        raise SystemExit("continuous service requires recovery_interval_steps")
    if config.learner.steps != 1_000_000:
        raise SystemExit("continuous service requires learner.steps: 1000000")
    if config.train.scheduler.total_steps != 1_000_000:
        raise SystemExit("continuous service requires scheduler.total_steps: 1000000")
    mixture = config.orchestration.ring_mixture
    if mixture.weights_for_step(999_999) is not None:
        raise SystemExit(
            "continuous ring weighting must not activate before step 1000000"
        )
    weights = mixture.weights_for_step(1_000_000)
    if weights is None or any(
        abs(actual - expected) > 1e-9
        for actual, expected in zip(weights, (0.1, 0.1, 0.1, 0.7), strict=True)
    ):
        raise SystemExit(
            "continuous service requires step-1000000 weights [0.1, 0.1, 0.1, 0.7]"
        )
    retention = config.orchestration.retention
    if retention.recovery_dry_run:
        raise SystemExit("continuous service requires bounded recovery retention")
    print(
        "continuous profile validated:",
        config.orchestration.run_id,
        config.orchestration.directories.root,
    )


if __name__ == "__main__":
    main()
