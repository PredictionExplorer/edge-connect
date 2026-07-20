#!/usr/bin/env python3
"""Reject a profile that is unsafe for an always-restarting continuous unit."""

from __future__ import annotations

import argparse

from startrain.config import ExperimentConfig, load_config


def _validate_autonomous_config(config: ExperimentConfig) -> None:
    autonomous = config.orchestration.autonomous
    if not autonomous.enabled:
        raise ValueError(
            "autonomous validator requires orchestration.autonomous.enabled"
        )
    if config.data.shards_per_batch < 2:
        raise ValueError("autonomous service requires cross-shard replay batches")
    selfplay = config.selfplay
    if (
        selfplay.max_considered_ring_exponent <= 0
        or selfplay.max_considered_cap < 48
        or selfplay.full_probability < 0.35
        or not selfplay.record_fast_policy_targets
        or not 0 < selfplay.fast_policy_weight <= 0.25
        or selfplay.policy_surprise_weight <= 0
    ):
        raise ValueError(
            "autonomous service requires large-board search and weighted fast targets"
        )
    learner = config.learner
    if (
        learner.target_updates_per_new_sample is None
        or learner.target_updates_per_new_sample > 1.25
        or learner.candidate_interval_examples is None
        or learner.selfplay_snapshot_interval_examples is None
        or learner.selfplay_snapshot_warmup_interval_examples is None
        or learner.selfplay_snapshot_warmup_examples <= 0
        or learner.selfplay_snapshot_interval_examples
        > learner.candidate_interval_examples
    ):
        raise ValueError(
            "autonomous service requires bounded update-to-data and decoupled cadence"
        )
    refresh = config.orchestration.model_refresh
    if (
        refresh.selfplay_source != "candidate_champion_history_mix"
        or refresh.history_probability <= 0
        or refresh.candidate_probability <= 0
    ):
        raise ValueError(
            "autonomous service requires candidate/champion/history self-play"
        )
    plateau = config.orchestration.plateau
    if (
        not plateau.enabled
        or plateau.action != "reduce_lr_keep_weights"
        or plateau.reset_learning_rate_scale > 0.5
        or not plateau.clear_optimizer_state_on_recovery
    ):
        raise ValueError("autonomous service requires non-destructive plateau recovery")
    mixture = config.orchestration.ring_mixture
    final_weights = mixture.weights_for_step(10**18)
    if (
        final_weights is None
        or final_weights[mixture.rings.index(10)] < 0.5
        or abs(sum(final_weights) - 1.0) > 1e-9
    ):
        raise ValueError("autonomous service requires a ring-10-weighted final mixture")
    retention = config.orchestration.retention
    if retention.candidate_manifests < 2 * refresh.history_pool_size:
        raise ValueError("autonomous retention cannot protect the history pool")
    historical = config.orchestration.historical_evaluation
    if historical.enabled and (
        historical.every_promotions < 4
        or historical.anchors_per_evaluation > 1
        or historical.pairs_per_ring > 10
        or historical.max_pairs_per_ring > 10
    ):
        raise ValueError("autonomous historical evaluation exceeds its compute budget")
    promotion = config.orchestration.promotion
    learner_gpu_ids = {
        gpu.gpu_id for gpu in config.orchestration.gpus if gpu.role == "learner"
    }
    if promotion.gpu_id in learner_gpu_ids and (
        promotion.max_waves_per_lease != 1
        or promotion.inter_wave_cooldown_seconds < 1_800
        or historical.enabled
    ):
        raise ValueError(
            "autonomous learner-shared promotion requires one-wave leases, "
            "a 30-minute catch-up interval, and disabled historical evaluation"
        )
    if config.arena.max_considered < 48:
        raise ValueError("autonomous arena must evaluate a broad action set")


def _validate_throughput_config(config: ExperimentConfig) -> None:
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
    arena = config.arena
    continuation = arena.continuation_pairs_per_ring or arena.pairs_per_ring
    remaining_after_minimum = arena.max_pairs_per_ring - arena.minimum_pairs_per_ring
    continuation_waves = (
        (remaining_after_minimum + continuation - 1) // continuation
        if remaining_after_minimum
        else 0
    )
    if continuation_waves > 1:
        raise ValueError(
            "continuous service requires arena continuation to reach the maximum "
            "within one post-minimum wave"
        )


def validate_continuous_config(config: ExperimentConfig) -> None:
    if config.profile != "continuous" or not config.orchestration.enabled:
        raise ValueError("continuous service requires an enabled continuous profile")
    if not config.learner.unlimited:
        raise ValueError("continuous service requires learner.unlimited: true")
    if config.learner.recovery_interval_steps is None:
        raise ValueError("continuous service requires recovery_interval_steps")
    if config.learner.steps != 1_000_000:
        raise ValueError("continuous service requires learner.steps: 1000000")
    if config.train.scheduler.total_steps != 1_000_000:
        raise ValueError("continuous service requires scheduler.total_steps: 1000000")
    promotion = config.orchestration.promotion
    if (
        not promotion.enabled
        or config.arena.minimum_pairs_per_ring < config.arena.pairs_per_ring
        or config.arena.max_pairs_per_ring < config.arena.minimum_pairs_per_ring
    ):
        raise ValueError("continuous service requires bounded promotion supervision")
    plateau = config.orchestration.plateau
    if (
        not plateau.enabled
        or plateau.max_learner_champion_lag_steps > config.learner.max_replay_lag_steps
    ):
        raise ValueError("continuous service plateau lag exceeds replay eligibility")
    retention = config.orchestration.retention
    if (
        not retention.enabled
        or retention.dry_run
        or retention.recovery_dry_run
        or retention.recovery_checkpoints <= 0
    ):
        raise ValueError("continuous service requires bounded active retention")
    if config.orchestration.autonomous.enabled:
        _validate_autonomous_config(config)
    else:
        _validate_throughput_config(config)


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
