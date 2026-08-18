from __future__ import annotations

import pytest

from startrain.config import ConfigError, LearnerConfig, TrainConfig


def test_example_normalized_ema_resolves_per_global_batch() -> None:
    config = TrainConfig(
        per_rank_batch_size=512,
        ema_decay=0.9,
        ema_half_life_examples=1_000_000,
    )

    decay = config.resolved_ema_decay(world_size=1)

    assert decay == pytest.approx(0.5 ** (512 / 1_000_000))
    assert decay ** (1_000_000 / 512) == pytest.approx(0.5)
    assert TrainConfig(ema_decay=0.99).resolved_ema_decay(1) == 0.99


@pytest.mark.parametrize("value", [0, -1, float("inf"), True])
def test_example_normalized_ema_rejects_invalid_half_life(value: object) -> None:
    with pytest.raises(ConfigError, match="ema_half_life_examples"):
        TrainConfig(ema_half_life_examples=value)  # type: ignore[arg-type]


def test_learner_replay_branch_cutoff_is_optional_and_nonnegative() -> None:
    assert LearnerConfig().minimum_replay_shard_id_exclusive is None
    assert (
        LearnerConfig(
            minimum_replay_shard_id_exclusive=0
        ).minimum_replay_shard_id_exclusive
        == 0
    )
    assert (
        LearnerConfig(
            minimum_replay_shard_id_exclusive=42
        ).minimum_replay_shard_id_exclusive
        == 42
    )
    with pytest.raises(ConfigError, match="minimum_replay_shard_id_exclusive"):
        LearnerConfig(minimum_replay_shard_id_exclusive=-1)
