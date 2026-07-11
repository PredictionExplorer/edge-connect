from dataclasses import fields
from typing import cast

import numpy as np
import pytest
import torch

from startrain.features import DoubleStarPosition, EncodedBatch, encode_batch
from startrain.native import (
    NativeFeatureDataProtocol,
    NativeScoreDataProtocol,
    encode_native_state_data,
    native_feature_path_stats,
    positions_from_native,
    reset_native_feature_path_stats,
    score_results_from_native,
    score_tensors_from_native_features,
)
from startrain.replay import (
    ReplayBatch,
    ReplaySample,
    augment_sample,
    collate_replay_samples,
)
from startrain.scoring import PlayerScore, ScoreResult
from startrain.symmetry import D5Transform
from startrain.topology import SUPPORTED_RINGS, get_topology


def assert_encoded_equal(actual: EncodedBatch, expected: EncodedBatch) -> None:
    for field in fields(EncodedBatch):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def assert_replay_equal(actual: ReplayBatch, expected: ReplayBatch) -> None:
    assert_encoded_equal(actual.inputs, expected.inputs)
    for field in fields(type(actual.targets)):
        actual_value = getattr(actual.targets, field.name)
        expected_value = getattr(expected.targets, field.name)
        if actual_value is None or expected_value is None:
            assert actual_value is expected_value
        else:
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def assert_native_scores_equal(feature_data: object, score_data: object) -> None:
    actual = score_tensors_from_native_features(
        cast(NativeFeatureDataProtocol, feature_data)
    )
    batch_size = int(getattr(score_data, "batch_size"))
    node_count = int(getattr(score_data, "node_count"))
    expected_components = torch.tensor(
        getattr(score_data, "components"), dtype=torch.int32
    ).reshape(batch_size, 14)
    expected_owner = torch.tensor(
        getattr(score_data, "node_owner"), dtype=torch.int8
    ).reshape(batch_size, node_count)
    expected_results = score_results_from_native(
        cast(NativeScoreDataProtocol, score_data)
    )
    expected_alive = torch.stack(
        [result.alive_stone for result in expected_results], dim=0
    )
    assert torch.equal(actual.components, expected_components)
    assert torch.equal(actual.node_owner[:, :node_count], expected_owner)
    assert torch.equal(actual.alive_stones[:, :node_count], expected_alive)


def live_sample(rings: int) -> ReplaySample:
    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    position = DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=1,
        moves_left=2,
        opening=False,
        terminal=False,
    )
    legal = stones.numpy() == -1
    policy = legal.astype(np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=ScoreResult(
            players=(
                PlayerScore(10, 3, 1, 1, 0, 11),
                PlayerScore(5, 2, 1, 0, 0, 5),
            ),
            node_owner=torch.zeros(topology.n, dtype=torch.int8),
            alive_stone=torch.zeros(topology.n, dtype=torch.bool),
            contested_peries=0,
            leader=0,
        ),
        search_provenance="mcts:native-feature-parity",
        policy_provenance="visits",
    )


@pytest.mark.native
def test_native_features_and_scores_match_oracle_all_rings_and_d5() -> None:
    native = pytest.importorskip("star_native")
    reset_native_feature_path_stats()
    for rings in SUPPORTED_RINGS:
        topology = get_topology(rings)
        states = native.StateBatch(rings, 3)
        states.apply_many([1], [0])
        states.apply_many([2] * topology.n, list(range(topology.n)))
        for transform_index in range(10):
            transformed = states.transformed(transform_index)
            state_data = transformed.data()
            expected = encode_batch(positions_from_native(state_data))
            actual = encode_native_state_data(state_data)
            assert_encoded_equal(actual, expected)
            assert_native_scores_equal(
                transformed.feature_data(), transformed.score_data()
            )
    stats = native_feature_path_stats()
    assert stats["native_state_batches"] == 40
    assert stats["native_state_rows"] == 120
    assert not any(key.startswith("python_") for key in stats)


@pytest.mark.native
def test_native_features_cover_full_board_terminals() -> None:
    native = pytest.importorskip("star_native")
    topology = get_topology(4)
    states = native.StateBatch(4, 1)
    states.apply_many([0] * topology.n, list(range(topology.n)))
    state_data = states.data()
    assert list(state_data.terminal) == [True]
    assert_encoded_equal(
        encode_native_state_data(state_data),
        encode_batch(positions_from_native(state_data)),
    )
    assert_native_scores_equal(states.feature_data(), states.score_data())


@pytest.mark.native
def test_heterogeneous_learner_batch_uses_native_exact_path() -> None:
    pytest.importorskip("star_native")
    samples = [
        augment_sample(live_sample(rings), D5Transform.from_index(index))
        for index, rings in enumerate(SUPPORTED_RINGS)
    ]
    topology = get_topology(4)
    terminal = DoubleStarPosition(
        rings=4,
        stones=torch.arange(topology.n, dtype=torch.int8) % 2,
        to_move=1,
        moves_left=0,
        opening=False,
        terminal=True,
    )
    samples.append(
        ReplaySample.from_position(
            terminal,
            policy=None,
            final_score=ScoreResult(
                players=(
                    PlayerScore(10, 3, 1, 1, 0, 11),
                    PlayerScore(5, 2, 1, 0, 0, 5),
                ),
                node_owner=torch.zeros(topology.n, dtype=torch.int8),
                alive_stone=torch.zeros(topology.n, dtype=torch.bool),
                contested_peries=0,
                leader=0,
            ),
            search_provenance="terminal:full-board",
            policy_provenance="none",
        )
    )

    reset_native_feature_path_stats()
    actual = collate_replay_samples(samples)
    expected = collate_replay_samples(samples, prefer_native=False)
    assert actual.feature_path == "rust"
    assert expected.feature_path == "python"
    assert_replay_equal(actual, expected)
    stats = native_feature_path_stats()
    assert stats["native_semantic_batches"] == 1
    assert stats["native_semantic_rows"] == len(samples)
