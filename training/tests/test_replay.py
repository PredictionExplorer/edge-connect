from dataclasses import replace

import numpy as np
import pytest
import torch

from startrain.actions import extract_sample_actions, relocate_sample_actions
from startrain.contracts import (
    FEATURE_SCHEMA_HASH,
    OUTCOME_LOSS,
    OUTCOME_WIN,
    RULES_HASH,
    SOFT_POLICY_TEMPERATURE,
    TARGET_OUTCOME,
    TARGET_POLICY,
    TARGET_SOFT_POLICY,
)
from startrain.features import DoubleStarPosition
from startrain.replay import (
    REPLAY_SCHEMA_VERSION,
    ReplaySample,
    ReplaySchemaError,
    augment_sample,
    collate_replay_samples,
    decode_replay_shard,
    read_replay_shard,
    write_replay_shard,
)
from startrain.scoring import PlayerScore, ScoreResult
from startrain.symmetry import D5Transform
from startrain.topology import get_topology


def live_position(rings: int = 4) -> DoubleStarPosition:
    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    return DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=1,
        moves_left=2,
        opening=False,
        terminal=False,
    )


def decisive_score(position: DoubleStarPosition, *, winner: int = 0) -> ScoreResult:
    topology = get_topology(position.rings)
    loser = 1 - winner
    players = [PlayerScore(5, 2, 1, 0, 0, 5) for _ in range(2)]
    players[winner] = PlayerScore(10, 3, 1, 1, 0, 11)
    owner = torch.full((topology.n,), loser, dtype=torch.int8)
    owner[: topology.peri_count] = winner
    return ScoreResult(
        players=(players[0], players[1]),
        node_owner=owner,
        alive_stone=torch.zeros(topology.n, dtype=torch.bool),
        contested_peries=0,
        leader=winner,
    )


def normalized_policy(position: DoubleStarPosition) -> np.ndarray:
    legal = position.stones.numpy() == -1
    policy = legal.astype(np.float32)
    policy /= policy.sum()
    return policy


def sample_for(rings: int = 4) -> ReplaySample:
    position = live_position(rings)
    return ReplaySample.from_position(
        position,
        policy=normalized_policy(position),
        final_score=decisive_score(position),
        search_provenance="gumbel-completed-q:test",
        policy_provenance="completed-q",
    )


def test_schema_v4_is_node_only_and_binary() -> None:
    sample = sample_for()
    topology = get_topology(4)
    assert sample.schema_version == REPLAY_SCHEMA_VERSION == 4
    assert sample.rules_hash == RULES_HASH
    assert sample.feature_schema_hash == FEATURE_SCHEMA_HASH
    assert sample.soft_policy_temperature == SOFT_POLICY_TEMPERATURE == 4
    assert sample.policy.shape == sample.soft_policy.shape == (topology.n,)
    assert sample.target_mask & TARGET_POLICY
    assert sample.target_mask & TARGET_SOFT_POLICY
    assert sample.target_mask & TARGET_OUTCOME
    assert sample.outcome == OUTCOME_LOSS

    batch = collate_replay_samples([sample])
    assert batch.targets.policy.shape == (1, topology.n)
    assert batch.targets.outcome.shape == (1,)
    assert batch.targets.outcome.tolist() == [OUTCOME_LOSS]


def test_zero_margin_quark_tiebreak_still_has_binary_outcome() -> None:
    position = live_position()
    topology = get_topology(4)
    final = ScoreResult(
        players=(
            PlayerScore(5, 3, 1, 1, 0, 6),
            PlayerScore(6, 2, 1, 0, 0, 6),
        ),
        node_owner=torch.full((topology.n,), -1, dtype=torch.int8),
        alive_stone=torch.zeros(topology.n, dtype=torch.bool),
        contested_peries=topology.peri_count,
        leader=0,
    )
    sample = ReplaySample.from_position(
        position,
        policy=normalized_policy(position),
        final_score=final,
        search_provenance="test",
        policy_provenance="test",
    )
    outcome, margin = sample.outcome_targets()
    assert outcome == OUTCOME_LOSS
    assert margin == 0


def test_opening_and_terminal_samples_round_trip(tmp_path) -> None:
    topology = get_topology(4)
    opening = DoubleStarPosition(
        rings=4,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
    )
    full = DoubleStarPosition(
        rings=4,
        stones=torch.arange(topology.n, dtype=torch.int8) % 2,
        to_move=1,
        moves_left=0,
        opening=False,
        terminal=True,
    )
    samples = [
        ReplaySample.from_position(
            opening,
            policy=normalized_policy(opening),
            final_score=decisive_score(opening, winner=1),
            search_provenance="opening",
            policy_provenance="completed-q",
        ),
        ReplaySample.from_position(
            full,
            policy=None,
            final_score=decisive_score(full, winner=1),
            search_provenance="terminal",
            policy_provenance="none",
        ),
    ]
    path = write_replay_shard(tmp_path / "v4.npz", samples)
    loaded = read_replay_shard(path)
    assert loaded[0].opening and not loaded[0].terminal
    assert loaded[1].terminal and loaded[1].outcome == OUTCOME_WIN
    assert not loaded[1].target_mask & (TARGET_POLICY | TARGET_SOFT_POLICY)
    assert not bool(collate_replay_samples(loaded).inputs.legal_action_mask[1].any())


def test_old_or_incomplete_shards_are_rejected(tmp_path) -> None:
    path = write_replay_shard(tmp_path / "current.npz", [sample_for()])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    metadata = arrays["metadata"].item()
    arrays["metadata"] = np.asarray(
        str(metadata).replace('"schema_version": 4', '"schema_version": 3')
    )
    old = tmp_path / "old.npz"
    np.savez_compressed(old, **arrays)
    with pytest.raises(ReplaySchemaError, match="schema_version"):
        read_replay_shard(old)

    arrays.pop("policy_weight")
    missing = tmp_path / "missing.npz"
    np.savez_compressed(missing, **arrays)
    with pytest.raises(ReplaySchemaError, match="missing arrays"):
        read_replay_shard(missing)


def test_decode_materializes_each_v4_npz_member_once(tmp_path, monkeypatch) -> None:
    expected = [sample_for(), sample_for(), sample_for()]
    path = write_replay_shard(tmp_path / "decode-once.npz", expected)
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    original_getitem = archive_type.__getitem__
    calls: dict[str, int] = {}

    def counted_getitem(archive, key):
        calls[str(key)] = calls.get(str(key), 0) + 1
        return original_getitem(archive, key)

    monkeypatch.setattr(archive_type, "__getitem__", counted_getitem)
    decoded = decode_replay_shard(path)

    assert len(decoded) == 3
    assert set(calls) == {"metadata", *decoded.arrays}
    assert set(calls.values()) == {1}
    np.testing.assert_array_equal(decoded.sample(1).stones, expected[1].stones)
    with pytest.raises(IndexError):
        decoded.sample(3)


def test_replay_validation_checks_types_shapes_and_integer_ranges_before_cast() -> None:
    sample = sample_for()
    with pytest.raises(ReplaySchemaError, match="integer before conversion"):
        replace(sample, schema_version=True)
    with pytest.raises(ReplaySchemaError, match="opening must be bool"):
        replace(sample, opening=1)
    with pytest.raises(ReplaySchemaError, match="stones must contain integers"):
        replace(sample, stones=sample.stones.astype(np.float32))
    with pytest.raises(ReplaySchemaError, match="stones must have shape"):
        replace(sample, stones=sample.stones[:-1])
    with pytest.raises(ReplaySchemaError, match="cannot be represented"):
        replace(
            sample,
            stones=np.full(sample.stones.shape, 255, dtype=np.uint16),
        )
    with pytest.raises(ReplaySchemaError, match="policy must be numeric"):
        replace(sample, policy=np.full(sample.policy.shape, "invalid"))


def test_node_only_action_padding_has_no_reserved_slot() -> None:
    values = torch.arange(70)
    padded = relocate_sample_actions(
        values,
        sample_nodes=70,
        batch_max_nodes=105,
        fill_value=-777,
    )
    assert padded.shape == (105,)
    assert torch.equal(padded[:70], values)
    assert torch.equal(padded[70:], torch.full((35,), -777))
    assert torch.equal(
        extract_sample_actions(padded, sample_nodes=70, batch_max_nodes=105),
        values,
    )


def test_replay_rejects_ties_invalid_rings_and_illegal_policy_support() -> None:
    position = live_position()
    tied = replace(decisive_score(position), leader=-1)
    with pytest.raises(ReplaySchemaError, match="tied"):
        ReplaySample.from_position(
            position,
            policy=normalized_policy(position),
            final_score=tied,
            search_provenance="test",
            policy_provenance="test",
        )
    with pytest.raises(ValueError, match="one of"):
        live_position(5)

    sample = sample_for()
    illegal = sample.policy.copy()
    illegal[0] = 0.1
    illegal[1:] *= 0.9 / illegal[1:].sum()
    with pytest.raises(ReplaySchemaError, match="illegal action"):
        replace(sample, policy=illegal)


def test_replay_augmentation_round_trips_all_d5_transforms() -> None:
    sample = sample_for(6)
    for index in range(10):
        transform = D5Transform.from_index(index)
        inverse = (
            transform
            if transform.reflected
            else D5Transform(rotation=-transform.rotation)
        )
        restored = augment_sample(augment_sample(sample, transform), inverse)
        np.testing.assert_array_equal(restored.stones, sample.stones)
        np.testing.assert_allclose(restored.policy, sample.policy)
        np.testing.assert_allclose(restored.soft_policy, sample.soft_policy)
