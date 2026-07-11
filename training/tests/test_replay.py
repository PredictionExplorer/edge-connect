from dataclasses import replace

import numpy as np
import pytest
import torch

from startrain.actions import extract_sample_actions, relocate_sample_actions
from startrain.contracts import (
    FEATURE_SCHEMA_HASH,
    RULES_HASH,
    SOFT_POLICY_TEMPERATURE,
    TARGET_POLICY,
    TARGET_SOFT_POLICY,
    WDL_LOSS,
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
from startrain.scoring import PlayerScore, ScoreResult, score_position
from startrain.symmetry import D5Transform
from startrain.topology import get_topology


def live_position(rings: int) -> DoubleStarPosition:
    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    return DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=1,
        moves_left=1,
        opening=False,
        pass_streak=0,
        terminal=False,
    )


def normalized_policy(position: DoubleStarPosition) -> np.ndarray:
    topology = get_topology(position.rings)
    legal = np.concatenate(
        ((position.stones.numpy() == -1), np.asarray([not position.terminal]))
    )
    policy = legal.astype(np.float32)
    policy /= policy.sum()
    assert policy.shape == (topology.n + 1,)
    return policy


def sample_for(rings: int) -> ReplaySample:
    position = live_position(rings)
    return ReplaySample.from_position(
        position,
        policy=normalized_policy(position),
        final_score=score_position(get_topology(rings), position.stones),
        search_provenance="mcts:puct-v1:sims=64",
        policy_provenance="root-visit-counts",
    )


def test_schema_v2_soft_policy_provenance_and_hashes() -> None:
    sample = sample_for(3)
    assert sample.schema_version == REPLAY_SCHEMA_VERSION == 3
    assert sample.rules_hash == RULES_HASH
    assert sample.feature_schema_hash == FEATURE_SCHEMA_HASH
    assert sample.soft_policy_temperature == SOFT_POLICY_TEMPERATURE == 4
    assert sample.target_mask & TARGET_POLICY
    assert sample.target_mask & TARGET_SOFT_POLICY
    expected = np.power(sample.policy, 0.25)
    expected /= expected.sum()
    np.testing.assert_allclose(sample.soft_policy, expected, atol=2e-6)
    assert sample.search_provenance.startswith("mcts:")
    assert sample.policy_provenance == "root-visit-counts"


def test_absolute_quark_tiebreak_defines_wdl_but_not_margin() -> None:
    position = live_position(3)
    topology = get_topology(3)
    players = (
        PlayerScore(5, 3, 1, 1, 0, 6),
        PlayerScore(6, 2, 1, 0, 0, 6),
    )
    final = ScoreResult(
        players=players,
        node_owner=torch.full((topology.n,), -1, dtype=torch.int8),
        alive_stone=torch.zeros(topology.n, dtype=torch.bool),
        contested_peries=topology.peri_count,
        leader=0,
    )
    sample = ReplaySample.from_position(
        position,
        policy=normalized_policy(position),
        final_score=final,
        search_provenance="mcts:test",
        policy_provenance="visits",
    )
    wdl, margin = sample.outcome_targets()
    assert position.to_move == 1
    assert wdl == WDL_LOSS
    assert margin == 0


def test_terminal_and_opening_states_round_trip_in_one_shard(tmp_path) -> None:
    topology = get_topology(3)
    opening = DoubleStarPosition(
        rings=3,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        pass_streak=0,
        terminal=False,
    )
    opening_sample = ReplaySample.from_position(
        opening,
        policy=normalized_policy(opening),
        final_score=score_position(topology, opening.stones),
        search_provenance="mcts:opening",
        policy_provenance="visits",
    )
    terminal = DoubleStarPosition(
        rings=3,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=1,
        moves_left=2,
        opening=False,
        pass_streak=2,
        terminal=True,
    )
    terminal_sample = ReplaySample.from_position(
        terminal,
        policy=None,
        final_score=score_position(topology, terminal.stones),
        search_provenance="terminal:no-search",
        policy_provenance="none",
    )
    path = write_replay_shard(
        tmp_path / "schema-v2.npz", [opening_sample, terminal_sample]
    )
    loaded = read_replay_shard(path)
    assert loaded[0].opening and not loaded[0].terminal
    assert loaded[1].terminal and loaded[1].pass_streak == 2
    assert not loaded[1].target_mask & (TARGET_POLICY | TARGET_SOFT_POLICY)
    np.testing.assert_array_equal(
        loaded[1].final_ownership, terminal_sample.final_ownership
    )
    batch = collate_replay_samples(loaded)
    assert batch.targets.policy_mask.tolist() == [True, False]
    assert not bool(batch.inputs.legal_action_mask[1].any())


def test_decode_materializes_each_npz_member_exactly_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_replay_shard(
        tmp_path / "decode-once.npz",
        [sample_for(4), sample_for(4), sample_for(4)],
    )
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
    np.testing.assert_array_equal(decoded.sample(1).stones, sample_for(4).stones)
    with pytest.raises(IndexError):
        decoded.sample(3)


def test_policy_confidence_weight_round_trips_and_legacy_shards_default_to_one(
    tmp_path,
) -> None:
    weighted = replace(sample_for(3), policy_weight=0.25)
    path = write_replay_shard(tmp_path / "weighted.npz", [weighted])

    loaded = read_replay_shard(path)

    assert loaded[0].policy_weight == pytest.approx(0.25)
    batch = collate_replay_samples(loaded)
    assert batch.targets.policy_weight is not None
    assert batch.targets.policy_weight.tolist() == pytest.approx([0.25])

    with np.load(path, allow_pickle=False) as archive:
        legacy_arrays = {
            name: archive[name] for name in archive.files if name != "policy_weight"
        }
    legacy_path = tmp_path / "legacy-without-policy-weight.npz"
    np.savez_compressed(legacy_path, **legacy_arrays)
    assert read_replay_shard(legacy_path)[0].policy_weight == 1.0


def test_action_relocation_has_one_pass_and_sentinel_gap() -> None:
    sample_nodes = 30
    batch_nodes = 75
    values = torch.arange(sample_nodes + 1)
    relocated = relocate_sample_actions(
        values,
        sample_nodes=sample_nodes,
        batch_max_nodes=batch_nodes,
        fill_value=-777,
    )
    assert torch.equal(relocated[:sample_nodes], values[:sample_nodes])
    assert torch.equal(
        relocated[sample_nodes:batch_nodes],
        torch.full((batch_nodes - sample_nodes,), -777),
    )
    assert relocated[batch_nodes] == values[sample_nodes]
    assert torch.equal(
        extract_sample_actions(
            relocated, sample_nodes=sample_nodes, batch_max_nodes=batch_nodes
        ),
        values,
    )

    small, large = sample_for(3), sample_for(5)
    small.policy[-1] = 0.25
    small.policy[:-1] *= 0.75 / small.policy[:-1].sum()
    legal = np.concatenate(((small.stones == -1), np.asarray([True])))
    small.soft_policy = np.power(np.where(legal, small.policy, 0), 0.25)
    small.soft_policy /= small.soft_policy.sum()
    small.__post_init__()
    batch = collate_replay_samples([small, large])
    assert batch.targets.policy[0, 30] == 0
    assert batch.targets.policy[0, batch.inputs.max_nodes] == pytest.approx(0.25)
    assert not bool(batch.inputs.legal_action_mask[0, 30])
    assert bool(batch.inputs.legal_action_mask[0, batch.inputs.max_nodes])


def test_validation_occurs_before_integer_cast_and_checks_policy_support() -> None:
    sample = sample_for(3)
    with pytest.raises(ReplaySchemaError, match="rules hash"):
        replace(sample, rules_hash=1)
    with pytest.raises(ReplaySchemaError, match="cannot be represented"):
        replace(
            sample,
            stones=np.full(sample.stones.shape, 255, dtype=np.uint16),
        )
    bad_policy = sample.policy.copy() * 0.9
    bad_policy[0] = 0.1
    with pytest.raises(ReplaySchemaError, match="illegal action"):
        replace(sample, policy=bad_policy)


def test_replay_augmentation_round_trips_all_d5_transforms() -> None:
    sample = sample_for(4)
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
        np.testing.assert_array_equal(restored.final_ownership, sample.final_ownership)
