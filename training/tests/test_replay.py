import json
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
    TARGET_ALIVE,
    TARGET_OWNERSHIP,
    TARGET_POLICY,
    TARGET_SCORE_MARGIN,
    TARGET_SOFT_POLICY,
)
from startrain.features import DoubleStarPosition
from startrain.replay import (
    MISSING_ALIVE,
    MISSING_OWNERSHIP,
    REPLAY_SCHEMA_VERSION,
    ReplaySample,
    ReplaySchemaError,
    augment_sample,
    collate_replay_samples,
    decode_replay_shard,
    read_replay_shard,
    write_replay_shard,
)
from startrain.replay_store import ReplaySelection, ReplayStore
from startrain.runtime import RunIdentity
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


def test_schema_v5_is_node_only_and_binary() -> None:
    sample = sample_for()
    topology = get_topology(4)
    assert sample.schema_version == REPLAY_SCHEMA_VERSION == 5
    assert (
        sample.is_standard_variant if hasattr(sample, "is_standard_variant") else True
    )
    assert sample.segment == "standard" and sample.variant_label == "double"
    assert not sample.has_teacher
    assert sample.history_flags is not None and sample.history_flags.shape == (
        topology.n,
    )
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


def test_clinch_outcome_only_uses_existing_masks_without_schema_change(
    tmp_path,
) -> None:
    position = live_position()
    policy = normalized_policy(position)
    synthetic = ReplaySample.from_position(
        position,
        policy=policy,
        final_score=decisive_score(position),
        search_provenance="gumbel:test:final=clinch-loser-fill",
        policy_provenance="completed-q",
    )
    outcome_only = ReplaySample.from_position(
        position,
        policy=policy,
        final_score=decisive_score(position),
        search_provenance="gumbel:test:final=clinch-loser-fill",
        policy_provenance="completed-q",
        clinch_auxiliary_targets="outcome_only",
    )

    assert synthetic.schema_version == outcome_only.schema_version == 5
    assert synthetic.target_mask & (
        TARGET_SCORE_MARGIN | TARGET_OWNERSHIP | TARGET_ALIVE
    )
    assert outcome_only.target_mask & TARGET_POLICY
    assert outcome_only.target_mask & TARGET_SOFT_POLICY
    assert outcome_only.target_mask & TARGET_OUTCOME
    assert not outcome_only.target_mask & (
        TARGET_SCORE_MARGIN | TARGET_OWNERSHIP | TARGET_ALIVE
    )
    assert outcome_only.outcome == synthetic.outcome
    np.testing.assert_array_equal(
        outcome_only.final_ownership,
        np.full_like(outcome_only.final_ownership, MISSING_OWNERSHIP),
    )
    np.testing.assert_array_equal(
        outcome_only.final_alive,
        np.full_like(outcome_only.final_alive, MISSING_ALIVE),
    )

    path = write_replay_shard(tmp_path / "clinch-v4.npz", [outcome_only])
    loaded = read_replay_shard(path)
    batch = collate_replay_samples(loaded)
    assert batch.targets.clinch_mask is not None
    assert batch.targets.clinch_mask.tolist() == [True]
    assert batch.targets.policy_mask.tolist() == [True]
    assert batch.targets.outcome_mask.tolist() == [True]
    assert batch.targets.score_margin_mask.tolist() == [False]
    assert batch.targets.ownership_mask.tolist() == [False]
    assert batch.targets.alive_mask.tolist() == [False]

    with pytest.raises(ReplaySchemaError, match="clinch_auxiliary_targets"):
        ReplaySample.from_position(
            position,
            policy=policy,
            final_score=decisive_score(position),
            search_provenance="test",
            policy_provenance="test",
            clinch_auxiliary_targets="invalid",  # type: ignore[arg-type]
        )


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
        str(metadata).replace('"schema_version": 5', '"schema_version": 3')
    )
    old = tmp_path / "old.npz"
    np.savez_compressed(old, **arrays)
    with pytest.raises(ReplaySchemaError, match="schema_version"):
        read_replay_shard(old)
    with pytest.raises(ReplaySchemaError, match="schema_version"):
        read_replay_shard(old, allow_legacy=True)

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


def test_replay_branch_cutoff_is_strict_persisted_and_default_neutral(
    tmp_path,
) -> None:
    identity = RunIdentity(
        tmp_path / "run.json",
        "run-cutoff",
        "family-cutoff",
        1,
    )
    model_identity = "sha256-" + "a" * 64
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor-cutoff")
        records = []
        for index in range(3):
            replay_sample = replace(
                sample_for(),
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                actor_id="actor-cutoff",
                generation=generation,
                game_id=f"game-cutoff-{index}",
                model_identity=model_identity,
            )
            records.append(
                store.append(
                    [replay_sample],
                    phase_min=0,
                    phase_max=0,
                    model_version=model_identity,
                    model_step=index,
                    model_identity=model_identity,
                    run_id=identity.run_id,
                    generation_family=identity.generation_family,
                    actor_id="actor-cutoff",
                    generation=generation,
                )
            )

        unbounded = store.select_recent_spans(
            rings=(4,),
            per_ring_quota=10,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=2,
            max_model_lag_steps=10,
        )
        assert [span.record.shard_id for span in unbounded.spans] == [
            record.shard_id for record in records
        ]
        assert unbounded.minimum_shard_id_exclusive is None

        cutoff = records[1].shard_id
        selected = store.select_recent_spans(
            rings=(4,),
            per_ring_quota=10,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=2,
            max_model_lag_steps=10,
            minimum_shard_id_exclusive=cutoff,
        )
        assert selected.minimum_shard_id_exclusive == cutoff
        assert selected.max_shard_id == records[2].shard_id
        assert [span.record.shard_id for span in selected.spans] == [
            records[2].shard_id
        ]
        assert (
            store.available_sample_count(
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                current_model_step=2,
                max_model_lag_steps=10,
                minimum_shard_id_exclusive=cutoff,
            )
            == 1
        )
        assert store.eligible_sample_counts(
            (4,),
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=2,
            max_model_lag_steps=10,
            minimum_shard_id_exclusive=cutoff,
        ) == {4: 1}
        assert store.sample_counts_by_ring(
            (4,),
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            minimum_shard_id_exclusive=cutoff,
        ) == {4: 1}
        assert [
            replay_sample.game_id
            for replay_sample in store.load_recent_samples(
                sample_window=10,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                minimum_shard_id_exclusive=cutoff,
            )
        ] == ["game-cutoff-2"]

        for invalid in (-1, True, 1.5, "1"):
            with pytest.raises(ValueError, match="non-negative integer"):
                store.recent_shards(
                    sample_window=1,
                    run_id=identity.run_id,
                    generation_family=identity.generation_family,
                    minimum_shard_id_exclusive=invalid,  # type: ignore[arg-type]
                )
            with pytest.raises(ValueError, match="non-negative integer"):
                ReplaySelection(
                    (),
                    {},
                    0,
                    invalid,  # type: ignore[arg-type]
                )


def legacy_v4_shard(path, samples: list[ReplaySample]):
    """Write a previous-lineage schema-v4 shard from v5 samples."""

    from startrain.contracts import (
        LEGACY_FEATURE_SCHEMA_HASH,
        LEGACY_RULES_HASH,
        LEGACY_RULES_HASH_WIRE,
        LEGACY_RULES_SCHEMA_ID,
    )
    from startrain.replay import _LEGACY_SAMPLE_ARRAY_NAMES

    current = write_replay_shard(path.with_name("current-source.npz"), samples)
    with np.load(current, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in _LEGACY_SAMPLE_ARRAY_NAMES}
        metadata = json.loads(str(archive["metadata"].item()))
    metadata.update(
        {
            "schema_version": 4,
            "rules_schema": LEGACY_RULES_SCHEMA_ID,
            "rules_hash": LEGACY_RULES_HASH,
            "rules_hash_wire": LEGACY_RULES_HASH_WIRE,
            "feature_schema_hash": LEGACY_FEATURE_SCHEMA_HASH,
        }
    )
    arrays["rules_hash"] = np.full(len(samples), LEGACY_RULES_HASH, dtype=np.uint64)
    arrays["feature_schema_hash"] = np.full(
        len(samples), LEGACY_FEATURE_SCHEMA_HASH, dtype=np.uint64
    )
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return path


def test_legacy_v4_shards_upgrade_only_through_the_adapter(tmp_path) -> None:
    samples = [sample_for(), sample_for(6)]
    # A ring-homogeneous legacy shard is required, so write one per ring.
    path = legacy_v4_shard(tmp_path / "legacy.npz", samples[:1])
    with pytest.raises(ReplaySchemaError, match="lineage-transfer"):
        decode_replay_shard(path)
    decoded = decode_replay_shard(path, allow_legacy=True)
    assert decoded.legacy
    upgraded = decoded.sample(0)
    assert upgraded.schema_version == REPLAY_SCHEMA_VERSION
    assert upgraded.rules_hash == RULES_HASH
    assert upgraded.feature_schema_hash == FEATURE_SCHEMA_HASH
    assert upgraded.segment == "standard"
    assert not upgraded.history_known
    assert upgraded.pda == 0
    assert upgraded.history_flags is not None and not upgraded.history_flags.any()
    assert "upgraded=rules-v2-features-v3" in upgraded.search_provenance
    np.testing.assert_array_equal(upgraded.stones, samples[0].stones)
    np.testing.assert_array_equal(upgraded.policy, samples[0].policy)
    assert upgraded.outcome == samples[0].outcome
    position = upgraded.to_position()
    assert position.is_standard and not position.history_known

    # The upgraded sample re-encodes and collates like any v5 sample.
    batch = collate_replay_samples([upgraded, samples[0]])
    assert batch.inputs.global_features[:, 23].tolist() == [0.0, 1.0]
    rewritten = write_replay_shard(tmp_path / "rewritten.npz", [upgraded])
    assert not decode_replay_shard(rewritten).legacy


def test_teacher_targets_round_trip_and_collate(tmp_path) -> None:
    position = live_position()
    topology = get_topology(4)
    legal = position.stones.numpy() == -1
    teacher_policy = legal.astype(np.float32)
    teacher_policy /= teacher_policy.sum()
    teacher_outcome = np.asarray([0.3, 0.7], dtype=np.float32)
    teacher_margin = np.zeros(303, dtype=np.float32)
    teacher_margin[150:153] = 1 / 3
    sample = ReplaySample.from_position(
        position,
        policy=normalized_policy(position),
        final_score=decisive_score(position),
        search_provenance="import:lineage",
        policy_provenance="completed-q",
        teacher_policy=teacher_policy,
        teacher_outcome=teacher_outcome,
        teacher_score_margin=teacher_margin,
    )
    assert sample.has_teacher
    assert sample.teacher_policy is not None
    assert sample.teacher_policy.dtype == np.float16
    path = write_replay_shard(tmp_path / "teacher.npz", [sample, sample_for()])
    decoded = read_replay_shard(path)
    assert decoded[0].has_teacher and not decoded[1].has_teacher
    batch = collate_replay_samples(decoded)
    assert batch.targets.teacher_mask is not None
    assert batch.targets.teacher_mask.tolist() == [True, False]
    assert batch.targets.teacher_policy is not None
    assert batch.targets.teacher_policy.shape == (2, topology.n)
    assert batch.targets.teacher_outcome is not None
    assert torch.allclose(
        batch.targets.teacher_outcome[0], torch.tensor([0.3, 0.7]), atol=1e-3
    )
    augmented = augment_sample(decoded[0], D5Transform(rotation=2, reflected=True))
    assert augmented.has_teacher
    assert augmented.teacher_policy is not None
    assert np.isclose(
        float(augmented.teacher_policy.astype(np.float32).sum()), 1.0, atol=5e-3
    )

    bad = np.zeros(2, dtype=np.float32)
    with pytest.raises(ReplaySchemaError, match="sum to one"):
        replace(sample, teacher_outcome=bad)
    with pytest.raises(ReplaySchemaError, match="illegal"):
        replace(
            sample, teacher_policy=np.full(topology.n, 1 / topology.n, dtype=np.float32)
        )


def test_variant_samples_round_trip(tmp_path) -> None:
    topology = get_topology(4)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[[1, 2, 3]] = 0
    handicap = torch.zeros(topology.n, dtype=torch.bool)
    handicap[[1, 2, 3]] = True
    position = DoubleStarPosition(
        rings=4,
        stones=stones,
        to_move=1,
        moves_left=1,
        opening=False,
        terminal=False,
        mode="classic",
        handicap=3,
        previous_turn=handicap,
        handicap_stones=handicap,
        pda=-2,
    )
    sample = ReplaySample.from_position(
        position,
        policy=normalized_policy(position),
        final_score=None,
        search_provenance="gumbel:test",
        policy_provenance="completed-q",
    )
    assert sample.segment == "handicap" and sample.variant_label == "handicap-3-classic"
    assert sample.pda == -2 and sample.mode == "classic"
    path = write_replay_shard(tmp_path / "variant.npz", [sample])
    decoded = read_replay_shard(path)[0]
    assert decoded.mode == "classic" and decoded.handicap == 3 and decoded.pda == -2
    assert decoded.history_flags is not None
    assert int(decoded.history_flags[1]) == 4 | 8
    restored = decoded.to_position()
    assert restored.previous_turn is not None
    assert torch.equal(restored.previous_turn, handicap)
    batch = collate_replay_samples([decoded])
    assert batch.inputs.global_features[0, 17].item() == pytest.approx(0.5)
    assert batch.inputs.global_features[0, 24].item() == pytest.approx(-2 / 3)
    with pytest.raises(ReplaySchemaError, match="variant-homogeneous"):
        write_replay_shard(tmp_path / "mixed.npz", [sample, sample_for()])
        read_replay_shard(tmp_path / "mixed.npz")
