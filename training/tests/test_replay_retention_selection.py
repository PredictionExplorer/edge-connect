from dataclasses import replace

import pytest

from startrain.replay_store import ReplaySelection, ReplaySpan, ReplayStore
from test_pipeline_core import append_replay, make_replay_sample, run_identity


def append_rows(store, identity, label, count, *, step=100, rings=4):
    mode, pie, handicap = {
        "standard": ("double", False, 1),
        "classic": ("classic", False, 1),
        "pie": ("double", True, 1),
        "handicap": ("double", False, 2),
    }[label]
    serial = store.connection.execute(
        "SELECT COALESCE(MAX(id), 0) FROM shards"
    ).fetchone()[0]
    samples = [
        replace(
            make_replay_sample(
                rings, identity=identity, game_id=f"retention-{serial}-{index}"
            ),
            mode=mode,
            pie=pie,
            handicap=handicap,
            moves_left=handicap,
        )
        for index in range(count)
    ]
    return append_replay(store, samples, identity, model_step=step)


def selection(store, identity, quota, weights=None, **changes):
    return store.select_recent_spans(
        rings=(4,),
        per_ring_quota=quota,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        current_model_step=100,
        max_model_lag_steps=10,
        segment_quotas=weights,
        **changes,
    )


def test_shortfall_is_proportional_deterministic_and_capacity_bounded(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        for label, count in (
            ("classic", 5),
            ("standard", 10),
            ("pie", 100),
            ("handicap", 100),
        ):
            append_rows(store, identity, label, count)
        weights = dict.fromkeys(("classic", "standard", "pie", "handicap"), 0.25)
        left = selection(store, identity, 60, weights)
        right = selection(store, identity, 60, dict(reversed(list(weights.items()))))
        assert left == right
        assert left.sample_count == 60
        assert left.samples_by_segment == {
            "classic": 5,
            "standard": 10,
            "handicap": 23,
            "pie": 22,
        }
        assert len({span.record.shard_id for span in left.spans}) == len(left.spans)


def test_sample_floor_preserves_many_tiny_shards_beyond_file_count(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        records = [append_rows(store, identity, "standard", 3) for _ in range(8)]
        before = selection(store, identity, 15)
        kwargs = dict(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=2,
            minimum_samples_per_ring=15,
            current_model_step=100,
            max_model_lag_steps=10,
        )
        dry = store.collect_garbage(**kwargs, dry_run=True)
        assert dry["sample_floor_rows"] == 15
        assert dry["sample_floor_shards"] == 5
        assert dry["candidate_shards"] == 3
        assert all(record.path.exists() for record in records)
        actual = store.collect_garbage(**kwargs, dry_run=False)
        assert actual["deleted_shards"] == 3
        assert selection(store, identity, 15) == before
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id, generation_family=identity.generation_family
            )
            == 24
        )


@pytest.mark.parametrize("column", ["rules_hash", "feature_schema_hash"])
def test_segment_capacity_only_counts_compatible_history(tmp_path, column):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        incompatible = append_rows(store, identity, "classic", 20)
        append_rows(store, identity, "standard", 20)
        store.connection.execute(
            f"UPDATE shards SET {column} = ? WHERE id = ?",
            ("0000000000000000", incompatible.shard_id),
        )
        store.connection.commit()
        selected = selection(store, identity, 20, {"classic": 0.5, "standard": 0.5})
        assert selected.sample_count == 20
        assert selected.samples_by_segment["classic"] == 0
        assert selected.samples_by_segment["standard"] == 20


def test_gc_floor_preserves_scarce_segment_and_pins_with_correct_age_and_cutoff(
    tmp_path,
):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        protected = append_rows(store, identity, "standard", 2, step=0)
        obsolete = append_rows(store, identity, "classic", 6, step=0)
        rare = append_rows(store, identity, "classic", 5)
        for _ in range(6):
            append_rows(store, identity, "standard", 2)
        store.set_gc_watermark(
            "active-reader",
            ReplaySelection(
                (ReplaySpan(protected, 0, 2),),
                {4: 2},
                protected.shard_id,
            ),
        )
        weights = {"classic": 0.5, "standard": 0.5}
        before = selection(
            store, identity, 10, weights, minimum_shard_id_exclusive=obsolete.shard_id
        )
        stats = store.collect_garbage(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            dry_run=False,
            minimum_samples_per_ring=10,
            current_model_step=100,
            max_model_lag_steps=10,
            minimum_shard_id_exclusive=obsolete.shard_id,
            segment_quotas=weights,
        )
        assert stats["deleted_shards"] > 0
        assert protected.path.exists() and rare.path.exists()
        assert not obsolete.path.exists()
        assert (
            selection(
                store,
                identity,
                10,
                weights,
                minimum_shard_id_exclusive=obsolete.shard_id,
            )
            == before
        )


def test_unfilled_sample_floor_keeps_all_available_eligible_history(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        records = [append_rows(store, identity, "standard", 2) for _ in range(5)]
        stats = store.collect_garbage(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            dry_run=False,
            minimum_samples_per_ring=100,
            current_model_step=100,
            max_model_lag_steps=10,
        )
        assert stats["deleted_shards"] == 0
        assert stats["sample_floor_rows"] == 10
        assert all(record.path.exists() for record in records)
        with pytest.raises(ValueError, match="model step and lag"):
            store.collect_garbage(
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                retain_shards_per_ring=1,
                dry_run=False,
                minimum_samples_per_ring=100,
            )
