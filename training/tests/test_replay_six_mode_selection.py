"""The declared six-mode objective survives selection, packing and retention."""

from collections import Counter
from dataclasses import replace

import pytest

from startrain.learner import LazyShardReplayDataset, UniqueReplayBatchSampler
from startrain.replay_store import ReplaySelection, ReplaySpan, ReplayStore
from test_pipeline_core import append_replay, make_replay_sample, run_identity


QUOTAS = {"standard": 1 / 6, "classic": 1 / 6, "handicap": 1 / 3, "pie": 1 / 3}
SHARES = {"handicap": 0.5, "pie": 0.5}
MODES = (
    ("double", 1, False),
    ("classic", 1, False),
    ("classic", 2, False),
    ("double", 2, False),
    ("classic", 1, True),
    ("double", 1, True),
)


def append_mode(
    store, identity, mode, count, *, handicap=1, pie=False, step=100, ring=4
):
    serial = store.connection.execute(
        "SELECT COALESCE(MAX(id), 0) FROM shards"
    ).fetchone()[0]
    samples = [
        replace(
            make_replay_sample(
                ring, identity=identity, game_id=f"six-mode-{serial}-{index}"
            ),
            mode=mode,
            handicap=handicap,
            pie=pie,
            moves_left=handicap,
        )
        for index in range(count)
    ]
    return append_replay(store, samples, identity, model_step=step)


def select(store, identity, total, **changes):
    options = dict(
        rings=(4,),
        per_ring_quota=total,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        current_model_step=100,
        max_model_lag_steps=10,
        segment_quotas=QUOTAS,
        within_segment_classic_shares=SHARES,
    )
    options.update(changes)
    return store.select_recent_spans(**options)


def mode_counts(selection):
    result = Counter()
    for span in selection.spans:
        variant = span.record.variant
        label = (
            "handicap-" + variant.rsplit("-", 1)[-1]
            if variant.startswith("handicap-")
            else variant
        )
        result[label] += span.sample_count
    return result


def test_six_modes_reach_actual_no_replacement_distributed_loader(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        for mode, handicap, pie in MODES:
            append_mode(store, identity, mode, 20, handicap=handicap, pie=pie)
        # Newer double shards used to consume the entire handicap/pie quota.
        legacy = select(store, identity, 60, within_segment_classic_shares=None)
        assert mode_counts(legacy) == {
            "double": 10,
            "classic": 10,
            "handicap-double": 20,
            "pie-double": 20,
        }
        selected = select(store, identity, 60)
        expected = dict.fromkeys(
            (
                "double",
                "classic",
                "handicap-double",
                "handicap-classic",
                "pie-double",
                "pie-classic",
            ),
            10,
        )
        assert mode_counts(selected) == expected
        assert selected.samples_by_segment == legacy.samples_by_segment
        assert selected == select(
            store,
            identity,
            60,
            segment_quotas=dict(reversed(list(QUOTAS.items()))),
            within_segment_classic_shares=dict(reversed(list(SHARES.items()))),
        )
        assert len({span.record.shard_id for span in selected.spans}) == 6
        assert all(span.sample_start == 10 for span in selected.spans)
        dataset = LazyShardReplayDataset(
            selected, seed=7, epoch=0, augmentation_enabled=False, shard_cache_size=8
        )
        indices_by_rank = [
            [
                index
                for batch in UniqueReplayBatchSampler(
                    dataset,
                    batch_size=6,
                    batches=5,
                    seed=7,
                    epoch=0,
                    ring_stratified=True,
                    shards_per_batch=4,
                    rank=rank,
                    world_size=2,
                )
                for index in batch
            ]
            for rank in (0, 1)
        ]
        assert set(indices_by_rank[0]).isdisjoint(indices_by_rank[1])
        assert set(indices_by_rank[0] + indices_by_rank[1]) == set(range(60))
        rows = [dataset[index] for indices in indices_by_rank for index in indices]
        assert len({row.game_id for row in rows}) == 60
        assert Counter(row.variant_label for row in rows) == {
            "double": 10,
            "classic": 10,
            "handicap-2-double": 10,
            "handicap-2-classic": 10,
            "pie-double": 10,
            "pie-classic": 10,
        }


@pytest.mark.parametrize(
    "classic,double,share,total,expected",
    [
        (20, 20, 0.5, 11, (6, 5)),
        (20, 20, 0.25, 10, (3, 7)),
        (2, 20, 0.5, 10, (2, 8)),
        (20, 2, 0.5, 10, (8, 2)),
        (0, 20, 0.5, 10, (0, 10)),
        (20, 0, 0.5, 10, (10, 0)),
        (2, 3, 0.5, 10, (2, 3)),
        (20, 20, 0, 10, (0, 10)),
        (20, 20, 1, 10, (10, 0)),
        (20, 2, 0, 10, (8, 2)),
        (2, 20, 1, 10, (2, 8)),
    ],
)
def test_mode_capacity_spills_inside_segment(
    tmp_path, classic, double, share, total, expected
):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        for mode, count in (("classic", classic), ("double", double)):
            if count:
                append_mode(store, identity, mode, count, pie=True)
        selected = select(
            store,
            identity,
            total,
            segment_quotas={"pie": 1.0},
            within_segment_classic_shares={"pie": share},
        )
        counts = mode_counts(selected)
        assert (counts["pie-classic"], counts["pie-double"]) == expected
        assert selected.sample_count == min(total, classic + double)


def test_aggregate_shortfall_remains_proportional_with_missing_modes(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        append_mode(store, identity, "double", 5)
        append_mode(store, identity, "classic", 5)
        append_mode(store, identity, "classic", 40, handicap=2)
        append_mode(store, identity, "double", 40, pie=True)
        selected = select(store, identity, 60)
        assert mode_counts(selected) == {
            "double": 5,
            "classic": 5,
            "handicap-classic": 25,
            "pie-double": 25,
        }
        assert selected.samples_by_segment == {
            "standard": 5,
            "classic": 5,
            "handicap": 25,
            "pie": 25,
        }


def test_handicap_severities_share_mode_recency_and_exact_selected_tails(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        old = append_mode(store, identity, "classic", 8, handicap=2)
        recent = append_mode(store, identity, "classic", 3, handicap=4)
        double = append_mode(store, identity, "double", 10, handicap=3)
        selected = select(store, identity, 12, segment_quotas={"handicap": 1.0})
        assert [
            (span.record.shard_id, span.sample_start, span.sample_count)
            for span in selected.spans
        ] == [
            (old.shard_id, 5, 3),
            (recent.shard_id, 0, 3),
            (double.shard_id, 4, 6),
        ]


@pytest.mark.parametrize(
    "column,value",
    [
        ("rules_hash", "0000000000000000"),
        ("feature_schema_hash", "0000000000000000"),
        ("run_id", "another-run"),
        ("generation_family", "another-family"),
        ("model_step", 89),
        ("model_step", 101),
        ("state", "quarantined"),
    ],
)
def test_mode_capacity_and_recency_keep_all_eligibility_filters(
    tmp_path, column, value
):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        cutoff = append_mode(store, identity, "classic", 20, pie=True)
        classic = append_mode(store, identity, "classic", 4, pie=True)
        double = append_mode(store, identity, "double", 20, pie=True)
        excluded = append_mode(store, identity, "classic", 20, pie=True)
        store.connection.execute(
            f"UPDATE shards SET {column} = ? WHERE id = ?", (value, excluded.shard_id)
        )
        store.connection.commit()
        append_mode(store, identity, "classic", 20, pie=True, ring=6)
        selected = select(
            store,
            identity,
            12,
            segment_quotas={"pie": 1.0},
            minimum_shard_id_exclusive=cutoff.shard_id,
        )
        assert mode_counts(selected) == {"pie-classic": 4, "pie-double": 8}
        assert {span.record.shard_id for span in selected.spans} == {
            classic.shard_id,
            double.shard_id,
        }


def test_mode_filtered_recency_obeys_upper_snapshot_boundary(tmp_path, monkeypatch):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        classic = append_mode(store, identity, "classic", 20, pie=True)
        double = append_mode(store, identity, "double", 20, pie=True)
        original = store.recent_shards
        appended = []

        def concurrent_append(**kwargs):
            if not appended:
                appended.append(append_mode(store, identity, "classic", 20, pie=True))
            return original(**kwargs)

        monkeypatch.setattr(store, "recent_shards", concurrent_append)
        selected = select(store, identity, 12, segment_quotas={"pie": 1.0})
        assert selected.max_shard_id == double.shard_id
        assert {span.record.shard_id for span in selected.spans} == {
            classic.shard_id,
            double.shard_id,
        }


def test_real_gc_preserves_six_mode_floor_pins_and_lifetime_sample_credit(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        pinned = append_mode(store, identity, "classic", 2, pie=True, step=0)
        cutoff = append_mode(store, identity, "classic", 20, pie=True)
        stale = append_mode(store, identity, "classic", 20, pie=True, step=89)
        for mode, handicap, pie in MODES:
            append_mode(store, identity, mode, 20, handicap=handicap, pie=pie)
        for _ in range(3):
            append_mode(store, identity, "double", 20, pie=True)
        store.set_gc_watermark(
            "active-reader",
            ReplaySelection((ReplaySpan(pinned, 0, 2),), {4: 2}, pinned.shard_id),
        )
        before = select(store, identity, 60, minimum_shard_id_exclusive=cutoff.shard_id)
        credit = store.total_committed_sample_count(
            run_id=identity.run_id, generation_family=identity.generation_family
        )
        kwargs = dict(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            minimum_samples_per_ring=60,
            current_model_step=100,
            max_model_lag_steps=10,
            minimum_shard_id_exclusive=cutoff.shard_id,
            segment_quotas=QUOTAS,
            within_segment_classic_shares=SHARES,
        )
        dry = store.collect_garbage(**kwargs, dry_run=True)
        assert dry["sample_floor_rows"] == 60 and dry["sample_floor_shards"] == 6
        assert pinned.path.exists() and stale.path.exists()
        actual = store.collect_garbage(**kwargs, dry_run=False)
        assert actual["deleted_shards"] == dry["candidate_shards"] > 0
        assert pinned.path.exists() and not stale.path.exists()
        assert not cutoff.path.exists()
        assert (
            select(store, identity, 60, minimum_shard_id_exclusive=cutoff.shard_id)
            == before
        )
        assert all(span.record.path.exists() for span in before.spans)
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id, generation_family=identity.generation_family
            )
            == credit
        )


@pytest.mark.parametrize(
    "shares",
    [
        [],
        {"standard": 0.5},
        {"pie": True},
        {"pie": "0.5"},
        {"pie": float("nan")},
        {"handicap": float("inf")},
        {"pie": -0.1},
        {"handicap": 1.1},
    ],
)
def test_invalid_mode_shares_are_rejected_before_selection_or_gc(tmp_path, shares):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        with pytest.raises(ValueError, match="classic shares"):
            select(store, identity, 60, within_segment_classic_shares=shares)
        with pytest.raises(ValueError, match="classic shares"):
            store.collect_garbage(
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                retain_shards_per_ring=1,
                dry_run=False,
                segment_quotas=QUOTAS,
                within_segment_classic_shares=shares,
            )


def test_mode_shares_require_aggregate_quota_authority(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        with pytest.raises(ValueError, match="require segment quotas"):
            select(store, identity, 60, segment_quotas=None)
        assert (
            select(
                store,
                identity,
                60,
                segment_quotas=None,
                within_segment_classic_shares=None,
            ).sample_count
            == 0
        )
