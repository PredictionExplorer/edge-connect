from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import assume, given, settings, strategies as st
from torch.utils.data import DataLoader

from startrain.inference import InferenceResponse
from startrain.learner import (
    LazyShardReplayDataset,
    LearnerLoop,
    SpawnedReplayLoaderPool,
    UniqueReplayBatchSampler,
    replay_selection_diagnostics,
)
from startrain.replay import collate_replay_samples
from startrain.replay_store import ReplaySelection, ReplaySpan, ReplayStore, ShardRecord
from startrain.runtime import RunIdentity
from startrain.selfplay import SelfPlayActor, SelfPlayConfig, SelfPlayIdentity


def selection_for(sizes, *, rings=None, starts=None):
    rings = rings or [10] * len(sizes)
    starts = starts or [0] * len(sizes)
    spans = []
    counts = {}
    for index, (count, ring, start) in enumerate(
        zip(sizes, rings, starts, strict=True)
    ):
        record = ShardRecord(
            shard_id=index + 1,
            path=Path(f"unused-{index}.npz"),
            created_ns=index,
            sample_count=start + count,
            ring=ring,
            phase_min=0,
            phase_max=275,
            model_version="sha256-" + "a" * 64,
            model_step=100 + index,
            model_identity="sha256-" + "a" * 64,
            run_id="run",
            generation_family="family",
            actor_id="actor",
            generation=0,
            game_count=1,
            checksum_sha256="a" * 64,
            state="ready",
            quarantine_reason=None,
        )
        spans.append(ReplaySpan(record, start, count))
        counts[ring] = counts.get(ring, 0) + count
    return ReplaySelection(tuple(spans), counts, len(spans))


def dataset_for(selection, epoch=0):
    return LazyShardReplayDataset(
        selection, seed=17, epoch=epoch, augmentation_enabled=False, shard_cache_size=16
    )


def sampler_for(
    dataset,
    *,
    batch_size=512,
    batches=None,
    diversity=4,
    rank=0,
    world_size=1,
    weights=None,
    epoch=0,
):
    if batches is None:
        batches = (
            sum(dataset.ring_count(r) // batch_size for r in dataset.rings)
            // world_size
        )
    return UniqueReplayBatchSampler(
        dataset,
        batch_size=batch_size,
        batches=batches,
        seed=17,
        epoch=epoch,
        ring_stratified=True,
        shards_per_batch=diversity,
        rank=rank,
        world_size=world_size,
        ring_weights=weights,
    )


@pytest.mark.parametrize(
    "sizes", [[128] * 4, [1] * 512, [50] * 20 + [24], [512] * 4 + [128] * 4, [513, 511]]
)
def test_all_short_mixed_and_tail_rows_are_packed_without_loss(sizes):
    selection = selection_for(sizes)
    dataset = dataset_for(selection)
    chunks = dataset.shard_batch_chunks(512)
    assert sum(chunk.sample_count for chunk in chunks) == sum(sizes)
    batches = list(sampler_for(dataset))
    flattened = [index for batch in batches for index in batch]
    assert len(flattened) == len(set(flattened)) == sum(sizes)
    assert set(flattened) == set(range(sum(sizes)))
    assert all(len(batch) == 512 for batch in batches)
    assert list(sampler_for(dataset)) == batches


@settings(max_examples=100, deadline=None)
@given(
    sizes=st.lists(st.integers(1, 80), min_size=1, max_size=16),
    batch_size=st.integers(1, 96),
    diversity=st.integers(1, 12),
)
def test_arbitrary_span_packing_matches_row_capacity_and_feasible_diversity(
    sizes, batch_size, diversity
):
    assume(sum(sizes) >= batch_size)
    dataset = dataset_for(selection_for(sizes))
    batches = list(sampler_for(dataset, batch_size=batch_size, diversity=diversity))
    remaining = set(range(sum(sizes)))
    sources = {index: dataset.reference(index).shard_id for index in remaining}
    assert len(batches) == sum(sizes) // batch_size
    for batch in batches:
        feasible = min(
            diversity, batch_size, len({sources[index] for index in remaining})
        )
        assert len(batch) == len(set(batch)) == batch_size
        assert set(batch) <= remaining
        assert len({sources[index] for index in batch}) >= feasible
        remaining.difference_update(batch)
    assert len(remaining) == sum(sizes) % batch_size


def test_diversity_is_a_minimum_when_feasible_and_can_require_more_files():
    dataset = dataset_for(selection_for([50] * 10 + [12]))
    batch = next(iter(sampler_for(dataset)))
    assert len({dataset.reference(index).shard_id for index in batch}) == 11
    one_file = dataset_for(selection_for([1024]))
    assert len(list(sampler_for(one_file))) == 2
    irregular = dataset_for(selection_for([20] * 8))
    for batch in sampler_for(irregular, batch_size=5, diversity=4):
        assert len(batch) == 5
    one_batch = next(iter(sampler_for(irregular, batch_size=5, batches=1, diversity=4)))
    assert len({irregular.reference(index).shard_id for index in one_batch}) >= 4


def test_partial_selected_spans_never_escape_their_original_row_bounds():
    selection = selection_for([127, 129, 256], starts=[31, 63, 17])
    dataset = dataset_for(selection)
    references = [
        dataset.reference(index) for index in next(iter(sampler_for(dataset)))
    ]
    for reference in references:
        span = selection.spans[reference.shard_id - 1]
        assert (
            span.sample_start
            <= reference.sample_offset
            < span.sample_start + span.sample_count
        )
    assert (
        len({(reference.shard_id, reference.sample_offset) for reference in references})
        == 512
    )


def test_rank_partitioning_and_weighted_homogeneous_rings_share_one_unique_plan():
    selection = selection_for([128] * 48, rings=[4] * 16 + [10] * 32)
    dataset = dataset_for(selection)
    flattened = []
    per_ring = {4: 0, 10: 0}
    for rank in range(2):
        batches = list(
            sampler_for(dataset, rank=rank, world_size=2, weights={4: 1, 10: 2})
        )
        assert len(batches) == 6
        for batch in batches:
            rings = {
                selection.spans[dataset.reference(index).shard_id - 1].record.ring
                for index in batch
            }
            assert len(rings) == 1
            per_ring[rings.pop()] += 1
            flattened.extend(batch)
    assert per_ring == {4: 4, 10: 8}
    assert len(flattened) == len(set(flattened)) == 6144


def test_capacity_readiness_and_metadata_use_actual_packable_rows():
    selection = selection_for([128] * 8)
    fake = SimpleNamespace(
        train_config=SimpleNamespace(per_rank_batch_size=512),
        data_config=SimpleNamespace(ring_stratified=True, shards_per_batch=4),
        learner_config=SimpleNamespace(
            steps_per_window=1000,
            minimum_unique_samples_per_ring=1,
            minimum_replay_samples=1,
        ),
        world_size=1,
        _active_ring_weights=lambda: {10: 1},
        _active_replay_counts=lambda counts: counts,
    )
    fake._available_batch_capacity = lambda counts: (
        LearnerLoop._available_batch_capacity(fake, counts)
    )
    capacity = LearnerLoop._maximum_unique_batches(fake, selection)
    assert capacity == 2
    assert LearnerLoop._replay_is_ready(fake, {10: 1024})
    assert not LearnerLoop._replay_is_ready(fake, {10: 511})
    assert len(list(sampler_for(dataset_for(selection), batches=capacity))) == capacity
    diagnostics = replay_selection_diagnostics(
        selection, batch_size=512, current_model_step=110
    )
    assert diagnostics["legacy_chunk_rows_by_ring"] == {"10": 0}
    assert diagnostics["short_span_rows_by_ring"] == {"10": 1024}
    assert diagnostics["packable_rows_by_ring"] == {"10": 1024}
    assert diagnostics["batch_capacity_by_ring"] == {"10": 2}
    assert diagnostics["selected_model_age"]["minimum"] == 3
    assert diagnostics["selected_model_age"]["maximum"] == 10


class UniformEvaluator:
    model_version = "sha256-" + "a" * 64
    model_identity = model_version
    model_step = 10

    def evaluate(self, requests):
        return InferenceResponse(
            list(requests.tokens),
            [0.0] * len(requests),
            list(requests.legal_offsets),
            [0.0] * len(requests.legal_actions),
        )


@pytest.fixture
def real_streamed_replay(tmp_path):
    native = pytest.importorskip("star_native")
    identity = RunIdentity(
        tmp_path / "run.json", "streamed-packing", "streamed-family", 1
    )
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor")
        actor = SelfPlayActor(
            native,
            UniformEvaluator(),
            store,
            SelfPlayConfig(
                rings=4,
                batch_size=1,
                games=12,
                stream_completed_games=True,
                fast_probability=0,
                full_probability=1,
                fast_simulations=1,
                full_simulations=1,
                simulation_ring_exponent=0,
                max_considered=1,
                shard_size=4096,
            ),
            SelfPlayIdentity(
                identity.run_id, identity.generation_family, "actor", generation
            ),
        )
        summaries = actor.run()
        selection = store.select_recent_spans(
            rings=(4,),
            per_ring_quota=1000,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=12,
            max_model_lag_steps=2,
        )
        assert len(selection.spans) == 12
        assert all(span.sample_count == 50 for span in selection.spans)
        assert selection.sample_count == 600
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id, generation_family=identity.generation_family
            )
            == 600
        )
        yield store, identity, selection, {summary.game_id for summary in summaries}


@pytest.mark.native
@pytest.mark.parametrize("persistent", [False, True])
def test_real_streamed_games_reach_production_512_row_loaders(
    real_streamed_replay, persistent
):
    store, identity, selection, expected_games = real_streamed_replay
    store.set_gc_watermark("packing-test", selection)
    pool = (
        SpawnedReplayLoaderPool(
            num_workers=2,
            augmentation_enabled=False,
            shard_cache_size=16,
            pin_memory=False,
            prefetch_factor=1,
        )
        if persistent
        else None
    )
    seen = set()
    try:
        for epoch in range(4):
            dataset = dataset_for(selection, epoch)
            sampler = sampler_for(dataset, epoch=epoch)
            indices = next(iter(sampler))
            seen.update(dataset[index].game_id for index in indices)
            if pool is not None:
                pool.rebind(dataset, sampler)
                loader = pool.loader
            else:
                loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=collate_replay_samples,
                    num_workers=0,
                )
            batch = next(iter(loader))
            assert batch.inputs.batch_size == 512
            assert batch.inputs.rings.tolist() == [4] * 512
            assert bool(batch.targets.outcome_mask.all())
            assert bool(batch.targets.policy_mask.all())
        assert seen == expected_games
        gc = store.collect_garbage(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            dry_run=False,
        )
        assert gc["deleted_shards"] == 0
    finally:
        if pool is not None:
            pool.shutdown()
        store.clear_gc_watermark("packing-test")
