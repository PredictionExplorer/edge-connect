from dataclasses import asdict, fields, replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from startrain.inference import InferenceResponse
from startrain.selfplay import (
    GameVariant,
    SelfPlayActor,
    SelfPlayConfig,
    SelfPlayIdentity,
    VariantMixtureConfig,
)


VARIANTS = (
    GameVariant(mode="classic"),
    GameVariant(mode="double"),
    GameVariant(mode="classic", pie=True),
    GameVariant(mode="double", pie=True),
    GameVariant(mode="classic", handicap=9),
    GameVariant(mode="double", handicap=9),
)


class Evaluator:
    model_version = "streaming-test"
    model_identity = "streaming-test"
    model_step = 7

    def evaluate(self, requests):
        return InferenceResponse(
            tokens=list(requests.tokens),
            values=[
                0.8 if int(seat) == 0 else -0.8 for seat in requests.states.to_move
            ],
            policy_offsets=list(requests.legal_offsets),
            policy_logits=[0.0] * len(requests.legal_actions),
        )


class Sink:
    def __init__(self, on_append=lambda _samples: None):
        self.samples = []
        self.on_append = on_append
        self.publications = []

    def append(self, samples, **metadata):
        self.samples.extend(samples)
        self.publications.append(
            (tuple(sample.game_id for sample in samples), metadata)
        )
        self.on_append(samples)
        return SimpleNamespace(sample_count=len(samples))


def config(**changes):
    return replace(
        SelfPlayConfig(
            rings=4,
            games=5,
            batch_size=3,
            fast_simulations=1,
            full_simulations=8,
            simulation_reference_rings=4,
            simulation_ring_exponent=0.0,
            max_considered=4,
            record_fast_policy_targets=True,
            policy_surprise_weight=0.5,
            shard_size=10_000,
            variants=VariantMixtureConfig(
                enabled=True,
                asymmetric_pda_fraction=1.0,
                swap_dead_zone=0.0,
            ),
        ),
        **changes,
    )


def actor(native, selected, sink):
    return SelfPlayActor(
        native,
        Evaluator(),
        sink,
        selected,
        SelfPlayIdentity("stream-test", "stream-family", "stream-actor", 12),
    )


def sample_fingerprints(samples):
    result = {}
    for sample in samples:
        fingerprint = hashlib.sha256()
        for field in fields(sample):
            value = getattr(sample, field.name)
            fingerprint.update(field.name.encode())
            if isinstance(value, np.ndarray):
                fingerprint.update(str((value.dtype, value.shape)).encode())
                fingerprint.update(value.tobytes())
            else:
                fingerprint.update(repr(value).encode())
        key = (sample.game_id, sample.ply)
        assert key not in result
        result[key] = fingerprint.hexdigest()
    return result


def summary_records(summaries):
    result = {}
    for summary in summaries:
        record = asdict(summary)
        del record["row"]  # Physical slots are intentionally reusable.
        result[summary.game_id] = record
    return result


def near_terminal_native(native, snapshots):
    def state_batch(rings, rows, **variant):
        states = native.StateBatch(rings, rows, **variant)
        for row in range(rows):
            placements = states.node_count - (2 * row + 1)
            states.apply_many([row] * placements, list(range(placements)))
        snapshots.append(states)
        return states

    return SimpleNamespace(StateBatch=state_batch, SearchBatch=native.SearchBatch)


@pytest.mark.native
@pytest.mark.parametrize("rings", [4, 6, 8, 10])
def test_streaming_publishes_early_and_preserves_legacy_trajectories(rings):
    native = pytest.importorskip("star_native")
    runs = []
    for streaming in (False, True):
        snapshots = []
        terminal_at_publication = []
        sink = Sink(
            lambda _samples: terminal_at_publication.append(
                list(snapshots[-1].data().terminal)
            )
        )
        worker = actor(
            near_terminal_native(native, snapshots),
            config(
                rings=rings, games=3, batch_size=3, stream_completed_games=streaming
            ),
            sink,
        )
        summaries = worker.run()
        runs.append((sample_fingerprints(sink.samples), summary_records(summaries)))
        assert worker.metrics_snapshot().completed_games == 3
        assert worker.metrics_snapshot().dropped_games == 0
        assert all(terminal_at_publication[0]) is (not streaming)
    assert runs[0] == runs[1]


@pytest.mark.native
@pytest.mark.parametrize("variant", VARIANTS, ids=lambda variant: variant.label)
def test_game_seed_contract_matches_across_rolling_and_fixed_packings(variant):
    native = pytest.importorskip("star_native")
    reference = None
    for slots, rolling in ((1, False), (3, False), (2, True), (3, True)):
        sink = Sink()
        worker = actor(
            native,
            config(
                games=5,
                batch_size=slots,
                stream_completed_games=True,
                rolling_game_slots=rolling,
                seed_contract="game-v1",
            ).with_variant(variant),
            sink,
        )
        summaries = worker.run()
        result = (sample_fingerprints(sink.samples), summary_records(summaries))
        if reference is None:
            reference = result
        assert result == reference
        metrics = worker.metrics_snapshot()
        assert metrics.started_games == metrics.completed_games == 5
        assert metrics.refilled_games == (5 - slots if rolling else 0)
        assert metrics.dropped_games == 0
        if variant.pie:
            assert metrics.pie_swaps == 5
        for sample in sink.samples:
            full = "gumbel-completed-q:full:" in sample.search_provenance
            base = worker.config.simulation_budget(full=full)
            if sample.pda:
                high, low = worker.config.playout_budgets(
                    simulations=base, pda=abs(sample.pda)
                )
                expected = high if sample.pda > 0 else low
            else:
                expected = base
            assert f"simulations={expected}:" in sample.search_provenance


@pytest.mark.native
@pytest.mark.parametrize("streaming", [False, True])
def test_stop_keeps_completed_games_and_counts_only_unfinished_rows(streaming):
    native = pytest.importorskip("star_native")
    snapshots = []
    sink = Sink()
    worker = actor(
        near_terminal_native(native, snapshots),
        config(games=2, batch_size=2, stream_completed_games=streaming),
        sink,
    )
    summaries = worker.run(
        stop_requested=lambda: (
            bool(snapshots) and bool(snapshots[-1].data().terminal[0])
        )
    )
    assert len(summaries) == 1
    assert len(sink.samples) == summaries[0].samples == 1
    assert {sample.game_id for sample in sink.samples} == {summaries[0].game_id}
    metrics = worker.metrics_snapshot()
    assert metrics.started_games == 2
    assert metrics.completed_games == metrics.dropped_games == 1
    assert metrics.dropped_decisions == 1


@pytest.mark.native
def test_refill_stops_issuing_but_drains_all_assigned_games():
    native = pytest.importorskip("star_native")
    sink = Sink()
    worker = actor(
        native,
        config(
            games=8,
            batch_size=3,
            seed_contract="game-v1",
            stream_completed_games=True,
            rolling_game_slots=True,
        ),
        sink,
    )
    summaries = worker.run(stop_refill_requested=lambda: bool(sink.samples))
    assert len(summaries) == 3
    metrics = worker.metrics_snapshot()
    assert metrics.started_games == metrics.completed_games == 3
    assert metrics.dropped_games == metrics.refilled_games == 0
    assert len(sink.samples) == sum(item.samples for item in summaries)


@pytest.mark.native
def test_rolling_short_quota_does_not_start_unused_slots():
    native = pytest.importorskip("star_native")
    sink = Sink()
    worker = actor(
        native,
        config(
            games=2,
            batch_size=8,
            seed_contract="game-v1",
            stream_completed_games=True,
            rolling_game_slots=True,
        ),
        sink,
    )
    assert len(worker.run()) == 2
    assert worker.metrics_snapshot().started_games == 2


@pytest.mark.native
@pytest.mark.parametrize(
    "termination",
    [
        {
            "clinch_finalization": "loser-fill",
            "clinch_auxiliary_targets": "outcome_only",
        },
        {"exact_endgame_max_empty": 2, "exact_endgame_max_nodes": 1000},
    ],
)
def test_refilled_slots_preserve_clinch_and_exact_endgame_provenance(termination):
    native = pytest.importorskip("star_native")
    runs = []
    for rolling in (False, True):
        sink = Sink()
        worker = actor(
            native,
            config(
                games=4,
                batch_size=2,
                stream_completed_games=True,
                rolling_game_slots=rolling,
                seed_contract="game-v1",
                **termination,
            ),
            sink,
        )
        summaries = worker.run()
        runs.append((sample_fingerprints(sink.samples), summary_records(summaries)))
        assert (
            worker.metrics_snapshot().clinch_empty_nodes > 0
            or worker.metrics_snapshot().exact_endgame_solved > 0
        )
    assert runs[0] == runs[1]


@pytest.mark.native
def test_model_pin_change_at_last_move_cannot_publish_mixed_model_game():
    native = pytest.importorskip("star_native")
    snapshots = []
    sink = Sink()
    worker = actor(
        near_terminal_native(native, snapshots),
        config(games=1, batch_size=1, stream_completed_games=True),
        sink,
    )
    original_evaluate = worker.evaluator.evaluate

    def change_model(requests):
        result = original_evaluate(requests)
        worker.evaluator.model_version = "unexpected-replacement"
        return result

    worker.evaluator.evaluate = change_model
    with pytest.raises(RuntimeError, match="model changed"):
        worker.run()
    assert sink.samples == []


@pytest.mark.native
def test_refill_progress_is_emitted_only_after_previous_game_is_durable():
    native = pytest.importorskip("star_native")
    sink = Sink()
    worker = actor(
        native,
        config(
            games=3,
            batch_size=1,
            stream_completed_games=True,
            rolling_game_slots=True,
            seed_contract="game-v1",
        ),
        sink,
    )
    refills = []

    def progress(**details):
        if details["phase"] == "selfplay_refill":
            committed = {sample.game_id for sample in sink.samples}
            assert len(committed) == details["completed_games"]
            assert details["persisted_decisions"] == len(sink.samples)
            refills.append(details["refilled_games"])

    worker.run(progress=progress)
    assert refills == [1, 2]


@pytest.mark.native
@pytest.mark.parametrize("full", [False, True])
def test_game_mode_probability_endpoints_cover_the_entire_seed_range(full):
    native = pytest.importorskip("star_native")
    sink = Sink()
    worker = actor(
        near_terminal_native(native, []),
        config(
            games=1,
            batch_size=1,
            seed_contract="game-v1",
            full_probability=float(full),
            fast_probability=float(not full),
        ),
        sink,
    )
    original_seed = worker._seed
    worker._seed = lambda purpose, *parts: (
        (1 << 64) - 1 if purpose == "mode-game-v1" else original_seed(purpose, *parts)
    )
    worker.run()
    assert worker.full_decisions == int(full)
    assert worker.fast_decisions == int(not full)


@pytest.mark.native
@pytest.mark.parametrize("handicap", range(2, 10))
@pytest.mark.parametrize("mode", ["classic", "double"])
def test_refill_preserves_every_handicap_severity_and_pda_assignment(mode, handicap):
    native = pytest.importorskip("star_native")
    runs = []
    for rolling in (False, True):
        sink = Sink()
        worker = actor(
            native,
            config(
                games=2,
                batch_size=1,
                stream_completed_games=True,
                rolling_game_slots=rolling,
                seed_contract="game-v1",
                fast_probability=1.0,
                full_probability=0.0,
            ).with_variant(GameVariant(mode=mode, handicap=handicap)),
            sink,
        )
        summaries = worker.run()
        pda = worker.config.variants.pda_for_handicap(handicap)
        assert {(summary.pda_seat0, summary.pda_seat1) for summary in summaries} == {
            (-pda, pda)
        }
        runs.append((sample_fingerprints(sink.samples), summary_records(summaries)))
    assert runs[0] == runs[1]


@pytest.mark.native
def test_game_seed_streams_are_unchanged_by_concurrent_native_tasks():
    native = pytest.importorskip("star_native")

    def task(index):
        sink = Sink()
        worker = SelfPlayActor(
            native,
            Evaluator(),
            sink,
            config(
                games=3,
                batch_size=2,
                stream_completed_games=True,
                rolling_game_slots=True,
                seed_contract="game-v1",
            ),
            SelfPlayIdentity(
                "stream-test", "stream-family", "stream-actor", 12 + index
            ),
        )
        summaries = worker.run()
        return sample_fingerprints(sink.samples), summary_records(summaries)

    expected = [task(index) for index in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(task, range(2)))
    assert actual == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"stream_completed_games": 1},
        {"rolling_game_slots": "true"},
        {"seed_contract": "unknown"},
        {"games": True},
        {"games": 2.0},
        {"batch_size": 1.5},
        {"rolling_game_slots": True},
        {"rolling_game_slots": True, "stream_completed_games": True},
    ],
)
def test_streaming_and_refill_require_explicit_valid_configuration(changes):
    with pytest.raises(ValueError):
        config(**changes)
