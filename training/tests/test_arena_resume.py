from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from startrain.arena import ArenaPair, ArenaRunner
from startrain.config import ArenaConfig
from startrain.inference import InferenceResponse
from startrain.selfplay import GameVariant


class Clock:
    def __init__(self):
        self.now = 0


class Evaluator:
    evaluator_calls = 0
    evaluator_rows = 0

    def __init__(self, name, clock=None):
        self.model_version = name
        self.clock = clock

    def evaluate(self, requests):
        self.evaluator_calls += 1
        self.evaluator_rows += len(requests)
        if self.clock is not None:
            self.clock.now += 1
        return InferenceResponse(
            list(requests.tokens),
            [0.0] * len(requests),
            list(requests.legal_offsets),
            [0.0] * len(requests.legal_actions),
        )


def config(**changes):
    values = dict(
        rings=(4,),
        pairs_per_ring=2,
        minimum_pairs_per_ring=2,
        max_pairs_per_ring=4,
        simulations=2,
        max_considered=2,
        bootstrap_samples=200,
        unforced_opening_fraction=0.5,
    )
    values.update(changes)
    return ArenaConfig(**values)


def runner(native, cfg=None, clock=None):
    return ArenaRunner(
        native_module=native,
        candidate=Evaluator("candidate", clock),
        baseline=Evaluator("baseline", clock),
        config=cfg or config(),
        stable_pair_seeds=True,
    )


@pytest.mark.native
def test_deadline_preserves_moves_and_resumes_across_rings():
    native = pytest.importorskip("star_native")
    cfg = config(rings=(4, 6))
    expected = runner(native, cfg).run(checkpoint=lambda _snapshot: None)
    clock = Clock()
    snapshots = []
    interrupted = runner(native, cfg, clock).run(
        checkpoint=snapshots.append,
        stop_requested=lambda: clock.now >= 15,
    )
    assert interrupted["interrupted"]
    assert interrupted["pairs"] == []
    saved = interrupted["resume_state"]
    moves = sum(len(game["actions"]) for game in saved["game_states"])
    assert moves > 0
    assert saved == snapshots[-1]
    # Earlier callback snapshots are immutable after subsequent moves.
    assert sum(len(game["actions"]) for game in snapshots[0]["game_states"]) < moves
    resumed = runner(native, cfg).run(
        resume_state=json.loads(json.dumps(saved)),
        checkpoint=lambda _snapshot: None,
    )
    assert not resumed["interrupted"]
    assert resumed["games"] == expected["games"]
    assert resumed["pairs"] == expected["pairs"]
    assert resumed["aggregate"] == expected["aggregate"]
    assert resumed["resume_state"] == expected["resume_state"]
    assert (
        resumed["evaluation_metrics"]["total_evaluator_calls"]
        < expected["evaluation_metrics"]["total_evaluator_calls"]
    )


@pytest.mark.native
def test_lone_finished_seat_survives_and_is_never_a_statistical_pair():
    native = pytest.importorskip("star_native")
    # Exercise the single-game compatibility path with real, replayable states.
    sequential = SimpleNamespace(
        StateBatch=lambda *args, **kwargs: native.StateBatch(*args, **kwargs),
        SearchBatch=native.SearchBatch,
    )
    stopping = False

    def checkpoint(snapshot):
        nonlocal stopping
        stopping = len(snapshot["games"]) == 1

    interrupted = runner(sequential).run(
        checkpoint=checkpoint,
        stop_requested=lambda: stopping,
    )
    assert interrupted["interrupted"]
    assert interrupted["games"] == interrupted["pairs"] == []
    saved = interrupted["resume_state"]
    assert len(saved["games"]) == 1
    assert saved["pairs"] == []
    assert saved["games"][0]["candidate_player"] == 0
    resumed = runner(sequential).run(resume_state=saved)
    complete = runner(sequential).run(checkpoint=lambda _snapshot: None)
    assert resumed["games"] == complete["games"]
    assert resumed["pairs"] == complete["pairs"]
    assert resumed["resume_state"] == complete["resume_state"]
    assert (
        resumed["evaluation_metrics"]["total_evaluator_calls"]
        + interrupted["evaluation_metrics"]["total_evaluator_calls"]
        == complete["evaluation_metrics"]["total_evaluator_calls"]
    )
    corrupted = json.loads(json.dumps(saved))
    stored = corrupted["game_states"][0]["result"]
    stored["winner"] = 1 - stored["winner"]
    stored["outcome"] = -stored["outcome"]
    repaired = runner(sequential).run(resume_state=corrupted)
    assert repaired["games"] == complete["games"]
    assert repaired["resume_state"] == complete["resume_state"]


@pytest.mark.native
def test_repeated_short_sessions_eventually_complete_without_replaying_old_moves():
    native = pytest.importorskip("star_native")
    cfg = config()
    clock = Clock()
    saved = None
    previous_moves = 0
    for _ in range(80):
        deadline = clock.now + 8
        result = runner(native, cfg, clock).run(
            resume_state=saved,
            checkpoint=lambda _snapshot: None,
            stop_requested=lambda: clock.now >= deadline,
        )
        saved = result["resume_state"]
        moves = sum(len(game["actions"]) for game in saved["game_states"])
        assert moves > previous_moves
        previous_moves = moves
        if not result["interrupted"]:
            break
    else:
        pytest.fail("bounded arena sessions failed to finish")
    expected = runner(native, cfg).run(checkpoint=lambda _snapshot: None)
    assert result["games"] == expected["games"]
    assert result["pairs"] == expected["pairs"]
    assert saved == expected["resume_state"]


@pytest.mark.native
def test_finished_later_pair_survives_an_unfinished_earlier_pair():
    native = pytest.importorskip("star_native")
    # This seed schedule gives pair 3 a forced opening and pairs 0..2 none,
    # so pair 3 finishes one search move before the earlier pairs.
    cfg = config(pairs_per_ring=4, minimum_pairs_per_ring=4)
    snapshots = []
    result = runner(native, cfg).run(
        checkpoint=snapshots.append,
        stop_requested=lambda: bool(snapshots and snapshots[-1]["games"]),
    )
    assert result["interrupted"]
    assert result["pairs"] == []  # Legacy planners still see a complete prefix.
    assert [pair["pair"] for pair in result["resume_state"]["pairs"]] == [3]
    assert {game["pair"] for game in result["resume_state"]["games"]} == {3}
    incomplete = [
        game for game in result["resume_state"]["game_states"] if game["result"] is None
    ]
    assert len(incomplete) == 6 and all(game["actions"] for game in incomplete)
    resumed = runner(native, cfg).run(resume_state=result["resume_state"])
    expected = runner(native, cfg).run(checkpoint=lambda _snapshot: None)
    assert resumed["games"] == expected["games"]
    assert resumed["pairs"] == expected["pairs"]
    assert resumed["resume_state"] == expected["resume_state"]


@pytest.mark.native
def test_resume_contract_rejects_changed_identity_budget_seed_and_malformed_games():
    native = pytest.importorskip("star_native")
    original = runner(native)
    saved = original.run(checkpoint=lambda _snapshot: None)["resume_state"]
    for changed in (
        runner(native, replace(original.config, seed=original.config.seed + 1)),
        runner(native, replace(original.config, simulations=4)),
    ):
        with pytest.raises(ValueError, match="evaluation contract"):
            changed.run(resume_state=saved)
    changed = runner(native)
    changed.candidate.model_version = "other-candidate"
    with pytest.raises(ValueError, match="evaluation contract"):
        changed.run(resume_state=saved)
    invalid = json.loads(json.dumps(saved))
    invalid["game_states"].append(invalid["game_states"][0])
    with pytest.raises(ValueError, match="duplicate"):
        runner(native).run(resume_state=invalid)
    invalid = json.loads(json.dumps(saved))
    invalid["game_states"][0]["opening_seed"] += 1
    with pytest.raises(ValueError, match="opening"):
        runner(native).run(resume_state=invalid)


@pytest.mark.native
def test_parallel_variant_checkpoints_are_serialized_and_keep_all_games():
    native = pytest.importorskip("star_native")

    class ParallelRunner(ArenaRunner):
        @contextmanager
        def _inference_owner(self):
            # Select the concurrent variant path without requiring neural GPU
            # models; native searches still run on independent producer groups.
            self._shared_broker = object()
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    yield executor
            finally:
                self._shared_broker = None

        def _evaluate_serialized(self, executor, evaluator, requests):
            return executor.submit(evaluator.evaluate, requests).result()

    cfg = config()
    subject = ParallelRunner(
        native_module=native,
        candidate=Evaluator("candidate"),
        baseline=Evaluator("baseline"),
        config=cfg,
        stable_pair_seeds=True,
    )
    active = threading.Lock()
    snapshots = []

    def checkpoint(snapshot):
        assert active.acquire(blocking=False), "checkpoint callbacks overlapped"
        try:
            time.sleep(0.0001)
            snapshots.append(snapshot)
        finally:
            active.release()

    subject._initialize_resume(None, checkpoint)
    games, pairs = [], []
    with subject._inference_owner() as executor:
        complete = subject._play_balanced_groups(
            4,
            {GameVariant(): [0], GameVariant(mode="classic"): [0]},
            games,
            pairs,
            progress=None,
            inference_executor=executor,
            stop_requested=lambda: False,
        )
    assert complete and len(games) == 4 and len(pairs) == 2
    assert len(snapshots[-1]["games"]) == 4
    assert len(snapshots[-1]["pairs"]) == 2
    counts = [
        sum(len(game["actions"]) for game in snapshot["game_states"])
        for snapshot in snapshots
    ]
    assert counts == sorted(counts)


@pytest.mark.native
@pytest.mark.parametrize(
    "mode,pie,handicap",
    [
        ("double", True, 1),
        ("classic", True, 1),
        ("double", False, 9),
        ("classic", False, 9),
    ],
)
def test_resumed_native_positions_preserve_pie_and_severe_handicap(mode, pie, handicap):
    native = pytest.importorskip("star_native")
    variant = GameVariant(mode=mode, pie=pie, handicap=handicap)

    class NegativeEvaluator(Evaluator):
        def evaluate(self, requests):
            response = super().evaluate(requests)
            value = 1.0 if mode == "classic" else -1.0
            return replace(response, values=[value] * len(requests))

    def play(saved=None, interrupt=False):
        subject = runner(native, config(simulations=1, max_considered=1))
        subject.candidate = NegativeEvaluator("candidate")
        subject.baseline = NegativeEvaluator("baseline")
        snapshots = []
        subject._initialize_resume(saved, snapshots.append)
        specifications = subject._pair_specifications(4, [0, 1], variant)
        with subject._inference_owner() as executor:
            games = subject._play_ring_batch(
                4,
                specifications,
                variant=variant,
                progress=None,
                inference_executor=executor,
                stop_requested=lambda: interrupt and bool(snapshots),
            )
        return games, snapshots[-1]

    expected_games, expected_state = play()
    _, partial = play(interrupt=True)
    assert partial["progress"]["completed_moves"] > 0
    assert partial["progress"]["completed_pairs"] == 0
    actual_games, actual_state = play(json.loads(json.dumps(partial)))
    assert actual_games == expected_games
    assert actual_state == expected_state
    if pie:
        assert any(game.swapped for game in actual_games)
    else:
        assert {game.pda for game in actual_games} == {3}


def test_resumability_requires_independent_pair_seed_streams():
    subject = ArenaRunner(
        native_module=object(),
        candidate=Evaluator("candidate"),
        baseline=Evaluator("baseline"),
        config=config(),
    )
    with pytest.raises(ValueError, match="stable pair seeds"):
        subject.run(checkpoint=lambda _snapshot: None)


@pytest.mark.native
def test_balanced_resume_skips_durable_pairs_without_counting_them_twice():
    native = pytest.importorskip("star_native")
    cfg = config(
        balanced_cells=True,
        rings=(4, 6, 8, 10),
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
    )
    counts = {4: 1, 6: 0, 8: 0, 10: 0}
    stopping = False

    def checkpoint(snapshot):
        nonlocal stopping
        stopping = bool(snapshot["pairs"])

    interrupted = runner(native, cfg).run(
        pair_counts=counts,
        checkpoint=checkpoint,
        stop_requested=lambda: stopping,
    )
    assert interrupted["interrupted"]
    assert len(interrupted["pairs"]) == 1
    prior = [ArenaPair(**pair) for pair in interrupted["pairs"]]
    resumed = runner(native, cfg).run(
        pair_counts=counts,
        resume_state=interrupted["resume_state"],
        previous_pairs=prior,
    )
    complete = runner(native, cfg).run(
        pair_counts=counts, checkpoint=lambda _state: None
    )
    assert len(resumed["pairs"]) == 5
    assert len(resumed["resume_state"]["pairs"]) == 6

    def key(value):
        return value["ring"], value["variant"], value["pair"]

    assert sorted(interrupted["pairs"] + resumed["pairs"], key=key) == sorted(
        complete["pairs"],
        key=key,
    )
    assert resumed["balanced_aggregate"] == complete["balanced_aggregate"]
    assert resumed["resume_state"] == complete["resume_state"]
