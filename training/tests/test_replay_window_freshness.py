from contextlib import contextmanager
from dataclasses import replace
import json
import time

import pytest

from startrain import learner as module
from startrain.config import DataConfig, LearnerConfig, SchedulerConfig, TrainConfig
from startrain.replay_store import ReplayStore
from test_pipeline_core import (
    append_replay,
    make_replay_sample,
    make_test_learner,
    run_identity,
)


def append_rows(store, identity, count, *, step=0, classic=False, sharp=False):
    serial = store.total_committed_sample_count(
        run_id=identity.run_id, generation_family=identity.generation_family
    )
    samples = []
    for index in range(count):
        sample = make_replay_sample(
            identity=identity, game_id=f"fresh-{serial}-{index}"
        )
        if classic:
            sample.mode = "classic"
        if sharp:
            sample.policy = sample.policy.copy()
            sample.policy[:] = 0
            sample.policy[0] = 1
            sample.soft_policy = sample.policy.copy()
        samples.append(sample)
    return append_replay(store, samples, identity, model_step=step)


@contextmanager
def fixture(tmp_path, *, workers=0, target=None):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        append_rows(store, identity, 16)
        learner = make_test_learner(
            store,
            identity,
            tmp_path / "learner",
            learner_config=LearnerConfig(
                minimum_replay_samples=1,
                recent_samples_per_ring=16,
                max_replay_lag_steps=10,
                steps_per_window=1000,
                candidate_interval=100,
                metrics_interval=1,
                replay_poll_seconds=0.001,
                target_updates_per_new_sample=target,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=2,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
            ),
            data_config=DataConfig(
                workers=workers,
                min_batches_for_workers=1,
                ring_stratified=False,
                d5_augmentation=False,
            ),
        )
        yield learner, store, identity
        learner._shutdown_loader_pool()


def open_window(learner):
    return learner._open_replay_window(
        learner._select_replay_spans(), batches=4, refresh_reason="initial"
    )


def age(window):
    window.opened_monotonic = (
        time.monotonic() - module.REPLAY_WINDOW_MAX_AGE_SECONDS - 1
    )
    window.freshness_last_checked_monotonic = None


def test_age_and_global_batch_threshold_gate_actual_new_selected_rows(tmp_path):
    with fixture(tmp_path) as (learner, store, identity):
        window = open_window(learner)
        try:
            learner.world_size = 2
            append_rows(store, identity, 3)
            assert learner._rank_zero_has_fresh_replay(window) is False
            assert window.freshness_probes == 0
            age(window)
            assert learner._rank_zero_has_fresh_replay(window) is False
            assert window.freshness_new_committed_rows == 3
            append_rows(store, identity, 1)
            # A no-change probe is retried on a bounded clock, not every spin.
            assert learner._rank_zero_has_fresh_replay(window) is False
            assert window.freshness_probes == 1
            window.freshness_last_checked_monotonic -= (
                module.REPLAY_FRESHNESS_RETRY_SECONDS + 1
            )
            assert learner._rank_zero_has_fresh_replay(window) is True
            assert window.freshness_additional_selected_rows == 4
            assert learner.examples_consumed == learner.step == 0
        finally:
            learner.world_size = 1
            learner._close_replay_window(window, reason="test")


def test_per_step_freshness_check_performs_no_sql_before_age_limit(
    tmp_path, monkeypatch
):
    with fixture(tmp_path) as (learner, store, _identity):
        window = open_window(learner)
        try:
            with monkeypatch.context() as context:
                context.setattr(
                    store,
                    "total_committed_sample_count",
                    lambda **_kwargs: pytest.fail("early SQL counter read"),
                )
                context.setattr(
                    learner,
                    "_rank_zero_select_replay_spans",
                    lambda: pytest.fail("early selection read"),
                )
                for _ in range(5):
                    assert learner._window_freshness_requested(window) is False
            assert window.freshness_probes == 0
        finally:
            learner._close_replay_window(window, reason="test")


@pytest.mark.parametrize(
    "ineligible", ["future_model", "zero_quota", "unchanged_selection", "zero_capacity"]
)
def test_ineligible_or_unchanged_successor_cannot_cause_refresh_loop(
    tmp_path, monkeypatch, ineligible
):
    with fixture(tmp_path) as (learner, store, identity):
        learner.learner_config = replace(
            learner.learner_config, segment_quotas={"standard": 1.0}
        )
        window = open_window(learner)
        try:
            append_rows(
                store,
                identity,
                4,
                step=100 if ineligible == "future_model" else 0,
                classic=ineligible == "zero_quota",
            )
            if ineligible == "unchanged_selection":
                monkeypatch.setattr(
                    learner,
                    "_rank_zero_select_replay_spans",
                    lambda: replace(window.selection, max_shard_id=9999),
                )
            elif ineligible == "zero_capacity":
                monkeypatch.setattr(
                    learner, "_maximum_unique_batches", lambda _selection: 0
                )
            age(window)
            for _ in range(3):
                assert learner._window_refresh_reason(window, target=None) is None
            assert window.freshness_probes == 1
            assert not window.closed
            assert (
                store.connection.execute(
                    "SELECT COUNT(*) FROM gc_watermarks"
                ).fetchone()[0]
                == 1
            )
        finally:
            learner._close_replay_window(window, reason="test")


def test_only_rank_zero_reads_clock_and_store_and_all_ranks_receive_same_decision(
    tmp_path, monkeypatch
):
    with fixture(tmp_path) as (learner, store, identity):
        window = open_window(learner)
        try:
            append_rows(store, identity, 4)
            age(window)
            payloads = []
            monkeypatch.setattr(
                learner,
                "_broadcast_object",
                lambda value: payloads.append(value) or value,
            )
            assert learner._window_freshness_requested(window) is True
            transmitted = payloads[-1]
            learner.rank = 1
            with monkeypatch.context() as context:
                context.setattr(
                    module.time,
                    "monotonic",
                    lambda: pytest.fail("nonzero rank read clock"),
                )
                context.setattr(
                    learner,
                    "_rank_zero_select_replay_spans",
                    lambda: pytest.fail("nonzero rank read manifest"),
                )
                context.setattr(
                    learner,
                    "_broadcast_object",
                    lambda value: (
                        transmitted
                        if value is None
                        else pytest.fail("nonzero rank authored decision")
                    ),
                )
                assert learner._window_freshness_requested(window) is True
        finally:
            learner.rank = 0
            learner._close_replay_window(window, reason="test")


def test_probe_error_is_broadcast_instead_of_stranding_other_ranks(
    tmp_path, monkeypatch
):
    with fixture(tmp_path) as (learner, store, identity):
        window = open_window(learner)
        try:
            append_rows(store, identity, 4)
            age(window)

            def broken():
                raise ValueError("manifest failure")

            monkeypatch.setattr(learner, "_rank_zero_select_replay_spans", broken)
            payloads = []
            monkeypatch.setattr(
                learner,
                "_broadcast_object",
                lambda value: payloads.append(value) or value,
            )
            with pytest.raises(RuntimeError, match="manifest failure"):
                learner._window_freshness_requested(window)
            assert payloads[-1] == {"freshness_error": "ValueError: manifest failure"}
        finally:
            learner._close_replay_window(window, reason="test")


@pytest.mark.parametrize("workers,utd_wait", [(0, False), (1, False), (0, True)])
def test_real_window_refresh_drains_old_prefetch_and_preserves_utd_and_consumed_counts(
    tmp_path, monkeypatch, workers, utd_wait
):
    target = 0.125 if utd_wait else 1.0
    steps = 2 if utd_wait else 3
    with fixture(tmp_path, workers=workers, target=target) as (
        learner,
        store,
        identity,
    ):
        windows, policies = [], []
        original_open, original_step = learner._open_replay_window, module.train_step
        original_close = learner._close_replay_window
        original_quiesce = module.SpawnedReplayLoaderPool.quiesce
        original_clear = store.clear_gc_watermark
        cleanup_order = []
        closing_reason = None
        segment_before = []
        appended = False

        def opened(*args, **kwargs):
            window = original_open(*args, **kwargs)
            windows.append(window)
            return window

        def trained(model, batch, *args, **kwargs):
            policies.append(batch.targets.policy[:, 0].tolist())
            return original_step(model, batch, *args, **kwargs)

        def clear(name):
            assert all(window.closed for window in windows)
            if workers and closing_reason == "fresh_replay":
                assert cleanup_order[-2:] == ["close:fresh_replay", "quiesce"]
            cleanup_order.append("clear")
            return original_clear(name)

        def close(window, *, reason):
            nonlocal closing_reason
            closing_reason = reason
            cleanup_order.append(f"close:{reason}")
            return original_close(window, reason=reason)

        def quiesce(pool):
            result = original_quiesce(pool)
            cleanup_order.append("quiesce")
            return result

        def progress(**payload):
            nonlocal appended
            if payload.get("phase") == "training" and learner.step == 1:
                age(windows[-1])
                segment_before.append(learner._utd_segment_state.as_dict())
            trigger = "update_to_data_wait" if utd_wait else "training"
            if payload.get("phase") == trigger and learner.step == 1 and not appended:
                append_rows(store, identity, 16, sharp=True)
                windows[-1].freshness_last_checked_monotonic = None
                appended = True

        monkeypatch.setattr(learner, "_open_replay_window", opened)
        monkeypatch.setattr(learner, "_close_replay_window", close)
        monkeypatch.setattr(module.SpawnedReplayLoaderPool, "quiesce", quiesce)
        monkeypatch.setattr(module, "train_step", trained)
        monkeypatch.setattr(store, "clear_gc_watermark", clear)
        assert learner.run(steps=steps, progress=progress) == steps
        assert appended and len(windows) == 2
        assert windows[1].refresh_reason == "fresh_replay"
        assert windows[0].batches_consumed == 1
        assert sum(window.batches_consumed for window in windows) == steps
        assert policies[0][0] < 1 and all(row == [1.0, 1.0] for row in policies[1:])
        assert learner.examples_consumed == steps * 2
        assert learner._utd_segment_state.as_dict() == segment_before[0]
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id, generation_family=identity.generation_family
            )
            == 32
        )
        assert (
            store.connection.execute("SELECT COUNT(*) FROM gc_watermarks").fetchone()[0]
            == 0
        )
        events = [
            json.loads(row) for row in learner.metrics.path.read_text().splitlines()
        ]
        refresh = next(
            row
            for row in events
            if row.get("window_refresh_reason") == "fresh_replay"
            and row.get("event") == "replay_window_refreshed"
        )
        assert refresh["freshness_additional_selected_rows"] == 16
        assert refresh["window_batches_remaining"] > 0


@pytest.mark.parametrize(
    "objective,expected",
    [
        ("ring10_priority", {"handicap": 0.5, "pie": 0.5}),
        ("ring10_only", None),
        ("generalist", None),
    ],
)
def test_selection_probe_and_retention_share_six_mode_objective(
    tmp_path, monkeypatch, objective, expected
):
    with fixture(tmp_path) as (learner, store, identity):
        learner.serialized_config = {
            "orchestration": {
                "training_objective": objective,
                "retention": {
                    "enabled": True,
                    "dry_run": True,
                    "gc_interval_windows": 1,
                },
            }
        }
        learner.learner_config = replace(
            learner.learner_config, segment_quotas={"standard": 1.0}
        )
        selected, collected = [], []
        original_select, original_collect = (
            store.select_recent_spans,
            store.collect_garbage,
        )

        def select(**kwargs):
            selected.append(kwargs.get("within_segment_classic_shares"))
            return original_select(**kwargs)

        def collect(**kwargs):
            collected.append(kwargs["within_segment_classic_shares"])
            return original_collect(**kwargs)

        monkeypatch.setattr(store, "select_recent_spans", select)
        monkeypatch.setattr(store, "collect_garbage", collect)
        window = open_window(learner)
        try:
            append_rows(store, identity, 4)
            age(window)
            assert learner._rank_zero_has_fresh_replay(window)
            learner._maybe_collect_replay_garbage()
            assert selected and all(shares == expected for shares in selected)
            assert collected == [expected]
        finally:
            learner._close_replay_window(window, reason="test")
