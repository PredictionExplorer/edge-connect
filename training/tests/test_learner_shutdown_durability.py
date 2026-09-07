import json
from contextlib import contextmanager

import pytest
import torch

import startrain.learner as learner_module
from startrain.config import LearnerConfig
from startrain.replay_store import ReplayStore
from test_pipeline_core import (
    append_replay,
    make_replay_sample,
    make_test_learner,
    run_identity,
)


@contextmanager
def trained_fixture(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id=f"shutdown-{index}",
                )
                for index in range(4)
            ],
            identity,
            model_step=0,
            generation=generation,
        )
        learner = make_test_learner(
            store,
            identity,
            tmp_path / "learner",
            learner_config=LearnerConfig(
                steps=4,
                minimum_replay_samples=1,
                recent_samples_per_ring=8,
                max_replay_lag_steps=10,
                steps_per_window=4,
                candidate_interval=100,
                device="cpu",
            ),
        )
        learner._maybe_write_recovery_checkpoint(force=True)
        yield learner


def recovery_pointer(learner):
    return json.loads((learner.publisher.root / "recovery.json").read_text())


def assert_latest_state_restores(learner, tmp_path):
    pointer = recovery_pointer(learner)
    assert pointer["step"] == learner.step == 1
    restored = make_test_learner(
        learner.store,
        learner.run_identity,
        tmp_path / "restored",
        learner_config=learner.learner_config,
    )
    restored.resume(
        learner.publisher.root / pointer["checkpoint"],
        expected_sha256=pointer["checkpoint_sha256"],
        expected_bytes=pointer["checkpoint_bytes"],
    )
    assert restored.step == learner.step
    assert restored.examples_consumed == learner.examples_consumed
    assert restored.scheduler.state_dict() == learner.scheduler.state_dict()
    for name, parameter in learner.model.state_dict().items():
        torch.testing.assert_close(
            restored.model.state_dict()[name], parameter, rtol=0, atol=0
        )
    for saved, actual in zip(
        restored.optimizer.state.values(), learner.optimizer.state.values(), strict=True
    ):
        for name, value in actual.items():
            torch.testing.assert_close(saved[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("failure_site", ("window", "pool"))
@pytest.mark.parametrize("completed", (False, True))
def test_completed_updates_survive_cleanup_failure(
    tmp_path, monkeypatch, failure_site, completed
):
    with trained_fixture(tmp_path) as learner:
        before = recovery_pointer(learner)
        original_close = learner._close_replay_window

        def close(window, *, reason):
            assert recovery_pointer(learner)["step"] == 1
            original_close(window, reason=reason)
            if failure_site == "window":
                raise RuntimeError("window cleanup failure")

        class FailingPool:
            worker_pids = (101,)

            def shutdown(self, *, strict):
                assert strict is False
                assert recovery_pointer(learner)["step"] == 1
                raise RuntimeError("pool cleanup failure")

        def progress(**message):
            if message["phase"] == "training" and failure_site == "pool":
                learner._loader_pool = FailingPool()

        monkeypatch.setattr(learner, "_close_replay_window", close)
        with pytest.raises(RuntimeError, match=f"{failure_site} cleanup failure"):
            learner.run(
                steps=1 if completed else 4,
                stop_requested=lambda: learner.step >= 1,
                progress=progress,
            )
        assert recovery_pointer(learner) != before
        assert_latest_state_restores(learner, tmp_path)
        assert not (learner.publisher.root / "learner-complete.json").exists()


def test_interrupted_checkpoint_retries_after_loader_teardown(tmp_path, monkeypatch):
    with trained_fixture(tmp_path) as learner:
        original_checkpoint = learner._maybe_write_recovery_checkpoint
        original_close = learner._close_replay_window
        events = []

        def checkpoint(*, force=False):
            if force:
                events.append("checkpoint")
                if events == ["checkpoint"]:
                    raise RuntimeError("DataLoader worker aborted during save")
            return original_checkpoint(force=force)

        def close(window, *, reason):
            events.append("cleanup")
            original_close(window, reason=reason)
            raise RuntimeError("secondary cleanup failure")

        monkeypatch.setattr(learner, "_maybe_write_recovery_checkpoint", checkpoint)
        monkeypatch.setattr(learner, "_close_replay_window", close)
        with pytest.raises(RuntimeError, match="DataLoader worker aborted during save"):
            learner.run(steps=4, stop_requested=lambda: learner.step >= 1)
        assert events == ["checkpoint", "cleanup", "checkpoint"]
        assert_latest_state_restores(learner, tmp_path)


def test_partially_failed_training_does_not_replace_valid_recovery(
    tmp_path, monkeypatch
):
    with trained_fixture(tmp_path) as learner:
        before = recovery_pointer(learner)
        original_step = learner_module.train_step
        original_close = learner._close_replay_window
        original_checkpoint = learner._maybe_write_recovery_checkpoint
        calls = 0
        checkpoint_calls = []

        def train_step(*args, **kwargs):
            nonlocal calls
            result = original_step(*args, **kwargs)
            calls += 1
            if calls == 2:
                raise ValueError("training failed after a partial state update")
            return result

        def close(window, *, reason):
            original_close(window, reason=reason)
            raise RuntimeError("secondary cleanup failure")

        def checkpoint(*, force=False):
            checkpoint_calls.append(force)
            return original_checkpoint(force=force)

        monkeypatch.setattr(learner_module, "train_step", train_step)
        monkeypatch.setattr(learner, "_close_replay_window", close)
        monkeypatch.setattr(learner, "_maybe_write_recovery_checkpoint", checkpoint)
        with pytest.raises(ValueError, match="training failed after a partial state"):
            learner.run(steps=4)
        assert learner.step == 1
        assert recovery_pointer(learner) == before
        assert checkpoint_calls == []


def test_finite_success_still_publishes_completion_after_durable_stop(tmp_path):
    with trained_fixture(tmp_path) as learner:
        assert learner.run(steps=1) == 1
        assert_latest_state_restores(learner, tmp_path)
        complete = json.loads(
            (learner.publisher.root / "learner-complete.json").read_text()
        )
        assert complete["candidate_step"] == 1


def test_checkpoint_retry_is_bounded_and_preserves_previous_recovery(
    tmp_path, monkeypatch
):
    with trained_fixture(tmp_path) as learner:
        before = recovery_pointer(learner)
        original_checkpoint = learner._maybe_write_recovery_checkpoint
        attempts = []

        def checkpoint(*, force=False):
            if force:
                attempts.append(learner.step)
                raise OSError("checkpoint storage unavailable")
            return original_checkpoint(force=force)

        monkeypatch.setattr(learner, "_maybe_write_recovery_checkpoint", checkpoint)
        with pytest.raises(OSError, match="checkpoint storage unavailable") as error:
            learner.run(steps=4, stop_requested=lambda: learner.step >= 1)
        assert attempts == [1, 1]
        assert recovery_pointer(learner) == before
        assert "retry after loader cleanup failed" in error.value.__notes__[0]
