"""Policy-only recovery from cleanly abandoned self-play, without outcome claims."""

from dataclasses import replace
import copy
import math
import re

import pytest
import torch

from startrain.actor_publication import PublicationProgress
from startrain.losses import LossWeights
from startrain.model import GraphResTNet, ModelConfig
from startrain.replay import (
    MISSING_OUTCOME,
    TARGET_POLICY,
    TARGET_SOFT_POLICY,
    collate_replay_samples,
    read_replay_shard,
)
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity
from startrain.selfplay import SelfPlayActor, SelfPlayConfig, SelfPlayIdentity
from startrain.training import train_step
from test_selfplay_streaming import Evaluator, Sink, near_terminal_native


def configuration(**updates):
    return replace(
        SelfPlayConfig(
            rings=4,
            games=2,
            batch_size=2,
            fast_probability=0,
            full_probability=1,
            fast_simulations=1,
            full_simulations=2,
            simulation_ring_exponent=0,
            max_considered=2,
            record_fast_policy_targets=True,
            preserve_interrupted_policy=True,
        ),
        **updates,
    )


def make_actor(native, config, sink, evaluator=None, source_role="candidate"):
    return SelfPlayActor(
        native,
        evaluator or Evaluator(),
        sink,
        config,
        SelfPlayIdentity("salvage-run", "salvage-family", "salvage-actor", 0),
        source_role=source_role,
    )


def tiny_training_model():
    with torch.random.fork_rng():
        torch.manual_seed(43)
        return GraphResTNet(
            ModelConfig(
                width=8,
                rrt_groups=1,
                attention_heads=2,
                kv_heads=1,
                local_blocks_per_group=1,
                adaln_hidden=4,
            )
        )


VALUE_HEADS = ("outcome_head.", "score_margin_head.", "ownership_head.", "alive_head.")


@pytest.mark.native
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize(
    "source_role", ["champion", "candidate", "history", "unattributed"]
)
def test_clean_stop_salvages_policy_masks_without_completing_games(
    enabled, fast, source_role
):
    native = pytest.importorskip("star_native")
    sink = Sink()
    actor = make_actor(
        native,
        configuration(
            preserve_interrupted_policy=enabled,
            fast_probability=int(fast),
            full_probability=int(not fast),
            fast_policy_weight=0.2,
        ),
        sink,
        source_role=source_role,
    )
    summaries = actor.run(
        stop_requested=lambda: actor.full_decisions + actor.fast_decisions >= 4
    )
    assert summaries == []
    metrics = actor.metrics_snapshot()
    assert metrics.started_games == metrics.dropped_games == 2
    assert metrics.completed_games == metrics.completed_decisions == 0
    assert (
        metrics.full_decisions + metrics.fast_decisions
        == metrics.dropped_decisions
        == 4
    )
    assert (
        metrics.salvaged_policy_decisions
        == actor.persisted_decisions
        == (4 if enabled else 0)
    )
    assert metrics.salvaged_games == (2 if enabled else 0)
    for role in ["champion", "candidate", "history", "unattributed"]:
        assert getattr(metrics, f"source_{role}_games") == 0
        assert getattr(metrics, f"source_{role}_samples") == (
            4 if enabled and source_role == role else 0
        )
    assert len(sink.samples) == (4 if enabled else 0)
    for sample in sink.samples:
        assert sample.target_mask == TARGET_POLICY | TARGET_SOFT_POLICY
        assert sample.outcome == MISSING_OUTCOME
        assert sample.game_id.startswith("abandoned-policy-")
        assert ":final=abandoned-policy-only:" in sample.search_provenance
        assert sample.policy_provenance.endswith("-abandoned-policy-only")
        assert sample.policy_weight == (0.2 if fast else 1.0)


@pytest.mark.native
def test_sparse_recorded_policies_have_contiguous_storage_and_original_move_provenance():
    native = pytest.importorskip("star_native")
    sink = Sink()
    actor = make_actor(
        native,
        configuration(
            games=1,
            batch_size=1,
            fast_probability=0.5,
            full_probability=0.5,
            record_fast_policy_targets=False,
        ),
        sink,
    )
    actor.run(stop_requested=lambda: actor.full_decisions + actor.fast_decisions >= 12)
    metrics = actor.metrics_snapshot()
    assert 0 < metrics.full_decisions < 12
    assert (
        len(sink.samples) == metrics.salvaged_policy_decisions == metrics.full_decisions
    )
    assert metrics.dropped_decisions == 12
    assert [sample.ply for sample in sink.samples] == list(range(len(sink.samples)))
    originals = [
        int(re.search(r":original_ply=(\d+):", sample.search_provenance)[1])
        for sample in sink.samples
    ]
    assert originals == sorted(set(originals))
    assert all(0 <= ply < 12 for ply in originals)
    assert len({sample.game_id for sample in sink.samples}) == 1
    assert all(sample.policy_weight == 1 for sample in sink.samples)


@pytest.mark.native
def test_no_recorded_policy_and_evaluator_failure_do_not_salvage():
    native = pytest.importorskip("star_native")
    sink = Sink()
    actor = make_actor(
        native,
        configuration(
            games=1,
            batch_size=1,
            fast_probability=1,
            full_probability=0,
            record_fast_policy_targets=False,
        ),
        sink,
    )
    actor.run(stop_requested=lambda: actor.fast_decisions >= 3)
    assert sink.samples == []
    assert actor.metrics_snapshot().salvaged_policy_decisions == 0

    class FailingEvaluator(Evaluator):
        calls = 0

        def evaluate(self, requests):
            self.calls += 1
            if self.calls == 7:
                raise RuntimeError("inference failure")
            return super().evaluate(requests)

    sink = Sink()
    actor = make_actor(
        native, configuration(games=1, batch_size=1), sink, FailingEvaluator()
    )
    with pytest.raises(RuntimeError, match="inference failure"):
        actor.run()
    assert actor.full_decisions > 0
    assert actor.metrics_snapshot().salvaged_policy_decisions == 0
    assert sink.samples == []


@pytest.mark.native
def test_mixed_completed_and_salvaged_publications_do_not_inflate_games():
    native = pytest.importorskip("star_native")
    snapshots = []
    sink = Sink()
    records = []
    callback = PublicationProgress(
        metadata={},
        base_games=0,
        base_samples=0,
        base_evaluator_rows=0,
        base_wall_seconds=0,
        task_started=0,
        evaluator_rows=lambda: 0,
        heartbeat=lambda **_: None,
        emit=records.append,
    )
    actor = make_actor(
        near_terminal_native(native, snapshots),
        configuration(
            stream_completed_games=True,
        ),
        sink,
    )
    summaries = actor.run(
        stop_requested=lambda: (
            bool(snapshots) and bool(snapshots[-1].data().terminal[0])
        ),
        progress=callback.progress,
    )
    callback.finish()
    callback.finish()
    assert len(summaries) == actor.completed_games == 1
    assert actor.completed_decisions == actor.salvaged_policy_decisions == 1
    assert actor.persisted_decisions == len(sink.samples) == 2
    assert sum(record["published_games"] for record in records) == 1
    assert sum(record["published_samples"] for record in records) == 2
    assert sum(record["published_policy_only_samples"] for record in records) == 1
    assert records[-1]["cumulative_games"] == 1
    assert records[-1]["cumulative_samples"] == 2
    assert records[-1]["cumulative_policy_only_samples"] == 1


@pytest.mark.native
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_real_replay_accepts_abandoned_policy_group_and_preserves_missing_labels(
    tmp_path,
    precision,
):
    native = pytest.importorskip("star_native")

    class StoredEvaluator(Evaluator):
        model_identity = model_version = "sha256-" + "b" * 64

    identity = RunIdentity(tmp_path / "run.json", "salvage-run", "salvage-family", 1)
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "salvage-actor")
        actor = SelfPlayActor(
            native,
            StoredEvaluator(),
            store,
            configuration(games=1, batch_size=1),
            SelfPlayIdentity(
                identity.run_id, identity.generation_family, "salvage-actor", generation
            ),
        )
        actor.run(stop_requested=lambda: actor.full_decisions >= 6)
        files = list(store.shard_directory.glob("*.npz"))
        assert len(files) == 1
        samples = read_replay_shard(files[0])
        assert [sample.ply for sample in samples] == list(range(6))
        batch = collate_replay_samples(samples)
        targets = batch.targets
        assert targets.policy_mask.all() and targets.soft_policy_mask.all()
        assert not targets.outcome_mask.any()
        assert not targets.score_margin_mask.any()
        assert not targets.ownership_mask.any() and not targets.alive_mask.any()
        # Ledger rows describe logical groups; they must carry the abandoned prefix.
        groups = store.connection.execute("SELECT game_id FROM games").fetchall()
        assert len(groups) == 1 and groups[0]["game_id"].startswith("abandoned-policy-")
        assert actor.completed_games == 0
        assert actor.metrics_snapshot().source_unattributed_games == 0
        assert actor.metrics_snapshot().source_unattributed_samples == 6

        # Exercise the actual learner consumer, including backward, finite
        # checks, clipping and optimizer.step, on an entirely policy-only batch.
        model = tiny_training_model()
        initial = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
        result = train_step(
            model,
            batch,
            optimizer,
            precision=precision,
            loss_weights=LossWeights(
                teacher_policy=0.5,
                teacher_outcome=0.25,
                teacher_score_margin=0.1,
            ),
        )
        metrics = result.to_host()
        assert all(math.isfinite(value) for value in metrics.losses.values())
        assert math.isfinite(metrics.gradient_norm) and metrics.gradient_norm > 0
        assert metrics.losses["policy"] > 0 and metrics.losses["soft_policy"] > 0
        for head in ["outcome", "score_margin", "ownership", "alive"]:
            assert metrics.losses[head] == 0
        assert targets.teacher_mask is None or not targets.teacher_mask.any()
        for head in ["teacher_policy", "teacher_outcome", "teacher_score_margin"]:
            # Collation can omit the entire absent-teacher branch.
            assert metrics.losses.get(head, 0) == 0
        assert not torch.equal(initial["node_policy.weight"], model.node_policy.weight)
        for name, parameter in model.named_parameters():
            assert torch.isfinite(parameter).all()
            assert parameter.grad is None or torch.isfinite(parameter.grad).all()
            if name.startswith(VALUE_HEADS):
                assert torch.equal(initial[name], parameter)
                assert (
                    parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
                )


@pytest.mark.native
def test_weighted_mixed_batches_get_value_gradients_only_from_completed_games():
    native = pytest.importorskip("star_native")
    normal_sink = Sink()
    normal_actor = make_actor(
        near_terminal_native(native, []), configuration(), normal_sink
    )
    normal_actor.run()
    assert len(normal_sink.samples) >= 2
    normal = [
        replace(normal_sink.samples[0], weight=0.5, policy_weight=0.25),
        replace(normal_sink.samples[1], weight=3.0, policy_weight=0.8),
    ]
    salvage_sink = Sink()
    salvage_actor = make_actor(
        native, configuration(games=1, batch_size=1), salvage_sink
    )
    salvage_actor.run(stop_requested=lambda: salvage_actor.full_decisions >= 2)
    salvage = [
        replace(sample, weight=7.0, policy_weight=0.2)
        for sample in salvage_sink.samples
    ]
    reference_batch = collate_replay_samples(normal)
    mixed_batch = collate_replay_samples([normal[0], *salvage, normal[1]])
    assert mixed_batch.targets.outcome_mask.tolist() == [True, False, False, True]
    reference_model = tiny_training_model()
    mixed_model = copy.deepcopy(reference_model)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.025)
    mixed_optimizer = torch.optim.SGD(mixed_model.parameters(), lr=0.025)
    # A shared clipping factor would intentionally couple all head gradients;
    # keep this probe below clipping to isolate missing-label/weight semantics.
    reference = train_step(
        reference_model,
        reference_batch,
        reference_optimizer,
        gradient_clip_norm=1_000_000,
    ).to_host()
    mixed = train_step(
        mixed_model, mixed_batch, mixed_optimizer, gradient_clip_norm=1_000_000
    ).to_host()
    assert all(math.isfinite(value) for value in mixed.losses.values())
    for head in ["outcome", "score_margin", "ownership", "alive"]:
        assert mixed.losses[head] == pytest.approx(
            reference.losses[head], rel=1e-5, abs=1e-6
        )
        assert mixed.losses[head] > 0
    reference_parameters = dict(reference_model.named_parameters())
    outcome_signal = 0.0
    for name, parameter in mixed_model.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
        if name.startswith(VALUE_HEADS):
            expected = reference_parameters[name]
            assert parameter.grad is not None and expected.grad is not None
            torch.testing.assert_close(
                parameter.grad, expected.grad, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)
            if name.startswith("outcome_head."):
                outcome_signal += float(parameter.grad.square().sum())
    assert outcome_signal > 0


def test_salvage_is_disabled_by_default_and_flag_is_strict_boolean():
    assert SelfPlayConfig().preserve_interrupted_policy is False
    for invalid in [1, None, "true"]:
        with pytest.raises(ValueError, match="boolean"):
            configuration(preserve_interrupted_policy=invalid)


@pytest.mark.native
def test_failed_durable_append_does_not_report_salvaged_rows():
    native = pytest.importorskip("star_native")

    class FailingSink(Sink):
        def append(self, samples, **metadata):
            raise OSError("test append failure")

    events = []
    actor = make_actor(native, configuration(), FailingSink())
    with pytest.raises(OSError, match="append failure"):
        actor.run(
            stop_requested=lambda: actor.full_decisions >= 4,
            progress=lambda **fields: events.append(fields),
        )
    assert actor.persisted_decisions == actor.salvaged_policy_decisions == 0
    assert not any(event["phase"] == "selfplay_policy_salvaged" for event in events)
