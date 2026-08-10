from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from startrain.contracts import TARGET_POLICY
from startrain.inference import InferenceResponse
from startrain.native import positions_from_native, score_results_from_native
from startrain.selfplay import SelfPlayActor, SelfPlayConfig
from startrain.topology import get_topology


def test_candidate_limit_scaling_is_explicit_and_capped() -> None:
    baseline = SelfPlayConfig(
        rings=10,
        max_considered=16,
        max_considered_ring_exponent=0.0,
        max_considered_cap=64,
    )
    assert baseline.considered_actions() == 16
    scaled = SelfPlayConfig(
        rings=10,
        simulation_reference_rings=6,
        max_considered=16,
        max_considered_ring_exponent=1.0,
        max_considered_cap=24,
    )
    assert scaled.considered_actions() == 24
    small_ring = SelfPlayConfig(
        rings=4,
        simulation_reference_rings=6,
        max_considered=16,
        max_considered_ring_exponent=1.0,
        max_considered_cap=32,
    )
    assert small_ring.considered_actions() == 16
    with pytest.raises(ValueError, match="candidate scaling"):
        SelfPlayConfig(max_considered=16, max_considered_cap=8)
    with pytest.raises(ValueError, match="fast_policy_weight"):
        SelfPlayConfig(fast_policy_weight=1.1)
    with pytest.raises(ValueError, match="policy-surprise"):
        SelfPlayConfig(policy_surprise_weight=1.1)
    with pytest.raises(ValueError, match="clinch_finalization"):
        SelfPlayConfig(clinch_finalization="random")  # type: ignore[arg-type]


def test_pending_clinch_is_not_counted_before_cohort_finalization() -> None:
    actor = object.__new__(SelfPlayActor)
    actor.clinched_games = 0
    actor.clinch_empty_nodes = 0
    states = SimpleNamespace(
        complete_clinches=lambda: SimpleNamespace(
            batch_size=1,
            clinched=[True],
            winner=[1],
            empty_nodes=[3],
            last_move=[12],
            turn_count=[24],
        )
    )
    finalizations = [None]

    actor._complete_clinches(states, finalizations)

    assert finalizations[0] is not None
    assert actor.clinched_games == 0
    assert actor.clinch_empty_nodes == 0


@pytest.mark.native
def test_fast_policy_target_ablation_records_completed_q_when_enabled() -> None:
    native = pytest.importorskip("star_native")

    class UniformEvaluator:
        model_version = "uniform"
        model_step = 0
        model_identity = "uniform"

        def evaluate(self, requests) -> InferenceResponse:
            return InferenceResponse(
                tokens=list(requests.tokens),
                values=[0.0] * len(requests),
                policy_offsets=list(requests.legal_offsets),
                policy_logits=[0.0] * len(requests.legal_actions),
            )

    class Sink:
        def __init__(self) -> None:
            self.samples = []

        def append(self, samples, **_metadata):
            self.samples.extend(samples)
            return SimpleNamespace(sample_count=len(samples))

    sink = Sink()
    config = SelfPlayConfig(
        rings=4,
        batch_size=1,
        games=1,
        fast_probability=1.0,
        full_probability=0.0,
        fast_simulations=2,
        full_simulations=2,
        simulation_reference_rings=4,
        max_considered=2,
        record_fast_policy_targets=True,
        fast_policy_weight=0.3,
        policy_surprise_weight=0.5,
        shard_size=128,
        seed=91,
    )
    summaries = SelfPlayActor(native, UniformEvaluator(), sink, config).run()
    assert summaries[0].policy_samples == summaries[0].samples
    assert sink.samples
    assert all(sample.target_mask & TARGET_POLICY for sample in sink.samples)
    assert all(
        sample.policy_provenance == "completed-q-fast" for sample in sink.samples
    )
    assert all(sample.policy_weight == pytest.approx(0.3) for sample in sink.samples)
    assert all(
        math.isfinite(sample.weight) and sample.weight > 0 for sample in sink.samples
    )
    assert sum(sample.weight for sample in sink.samples) == pytest.approx(
        len(sink.samples)
    )


@pytest.mark.native
def test_loser_fill_clinch_stops_search_and_supplies_conservative_targets() -> None:
    native = pytest.importorskip("star_native")

    class UniformEvaluator:
        model_version = "uniform"
        model_step = 0
        model_identity = "uniform"

        def evaluate(self, requests) -> InferenceResponse:
            return InferenceResponse(
                tokens=list(requests.tokens),
                values=[0.0] * len(requests),
                policy_offsets=list(requests.legal_offsets),
                policy_logits=[0.0] * len(requests.legal_actions),
            )

    class Sink:
        def __init__(self) -> None:
            self.samples = []

        def append(self, samples, **_metadata):
            self.samples.extend(samples)
            return SimpleNamespace(sample_count=len(samples))

    config = SelfPlayConfig(
        rings=4,
        batch_size=1,
        games=1,
        fast_probability=1.0,
        full_probability=0.0,
        fast_simulations=1,
        full_simulations=1,
        simulation_reference_rings=4,
        max_considered=1,
        record_fast_policy_targets=True,
        clinch_finalization="loser-fill",
        shard_size=128,
        seed=91,
    )
    clinch_sink = Sink()
    clinch_actor = SelfPlayActor(native, UniformEvaluator(), clinch_sink, config)
    clinch_summary = clinch_actor.run()[0]
    full_sink = Sink()
    full_summary = SelfPlayActor(
        native,
        UniformEvaluator(),
        full_sink,
        replace(config, clinch_finalization="disabled"),
    ).run()[0]

    assert clinch_summary.finish_reason == "clinch"
    assert clinch_summary.empty_nodes_saved > 0
    assert (
        clinch_summary.samples + clinch_summary.empty_nodes_saved == get_topology(4).n
    )
    assert full_summary.finish_reason == "board-full"
    assert full_summary.samples == get_topology(4).n
    assert full_summary.winner == clinch_summary.winner
    assert len(clinch_sink.samples) == clinch_summary.samples
    assert all(
        "final=clinch-loser-fill" in sample.search_provenance
        for sample in clinch_sink.samples
    )
    metrics = clinch_actor.metrics_snapshot()
    assert metrics.clinched_games == 1
    assert metrics.clinch_empty_nodes == clinch_summary.empty_nodes_saved

    proof_states = native.StateBatch(4, 1)
    actions = []
    for current, following in zip(
        clinch_sink.samples,
        clinch_sink.samples[1:],
        strict=False,
    ):
        changed = np.flatnonzero(current.stones != following.stones)
        assert changed.size == 1
        actions.append(int(changed[0]))
    actions.append(clinch_summary.last_move)
    for action in actions:
        proof_states.apply_many([0], [action])
    live_position = positions_from_native(proof_states.data())[0]
    proof_result = proof_states.complete_clinches()
    proof_position = positions_from_native(proof_states.data())[0]
    proof_score = score_results_from_native(proof_states.score_data())[0]
    loser = 1 - clinch_summary.winner
    live_stones = live_position.stones.numpy()
    proof_stones = proof_position.stones.numpy()

    assert proof_result.clinched == [True]
    assert proof_result.winner == [clinch_summary.winner]
    assert proof_result.empty_nodes == [clinch_summary.empty_nodes_saved]
    assert np.all(proof_stones[live_stones == -1] == loser)
    sample = clinch_sink.samples[0]
    np.testing.assert_array_equal(
        sample.final_scores,
        [player.total for player in proof_score.players],
    )
    np.testing.assert_array_equal(
        sample.final_ownership, proof_score.node_owner.numpy()
    )
    np.testing.assert_array_equal(
        sample.final_alive,
        proof_score.alive_stone.numpy().astype(np.uint8),
    )
