from __future__ import annotations

from types import SimpleNamespace

import pytest

from startrain.contracts import TARGET_POLICY
from startrain.inference import InferenceResponse
from startrain.selfplay import SelfPlayActor, SelfPlayConfig


def test_candidate_limit_scaling_is_explicit_and_capped() -> None:
    baseline = SelfPlayConfig(
        rings=12,
        max_considered=16,
        max_considered_ring_exponent=0.0,
        max_considered_cap=64,
    )
    assert baseline.considered_actions() == 16
    scaled = SelfPlayConfig(
        rings=12,
        simulation_reference_rings=6,
        max_considered=16,
        max_considered_ring_exponent=1.0,
        max_considered_cap=24,
    )
    assert scaled.considered_actions() == 24
    with pytest.raises(ValueError, match="candidate scaling"):
        SelfPlayConfig(max_considered=16, max_considered_cap=8)
    with pytest.raises(ValueError, match="fast_policy_weight"):
        SelfPlayConfig(fast_policy_weight=1.1)


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
        rings=3,
        batch_size=1,
        games=1,
        fast_probability=1.0,
        full_probability=0.0,
        fast_simulations=2,
        full_simulations=2,
        simulation_reference_rings=3,
        max_considered=2,
        record_fast_policy_targets=True,
        fast_policy_weight=0.3,
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
