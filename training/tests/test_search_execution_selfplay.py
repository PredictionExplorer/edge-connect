from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from startrain.inference import InferenceResponse
from startrain.search_options import FullSearchBudgetConfig, SearchExecutionConfig
from startrain.selfplay import SelfPlayActor, SelfPlayConfig, SelfPlayIdentity
from test_selfplay_streaming import Sink, sample_fingerprints


class DeterministicEvaluator:
    model_version = "search-experiment-test"
    model_identity = "search-experiment-test"
    model_step = 3

    def __init__(self, *, sharp=False):
        self.sharp = sharp
        self.rows = []

    def evaluate(self, requests):
        self.rows.append(len(requests))
        logits = []
        for start, end in zip(
            requests.legal_offsets[:-1], requests.legal_offsets[1:], strict=True
        ):
            logits.extend([0.0] + [-10000.0 if self.sharp else 0.0] * (end - start - 1))
        return InferenceResponse(
            list(requests.tokens),
            [0.8 if side == 0 else -0.8 for side in requests.states.to_move],
            list(requests.legal_offsets),
            logits,
        )


def run_tail(native, execution, *, sharp=False):
    observations = SimpleNamespace(constructors=[], advances=[], roots=0, results=[])

    class Search:
        def __init__(self, states, **options):
            observations.constructors.append(options)
            self.inner = native.SearchBatch(states, **options)

        def advance(self, states, **options):
            observations.advances.append(options)
            return self.inner.advance(states, **options)

        def root_requests(self):
            observations.roots += 1
            return self.inner.root_requests()

        def results(self):
            result = self.inner.results()
            observations.results.append(result)
            return result

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def state_batch(rings, rows, **options):
        states = native.StateBatch(rings, rows, **options)
        for row in range(rows):
            placed = states.node_count - 8 - row
            states.apply_many([row] * placed, list(range(placed)))
        return states

    module = SimpleNamespace(
        StateBatch=state_batch,
        SearchBatch=Search,
        native_search_execution_version=native.native_search_execution_version,
    )
    config = SelfPlayConfig(
        rings=4,
        games=2,
        batch_size=2,
        fast_probability=0,
        full_probability=1,
        fast_simulations=2,
        full_simulations=8,
        simulation_ring_exponent=0,
        max_considered=4,
        search_execution=execution,
    )
    evaluator = DeterministicEvaluator(sharp=sharp)
    sink = Sink()
    actor = SelfPlayActor(
        module, evaluator, sink, config, SelfPlayIdentity("run", "family", "actor", 0)
    )
    summaries = actor.run()
    return sink, summaries, evaluator, observations


@pytest.mark.native
def test_first_visit_prefetch_preserves_selfplay_results_and_respects_row_cap():
    native = pytest.importorskip("star_native")
    baseline = run_tail(native, SearchExecutionConfig())
    batched = run_tail(native, SearchExecutionConfig(first_visit_batch_size=4))
    assert [asdict(summary) for summary in baseline[1]] == [
        asdict(summary) for summary in batched[1]
    ]
    # The experiment label is intentionally different; targets and semantic
    # replay remain identical for this exact first-visit execution treatment.
    before = [
        replace(sample, search_provenance="same") for sample in baseline[0].samples
    ]
    after = [replace(sample, search_provenance="same") for sample in batched[0].samples]
    assert sample_fingerprints(before) == sample_fingerprints(after)
    assert max(batched[2].rows) <= 4
    assert all(
        "first_visit_batch_size" not in options for options in baseline[3].constructors
    )
    assert all(
        options["first_visit_batch_size"] == 4 for options in batched[3].constructors
    )
    assert all(
        ":first_visit_batch_size=4:" in sample.search_provenance
        for sample in batched[0].samples
    )


@pytest.mark.native
def test_entropy_full_cap_records_actual_budget_without_relabeling_full_targets():
    native = pytest.importorskip("star_native")
    sink, summaries, _, _ = run_tail(
        native,
        SearchExecutionConfig(full_budget=FullSearchBudgetConfig(mode="root-entropy")),
        sharp=True,
    )
    assert summaries and sink.samples
    for sample in sink.samples:
        assert "gumbel-completed-q:full:simulations=4:" in sample.search_provenance
        assert (
            ":full_budget=root-entropy-v1:root_entropy=0:" in sample.search_provenance
        )
        assert ":planned_base=8:effective_base=4:full_cap=4" in sample.search_provenance
        assert sample.policy_provenance.startswith("completed-q-full")
        assert sample.policy_weight == 1


@pytest.mark.native
def test_reuse_keeps_one_session_and_evaluates_each_new_root():
    native = pytest.importorskip("star_native")
    sink, summaries, _, observed = run_tail(
        native, SearchExecutionConfig(first_visit_batch_size=2, subtree_reuse=True)
    )
    assert summaries and sink.samples
    assert len(observed.constructors) == 1
    assert observed.advances
    assert observed.roots == len(observed.advances) + 1
    assert len({options["model_context"] for options in observed.advances}) == 1
    assert all(
        options["reuse_tree"] and options["max_reused_nodes"] == 4096
        for options in observed.advances
    )
    assert any(
        any(count > 0 for count in result.reused_nodes) for result in observed.results
    )
    assert all(
        ":subtree_reuse=true:" in sample.search_provenance for sample in sink.samples
    )
