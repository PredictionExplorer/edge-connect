"""Pie decisions follow the searched keep move, independently of exploration."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from startrain.arena import ArenaRunner
from startrain.config import ArenaConfig
from startrain.contracts import OUTCOME_WIN, SEARCH_ALGORITHM_ID
from startrain.inference import InferenceResponse
from startrain.native import validate_native_module
from startrain.selfplay import (
    GameVariant,
    SelfPlayActor,
    SelfPlayConfig,
    VariantMixtureConfig,
)


class UniformEvaluator:
    model_version = "pie-selected-keep-test"
    model_identity = model_version
    model_step = 0

    def evaluate(self, requests):
        return InferenceResponse(
            tokens=list(requests.tokens),
            values=[0.0] * len(requests),
            policy_offsets=list(requests.legal_offsets),
            policy_logits=[0.0] * len(requests.legal_actions),
        )


class Sink:
    def __init__(self):
        self.samples = []

    def append(self, samples, **_metadata):
        self.samples.extend(samples)
        return SimpleNamespace(sample_count=len(samples))


def native_with_conflicting_mean(native, keep_value):
    """Run real search, then make only responder decision estimates disagree."""

    class Search:
        def __init__(self, states, **options):
            self.inner = native.SearchBatch(states, **options)
            self.swap_available = list(states.data().swap_available)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def results(self):
            result = self.inner.results()
            fields = (
                "selected_actions",
                "terminal",
                "action_offsets",
                "actions",
                "visits",
                "q_values",
                "priors",
                "policy_target",
                "root_values",
                "selected_action_values",
            )
            data = {name: getattr(result, name) for name in fields}
            for row, available in enumerate(self.swap_available):
                if available:
                    data["selected_action_values"][row] = keep_value
                    data["root_values"][row] = -0.75 if keep_value >= 0 else 0.75
            return SimpleNamespace(**data)

    return SimpleNamespace(StateBatch=native.StateBatch, SearchBatch=Search)


@pytest.mark.native
@pytest.mark.parametrize("mode", ["classic", "double"])
@pytest.mark.parametrize(
    "keep_value,swaps", [(1.0, False), (-0.01, False), (-1.0, True)]
)
def test_selfplay_pie_uses_selected_keep_value_and_preserves_replay_labels(
    mode, keep_value, swaps
):
    native = pytest.importorskip("star_native")
    validate_native_module(native)
    config = replace(
        SelfPlayConfig.cpu_smoke(seed=17),
        batch_size=1,
        games=1,
        full_simulations=4,
        max_considered=4,
        variants=VariantMixtureConfig(enabled=True, swap_dead_zone=0.02),
    ).with_variant(GameVariant(mode=mode, pie=True))
    sink = Sink()
    actor = SelfPlayActor(
        native_with_conflicting_mean(native, keep_value),
        UniformEvaluator(),
        sink,
        config,
    )
    (summary,) = actor.run()
    assert summary.swapped == swaps
    assert actor.pie_decisions == 1
    assert actor.pie_swaps == int(swaps)
    (responder,) = [sample for sample in sink.samples if sample.swap_available]
    assert ("swap=taken" in responder.search_provenance) == swaps
    assert f"algorithm={SEARCH_ALGORITHM_ID}" in responder.search_provenance
    # The policy remains a conditional placement target; actual outcomes retain
    # fixed-player ownership across the optional swap, with no phantom action.
    assert responder.policy.shape == responder.stones.shape
    assert responder.policy.sum() == pytest.approx(1.0)
    assert (responder.outcome == OUTCOME_WIN) == (summary.winner == responder.to_move)


@pytest.mark.native
@pytest.mark.parametrize("mode", ["classic", "double"])
@pytest.mark.parametrize(
    "keep_value,swaps", [(1.0, False), (-0.01, False), (-1.0, True)]
)
def test_arena_pie_uses_selected_keep_value(mode, keep_value, swaps):
    native = pytest.importorskip("star_native")
    validate_native_module(native)
    states = native.StateBatch(4, 1, mode=mode, pie=True)
    states.apply_many([0], [0])
    evaluator = UniformEvaluator()
    runner = ArenaRunner(
        native_module=native_with_conflicting_mean(native, keep_value),
        candidate=evaluator,
        baseline=evaluator,
        config=ArenaConfig(
            rings=(4,), simulations=4, max_considered=4, swap_dead_zone=0.02
        ),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = runner._search_group(
            states,
            evaluator,
            runner.candidate_search,
            17,
            1,
            [0],
            executor,
            lambda: False,
            swap_available=[True],
            node_count=states.node_count,
        )
    assert result is not None
    rows, actions = result
    assert rows == [0]
    assert (actions[0] == states.node_count) == swaps
    assert swaps or 0 <= actions[0] < states.node_count
