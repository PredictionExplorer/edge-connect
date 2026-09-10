"""Real native integration for bounded prefetch and context-safe tree reuse."""

import math

import pytest

from startrain.inference import InferenceResponse
from startrain.native import BITBOARD_WORDS


def response(requests, *, concentrated=False):
    values, logits = [], []
    for row in range(len(requests)):
        code = sum(
            requests.states.zero_bits[row * BITBOARD_WORDS : (row + 1) * BITBOARD_WORDS]
        )
        code ^= (
            sum(
                requests.states.one_bits[
                    row * BITBOARD_WORDS : (row + 1) * BITBOARD_WORDS
                ]
            )
            << 1
        )
        actions = requests.legal_actions[
            requests.legal_offsets[row] : requests.legal_offsets[row + 1]
        ]
        values.append(0.0 if concentrated else (code % 101 - 50) / 100)
        logits.extend(
            (8.0 if index == 0 else -2.0)
            if concentrated
            else ((action * 13 + code) % 17) / 10 - 0.8
            for index, action in enumerate(actions)
        )
    return InferenceResponse(
        tokens=list(requests.tokens),
        values=values,
        policy_offsets=list(requests.legal_offsets),
        policy_logits=logits,
    )


def complete(search, *, max_rows=None, concentrated=False):
    roots = search.root_requests()
    search.initialize_roots(*response(roots, concentrated=concentrated).submit_args())
    sizes = []
    while not search.is_done():
        requests = search.next_requests(max_rows=max_rows)
        if not len(requests):
            continue
        sizes.append(len(requests))
        search.submit(*response(requests, concentrated=concentrated).submit_args())
    return search.results(), sizes


def fingerprint(result):
    return tuple(
        getattr(result, name)
        for name in (
            "selected_actions",
            "terminal",
            "terminal_values",
            "root_values",
            "selected_action_values",
            "action_offsets",
            "actions",
            "visits",
            "q_values",
            "priors",
            "policy_target",
        )
    )


@pytest.mark.native
@pytest.mark.parametrize(
    "mode,handicap,pie",
    [
        (0, 1, False),
        (1, 1, False),
        (0, 4, False),
        (1, 4, False),
        (0, 1, True),
        (1, 1, True),
    ],
)
@pytest.mark.parametrize("width,cap", [(2, 3), (8, 3), (8, 24)])
def test_prefetch_preserves_native_results_with_global_caps(
    mode, handicap, pie, width, cap
):
    native = pytest.importorskip("star_native")
    states = native.StateBatch(
        4, 3, mode="classic" if mode == 0 else "double", handicap=handicap, pie=pie
    )
    options = dict(
        simulations=64,
        max_considered=8,
        simulations_per_root=[5, 17, 64],
        seeds_per_root=[17, 23, 31],
        pda_by_seat=[(0, 0), (2, -2), (-1, 1)],
    )
    baseline, old_sizes = complete(native.SearchBatch(states, **options))
    search = native.SearchBatch(states, **options, first_visit_batch_size=width)
    actual, sizes = complete(search, max_rows=cap)
    assert fingerprint(actual) == fingerprint(baseline)
    assert sum(sizes) == sum(old_sizes)
    assert max(sizes) <= cap
    assert actual.reused_nodes == [0, 0, 0]
    for row, budget in enumerate([5, 17, 64]):
        assert (
            sum(
                actual.visits[
                    actual.action_offsets[row] : actual.action_offsets[row + 1]
                ]
            )
            == budget
        )


@pytest.mark.native
def test_single_root_prefetch_reduces_round_trips_without_extra_evaluations():
    native = pytest.importorskip("star_native")
    states = native.StateBatch(4, 1)
    options = dict(simulations=16, max_considered=16, deterministic_seed=17)
    baseline, old_sizes = complete(native.SearchBatch(states, **options))
    actual, sizes = complete(
        native.SearchBatch(states, **options, first_visit_batch_size=8)
    )
    assert fingerprint(actual) == fingerprint(baseline)
    assert len(old_sizes) == 16 and sizes == [8, 8]
    assert sum(sizes) == sum(old_sizes)


@pytest.mark.native
@pytest.mark.parametrize(
    "invalidate", [None, "model", "pda", "parameters", "cap", "same-root"]
)
def test_reuse_keeps_work_separate_and_invalidates_unsafe_context(invalidate):
    native = pytest.importorskip("star_native")
    states = native.StateBatch(4, 1)
    states.apply_many([0], [0])
    options = dict(
        simulations=64,
        max_considered=4,
        deterministic_seed=17,
        first_visit_batch_size=4,
        model_context="immutable-model-a",
    )
    search = native.SearchBatch(states, **options)
    before, _ = complete(search, concentrated=True)
    if invalidate != "same-root":
        states.apply_many([0], before.selected_actions)
    advanced = dict(options, simulations=7, deterministic_seed=23)
    if invalidate == "model":
        advanced["model_context"] = "immutable-model-b"
    elif invalidate == "pda":
        advanced["pda_by_seat"] = [(2, -2)]
    elif invalidate == "parameters":
        advanced["c_scale"] = 0.25
    assert search.can_reuse(states, "immutable-model-a") is (invalidate != "same-root")
    search.advance(
        states,
        **advanced,
        reuse_tree=True,
        max_reused_nodes=1 if invalidate == "cap" else 4096,
    )
    actual, _ = complete(search, concentrated=True)
    assert sum(actual.visits) == 7
    assert actual.total_visits == [
        old + new for old, new in zip(actual.inherited_visits, actual.visits)
    ]
    if invalidate is None:
        assert actual.reused_nodes[0] > 0 and actual.reused_simulations[0] > 0
        assert sum(actual.inherited_visits) == actual.reused_simulations[0]
    else:
        assert actual.reused_nodes == [0] and actual.reused_simulations == [0]
        fresh, _ = complete(native.SearchBatch(states, **advanced), concentrated=True)
        assert fingerprint(actual) == fingerprint(fresh)


@pytest.mark.native
def test_experimental_responses_are_atomic_and_cancellation_rejects_stale_tokens():
    native = pytest.importorskip("star_native")
    states = native.StateBatch(4, 2)
    options = dict(
        simulations=8, max_considered=8, deterministic_seed=17, first_visit_batch_size=4
    )
    search = native.SearchBatch(states, **options)
    roots = search.root_requests()
    search.initialize_roots(*response(roots).submit_args())
    pending = search.next_requests()
    valid = response(pending)
    invalid_values = list(valid.values)
    invalid_values[-1] = math.nan
    with pytest.raises(ValueError, match="finite"):
        search.submit(
            valid.tokens, invalid_values, valid.policy_offsets, valid.policy_logits
        )
    search.submit(*valid.submit_args())
    pending = search.next_requests()
    stale = response(pending)
    search.cancel_pending()
    with pytest.raises(RuntimeError, match="pending"):
        search.submit(*stale.submit_args())
    while not search.is_done():
        request = search.next_requests()
        if len(request):
            search.submit(*response(request).submit_args())
    baseline, _ = complete(native.SearchBatch(states, **options))
    assert fingerprint(search.results()) == fingerprint(baseline)


@pytest.mark.native
def test_budget_overrides_only_before_initialization_and_advance_is_atomic():
    native = pytest.importorskip("star_native")
    states = native.StateBatch(4, 1)
    search = native.SearchBatch(
        states, simulations=16, first_visit_batch_size=8, model_context="a"
    )
    with pytest.raises(RuntimeError, match="completed"):
        search.advance(states, model_context="a")
    search.set_simulations_per_root([3])
    with pytest.raises(ValueError, match="positive"):
        search.set_simulations_per_root([0])
    assert search.budgets == [3]
    result, _ = complete(search)
    assert sum(result.visits) == 3
    with pytest.raises(RuntimeError, match="before"):
        search.set_simulations_per_root([5])
    with pytest.raises(ValueError, match="model_context"):
        search.advance(states, reuse_tree=True)
    assert fingerprint(search.results()) == fingerprint(result)
