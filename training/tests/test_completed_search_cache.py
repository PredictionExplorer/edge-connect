from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import weakref

import pytest

from startrain.search_sessions import CompletedSearchCache


class Session:
    def __init__(self, nodes, *, context="model-a", pda=((0, 0),), states="descendant"):
        self.unique_state_counts = list(nodes)
        self.done = True
        self.context, self.pda, self.states = context, list(pda), states

    def is_done(self):
        return self.done

    def can_reuse(self, states, *, model_context, pda_by_seat):
        return (states, model_context, pda_by_seat) == (
            self.states,
            self.context,
            self.pda,
        )


def test_pool_enforces_total_nodes_and_entry_bounds_with_oldest_eviction():
    cache = CompletedSearchCache(capacity=2, max_nodes=5)
    oldest, second, newest = Session([1, 1]), Session([2]), Session([1, 2])
    assert cache.put(oldest) and cache.put(second) and cache.put(newest)
    metrics = cache.metrics_snapshot()
    assert metrics["entries"] == 2
    assert metrics["retained_nodes"] == 5
    assert metrics["evictions"] == 1
    assert cache.take("descendant", "model-a", [(0, 0)]) is newest
    assert cache.take("descendant", "model-a", [(0, 0)]) is second
    assert cache.take("descendant", "model-a", [(0, 0)]) is None
    assert cache.metrics_snapshot()["retained_nodes"] == 0


def test_model_pda_and_semantic_mismatches_never_checkout_a_session():
    cache = CompletedSearchCache()
    session = Session([4])
    assert cache.put(session)
    assert cache.take("descendant", "model-b", [(0, 0)]) is None
    assert cache.take("descendant", "model-a", [(1, -1)]) is None
    assert cache.take("unrelated", "model-a", [(0, 0)]) is None
    assert cache.take("descendant", "model-a", [(0, 0)]) is session


def test_incomplete_oversized_and_duplicate_sessions_are_not_admitted():
    cache = CompletedSearchCache(max_nodes=5)
    pending = Session([2])
    pending.done = False
    assert cache.put(pending) is False
    assert cache.put(Session([3, 3])) is False
    complete = Session([2])
    assert cache.put(complete) is True
    assert cache.put(complete) is False
    assert cache.metrics_snapshot()["entries"] == 1
    assert cache.metrics_snapshot()["rejected"] == 3
    complete.done = False
    assert cache.take("descendant", "model-a", [(0, 0)]) is None
    assert cache.metrics_snapshot()["entries"] == 0


def test_concurrent_checkout_has_one_exclusive_owner():
    cache = CompletedSearchCache()
    session = Session([3])
    cache.put(session)
    barrier = threading.Barrier(8)

    def take():
        barrier.wait(timeout=3)
        return cache.take("descendant", "model-a", [(0, 0)])

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: take(), range(8)))
    assert sum(response is session for response in responses) == 1
    assert sum(response is None for response in responses) == 7
    assert cache.metrics_snapshot()["hits"] == 1
    assert cache.metrics_snapshot()["misses"] == 7


def test_clear_releases_all_owned_native_sessions():
    cache = CompletedSearchCache()
    session = Session([4])
    retained = weakref.ref(session)
    cache.put(session)
    del session
    assert retained() is not None
    cache.clear()
    gc.collect()
    assert retained() is None
    assert cache.metrics_snapshot()["retained_nodes"] == 0


@pytest.mark.parametrize(
    "options",
    [{"capacity": 0}, {"capacity": True}, {"max_nodes": 0}, {"max_nodes": 65_537}],
)
def test_pool_limits_are_strict_and_bounded(options):
    with pytest.raises(ValueError):
        CompletedSearchCache(**options)
