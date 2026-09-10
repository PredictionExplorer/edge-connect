"""Bounded ownership transfer for completed native search sessions."""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any, Sequence


class CompletedSearchCache:
    """Thread-safe pool; callers relinquish a session after successful ``put``.

    Checked-out sessions are exclusively owned by their caller. Only successfully
    completed searches may be returned; failed or cancelled work is discarded by
    the caller. The node limit charges all retained roots across the whole pool.
    """

    def __init__(self, capacity: int = 8, max_nodes: int = 4_096) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 1_024:
            raise ValueError("completed search capacity must be an integer in 1..1024")
        if type(max_nodes) is not int or not 1 <= max_nodes <= 65_536:
            raise ValueError("completed search max_nodes must be in 1..65536")
        self.capacity = capacity
        self.max_nodes = max_nodes
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, tuple[Any, int]] = OrderedDict()
        self._nodes = 0
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0
        self._rejected = 0

    @staticmethod
    def _node_count(search: Any) -> int:
        counts = list(search.unique_state_counts)
        if not counts or any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("native search unique-state counts are invalid")
        return sum(counts)

    def take(
        self, states: Any, model_context: str, pda_by_seat: Sequence[tuple[int, int]]
    ) -> Any | None:
        """Remove the newest compatible complete session, without advancing it."""
        if not isinstance(model_context, str) or not model_context:
            raise ValueError("completed search reuse requires a nonempty model context")
        with self._lock:
            for key in reversed(tuple(self._entries)):
                search, nodes = self._entries[key]
                if not search.is_done() or self._node_count(search) != nodes:
                    # A caller violated ownership after put. Never hand out a
                    # pending session or trust its stale resource accounting.
                    del self._entries[key]
                    self._nodes -= nodes
                    self._rejected += 1
                    continue
                if search.can_reuse(
                    states, model_context=model_context, pda_by_seat=list(pda_by_seat)
                ):
                    del self._entries[key]
                    self._nodes -= nodes
                    self._hits += 1
                    return search
            self._misses += 1
            return None

    def put(self, search: Any) -> bool:
        """Transfer a completed session to the pool, or reject it unchanged."""
        with self._lock:
            key = id(search)
            if key in self._entries or not search.is_done():
                self._rejected += 1
                return False
            nodes = self._node_count(search)
            if not 0 < nodes <= self.max_nodes:
                self._rejected += 1
                return False
            while self._entries and (
                len(self._entries) >= self.capacity
                or self._nodes + nodes > self.max_nodes
            ):
                _, (_, removed) = self._entries.popitem(last=False)
                self._nodes -= removed
                self._evictions += 1
            self._entries[key] = (search, nodes)
            self._nodes += nodes
            self._puts += 1
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._nodes = 0

    def metrics_snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "retained_nodes": self._nodes,
                "capacity": self.capacity,
                "max_nodes": self.max_nodes,
                "hits": self._hits,
                "misses": self._misses,
                "puts": self._puts,
                "evictions": self._evictions,
                "rejected": self._rejected,
            }
