"""Bounded, non-barrier work bundles for independently running actor cohorts.

A bundle contains a fixed number of compatible leases. Any free producer may
claim the next lease; it never waits for another producer to finish a game.
The bundle owns one resource reservation until every lease has acquired its
own pin. Model/board choices are joint; conditional per-lease mode choices avoid
both single-mode bundles and correlations with the model schedule.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
import hashlib
import math
import random
import threading
from types import MappingProxyType
from typing import Any


class WeightedFairChoice:
    """Smooth weighted selection with deterministic, seed-specific tie breaks."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._weights: dict[Hashable, float] = {}
        self._credits: dict[Hashable, float] = {}

    def choose(self, weights: Mapping[Hashable, float]) -> Any:
        if not weights or any(
            not math.isfinite(value) or value < 0 for value in weights.values()
        ):
            raise ValueError("work weights must be finite, nonnegative and nonempty")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("work weights must have positive mass")
        normalized = {key: value / total for key, value in weights.items() if value > 0}
        if normalized != self._weights:
            # A curriculum/availability change begins a new weighted segment;
            # obsolete debt must not force work outside its current support.
            self._weights = normalized
            self._credits = dict.fromkeys(normalized, 0.0)
        for key, value in normalized.items():
            self._credits[key] += value
        selected = max(
            normalized,
            key=lambda key: (
                self._credits[key],
                hashlib.sha256(f"{self.seed}:{key!r}".encode()).digest(),
            ),
        )
        self._credits[selected] -= 1.0
        return selected


@dataclass(frozen=True)
class WorkBundle:
    metadata: Mapping[str, Any]
    acquire: Callable[[], Any]
    release_reservation: Callable[[], None]
    lease_metadata: tuple[Mapping[str, Any], ...] | None = None


@dataclass(frozen=True)
class WorkLease:
    bundle_id: int
    index: int
    metadata: Mapping[str, Any]
    resource: Any


class CompatibleWorkCoordinator:
    def __init__(self, *, cohort_count: int, seed: int) -> None:
        if type(cohort_count) is not int or not 2 <= cohort_count <= 32:
            raise ValueError("compatible work requires 2..32 producer cohorts")
        self.cohort_count = cohort_count
        self.random = random.Random(seed)
        self.choice = WeightedFairChoice(seed)
        self._mode_choices: dict[Hashable, WeightedFairChoice] = {}
        self._severity_choices: dict[Hashable, WeightedFairChoice] = {}
        self._lock = threading.Lock()
        self._bundle: WorkBundle | None = None
        self._bundle_id = 0
        self._next_index = 0
        self._closed = False
        self._issued = 0
        self._requested_roles: Counter[str] = Counter()
        self._actual_roles: Counter[str] = Counter()
        self._rings: Counter[str] = Counter()
        self._modes: Counter[str] = Counter()
        self._outstanding: dict[tuple[int, int], int] = {}
        self._planned: Counter[str] = Counter()
        self._completed: Counter[str] = Counter()
        self._planned_rings: Counter[str] = Counter()
        self._planned_modes: Counter[str] = Counter()
        self._completed_rings: Counter[str] = Counter()
        self._completed_modes: Counter[str] = Counter()
        self._completed_actual_roles: Counter[str] = Counter()
        self._started = 0
        self._dropped = 0
        self._cancelled_unstarted = 0

    def choose_severity(self, key: Hashable, minimum: int, maximum: int) -> int:
        if not 2 <= minimum <= maximum <= 9:
            raise ValueError("invalid work handicap severity range")
        if key not in self._severity_choices:
            self._severity_choices[key] = WeightedFairChoice(self.choice.seed)
        return self._severity_choices[key].choose(
            dict.fromkeys(range(minimum, maximum + 1), 1.0)
        )

    def choose_mode(self, role_and_ring: Hashable, weights: Mapping[str, float]) -> str:
        if role_and_ring not in self._mode_choices:
            salt = int.from_bytes(
                hashlib.sha256(repr(role_and_ring).encode()).digest()[:8], "big"
            )
            self._mode_choices[role_and_ring] = WeightedFairChoice(
                self.choice.seed ^ salt
            )
        return self._mode_choices[role_and_ring].choose(
            {key: value for key, value in weights.items()}
        )

    def acquire(
        self, factory: Callable[["CompatibleWorkCoordinator"], WorkBundle]
    ) -> WorkLease:
        with self._lock:
            if self._closed:
                raise RuntimeError("work coordinator is closed")
            if self._bundle is None:
                created = factory(self)
                if (
                    created.lease_metadata is not None
                    and len(created.lease_metadata) != self.cohort_count
                ):
                    created.release_reservation()
                    raise ValueError(
                        "per-lease metadata count differs from producer count"
                    )
                self._bundle = WorkBundle(
                    MappingProxyType(dict(created.metadata)),
                    created.acquire,
                    created.release_reservation,
                    tuple(
                        MappingProxyType(dict(item)) for item in created.lease_metadata
                    )
                    if created.lease_metadata is not None
                    else None,
                )
                self._bundle_id += 1
                self._next_index = 0
            bundle = self._bundle
            metadata = dict(bundle.metadata)
            if bundle.lease_metadata is not None:
                metadata.update(bundle.lease_metadata[self._next_index])
            metadata = MappingProxyType(metadata)
            # Acquire the individual model pin before releasing the final bundle
            # reservation, including when the first producer finished early.
            resource = bundle.acquire()
            lease = WorkLease(self._bundle_id, self._next_index, metadata, resource)
            self._next_index += 1
            self._issued += 1
            self._requested_roles[str(metadata["requested_model_role"])] += 1
            self._actual_roles[str(metadata["model_role"])] += 1
            self._rings[str(metadata["ring"])] += 1
            self._modes[str(metadata["mode_category"])] += 1
            games = int(metadata["games"])
            self._outstanding[(lease.bundle_id, lease.index)] = games
            self._planned[str(metadata["requested_model_role"])] += games
            self._planned_rings[str(metadata["ring"])] += games
            self._planned_modes[str(metadata["mode_category"])] += games
            if self._next_index == self.cohort_count:
                self._bundle = None
                bundle.release_reservation()
            return lease

    def record_outcome(
        self,
        lease: WorkLease,
        *,
        requested: int,
        started: int,
        completed: int,
        dropped: int,
        cancelling: bool,
    ) -> int:
        """Return unissued quota for a fresh-model continuation, never new debt."""
        if (
            min(requested, started, completed, dropped) < 0
            or started != completed + dropped
            or started > requested
        ):
            raise ValueError("invalid compatible work game accounting")
        with self._lock:
            key = (lease.bundle_id, lease.index)
            if self._outstanding.get(key) != requested:
                raise ValueError("work outcome does not match outstanding quota")
            self._started += started
            self._dropped += dropped
            self._completed[str(lease.metadata["requested_model_role"])] += completed
            self._completed_rings[str(lease.metadata["ring"])] += completed
            self._completed_modes[str(lease.metadata["mode_category"])] += completed
            self._completed_actual_roles[str(lease.metadata["model_role"])] += completed
            remaining = requested - started
            if cancelling:
                self._cancelled_unstarted += remaining
                remaining = 0
            if remaining:
                self._outstanding[key] = remaining
            else:
                del self._outstanding[key]
            return remaining

    def metrics_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "closed": self._closed,
                "bundles_created": self._bundle_id,
                "leases_issued": self._issued,
                "pending_leases": self.cohort_count - self._next_index
                if self._bundle is not None
                else 0,
                "requested_model_roles": dict(self._requested_roles),
                "actual_model_roles": dict(self._actual_roles),
                "rings": dict(self._rings),
                "mode_categories": dict(self._modes),
                "planned_games_by_requested_role": dict(self._planned),
                "completed_games_by_requested_role": dict(self._completed),
                "completed_games_by_actual_role": dict(self._completed_actual_roles),
                "planned_games_by_ring": dict(self._planned_rings),
                "completed_games_by_ring": dict(self._completed_rings),
                "planned_games_by_mode": dict(self._planned_modes),
                "completed_games_by_mode": dict(self._completed_modes),
                "game_accounting_scope": "completed_task_outcomes; active durable publications appear in actor metrics",
                "started_games": self._started,
                "dropped_games": self._dropped,
                "cancelled_unstarted_games": self._cancelled_unstarted,
                "outstanding_promised_games": sum(self._outstanding.values()),
                "pending_model_identity": self._bundle.metadata.get("model_identity")
                if self._bundle
                else None,
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            bundle, self._bundle = self._bundle, None
            if bundle is not None:
                bundle.release_reservation()
