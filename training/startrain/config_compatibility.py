"""Explicit compatibility with profiles predating optional efficiency services.

Only semantically inert defaults may be omitted. This is a single release epoch,
not an exponential search over arbitrary configuration edits. Unknown and
non-default values remain part of the hash and cannot acquire legacy authority.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any


def without_efficiency_defaults(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the prior-epoch representation without changing enabled features."""

    result = deepcopy(dict(payload))

    def omit(parent: object, key: str, default: object) -> None:
        if not isinstance(parent, dict) or key not in parent:
            return
        value = parent[key]
        # Tuples become lists in serialized profiles; compare their JSON shape.
        equal = value == default and (
            type(value) is type(default)
            or isinstance(value, (list, tuple))
            and isinstance(default, (list, tuple))
        )
        if equal:
            del parent[key]

    orchestration = result.get("orchestration", {})
    if isinstance(orchestration, dict):
        omit(orchestration, "cpu_actors", ())
        omit(orchestration.get("promotion", {}), "cpu_affinity", None)
        for gpu in orchestration.get("gpus", ()):
            omit(gpu, "actor_cohorts", 1)
            omit(gpu, "native_threads", None)
            omit(gpu, "blas_threads", None)
        refresh = orchestration.get("model_refresh", {})
        if isinstance(refresh, dict):
            # Whole nested service did not exist in earlier profiles. Never
            # strip an enabled service, even if some of its limits are defaulted.
            defaults = {
                "cache_max_entries": 0,
                "cache_max_bytes": 0,
                "deduplicate": False,
                "pinned_transfers": False,
                "pinned_buffer_slots": 2,
                "homogeneous_relational_bias": False,
                "shared_batching": False,
                "max_batch_rows": 256,
                "max_pending_requests": 16,
                "max_wait_seconds": 0.002,
            }
            omit(refresh, "inference", defaults)
    arena = result.get("arena", {})
    omit(arena, "balanced_cells", False)
    omit(arena, "cell_regression_floor_elo", -100.0)
    omit(arena, "handicap_severity_cycle", (2, 4, 6, 9))
    omit(arena, "strength_simulations", 1024)
    selfplay = result.get("selfplay", {})
    omit(selfplay, "exact_endgame_max_empty", 0)
    omit(selfplay, "exact_endgame_max_nodes", 100000)
    return result
