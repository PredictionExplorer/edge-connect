"""Explicit compatibility with profiles predating additive configuration epochs.

Only known defaults may be omitted through explicit release representations.
This is not an exponential search over arbitrary configuration edits. Unknown
and non-default values remain in the hash and cannot acquire legacy authority.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any


def without_pause_strategy_default(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Represent releases whose actor pause strategy was always terminate."""

    result = deepcopy(dict(payload))
    orchestration = result.get("orchestration")
    if isinstance(orchestration, dict):
        promotion = orchestration.get("promotion")
        if (
            isinstance(promotion, dict)
            and promotion.get("pause_strategy") == "terminate"
        ):
            del promotion["pause_strategy"]
    return result


def without_evaluation_session_defaults(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Represent the release before resumable evaluation scheduling defaults.

    Scheduling limits do not change evaluation evidence, model identity, or
    optimizer budgets. Preserve every explicit non-default in hash authority.
    """

    result = deepcopy(dict(payload))
    orchestration = result.get("orchestration")
    if not isinstance(orchestration, dict):
        return result
    for section, defaults in (
        ("promotion", {"session_seconds": 300.0}),
        (
            "historical_evaluation",
            {"session_seconds": 300.0, "cooldown_seconds": 1_800.0},
        ),
    ):
        parent = orchestration.get(section)
        if not isinstance(parent, dict):
            continue
        for name, default in defaults.items():
            if type(parent.get(name)) is type(default) and parent[name] == default:
                del parent[name]
    return result


def compatible_config_epoch_payloads(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Preserve existing guards across scheduling and actor-pause releases.

    Existing independent compatibility guards for older additions can operate
    on each representation. The new scheduling fields strip as one release
    block, without multiplying by every newly added field or dropping a
    previously accepted representation that retained scheduling defaults.
    """

    pre_session = without_evaluation_session_defaults(payload)
    variants = (
        deepcopy(dict(payload)),
        without_efficiency_defaults(payload),
        pre_session,
        without_efficiency_defaults(pre_session),
    )
    if without_pause_strategy_default(payload) == payload:
        return variants
    return (*variants, *(without_pause_strategy_default(row) for row in variants))


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
