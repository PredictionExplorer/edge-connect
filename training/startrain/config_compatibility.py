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
    """Preserve existing guards across additive performance/scheduling releases.

    The pre-broadcast representation receives the same scheduling, pause and
    efficiency omissions as the current one. Older independent guards can
    operate on every representation without losing any previously accepted
    epoch or searching arbitrary subsets of newly added fields.
    """

    current = deepcopy(dict(payload))
    representations = [current]
    pre_clipping = without_gradient_clipping_defaults(payload)
    if pre_clipping != current:
        representations.append(pre_clipping)
    sources: list[dict[str, Any]] = []
    for representation in representations:
        sources.append(representation)
        pre_broadcast = without_broadcast_topology_default(representation)
        if pre_broadcast != representation:
            sources.append(pre_broadcast)
    variants: list[dict[str, Any]] = []
    for source in sources:
        pre_session = without_evaluation_session_defaults(source)
        previous = (
            source,
            without_efficiency_defaults(source),
            pre_session,
            without_efficiency_defaults(pre_session),
        )
        variants.extend(previous)
        if without_pause_strategy_default(source) != source:
            variants.extend(without_pause_strategy_default(row) for row in previous)
    return tuple(variants)


def without_gradient_clipping_defaults(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Represent the release before opt-in clipping and diagnostic settings.

    Every non-default, unknown field, and untyped lookalike remains authoritative.
    Diagnostics are deliberately an explicit profile change when enabled.
    """
    result = deepcopy(dict(payload))
    train = result.get("train")
    if not isinstance(train, dict):
        return result
    defaults = {
        "mode": "global",
        "beta": 0.99,
        "multiplier": 1.04,
        "warmup_steps": 100,
    }
    value = train.get("gradient_clipping")
    if (
        isinstance(value, dict)
        and value.keys() == defaults.keys()
        and all(
            type(value[key]) is type(default) and value[key] == default
            for key, default in defaults.items()
        )
    ):
        del train["gradient_clipping"]
    if train.get("gradient_diagnostics") is False:
        del train["gradient_diagnostics"]
    return result


def without_broadcast_topology_default(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Represent releases predating the opt-in shared-topology optimization.

    Only the exact disabled boolean is additive. Enabling the optimization
    requires an explicit profile migration even when other services are active.
    """

    result = deepcopy(dict(payload))
    parent: object = result
    for name in ("orchestration", "model_refresh", "inference"):
        if not isinstance(parent, dict):
            return result
        parent = parent.get(name)
    if isinstance(parent, dict) and parent.get("preserve_broadcast_topology") is False:
        del parent["preserve_broadcast_topology"]
    return result


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
            inference = refresh.get("inference")
            if (
                isinstance(inference, dict)
                and inference.get("preserve_broadcast_topology") is False
            ):
                omit(
                    refresh,
                    "inference",
                    {**defaults, "preserve_broadcast_topology": False},
                )
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
