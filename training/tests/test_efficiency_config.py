from dataclasses import replace
from pathlib import Path

import pytest

from startrain.config import (
    ActorInferenceConfig,
    CPUActorConfig,
    ConfigError,
    load_config,
)
from startrain.config_compatibility import without_efficiency_defaults


def profile():
    return load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-variant-stage-a.yaml"
    )


@pytest.mark.parametrize(
    "values",
    [
        {"cache_max_entries": 10},
        {"cache_max_bytes": 10},
        {"cache_max_entries": True, "cache_max_bytes": 10},
        {"max_wait_seconds": float("nan")},
        {"max_pending_requests": 0},
        {"pinned_transfers": 1},
    ],
)
def test_inference_bounds_reject_invalid_or_unbounded_settings(values):
    with pytest.raises(ConfigError):
        ActorInferenceConfig(**values)


def test_cpu_reservation_and_shared_actor_config_are_enforced():
    config = profile()
    cpu = CPUActorConfig(actor_id="actor-cpu-small", cpu_affinity="0-15")
    with pytest.raises(ConfigError, match="disjoint"):
        replace(config.orchestration, cpu_actors=(cpu,))
    with pytest.raises(ConfigError, match="reserved"):
        replace(cpu, actor_id="learner")
    gpu = replace(config.orchestration.actor_gpus[0], actor_lanes=1, actor_cohorts=2)
    gpus = tuple(
        gpu if old.gpu_id == gpu.gpu_id else old for old in config.orchestration.gpus
    )
    with pytest.raises(ConfigError, match="shared inference"):
        replace(config.orchestration, gpus=gpus)
    inference = replace(
        config.orchestration.model_refresh.inference, shared_batching=True
    )
    replace(
        config.orchestration,
        gpus=gpus,
        model_refresh=replace(config.orchestration.model_refresh, inference=inference),
    )


def test_prior_epoch_compatibility_does_not_erase_enabled_features():
    old = profile().as_dict()
    stripped = without_efficiency_defaults(old)
    assert "inference" not in stripped["orchestration"]["model_refresh"]
    assert "balanced_cells" not in stripped["arena"]
    assert "actor_cohorts" not in stripped["orchestration"]["gpus"][1]
    old["orchestration"]["model_refresh"]["inference"]["deduplicate"] = True
    old["arena"]["balanced_cells"] = True
    old["orchestration"]["gpus"][1]["actor_cohorts"] = 2
    selected = without_efficiency_defaults(old)
    assert (
        selected["orchestration"]["model_refresh"]["inference"]["deduplicate"] is True
    )
    assert selected["arena"]["balanced_cells"] is True
    assert selected["orchestration"]["gpus"][1]["actor_cohorts"] == 2
    assert old["arena"]["strength_simulations"] == 1024


def test_balanced_arena_requires_complete_rule_family_and_all_rings():
    config = profile()
    arena = replace(config.arena, balanced_cells=True)
    replace(config, arena=arena)
    with pytest.raises(ConfigError, match="balanced"):
        replace(arena, rings=(10,))
    with pytest.raises(ConfigError, match="game family"):
        replace(
            config,
            arena=arena,
            game=replace(
                config.game, variants=replace(config.game.variants, pie_allowed=False)
            ),
        )
