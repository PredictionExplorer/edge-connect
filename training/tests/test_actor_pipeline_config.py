from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import migrate_continuous_profile as migration
from startrain.balanced_evaluation import evaluation_contract
from startrain.config import (
    ActorInferenceConfig,
    ActorPipelineConfig,
    ConfigError,
    GPUWorkerConfig,
    load_config,
)
from startrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_selfplay_pipeline_defaults,
)
from startrain.learner import UTDSegmentState
from test_continuous_profile_migration import _fixture, _snapshot, _write_json


PROFILE = Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
LEGACY_PRODUCTION_CONFIG_SHA256 = (
    # Canonical materialized profile from the actual 0c734d1 package.
    "4d85be61d94ad886e7ff8d8ddcfb5b89680719eb2c63e6a7f48b3525ece7d0f9"
)


def pipeline():
    return ActorPipelineConfig(
        compatible_work=True,
        stream_completed_games=True,
        rolling_game_slots=True,
        seed_contract="game-v1",
        games_per_task=512,
        cuda_graphs=True,
        max_model_pin_seconds=1200.0,
    )


def test_per_gpu_pipeline_round_trips_and_is_isolated_to_selected_actor(tmp_path):
    raw = yaml.safe_load(PROFILE.read_text())
    selected = next(gpu for gpu in raw["orchestration"]["gpus"] if gpu["gpu_id"] == 7)
    selected["actor_pipeline"] = asdict(pipeline())
    path = tmp_path / "canary.yaml"
    path.write_text(yaml.safe_dump(raw))
    parsed = load_config(path)
    assert parsed.orchestration.gpus[7].actor_pipeline == pipeline()
    assert all(gpu.actor_pipeline is None for gpu in parsed.orchestration.gpus[:7])
    path.write_text(yaml.safe_dump(parsed.as_dict()))
    assert load_config(path) == parsed
    selected["actor_pipeline"]["unknown_control"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize(
    "field",
    ["compatible_work", "stream_completed_games", "rolling_game_slots", "cuda_graphs"],
)
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_pipeline_switches_require_strict_booleans(field, value):
    with pytest.raises(ConfigError, match="boolean"):
        ActorPipelineConfig(**{field: value})


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "512"])
def test_pipeline_task_quota_requires_a_positive_integer(value):
    with pytest.raises(ConfigError, match="games_per_task"):
        ActorPipelineConfig(games_per_task=value)


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, float("nan"), float("inf"), "10"]
)
def test_model_pin_age_requires_finite_positive_duration(value):
    with pytest.raises(ConfigError, match="max_model_pin_seconds"):
        ActorPipelineConfig(max_model_pin_seconds=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"rolling_game_slots": True},
        {"rolling_game_slots": True, "stream_completed_games": True},
        {"rolling_game_slots": True, "seed_contract": "game-v1"},
        {"seed_contract": "future-version"},
    ],
)
def test_rolling_requires_explicit_streaming_and_game_seed_contract(changes):
    with pytest.raises(ConfigError):
        ActorPipelineConfig(**changes)


def test_pipeline_is_forbidden_on_learner_and_rejects_untyped_direct_settings():
    with pytest.raises(ConfigError, match="actor GPU"):
        GPUWorkerConfig(
            gpu_id=0, role="learner", cpu_threads=4, actor_pipeline=pipeline()
        )
    with pytest.raises(ConfigError, match="typed settings"):
        GPUWorkerConfig(
            gpu_id=1,
            role="actor",
            cpu_threads=4,
            actor_batch_size=128,
            actor_pipeline={},
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"cuda_graphs": 1},
        {"cuda_graphs": "true"},
        {"cuda_graph_max_entries": 0},
        {"cuda_graph_max_entries": 65},
        {"cuda_graph_max_entries": True},
        {"cuda_graph_max_entries": 8.0},
        {"cuda_graph_max_bytes": 0},
        {"cuda_graph_max_bytes": -1},
        {"cuda_graph_max_bytes": True},
        {"cuda_graph_max_bytes": float("inf")},
        {"cuda_graph_max_bytes": 1024.0},
    ],
)
def test_global_cuda_graph_storage_is_explicit_and_bounded(changes):
    with pytest.raises(ConfigError):
        ActorInferenceConfig(**changes)
    assert (
        ActorInferenceConfig(
            cuda_graphs=True, cuda_graph_max_entries=64, cuda_graph_max_bytes=1
        ).cuda_graph_max_entries
        == 64
    )


def test_pipeline_omission_matches_the_exact_previous_production_config_hash():
    config = load_config(PROFILE)
    payload = config.as_dict()
    before = deepcopy(payload)
    old = without_selfplay_pipeline_defaults(payload)
    checksum = hashlib.sha256(migration._canonical_config_bytes(old)).hexdigest()
    assert checksum == LEGACY_PRODUCTION_CONFIG_SHA256
    assert checksum in migration._compatible_source_config_sha256s(config)
    assert old in compatible_config_epoch_payloads(payload)
    assert payload == before
    assert old["model"] == payload["model"]
    assert old["train"] == payload["train"]
    assert old["arena"] == payload["arena"]
    assert (
        old["orchestration"]["model_refresh"]["inference"][
            "preserve_broadcast_topology"
        ]
        is True
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("selfplay", "stream_completed_games"), True),
        (("selfplay", "stream_completed_games"), 0),
        (("selfplay", "rolling_game_slots"), 0),
        (("selfplay", "seed_contract"), "game-v1"),
        (("orchestration", "model_refresh", "compatible_cohort_work"), 0),
        (("orchestration", "model_refresh", "inference", "cuda_graphs"), True),
        (("orchestration", "model_refresh", "inference", "cuda_graphs"), 0),
        (
            ("orchestration", "model_refresh", "inference", "cuda_graph_max_entries"),
            8.0,
        ),
        (("orchestration", "model_refresh", "inference", "cuda_graph_max_entries"), 16),
        (
            ("orchestration", "model_refresh", "inference", "cuda_graph_max_bytes"),
            float(2 * 1024**3),
        ),
    ],
)
def test_pipeline_compatibility_never_discards_enabled_or_untyped_values(path, value):
    payload = load_config(PROFILE).as_dict()
    parent = payload
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = value
    for result in compatible_config_epoch_payloads(payload):
        actual = result
        for part in path:
            actual = actual[part]
        assert actual == value and type(actual) is type(value)


def test_pipeline_compatibility_preserves_explicit_nested_and_unknown_settings():
    payload = load_config(PROFILE).as_dict()
    payload["orchestration"]["gpus"][7]["actor_pipeline"] = asdict(
        ActorPipelineConfig()
    )
    payload["selfplay"]["unknown_control"] = False
    payload["orchestration"]["model_refresh"]["inference"]["unknown_cuda_control"] = 0
    result = without_selfplay_pipeline_defaults(payload)
    assert result["orchestration"]["gpus"][7]["actor_pipeline"] == asdict(
        ActorPipelineConfig()
    )
    assert result["selfplay"]["unknown_control"] is False
    assert (
        result["orchestration"]["model_refresh"]["inference"]["unknown_cuda_control"]
        == 0
    )


def test_gpu7_canary_and_global_streaming_migration_preserve_training_and_pending_work(
    tmp_path,
):
    fixture = _fixture(tmp_path, PROFILE.name)
    old = load_config(fixture.old_profile)
    raw = yaml.safe_load(fixture.old_profile.read_text())
    raw["selfplay"]["stream_completed_games"] = True
    raw["orchestration"]["gpus"][7]["actor_pipeline"] = asdict(pipeline())
    fixture.candidate_profile.write_text(yaml.safe_dump(raw))
    _write_json(fixture.root / "arena/promotion-status.json", {"terminal": False})
    _write_json(
        fixture.root / "arena/pending.resume.json",
        {"arena_state": {"game_states": [{"actions": [1, 2, 3]}]}},
    )
    _write_json(
        fixture.root / "utd-segment.json",
        UTDSegmentState(
            run_id="continuous-test-run",
            generation_family="family-continuous-test",
            target_updates_per_new_sample=old.learner.target_updates_per_new_sample,
            baseline_examples_consumed=0,
            baseline_committed_replay_samples=0,
        ).as_dict(),
    )
    retained = [
        fixture.checkpoint,
        *(
            fixture.root / name
            for name in (
                "run.json",
                "learner/recovery.json",
                "learner/champion.json",
                "utd-segment.json",
                "arena/promotion-status.json",
                "arena/pending.resume.json",
            )
        ),
    ]
    before = {path: path.read_bytes() for path in retained}
    snapshot = _snapshot(fixture.root)
    plan = migration.plan_migration(fixture.request)
    assert _snapshot(fixture.root) == snapshot
    migration.apply_migration(plan)
    active = load_config(plan.target_profile)
    for name in ("model", "optimizer", "train", "arena", "learner", "game"):
        assert getattr(active, name) == getattr(old, name)
    assert evaluation_contract(active.arena) == evaluation_contract(old.arena)
    assert {path: path.read_bytes() for path in retained} == before
    assert active.selfplay.stream_completed_games is True
    assert active.selfplay.seed_contract == "cohort-v1"
    assert active.orchestration.gpus[7].actor_pipeline == pipeline()
    assert active.orchestration.gpus[:7] == old.orchestration.gpus[:7]
    record = json.loads((fixture.root / "continuous-migrations.jsonl").read_text())
    assert {change["path"] for change in record["changes"]} == {
        "selfplay.stream_completed_games",
        "orchestration.gpus.7.actor_pipeline",
    }
    assert "utd_segment" not in record
    assert "evaluation_contract_transition" not in record
