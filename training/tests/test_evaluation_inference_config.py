from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.run_lineage_arena import _profile_inference_config, load_candidate
from startrain.checkpoint import ExponentialMovingAverage, save_checkpoint
from startrain.config import ActorInferenceConfig, GameConfig, load_config
from startrain.contracts import FEATURE_SCHEMA_VERSION, LEGACY_FEATURE_SCHEMA_VERSION
from startrain.model import GraphResTNet, ModelConfig
from startrain.promotion import load_manifest_evaluator


def test_promotion_evaluator_receives_explicit_inference_flags(tmp_path, monkeypatch):
    import startrain.promotion as promotion

    experiment = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    flags = ActorInferenceConfig(
        cache_max_entries=32,
        cache_max_bytes=1_000_000,
        deduplicate=True,
        pinned_transfers=True,
        pinned_buffer_slots=3,
        preserve_broadcast_topology=True,
        homogeneous_relational_bias=True,
    )
    experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            model_refresh=replace(
                experiment.orchestration.model_refresh, inference=flags
            ),
        ),
    )
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(promotion, "GraphResTNet", lambda _: model)
    monkeypatch.setattr(promotion, "load_ema_checkpoint", lambda *_, **__: {"step": 3})
    monkeypatch.setattr(promotion, "maybe_compile_model", lambda model, **_: model)
    manifest = SimpleNamespace(
        checkpoint=tmp_path / "unused.pt",
        checkpoint_sha256="a" * 64,
        checkpoint_bytes=1,
        model_step=3,
        model_identity="a",
        model_version="a",
        run_id="run",
        generation_family="family",
    )
    evaluator = load_manifest_evaluator(experiment, manifest, device="cpu")
    for name in (
        "cache_max_entries",
        "cache_max_bytes",
        "deduplicate",
        "pinned_transfers",
        "pinned_buffer_slots",
        "preserve_broadcast_topology",
    ):
        assert getattr(evaluator.config, name) == getattr(flags, name)
    assert evaluator.homogeneous_relational_bias is True


def test_lineage_runtime_profile_preserves_each_feature_schema_and_fp32(tmp_path):
    profile = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    flags = ActorInferenceConfig(
        cache_max_entries=16,
        cache_max_bytes=500000,
        deduplicate=True,
        homogeneous_relational_bias=True,
    )
    profile = replace(
        profile,
        orchestration=replace(
            profile.orchestration,
            model_refresh=replace(profile.orchestration.model_refresh, inference=flags),
        ),
    )
    current = _profile_inference_config(profile, schema=FEATURE_SCHEMA_VERSION)
    legacy = _profile_inference_config(profile, schema=LEGACY_FEATURE_SCHEMA_VERSION)
    assert current.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert legacy.feature_schema_version == LEGACY_FEATURE_SCHEMA_VERSION
    assert current.precision == legacy.precision == "fp32"
    assert current.cache_max_entries == legacy.cache_max_entries == 16
    model_config = ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    model = GraphResTNet(model_config)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        ema=ExponentialMovingAverage(model),
        step=3,
        config={"model": asdict(model_config), "game": asdict(GameConfig())},
    )
    adapter, metadata = load_candidate(
        checkpoint,
        device=torch.device("cpu"),
        inference_config=current,
        homogeneous_relational_bias=True,
    )
    assert adapter.config == current and adapter.homogeneous_relational_bias
    assert metadata["precision"] == "fp32"
