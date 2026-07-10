import pytest

from startrain.arena import ArenaRunner
from startrain.config import ArenaConfig
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.model import GraphResTNet, ModelConfig
from startrain.native import validate_native_module
from startrain.optim import OptimizerConfig, build_optimizer
from startrain.replay import collate_replay_samples
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity
from startrain.selfplay import SelfPlayActor, SelfPlayConfig, SelfPlayIdentity
from startrain.training import train_step


def test_true_native_tiny_game_replay_and_train_step_when_available(tmp_path) -> None:
    native = pytest.importorskip("star_native")
    validate_native_module(native)
    model = GraphResTNet(
        ModelConfig(
            width=8,
            rrt_groups=1,
            attention_heads=2,
            kv_heads=1,
        )
    )
    evaluator = GraphInferenceAdapter(
        model,
        config=InferenceConfig(
            precision="fp32",
            initial_pass_logit_penalty=3.0,
        ),
        model_version="sha256-" + "a" * 64,
        model_step=0,
        model_identity="sha256-" + "a" * 64,
    )
    identity = RunIdentity(tmp_path / "run.json", "run-native", "family-native", 1)
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor-native")
        summaries = SelfPlayActor(
            native,
            evaluator,
            store,
            SelfPlayConfig.cpu_smoke(seed=123),
            SelfPlayIdentity(
                identity.run_id,
                identity.generation_family,
                "actor-native",
                generation,
            ),
        ).run()
        assert summaries
        samples = store.load_recent_samples(
            sample_window=128,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=0,
            max_model_lag_steps=0,
        )
        assert samples
        optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
        result = train_step(
            model,
            collate_replay_samples([samples[0]]),
            optimizer,
            precision="fp32",
        )
        assert result.losses["total"] >= 0


def test_true_native_tiny_paired_arena_when_available() -> None:
    native = pytest.importorskip("star_native")
    validate_native_module(native)
    model = GraphResTNet(
        ModelConfig(
            width=8,
            rrt_groups=1,
            attention_heads=2,
            kv_heads=1,
        )
    )
    evaluator = GraphInferenceAdapter(
        model,
        config=InferenceConfig(precision="fp32"),
        model_version="sha256-" + "b" * 64,
        model_step=0,
        model_identity="sha256-" + "b" * 64,
    )
    result = ArenaRunner(
        native_module=native,
        candidate=evaluator,
        baseline=evaluator,
        config=ArenaConfig(
            rings=(3,),
            pairs_per_ring=2,
            simulations=1,
            max_considered=2,
            regression_floor_elo=-2_500.0,
        ),
    ).run()
    assert result["aggregate"]["games"] == 4
    first, second = result["games"][:2]
    assert first["opening_seed"] == second["opening_seed"]
    assert first["opening_action"] == second["opening_action"]
