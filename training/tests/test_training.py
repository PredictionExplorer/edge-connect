from pathlib import Path

import numpy as np
import pytest
import torch

from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    save_checkpoint,
)
from startrain.config import SchedulerConfig, load_config
from startrain.export import ONNX_INPUT_NAMES, ONNXStarModel, export_onnx
from startrain.features import DoubleStarPosition, encode_batch
from startrain.model import GraphResTNet, ModelConfig
from startrain.optim import (
    MuonAdamW,
    OptimizerConfig,
    build_optimizer,
    split_decay_parameters,
)
from startrain.replay import ReplaySample, collate_replay_samples
from startrain.sampling import RingStratifiedSampler
from startrain.scoring import score_position
from startrain.topology import get_topology
from startrain.training import (
    build_scheduler,
    maybe_compile_model,
    train_step,
)


CONFIGS = Path(__file__).parents[1] / "configs"


def tiny_model() -> GraphResTNet:
    return GraphResTNet(
        ModelConfig(
            width=16,
            rrt_groups=5,
            attention_heads=4,
            kv_heads=1,
        )
    )


def sample() -> ReplaySample:
    topology = get_topology(3)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    position = DoubleStarPosition(
        rings=3,
        stones=stones,
        to_move=1,
        moves_left=2,
        opening=False,
        pass_streak=0,
        terminal=False,
    )
    legal = np.concatenate(((stones.numpy() == -1), np.asarray([True])))
    policy = legal.astype(np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=score_position(topology, stones),
        search_provenance="mcts:test",
        policy_provenance="root-visits",
    )


def test_yaml_configs_load_strictly() -> None:
    small = load_config(CONFIGS / "small.yaml")
    h100 = load_config(CONFIGS / "h100.yaml")
    assert small.schema_version == h100.schema_version == 2
    assert small.model.rrt_groups == h100.model.rrt_groups == 5
    assert h100.model.kv_heads < h100.model.attention_heads
    assert small.train.precision == "fp32"
    assert h100.train.precision == "bf16"
    assert h100.train.global_batch_size(4) == (h100.train.per_rank_batch_size * 4)
    assert h100.profile == "standalone-smoke"
    assert h100.data.ring_stratified is False
    continuous = load_config(CONFIGS / "h100-8gpu.yaml")
    assert continuous.profile == "continuous"
    assert continuous.data.ring_stratified is True
    assert continuous.orchestration.model_refresh.selfplay_source == "champion"
    assert continuous.orchestration.model_refresh.candidate_probability == 0.8
    assert continuous.selfplay.record_fast_policy_targets is False
    assert continuous.selfplay.max_considered_cap == 64


def test_optimizer_decay_groups_and_muon_selection() -> None:
    model = tiny_model()
    decay, no_decay = split_decay_parameters(model)
    no_decay_names = {name for name, _ in no_decay}
    assert "global_token" in no_decay_names
    assert all(parameter.ndim >= 2 for _, parameter in decay)
    assert any("edge_embedding" in name for name in no_decay_names)

    adamw = build_optimizer(model, OptimizerConfig(kind="adamw"))
    assert {float(group["weight_decay"]) for group in adamw.param_groups} == {
        0.0,
        0.01,
    }
    muon = build_optimizer(
        model,
        OptimizerConfig(kind="muon_adamw", min_muon_elements=1, muon_ns_steps=2),
    )
    assert isinstance(muon, MuonAdamW)
    assert any(group["algorithm"] == "muon" for group in muon.param_groups)
    assert any(
        group["algorithm"] == "adamw" and group["weight_decay"] == 0
        for group in muon.param_groups
    )


def test_bf16_compiled_train_step_scheduler_and_checkpoint(tmp_path) -> None:
    torch.manual_seed(3)
    model = tiny_model()
    compiled = maybe_compile_model(
        model, enabled=True, dynamic=True, fullgraph=True, backend="eager"
    )
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(warmup_steps=1, total_steps=10)
    )
    ema = ExponentialMovingAverage(model, decay=0.9)
    result = train_step(
        compiled,
        collate_replay_samples([sample()]),
        optimizer,
        precision="bf16",
        gradient_clip_norm=0.5,
        scheduler=scheduler,
        ema=ema,
    )
    assert all(np.isfinite(value) for value in result.losses.values())
    assert np.isfinite(result.gradient_norm)
    assert ema.num_updates == 1

    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=9,
        epoch=2,
        config={"schema_version": 2},
    )
    restored = tiny_model()
    restored_optimizer = build_optimizer(restored, OptimizerConfig(kind="adamw"))
    restored_scheduler = build_scheduler(
        restored_optimizer, SchedulerConfig(warmup_steps=1, total_steps=10)
    )
    restored_ema = ExponentialMovingAverage(restored)
    metadata = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        ema=restored_ema,
    )
    assert metadata["step"] == 9
    assert metadata["epoch"] == 2
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for expected, actual in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(expected, actual)

    tampered_path = tmp_path / "tampered-checkpoint.pt"
    payload = torch.load(path, weights_only=True)
    payload["rules_hash"] = 1
    torch.save(payload, tampered_path)
    with pytest.raises(ValueError, match="rules hash"):
        load_checkpoint(tampered_path, model=tiny_model())


def test_ring_stratified_sampler_remains_balanced() -> None:
    rings = [3] * 8 + [12] * 2
    sampler = RingStratifiedSampler(rings, num_samples=20, seed=11)
    sampled = [rings[index] for index in sampler]
    assert sampled.count(3) == sampled.count(12) == 10


def test_onnx_parity_when_runtime_is_available(tmp_path) -> None:
    runtime = pytest.importorskip("onnxruntime")
    model = tiny_model().eval()
    position = sample().to_position()
    batch = encode_batch([position])
    path = export_onnx(model, batch, tmp_path / "model.onnx")
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    topology4 = get_topology(4)
    position4 = DoubleStarPosition(
        rings=4,
        stones=torch.full((topology4.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=2,
        opening=False,
        pass_streak=0,
        terminal=False,
    )
    for inference_batch in (batch, encode_batch([position4])):
        feed = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                ONNX_INPUT_NAMES, inference_batch.model_args(), strict=True
            )
        }
        actual = session.run(None, feed)
        with torch.no_grad():
            expected = ONNXStarModel(model)(*inference_batch.model_args())
        for expected_tensor, actual_array in zip(expected, actual, strict=True):
            np.testing.assert_allclose(
                expected_tensor.numpy(), actual_array, atol=2e-4, rtol=2e-4
            )
