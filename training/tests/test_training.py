from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.validate_continuous_profile import validate_continuous_config
from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    save_checkpoint,
)
from startrain.config import ConfigError, SchedulerConfig, load_config
from startrain.export import ONNX_INPUT_NAMES, ONNXStarModel, export_onnx
from startrain.features import DoubleStarPosition, encode_batch
from startrain.model import GraphResTNet, ModelConfig
from startrain.optim import (
    MuonAdamW,
    OptimizerConfig,
    build_optimizer,
    split_decay_parameters,
)
from startrain.replay import ReplayBatch, ReplaySample, collate_replay_samples
from startrain.sampling import RingStratifiedSampler
from startrain.scoring import PlayerScore, ScoreResult
from startrain.topology import get_topology
from startrain.training import (
    DeviceBatchPrefetcher,
    build_scheduler,
    maybe_compile_model,
    train_step,
    unwrap_model,
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


def test_scheduler_holds_minimum_rate_after_configured_horizon() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(warmup_steps=0, total_steps=10, min_lr_ratio=0.05),
    )
    for _ in range(25):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def sample(rings: int = 4) -> ReplaySample:
    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    position = DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=1,
        moves_left=2,
        opening=False,
        terminal=False,
    )
    legal = stones.numpy() == -1
    policy = legal.astype(np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=ScoreResult(
            players=(
                PlayerScore(10, 3, 1, 1, 0, 11),
                PlayerScore(5, 2, 1, 0, 0, 5),
            ),
            node_owner=torch.zeros(topology.n, dtype=torch.int8),
            alive_stone=torch.zeros(topology.n, dtype=torch.bool),
            contested_peries=0,
            leader=0,
        ),
        search_provenance="mcts:test",
        policy_provenance="root-visits",
    )


def test_yaml_configs_load_strictly() -> None:
    small = load_config(CONFIGS / "small.yaml")
    h100 = load_config(CONFIGS / "h100.yaml")
    assert small.schema_version == h100.schema_version == 3
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
    assert continuous.learner.use_ring_mixture_curriculum is False
    assert continuous.orchestration.model_refresh.selfplay_source == "champion"
    assert continuous.orchestration.model_refresh.candidate_probability == 0.8
    assert continuous.selfplay.record_fast_policy_targets is False
    assert continuous.selfplay.max_considered_cap == 64
    optimized = load_config(CONFIGS / "h100-8gpu-optimized.yaml")
    assert optimized.learner.use_ring_mixture_curriculum is True
    assert len(optimized.orchestration.actor_gpus) == 7
    assert {gpu.actor_batch_size for gpu in optimized.orchestration.actor_gpus} == {128}
    assert optimized.orchestration.actor_games_per_batch == 128
    assert optimized.orchestration.promotion.gpu_id == 7
    assert optimized.orchestration.promotion.pause_sharing_mode is True
    throughput = load_config(CONFIGS / "h100-8gpu-throughput.yaml")
    assert sum(gpu.actor_lanes for gpu in throughput.orchestration.actor_gpus) == 13
    assert throughput.data.shard_cache_size == 8
    assert throughput.orchestration.learner_gpus[0].cpu_affinity == "0-103"
    assert (
        next(
            gpu for gpu in throughput.orchestration.actor_gpus if gpu.gpu_id == 7
        ).actor_lanes
        == 1
    )
    assert throughput.selfplay.fast_policy_weight == 0.25
    assert throughput.selfplay.record_fast_policy_targets is True
    assert throughput.selfplay.considered_actions() == 16
    assert replace(throughput.selfplay, rings=10).considered_actions() == 27
    assert throughput.learner.unlimited is True
    assert throughput.orchestration.ring_mixture.weights_for_step(360_000) == (
        0.15,
        0.15,
        0.15,
        0.55,
    )
    assert throughput.orchestration.ring_mixture.weights_for_step(1_000_000) == (
        0.1,
        0.1,
        0.1,
        0.7,
    )
    assert throughput.orchestration.plateau.consecutive_terminal_rejections == 2
    assert throughput.orchestration.plateau.reset_learning_rate_scale == 0.5
    validate_continuous_config(throughput)
    autonomous = load_config(CONFIGS / "h100-8gpu-autonomous.yaml")
    assert autonomous.orchestration.autonomous.enabled is True
    assert autonomous.orchestration.model_refresh.selfplay_source == (
        "candidate_champion_history_mix"
    )
    assert autonomous.orchestration.model_refresh.history_probability == 0.25
    assert autonomous.orchestration.plateau.action == "reduce_lr_keep_weights"
    assert autonomous.learner.target_updates_per_new_sample == 1.0
    assert autonomous.learner.candidate_interval_examples == 20_000_000
    assert autonomous.data.shards_per_batch == 4
    assert replace(autonomous.selfplay, rings=10).considered_actions() == 53
    validate_continuous_config(autonomous)

    with pytest.raises(ValueError, match="cross-shard"):
        validate_continuous_config(
            replace(
                autonomous,
                data=replace(autonomous.data, shards_per_batch=1),
            )
        )
    with pytest.raises(ValueError, match="update-to-data"):
        validate_continuous_config(
            replace(
                autonomous,
                learner=replace(
                    autonomous.learner,
                    target_updates_per_new_sample=None,
                ),
            )
        )


def test_yaml_parses_opt_in_learner_ring_mixture_curriculum(tmp_path) -> None:
    source = (CONFIGS / "small.yaml").read_text(encoding="utf-8")
    configured = tmp_path / "curriculum.yaml"
    configured.write_text(
        source.replace(
            "learner:\n",
            "learner:\n  use_ring_mixture_curriculum: true\n",
            1,
        ),
        encoding="utf-8",
    )

    experiment = load_config(configured)
    assert experiment.learner.use_ring_mixture_curriculum is True


def test_old_config_schema_and_noncanonical_rings_are_rejected(tmp_path) -> None:
    source = (CONFIGS / "small.yaml").read_text(encoding="utf-8")
    old = tmp_path / "old.yaml"
    old.write_text(source.replace("schema_version: 3", "schema_version: 2", 1))
    with pytest.raises(ConfigError, match="schema_version must be 3"):
        load_config(old)

    odd = tmp_path / "odd.yaml"
    odd.write_text(source.replace("rings: 4", "rings: 5", 1))
    with pytest.raises(ConfigError, match="one of"):
        load_config(odd)


def test_plateau_candidate_cadence_fits_replay_lag() -> None:
    experiment = load_config(CONFIGS / "h100-8gpu.yaml")
    with pytest.raises(ConfigError, match="reset-triggering candidate"):
        replace(
            experiment,
            learner=replace(experiment.learner, candidate_interval=10_000),
        )
    with pytest.raises(ConfigError, match="reset-triggering candidate"):
        replace(
            experiment,
            learner=replace(
                experiment.learner,
                candidate_interval=1,
                candidate_interval_examples=10_000
                * experiment.train.per_rank_batch_size,
            ),
        )
    with pytest.raises(ConfigError, match="replay lag eligibility"):
        replace(
            experiment,
            learner=replace(
                experiment.learner,
                candidate_interval=20_000,
            ),
            orchestration=replace(
                experiment.orchestration,
                plateau=replace(
                    experiment.orchestration.plateau,
                    max_learner_champion_lag_steps=50_000,
                ),
            ),
        )


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


def test_compile_forwards_isolated_recompile_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    calls: list[dict[str, object]] = []

    def compile_model(module, **options):
        assert module is model
        calls.append(options)
        return module

    monkeypatch.setattr(torch, "compile", compile_model)
    compiled = maybe_compile_model(
        model,
        enabled=True,
        dynamic=False,
        fullgraph=True,
        backend="eager",
        recompile_limit=10,
        isolate_recompiles=True,
    )
    assert compiled is model
    assert calls == [
        {
            "dynamic": False,
            "fullgraph": True,
            "backend": "eager",
            "recompile_limit": 10,
            "isolate_recompiles": True,
        }
    ]
    with pytest.raises(ValueError, match="recompile_limit"):
        maybe_compile_model(model, enabled=True, recompile_limit=0)


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
    assert all(
        isinstance(value, torch.Tensor) for value in result.loss_tensors.values()
    )
    host_metrics = result.to_host()
    assert all(np.isfinite(value) for value in host_metrics.losses.values())
    assert np.isfinite(host_metrics.gradient_norm)
    assert result.losses == host_metrics.losses
    assert ema.num_updates == 1

    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=9,
        epoch=2,
        config={"schema_version": 3},
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

    old_path = tmp_path / "old-checkpoint.pt"
    payload = torch.load(path, weights_only=True)
    payload["version"] = 2
    torch.save(payload, old_path)
    with pytest.raises(ValueError, match="checkpoint version"):
        load_checkpoint(old_path, model=tiny_model())


def test_public_train_step_validates_untrusted_target_weights() -> None:
    model = tiny_model()
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    batch = collate_replay_samples([sample()])
    assert batch.targets.policy_weight is not None
    batch.targets.policy_weight[0] = -1
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(ValueError, match="policy weights"):
        train_step(model, batch, optimizer)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_train_step_fails_all_ranks_before_optimizer_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    batch = collate_replay_samples([sample()])
    reductions: list[torch.Tensor] = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def reject_one_rank(tensor: torch.Tensor, **_kwargs) -> None:
        reductions.append(tensor.clone())
        tensor[0] = 0

    monkeypatch.setattr(torch.distributed, "all_reduce", reject_one_rank)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(FloatingPointError, match="at least one rank"):
        train_step(model, batch, optimizer)

    assert len(reductions) == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_device_prefetcher_cpu_path_preserves_batches() -> None:
    batches = [collate_replay_samples([sample()]) for _ in range(3)]

    observed = list(DeviceBatchPrefetcher(batches, device="cpu"))

    assert [batch.feature_path for batch in observed] == [
        batch.feature_path for batch in batches
    ]
    for expected, actual in zip(batches, observed, strict=True):
        torch.testing.assert_close(
            expected.inputs.node_features, actual.inputs.node_features
        )
        torch.testing.assert_close(expected.targets.policy, actual.targets.policy)


def test_prefetch_transfer_reuses_homogeneous_topology_and_handles_mixed_rings() -> (
    None
):
    prefetcher = object.__new__(DeviceBatchPrefetcher)
    prefetcher.device = torch.device("cpu")
    prefetcher._topology_cache = {}

    homogeneous = collate_replay_samples([sample(4), sample(4)])
    first = prefetcher._to_device(homogeneous)
    second = prefetcher._to_device(homogeneous)

    assert len(prefetcher._topology_cache) == 1
    assert (
        first.inputs.neighbor_index.data_ptr()
        == second.inputs.neighbor_index.data_ptr()
    )
    assert (
        first.inputs.neighbor_mask.data_ptr() == second.inputs.neighbor_mask.data_ptr()
    )
    assert (
        first.inputs.neighbor_edge_type.data_ptr()
        == second.inputs.neighbor_edge_type.data_ptr()
    )
    assert first.inputs.node_mask.data_ptr() == second.inputs.node_mask.data_ptr()
    torch.testing.assert_close(first.targets.policy, homogeneous.targets.policy)

    mixed = collate_replay_samples([sample(4), sample(6)])
    transferred = prefetcher._to_device(mixed)
    assert transferred.inputs.rings.tolist() == [4, 6]
    assert transferred.inputs.max_nodes == get_topology(6).n
    assert len(prefetcher._topology_cache) == 1


def test_async_prefetch_hands_off_events_and_exhausts_cleanly(monkeypatch) -> None:
    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing
            self.recorded_on: list[object] = []

        def record(self, stream: object) -> None:
            self.recorded_on.append(stream)

        def elapsed_time(self, _completed: object) -> float:
            return 2.5

    class FakeCurrentStream:
        def __init__(self) -> None:
            self.waited_for: list[object] = []

        def wait_stream(self, stream: object) -> None:
            self.waited_for.append(stream)

    fake_copy_stream = object()
    current_stream = FakeCurrentStream()
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext(stream))
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: current_stream,
    )
    recorded_batches: list[ReplayBatch] = []
    monkeypatch.setattr(
        ReplayBatch,
        "record_stream",
        lambda batch, _stream: recorded_batches.append(batch),
    )
    monkeypatch.setattr(
        DeviceBatchPrefetcher,
        "_to_device",
        lambda _prefetcher, source: source,
    )

    sources = [
        collate_replay_samples([sample()]),
        collate_replay_samples([sample()]),
    ]
    prefetcher = object.__new__(DeviceBatchPrefetcher)
    prefetcher._batches = iter(sources)
    prefetcher.device = torch.device("cpu")
    prefetcher._stream = fake_copy_stream
    prefetcher._consumed_copy_events = []
    prefetcher._next_copy_event = None
    prefetcher._topology_cache = {}
    prefetcher._next_batch = None
    prefetcher._next_source = None
    prefetcher._preload()

    assert next(prefetcher) is sources[0]
    assert next(prefetcher) is sources[1]
    with pytest.raises(StopIteration):
        next(prefetcher)
    assert recorded_batches == sources
    assert current_stream.waited_for == [fake_copy_stream, fake_copy_stream]
    assert prefetcher.pop_copy_seconds() == pytest.approx(0.005)
    assert prefetcher.pop_copy_events() == []


def test_training_rejects_invalid_step_controls_and_unwraps_nested_models() -> None:
    model = tiny_model()
    wrapper = torch.nn.Module()
    wrapper.module = model
    outer = torch.nn.Module()
    outer._orig_mod = wrapper
    assert unwrap_model(outer) is model

    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    batch = collate_replay_samples([sample()])
    with pytest.raises(ValueError, match="precision"):
        train_step(model, batch, optimizer, precision="fp16")
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        train_step(model, batch, optimizer, gradient_clip_norm=0)


@pytest.mark.cuda
def test_cuda_prefetcher_reuses_ring_topology_and_tracks_copy_time() -> None:
    sources = [
        collate_replay_samples([sample(), sample()]).pin_memory() for _ in range(3)
    ]
    assert sources[0].inputs.node_features.is_pinned()
    assert sources[0].inputs.legal_action_mask.is_pinned()
    assert not sources[0].inputs.neighbor_index.is_pinned()
    assert not sources[0].inputs.node_mask.is_pinned()
    prefetcher = DeviceBatchPrefetcher(sources, device="cuda")

    first = next(prefetcher)
    second = next(prefetcher)
    torch.cuda.synchronize()
    consumed_events = prefetcher.pop_copy_events()

    assert first.inputs.node_features.is_cuda
    assert second.targets.policy.is_cuda
    assert (
        first.inputs.neighbor_index.data_ptr()
        == second.inputs.neighbor_index.data_ptr()
    )
    assert first.inputs.node_mask.data_ptr() == second.inputs.node_mask.data_ptr()
    assert len(consumed_events) == 2
    assert sum(start.elapsed_time(end) for start, end in consumed_events) / 1_000 >= 0
    next(prefetcher)
    torch.cuda.synchronize()
    assert len(prefetcher.pop_copy_events()) == 1


def test_ema_foreach_update_matches_exact_lerp() -> None:
    model = tiny_model()
    ema = ExponentialMovingAverage(model, decay=0.75)
    before = {name: value.clone() for name, value in ema.shadow.items()}
    with torch.no_grad():
        for value in model.parameters():
            value.add_(0.125)

    ema.update(model)

    state = model.state_dict()
    assert ema.num_updates == 1
    for name, average in ema.shadow.items():
        expected = before[name].lerp(
            state[name].detach().to(dtype=average.dtype),
            0.25,
        )
        torch.testing.assert_close(average, expected)


def test_ring_stratified_sampler_remains_balanced() -> None:
    rings = [4] * 8 + [10] * 2
    sampler = RingStratifiedSampler(rings, num_samples=20, seed=11)
    sampled = [rings[index] for index in sampler]
    assert sampled.count(4) == sampled.count(10) == 10


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
