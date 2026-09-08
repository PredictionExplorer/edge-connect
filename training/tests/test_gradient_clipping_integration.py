from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import pytest
import torch

from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    save_checkpoint,
)
from startrain.config import SchedulerConfig
from startrain.gradient_clipping import GradientClipper, GradientClippingConfig
from startrain.model import GraphResTNet, ModelConfig
from startrain.optim import MuonAdamW, build_optimizer
from startrain.replay import ReplayBatch, collate_replay_samples
from startrain.training import (
    DeviceBatchPrefetcher,
    NonFiniteTrainingError,
    build_scheduler,
    train_step,
)
from test_training import sample


@dataclass
class Runtime:
    model: GraphResTNet
    optimizer: MuonAdamW
    scheduler: object
    ema: ExponentialMovingAverage
    clipper: GradientClipper | None


def runtime(*, adaptive=True, seed=71):
    torch.manual_seed(seed)
    model = GraphResTNet(
        ModelConfig(
            width=16, rrt_groups=1, attention_heads=4, kv_heads=1, adaln_hidden=4
        )
    )
    optimizer = build_optimizer(model)
    assert isinstance(optimizer, MuonAdamW)
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(warmup_steps=2, total_steps=30, min_lr_ratio=0.1)
    )
    clipper = (
        GradientClipper(
            model.named_parameters(),
            config=GradientClippingConfig(
                mode="adagc", warmup_steps=3, beta=0.9, multiplier=1.04
            ),
        )
        if adaptive
        else None
    )
    return Runtime(
        model, optimizer, scheduler, ExponentialMovingAverage(model, decay=0.9), clipper
    )


def run_step(state, index=0, *, diagnostics=False):
    ring = 4 if index % 2 == 0 else 6
    batch = collate_replay_samples([sample(ring), sample(ring)], prefer_native=False)
    return train_step(
        state.model,
        batch,
        state.optimizer,
        scheduler=state.scheduler,
        ema=state.ema,
        gradient_clipper=state.clipper,
        collect_diagnostics=diagnostics,
        collect_gradient_diagnostics=diagnostics,
    )


def state_dict(state):
    return deepcopy(
        {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict(),
            "ema": state.ema.state_dict(),
            "clipping": state.clipper.state_dict() if state.clipper else None,
        }
    )


def assert_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_tree(first, second)
    else:
        assert left == right


def save(state, path, step):
    config = {
        "model": asdict(state.model.config),
        "train": {
            "gradient_clipping": asdict(state.clipper.config)
            if state.clipper
            else {"mode": "global"}
        },
    }
    return save_checkpoint(
        path,
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        ema=state.ema,
        gradient_clipper=state.clipper,
        step=step,
        config=config,
    )


def load(state, path, **kwargs):
    return load_checkpoint(
        path,
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        ema=state.ema,
        gradient_clipper=state.clipper,
        **kwargs,
    )


def test_global_diagnostics_do_not_change_model_muon_moments_scheduler_or_ema():
    plain, observed = runtime(adaptive=False), runtime(adaptive=False)
    for index in range(3):
        expected = run_step(plain, index).to_host()
        measured = run_step(observed, index, diagnostics=True).to_host()
        assert measured.losses["total"] == expected.losses["total"]
        assert measured.gradient_norm == expected.gradient_norm
        assert measured.gradient_diagnostics["gradient_stage"] == "before_clipping"
        assert measured.gradient_diagnostics["batch"]["six_mode_counts_exact"] is True
        assert_tree(state_dict(plain), state_dict(observed))


@pytest.mark.parametrize("checkpoint_after", (1, 4))
def test_adagc_checkpoint_restores_exact_muon_and_adamw_continuation(
    tmp_path, checkpoint_after
):
    uninterrupted = runtime()
    for index in range(checkpoint_after):
        run_step(uninterrupted, index)
    checkpoint = save(uninterrupted, tmp_path / "resume.pt", checkpoint_after)
    resumed = runtime(seed=913)
    metadata = load(resumed, checkpoint)
    assert metadata["step"] == checkpoint_after
    assert_tree(state_dict(uninterrupted), state_dict(resumed))
    for index in range(checkpoint_after, checkpoint_after + 3):
        expected = run_step(uninterrupted, index).to_host()
        actual = run_step(resumed, index, diagnostics=True).to_host()
        assert expected.gradient_clipping == actual.gradient_clipping
        assert actual.gradient_clipping["warmup"] == (index < 3)
        assert_tree(state_dict(uninterrupted), state_dict(resumed))


@pytest.mark.parametrize("corruption", ("missing", "negative_history", "wrong_mode"))
def test_corrupt_adaptive_state_is_rejected_before_any_live_mutation(
    tmp_path, corruption
):
    source = runtime()
    run_step(source)
    checkpoint = save(source, tmp_path / "corrupt.pt", 1)
    payload = torch.load(checkpoint, weights_only=True)
    if corruption == "missing":
        payload["gradient_clipping"] = None
    elif corruption == "negative_history":
        name = next(iter(payload["gradient_clipping"]["ema_norms"]))
        payload["gradient_clipping"]["ema_norms"][name] = torch.tensor(-1.0)
    else:
        payload["gradient_clipping"]["config"]["mode"] = "global"
    torch.save(payload, checkpoint)
    destination = runtime(seed=913)
    run_step(destination)
    before = state_dict(destination)
    with pytest.raises(ValueError, match="gradient|adaptive"):
        load(destination, checkpoint, allow_gradient_clipping_cold_start=True)
    assert_tree(before, state_dict(destination))


def test_missing_legacy_state_requires_explicit_cold_start_and_resets_used_history(
    tmp_path,
):
    legacy = runtime(adaptive=False)
    run_step(legacy)
    checkpoint = save(legacy, tmp_path / "legacy.pt", 1)
    # Reproduce a genuine old checkpoint, which predates the optional field.
    payload = torch.load(checkpoint, weights_only=True)
    del payload["gradient_clipping"]
    torch.save(payload, checkpoint)
    destination = runtime(seed=913)
    for index in range(4):
        run_step(destination, index)
    before = state_dict(destination)
    with pytest.raises(ValueError, match="explicit opt-in"):
        load(destination, checkpoint)
    assert_tree(before, state_dict(destination))
    load(destination, checkpoint, allow_gradient_clipping_cold_start=True)
    assert destination.clipper.steps == 0
    history = destination.clipper.state_dict()["ema_norms"]
    assert set(history) == dict(destination.model.named_parameters()).keys()
    assert all(value.item() == 0 for value in history.values())
    for name in ("model", "optimizer", "scheduler", "ema"):
        assert_tree(state_dict(legacy)[name], state_dict(destination)[name])
    assert (
        run_step(destination, diagnostics=True).to_host().gradient_clipping["warmup"]
        is True
    )


def test_adaptive_optimizer_resume_cannot_drop_its_clipping_controller(tmp_path):
    source = runtime()
    run_step(source)
    checkpoint = save(source, tmp_path / "adaptive.pt", 1)
    destination = runtime(adaptive=False, seed=913)
    before = state_dict(destination)
    with pytest.raises(ValueError, match="requires its gradient clipper"):
        load(destination, checkpoint)
    assert_tree(before, state_dict(destination))


@pytest.mark.parametrize("after_steps", (1, 4))
def test_nonfinite_gradient_aborts_without_advancing_any_adaptive_state(after_steps):
    state = runtime()
    for index in range(after_steps):
        run_step(state, index)
    before = state_dict(state)
    parameter = next(state.model.parameters())
    hook = parameter.register_hook(lambda gradient: gradient * float("nan"))
    try:
        with pytest.raises(NonFiniteTrainingError):
            run_step(state, diagnostics=True)
    finally:
        hook.remove()
    assert_tree(before, state_dict(state))


def test_prefetch_topology_fast_path_preserves_original_variant_labels():
    # Exercise the same homogeneous-topology reconstruction used by CUDA,
    # keeping tensor transfers on CPU so this remains runnable everywhere.
    prefetcher = object.__new__(DeviceBatchPrefetcher)
    prefetcher.device = torch.device("cpu")
    prefetcher._topology_cache = {}
    standard = sample(4)
    source = collate_replay_samples(
        [standard, replace(standard, pie=True)], prefer_native=False
    )
    first, second = prefetcher._to_device(source), prefetcher._to_device(source)
    assert first.variant_labels == second.variant_labels == ("double", "pie-double")
    assert (
        first.inputs.neighbor_index.data_ptr()
        == second.inputs.neighbor_index.data_ptr()
    )
    legacy = ReplayBatch(source.inputs, source.targets)
    assert prefetcher._to_device(legacy).variant_labels is None


@pytest.mark.cuda
def test_cuda_prefetch_preserves_variant_metadata_and_cuda_placement():
    standard = sample(4)
    source = collate_replay_samples(
        [standard, replace(standard, pie=True)], prefer_native=False
    )
    prefetcher = DeviceBatchPrefetcher([source], device="cuda:0")
    try:
        observed = next(prefetcher)
        assert observed.variant_labels == ("double", "pie-double")
        assert observed.inputs.node_features.is_cuda
    finally:
        prefetcher.close()
