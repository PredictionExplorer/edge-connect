"""Shared training geometry is CPU-certified and preserves trainable bias."""

from copy import deepcopy
from dataclasses import replace
import pickle
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from startrain.losses import compute_losses
from startrain.model import GraphResTNet, ModelConfig
from startrain.replay import (
    ReplayBatch,
    _validate_homogeneous_geometry,
    collate_replay_samples,
)
from startrain.training import DeviceBatchPrefetcher, train_step
from test_training import sample


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
@pytest.mark.parametrize("native", [False, True])
def test_collation_certifies_canonical_cpu_geometry(ring, native):
    batch = collate_replay_samples([sample(ring)] * 3, prefer_native=native)
    assert batch.homogeneous_ring == ring
    assert batch.to("cpu", feature_dtype=torch.bfloat16).homogeneous_ring == ring
    # A device without readable tensor values demonstrates metadata propagation
    # does not require .item(), bool(tensor), or a device-to-host transfer.
    assert batch.to("meta").homogeneous_ring == ring
    restored = pickle.loads(pickle.dumps(batch))
    assert restored.homogeneous_ring == ring
    assert restored.inputs.node_features.shape == batch.inputs.node_features.shape


def test_worker_ipc_preserves_validated_geometry():
    loader = DataLoader(
        [sample(6)] * 4,
        batch_size=2,
        collate_fn=collate_replay_samples,
        num_workers=1,
        multiprocessing_context="spawn",
    )
    batches = list(loader)
    assert [batch.homogeneous_ring for batch in batches] == [6, 6]


@pytest.mark.parametrize(
    "field",
    ["rings", "node_mask", "neighbor_mask", "neighbor_index", "neighbor_edge_type"],
)
def test_static_replacement_or_mutation_invalidates_proof(field):
    batch = collate_replay_samples([sample(4)] * 2, prefer_native=False)
    tensor = getattr(batch.inputs, field)
    replaced = replace(batch, inputs=replace(batch.inputs, **{field: tensor.clone()}))
    assert replaced.homogeneous_ring is None
    # Even a value-preserving in-place operation invalidates the version guard.
    tensor.copy_(tensor.clone())
    assert batch.homogeneous_ring is None
    assert batch.to("meta").homogeneous_ring is None


def test_uncertified_or_noncanonical_batches_keep_general_path():
    batch = collate_replay_samples([sample(4)] * 2, prefer_native=False)
    assert ReplayBatch(batch.inputs, batch.targets).homogeneous_ring is None
    assert collate_replay_samples([sample(4), sample(6)]).homogeneous_ring is None
    changed_mask = batch.inputs.node_mask.clone()
    changed_mask[1, -1] = False
    changed = replace(batch.inputs, node_mask=changed_mask)
    assert _validate_homogeneous_geometry(changed) is None
    wrong_edges = batch.inputs.neighbor_edge_type.clone()
    wrong_edges[:, 0, 0] += 1
    assert (
        _validate_homogeneous_geometry(
            replace(batch.inputs, neighbor_edge_type=wrong_edges)
        )
        is None
    )
    with torch.inference_mode():
        inference_inputs = replace(batch.inputs, rings=batch.inputs.rings.clone())
    assert _validate_homogeneous_geometry(inference_inputs) is None


def test_prefetch_cache_uses_only_certified_geometry_and_preserves_proof():
    prefetcher = object.__new__(DeviceBatchPrefetcher)
    prefetcher.device = torch.device("cpu")
    prefetcher._topology_cache = {}
    batch = collate_replay_samples([sample(6)] * 2, prefer_native=False)
    first = prefetcher._to_device(batch)
    second = prefetcher._to_device(batch)
    assert first.homogeneous_ring == second.homogeneous_ring == 6
    assert first.inputs.node_mask is second.inputs.node_mask
    batch.inputs.node_mask[1, -1] = False
    transferred = prefetcher._to_device(batch)
    assert transferred.homogeneous_ring is None
    assert not transferred.inputs.node_mask[1, -1]
    assert len(prefetcher._topology_cache) == 1


def test_mutated_cached_topology_is_replaced_from_validated_source():
    prefetcher = object.__new__(DeviceBatchPrefetcher)
    prefetcher.device = torch.device("cpu")
    prefetcher._topology_cache = {}
    batch = collate_replay_samples([sample(6)] * 2, prefer_native=False)
    first = prefetcher._to_device(batch)
    first.inputs.node_mask[1, -1] = False
    assert first.homogeneous_ring is None
    assert batch.homogeneous_ring == 6
    second = prefetcher._to_device(batch)
    assert second.homogeneous_ring == 6
    assert second.inputs.node_mask[1, -1]
    assert second.inputs.node_mask is not first.inputs.node_mask


def test_pinning_rebinds_geometry_without_revalidation(monkeypatch):
    batch = collate_replay_samples([sample(4)] * 2)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor.clone())
    with patch("startrain.replay._validate_homogeneous_geometry") as validate:
        pinned = batch.pin_memory()
    validate.assert_not_called()
    assert pinned.homogeneous_ring == 4
    assert pinned.inputs.node_features is not batch.inputs.node_features


def _model():
    torch.manual_seed(891)
    model = GraphResTNet(
        ModelConfig(width=16, rrt_groups=2, attention_heads=4, kv_heads=1)
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "relation_bias" in name or "modulation" in name:
                parameter.normal_(0, 0.2)
            elif "layer_scale" in name or "ff_scale" in name:
                parameter.fill_(0.2)
    return model


def _forward_and_gradients(model, batch, *, shared, bf16=False):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        outputs = model(
            *batch.inputs.model_args(),
            **({"homogeneous_ring": batch.homogeneous_ring} if shared else {}),
        )
        losses = compute_losses(
            outputs,
            batch.targets,
            legal_action_mask=batch.inputs.legal_action_mask,
            node_mask=batch.inputs.node_mask,
        )
    losses["total"].backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return tuple(output.detach() for output in outputs), gradients


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
def test_fp32_shared_outputs_and_all_parameter_gradients_match(ring):
    batch = collate_replay_samples([sample(ring)] * 3)
    # Different examples matter: a broadcast bias must sum different row VJPs.
    batch.inputs.global_features[1].mul_(0.8)
    batch.inputs.node_features[2].mul_(0.7)
    model = _model()
    expected, reference_grads = _forward_and_gradients(model, batch, shared=False)
    actual, shared_grads = _forward_and_gradients(model, batch, shared=True)
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    assert shared_grads.keys() == reference_grads.keys()
    for name, reference in reference_grads.items():
        candidate = shared_grads[name]
        # Reassociating the FP32 batch reduction can change final mantissa bits.
        torch.testing.assert_close(candidate, reference, rtol=2e-5, atol=2e-8)
        if "relation_bias" in name:
            assert reference.norm() > 1e-5
            assert (candidate - reference).norm() / reference.norm() < 2e-5


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
def test_bf16_bias_gradients_stay_close_to_fp32_oracle(ring):
    batch = collate_replay_samples([sample(ring)] * 3)
    batch.inputs.global_features[1].mul_(0.8)
    batch.inputs.node_features[2].mul_(0.7)
    model = _model()
    _, fp32 = _forward_and_gradients(model, batch, shared=False)
    expected, bf16 = _forward_and_gradients(model, batch, shared=False, bf16=True)
    actual, shared = _forward_and_gradients(model, batch, shared=True, bf16=True)
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    for name, reference in fp32.items():
        assert torch.isfinite(shared[name]).all()
        if "relation_bias" not in name:
            continue
        # The shared VJP rounds once after summing FP32 rows; the old VJP
        # rounded each BF16 row first. Both must track the same FP32 oracle.
        assert (shared[name] - reference).norm() / reference.norm() < 0.05
        assert (bf16[name] - reference).norm() / reference.norm() < 0.05
        assert (shared[name] - bf16[name]).norm() / reference.norm() < 0.05


@pytest.mark.parametrize("shared", [False, True])
def test_train_step_only_forwards_certified_opt_in_to_supported_models(shared):
    batch = collate_replay_samples([sample(4)] * 2)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    with patch.object(model, "forward", wraps=model.forward) as forward:
        train_step(model, batch, optimizer, share_homogeneous_geometry=shared)
    assert forward.call_args.kwargs == ({"homogeneous_ring": 4} if shared else {})
    with patch.object(model, "forward", wraps=model.forward) as forward:
        train_step(
            model,
            ReplayBatch(batch.inputs, batch.targets),
            optimizer,
            share_homogeneous_geometry=shared,
        )
    assert forward.call_args.kwargs == {}

    class GenericModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args):
            return self.inner(*args)

    generic = GenericModel(deepcopy(model))
    train_step(
        generic,
        batch,
        torch.optim.SGD(generic.parameters(), lr=0.0),
        share_homogeneous_geometry=shared,
    )


def test_compiled_wrapper_receives_shared_ring_and_preserves_ring6_gradients():
    batch = collate_replay_samples([sample(6)] * 2)
    model = _model()
    expected = deepcopy(model)
    train_step(
        expected,
        batch,
        torch.optim.SGD(expected.parameters(), lr=0.0),
        precision="bf16",
        share_homogeneous_geometry=True,
    )
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    train_step(
        compiled,
        batch,
        torch.optim.SGD(model.parameters(), lr=0.0),
        precision="bf16",
        share_homogeneous_geometry=True,
    )
    for candidate, reference in zip(
        model.parameters(), expected.parameters(), strict=True
    ):
        if reference.grad is not None:
            torch.testing.assert_close(candidate.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.cuda
def test_cuda_pinned_prefetch_and_compiled_shared_ring6():
    batch = collate_replay_samples([sample(6)] * 2).pin_memory()
    assert batch.homogeneous_ring == 6
    staged = list(DeviceBatchPrefetcher([batch, batch], device="cuda"))
    assert [item.homogeneous_ring for item in staged] == [6, 6]
    model = _model().cuda()
    expected = deepcopy(model)
    train_step(
        expected,
        staged[0],
        torch.optim.SGD(expected.parameters(), lr=0.0),
        precision="bf16",
        share_homogeneous_geometry=True,
    )
    compiled = torch.compile(model, fullgraph=True)
    train_step(
        compiled,
        staged[1],
        torch.optim.SGD(model.parameters(), lr=0.0),
        precision="bf16",
        share_homogeneous_geometry=True,
    )
    for name, candidate in model.named_parameters():
        reference = dict(expected.named_parameters())[name]
        if reference.grad is not None:
            assert candidate.grad is not None and torch.isfinite(candidate.grad).all()
            if "relation_bias" in name:
                assert (
                    candidate.grad - reference.grad
                ).norm() / reference.grad.norm() < 0.03
