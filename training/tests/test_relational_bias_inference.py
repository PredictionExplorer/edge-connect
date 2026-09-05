from copy import deepcopy

import pytest
import torch

from startrain.features import encode_batch
from startrain.model import GraphResTNet, ModelConfig, model_parameter_count
from startrain.topology import SUPPORTED_RINGS
from test_inference_efficiency import position


def model() -> GraphResTNet:
    torch.manual_seed(51)
    result = GraphResTNet(
        ModelConfig(width=16, rrt_groups=2, attention_heads=4, kv_heads=1)
    )
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            if "relation_bias" in name or "modulation" in name:
                parameter.normal_(std=0.25)
    return result


@pytest.mark.parametrize("ring", SUPPORTED_RINGS)
def test_shared_relational_bias_preserves_forward_and_all_parameter_gradients(ring):
    original = model().double()
    shared = deepcopy(original)
    batch = encode_batch([position(ring), position(ring, pda=2)]).to(
        "cpu", feature_dtype=torch.float64
    )
    expected = original(*batch.model_args())
    actual = shared(*batch.model_args(), homogeneous_ring=ring)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)
    torch.manual_seed(59)
    upstream = [torch.randn_like(tensor) for tensor in expected]
    sum(
        (value * gradient).sum()
        for value, gradient in zip(expected, upstream, strict=True)
    ).backward()
    sum(
        (value * gradient).sum()
        for value, gradient in zip(actual, upstream, strict=True)
    ).backward()
    original_parameters = dict(original.named_parameters())
    for name, parameter in shared.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            original_parameters[name].grad,
            atol=1e-9,
            rtol=1e-9,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_cached_bias_reuses_storage_invalidates_after_weight_load_and_keeps_state_tree():
    network = model().eval()
    state_keys = tuple(network.state_dict())
    batch = encode_batch([position(4), position(4, pda=2)])
    cached = network.prepare_inference_relational_bias(4, dtype=torch.float32)
    assert cached is not None
    again = network.prepare_inference_relational_bias(4, dtype=torch.float32)
    assert again is cached
    with torch.inference_mode():
        expected = network(*batch.model_args())
        actual = network(
            *batch.model_args(), homogeneous_ring=4, inference_relation_bias=cached
        )
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    with torch.no_grad():
        network.rrt_groups[0].global_block.relation_bias.weight.add_(0.2)
    refreshed = network.prepare_inference_relational_bias(4, dtype=torch.float32)
    assert refreshed is not None and refreshed is not cached
    assert not torch.equal(refreshed[0], cached[0])
    assert tuple(network.state_dict()) == state_keys
    checkpoint = {name: tensor.clone() for name, tensor in network.state_dict().items()}
    network.load_state_dict(checkpoint, strict=True)
    assert (
        network.prepare_inference_relational_bias(4, dtype=torch.float32)
        is not refreshed
    )
    with pytest.raises(ValueError, match="inference without gradients"):
        network(*batch.model_args(), homogeneous_ring=4, inference_relation_bias=cached)
    network.clear_inference_caches()
    assert network._inference_relation_bias_cache == {}


def test_compiled_homogeneous_inference_and_mixed_ring_fallback_match():
    network = model().eval()
    compiled = torch.compile(network, backend="eager", dynamic=True, fullgraph=True)
    with torch.inference_mode():
        for ring in (4, 10):
            batch = encode_batch([position(ring)] * 2)
            biases = network.prepare_inference_relational_bias(
                ring, dtype=torch.float32
            )
            actual = compiled(
                *batch.model_args(),
                homogeneous_ring=ring,
                inference_relation_bias=biases,
            )
            expected = network(*batch.model_args())
            for left, right in zip(expected, actual, strict=True):
                torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
        mixed = encode_batch([position(4), position(10)])
        for left, right in zip(
            network(*mixed.model_args()), compiled(*mixed.model_args()), strict=True
        ):
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    assert (
        model_parameter_count(
            ModelConfig(
                width=384,
                rrt_groups=8,
                attention_heads=12,
                kv_heads=3,
                ff_multiplier=2.5,
            )
        )
        == 17_402_775
    )
