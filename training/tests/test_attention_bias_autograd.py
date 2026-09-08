from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from startrain.attention_bias_autograd import (
    _relation_bias_vjp,
    relation_bias_gradient_carrier,
)
from startrain.model import _explicit_attention_in_fp32, model_parameter_count


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
def test_opaque_bias_vjp_matches_explicit_attention(groups, dtype, shared):
    torch.manual_seed(851)
    batch, heads, length, width = 2, 4, 11, 8
    query = torch.randn(batch, heads, length, width, dtype=dtype, requires_grad=True)
    key = torch.randn(
        batch, heads // groups, length, width, dtype=dtype, requires_grad=True
    )
    value = torch.randn(
        batch, heads // groups, length, 5, dtype=dtype, requires_grad=True
    )
    # Noncontiguous additive mask, including broadcast over the batch dimension.
    mask = torch.randn(
        1 if shared else batch, length, length, heads, dtype=dtype
    ).permute(0, 3, 1, 2)
    mask[..., -2:] = -torch.inf
    mask.requires_grad_()
    upstream = torch.randn(batch, heads, length, 5, dtype=dtype)
    expected_output = _explicit_attention_in_fp32(
        query.detach(), key.detach(), value.detach(), mask, groups
    )
    expected = torch.autograd.grad((expected_output * upstream).float().sum(), mask)[0]
    carrier = relation_bias_gradient_carrier(query, key, value, mask, groups)
    assert torch.count_nonzero(carrier) == 0
    (carrier * upstream).float().sum().backward()
    assert query.grad is None and key.grad is None and value.grad is None
    assert mask.grad is not None
    if dtype == torch.float32:
        torch.testing.assert_close(mask.grad, expected, rtol=1e-5, atol=1e-6)
    else:
        # Different FP32 softmax VJP algebra may round one BF16 ULP differently.
        assert (
            mask.grad.float() - expected.float()
        ).norm() / expected.float().norm() < 0.005
    assert torch.count_nonzero(mask.grad[..., -2:]) == 0


def test_key_only_broadcast_mask_reduces_all_expanded_dimensions():
    torch.manual_seed(859)
    query = torch.randn(3, 4, 9, 8)
    key = torch.randn(3, 2, 9, 8)
    value = torch.randn(3, 2, 9, 8)
    mask = torch.randn(1, 1, 1, 9, requires_grad=True)
    upstream = torch.randn_like(query)
    output = _explicit_attention_in_fp32(query, key, value, mask, 2)
    expected = torch.autograd.grad((output * upstream).sum(), mask)[0]
    actual = _relation_bias_vjp(query, key, value, mask.detach(), upstream, 2)
    assert actual.shape == mask.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_custom_operators_pass_schema_fake_autograd_and_aot_checks():
    query = torch.randn(2, 4, 7, 8)
    key = torch.randn(2, 2, 7, 8)
    value = torch.randn(2, 2, 7, 8)
    mask = torch.randn(1, 4, 7, 7, requires_grad=True)
    for operator, arguments in (
        (relation_bias_gradient_carrier, (query, key, value, mask, 2)),
        (
            _relation_bias_vjp,
            (query, key, value, mask.detach(), torch.randn_like(query), 2),
        ),
    ):
        assert set(torch.library.opcheck(operator, arguments).values()) == {"SUCCESS"}


def test_compiled_carrier_preserves_zero_forward_and_bias_gradient():
    torch.manual_seed(863)
    query = torch.randn(2, 8, 106, 16)
    key = torch.randn(2, 2, 106, 16)
    value = torch.randn(2, 2, 106, 16)
    mask = torch.randn(1, 8, 106, 106, requires_grad=True)
    upstream = torch.randn_like(query)

    def objective(mask):
        return (
            relation_bias_gradient_carrier(query, key, value, mask, 4) * upstream
        ).sum()

    expected = torch.autograd.grad(objective(mask), mask)[0]
    compiled = torch.compile(objective, backend="aot_eager", fullgraph=True)
    actual = torch.autograd.grad(compiled(mask), mask)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.sum(dim=-1).norm() / actual.norm() < 1e-6


def test_bias_fix_retains_approved_parameter_count():
    from startrain.config import load_config

    profile = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    assert model_parameter_count(profile.model) == 17_402_775


def test_model_forward_is_unchanged_from_explicit_zero_carrier():
    from startrain.features import encode_batch
    from startrain.model import GraphResTNet, ModelConfig
    from test_inference_efficiency import position

    torch.manual_seed(869)
    model = GraphResTNet(
        ModelConfig(width=16, rrt_groups=2, attention_heads=4, kv_heads=1)
    )
    original = deepcopy(model)
    batch = encode_batch([position(4), position(6)])

    def original_carrier(query, key, value, mask, *, groups):
        explicit = _explicit_attention_in_fp32(query, key, value, mask, groups)
        return explicit - explicit.detach()

    with patch("startrain.model._relation_bias_gradient_carrier", original_carrier):
        expected = original(*batch.model_args())
    actual = model(*batch.model_args())
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    model.eval()
    original.eval()
    with torch.no_grad():
        actual = model(*batch.model_args())
        expected = original(*batch.model_args())
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)


@pytest.mark.cuda
def test_cuda_inductor_keeps_bias_vjp_opaque():
    if not torch.cuda.is_available():
        pytest.skip("requires isolated CUDA")
    torch.manual_seed(877)
    query = torch.randn(2, 8, 106, 48, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 2, 106, 48, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.randn(
        2, 8, 106, 106, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    upstream = torch.randn_like(query)

    def objective(mask):
        return (
            (relation_bias_gradient_carrier(query, key, value, mask, 4) * upstream)
            .float()
            .sum()
        )

    expected = torch.autograd.grad(objective(mask), mask)[0]
    compiled = torch.compile(objective, fullgraph=True)
    actual = torch.autograd.grad(compiled(mask), mask)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
