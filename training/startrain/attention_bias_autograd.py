"""Opaque first-order relation-bias gradients for compiled fused attention.

The model obtains attention values and Q/K/V gradients from fused SDPA. These
custom operators contribute a zero forward value and the additive-mask VJP
only. Keeping the VJP opaque prevents Inductor from rewriting the explicit
softmax backward, which produced incorrect relation-table gradients in the
production BF16 six-ring workload. No attention matrix is saved for backward.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


@torch.library.custom_op("startrain::relation_bias_vjp", mutates_args=())
def _relation_bias_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor,
    grad_output: Tensor,
    groups: int,
) -> Tensor:
    """Compute only d attention / d additive mask, outside compiler lowering."""

    if groups < 1 or query.shape[1] != key.shape[1] * groups:
        raise ValueError("relation-bias VJP has inconsistent GQA groups")
    with torch.autocast(device_type=query.device.type, enabled=False):
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        scores = (
            torch.matmul(query.float(), key.float().transpose(-2, -1))
            * query.shape[-1] ** -0.5
        )
        probabilities = torch.softmax(scores + attn_mask.float(), dim=-1)
        probability_gradient = torch.matmul(
            grad_output.float(), value.float().transpose(-2, -1)
        )
        centered = probability_gradient - (probability_gradient * probabilities).sum(
            dim=-1, keepdim=True
        )
        score_gradient = probabilities * centered
        result = score_gradient.sum_to_size(attn_mask.shape).to(attn_mask.dtype)
    # The score gradient owns new storage; no input is returned or mutated.
    return result.contiguous()


@_relation_bias_vjp.register_fake
def _relation_bias_vjp_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor,
    grad_output: Tensor,
    groups: int,
) -> Tensor:
    return torch.empty_like(attn_mask, memory_format=torch.contiguous_format)


@torch.library.custom_op("startrain::relation_bias_carrier", mutates_args=())
def relation_bias_gradient_carrier(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor,
    groups: int,
) -> Tensor:
    """Return zero while carrying only the mask gradient during backward."""

    return query.new_zeros((*query.shape[:-1], value.shape[-1]))


@relation_bias_gradient_carrier.register_fake
def _relation_bias_carrier_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor,
    groups: int,
) -> Tensor:
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    query, key, value, attn_mask, groups = inputs
    ctx.save_for_backward(query, key, value, attn_mask)
    ctx.groups = groups


def _backward(ctx: Any, grad_output: Tensor) -> tuple[None, None, None, Tensor, None]:
    query, key, value, attn_mask = ctx.saved_tensors
    mask_gradient = _relation_bias_vjp(
        query, key, value, attn_mask, grad_output, ctx.groups
    )
    return None, None, None, mask_gradient, None


relation_bias_gradient_carrier.register_autograd(
    _backward, setup_context=_setup_context
)
