"""The only sample-to-batch action-layout relocation implementation."""

from __future__ import annotations

import torch
from torch import Tensor


def relocate_sample_actions(
    values: Tensor,
    *,
    sample_nodes: int,
    batch_max_nodes: int,
    fill_value: float | int | bool,
) -> Tensor:
    """Pad a node-only sample policy from ``N`` to ``batch_max_nodes``."""

    if sample_nodes < 0 or batch_max_nodes < sample_nodes:
        raise ValueError("invalid sample/batch node counts")
    if values.ndim < 1 or values.shape[-1] != sample_nodes:
        raise ValueError("sample action layout must be node[0:N]")
    output_shape = (*values.shape[:-1], batch_max_nodes)
    output = torch.full(
        output_shape,
        fill_value=fill_value,
        dtype=values.dtype,
        device=values.device,
    )
    output[..., :sample_nodes] = values
    return output


def extract_sample_actions(
    values: Tensor,
    *,
    sample_nodes: int,
    batch_max_nodes: int,
) -> Tensor:
    """Extract one node-only sample policy from a padded batch."""

    if values.ndim < 1 or values.shape[-1] != batch_max_nodes:
        raise ValueError("batch action layout must be node[0:maxN]")
    if sample_nodes < 0 or sample_nodes > batch_max_nodes:
        raise ValueError("invalid sample node count")
    return values[..., :sample_nodes]
