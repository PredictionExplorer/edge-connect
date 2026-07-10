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
    """Move sample pass ``N`` to batch pass ``max_nodes``.

    All slots between ``sample_nodes`` and ``batch_max_nodes`` retain the
    caller-provided sentinel. The action dimension is always the final axis.
    """

    if sample_nodes < 0 or batch_max_nodes < sample_nodes:
        raise ValueError("invalid sample/batch node counts")
    if values.ndim < 1 or values.shape[-1] != sample_nodes + 1:
        raise ValueError("sample action layout must be node[0:N], pass[N]")
    output_shape = (*values.shape[:-1], batch_max_nodes + 1)
    output = torch.full(
        output_shape,
        fill_value=fill_value,
        dtype=values.dtype,
        device=values.device,
    )
    output[..., :sample_nodes] = values[..., :sample_nodes]
    output[..., batch_max_nodes] = values[..., sample_nodes]
    return output


def extract_sample_actions(
    values: Tensor,
    *,
    sample_nodes: int,
    batch_max_nodes: int,
) -> Tensor:
    """Inverse relocation for one item in a padded batch."""

    if values.ndim < 1 or values.shape[-1] != batch_max_nodes + 1:
        raise ValueError("batch action layout must be node[0:maxN], pass[maxN]")
    if sample_nodes < 0 or sample_nodes > batch_max_nodes:
        raise ValueError("invalid sample node count")
    return torch.cat(
        (
            values[..., :sample_nodes],
            values[..., batch_max_nodes : batch_max_nodes + 1],
        ),
        dim=-1,
    )
