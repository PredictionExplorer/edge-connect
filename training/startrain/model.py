"""Approved five-group D5-equivariant local/global graph RRT trunk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .contracts import (
    SCORE_MARGIN_MAX,
    SCORE_MARGIN_MIN,
    SOFT_POLICY_TEMPERATURE,
)
from .features import GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM
from .topology import EDGE_CLASS_COUNT


@dataclass(frozen=True, slots=True)
class ModelConfig:
    node_feature_dim: int = NODE_FEATURE_DIM
    global_feature_dim: int = GLOBAL_FEATURE_DIM
    width: int = 128
    rrt_groups: int = 5
    attention_heads: int = 8
    kv_heads: int = 2
    bottleneck_ratio: float = 0.5
    ff_multiplier: float = 2.0
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    score_margin_min: int = SCORE_MARGIN_MIN
    score_margin_max: int = SCORE_MARGIN_MAX
    soft_policy_temperature: float = SOFT_POLICY_TEMPERATURE

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.rrt_groups <= 0:
            raise ValueError("rrt_groups must be positive")
        if self.width % self.attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")
        if not 0.0 < self.bottleneck_ratio <= 1.0:
            raise ValueError("bottleneck_ratio must be in (0, 1]")
        if self.ff_multiplier <= 0:
            raise ValueError("ff_multiplier must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if (
            self.score_margin_min != SCORE_MARGIN_MIN
            or self.score_margin_max != SCORE_MARGIN_MAX
        ):
            raise ValueError("score-margin support is fixed at [-181, 181]")
        if self.soft_policy_temperature != SOFT_POLICY_TEMPERATURE:
            raise ValueError("the single KataGo soft-policy temperature is fixed at 4")

    @property
    def score_margin_bins(self) -> int:
        return self.score_margin_max - self.score_margin_min + 1


class StarModelOutput(NamedTuple):
    policy_logits: Tensor
    wdl_logits: Tensor
    score_margin_logits: Tensor
    ownership_logits: Tensor
    alive_logits: Tensor
    soft_policy_logits: Tensor


class SwiGLU(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_width, 2 * hidden_width, bias=False)
        self.output_projection = nn.Linear(hidden_width, output_width, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.input_projection(inputs).chunk(2, dim=-1)
        return self.output_projection(functional.silu(gate) * value)


def _gather_neighbors(inputs: Tensor, neighbor_index: Tensor) -> Tensor:
    batch, nodes, channels = inputs.shape
    degree = neighbor_index.shape[-1]
    flattened = neighbor_index.reshape(batch, nodes * degree)
    gather_index = flattened.unsqueeze(-1).expand(-1, -1, channels)
    return inputs.gather(1, gather_index).reshape(batch, nodes, degree, channels)


class LocalEdgeBlock(nn.Module):
    """Bottleneck residual message passing over invariant edge classes."""

    def __init__(
        self,
        width: int,
        bottleneck_ratio: float,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        bottleneck = max(8, int(width * bottleneck_ratio))
        self.norm = nn.RMSNorm(width, eps=norm_eps)
        self.self_projection = nn.Linear(width, bottleneck, bias=False)
        self.neighbor_projection = nn.Linear(width, bottleneck, bias=False)
        self.edge_embedding = nn.Embedding(EDGE_CLASS_COUNT, bottleneck)
        self.update = SwiGLU(bottleneck, bottleneck, width)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((width,), 1e-2))

    def forward(
        self,
        inputs: Tensor,
        neighbor_index: Tensor,
        neighbor_mask: Tensor,
        neighbor_edge_type: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        normalized = self.norm(inputs)
        neighbors = _gather_neighbors(normalized, neighbor_index)
        messages = self.neighbor_projection(neighbors)
        messages = functional.silu(messages + self.edge_embedding(neighbor_edge_type))
        weights = neighbor_mask.unsqueeze(-1).to(dtype=messages.dtype)
        aggregated = (messages * weights).sum(dim=2)
        aggregated = aggregated / weights.sum(dim=2).clamp_min(1.0)
        update = self.update(self.self_projection(normalized) + aggregated)
        output = inputs + self.dropout(update) * self.layer_scale
        return output * node_mask.unsqueeze(-1).to(dtype=output.dtype)


class GlobalGQABlock(nn.Module):
    """Masked global-token attention using fused scaled-dot-product attention."""

    def __init__(
        self,
        width: int,
        query_heads: int,
        kv_heads: int,
        ff_multiplier: float,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_width = width // query_heads
        self.dropout = dropout
        self.attention_norm = nn.RMSNorm(width, eps=norm_eps)
        self.query = nn.Linear(width, query_heads * self.head_width, bias=False)
        self.key = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.value = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)
        hidden = max(width, int(width * ff_multiplier))
        self.ff_norm = nn.RMSNorm(width, eps=norm_eps)
        self.ff = SwiGLU(width, hidden, width)
        self.attention_scale = nn.Parameter(torch.full((width,), 1e-2))
        self.ff_scale = nn.Parameter(torch.full((width,), 1e-2))

    def forward(
        self,
        token: Tensor,
        nodes: Tensor,
        node_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        sequence = torch.cat((token, nodes), dim=1)
        token_mask = torch.ones(
            (node_mask.shape[0], 1), dtype=torch.bool, device=node_mask.device
        )
        sequence_mask = torch.cat((token_mask, node_mask), dim=1)
        normalized = self.attention_norm(sequence)
        batch, length, width = normalized.shape
        query = self.query(normalized).reshape(
            batch, length, self.query_heads, self.head_width
        )
        key = self.key(normalized).reshape(
            batch, length, self.kv_heads, self.head_width
        )
        value = self.value(normalized).reshape(
            batch, length, self.kv_heads, self.head_width
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=sequence_mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=self.query_heads != self.kv_heads,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        sequence = sequence + self.attention_output(attended) * self.attention_scale
        sequence = sequence * sequence_mask.unsqueeze(-1).to(sequence.dtype)
        sequence = sequence + self.ff(self.ff_norm(sequence)) * self.ff_scale
        sequence = sequence * sequence_mask.unsqueeze(-1).to(sequence.dtype)
        return sequence[:, :1], sequence[:, 1:]


class RRTGroup(nn.Module):
    """Exactly two local edge blocks followed by one global GQA block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.local_blocks = nn.ModuleList(
            [
                LocalEdgeBlock(
                    config.width,
                    config.bottleneck_ratio,
                    config.dropout,
                    config.rms_norm_eps,
                )
                for _ in range(2)
            ]
        )
        self.global_block = GlobalGQABlock(
            config.width,
            config.attention_heads,
            config.kv_heads,
            config.ff_multiplier,
            config.dropout,
            config.rms_norm_eps,
        )

    def forward(
        self,
        token: Tensor,
        nodes: Tensor,
        neighbor_index: Tensor,
        neighbor_mask: Tensor,
        neighbor_edge_type: Tensor,
        node_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        for block in self.local_blocks:
            nodes = block(
                nodes,
                neighbor_index,
                neighbor_mask,
                neighbor_edge_type,
                node_mask,
            )
        return self.global_block(token, nodes, node_mask)


def _mask_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


class GraphResTNet(nn.Module):
    """Shared model for rings 3..12 with five configurable-width RRT groups."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.node_projection = nn.Linear(config.node_feature_dim, width)
        self.global_projection = nn.Linear(config.global_feature_dim, width)
        self.global_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.global_token, std=0.02)
        self.rrt_groups = nn.ModuleList(
            [RRTGroup(config) for _ in range(config.rrt_groups)]
        )
        self.final_node_norm = nn.RMSNorm(width, eps=config.rms_norm_eps)
        self.final_token_norm = nn.RMSNorm(width, eps=config.rms_norm_eps)

        self.node_policy = nn.Linear(width, 1)
        self.pass_policy = nn.Linear(width, 1)
        self.wdl_head = nn.Linear(width, 3)
        self.score_margin_head = nn.Linear(width, config.score_margin_bins)
        self.ownership_head = nn.Linear(width, 3)
        self.alive_head = nn.Linear(width, 1)
        self.soft_node_policy = nn.Linear(width, 1)
        self.soft_pass_policy = nn.Linear(width, 1)

    def forward(
        self,
        node_features: Tensor,
        global_features: Tensor,
        neighbor_index: Tensor,
        neighbor_mask: Tensor,
        neighbor_edge_type: Tensor,
        node_mask: Tensor,
        legal_action_mask: Tensor,
    ) -> StarModelOutput:
        mask_values = node_mask.unsqueeze(-1).to(dtype=node_features.dtype)
        nodes = self.node_projection(node_features) * mask_values
        token = self.global_token.to(dtype=nodes.dtype).expand(
            node_features.shape[0], -1, -1
        )
        token = token + self.global_projection(global_features).unsqueeze(1)

        for group in self.rrt_groups:
            token, nodes = group(
                token,
                nodes,
                neighbor_index,
                neighbor_mask,
                neighbor_edge_type,
                node_mask,
            )

        nodes = self.final_node_norm(nodes) * mask_values
        pooled = self.final_token_norm(token[:, 0])
        policy_logits = _mask_logits(
            torch.cat(
                (self.node_policy(nodes).squeeze(-1), self.pass_policy(pooled)),
                dim=1,
            ),
            legal_action_mask,
        )
        soft_policy_logits = _mask_logits(
            torch.cat(
                (
                    self.soft_node_policy(nodes).squeeze(-1),
                    self.soft_pass_policy(pooled),
                ),
                dim=1,
            ),
            legal_action_mask,
        )
        return StarModelOutput(
            policy_logits=policy_logits,
            wdl_logits=self.wdl_head(pooled),
            score_margin_logits=self.score_margin_head(pooled),
            ownership_logits=self.ownership_head(nodes),
            alive_logits=self.alive_head(nodes).squeeze(-1),
            soft_policy_logits=soft_policy_logits,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
