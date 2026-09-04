"""D5-equivariant local/global graph RRT trunk with relational attention.

Architecture v3 adds two structural elements to the approved five-group
GraphResTNet: a D5-invariant relative attention bias in every global block
(keyed by the pairwise relations of :mod:`startrain.topology`) and adaLN-Zero
rule conditioning of every local block from the global scalars. Both are
optional so the previous lineage's checkpoints (feature schema v3, no bias, no
conditioning) still build the exact module tree they were trained with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .contracts import (
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_SCHEMA_VERSION,
    SCORE_MARGIN_MAX,
    SCORE_MARGIN_MIN,
    SOFT_POLICY_TEMPERATURE,
)
from .features import GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM
from .features_v3 import LEGACY_GLOBAL_FEATURE_DIM, LEGACY_NODE_FEATURE_DIM
from .topology import (
    EDGE_CLASS_COUNT,
    relation_count,
    relation_tables,
    ring_slots,
)

MODEL_SCHEMA_VERSION = 3
LocalOperator = Literal["mean", "source_gated"]
LOCAL_OPERATORS: tuple[LocalOperator, ...] = ("mean", "source_gated")
FEATURE_DIMENSIONS: dict[int, tuple[int, int]] = {
    FEATURE_SCHEMA_VERSION: (NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM),
    LEGACY_FEATURE_SCHEMA_VERSION: (LEGACY_NODE_FEATURE_DIM, LEGACY_GLOBAL_FEATURE_DIM),
}


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
    local_operator: LocalOperator = "mean"
    local_blocks_per_group: int = 2
    # Architecture v3. The legacy lineage is (3, False, 0).
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    relational_bias: bool = True
    adaln_hidden: int = 32

    def __post_init__(self) -> None:
        if (
            isinstance(self.feature_schema_version, bool)
            or not isinstance(self.feature_schema_version, int)
            or self.feature_schema_version not in FEATURE_DIMENSIONS
        ):
            raise ValueError(
                "feature_schema_version must be the production schema "
                f"{FEATURE_SCHEMA_VERSION} or the legacy schema "
                f"{LEGACY_FEATURE_SCHEMA_VERSION}"
            )
        node_dim, global_dim = FEATURE_DIMENSIONS[self.feature_schema_version]
        if self.node_feature_dim != node_dim or self.global_feature_dim != global_dim:
            raise ValueError(
                "model feature dimensions must match feature schema "
                f"v{self.feature_schema_version} ({node_dim}, {global_dim})"
            )
        if type(self.relational_bias) is not bool:
            raise TypeError("relational_bias must be boolean")
        if isinstance(self.adaln_hidden, bool) or not isinstance(
            self.adaln_hidden, int
        ):
            raise TypeError("adaln_hidden must be an integer")
        if self.adaln_hidden < 0:
            raise ValueError("adaln_hidden must be non-negative")
        if self.feature_schema_version == LEGACY_FEATURE_SCHEMA_VERSION and (
            self.relational_bias or self.adaln_hidden
        ):
            raise ValueError(
                "the legacy feature schema predates relational bias and rule conditioning"
            )
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
            raise ValueError("score-margin support is fixed at [-151, 151]")
        if self.soft_policy_temperature != SOFT_POLICY_TEMPERATURE:
            raise ValueError("the single KataGo soft-policy temperature is fixed at 4")
        if type(self.local_operator) is not str:
            raise TypeError("local_operator must be a string")
        if self.local_operator not in LOCAL_OPERATORS:
            choices = ", ".join(LOCAL_OPERATORS)
            raise ValueError(f"local_operator must be one of {{{choices}}}")
        if type(self.local_blocks_per_group) is not int:
            raise TypeError("local_blocks_per_group must be an integer")
        if self.local_blocks_per_group <= 0:
            raise ValueError("local_blocks_per_group must be positive")

    @classmethod
    def legacy(cls, **overrides: object) -> "ModelConfig":
        """Configuration of the previous lineage (feature schema v3)."""

        values: dict[str, object] = {
            "node_feature_dim": LEGACY_NODE_FEATURE_DIM,
            "global_feature_dim": LEGACY_GLOBAL_FEATURE_DIM,
            "feature_schema_version": LEGACY_FEATURE_SCHEMA_VERSION,
            "relational_bias": False,
            "adaln_hidden": 0,
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    @property
    def is_legacy(self) -> bool:
        return self.feature_schema_version == LEGACY_FEATURE_SCHEMA_VERSION

    @property
    def score_margin_bins(self) -> int:
        return self.score_margin_max - self.score_margin_min + 1

    @property
    def local_bottleneck_width(self) -> int:
        return max(8, int(self.width * self.bottleneck_ratio))

    @property
    def attention_head_width(self) -> int:
        return self.width // self.attention_heads

    @property
    def ff_hidden_width(self) -> int:
        return max(self.width, int(self.width * self.ff_multiplier))

    @property
    def uses_rule_conditioning(self) -> bool:
        return self.adaln_hidden > 0


class StarModelOutput(NamedTuple):
    policy_logits: Tensor
    outcome_logits: Tensor
    score_margin_logits: Tensor
    ownership_logits: Tensor
    alive_logits: Tensor
    soft_policy_logits: Tensor


class ModelParameterCounts(NamedTuple):
    input_and_output: int
    local_blocks: int
    global_blocks: int
    relational_bias: int
    rule_conditioning: int
    total: int


def model_parameter_counts(config: ModelConfig) -> ModelParameterCounts:
    """Return an exact allocation count without materializing model tensors."""

    width = config.width
    bottleneck = config.local_bottleneck_width
    local_per_block = (
        width
        + 2 * width * bottleneck
        + EDGE_CLASS_COUNT * bottleneck
        + 2 * bottleneck * bottleneck
        + bottleneck * width
        + width
    )
    if config.local_operator == "source_gated":
        local_per_block += width * bottleneck
    local_block_count = config.rrt_groups * config.local_blocks_per_group
    local_blocks = local_block_count * local_per_block

    kv_width = config.kv_heads * config.attention_head_width
    global_per_block = (
        4 * width
        + 2 * width * width
        + 2 * width * kv_width
        + 3 * width * config.ff_hidden_width
    )
    global_blocks = config.rrt_groups * global_per_block

    projection_parameters = (
        (config.node_feature_dim + 1) * width
        + (config.global_feature_dim + 1) * width
        + width
    )
    final_norm_parameters = 2 * width
    head_outputs = 1 + 2 + config.score_margin_bins + 3 + 1 + 1
    head_parameters = (width + 1) * head_outputs
    input_and_output = projection_parameters + final_norm_parameters + head_parameters

    relational_bias = (
        config.rrt_groups * relation_count() * config.attention_heads
        if config.relational_bias
        else 0
    )
    rule_conditioning = 0
    if config.uses_rule_conditioning:
        hidden = config.adaln_hidden
        rule_conditioning = (
            (config.global_feature_dim + 1) * hidden
            + (hidden + 1) * hidden
            + local_block_count * (hidden + 1) * 2 * width
        )
    return ModelParameterCounts(
        input_and_output=input_and_output,
        local_blocks=local_blocks,
        global_blocks=global_blocks,
        relational_bias=relational_bias,
        rule_conditioning=rule_conditioning,
        total=input_and_output
        + local_blocks
        + global_blocks
        + relational_bias
        + rule_conditioning,
    )


def model_parameter_count(config: ModelConfig) -> int:
    """Return the exact parameter total for ``config``."""

    return model_parameter_counts(config).total


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
    """Bottleneck residual message passing over invariant edge classes.

    With ``adaln_hidden > 0`` the RMSNorm output is modulated by a zero-
    initialized scale and shift computed from the shared rule-conditioning
    vector (adaLN-Zero), so every block starts as the unconditioned block and
    learns how the variant should change local message passing.
    """

    def __init__(
        self,
        width: int,
        bottleneck_ratio: float,
        dropout: float,
        norm_eps: float,
        local_operator: LocalOperator = "mean",
        adaln_hidden: int = 0,
    ) -> None:
        super().__init__()
        bottleneck = max(8, int(width * bottleneck_ratio))
        self.local_operator = local_operator
        self.norm = nn.RMSNorm(width, eps=norm_eps)
        self.self_projection = nn.Linear(width, bottleneck, bias=False)
        self.neighbor_projection = nn.Linear(width, bottleneck, bias=False)
        self.edge_embedding = nn.Embedding(EDGE_CLASS_COUNT, bottleneck)
        self.source_gate_projection: nn.Linear | None = None
        if local_operator == "source_gated":
            self.source_gate_projection = nn.Linear(width, bottleneck, bias=False)
        self.update = SwiGLU(bottleneck, bottleneck, width)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((width,), 1e-2))
        self.modulation: nn.Linear | None = None
        if adaln_hidden > 0:
            self.modulation = nn.Linear(adaln_hidden, 2 * width)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        inputs: Tensor,
        neighbor_index: Tensor,
        neighbor_mask: Tensor,
        neighbor_edge_type: Tensor,
        node_mask: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        normalized = self.norm(inputs)
        if self.modulation is not None and condition is not None:
            scale, shift = self.modulation(condition).unsqueeze(1).chunk(2, dim=-1)
            normalized = normalized * (1.0 + scale) + shift
        neighbors = _gather_neighbors(normalized, neighbor_index)
        messages = self.neighbor_projection(neighbors)
        messages = messages + self.edge_embedding(neighbor_edge_type)
        if self.source_gate_projection is not None:
            source_gate = self.source_gate_projection(normalized).unsqueeze(2)
            messages = functional.silu(messages) * torch.sigmoid(source_gate + messages)
        else:
            messages = functional.silu(messages)
        weights = neighbor_mask.unsqueeze(-1).to(dtype=messages.dtype)
        aggregated = (messages * weights).sum(dim=2)
        aggregated = aggregated / weights.sum(dim=2).clamp_min(1.0)
        update = self.update(self.self_projection(normalized) + aggregated)
        output = inputs + self.dropout(update) * self.layer_scale
        return output * node_mask.unsqueeze(-1).to(dtype=output.dtype)


def _explicit_attention_in_fp32(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor,
    groups: int,
) -> Tensor:
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    with torch.autocast(device_type=query.device.type, enabled=False):
        scale = query.shape[-1] ** -0.5
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        probabilities = torch.softmax(scores + attn_mask.float(), dim=-1)
        explicit = torch.matmul(probabilities, value.float())
    return explicit.to(query.dtype)


def _relation_bias_gradient_carrier(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor,
    *,
    groups: int,
) -> Tensor:
    """Zero-valued tensor whose gradient is exactly ``d attention / d attn_mask``.

    ``query``, ``key``, and ``value`` are detached, so the explicit fp32
    softmax attention contributes gradient only to the additive mask (and
    through it to the relation-bias table). Its value is subtracted from
    itself, so the forward result is unchanged. The explicit attention runs
    under activation checkpointing: nothing of its ``[batch, heads, length,
    length]`` intermediates survives the forward pass, and the backward pass
    recomputes one layer at a time.
    """

    explicit = checkpoint(
        _explicit_attention_in_fp32,
        query,
        key,
        value,
        attn_mask,
        groups,
        use_reentrant=False,
    )
    return explicit - explicit.detach()


class GlobalGQABlock(nn.Module):
    """Masked global-token attention using fused scaled-dot-product attention.

    With ``relational_bias`` a learned per-head bias indexed by the D5-invariant
    pairwise relation of every (query, key) pair is added through the additive
    attention mask, so attention can attend by geometry (ring difference,
    angular offset, shortest-path distance) rather than only by content.
    """

    def __init__(
        self,
        width: int,
        query_heads: int,
        kv_heads: int,
        ff_multiplier: float,
        dropout: float,
        norm_eps: float,
        relational_bias: bool = False,
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
        self.relation_bias: nn.Embedding | None = None
        if relational_bias:
            self.relation_bias = nn.Embedding(relation_count(), query_heads)
            nn.init.zeros_(self.relation_bias.weight)

    def forward(
        self,
        token: Tensor,
        nodes: Tensor,
        node_mask: Tensor,
        relation_index: Tensor | None = None,
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
        key_mask = sequence_mask[:, None, None, :]
        attn_mask: Tensor
        relational = self.relation_bias is not None and relation_index is not None
        if relational:
            assert self.relation_bias is not None
            bias = self.relation_bias(relation_index).permute(0, 3, 1, 2)
            attn_mask = bias.to(dtype=query.dtype).masked_fill(
                ~key_mask, torch.finfo(query.dtype).min
            )
        else:
            attn_mask = key_mask
        dropout = self.dropout if self.training else 0.0
        carry_bias_gradient = (
            relational
            and dropout == 0.0
            and torch.is_grad_enabled()
            and attn_mask.requires_grad
        )
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            # Fused attention kernels do not differentiate the additive mask;
            # a mask that requires grad forces the memory-hungry math path. The
            # value comes from the fused kernel with a detached mask and the
            # bias gradient travels through a zero-valued explicit carrier.
            attn_mask=attn_mask.detach() if carry_bias_gradient else attn_mask,
            dropout_p=dropout,
            is_causal=False,
            enable_gqa=self.query_heads != self.kv_heads,
        )
        if carry_bias_gradient:
            attended = attended + _relation_bias_gradient_carrier(
                query.detach(),
                key.detach(),
                value.detach(),
                attn_mask,
                groups=self.query_heads // self.kv_heads,
            )
        attended = attended.transpose(1, 2)
        if torch.compiler.is_exporting():
            # With an additive mask the math kernel hands back a transposed
            # layout that symbolic tracing cannot prove viewable; an explicit
            # contiguous clone keeps the ONNX export well-defined. Eager and
            # compiled runs keep the copy-free reshape.
            attended = attended.clone(memory_format=torch.contiguous_format)
            attended = attended.view(batch, length, width)
        else:
            attended = attended.reshape(batch, length, width)
        sequence = sequence + self.attention_output(attended) * self.attention_scale
        sequence = sequence * sequence_mask.unsqueeze(-1).to(sequence.dtype)
        sequence = sequence + self.ff(self.ff_norm(sequence)) * self.ff_scale
        sequence = sequence * sequence_mask.unsqueeze(-1).to(sequence.dtype)
        return sequence[:, :1], sequence[:, 1:]


class RRTGroup(nn.Module):
    """Configured local edge blocks followed by one global GQA block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.local_blocks = nn.ModuleList(
            [
                LocalEdgeBlock(
                    config.width,
                    config.bottleneck_ratio,
                    config.dropout,
                    config.rms_norm_eps,
                    config.local_operator,
                    config.adaln_hidden,
                )
                for _ in range(config.local_blocks_per_group)
            ]
        )
        self.global_block = GlobalGQABlock(
            config.width,
            config.attention_heads,
            config.kv_heads,
            config.ff_multiplier,
            config.dropout,
            config.rms_norm_eps,
            config.relational_bias,
        )

    def forward(
        self,
        token: Tensor,
        nodes: Tensor,
        neighbor_index: Tensor,
        neighbor_mask: Tensor,
        neighbor_edge_type: Tensor,
        node_mask: Tensor,
        condition: Tensor | None = None,
        relation_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        for block in self.local_blocks:
            nodes = block(
                nodes,
                neighbor_index,
                neighbor_mask,
                neighbor_edge_type,
                node_mask,
                condition,
            )
        return self.global_block(token, nodes, node_mask, relation_index)


def _mask_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


class GraphResTNet(nn.Module):
    """Shared model for rings 4, 6, 8, and 10 and every rule variant."""

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
        self.outcome_head = nn.Linear(width, 2)
        self.score_margin_head = nn.Linear(width, config.score_margin_bins)
        self.ownership_head = nn.Linear(width, 3)
        self.alive_head = nn.Linear(width, 1)
        self.soft_node_policy = nn.Linear(width, 1)

        self.rule_conditioner: nn.Sequential | None = None
        if config.uses_rule_conditioning:
            hidden = config.adaln_hidden
            self.rule_conditioner = nn.Sequential(
                nn.Linear(config.global_feature_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
        # Non-persistent buffers: derived from the board, never checkpointed,
        # and absent from legacy module trees.
        self.relation_tables: Tensor | None
        self.ring_slots: Tensor | None
        if config.relational_bias:
            self.register_buffer("relation_tables", relation_tables(), persistent=False)
            self.register_buffer("ring_slots", ring_slots(), persistent=False)
        else:
            self.relation_tables = None
            self.ring_slots = None

    def relation_index_for(self, rings: Tensor, max_nodes: int) -> Tensor | None:
        """Per-sample `(B, 1 + max_nodes, 1 + max_nodes)` relation ids."""

        if self.relation_tables is None or self.ring_slots is None:
            return None
        slots = self.ring_slots[rings]
        # index_select (rather than a slice) keeps the exported graph free of a
        # specialization on whether the batch reaches the largest board.
        positions = torch.arange(max_nodes + 1, device=rings.device)
        tables = self.relation_tables.index_select(1, positions).index_select(
            2, positions
        )
        return tables[slots]

    def forward(
        self,
        node_features: Tensor,
        global_features: Tensor,
        neighbor_index: Tensor,
        neighbor_mask: Tensor,
        neighbor_edge_type: Tensor,
        node_mask: Tensor,
        legal_action_mask: Tensor,
        rings: Tensor | None = None,
    ) -> StarModelOutput:
        mask_values = node_mask.unsqueeze(-1).to(dtype=node_features.dtype)
        nodes = self.node_projection(node_features) * mask_values
        token = self.global_token.to(dtype=nodes.dtype).expand(
            node_features.shape[0], -1, -1
        )
        token = token + self.global_projection(global_features).unsqueeze(1)
        condition = (
            self.rule_conditioner(global_features)
            if self.rule_conditioner is not None
            else None
        )
        relation_index: Tensor | None = None
        if self.relation_tables is not None:
            if rings is None:
                raise ValueError("relational bias requires the per-sample rings tensor")
            relation_index = self.relation_index_for(rings, node_features.shape[1])

        for group in self.rrt_groups:
            token, nodes = group(
                token,
                nodes,
                neighbor_index,
                neighbor_mask,
                neighbor_edge_type,
                node_mask,
                condition,
                relation_index,
            )

        nodes = self.final_node_norm(nodes) * mask_values
        pooled = self.final_token_norm(token[:, 0])
        policy_logits = _mask_logits(
            self.node_policy(nodes).squeeze(-1), legal_action_mask
        )
        soft_policy_logits = _mask_logits(
            self.soft_node_policy(nodes).squeeze(-1),
            legal_action_mask,
        )
        return StarModelOutput(
            policy_logits=policy_logits,
            outcome_logits=self.outcome_head(pooled),
            score_margin_logits=self.score_margin_head(pooled),
            ownership_logits=self.ownership_head(nodes),
            alive_logits=self.alive_head(nodes).squeeze(-1),
            soft_policy_logits=soft_policy_logits,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
