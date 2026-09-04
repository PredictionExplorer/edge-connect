from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as functional

from startrain.features import DoubleStarPosition, encode_batch
from startrain.model import (
    MODEL_SCHEMA_VERSION,
    GraphResTNet,
    LocalEdgeBlock,
    LocalOperator,
    ModelConfig,
    StarModelOutput,
    _relation_bias_gradient_carrier,
    model_parameter_count,
    model_parameter_counts,
)
from startrain.symmetry import (
    D5Transform,
    permute_actions,
    permute_nodes,
    transform_position,
)
from startrain.topology import (
    EDGE_CLASS_COUNT,
    SUPPORTED_RINGS,
    get_topology,
    relation_count,
)


def position(rings: int) -> DoubleStarPosition:
    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    stones[topology.n - 2] = 1
    return DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=0,
        moves_left=2,
        opening=False,
        terminal=False,
    )


def tiny_model(
    *,
    local_operator: LocalOperator = "mean",
    local_blocks_per_group: int = 2,
    rrt_groups: int = 5,
) -> GraphResTNet:
    return GraphResTNet(
        ModelConfig(
            width=16,
            rrt_groups=rrt_groups,
            attention_heads=4,
            kv_heads=1,
            bottleneck_ratio=0.5,
            local_operator=local_operator,
            local_blocks_per_group=local_blocks_per_group,
        )
    )


def production_config(**changes: object) -> ModelConfig:
    return replace(
        ModelConfig(
            width=384,
            rrt_groups=5,
            attention_heads=12,
            kv_heads=3,
            bottleneck_ratio=0.5,
            ff_multiplier=2.5,
        ),
        **changes,
    )


def legacy_production_config(**changes: object) -> ModelConfig:
    """The previous lineage's 384x5 trunk: feature schema v3, no v3 additions."""

    return replace(
        ModelConfig.legacy(
            width=384,
            rrt_groups=5,
            attention_heads=12,
            kv_heads=3,
            bottleneck_ratio=0.5,
            ff_multiplier=2.5,
        ),
        **changes,
    )


def randomize_v3_parameters(model: GraphResTNet) -> None:
    """Give the zero-initialized v3 additions real values so tests exercise them."""

    generator = torch.Generator().manual_seed(23)
    for name, parameter in model.named_parameters():
        if "relation_bias" in name or "modulation" in name:
            with torch.no_grad():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)


def assert_global_outputs_close(
    actual: StarModelOutput,
    expected: StarModelOutput,
    *,
    actual_index: int,
    expected_index: int,
    atol: float = 3e-5,
    rtol: float = 3e-5,
) -> None:
    torch.testing.assert_close(
        actual.outcome_logits[actual_index],
        expected.outcome_logits[expected_index],
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        actual.score_margin_logits[actual_index],
        expected.score_margin_logits[expected_index],
        atol=atol,
        rtol=rtol,
    )


def test_approved_trunk_shapes_masks_and_single_soft_policy() -> None:
    model = tiny_model().eval()
    assert model.config.local_operator == "mean"
    assert model.config.local_blocks_per_group == 2
    assert len(model.rrt_groups) == 5
    assert all(len(group.local_blocks) == 2 for group in model.rrt_groups)
    batch = encode_batch([position(rings) for rings in SUPPORTED_RINGS])
    with patch.object(
        functional,
        "scaled_dot_product_attention",
        wraps=functional.scaled_dot_product_attention,
    ) as fused_attention:
        output = model(*batch.model_args())
    assert fused_attention.call_count == 5
    assert output.policy_logits.shape == (4, batch.max_nodes)
    assert output.soft_policy_logits.shape == (4, batch.max_nodes)
    assert output.outcome_logits.shape == (4, 2)
    assert output.score_margin_logits.shape == (4, 303)
    assert output.ownership_logits.shape == (4, batch.max_nodes, 3)
    assert output.alive_logits.shape == (4, batch.max_nodes)
    illegal = ~batch.legal_action_mask
    minimum = torch.finfo(output.policy_logits.dtype).min
    assert torch.equal(
        output.policy_logits[illegal],
        torch.full_like(output.policy_logits[illegal], minimum),
    )
    assert torch.equal(
        output.soft_policy_logits[illegal],
        torch.full_like(output.soft_policy_logits[illegal], minimum),
    )


def test_default_mean_preserves_v2_state_and_output_parity() -> None:
    legacy_config = ModelConfig(
        width=16,
        rrt_groups=1,
        attention_heads=4,
        kv_heads=1,
        bottleneck_ratio=0.5,
    )
    explicit_config = replace(
        legacy_config,
        local_operator="mean",
        local_blocks_per_group=2,
    )
    assert MODEL_SCHEMA_VERSION == 3
    assert legacy_config == explicit_config

    torch.manual_seed(19)
    legacy_model = GraphResTNet(legacy_config).eval()
    torch.manual_seed(19)
    explicit_model = GraphResTNet(explicit_config).eval()
    assert tuple(legacy_model.state_dict()) == tuple(explicit_model.state_dict())
    assert all(
        "source_gate_projection" not in name for name in legacy_model.state_dict()
    )
    explicit_model.load_state_dict(legacy_model.state_dict(), strict=True)

    batch = encode_batch([position(4), position(10)])
    with torch.no_grad():
        legacy_output = legacy_model(*batch.model_args())
        explicit_output = explicit_model(*batch.model_args())
    for legacy_tensor, explicit_tensor in zip(
        legacy_output, explicit_output, strict=True
    ):
        torch.testing.assert_close(
            legacy_tensor,
            explicit_tensor,
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize("local_operator", ("mean", "source_gated"))
def test_model_equivariance_and_invariance_for_all_rings_and_d5_transforms(
    local_operator: LocalOperator,
) -> None:
    torch.manual_seed(7)
    model = tiny_model(local_operator=local_operator, rrt_groups=1).eval()
    randomize_v3_parameters(model)
    sources = [position(rings) for rings in SUPPORTED_RINGS]
    source_batch = encode_batch(sources)
    with torch.no_grad():
        baseline = model(*source_batch.model_args())
    for transform_index in range(10):
        transform = D5Transform.from_index(transform_index)
        transformed_batch = encode_batch(
            [transform_position(source, transform) for source in sources]
        )
        with torch.no_grad():
            transformed = model(*transformed_batch.model_args())
        for batch_index, rings in enumerate(SUPPORTED_RINGS):
            topology = get_topology(rings)
            permutation = topology.d5_permutation(
                transform.rotation, transform.reflected
            )
            torch.testing.assert_close(
                transformed.policy_logits[batch_index, : topology.n],
                permute_actions(
                    baseline.policy_logits[batch_index, : topology.n],
                    permutation,
                ),
                atol=3e-5,
                rtol=3e-5,
            )
            torch.testing.assert_close(
                transformed.soft_policy_logits[batch_index, : topology.n],
                permute_actions(
                    baseline.soft_policy_logits[batch_index, : topology.n],
                    permutation,
                ),
                atol=3e-5,
                rtol=3e-5,
            )
            torch.testing.assert_close(
                transformed.ownership_logits[batch_index, : topology.n],
                permute_nodes(
                    baseline.ownership_logits[batch_index, : topology.n],
                    permutation,
                ),
                atol=3e-5,
                rtol=3e-5,
            )
            torch.testing.assert_close(
                transformed.alive_logits[batch_index, : topology.n],
                permute_nodes(
                    baseline.alive_logits[batch_index, : topology.n],
                    permutation,
                ),
                atol=3e-5,
                rtol=3e-5,
            )
            assert_global_outputs_close(
                transformed,
                baseline,
                actual_index=batch_index,
                expected_index=batch_index,
            )


@pytest.mark.parametrize("local_operator", ("mean", "source_gated"))
def test_all_ring_padding_matches_unpadded_inference(
    local_operator: LocalOperator,
) -> None:
    torch.manual_seed(23)
    model = tiny_model(local_operator=local_operator, rrt_groups=1).eval()
    positions = [position(rings) for rings in SUPPORTED_RINGS]
    mixed_batch = encode_batch(positions)
    with torch.no_grad():
        mixed = model(*mixed_batch.model_args())

    minimum = torch.finfo(mixed.policy_logits.dtype).min
    for batch_index, source in enumerate(positions):
        topology = get_topology(source.rings)
        single_batch = encode_batch([source])
        with torch.no_grad():
            single = model(*single_batch.model_args())
        for mixed_tensor, single_tensor in (
            (mixed.policy_logits, single.policy_logits),
            (mixed.soft_policy_logits, single.soft_policy_logits),
            (mixed.ownership_logits, single.ownership_logits),
            (mixed.alive_logits, single.alive_logits),
        ):
            torch.testing.assert_close(
                mixed_tensor[batch_index, : topology.n],
                single_tensor[0],
                atol=3e-5,
                rtol=3e-5,
            )
        assert_global_outputs_close(
            mixed,
            single,
            actual_index=batch_index,
            expected_index=0,
        )
        assert torch.equal(
            mixed.policy_logits[batch_index, topology.n :],
            torch.full_like(
                mixed.policy_logits[batch_index, topology.n :],
                minimum,
            ),
        )
        assert torch.equal(
            mixed.soft_policy_logits[batch_index, topology.n :],
            torch.full_like(
                mixed.soft_policy_logits[batch_index, topology.n :],
                minimum,
            ),
        )


def test_source_gated_local_messages_are_edge_local_and_masked() -> None:
    torch.manual_seed(29)
    model = tiny_model(
        local_operator="source_gated",
        local_blocks_per_group=1,
        rrt_groups=1,
    ).eval()
    batch = encode_batch([position(rings) for rings in SUPPORTED_RINGS])
    mask_values = batch.node_mask.unsqueeze(-1).to(batch.node_features.dtype)
    nodes = model.node_projection(batch.node_features) * mask_values
    block = model.rrt_groups[0].local_blocks[0]
    assert isinstance(block, LocalEdgeBlock)
    assert block.source_gate_projection is not None

    with (
        patch.object(
            block.neighbor_projection,
            "forward",
            wraps=block.neighbor_projection.forward,
        ) as neighbor_projection,
        patch.object(
            block.source_gate_projection,
            "forward",
            wraps=block.source_gate_projection.forward,
        ) as source_projection,
    ):
        baseline = block(
            nodes,
            batch.neighbor_index,
            batch.neighbor_mask,
            batch.neighbor_edge_type,
            batch.node_mask,
        )
    neighbor_inputs = neighbor_projection.call_args.args[0]
    source_inputs = source_projection.call_args.args[0]
    assert neighbor_inputs.shape == (
        batch.batch_size,
        batch.max_nodes,
        batch.neighbor_index.shape[-1],
        model.config.width,
    )
    assert source_inputs.shape == (
        batch.batch_size,
        batch.max_nodes,
        model.config.width,
    )
    assert batch.neighbor_index.shape[-1] < batch.max_nodes
    assert torch.count_nonzero(baseline[~batch.node_mask]) == 0

    masked_index = batch.neighbor_index.clone()
    masked_edge_type = batch.neighbor_edge_type.clone()
    masked_index[~batch.neighbor_mask] = batch.max_nodes - 1
    masked_edge_type[~batch.neighbor_mask] = EDGE_CLASS_COUNT - 1
    masked_values_changed = block(
        nodes,
        masked_index,
        batch.neighbor_mask,
        masked_edge_type,
        batch.node_mask,
    )
    torch.testing.assert_close(
        masked_values_changed,
        baseline,
        atol=0.0,
        rtol=0.0,
    )

    changed_edge_type = batch.neighbor_edge_type.clone()
    first_valid = tuple(
        int(value) for value in torch.nonzero(batch.neighbor_mask, as_tuple=False)[0]
    )
    changed_edge_type[first_valid] = (
        changed_edge_type[first_valid] + 1
    ) % EDGE_CLASS_COUNT
    edge_class_changed = block(
        nodes,
        batch.neighbor_index,
        batch.neighbor_mask,
        changed_edge_type,
        batch.node_mask,
    )
    assert not torch.equal(edge_class_changed, baseline)


def test_configured_local_block_count_and_checkpoint_shape_gate() -> None:
    deeper = tiny_model(local_blocks_per_group=3, rrt_groups=2)
    assert all(len(group.local_blocks) == 3 for group in deeper.rrt_groups)

    mean = tiny_model(rrt_groups=1)
    source_gated = tiny_model(local_operator="source_gated", rrt_groups=1)
    with pytest.raises(RuntimeError, match="source_gate_projection"):
        source_gated.load_state_dict(mean.state_dict(), strict=True)


def test_exact_production_parameter_counts() -> None:
    legacy_variants = {
        "current-384x5-kv3-ff2.5": (
            legacy_production_config(),
            10_476_983,
        ),
        "matched-attention-384x5-kv12-ff2.0": (
            legacy_production_config(kv_heads=12, ff_multiplier=2.0),
            10_476_983,
        ),
        "depth-384x7-kv3-ff2.5": (
            legacy_production_config(rrt_groups=7),
            14_614_199,
        ),
        "width-512x5-kv4-ff2.5": (
            legacy_production_config(
                width=512,
                attention_heads=16,
                kv_heads=4,
            ),
            18_556_727,
        ),
        "width-512x5-kv16-ff2.0": (
            legacy_production_config(
                width=512,
                attention_heads=16,
                kv_heads=16,
                ff_multiplier=2.0,
            ),
            18_556_727,
        ),
    }
    for config, expected in legacy_variants.values():
        counts = model_parameter_counts(config)
        assert counts.total == expected
        assert counts.relational_bias == 0 and counts.rule_conditioning == 0
        assert sum(counts[:5]) == counts.total
        assert model_parameter_count(config) == expected
        assert GraphResTNet(config).parameter_count() == expected

    # Architecture v3: Stage A keeps the 384x5 trunk and adds the relational
    # bias and adaLN-Zero conditioning; Stage B is the 512x6 GQA 16/4 trunk.
    stage_a = production_config()
    stage_a_counts = model_parameter_counts(stage_a)
    assert stage_a_counts.local_blocks == 2_962_560
    assert stage_a_counts.global_blocks == 7_380_480
    assert stage_a_counts.relational_bias == 5 * relation_count() * 12
    assert stage_a_counts.rule_conditioning == (26 * 32 + 33 * 32 + 10 * 33 * 2 * 384)
    assert stage_a_counts.total == 10_929_399
    assert GraphResTNet(stage_a).parameter_count() == stage_a_counts.total
    stage_b = production_config(width=512, rrt_groups=6, attention_heads=16, kv_heads=4)
    stage_b_counts = model_parameter_counts(stage_b)
    assert stage_b_counts.total == 22_953_879
    assert GraphResTNet(stage_b).parameter_count() == stage_b_counts.total


def test_legacy_config_rebuilds_the_previous_lineage_module_tree() -> None:
    legacy = GraphResTNet(legacy_production_config())
    assert legacy.config.is_legacy
    assert legacy.rule_conditioner is None
    assert legacy.relation_tables is None
    keys = list(legacy.state_dict())
    assert all(
        "relation" not in key and "modulation" not in key and "conditioner" not in key
        for key in keys
    )
    assert legacy.node_projection.in_features == 15
    assert legacy.global_projection.in_features == 25 - 8
    with pytest.raises(ValueError, match="legacy feature schema"):
        ModelConfig.legacy(relational_bias=True)
    with pytest.raises(ValueError, match="legacy feature schema"):
        ModelConfig.legacy(adaln_hidden=16)
    with pytest.raises(ValueError, match="feature dimensions"):
        ModelConfig(node_feature_dim=15, global_feature_dim=17)
    with pytest.raises(ValueError, match="feature_schema_version"):
        ModelConfig(feature_schema_version=2)

    # A v3 network without the additions is the legacy trunk with wider inputs.
    plain = GraphResTNet(production_config(relational_bias=False, adaln_hidden=0))
    plain_keys = [key for key in plain.state_dict()]
    assert plain_keys == keys
    assert plain.parameter_count() - legacy.parameter_count() == (4 + 8) * 384


def test_relational_bias_and_conditioning_start_as_the_identity() -> None:
    torch.manual_seed(5)
    with_additions = tiny_model(rrt_groups=2).eval()
    without = GraphResTNet(
        replace(with_additions.config, relational_bias=False, adaln_hidden=0)
    ).eval()
    shared = {
        key: value
        for key, value in with_additions.state_dict().items()
        if "relation" not in key
        and "modulation" not in key
        and "conditioner" not in key
    }
    without.load_state_dict(shared, strict=True)
    batch = encode_batch([position(4), position(10)])
    with torch.no_grad():
        first = with_additions(*batch.model_args())
        second = without(*batch.model_args())
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)

    # Once the bias table has values, attention depends on the relation index.
    randomize_v3_parameters(with_additions)
    with torch.no_grad():
        third = with_additions(*batch.model_args())
    assert not torch.allclose(third.policy_logits, first.policy_logits)
    with pytest.raises(ValueError, match="rings"):
        with_additions(*batch.model_args()[:-1])


def test_relation_bias_gradient_carrier_matches_the_explicit_attention_gradient() -> (
    None
):
    """The fused-attention value plus the carrier differentiates exactly like math attention."""

    torch.manual_seed(11)
    batch, heads, kv_heads, length, width = 2, 4, 2, 9, 8
    query = torch.randn(batch, heads, length, width, dtype=torch.float64)
    key = torch.randn(batch, kv_heads, length, width, dtype=torch.float64)
    value = torch.randn(batch, kv_heads, length, width, dtype=torch.float64)
    table = torch.randn(7, heads, dtype=torch.float64, requires_grad=True)
    relation = torch.randint(0, 7, (batch, length, length))
    key_mask = torch.ones(batch, 1, 1, length, dtype=torch.bool)
    key_mask[1, ..., -2:] = False
    upstream = torch.randn(batch, heads, length, width, dtype=torch.float64)

    def mask() -> torch.Tensor:
        bias = table[relation].permute(0, 3, 1, 2)
        return bias.masked_fill(~key_mask, torch.finfo(torch.float64).min)

    reference = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask(), enable_gqa=True
    )
    (reference * upstream).sum().backward()
    assert table.grad is not None
    expected = table.grad.clone()
    table.grad = None

    attn_mask = mask()
    fused = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask.detach(), enable_gqa=True
    )
    carrier = _relation_bias_gradient_carrier(
        query.detach(),
        key.detach(),
        value.detach(),
        attn_mask,
        groups=heads // kv_heads,
    )
    torch.testing.assert_close(carrier, torch.zeros_like(carrier), atol=0.0, rtol=0.0)
    ((fused + carrier) * upstream).sum().backward()
    assert table.grad is not None
    # The carrier evaluates its explicit attention in fp32 by design.
    torch.testing.assert_close(table.grad, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fused, reference, atol=1e-12, rtol=1e-12)

    # The whole model: training-mode gradients of the bias table equal the
    # gradients of a pure math-attention reference built from the same weights.
    model = tiny_model(rrt_groups=2).train()
    randomize_v3_parameters(model)
    inputs = encode_batch([position(4), position(6)])
    output = model(*inputs.model_args())
    output.policy_logits.sum().backward()
    grads = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if "relation_bias" in name and parameter.grad is not None
    }
    assert grads
    model.zero_grad(set_to_none=True)
    with patch("startrain.model._relation_bias_gradient_carrier") as carrier_mock:
        carrier_mock.side_effect = AssertionError("carrier must not run in eval mode")
        model.eval()
        with torch.no_grad():
            model(*inputs.model_args())
    model.train()
    with patch.object(torch, "is_grad_enabled", return_value=False):
        # Forcing the math path (mask keeps its gradient) must give the same table gradients.
        output = model(*inputs.model_args())
        output.policy_logits.sum().backward()
    for name, parameter in model.named_parameters():
        if name in grads:
            assert parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, grads[name], atol=1e-5, rtol=1e-4
            )


def test_parameter_matched_relational_variants_report_exact_deltas() -> None:
    phase_two = legacy_production_config(kv_heads=12, ff_multiplier=2.0)
    local_heavy = replace(
        phase_two,
        bottleneck_ratio=35 / 64,
        ff_multiplier=419 / 384,
        local_blocks_per_group=3,
    )
    source_gated = replace(
        phase_two,
        ff_multiplier=5 / 3,
        local_operator="source_gated",
    )

    assert local_heavy.local_bottleneck_width == 210
    assert local_heavy.ff_hidden_width == 419
    assert source_gated.local_bottleneck_width == 192
    assert source_gated.ff_hidden_width == 640

    reference_count = model_parameter_count(phase_two)
    local_heavy_count = model_parameter_count(local_heavy)
    source_gated_count = model_parameter_count(source_gated)
    assert reference_count == 10_476_983
    assert local_heavy_count == 10_476_953
    assert local_heavy_count - reference_count == -30
    assert local_heavy_count != reference_count
    assert abs(local_heavy_count - reference_count) / reference_count < 0.001
    assert source_gated_count == reference_count
    assert GraphResTNet(local_heavy).parameter_count() == local_heavy_count
    assert GraphResTNet(source_gated).parameter_count() == source_gated_count


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"node_feature_dim": 16}, "feature dimensions"),
        ({"width": 0}, "width must be positive"),
        ({"rrt_groups": 0}, "rrt_groups"),
        ({"width": 10, "attention_heads": 4}, "divisible by attention_heads"),
        (
            {"width": 12, "attention_heads": 4, "kv_heads": 3},
            "attention_heads must be divisible",
        ),
        ({"bottleneck_ratio": 0.0}, "bottleneck_ratio"),
        ({"ff_multiplier": 0.0}, "ff_multiplier"),
        ({"dropout": 1.0}, "dropout"),
        ({"score_margin_min": -150}, r"\[-151, 151\]"),
        ({"soft_policy_temperature": 2.0}, "temperature is fixed at 4"),
        ({"local_operator": "sum"}, "local_operator must be one of"),
        ({"local_blocks_per_group": 0}, "local_blocks_per_group must be positive"),
    ],
)
def test_model_config_enforces_v2_head_and_feature_contracts(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelConfig(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"local_operator": 1}, "local_operator must be a string"),
        (
            {"local_blocks_per_group": True},
            "local_blocks_per_group must be an integer",
        ),
        (
            {"local_blocks_per_group": 2.0},
            "local_blocks_per_group must be an integer",
        ),
    ],
)
def test_model_config_strictly_types_new_local_fields(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        ModelConfig(**changes)
