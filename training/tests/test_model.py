from unittest.mock import patch

import torch
import torch.nn.functional as functional

from startrain.features import DoubleStarPosition, encode_batch
from startrain.model import GraphResTNet, ModelConfig
from startrain.symmetry import (
    D5Transform,
    permute_actions,
    permute_nodes,
    transform_position,
)
from startrain.topology import get_topology


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
        pass_streak=0,
        terminal=False,
    )


def tiny_model() -> GraphResTNet:
    return GraphResTNet(
        ModelConfig(
            width=16,
            rrt_groups=5,
            attention_heads=4,
            kv_heads=1,
            bottleneck_ratio=0.5,
        )
    )


def test_approved_trunk_shapes_masks_and_single_soft_policy() -> None:
    model = tiny_model().eval()
    assert len(model.rrt_groups) == 5
    assert all(len(group.local_blocks) == 2 for group in model.rrt_groups)
    batch = encode_batch([position(3), position(4)])
    with patch.object(
        functional,
        "scaled_dot_product_attention",
        wraps=functional.scaled_dot_product_attention,
    ) as fused_attention:
        output = model(*batch.model_args())
    assert fused_attention.call_count == 5
    assert output.policy_logits.shape == (2, batch.max_nodes + 1)
    assert output.soft_policy_logits.shape == (2, batch.max_nodes + 1)
    assert output.wdl_logits.shape == (2, 3)
    assert output.score_margin_logits.shape == (2, 363)
    assert output.ownership_logits.shape == (2, batch.max_nodes, 3)
    assert output.alive_logits.shape == (2, batch.max_nodes)
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


def test_model_equivariance_and_invariance_for_all_d5_transforms() -> None:
    torch.manual_seed(7)
    model = tiny_model().eval()
    source = position(3)
    source_batch = encode_batch([source])
    with torch.no_grad():
        baseline = model(*source_batch.model_args())
    topology = get_topology(3)
    for transform_index in range(10):
        transform = D5Transform.from_index(transform_index)
        permutation = topology.d5_permutation(transform.rotation, transform.reflected)
        transformed_batch = encode_batch([transform_position(source, transform)])
        with torch.no_grad():
            transformed = model(*transformed_batch.model_args())
        torch.testing.assert_close(
            transformed.policy_logits[0],
            permute_actions(baseline.policy_logits[0], permutation),
            atol=3e-5,
            rtol=3e-5,
        )
        torch.testing.assert_close(
            transformed.soft_policy_logits[0],
            permute_actions(baseline.soft_policy_logits[0], permutation),
            atol=3e-5,
            rtol=3e-5,
        )
        torch.testing.assert_close(
            transformed.ownership_logits[0],
            permute_nodes(baseline.ownership_logits[0], permutation),
            atol=3e-5,
            rtol=3e-5,
        )
        torch.testing.assert_close(
            transformed.alive_logits[0],
            permute_nodes(baseline.alive_logits[0], permutation),
            atol=3e-5,
            rtol=3e-5,
        )
        torch.testing.assert_close(
            transformed.wdl_logits, baseline.wdl_logits, atol=3e-5, rtol=3e-5
        )
        torch.testing.assert_close(
            transformed.score_margin_logits,
            baseline.score_margin_logits,
            atol=3e-5,
            rtol=3e-5,
        )
