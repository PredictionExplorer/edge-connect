from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from startrain.actions import extract_sample_actions, relocate_sample_actions
from startrain.losses import LossWeights, TrainingTargets, compute_losses
from startrain.model import StarModelOutput


@settings(max_examples=100, deadline=None)
@given(
    sample_nodes=st.integers(min_value=0, max_value=275),
    extra_nodes=st.integers(min_value=0, max_value=275),
)
def test_action_relocation_is_an_exact_inverse(
    sample_nodes: int, extra_nodes: int
) -> None:
    batch_nodes = sample_nodes + extra_nodes
    values = torch.arange(sample_nodes, dtype=torch.float32)
    relocated = relocate_sample_actions(
        values,
        sample_nodes=sample_nodes,
        batch_max_nodes=batch_nodes,
        fill_value=float("nan"),
    )
    restored = extract_sample_actions(
        relocated,
        sample_nodes=sample_nodes,
        batch_max_nodes=batch_nodes,
    )
    torch.testing.assert_close(restored, values)
    if extra_nodes:
        assert bool(torch.isnan(relocated[sample_nodes:batch_nodes]).all())


@settings(max_examples=75, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=5),
    actions=st.integers(min_value=2, max_value=12),
    target_action=st.integers(min_value=0, max_value=100),
)
def test_masked_policy_loss_is_finite_and_never_trains_illegal_logits(
    batch_size: int, actions: int, target_action: int
) -> None:
    nodes = actions
    selected = target_action % actions
    policy_logits = torch.randn(batch_size, actions, requires_grad=True)
    legal = torch.zeros(batch_size, actions, dtype=torch.bool)
    legal[:, selected] = True
    policy = torch.zeros(batch_size, actions)
    policy[:, selected] = 1.0
    output = StarModelOutput(
        policy_logits=policy_logits,
        outcome_logits=torch.zeros(batch_size, 2, requires_grad=True),
        score_margin_logits=torch.zeros(batch_size, 303, requires_grad=True),
        ownership_logits=torch.zeros(batch_size, nodes, 3, requires_grad=True),
        alive_logits=torch.zeros(batch_size, nodes, requires_grad=True),
        soft_policy_logits=policy_logits.clone(),
    )
    unavailable = torch.zeros(batch_size, dtype=torch.bool)
    targets = TrainingTargets(
        policy=policy,
        outcome=torch.zeros(batch_size, dtype=torch.long),
        score_margin=torch.zeros(batch_size, dtype=torch.long),
        ownership=torch.full((batch_size, nodes), -100, dtype=torch.long),
        alive=torch.full((batch_size, nodes), -1.0),
        soft_policy=policy,
        policy_mask=torch.ones(batch_size, dtype=torch.bool),
        outcome_mask=unavailable,
        score_margin_mask=unavailable,
        ownership_mask=unavailable,
        alive_mask=unavailable,
        soft_policy_mask=unavailable,
    )
    losses = compute_losses(
        output,
        targets,
        legal_action_mask=legal,
        node_mask=torch.ones(batch_size, nodes, dtype=torch.bool),
        weights=LossWeights(1, 0, 0, 0, 0, 0),
    )
    losses["total"].backward()
    assert bool(torch.isfinite(losses["total"]))
    assert policy_logits.grad is not None
    assert bool((policy_logits.grad[~legal] == 0).all())
