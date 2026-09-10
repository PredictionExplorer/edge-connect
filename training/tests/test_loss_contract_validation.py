"""Stored-teacher loss contracts, checked against independent KL arithmetic."""

from dataclasses import replace
import math

import pytest
import torch

from startrain.losses import LossWeights, compute_losses
from test_losses import outputs, targets


def teacher_targets():
    target = targets()
    target.policy[:, 0] = 1
    margin = torch.zeros(2, 303)
    margin[0, 0] = 1
    margin[1, :2] = 1
    return replace(
        target,
        teacher_mask=torch.ones(2, dtype=torch.bool),
        teacher_policy=torch.tensor([[1.0, 3.0, 100.0], [3.0, 0.0, 1.0]]),
        teacher_outcome=torch.tensor([[1.0, 0.0], [1.0, 3.0]]),
        teacher_score_margin=margin,
    )


def test_teacher_kl_masks_illegal_mass_and_preserves_independent_head_weights():
    output = outputs()
    target = replace(
        teacher_targets(),
        sample_weight=torch.tensor([1.0, 3.0]),
        policy_weight=torch.tensor([2.0, 0.0]),
    )
    weights = LossWeights(
        teacher_policy=0.5, teacher_outcome=0.75, teacher_score_margin=0.25
    )
    loss = compute_losses(
        output,
        target,
        legal_action_mask=torch.tensor([[True, True, False], [True, False, False]]),
        node_mask=torch.ones(2, 3, dtype=torch.bool),
        weights=weights,
    )

    # A uniform student's KL is log(class count) minus teacher entropy.
    # Illegal mass 100 is excluded; remaining policy mass renormalizes to 1:3.
    entropy = -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
    policy_kl = math.log(2) - entropy
    outcome_kl = math.log(2) - 0.75 * entropy
    margin_kl = math.log(303) - 0.75 * math.log(2)
    assert loss["teacher_policy"].item() == pytest.approx(policy_kl, abs=1e-6)
    assert loss["teacher_outcome"].item() == pytest.approx(outcome_kl, abs=1e-6)
    assert loss["teacher_score_margin"].item() == pytest.approx(margin_kl, abs=1e-6)
    assert loss["teacher_samples"].item() == 2
    assert loss["total"].item() == pytest.approx(
        0.5 * policy_kl + 0.75 * outcome_kl + 0.25 * margin_kl, abs=1e-6
    )

    loss["total"].backward()
    # d KL / d logit = student probability - teacher probability, scaled
    # by head weight and that head's normalized sample weight.
    torch.testing.assert_close(
        output.policy_logits.grad,
        torch.tensor([[0.125, -0.125, 0.0], [0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        output.outcome_logits.grad,
        torch.tensor([[-0.09375, 0.09375], [0.140625, -0.140625]]),
    )
    expected_margin_gradient = torch.full((2, 303), 1 / 303)
    expected_margin_gradient[0, 0] -= 1
    expected_margin_gradient[1, :2] -= 0.5
    expected_margin_gradient *= torch.tensor([[0.0625], [0.1875]])
    torch.testing.assert_close(
        output.score_margin_logits.grad, expected_margin_gradient
    )


def test_empty_teacher_distributions_and_all_illegal_policy_have_zero_gradient():
    output = outputs()
    target = replace(
        teacher_targets(),
        teacher_policy=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        teacher_outcome=torch.zeros(2, 2),
        teacher_score_margin=torch.zeros(2, 303),
    )
    loss = compute_losses(
        output,
        target,
        legal_action_mask=torch.tensor([[True, True, True], [False, False, False]]),
        node_mask=torch.ones(2, 3, dtype=torch.bool),
        weights=LossWeights(
            teacher_policy=1, teacher_outcome=1, teacher_score_margin=1
        ),
    )
    for name in ("teacher_policy", "teacher_outcome", "teacher_score_margin", "total"):
        assert loss[name].item() == 0
    loss["total"].backward()
    for logits in output:
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.count_nonzero(logits.grad).item() == 0


def test_teacher_availability_excludes_unavailable_rows_from_loss_and_gradients():
    output = outputs()
    target = replace(teacher_targets(), teacher_mask=torch.tensor([False, True]))
    loss = compute_losses(
        output,
        target,
        legal_action_mask=torch.ones(2, 3, dtype=torch.bool),
        node_mask=torch.ones(2, 3, dtype=torch.bool),
        weights=LossWeights(teacher_outcome=1),
    )
    entropy = -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
    assert loss["total"].item() == pytest.approx(math.log(2) - entropy, abs=1e-6)
    assert loss["teacher_samples"].item() == 1
    loss["total"].backward()
    torch.testing.assert_close(
        output.outcome_logits.grad, torch.tensor([[0.0, 0.0], [0.25, -0.25]])
    )


@pytest.mark.parametrize("teacher_available", [False, True])
def test_absent_or_disabled_teacher_preserves_hard_target_loss(teacher_available):
    target = teacher_targets() if teacher_available else targets()
    target.outcome[:] = 1
    target.outcome_mask[:] = True
    weights = LossWeights(teacher_outcome=0 if teacher_available else 1)
    loss = compute_losses(
        outputs(),
        target,
        legal_action_mask=torch.ones(2, 3, dtype=torch.bool),
        node_mask=torch.ones(2, 3, dtype=torch.bool),
        weights=weights,
    )
    assert not any(name.startswith("teacher_") for name in loss)
    assert loss["total"].item() == pytest.approx(math.log(2), abs=1e-6)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clinch_mask", torch.ones(2, 1, dtype=torch.bool), "clinch mask"),
        ("teacher_mask", torch.ones(2, 1, dtype=torch.bool), "teacher mask"),
        ("teacher_policy", None, "teacher policy"),
        ("teacher_policy", torch.zeros(2, 4), "teacher policy"),
        ("teacher_outcome", None, "teacher outcome"),
        ("teacher_outcome", torch.zeros(2, 3), "teacher outcome"),
        ("teacher_score_margin", None, "teacher score margin"),
        ("teacher_score_margin", torch.zeros(2, 302), "teacher score margin"),
    ],
)
def test_optional_target_contract_rejects_missing_or_misaligned_heads(
    field, value, message
):
    with pytest.raises(ValueError, match=message):
        compute_losses(
            outputs(),
            replace(teacher_targets(), **{field: value}),
            legal_action_mask=torch.ones(2, 3, dtype=torch.bool),
            node_mask=torch.ones(2, 3, dtype=torch.bool),
            weights=LossWeights(teacher_outcome=1),
        )
