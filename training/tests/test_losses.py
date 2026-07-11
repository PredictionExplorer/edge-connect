import math
from dataclasses import replace

import pytest
import torch

from startrain.contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from startrain.losses import LossWeights, TrainingTargets, compute_losses
from startrain.model import StarModelOutput


def test_loss_weights_reject_negative_nonfinite_and_all_zero() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LossWeights(policy=-1.0)
    with pytest.raises(ValueError, match="finite"):
        LossWeights(policy=math.inf)
    with pytest.raises(ValueError, match="at least one"):
        LossWeights(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def outputs(batch: int = 2, nodes: int = 3, actions: int = 3) -> StarModelOutput:
    return StarModelOutput(
        policy_logits=torch.zeros(batch, actions, requires_grad=True),
        outcome_logits=torch.zeros(batch, 2, requires_grad=True),
        score_margin_logits=torch.zeros(batch, 303, requires_grad=True),
        ownership_logits=torch.zeros(batch, nodes, 3, requires_grad=True),
        alive_logits=torch.zeros(batch, nodes, requires_grad=True),
        soft_policy_logits=torch.zeros(batch, actions, requires_grad=True),
    )


def targets(batch: int = 2, nodes: int = 3, actions: int = 3) -> TrainingTargets:
    false = torch.zeros(batch, dtype=torch.bool)
    return TrainingTargets(
        policy=torch.zeros(batch, actions),
        outcome=torch.zeros(batch, dtype=torch.long),
        score_margin=torch.zeros(batch, dtype=torch.long),
        ownership=torch.full((batch, nodes), -100, dtype=torch.long),
        alive=torch.full((batch, nodes), -1.0),
        soft_policy=torch.zeros(batch, actions),
        policy_mask=false.clone(),
        outcome_mask=false.clone(),
        score_margin_mask=false.clone(),
        ownership_mask=false.clone(),
        alive_mask=false.clone(),
        soft_policy_mask=false.clone(),
        sample_weight=torch.ones(batch),
        policy_weight=torch.ones(batch),
    )


def test_true_minus_100_margin_is_not_a_missing_sentinel() -> None:
    output = outputs(batch=1)
    target = targets(batch=1)
    target.score_margin[0] = -100
    target.score_margin_mask[0] = True
    losses = compute_losses(
        output,
        target,
        legal_action_mask=torch.zeros(1, 3, dtype=torch.bool),
        node_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    expected_class = -100 - SCORE_MARGIN_MIN
    expected = torch.nn.functional.cross_entropy(
        output.score_margin_logits, torch.tensor([expected_class])
    )
    torch.testing.assert_close(losses["score_margin"], expected)

    target.score_margin[0] = SCORE_MARGIN_MAX + 1
    with pytest.raises(ValueError, match="score margins"):
        compute_losses(
            output,
            target,
            legal_action_mask=torch.zeros(1, 3, dtype=torch.bool),
            node_mask=torch.ones(1, 3, dtype=torch.bool),
        )
    target.score_margin_mask[0] = False
    compute_losses(
        output,
        target,
        legal_action_mask=torch.zeros(1, 3, dtype=torch.bool),
        node_mask=torch.ones(1, 3, dtype=torch.bool),
    )


def test_weights_below_one_are_normalized_not_clamped() -> None:
    output = outputs(batch=1)
    target = targets(batch=1)
    target.outcome[0] = 1
    target.outcome_mask[0] = True
    target.sample_weight[0] = 0.25
    losses = compute_losses(
        output,
        target,
        legal_action_mask=torch.zeros(1, 3, dtype=torch.bool),
        node_mask=torch.ones(1, 3, dtype=torch.bool),
        weights=LossWeights(
            policy=0,
            outcome=1,
            score_margin=0,
            ownership=0,
            alive=0,
            soft_policy=0,
        ),
    )
    assert losses["outcome"].item() == pytest.approx(math.log(2), rel=1e-6)

    target.outcome[0] = 2
    with pytest.raises(ValueError, match="loss=0 or win=1"):
        compute_losses(
            output,
            target,
            legal_action_mask=torch.zeros(1, 3, dtype=torch.bool),
            node_mask=torch.ones(1, 3, dtype=torch.bool),
        )


def test_policy_confidence_weights_affect_only_policy_heads() -> None:
    output = outputs()
    policy_logits = output.policy_logits.detach().clone()
    policy_logits[0, 0] = 3
    policy_logits[1, 1] = 3
    output = output._replace(policy_logits=policy_logits.requires_grad_())
    target = targets()
    target.policy[:, 0] = 1
    target.policy_mask[:] = True
    target.outcome[:] = torch.tensor([0, 1])
    target.outcome_mask[:] = True
    assert target.policy_weight is not None
    target.policy_weight[:] = torch.tensor([1.0, 0.0])
    legal = torch.ones(2, 3, dtype=torch.bool)

    losses = compute_losses(
        output,
        target,
        legal_action_mask=legal,
        node_mask=torch.ones(2, 3, dtype=torch.bool),
        weights=LossWeights(1, 1, 0, 0, 0, 0),
    )

    expected_policy = torch.nn.functional.cross_entropy(
        policy_logits[:1], torch.tensor([0])
    )
    torch.testing.assert_close(losses["policy"], expected_policy)
    assert losses["outcome"].item() == pytest.approx(math.log(2), rel=1e-6)

    target.policy_weight[1] = -1
    with pytest.raises(ValueError, match="policy weights"):
        compute_losses(
            output,
            target,
            legal_action_mask=legal,
            node_mask=torch.ones(2, 3, dtype=torch.bool),
        )


@pytest.mark.parametrize("head", ["ownership", "alive"])
def test_spatial_losses_average_each_sample_before_weighting(head: str) -> None:
    output = outputs()
    target = targets()
    node_mask = torch.tensor([[True, False, False], [True, True, True]])
    if head == "ownership":
        logits = output.ownership_logits.detach().clone()
        logits[1, :, 0] = 10
        output = output._replace(ownership_logits=logits.requires_grad_())
        target.ownership[:] = 0
        target.ownership[0, 1:] = -100
        target.ownership_mask[:] = True
        first = math.log(3)
        second = float(
            torch.nn.functional.cross_entropy(
                torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([0])
            )
        )
        selected_weights = LossWeights(0, 0, 0, 1, 0, 0)
    else:
        logits = output.alive_logits.detach().clone()
        logits[1, :] = 10
        output = output._replace(alive_logits=logits.requires_grad_())
        target.alive[:] = 1
        target.alive[0, 1:] = -1
        target.alive_mask[:] = True
        first = math.log(2)
        second = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.tensor([10.0]), torch.tensor([1.0])
            )
        )
        selected_weights = LossWeights(0, 0, 0, 0, 1, 0)

    losses = compute_losses(
        output,
        target,
        legal_action_mask=torch.zeros(2, 3, dtype=torch.bool),
        node_mask=node_mask,
        weights=selected_weights,
    )
    assert losses[head].item() == pytest.approx((first + second) / 2, rel=1e-5)
    losses["total"].backward()
    assert torch.isfinite(
        output.ownership_logits.grad
        if head == "ownership"
        else output.alive_logits.grad
    ).all()


def test_loss_schema_rejects_malformed_masks_heads_targets_and_weights() -> None:
    output = outputs()
    target = targets()
    legal = torch.ones(2, 3, dtype=torch.bool)
    nodes = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="rank-two"):
        compute_losses(
            output,
            target,
            legal_action_mask=legal[0],
            node_mask=nodes,
        )
    with pytest.raises(ValueError, match="boolean"):
        compute_losses(
            output,
            target,
            legal_action_mask=legal.float(),
            node_mask=nodes,
        )
    with pytest.raises(ValueError, match="batch dimensions"):
        compute_losses(
            output,
            target,
            legal_action_mask=legal,
            node_mask=torch.ones(3, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="policy logits"):
        compute_losses(
            output._replace(policy_logits=torch.zeros(2, 2)),
            target,
            legal_action_mask=legal,
            node_mask=nodes,
        )
    with pytest.raises(ValueError, match="policy target"):
        compute_losses(
            output,
            replace(target, policy=torch.zeros(2, 2)),
            legal_action_mask=legal,
            node_mask=nodes,
        )
    with pytest.raises(ValueError, match="sample weights"):
        compute_losses(
            output,
            replace(target, sample_weight=torch.ones(3)),
            legal_action_mask=legal,
            node_mask=nodes,
        )
