import math

import pytest
import torch
from torch import nn

from startrain.optim import (
    MuonAdamW,
    OptimizerConfig,
    build_optimizer,
    capture_optimizer_diagnostic_snapshot,
    finalize_optimizer_step_diagnostics,
    optimizer_routing_metadata,
)


class RoutingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 4, bias=False)
        self.norm = nn.LayerNorm(4)
        self.policy_head = nn.Linear(4, 2)


def _reference_newton_schulz(gradient: torch.Tensor, steps: int) -> torch.Tensor:
    update = gradient.float()
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.transpose(0, 1)
    update = update / update.norm().clamp_min(1e-7)
    for _ in range(steps):
        covariance = update @ update.transpose(0, 1)
        update = (
            3.4445 * update
            + (-4.7750 * covariance + 2.0315 * covariance @ covariance) @ update
        )
    return update.transpose(0, 1) if transposed else update


def test_routing_metadata_is_stable_complete_and_auditable() -> None:
    first = build_optimizer(
        RoutingModel(),
        OptimizerConfig(kind="muon_adamw", min_muon_elements=1),
    )
    second = build_optimizer(
        RoutingModel(),
        OptimizerConfig(kind="muon_adamw", min_muon_elements=1),
    )

    metadata = optimizer_routing_metadata(first)
    assert metadata == optimizer_routing_metadata(second)
    assert metadata.requested_kind == metadata.implementation == "muon_adamw"
    assert metadata.fallback_used is False
    assert metadata.routing_hash.startswith("sha256-")
    assert len(metadata.routing_hash) == len("sha256-") + 64
    assert metadata.parameter_tensors == 5
    assert metadata.parameter_elements == 34
    assert [group.name for group in metadata.groups] == [
        "muon",
        "adamw_decay",
        "adamw_no_decay",
    ]
    assert metadata.groups[0].parameter_names == ("trunk.weight",)
    assert metadata.groups[1].parameter_names == ("policy_head.weight",)
    assert set(metadata.groups[2].parameter_names) == {
        "norm.weight",
        "norm.bias",
        "policy_head.bias",
    }
    compact = metadata.as_dict()
    assert "parameter_names" not in compact["groups"][0]
    assert metadata.as_dict(include_parameter_names=True)["groups"][0][
        "parameter_names"
    ] == ["trunk.weight"]


def test_muon_fallback_routing_is_explicit_and_deterministic() -> None:
    model = nn.LayerNorm(4)
    config = OptimizerConfig(kind="muon_adamw", min_muon_elements=1)

    with pytest.warns(RuntimeWarning, match="using AdamW"):
        optimizer = build_optimizer(model, config)

    metadata = optimizer_routing_metadata(optimizer)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert metadata.requested_kind == "muon_adamw"
    assert metadata.implementation == "torch_adamw"
    assert metadata.fallback_used is True
    assert [group.name for group in metadata.groups] == ["adamw_no_decay"]
    with pytest.raises(ValueError, match="no StarTrain routing"):
        optimizer_routing_metadata(torch.optim.SGD(model.parameters(), lr=0.1))


def test_custom_adamw_route_matches_torch_reference_across_steps() -> None:
    muon_parameter = nn.Parameter(torch.tensor([[0.2, -0.1], [0.3, 0.4]]))
    actual = nn.Parameter(torch.tensor([0.4, -0.7, 1.2]))
    expected = nn.Parameter(actual.detach().clone())
    config = OptimizerConfig(
        adamw_lr=3e-3,
        muon_lr=1e-2,
        weight_decay=0.07,
        betas=(0.8, 0.91),
        eps=1e-6,
        muon_momentum=0.7,
        muon_ns_steps=2,
    )
    optimizer = MuonAdamW(
        muon_params=[muon_parameter],
        adamw_decay_params=[actual],
        adamw_no_decay_params=[],
        config=config,
    )
    reference = torch.optim.AdamW(
        [expected],
        lr=config.adamw_lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=False,
    )

    for step in range(1, 5):
        gradient = torch.tensor([0.1 * step, -0.2, 0.05 * (step + 1)])
        actual.grad = gradient.clone()
        expected.grad = gradient.clone()
        muon_parameter.grad = torch.full_like(muon_parameter, 0.01 * step)
        optimizer.step()
        reference.step()
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)


def test_muon_route_matches_independent_multistep_reference() -> None:
    initial = torch.tensor([[0.2, -0.3], [0.5, 0.7], [-0.4, 0.1]])
    parameter = nn.Parameter(initial.clone())
    config = OptimizerConfig(
        muon_lr=0.02,
        weight_decay=0.1,
        muon_momentum=0.8,
        muon_nesterov=True,
        muon_ns_steps=3,
    )
    optimizer = MuonAdamW(
        muon_params=[parameter],
        adamw_decay_params=[],
        adamw_no_decay_params=[],
        config=config,
    )
    expected = initial.clone()
    momentum = torch.zeros_like(initial)
    aspect_scale = math.sqrt(initial.shape[0] / initial.shape[1])

    for gradient in (
        torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]]),
        torch.tensor([[-0.2, 0.1], [0.05, -0.3], [0.2, 0.4]]),
    ):
        parameter.grad = gradient.clone()
        momentum.mul_(config.muon_momentum).add_(gradient)
        source = gradient + config.muon_momentum * momentum
        update = _reference_newton_schulz(source, config.muon_ns_steps)
        expected.mul_(1.0 - config.muon_lr * config.weight_decay)
        expected.add_(update, alpha=-config.muon_lr * aspect_scale)
        optimizer.step()
        torch.testing.assert_close(parameter, expected, atol=2e-7, rtol=2e-7)


def test_optimizer_diagnostics_observe_actual_update_and_effective_lr() -> None:
    parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.2)
    parameter.grad = torch.tensor([0.6, 0.8])
    snapshot = capture_optimizer_diagnostic_snapshot(optimizer)

    optimizer.step()
    diagnostics = finalize_optimizer_step_diagnostics(optimizer, snapshot).to_host()

    assert len(diagnostics) == 1
    group = diagnostics[0]
    assert group.name == "group_0"
    assert group.algorithm == "sgd"
    assert group.parameter_tensors == 1
    assert group.parameter_elements == 2
    assert group.configured_learning_rate == pytest.approx(0.2)
    assert group.weight_norm == pytest.approx(5.0)
    assert group.gradient_norm == pytest.approx(1.0)
    assert group.update_norm == pytest.approx(0.2)
    assert group.effective_learning_rate == pytest.approx(0.2)
    assert group.update_to_weight_ratio == pytest.approx(0.04)

    other = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1)
    with pytest.raises(ValueError, match="another optimizer"):
        finalize_optimizer_step_diagnostics(other, snapshot)
