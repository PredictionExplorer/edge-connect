from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from startrain.features import DoubleStarPosition
from startrain.gradient_diagnostics import collect_gradient_diagnostics
from startrain.optim import build_optimizer
from startrain.replay import ReplayBatch, ReplaySample, collate_replay_samples
from startrain.topology import get_topology
from test_replay import decisive_score, normalized_policy, sample_for


class RoutedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.ones(8, 8))
        self.scale = nn.Parameter(torch.zeros(2))
        self.bias = nn.Parameter(torch.ones(2))
        self.unused = nn.Parameter(torch.ones(3))
        self.matrix.grad = torch.zeros_like(self.matrix)
        self.matrix.grad.reshape(-1)[:2] = torch.tensor([3.0, 4.0])
        self.scale.grad = torch.tensor([12.0, 0.0])
        self.bias.grad = torch.zeros_like(self.bias)


def batch():
    return collate_replay_samples([sample_for(4), sample_for(10)], prefer_native=False)


def variant_sample(mode, handicap, pie, ring=4):
    position = DoubleStarPosition(
        rings=ring,
        stones=torch.full((get_topology(ring).n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=handicap,
        opening=True,
        terminal=False,
        mode=mode,
        handicap=handicap,
        pie=pie,
    )
    return ReplaySample.from_position(
        position,
        policy=normalized_policy(position),
        final_score=decisive_score(position),
        search_provenance="diagnostic-test",
        policy_provenance="completed-q",
    )


def test_preclip_attribution_routes_shares_zero_missing_and_snapshot_ownership():
    model = RoutedModel()
    optimizer = build_optimizer(model)
    before = {
        name: (
            value.detach().clone(),
            value.grad.clone() if value.grad is not None else None,
        )
        for name, value in model.named_parameters()
    }
    snapshot = collect_gradient_diagnostics(model, batch(), optimizer)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name][0], rtol=0, atol=0)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, before[name][1], rtol=0, atol=0)
    assert not optimizer.state
    # The diagnostic must retain pre-clip values after the real optimizer runs.
    with torch.no_grad():
        model.scale.grad.zero_()
        model.matrix.fill_(30)
    report = snapshot.to_host()
    assert report["pre_clip_global_norm"] == 13.0
    assert report["missing_gradient_tensors"] == 1
    assert report["zero_gradient_tensors"] == 1
    assert [value["name"] for value in report["parameters"]] == [
        "scale",
        "matrix",
        "bias",
    ]
    first, second = report["parameters"][:2]
    assert first["optimizer_group"] == "adamw_no_decay"
    assert first["algorithm"] == "adamw"
    assert first["pre_clip_gradient_norm"] == 12
    assert first["gradient_squared_share"] == pytest.approx(144 / 169)
    assert first["parameter_norm"] == 0 and first["gradient_to_parameter_ratio"] is None
    assert second["optimizer_group"] == second["algorithm"] == "muon"
    assert second["parameter_norm"] == 8
    assert second["gradient_squared_share"] == pytest.approx(25 / 169)
    json.dumps(report, allow_nan=False)


def test_reuses_precomputed_norms_without_reducing_gradients_again(monkeypatch):
    import startrain.gradient_diagnostics as diagnostics

    model = RoutedModel()
    optimizer = build_optimizer(model)
    norms = {
        id(model.matrix): torch.tensor(5.0),
        id(model.scale): torch.tensor(12.0),
        id(model.bias): torch.tensor(0.0),
    }
    seen = []
    original = diagnostics._norms

    def record(values):
        seen.extend(id(value) for value in values)
        return original(values)

    monkeypatch.setattr(diagnostics, "_norms", record)
    snapshot = collect_gradient_diagnostics(
        model, batch(), optimizer, parameter_norms=norms, total_norm=torch.tensor(13.0)
    )
    for value in norms.values():
        value.zero_()
    assert not set(seen) & {id(parameter.grad) for parameter in model.parameters()}
    report = snapshot.to_host()
    assert report["reused_gradient_norms"] == 3
    assert report["pre_clip_global_norm"] == 13
    assert report["parameters"][0]["pre_clip_gradient_norm"] == 12


@pytest.mark.parametrize("bad_norm", (float("nan"), float("inf")))
def test_nonfinite_attribution_is_json_safe_and_visible(bad_norm):
    model = RoutedModel()
    model.scale.grad[0] = bad_norm
    report = collect_gradient_diagnostics(
        model, batch(), build_optimizer(model)
    ).to_host()
    assert report["pre_clip_global_norm"] is None
    assert report["global_norm_finite"] is False
    assert report["nonfinite_gradient_tensors"] == 1
    assert report["parameters"][0]["name"] == "scale"
    assert report["parameters"][0]["pre_clip_gradient_norm"] is None
    assert report["parameters"][0]["gradient_squared_share"] is None
    json.dumps(report, allow_nan=False)


def test_zero_and_missing_gradients_have_no_fabricated_contributions():
    model = RoutedModel()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.zero_()
    report = collect_gradient_diagnostics(
        model, batch(), build_optimizer(model), top_n=2
    ).to_host()
    assert report["pre_clip_global_norm"] == 0
    assert report["missing_gradient_tensors"] == 1
    assert report["zero_gradient_tensors"] == 3
    assert len(report["parameters"]) == 2
    assert all(value["gradient_squared_share"] == 0 for value in report["parameters"])
    for parameter in model.parameters():
        parameter.grad = None
    report = collect_gradient_diagnostics(
        model, batch(), build_optimizer(model)
    ).to_host()
    assert report["parameters"] == [] and report["pre_clip_global_norm"] == 0


def test_sparse_gradients_and_external_optimizer_routes_are_supported():
    model = nn.Embedding(5, 2, sparse=True)
    model(torch.tensor([1, 1, 2])).sum().backward()
    report = collect_gradient_diagnostics(
        model, batch(), torch.optim.SGD(model.parameters(), lr=0.1)
    ).to_host()
    assert report["pre_clip_global_norm"] == pytest.approx(10**0.5)
    assert report["parameters"][0]["optimizer_group"] == "group_0"
    assert report["parameters"][0]["algorithm"] == "sgd"
    assert model.weight.grad.is_sparse


@pytest.mark.parametrize("prefer_native", (False, True))
def test_exact_six_modes_survive_collation_and_transfers(prefer_native, monkeypatch):
    if prefer_native:
        pytest.importorskip("star_native")
    variants = (
        ("classic", 1, False),
        ("double", 1, False),
        ("classic", 1, True),
        ("double", 1, True),
        ("classic", 9, False),
        ("double", 3, False),
    )
    samples = [
        variant_sample(*variant, ring=4 if index < 3 else 10)
        for index, variant in enumerate(variants)
    ]
    source = collate_replay_samples(samples, prefer_native=prefer_native)
    expected = tuple(sample.variant_label for sample in samples)
    assert source.variant_labels == expected
    assert source.to("cpu", feature_dtype=torch.bfloat16).variant_labels == expected
    # Pinning needs a CUDA runtime; mock storage transfer, not metadata behavior.
    monkeypatch.setattr(type(source.inputs), "pin_memory", lambda self, **kwargs: self)
    monkeypatch.setattr(type(source.targets), "pin_memory", lambda self: self)
    assert source.pin_memory().variant_labels == expected
    model = RoutedModel()
    result = collect_gradient_diagnostics(
        model, source, build_optimizer(model)
    ).to_host()["batch"]
    assert result["rings"] == {"4": 3, "10": 3}
    assert result["modes"] == {"classic": 3, "double": 3}
    assert all(value == 1 for value in result["six_modes"].values())
    assert result["six_mode_counts_exact"] is True
    assert result["six_mode_unknown"] == {}
    assert result["handicap_severity"] == {"1": 4, "3": 1, "9": 1}
    assert result["label_availability"] == {
        name: 6
        for name in (
            "policy",
            "outcome",
            "score_margin",
            "ownership",
            "alive",
            "soft_policy",
        )
    }


def test_resolved_pie_metadata_is_not_inferred_as_standard():
    # Both have identical relevant global features once the swap window closes.
    standard = sample_for(4)
    pie = replace(standard, pie=True)
    source = collate_replay_samples([standard, pie], prefer_native=False)
    # Explicit original metadata distinguishes them even without a pending flag.
    model = RoutedModel()
    exact = collect_gradient_diagnostics(
        model, source, build_optimizer(model)
    ).to_host()["batch"]
    assert exact["six_modes"]["double"] == exact["six_modes"]["pie-double"] == 1
    manual = ReplayBatch(source.inputs, source.targets)
    unknown = collect_gradient_diagnostics(
        model, manual, build_optimizer(model)
    ).to_host()["batch"]
    assert unknown["six_mode_counts_exact"] is False
    assert unknown["six_mode_unknown"] == {"standard_or_resolved_pie-double": 2}
    assert unknown["modes"] == {"double": 2}


def test_manual_batch_labels_and_teacher_availability_are_explicit():
    source = batch()
    targets = replace(
        source.targets,
        teacher_mask=torch.tensor([True, False]),
        policy_mask=torch.tensor([False, True]),
        sample_weight=torch.tensor([0.5, 0.0]),
        policy_weight=torch.tensor([0.25, 1.0]),
    )
    manual = ReplayBatch(source.inputs, targets)
    model = RoutedModel()
    result = collect_gradient_diagnostics(
        model, manual, build_optimizer(model), variant_labels=("pie-double", "double")
    ).to_host()["batch"]
    assert result["teacher_samples"] == 1
    assert result["label_availability"]["policy"] == 1
    assert result["sample_weight_sum"] == result["teacher_sample_weight_sum"] == 0.5
    assert result["policy_sample_weight_sum"] == 0.125
    assert (
        result["positive_weight_samples"]
        == result["positive_policy_weight_samples"]
        == 1
    )


@pytest.mark.parametrize(
    "labels", (("double",), ("bad", "double"), ("handicap-10-double", "double"))
)
def test_invalid_variant_metadata_is_rejected(labels):
    model = RoutedModel()
    with pytest.raises(ValueError, match="variant"):
        collect_gradient_diagnostics(
            model, batch(), build_optimizer(model), variant_labels=labels
        )


@pytest.mark.parametrize("top_n", (0, 129, True))
def test_invalid_top_n_is_rejected(top_n):
    model = RoutedModel()
    with pytest.raises(ValueError, match="top_n"):
        collect_gradient_diagnostics(
            model, batch(), build_optimizer(model), top_n=top_n
        )


def test_unknown_feature_schema_does_not_invent_modes():
    source = batch()
    source = ReplayBatch(
        replace(source.inputs, global_features=source.inputs.global_features[:, :17]),
        source.targets,
    )
    model = RoutedModel()
    result = collect_gradient_diagnostics(
        model, source, build_optimizer(model)
    ).to_host()["batch"]
    assert result["variant_source"] == "unavailable"
    assert result["six_mode_unknown"] == {"unsupported_feature_schema": 2}
    assert result["modes"] == {}
