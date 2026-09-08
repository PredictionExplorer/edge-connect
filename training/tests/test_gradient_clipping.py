from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from torch import nn

from startrain.gradient_clipping import GradientClipper, GradientClippingConfig


def parameters(*, dtype=torch.float64):
    return {
        "first": nn.Parameter(torch.zeros(2, dtype=dtype)),
        "second": nn.Parameter(torch.zeros(3, dtype=dtype)),
    }


def assign(values, gradients):
    for name, parameter in values.items():
        gradient = gradients.get(name)
        parameter.grad = (
            None
            if gradient is None
            else torch.tensor(gradient, device=parameter.device, dtype=parameter.dtype)
        )


def assert_state_equal(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        if key == "ema_norms":
            assert set(actual[key]) == set(expected[key])
            for name in actual[key]:
                torch.testing.assert_close(
                    actual[key][name], expected[key][name], rtol=0, atol=0
                )
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("scale", [0.0, 0.01, 1.0, 17.0])
def test_global_clipping_matches_pytorch_bitwise(dtype, scale):
    actual = parameters(dtype=dtype)
    expected = parameters(dtype=dtype)
    gradients = {
        "first": [3 * scale, 4 * scale],
        "second": [-2 * scale, 1 * scale, 3 * scale],
    }
    assign(actual, gradients)
    assign(expected, gradients)
    clipper = GradientClipper(actual.items(), max_norm=1.0)
    measurement = clipper.measure()
    reference = torch.nn.utils.clip_grad_norm_(expected.values(), 1.0)
    result = clipper.apply_(measurement)
    torch.testing.assert_close(result.pre_clip_norm, reference, rtol=0, atol=0)
    for name in actual:
        torch.testing.assert_close(
            actual[name].grad, expected[name].grad, rtol=0, atol=0
        )
        assert id(actual[name]) in measurement.parameter_norms
    post = torch.nn.utils.get_total_norm(
        [parameter.grad for parameter in expected.values()]
    )
    torch.testing.assert_close(result.post_clip_norm, post, rtol=0, atol=0)
    assert result.mode == "global" and result.warmup is False
    assert clipper.state_dict()["ema_norms"] == {}


def test_global_mixed_dtype_grouping_matches_pytorch_bitwise():
    generator = torch.Generator().manual_seed(881)
    dtypes = (
        torch.float64,
        torch.float32,
        torch.float16,
        torch.float64,
        torch.bfloat16,
    )
    actual = {
        str(index): nn.Parameter(torch.zeros(37 + index, dtype=dtype))
        for index, dtype in enumerate(dtypes)
    }
    expected = {
        name: nn.Parameter(torch.zeros_like(parameter))
        for name, parameter in actual.items()
    }
    for name, parameter in actual.items():
        parameter.grad = torch.randn(
            parameter.shape, dtype=parameter.dtype, generator=generator
        )
        expected[name].grad = parameter.grad.clone()
    clipper = GradientClipper(actual.items(), max_norm=1.25)
    measurement = clipper.measure()
    total = torch.nn.utils.clip_grad_norm_(expected.values(), 1.25)
    clipper.apply_(measurement)
    torch.testing.assert_close(measurement.total_norm, total, rtol=0, atol=0)
    for name in actual:
        torch.testing.assert_close(
            actual[name].grad, expected[name].grad, rtol=0, atol=0
        )


def test_adagc_matches_reference_minimum_warmup_and_post_clip_ema():
    actual = parameters()
    expected = parameters()
    config = GradientClippingConfig(
        mode="adagc", beta=0.9, multiplier=1.04, warmup_steps=3
    )
    clipper = GradientClipper(actual.items(), config=config, max_norm=10.0)
    history = {}
    sequence = [
        {"first": [3, 4], "second": [0, 0, 12]},
        {"first": [1, 0], "second": [1, 0, 0]},
        {"first": [8, 0], "second": [8, 0, 0]},
        {"first": [100, 0], "second": [1, 0, 0]},
        {"first": [1.005, 0], "second": [0.98, 0, 0]},
        {"first": [8, 0], "second": [20, 0, 0]},
    ]
    for step, gradients in enumerate(sequence):
        assign(actual, gradients)
        assign(expected, gradients)
        if step < config.warmup_steps:
            torch.nn.utils.clip_grad_norm_(expected.values(), 10.0)
        else:
            for name, parameter in expected.items():
                norm = torch.linalg.vector_norm(parameter.grad)
                coefficient = torch.clamp(
                    config.multiplier * history[name] / norm, max=1.0
                )
                parameter.grad.mul_(coefficient)
        for name, parameter in expected.items():
            norm = torch.linalg.vector_norm(parameter.grad)
            if name not in history:
                history[name] = norm
            elif step < config.warmup_steps:
                history[name] = torch.minimum(history[name], norm)
            else:
                history[name] = config.beta * history[name] + (1.0 - config.beta) * norm
        result = clipper.apply_(clipper.measure())
        assert result.warmup is (step < config.warmup_steps)
        assert result.steps == step + 1
        for name in actual:
            torch.testing.assert_close(
                actual[name].grad, expected[name].grad, rtol=0, atol=0
            )
            torch.testing.assert_close(
                clipper.state_dict()["ema_norms"][name], history[name], rtol=0, atol=0
            )
    assert clipper.state_dict()["ema_norms"]["first"] < 1.1


def test_gradual_increase_follows_history_while_spike_is_bounded():
    parameter = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    clipper = GradientClipper(
        [("weight", parameter)],
        config=GradientClippingConfig(mode="adagc", warmup_steps=1),
        max_norm=100.0,
    )
    for index in range(300):
        parameter.grad = torch.tensor([1.0 + index * 0.0001], dtype=torch.float64)
        result = clipper.apply_(clipper.measure())
        assert result.coefficients["weight"] == 1
    previous = clipper.state_dict()["ema_norms"]["weight"].clone()
    parameter.grad.fill_(1_000_000)
    result = clipper.apply_(clipper.measure())
    torch.testing.assert_close(parameter.grad[0], previous * 1.04, rtol=0, atol=0)
    expected_history = previous * 0.99 + parameter.grad[0] * 0.01
    torch.testing.assert_close(
        clipper.state_dict()["ema_norms"]["weight"], expected_history
    )
    assert result.parameter_clip_fraction == 1
    assert result.minimum_coefficient < 0.00001


def test_zero_and_missing_gradients_preserve_history_and_late_activation_bootstraps():
    values = parameters()
    clipper = GradientClipper(
        values.items(),
        config=GradientClippingConfig(mode="adagc", warmup_steps=2),
        max_norm=1.0,
    )
    for _ in range(4):
        assign(values, {"first": [0.0, 0.0]})
        result = clipper.apply_(clipper.measure())
        assert result.post_clip_norm == 0
        assert result.parameter_clip_fraction == 0
    assert clipper.state_dict()["ema_norms"]["second"] == 0
    assign(values, {"first": [3, 4], "second": [0, 0, 12]})
    result = clipper.apply_(clipper.measure())
    assert result.post_clip_norm <= 1.0
    positive = clipper.state_dict()
    assert all(value > 0 for value in positive["ema_norms"].values())
    assign(values, {"first": [0, 0]})
    clipper.apply_(clipper.measure())
    for name, value in positive["ema_norms"].items():
        torch.testing.assert_close(
            clipper.state_dict()["ema_norms"][name], value, rtol=0, atol=0
        )


@pytest.mark.parametrize("mode", ["global", "adagc"])
def test_no_gradients_and_empty_parameter_sets(mode):
    for values in ({}, parameters()):
        clipper = GradientClipper(
            values.items(), config=GradientClippingConfig(mode=mode)
        )
        result = clipper.apply_(clipper.measure())
        assert result.pre_clip_norm == result.post_clip_norm == 0
        assert result.coefficients == result.clipped == {}
        assert result.parameter_clip_fraction == 0
        assert result.minimum_coefficient == result.median_coefficient == 1


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("mode", ["global", "adagc"])
def test_nonfinite_rejects_step_without_mutating_any_gradient_or_history(
    nonfinite, mode
):
    values = parameters()
    clipper = GradientClipper(
        values.items(), config=GradientClippingConfig(mode=mode, warmup_steps=1)
    )
    assign(values, {"first": [1, 0], "second": [1, 0, 0]})
    clipper.apply_(clipper.measure())
    saved = clipper.state_dict()
    assign(values, {"first": [7, 8], "second": [1, nonfinite, 0]})
    gradients = {name: parameter.grad.clone() for name, parameter in values.items()}
    with pytest.raises(ValueError, match="nonfinite"):
        clipper.apply_(clipper.measure())
    assert_state_equal(clipper.state_dict(), saved)
    for name, parameter in values.items():
        torch.testing.assert_close(
            parameter.grad, gradients[name], rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("split", [1, 3, 7])
def test_state_checkpoint_resumes_exactly_during_and_after_warmup(tmp_path, split):
    values = parameters(dtype=torch.float32)
    config = GradientClippingConfig(mode="adagc", warmup_steps=3)
    uninterrupted = GradientClipper(values.items(), config=config, max_norm=2.0)
    generator = torch.Generator().manual_seed(41)
    sequence = [
        {
            name: torch.randn(parameter.shape, generator=generator).tolist()
            for name, parameter in values.items()
        }
        for _ in range(10)
    ]
    for gradients in sequence[:split]:
        assign(values, gradients)
        uninterrupted.apply_(uninterrupted.measure())
    path = tmp_path / "clipping.pt"
    torch.save(uninterrupted.state_dict(), path)
    resumed_values = parameters(dtype=torch.float32)
    resumed = GradientClipper(resumed_values.items(), config=config, max_norm=2.0)
    payload = torch.load(path, weights_only=True)
    before = resumed.state_dict()
    resumed.validate_state_dict(payload)
    assert_state_equal(resumed.state_dict(), before)
    resumed.load_state_dict(payload)
    for gradients in sequence[split:]:
        assign(values, gradients)
        assign(resumed_values, gradients)
        expected = uninterrupted.apply_(uninterrupted.measure())
        actual = resumed.apply_(resumed.measure())
        torch.testing.assert_close(
            actual.post_clip_norm, expected.post_clip_norm, rtol=0, atol=0
        )
        for name in values:
            torch.testing.assert_close(
                resumed_values[name].grad, values[name].grad, rtol=0, atol=0
            )
        assert_state_equal(resumed.state_dict(), uninterrupted.state_dict())


@pytest.mark.parametrize(
    "corruption",
    [
        "name",
        "shape",
        "shape_bool",
        "dtype",
        "beta",
        "mode",
        "warmup",
        "max_norm",
        "updates",
        "version",
        "unknown",
        "nan",
        "negative",
        "nonscalar",
        "wrong_norm_dtype",
        "extra_name",
        "missing_name",
    ],
)
def test_malformed_state_validation_is_transactional(corruption):
    values = parameters()
    clipper = GradientClipper(
        values.items(), config=GradientClippingConfig(mode="adagc", warmup_steps=1)
    )
    assign(values, {"first": [1, 0], "second": [1, 0, 0]})
    clipper.apply_(clipper.measure())
    before = clipper.state_dict()
    bad = deepcopy(before)
    if corruption in ("name", "shape", "dtype"):
        bad["parameters"][0][corruption] = {
            "name": "renamed",
            "shape": [3],
            "dtype": "torch.float32",
        }[corruption]
    elif corruption == "shape_bool":
        bad["parameters"][0]["shape"] = [True, True]
    elif corruption in ("beta", "mode", "warmup"):
        bad["config"]["warmup_steps" if corruption == "warmup" else corruption] = {
            "beta": 0.5,
            "mode": "global",
            "warmup": 2,
        }[corruption]
    elif corruption in ("max_norm", "updates", "version"):
        bad[corruption] = {"max_norm": 2.0, "updates": -1, "version": True}[corruption]
    elif corruption == "unknown":
        bad["unknown"] = 1
    elif corruption == "extra_name":
        bad["ema_norms"]["unknown"] = torch.tensor(1.0)
    elif corruption == "missing_name":
        del bad["ema_norms"]["second"]
    else:
        bad["ema_norms"]["second"] = {
            "nan": torch.tensor(float("nan"), dtype=torch.float64),
            "negative": torch.tensor(-1.0, dtype=torch.float64),
            "nonscalar": torch.tensor([1.0], dtype=torch.float64),
            "wrong_norm_dtype": torch.tensor(1.0, dtype=torch.float32),
        }[corruption]
    for method in (clipper.validate_state_dict, clipper.load_state_dict):
        with pytest.raises(ValueError):
            method(bad)
        assert_state_equal(clipper.state_dict(), before)


def test_parameter_identity_and_measurement_lifetime_are_checked_before_mutation():
    parameter = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="unique Parameters"):
        GradientClipper([("one", parameter), ("two", parameter)])
    with pytest.raises(ValueError, match="unique"):
        GradientClipper([("one", parameter), ("one", nn.Parameter(torch.zeros(1)))])
    clipper = GradientClipper([("one", parameter)])
    parameter.grad = torch.ones_like(parameter)
    measured = clipper.measure()
    parameter.grad.mul_(2)
    with pytest.raises(ValueError, match="changed"):
        clipper.apply_(measured)
    measured = clipper.measure()
    clipper.apply_(measured)
    with pytest.raises(ValueError, match="stale"):
        clipper.apply_(measured)
    clipper.reset()
    assert clipper.steps == 0 and clipper.state_dict()["ema_norms"] == {}
    with pytest.raises(ValueError, match="belongs"):
        clipper.apply_(measured)


@pytest.mark.parametrize(
    "values",
    [
        {"beta": True},
        {"beta": -0.1},
        {"beta": 1.0},
        {"beta": float("nan")},
        {"multiplier": float("inf")},
        {"multiplier": 0.9},
        {"warmup_steps": 0},
        {"warmup_steps": True},
        {"warmup_steps": 1.0},
        {"mode": "adaptive"},
    ],
)
def test_invalid_config_is_rejected(values):
    with pytest.raises(ValueError):
        GradientClippingConfig(**values)


def test_default_config_is_explicit_legacy_global():
    assert asdict(GradientClippingConfig()) == {
        "mode": "global",
        "beta": 0.99,
        "multiplier": 1.04,
        "warmup_steps": 100,
    }


def test_inactive_parameters_do_not_retain_quadratic_history_storage():
    values = {str(index): nn.Parameter(torch.zeros(1)) for index in range(32)}
    clipper = GradientClipper(
        values.items(), config=GradientClippingConfig(mode="adagc", warmup_steps=1)
    )
    for turn in range(32):
        for index, parameter in enumerate(values.values()):
            parameter.grad = torch.ones_like(parameter) if index >= turn else None
        clipper.apply_(clipper.measure())
    storages = {
        value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
        for value in clipper._history.values()
    }
    assert sum(storages.values()) == len(values) * torch.tensor(0.0).element_size()


def test_collectively_validated_hot_path_does_not_read_device_scalars(monkeypatch):
    values = parameters(dtype=torch.float32)
    clipper = GradientClipper(
        values.items(), config=GradientClippingConfig(mode="adagc", warmup_steps=1)
    )

    def host_read(*args, **kwargs):
        raise AssertionError("clipping hot path must not read a device scalar")

    for name in ("__bool__", "item", "tolist", "cpu"):
        monkeypatch.setattr(torch.Tensor, name, host_read)
    for _ in range(3):
        assign(values, {"first": [30, 40], "second": [1, 0, 0]})
        clipper.apply_(clipper.measure(), finite_checked=True)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_adaptive_clipping_and_state_support_float_dtypes(dtype):
    values = parameters(dtype=dtype)
    clipper = GradientClipper(
        values.items(), config=GradientClippingConfig(mode="adagc", warmup_steps=1)
    )
    assign(values, {"first": [3, 4], "second": [1, 0, 0]})
    clipper.apply_(clipper.measure())
    assign(values, {"first": [30, 40], "second": [1, 0, 0]})
    result = clipper.apply_(clipper.measure())
    assert result.clipped["first"]
    assert torch.isfinite(result.post_clip_norm)
    saved = clipper.state_dict()
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert all(value.dtype == expected_dtype for value in saved["ema_norms"].values())
    clipper.validate_state_dict(saved)


@pytest.mark.cuda
def test_cuda_clipping_resumes_cpu_checkpoint_and_checks_mixed_devices(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cpu = nn.Parameter(torch.zeros(2))
    gpu = nn.Parameter(torch.zeros(3, device="cuda"))
    cpu.grad = torch.tensor([3.0, 4.0])
    gpu.grad = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    reference_cpu = nn.Parameter(torch.zeros_like(cpu))
    reference_gpu = nn.Parameter(torch.zeros_like(gpu))
    reference_cpu.grad = cpu.grad.clone()
    reference_gpu.grad = gpu.grad.clone()
    global_clipper = GradientClipper([("cpu", cpu), ("gpu", gpu)])
    measured = global_clipper.measure()
    expected = torch.nn.utils.clip_grad_norm_([reference_cpu, reference_gpu], 1.0)
    global_clipper.apply_(measured)
    torch.testing.assert_close(measured.total_norm, expected, rtol=0, atol=0)
    torch.testing.assert_close(cpu.grad, reference_cpu.grad, rtol=0, atol=0)
    torch.testing.assert_close(gpu.grad, reference_gpu.grad, rtol=0, atol=0)
    adaptive = GradientClipper(
        [("gpu", gpu)],
        config=GradientClippingConfig(mode="adagc", warmup_steps=1),
    )
    adaptive.apply_(adaptive.measure())
    checkpoint = tmp_path / "adaptive.pt"
    torch.save(adaptive.state_dict(), checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["ema_norms"]["gpu"].device.type == "cpu"
    adaptive.load_state_dict(payload)
    gpu.grad.fill_(10)
    result = adaptive.apply_(adaptive.measure(), finite_checked=True)
    assert result.post_clip_norm.device.type == "cuda"
    assert result.clipped["gpu"]
    gpu.grad = None
    empty = adaptive.measure()
    assert empty.total_norm.device.type == "cuda"
    empty_result = adaptive.apply_(empty, finite_checked=True)
    assert empty_result.post_clip_norm.device.type == "cuda"
