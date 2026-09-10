import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from scripts import benchmark_training_shared_geometry as benchmark
from startrain.config import load_config
from startrain.model import model_parameter_count


def test_plan_import_does_not_load_torch_or_initialize_devices():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from scripts import benchmark_training_shared_geometry as b; assert 'torch' not in sys.modules; b.main(['--config','/nonexistent/offline.yaml'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(process.stdout)
    assert plan["execute"] is False
    assert plan["adoptable"] is False
    assert len(plan["cases"]) == 10


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--accuracy-batch-size", "33"),
        ("--timing-batch-sizes", "513"),
        ("--max-memory-gib", "77"),
        ("--max-memory-gib", "nan"),
        ("--case-timeout-seconds", "901"),
        ("--timeout-seconds", "3601"),
        ("--repeats", "1"),
        ("--rings", "5"),
    ],
)
def test_resource_and_shape_bounds(flag, value):
    args = benchmark.parser().parse_args(["--config", "offline.yaml", flag, value])
    with pytest.raises(ValueError):
        benchmark.validate(args)


@pytest.mark.parametrize(
    "stdout,verified",
    [
        ("GPU-example, 123\n", True),
        ("GPU-example, 123\nGPU-example, 456\n", False),
        ("", False),
        ("GPU-other, 123\n", False),
    ],
)
def test_exclusivity_requires_verified_current_process_only(stdout, verified):
    completed = subprocess.CompletedProcess([], 0, stdout, "")
    with patch.object(benchmark.subprocess, "run", return_value=completed) as run:
        result = benchmark.gpu_ownership("GPU-example", own_pid=123)
    assert result["verified"] is verified
    assert "--query-compute-apps=gpu_uuid,pid" in run.call_args.args[0]
    assert run.call_args.kwargs["timeout"] == 5


def test_unknown_process_inventory_fails_closed():
    with patch.object(benchmark.subprocess, "run", side_effect=FileNotFoundError):
        assert benchmark.gpu_ownership("GPU-example")["verified"] is False


def test_tensor_error_metrics_match_independent_expected_math():
    result = benchmark.compare_tensors(
        {"x": torch.tensor([3.0, 0.0])}, {"x": torch.tensor([0.0, 4.0])}
    )
    assert result["relative_l2"] == pytest.approx(5 / 4)
    assert result["cosine"] == 0
    assert result["max_absolute"] == 4
    assert result["worst_tensor"] == "x"
    bad = benchmark.compare_tensors(
        {"x": torch.tensor([float("inf")])}, {"x": torch.zeros(1)}
    )
    assert bad["finite"] is False


def test_numerical_gate_rejects_broken_bias_even_with_exact_trunk():
    comparisons = {
        f"{label}_{kind}": {"finite": True, "relative_l2": 0.0, "cosine": 1.0}
        for label in ("fp32_shared", "bf16_shared", "bf16_oracle", "bf16_shared_oracle")
        for kind in ("outputs", "gradients", "bias_gradients")
    }
    assert benchmark.numerical_gate(comparisons)["passed"]
    comparisons["bf16_shared_bias_gradients"]["relative_l2"] = 0.051
    gate = benchmark.numerical_gate(comparisons)
    assert not gate["passed"]
    assert gate["failures"] == ["bf16_shared_bias_gradients"]


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
def test_synthetic_fixture_is_deterministic_validated_and_varied(ring):
    first = benchmark.make_batch(ring, 6, 42)
    second = benchmark.make_batch(ring, 6, 42)
    assert first.homogeneous_ring == ring
    assert len(set(first.variant_labels)) == 6
    for a, b in zip(first.inputs.model_args(), second.inputs.model_args(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(first.inputs.node_features[0], first.inputs.node_features[1])
    torch.testing.assert_close(first.targets.policy.sum(dim=1), torch.ones(6))


def test_config_fixture_is_actual_production_architecture():
    config = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    assert (
        config.model.width,
        config.model.rrt_groups,
        config.model.attention_heads,
        config.model.kv_heads,
    ) == (384, 8, 12, 3)
    assert model_parameter_count(config.model) == 17_402_775


def test_owned_child_timeout_is_enforced():
    import os

    with pytest.raises(subprocess.TimeoutExpired):
        benchmark._run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=dict(os.environ),
            timeout=0.1,
        )


def test_manifest_loads_verified_ema_and_rejects_checkpoint_drift(tmp_path):
    from dataclasses import replace
    from startrain.checkpoint import ExponentialMovingAverage
    from startrain.learner import ImmutableModelPublisher
    from startrain.model import GraphResTNet, ModelConfig
    from startrain.runtime import RunIdentity

    config = replace(
        load_config(Path(__file__).parents[1] / "configs/small.yaml"),
        model=ModelConfig(width=16, rrt_groups=2, attention_heads=4, kv_heads=1),
    )
    model = GraphResTNet(config.model)
    ema = ExponentialMovingAverage(model)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    manifest = ImmutableModelPublisher(
        tmp_path / "learner", RunIdentity(tmp_path / "run.json", "benchmark", "test", 1)
    ).publish(
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        ema=ema,
        step=7,
        epoch=0,
        config=config.as_dict(),
    )
    restored = GraphResTNet(config.model)
    metadata, identity = benchmark.load_weights(manifest.path, restored, config)
    assert metadata["step"] == 7
    assert identity["manifest_sha256"] == manifest.manifest_sha256
    assert identity["checkpoint_sha256"] == manifest.checkpoint_sha256
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, ema.shadow[name], rtol=0, atol=0)
    assert not torch.equal(next(model.parameters()), next(restored.parameters()))
    with patch(
        "startrain.checkpoint.load_model_manifest",
        return_value=replace(manifest, model_step=8),
    ):
        with pytest.raises(ValueError, match="step disagree"):
            benchmark.load_weights(manifest.path, restored, config)
    wrong_config = replace(config, model=replace(config.model, width=32))
    with pytest.raises(ValueError, match="model"):
        benchmark.load_weights(
            manifest.path, GraphResTNet(wrong_config.model), wrong_config
        )
    with manifest.checkpoint.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        benchmark.load_weights(manifest.path, restored, config)


@pytest.mark.parametrize(
    "load_context,adoptable",
    [("isolated", True), ("shared", False), ("unverified", False)],
)
def test_cutover_controller_one_owned_case_and_shared_results_never_adoptable(
    tmp_path, capsys, load_context, adoptable
):
    timing = [
        {
            "kind": "timing",
            "ring": 10,
            "batch_size": size,
            "shared": shared,
            "status": "ok",
            "median_step_seconds": 0.8 if shared else 1.0,
            "peak_allocated_bytes": 80 if shared else 100,
        }
        for size in (128, 512)
        for shared in (False, True)
    ]
    child = {
        "kind": "cutover",
        "ring": 10,
        "batch_size": 16,
        "status": "ok",
        "numerical_gate": {"passed": True},
        "timing_results": timing,
        "isolation_verified": True,
        "checkpoint_sha256": "checkpoint",
        "config_sha256": "config",
        "parameter_count": 17_402_775,
        "compile_settings": {
            "dynamic": False,
            "fullgraph": True,
            "isolate_recompiles": True,
        },
        "measured_math": {
            "float32_matmul_precision": "high",
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
        },
    }
    output = tmp_path / "report.json"
    with patch.object(
        benchmark,
        "_run_process",
        return_value=subprocess.CompletedProcess([], 0, json.dumps(child), ""),
    ) as run:
        status = benchmark.main(
            [
                "--config",
                str(tmp_path / "profile.yaml"),
                "--checkpoint",
                str(tmp_path / "manifest.json"),
                "--execute",
                "--cutover",
                "--compile",
                "--rings",
                "10",
                "--timeout-seconds",
                "540",
                "--load-context",
                load_context,
                "--output",
                str(output),
            ]
        )
    assert status == 0
    assert run.call_count == 1
    assert 0 < run.call_args.kwargs["timeout"] <= 535
    command = run.call_args.args[0]
    assert json.loads(command[command.index("--child-case") + 1])["kind"] == "cutover"
    report = json.loads(output.read_text())
    assert report["adoptable"] is adoptable
    assert len(report["comparisons"]) == 2
    assert all(row["speedup"] == 1.25 for row in report["comparisons"])
    capsys.readouterr()


def test_cutover_requires_one_ring_and_whole_run_deadline():
    for extra in ([], ["--rings", "10", "--timeout-seconds", "601"]):
        args = benchmark.parser().parse_args(
            ["--config", "offline", "--cutover", "--compile", *extra]
        )
        with pytest.raises(ValueError, match="cutover"):
            benchmark.validate(args)
    assert not benchmark.numerical_gate({})["passed"]


def test_cutover_timeout_retains_completed_accuracy_without_adoption(tmp_path, capsys):
    def timeout_after_accuracy(command, *, env, timeout):
        progress = Path(command[command.index("--child-progress") + 1])
        progress.write_text(
            json.dumps({"numerical_gate": {"passed": True}, "timing_results": []})
        )
        raise subprocess.TimeoutExpired(command, timeout)

    output = tmp_path / "report.json"
    with patch.object(benchmark, "_run_process", side_effect=timeout_after_accuracy):
        status = benchmark.main(
            [
                "--config",
                "offline.yaml",
                "--checkpoint",
                "manifest.json",
                "--execute",
                "--cutover",
                "--compile",
                "--rings",
                "10",
                "--max-memory-gib",
                "75",
                "--output",
                str(output),
            ]
        )
    report = json.loads(output.read_text())
    assert status == 1
    assert report["results"][0]["numerical_gate"]["passed"]
    assert report["results"][0]["status"] == "case-timeout"
    assert not report["adoptable"]
    capsys.readouterr()


def test_oracle_disables_tf32_and_measured_math_uses_production_flags():
    previous = benchmark.math_settings()
    try:
        oracle = benchmark.configure_math(oracle=True, device="cuda:2")
        assert oracle == {
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        with patch(
            "startrain.device.enable_fast_math",
            wraps=__import__(
                "startrain.device", fromlist=["enable_fast_math"]
            ).enable_fast_math,
        ) as fast_math:
            measured = benchmark.configure_math(oracle=False, device="cuda:2")
        fast_math.assert_called_once_with("cuda:2")
        assert measured == {
            "float32_matmul_precision": "high",
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
        }
    finally:
        torch.set_float32_matmul_precision(previous["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = previous["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]


def test_compile_matches_static_learner_with_room_for_both_arms():
    args = benchmark.parser().parse_args(
        ["--config", "offline.yaml", "--compile", "--device", "cpu"]
    )
    with patch(
        "startrain.training.maybe_compile_model",
        side_effect=lambda model, **kwargs: model,
    ) as compile_model:
        model, runner = benchmark._execution(args, torch.nn.Linear(4, 4))
    assert runner is model
    assert compile_model.call_args.kwargs == {
        "enabled": True,
        "dynamic": False,
        "fullgraph": True,
        "isolate_recompiles": True,
        "recompile_limit": 6,
    }


@pytest.mark.parametrize("incorrect", ["dynamic", "tf32", "precision", "isolation"])
def test_nonproduction_execution_is_not_eligible(incorrect):
    result = {
        "compile_settings": {
            "dynamic": False,
            "fullgraph": True,
            "isolate_recompiles": True,
        },
        "measured_math": {
            "float32_matmul_precision": "high",
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
        },
    }
    assert benchmark.production_execution_matches(result)
    if incorrect == "dynamic":
        result["compile_settings"]["dynamic"] = True
    elif incorrect == "tf32":
        result["measured_math"]["cuda_matmul_allow_tf32"] = False
    elif incorrect == "precision":
        result["measured_math"]["float32_matmul_precision"] = "highest"
    else:
        result["compile_settings"]["isolate_recompiles"] = False
    assert not benchmark.production_execution_matches(result)
