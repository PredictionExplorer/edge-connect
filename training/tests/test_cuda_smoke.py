from __future__ import annotations

import torch

import pytest
from startrain.arena import ArenaRunner
from startrain.config import ArenaConfig
from startrain.features import DoubleStarPosition, encode_batch
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.model import GraphResTNet, ModelConfig
from startrain.native import load_star_native
from startrain.topology import get_topology
from startrain.training import maybe_compile_model, unwrap_model


def _batch(batch_size: int = 4, *, rings: int = 4):
    topology = get_topology(rings)
    position = DoubleStarPosition(
        rings=rings,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
    )
    return encode_batch([position] * batch_size).to("cuda")


def _model() -> GraphResTNet:
    return GraphResTNet(
        ModelConfig(
            width=32,
            rrt_groups=1,
            attention_heads=4,
            kv_heads=1,
        )
    ).cuda()


@pytest.mark.cuda
@pytest.mark.timeout(600)
def test_cuda_bf16_compiled_forward_backward_is_finite() -> None:
    model = _model()
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    compiled = maybe_compile_model(
        model,
        enabled=True,
        dynamic=False,
        fullgraph=True,
        backend="inductor",
        recompile_limit=10,
        isolate_recompiles=True,
    )
    assert unwrap_model(compiled) is model
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    for rings in (4, 6, 8, 10, 10, 8, 6, 4):
        model.zero_grad(set_to_none=True)
        batch = _batch(batch_size=2, rings=rings)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = compiled(*batch.model_args())
            loss = (
                output.outcome_logits.float().square().mean()
                + output.score_margin_logits.float().square().mean()
                + output.ownership_logits.float().square().mean()
            )
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )


@pytest.mark.cuda
@pytest.mark.native
@pytest.mark.timeout(600)
def test_cuda_dual_compiled_arena_first_wave_completes() -> None:
    native = load_star_native(required=True)
    assert native is not None
    evaluators = []
    for version in ("candidate", "baseline"):
        model = _model().eval()
        compiled = maybe_compile_model(
            model,
            enabled=True,
            dynamic=True,
            fullgraph=True,
            backend="inductor",
        )
        evaluators.append(
            GraphInferenceAdapter(
                compiled,
                device="cuda",
                config=InferenceConfig(precision="bf16"),
                model_version=version,
            )
        )
    progress: list[dict[str, object]] = []

    result = ArenaRunner(
        native_module=native,
        candidate=evaluators[0],
        baseline=evaluators[1],
        config=ArenaConfig(
            rings=(4,),
            pairs_per_ring=2,
            simulations=2,
            max_considered=2,
            minimum_pairs_per_ring=2,
            max_pairs_per_ring=2,
            bootstrap_samples=200,
            regression_floor_elo=-2_500,
        ),
    ).run(progress=lambda **details: progress.append(details))

    assert len(result["pairs"]) == 2
    assert result["evaluation_metrics"]["serialized_inference_calls"] > 0
    assert result["evaluation_metrics"]["total_evaluator_rows"] > 0
    assert result["search"]["search_workers"] == 2
    assert result["search"]["inference_workers"] == 1
    assert any(item.get("completed_pairs") == 2 for item in progress)


@pytest.mark.cuda
@pytest.mark.soak
def test_cuda_repeated_inference_stays_finite_and_memory_bounded() -> None:
    model = _model().eval()
    batch = _batch(batch_size=16)
    torch.cuda.reset_peak_memory_stats()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        for _ in range(500):
            output = model(*batch.model_args())
            assert bool(torch.isfinite(output.outcome_logits).all())
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    allocated = torch.cuda.memory_allocated()
    assert allocated <= peak
    assert peak < 2 * 1024**3
