"""Plan boundaries for the variable request sizes observed in production."""

import torch
from torch import nn

from startrain.inference import GraphInferenceAdapter, InferenceConfig


def _planner(*, graphs: bool, device: str) -> GraphInferenceAdapter:
    # Planning is host-only; no CUDA context is needed for these boundary tests.
    adapter = GraphInferenceAdapter(
        nn.Identity(), config=InferenceConfig(cuda_graphs=graphs)
    )
    adapter.device = torch.device(device)
    return adapter


def test_graph_plans_reduce_measured_tail_padding_with_bounded_shapes():
    adapter = _planner(graphs=True, device="cuda")
    assert [
        adapter._inference_batch_rows(rows)
        for rows in (65, 90, 96, 97, 128, 129, 135, 180, 192, 193, 256)
    ] == [96, 96, 96, 128, 128, 192, 192, 192, 192, 256, 256]
    plans = {adapter._inference_batch_rows(rows) for rows in range(1, 257)}
    assert plans == {1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256}
    for rows in range(65, 257):
        assert rows <= adapter._inference_batch_rows(rows) < rows * 1.5


def test_disabled_graphs_preserve_existing_cuda_plans_and_cpu_never_pads():
    legacy = _planner(graphs=False, device="cuda")
    cpu = _planner(graphs=True, device="cpu")
    for rows in range(1, 257):
        assert legacy._inference_batch_rows(rows) == 1 << (rows - 1).bit_length()
        assert cpu._inference_batch_rows(rows) == rows
