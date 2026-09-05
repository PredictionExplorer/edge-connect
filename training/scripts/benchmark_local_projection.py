#!/usr/bin/env python3
"""Exploratory paired BF16 local-block benchmark; no training state is read.

Run from the training directory with PYTHONPATH=. and its Python environment.
Both projection orders are implemented here independently of the production
forward method. This microbenchmark does not establish whole-actor throughput;
concurrent GPU load can reverse apparent wins, even with alternating pair order.

Compilation and warmup are outside timed samples and their cost is reported
separately. An external wall timeout bounds those operations as well, for example
on the Linux training host:

    PYTHONPATH=. timeout --signal=TERM --kill-after=10s 180s .venv/bin/python \\
        scripts/benchmark_local_projection.py --device cuda:7 --compile \\
        --load-context shared

Internal checks cap accumulated timed CUDA-event intervals at 45 seconds and
timed host intervals at 50 seconds; they cannot preempt a CUDA call or compilation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
import json
import statistics
import time

import torch
import torch.nn.functional as functional
from torch import Tensor

from startrain.model import LocalEdgeBlock
from startrain.topology import SUPPORTED_RINGS, get_topology


class OriginalLocalEdgeBlock(LocalEdgeBlock):
    project_before_gather = False

    def forward(
        self,
        inputs: Tensor,
        neighbor_index: Tensor,
        neighbor_mask: Tensor,
        neighbor_edge_type: Tensor,
        node_mask: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        normalized = self.norm(inputs)
        if self.modulation is not None and condition is not None:
            scale, shift = self.modulation(condition).unsqueeze(1).chunk(2, dim=-1)
            normalized = normalized * (1.0 + scale) + shift
        gather_source = (
            self.neighbor_projection(normalized)
            if self.project_before_gather
            else normalized
        )
        batch, nodes, width = gather_source.shape
        degree = neighbor_index.shape[-1]
        gather_index = neighbor_index.reshape(batch, nodes * degree)
        neighbors = gather_source.gather(
            1, gather_index.unsqueeze(-1).expand(-1, -1, width)
        ).reshape(batch, nodes, degree, width)
        messages = (
            neighbors
            if self.project_before_gather
            else self.neighbor_projection(neighbors)
        )
        messages = messages + self.edge_embedding(neighbor_edge_type)
        if self.source_gate_projection is not None:
            source_gate = self.source_gate_projection(normalized).unsqueeze(2)
            messages = functional.silu(messages) * torch.sigmoid(source_gate + messages)
        else:
            messages = functional.silu(messages)
        weights = neighbor_mask.unsqueeze(-1).to(dtype=messages.dtype)
        aggregated = (messages * weights).sum(dim=2)
        aggregated = aggregated / weights.sum(dim=2).clamp_min(1.0)
        update = self.update(self.self_projection(normalized) + aggregated)
        output = inputs + self.dropout(update) * self.layer_scale
        return output * node_mask.unsqueeze(-1).to(dtype=output.dtype)


class ProjectionFirstLocalEdgeBlock(OriginalLocalEdgeBlock):
    # Self-contained so an existing release can benchmark without being edited.
    project_before_gather = True


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rings", type=int, nargs="+", default=list(SUPPORTED_RINGS))
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument(
        "--load-context",
        choices=("unverified", "shared", "isolated"),
        default="unverified",
        help="Operator-supplied GPU load context; the benchmark does not stop workers.",
    )
    args = parser.parse_args()
    if (
        args.batch_size != 128
        or not 1 <= args.rounds <= 12
        or not 1 <= args.iterations <= 8
    ):
        parser.error("bounded benchmark requires batch128, rounds1..12, iterations1..8")
    if not 1 <= args.warmups <= 5 or any(
        ring not in SUPPORTED_RINGS for ring in args.rings
    ):
        parser.error("invalid rings or warmup count")
    if len(args.rings) > 4 or len(set(args.rings)) != len(args.rings):
        parser.error("at most four unique rings are allowed")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; no CPU timing is reported")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < 2 * 1024**3:
        raise RuntimeError("benchmark requires at least 2 GiB free GPU memory")
    torch.manual_seed(913)
    torch.backends.cuda.matmul.allow_tf32 = True
    original = OriginalLocalEdgeBlock(
        width=384,
        bottleneck_ratio=0.5,
        dropout=0.0,
        norm_eps=1e-6,
        local_operator="mean",
        adaln_hidden=32,
    ).eval()
    with torch.no_grad():
        original.layer_scale.fill_(0.3)
        assert original.modulation is not None
        original.modulation.weight.normal_(std=0.03)
        original.modulation.bias.normal_(std=0.03)
    optimized = ProjectionFirstLocalEdgeBlock(
        width=384,
        bottleneck_ratio=0.5,
        dropout=0.0,
        norm_eps=1e-6,
        local_operator="mean",
        adaln_hidden=32,
    ).eval()
    optimized.load_state_dict(deepcopy(original.state_dict()), strict=True)
    models: dict[str, Callable[..., Tensor]] = {
        "original": original.to(device),
        "optimized": optimized.to(device),
    }
    if args.compile_model:
        models = {
            name: torch.compile(model, dynamic=True, mode="default")
            for name, model in models.items()
        }
    gpu_ms = 0.0
    timed_wall_seconds = 0.0
    warmup_wall_seconds = 0.0
    results: list[dict[str, object]] = []

    def measure(
        name: str, tensors: tuple[Tensor, ...], count: int
    ) -> tuple[float, Tensor]:
        nonlocal gpu_ms, timed_wall_seconds
        if timed_wall_seconds >= 50.0 or gpu_ms >= 45_000:
            raise TimeoutError("bounded benchmark time budget exhausted")
        sample_started = time.monotonic()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = models[name](*tensors)
        for _ in range(count - 1):
            output = models[name](*tensors)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        gpu_ms += elapsed
        timed_wall_seconds += time.monotonic() - sample_started
        return elapsed / count, output

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ring in args.rings:
            topology = get_topology(ring)

            def expand(tensor: Tensor) -> Tensor:
                return (
                    tensor.to(device)
                    .unsqueeze(0)
                    .expand(args.batch_size, *tensor.shape)
                    .contiguous()
                )

            tensors = (
                torch.randn(
                    (args.batch_size, topology.n, 384),
                    device=device,
                    dtype=torch.bfloat16,
                ),
                expand(topology.neighbor_index),
                expand(topology.neighbor_mask),
                expand(topology.neighbor_edge_type),
                torch.ones(
                    (args.batch_size, topology.n), device=device, dtype=torch.bool
                ),
                torch.randn((args.batch_size, 32), device=device, dtype=torch.bfloat16),
            )
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            # Every new ring shape gets its own unmeasured compile/warmup calls.
            warmup_started = time.monotonic()
            old_output = models["original"](*tensors)
            new_output = models["optimized"](*tensors)
            for _ in range(args.warmups - 1):
                old_output = models["original"](*tensors)
                new_output = models["optimized"](*tensors)
            torch.cuda.synchronize(device)
            ring_warmup_seconds = time.monotonic() - warmup_started
            warmup_wall_seconds += ring_warmup_seconds
            max_abs_difference = (
                (old_output.float() - new_output.float()).abs().max().item()
            )
            torch.testing.assert_close(new_output, old_output, atol=0.015, rtol=0.02)
            del old_output, new_output
            latencies: dict[str, list[float]] = {"original": [], "optimized": []}
            for round_index in range(args.rounds):
                order = (
                    ("original", "optimized")
                    if round_index % 2 == 0
                    else ("optimized", "original")
                )
                for name in order:
                    latency, output = measure(name, tensors, args.iterations)
                    latencies[name].append(latency)
                    del output
            old_ms = statistics.median(latencies["original"])
            new_ms = statistics.median(latencies["optimized"])
            results.append(
                {
                    "ring": ring,
                    "batch_size": args.batch_size,
                    "compiled": args.compile_model,
                    "load_context": args.load_context,
                    "compile_and_warmup_wall_seconds": ring_warmup_seconds,
                    "original_median_ms": old_ms,
                    "optimized_median_ms": new_ms,
                    "speedup": old_ms / new_ms,
                    "max_abs_output_difference": max_abs_difference,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                    / 1024**2,
                    "round_ms": latencies,
                }
            )
            print(json.dumps(results[-1]), flush=True)
            del tensors
    print(
        json.dumps(
            {
                "benchmark": "bf16-local-edge-block-projection-order",
                "evidence_status": "exploratory",
                "compiled": args.compile_model,
                "compile_dynamic": True if args.compile_model else None,
                "compile_mode": "default" if args.compile_model else None,
                "load_context": args.load_context,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "timed_cuda_event_seconds_excluding_warmup": gpu_ms / 1000,
                "timed_wall_seconds": timed_wall_seconds,
                "compile_and_warmup_wall_seconds": warmup_wall_seconds,
                "total_wall_seconds": time.monotonic() - started,
                "note": "Exploratory microbenchmark; concurrent load can distort timing. "
                "Measure isolated compiled actor throughput before rollout.",
                "results": results,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
