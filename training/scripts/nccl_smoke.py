#!/usr/bin/env python3
"""Run one real GraphResTNet DDP step and verify rank agreement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from startrain.config import load_config
from startrain.features import DoubleStarPosition, encode_batch
from startrain.model import GraphResTNet
from startrain.topology import get_topology


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        raise SystemExit("batch-size must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise SystemExit("NCCL smoke requires at least two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group(backend="nccl")
    try:
        identity = torch.tensor(float(rank + 1), device=device)
        distributed.all_reduce(identity)
        expected = world_size * (world_size + 1) / 2
        if identity.item() != expected:
            raise RuntimeError("NCCL all-reduce returned an incorrect sum")

        experiment = load_config(arguments.config)
        torch.manual_seed(experiment.train.seed)
        model = GraphResTNet(experiment.model).to(device)
        wrapped = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        topology = get_topology(3)
        position = DoubleStarPosition(
            rings=3,
            stones=torch.full((topology.n,), -1, dtype=torch.int8),
            to_move=0,
            moves_left=1,
            opening=True,
            pass_streak=0,
            terminal=False,
        )
        batch = encode_batch([position] * arguments.batch_size).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = wrapped(*batch.model_args())
            loss = (
                output.wdl_logits.float().square().mean()
                + output.score_margin_logits.float().square().mean()
                + output.ownership_logits.float().square().mean()
                + output.alive_logits.float().square().mean()
            )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)

        checksum = torch.stack(
            [parameter.detach().float().sum() for parameter in model.parameters()]
        ).sum()
        minimum = checksum.clone()
        maximum = checksum.clone()
        distributed.all_reduce(minimum, op=distributed.ReduceOp.MIN)
        distributed.all_reduce(maximum, op=distributed.ReduceOp.MAX)
        if not torch.allclose(minimum, maximum, atol=1e-5, rtol=1e-6):
            raise RuntimeError("DDP parameters diverged across ranks")
        if rank == 0:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "world_size": world_size,
                        "device": torch.cuda.get_device_name(device),
                        "loss": float(loss.detach()),
                        "parameter_checksum": float(checksum),
                        "passed": True,
                    },
                    sort_keys=True,
                )
            )
    finally:
        distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
