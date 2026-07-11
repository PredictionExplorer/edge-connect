from __future__ import annotations

import numpy as np
import torch

from startrain.features import DoubleStarPosition
from startrain.losses import LossWeights, compute_losses
from startrain.model import GraphResTNet, ModelConfig
from startrain.optim import OptimizerConfig, build_optimizer
from startrain.replay import ReplaySample, collate_replay_samples
from startrain.scoring import PlayerScore, ScoreResult
from startrain.topology import get_topology
from startrain.training import train_step


def test_tiny_model_overfits_a_fixed_search_target() -> None:
    torch.manual_seed(23)
    topology = get_topology(4)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    position = DoubleStarPosition(
        rings=4,
        stones=stones,
        to_move=1,
        moves_left=2,
        opening=False,
        terminal=False,
    )
    policy = np.zeros(topology.n, dtype=np.float32)
    # Ring-2 node 5 lies on the reflection axis fixed by the existing stone,
    # so the exact D5-equivariant model can distinguish this target uniquely.
    policy[5] = 1.0
    sample = ReplaySample.from_position(
        position,
        policy=policy,
        final_score=ScoreResult(
            players=(
                PlayerScore(10, 3, 1, 1, 0, 11),
                PlayerScore(5, 2, 1, 0, 0, 5),
            ),
            node_owner=torch.zeros(topology.n, dtype=torch.int8),
            alive_stone=torch.zeros(topology.n, dtype=torch.bool),
            contested_peries=0,
            leader=0,
        ),
        search_provenance="learning-smoke",
        policy_provenance="completed-q",
    )
    batch = collate_replay_samples([sample] * 8)
    model = GraphResTNet(
        ModelConfig(
            width=16,
            rrt_groups=1,
            attention_heads=4,
            kv_heads=1,
        )
    )
    optimizer = build_optimizer(
        model,
        OptimizerConfig(kind="adamw", adamw_lr=0.01, weight_decay=0.0),
    )
    weights = LossWeights(1, 1, 0, 0, 0, 0)

    with torch.no_grad():
        initial = compute_losses(
            model(*batch.inputs.model_args()),
            batch.targets,
            legal_action_mask=batch.inputs.legal_action_mask,
            node_mask=batch.inputs.node_mask,
            weights=weights,
        )["total"].item()
    for _ in range(40):
        result = train_step(
            model,
            batch,
            optimizer,
            loss_weights=weights,
            gradient_clip_norm=10.0,
        )
    assert result.losses["total"] < initial * 0.25
