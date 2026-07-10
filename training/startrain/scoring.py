"""Reference-compatible scoring annotations for Double *Star positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .topology import StarTopology

EMPTY = -1


@dataclass(frozen=True, slots=True)
class PlayerScore:
    peries: int
    quarks: int
    stars: int
    quark_peri: int
    award: int
    total: int


@dataclass(frozen=True, slots=True)
class ScoreResult:
    players: tuple[PlayerScore, PlayerScore]
    node_owner: Tensor
    alive_stone: Tensor
    contested_peries: int
    leader: int


def _as_stone_list(stones: Sequence[int] | Tensor, expected: int) -> list[int]:
    if isinstance(stones, Tensor):
        values = stones.detach().to(device="cpu", dtype=torch.int8).tolist()
    else:
        values = [int(value) for value in stones]
    if len(values) != expected:
        raise ValueError(f"expected {expected} stones, got {len(values)}")
    if any(value not in (EMPTY, 0, 1) for value in values):
        raise ValueError("stones must contain only -1, 0, or 1")
    return values


def score_position(
    topology: StarTopology,
    stones: Sequence[int] | Tensor,
) -> ScoreResult:
    """Score a position with the same algorithm as the TypeScript engine."""

    values = _as_stone_list(stones, topology.n)
    offsets = topology.adjacency_offsets.tolist()
    adjacency = topology.adjacency.tolist()
    periphery = topology.is_peri.tolist()
    quarks_mask = topology.is_quark.tolist()
    n = topology.n

    parent = list(range(n))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    for node in range(n):
        color = values[node]
        if color == EMPTY:
            continue
        for edge in range(offsets[node], offsets[node + 1]):
            neighbor = adjacency[edge]
            if neighbor > node and values[neighbor] == color:
                node_root = find(node)
                neighbor_root = find(neighbor)
                if node_root != neighbor_root:
                    parent[neighbor_root] = node_root

    occupied_peries = [0] * n
    for node in range(n):
        if values[node] != EMPTY and periphery[node]:
            occupied_peries[find(node)] += 1

    alive = [False] * n
    for node in range(n):
        if values[node] != EMPTY and occupied_peries[find(node)] >= 2:
            alive[node] = True

    region_of = [-1] * n
    region_color: list[int] = []
    stack: list[int] = []
    for start in range(n):
        if alive[start] or region_of[start] != -1:
            continue
        region = len(region_color)
        color = -2
        stack.append(start)
        region_of[start] = region
        while stack:
            node = stack.pop()
            for edge in range(offsets[node], offsets[node + 1]):
                neighbor = adjacency[edge]
                if alive[neighbor]:
                    neighbor_color = values[neighbor]
                    if color == -2:
                        color = neighbor_color
                    elif color != neighbor_color:
                        color = -1
                elif region_of[neighbor] == -1:
                    region_of[neighbor] = region
                    stack.append(neighbor)
        region_color.append(color)

    node_owner = [-1] * n
    alive_stone = [False] * n
    peries = [0, 0]
    quarks = [0, 0]
    stars = [0, 0]
    contested_peries = 0

    for node in range(n):
        if alive[node]:
            owner = values[node]
            alive_stone[node] = True
            if parent[node] == node:
                stars[owner] += 1
        else:
            owner = region_color[region_of[node]]
        if owner in (0, 1):
            node_owner[node] = owner
            if periphery[node]:
                peries[owner] += 1
                if quarks_mask[node]:
                    quarks[owner] += 1
        elif periphery[node]:
            contested_peries += 1

    scores: list[PlayerScore] = []
    for player in (0, 1):
        quark_peri = int(quarks[player] >= 3)
        award = 2 * (stars[1 - player] - stars[player])
        scores.append(
            PlayerScore(
                peries=peries[player],
                quarks=quarks[player],
                stars=stars[player],
                quark_peri=quark_peri,
                award=award,
                total=peries[player] + quark_peri + award,
            )
        )

    if scores[0].total != scores[1].total:
        leader = 0 if scores[0].total > scores[1].total else 1
    elif scores[0].quarks != scores[1].quarks:
        leader = 0 if scores[0].quarks > scores[1].quarks else 1
    else:
        leader = -1

    return ScoreResult(
        players=(scores[0], scores[1]),
        node_owner=torch.tensor(node_owner, dtype=torch.int8),
        alive_stone=torch.tensor(alive_stone, dtype=torch.bool),
        contested_peries=contested_peries,
        leader=leader,
    )
