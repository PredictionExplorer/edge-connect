"""Deterministic D5 augmentation for board and action tensors."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .features import DoubleStarPosition
from .topology import StarTopology, get_topology


@dataclass(frozen=True, slots=True)
class D5Transform:
    rotation: int = 0
    reflected: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.rotation, bool) or not isinstance(self.rotation, int):
            raise TypeError("rotation must be an integer")
        object.__setattr__(self, "rotation", self.rotation % 5)

    @property
    def index(self) -> int:
        return self.rotation + (5 if self.reflected else 0)

    @classmethod
    def from_index(cls, index: int) -> "D5Transform":
        if not 0 <= index < 10:
            raise ValueError("D5 transform index must be in 0..9")
        return cls(rotation=index % 5, reflected=index >= 5)


ALL_D5_TRANSFORMS = tuple(D5Transform.from_index(index) for index in range(10))


def inverse_permutation(source_to_destination: Tensor) -> Tensor:
    permutation = source_to_destination.to(dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), device=permutation.device, dtype=torch.long
    )
    return inverse


def permute_nodes(
    values: Tensor,
    source_to_destination: Tensor,
    *,
    node_dim: int = 0,
) -> Tensor:
    """Move each source node value to its transformed destination."""

    inverse = inverse_permutation(source_to_destination.to(values.device))
    return values.index_select(node_dim, inverse)


def permute_actions(
    values: Tensor,
    source_to_destination: Tensor,
    *,
    action_dim: int = -1,
) -> Tensor:
    """Permute a node-only action tensor."""

    action_dim %= values.ndim
    node_count = source_to_destination.numel()
    if values.shape[action_dim] != node_count:
        raise ValueError("action dimension must contain exactly N node actions")
    return permute_nodes(
        values,
        source_to_destination,
        node_dim=action_dim,
    )


def transform_position(
    position: DoubleStarPosition,
    transform: D5Transform,
) -> DoubleStarPosition:
    topology = get_topology(position.rings)
    permutation = topology.d5_permutation(
        rotation=transform.rotation, reflected=transform.reflected
    )
    return replace(
        position,
        stones=permute_nodes(position.stones, permutation),
    )


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def deterministic_transform(
    *, seed: int, sample_index: int, epoch: int = 0
) -> D5Transform:
    """Map a sample/epoch pair to one of ten transforms without global RNG."""

    if sample_index < 0 or epoch < 0:
        raise ValueError("sample_index and epoch must be non-negative")
    mixed = (
        (seed & ((1 << 64) - 1))
        ^ ((sample_index + 1) * 0xD1342543DE82EF95)
        ^ ((epoch + 1) * 0xA24BAED4963EE407)
    )
    return D5Transform.from_index(_splitmix64(mixed) % 10)


class D5Augmenter:
    """Stateless, worker-safe deterministic augmentation."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def transform_for(self, sample_index: int, epoch: int = 0) -> D5Transform:
        return deterministic_transform(
            seed=self.seed, sample_index=sample_index, epoch=epoch
        )

    def position(
        self,
        position: DoubleStarPosition,
        *,
        sample_index: int,
        epoch: int = 0,
    ) -> DoubleStarPosition:
        return transform_position(
            position, self.transform_for(sample_index=sample_index, epoch=epoch)
        )


def permutation_for(topology: StarTopology, transform: D5Transform) -> Tensor:
    return topology.d5_permutation(transform.rotation, transform.reflected)
