"""Static Double *Star board topology, matching ``src/lib/star/board.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

MIN_RINGS = 3
MAX_RINGS = 12
SECTOR_CHARS = ("*", "S", "T", "A", "R")
EDGE_TANGENTIAL = 0
EDGE_RADIAL_DIAGONAL = 1
EDGE_BRIDGE = 2
EDGE_CLASS_COUNT = 3


def ring_start(ring: int) -> int:
    """Dense id of the first node on ``ring``."""

    return 5 * ring * (ring - 1) // 2


def node_count(rings: int) -> int:
    return ring_start(rings + 1)


@dataclass(frozen=True, slots=True)
class StarTopology:
    """Immutable CPU tensors describing one board size.

    Node ids and edges intentionally use the same ordering as the TypeScript
    engine: ring-major, then sector-major, then clockwise position.
    """

    rings: int
    n: int
    peri_count: int
    sector_of: Tensor
    ring_of: Tensor
    pos_of: Tensor
    is_peri: Tensor
    is_quark: Tensor
    labels: tuple[str, ...]
    adjacency_offsets: Tensor
    adjacency: Tensor
    adjacency_edge_type: Tensor
    edge_index: Tensor
    edge_type: Tensor
    neighbor_index: Tensor
    neighbor_mask: Tensor
    neighbor_edge_type: Tensor
    bridge: tuple[int, ...]

    @property
    def max_degree(self) -> int:
        return int(self.neighbor_index.shape[1])

    def idx(self, sector: int, ring: int, position: int) -> int:
        if ring < 1 or ring > self.rings:
            raise ValueError(f"ring must be in 1..{self.rings}")
        if position < 0 or position >= ring:
            raise ValueError(f"position must be in 0..{ring - 1}")
        return ring_start(ring) + (sector % 5) * ring + position

    def label_to_id(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise ValueError(f"unknown node label: {label}") from exc

    def d5_permutation(self, rotation: int = 0, reflected: bool = False) -> Tensor:
        """Return a source-to-destination node permutation for a D5 action."""

        rotation %= 5
        permutation = torch.empty(self.n, dtype=torch.long)
        for ring in range(1, self.rings + 1):
            width = 5 * ring
            start = ring_start(ring)
            for angular_index in range(width):
                transformed = -angular_index if reflected else angular_index
                transformed = (transformed + rotation * ring) % width
                permutation[start + angular_index] = start + transformed
        return permutation


def _ring_char(ring: int) -> str:
    return "0" if ring == 10 else str(ring)


@lru_cache(maxsize=MAX_RINGS - MIN_RINGS + 1)
def get_topology(rings: int) -> StarTopology:
    """Build and cache a board for ``rings`` in the supported 3..12 range."""

    if isinstance(rings, bool) or not isinstance(rings, int):
        raise TypeError("rings must be an integer")
    if not MIN_RINGS <= rings <= MAX_RINGS:
        raise ValueError(f"rings must be in {MIN_RINGS}..{MAX_RINGS}, got {rings}")

    n = node_count(rings)
    sector_of = torch.empty(n, dtype=torch.long)
    ring_of = torch.empty(n, dtype=torch.long)
    pos_of = torch.empty(n, dtype=torch.long)
    is_peri = torch.zeros(n, dtype=torch.bool)
    is_quark = torch.zeros(n, dtype=torch.bool)
    labels: list[str] = [""] * n

    def idx(sector: int, ring: int, position: int) -> int:
        return ring_start(ring) + (sector % 5) * ring + position

    for ring in range(1, rings + 1):
        for sector in range(5):
            for position in range(ring):
                node = idx(sector, ring, position)
                sector_of[node] = sector
                ring_of[node] = ring
                pos_of[node] = position
                if ring == rings:
                    is_peri[node] = True
                    is_quark[node] = position == 0
                labels[node] = f"{SECTOR_CHARS[sector]}{_ring_char(ring)}{position}"

    edge_indices: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int, int]] = []

    def add_edge(
        first: int,
        second: int,
        edge_class: int,
        *,
        override: bool = False,
    ) -> None:
        edge = (min(first, second), max(first, second))
        existing = edge_indices.get(edge)
        if existing is None:
            edge_indices[edge] = len(edges)
            edges.append((edge[0], edge[1], edge_class))
        elif override:
            edges[existing] = (edge[0], edge[1], edge_class)

    for ring in range(1, rings + 1):
        for sector in range(5):
            for position in range(ring):
                node = idx(sector, ring, position)
                successor = (
                    idx(sector, ring, position + 1)
                    if position < ring - 1
                    else idx(sector + 1, ring, 0)
                )
                add_edge(node, successor, EDGE_TANGENTIAL)
                if ring >= 2:
                    if position <= ring - 2:
                        add_edge(
                            node,
                            idx(sector, ring - 1, position),
                            EDGE_RADIAL_DIAGONAL,
                        )
                    if position >= 1:
                        add_edge(
                            node,
                            idx(sector, ring - 1, position - 1),
                            EDGE_RADIAL_DIAGONAL,
                        )
                    if position == ring - 1:
                        add_edge(
                            node,
                            idx(sector + 1, ring - 1, 0),
                            EDGE_RADIAL_DIAGONAL,
                        )

    bridge = tuple(idx(sector, 1, 0) for sector in range(5))
    for left in range(5):
        for right in range(left + 1, 5):
            add_edge(
                bridge[left],
                bridge[right],
                EDGE_BRIDGE,
                override=True,
            )

    neighbors: list[list[int]] = [[] for _ in range(n)]
    neighbor_classes: list[list[int]] = [[] for _ in range(n)]
    for first, second, edge_class in edges:
        neighbors[first].append(second)
        neighbor_classes[first].append(edge_class)
        neighbors[second].append(first)
        neighbor_classes[second].append(edge_class)

    offsets = [0]
    adjacency_values: list[int] = []
    adjacency_classes: list[int] = []
    for node_neighbors, node_classes in zip(neighbors, neighbor_classes, strict=True):
        adjacency_values.extend(node_neighbors)
        adjacency_classes.extend(node_classes)
        offsets.append(len(adjacency_values))

    max_degree = max(map(len, neighbors))
    neighbor_index = torch.zeros((n, max_degree), dtype=torch.long)
    neighbor_mask = torch.zeros((n, max_degree), dtype=torch.bool)
    neighbor_edge_type = torch.zeros((n, max_degree), dtype=torch.long)
    for node, (node_neighbors, node_classes) in enumerate(
        zip(neighbors, neighbor_classes, strict=True)
    ):
        degree = len(node_neighbors)
        neighbor_index[node, :degree] = torch.tensor(node_neighbors, dtype=torch.long)
        neighbor_mask[node, :degree] = True
        neighbor_edge_type[node, :degree] = torch.tensor(node_classes, dtype=torch.long)

    adjacency = torch.tensor(adjacency_values, dtype=torch.long)
    adjacency_edge_type = torch.tensor(adjacency_classes, dtype=torch.long)
    sources = torch.repeat_interleave(
        torch.arange(n, dtype=torch.long),
        torch.tensor([len(values) for values in neighbors], dtype=torch.long),
    )
    edge_index = torch.stack((sources, adjacency), dim=0)

    topology = StarTopology(
        rings=rings,
        n=n,
        peri_count=5 * rings,
        sector_of=sector_of,
        ring_of=ring_of,
        pos_of=pos_of,
        is_peri=is_peri,
        is_quark=is_quark,
        labels=tuple(labels),
        adjacency_offsets=torch.tensor(offsets, dtype=torch.long),
        adjacency=adjacency,
        adjacency_edge_type=adjacency_edge_type,
        edge_index=edge_index,
        edge_type=adjacency_edge_type,
        neighbor_index=neighbor_index,
        neighbor_mask=neighbor_mask,
        neighbor_edge_type=neighbor_edge_type,
        bridge=bridge,
    )
    _validate_symmetry(topology)
    return topology


def _validate_symmetry(topology: StarTopology) -> None:
    """Fail fast if topology changes break a D5 automorphism."""

    edge_classes = {
        (int(topology.edge_index[0, edge]), int(topology.edge_index[1, edge])): int(
            topology.edge_type[edge]
        )
        for edge in range(topology.edge_index.shape[1])
    }
    for reflected in (False, True):
        permutation = topology.d5_permutation(rotation=1, reflected=reflected)
        for (source, destination), edge_class in edge_classes.items():
            transformed = (int(permutation[source]), int(permutation[destination]))
            if edge_classes.get(transformed) != edge_class:
                raise RuntimeError("topology edge classes are not D5 symmetric")
