"""Static Double *Star board topology, matching ``src/lib/star/board.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

SUPPORTED_RINGS = (4, 6, 8, 10)
MIN_RINGS = SUPPORTED_RINGS[0]
MAX_RINGS = SUPPORTED_RINGS[-1]
MAX_NODES = 275
SECTOR_CHARS = ("*", "S", "T", "A", "R")
EDGE_TANGENTIAL = 0
EDGE_RADIAL_DIAGONAL = 1
EDGE_BRIDGE = 2
EDGE_CLASS_COUNT = 3

# D5-invariant pairwise relations for the global attention bias. Every node
# pair maps to one vocabulary entry keyed by ring difference, the canonical
# angular offset bucket, the shortest-path-distance bucket, and the two
# perimeter flags; the same vocabulary is shared by all four board sizes.
RELATION_TOKEN_TOKEN = 0
RELATION_TOKEN_TO_NODE = 1
RELATION_NODE_TO_TOKEN = 2
RELATION_PAD = 3
RELATION_NODE_OFFSET = 4
# lcm(1..10): every ring's angular coordinate becomes an exact integer.
ANGULAR_DENOMINATOR = 2520
ANGULAR_CIRCLE = 5 * ANGULAR_DENOMINATOR
# One bucket per tenth of a sector over the canonical offset range [0, 2.5].
ANGULAR_BUCKET_WIDTH = ANGULAR_DENOMINATOR // 10
ANGULAR_BUCKETS = ANGULAR_CIRCLE // 2 // ANGULAR_BUCKET_WIDTH + 1
SHORTEST_PATH_CAP = 12
RING_DIFFERENCE_CAP = MAX_RINGS - 1


def ring_start(ring: int) -> int:
    """Dense id of the first node on ``ring``."""

    return 5 * ring * (ring - 1) // 2


def node_count(rings: int) -> int:
    if isinstance(rings, bool) or not isinstance(rings, int):
        raise TypeError("rings must be an integer")
    if rings not in SUPPORTED_RINGS:
        supported = ", ".join(str(value) for value in SUPPORTED_RINGS)
        raise ValueError(f"rings must be one of {{{supported}}}, got {rings}")
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
    shortest_path: Tensor
    relation_index: Tensor

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


@lru_cache(maxsize=len(SUPPORTED_RINGS))
def get_topology(rings: int) -> StarTopology:
    """Build and cache a board for one canonical supported ring count."""

    n = node_count(rings)
    if n > MAX_NODES:
        raise RuntimeError("canonical topology exceeds MAX_NODES")
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
    shortest_path = _shortest_paths(neighbors)
    relation_index = _relation_index(
        rings, ring_of, sector_of, pos_of, is_peri, shortest_path
    )

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
        shortest_path=shortest_path,
        relation_index=relation_index,
    )
    _validate_symmetry(topology)
    return topology


def _shortest_paths(neighbors: list[list[int]]) -> Tensor:
    """All-pairs shortest path lengths by breadth-first search."""

    n = len(neighbors)
    distances = torch.full((n, n), -1, dtype=torch.long)
    for source in range(n):
        distances[source, source] = 0
        frontier = [source]
        depth = 0
        while frontier:
            depth += 1
            next_frontier: list[int] = []
            for node in frontier:
                for neighbor in neighbors[node]:
                    if int(distances[source, neighbor]) < 0:
                        distances[source, neighbor] = depth
                        next_frontier.append(neighbor)
            frontier = next_frontier
    if bool((distances < 0).any()):
        raise RuntimeError("board graph is not connected")
    return distances


def _pair_relation_keys(
    rings: int,
    ring_of: Tensor,
    sector_of: Tensor,
    pos_of: Tensor,
    is_peri: Tensor,
    shortest_path: Tensor,
) -> Tensor:
    """Return the `(n, n, 5)` integer relation key of every node pair.

    The key is invariant under every D5 transform applied to both nodes:
    rotations shift the angular coordinate of both nodes equally, reflections
    negate the offset (folded away by taking the canonical minimum), and the
    shortest-path distance, ring, and perimeter flags are automorphism
    invariants.
    """

    angular_index = sector_of * ring_of + pos_of
    scaled = angular_index * (ANGULAR_DENOMINATOR // ring_of)
    offset = (scaled.unsqueeze(0) - scaled.unsqueeze(1)) % ANGULAR_CIRCLE
    canonical = torch.minimum(offset, ANGULAR_CIRCLE - offset)
    angular_bucket = (canonical // ANGULAR_BUCKET_WIDTH).clamp(max=ANGULAR_BUCKETS - 1)
    ring_difference = (ring_of.unsqueeze(0) - ring_of.unsqueeze(1)).clamp(
        -RING_DIFFERENCE_CAP, RING_DIFFERENCE_CAP
    ) + RING_DIFFERENCE_CAP
    distance_bucket = shortest_path.clamp(max=SHORTEST_PATH_CAP)
    peri_source = is_peri.to(torch.long).unsqueeze(1).expand(-1, is_peri.numel())
    peri_target = is_peri.to(torch.long).unsqueeze(0).expand(is_peri.numel(), -1)
    del rings
    return torch.stack(
        (ring_difference, angular_bucket, distance_bucket, peri_source, peri_target),
        dim=-1,
    )


def _relation_key_code(keys: Tensor) -> Tensor:
    """Pack the five key components into one integer per pair."""

    return (
        (
            (keys[..., 0] * ANGULAR_BUCKETS + keys[..., 1]) * (SHORTEST_PATH_CAP + 1)
            + keys[..., 2]
        )
        * 4
        + keys[..., 3] * 2
        + keys[..., 4]
    )


@lru_cache(maxsize=1)
def relation_vocabulary() -> dict[int, int]:
    """Dense relation ids for every pair key occurring on any supported board.

    Ids `0..3` are reserved for the token-token, token-to-node, node-to-token,
    and padding relations; node pairs start at `RELATION_NODE_OFFSET`.
    """

    codes: set[int] = set()
    for rings in SUPPORTED_RINGS:
        codes.update(int(code) for code in _raw_relation_codes(rings).unique())
    return {
        code: RELATION_NODE_OFFSET + index for index, code in enumerate(sorted(codes))
    }


def relation_count() -> int:
    """Total number of relation ids, including the reserved ones."""

    return RELATION_NODE_OFFSET + len(relation_vocabulary())


@lru_cache(maxsize=len(SUPPORTED_RINGS))
def _raw_relation_codes(rings: int) -> Tensor:
    n = node_count(rings)
    sector_of = torch.empty(n, dtype=torch.long)
    ring_of = torch.empty(n, dtype=torch.long)
    pos_of = torch.empty(n, dtype=torch.long)
    is_peri = torch.zeros(n, dtype=torch.bool)
    neighbors: list[list[int]] = [[] for _ in range(n)]

    def idx(sector: int, ring: int, position: int) -> int:
        return ring_start(ring) + (sector % 5) * ring + position

    def link(first: int, second: int) -> None:
        if second not in neighbors[first]:
            neighbors[first].append(second)
            neighbors[second].append(first)

    for ring in range(1, rings + 1):
        for sector in range(5):
            for position in range(ring):
                node = idx(sector, ring, position)
                sector_of[node] = sector
                ring_of[node] = ring
                pos_of[node] = position
                is_peri[node] = ring == rings
                successor = (
                    idx(sector, ring, position + 1)
                    if position < ring - 1
                    else idx(sector + 1, ring, 0)
                )
                link(node, successor)
                if ring >= 2:
                    if position <= ring - 2:
                        link(node, idx(sector, ring - 1, position))
                    if position >= 1:
                        link(node, idx(sector, ring - 1, position - 1))
                    if position == ring - 1:
                        link(node, idx(sector + 1, ring - 1, 0))
    for left in range(5):
        for right in range(left + 1, 5):
            link(idx(left, 1, 0), idx(right, 1, 0))
    shortest_path = _shortest_paths(neighbors)
    keys = _pair_relation_keys(
        rings, ring_of, sector_of, pos_of, is_peri, shortest_path
    )
    return _relation_key_code(keys)


def _relation_index(
    rings: int,
    ring_of: Tensor,
    sector_of: Tensor,
    pos_of: Tensor,
    is_peri: Tensor,
    shortest_path: Tensor,
) -> Tensor:
    keys = _pair_relation_keys(
        rings, ring_of, sector_of, pos_of, is_peri, shortest_path
    )
    codes = _relation_key_code(keys)
    vocabulary = relation_vocabulary()
    sorted_codes = torch.tensor(sorted(vocabulary), dtype=torch.long)
    ids = torch.tensor(
        [vocabulary[int(code)] for code in sorted_codes], dtype=torch.long
    )
    positions = torch.searchsorted(sorted_codes, codes.reshape(-1))
    if not torch.equal(sorted_codes[positions], codes.reshape(-1)):
        raise RuntimeError("relation key missing from the shared vocabulary")
    return ids[positions].reshape(codes.shape)


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
        relation = topology.relation_index
        transformed_relation = relation[permutation][:, permutation]
        if not torch.equal(relation, transformed_relation):
            raise RuntimeError("topology relations are not D5 invariant")
        if not torch.equal(
            topology.shortest_path, topology.shortest_path[permutation][:, permutation]
        ):
            raise RuntimeError("shortest paths are not D5 invariant")


def relation_tables() -> Tensor:
    """Stacked `(len(SUPPORTED_RINGS), MAX_NODES + 1, MAX_NODES + 1)` relations.

    Row and column zero hold the global token; nodes follow in dense order and
    padded nodes use `RELATION_PAD`. The model indexes this buffer by the
    per-sample ring slot from :func:`ring_slots`.
    """

    tables = torch.full(
        (len(SUPPORTED_RINGS), MAX_NODES + 1, MAX_NODES + 1),
        RELATION_PAD,
        dtype=torch.long,
    )
    for slot, rings in enumerate(SUPPORTED_RINGS):
        topology = get_topology(rings)
        n = topology.n
        tables[slot, 0, 0] = RELATION_TOKEN_TOKEN
        tables[slot, 0, 1 : n + 1] = RELATION_TOKEN_TO_NODE
        tables[slot, 1 : n + 1, 0] = RELATION_NODE_TO_TOKEN
        tables[slot, 1 : n + 1, 1 : n + 1] = topology.relation_index
    return tables


def ring_slots() -> Tensor:
    """Map a ring count to its slot in :func:`relation_tables` (`-1` if unsupported)."""

    slots = torch.full((MAX_RINGS + 1,), -1, dtype=torch.long)
    for slot, rings in enumerate(SUPPORTED_RINGS):
        slots[rings] = slot
    return slots
