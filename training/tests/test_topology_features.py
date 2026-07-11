from dataclasses import fields

import pytest
import torch

from startrain.contracts import (
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    RULES_CONTRACT,
    RULES_HASH,
    fnv1a64,
)
from startrain.features import (
    GLOBAL_FEATURE_DIM,
    GLOBAL_FEATURE_NAMES,
    DoubleStarPosition,
    encode_position,
)
from startrain.symmetry import D5Transform, permute_nodes, transform_position
from startrain.topology import (
    EDGE_BRIDGE,
    EDGE_CLASS_COUNT,
    MAX_NODES,
    MAX_RINGS,
    MIN_RINGS,
    SUPPORTED_RINGS,
    get_topology,
)


def live_position(rings: int) -> DoubleStarPosition:
    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    stones[topology.n - 1] = 1
    return DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=0,
        moves_left=1,
        opening=False,
        terminal=False,
    )


def test_schema_v3_is_exactly_the_canonical_semantic_key() -> None:
    assert FEATURE_SCHEMA_VERSION == 3
    assert FEATURE_SCHEMA_HASH != 0
    assert GLOBAL_FEATURE_DIM == 17
    assert all("pass" not in name for name in GLOBAL_FEATURE_NAMES)
    assert RULES_HASH == fnv1a64(RULES_CONTRACT)
    assert [field.name for field in fields(DoubleStarPosition)] == [
        "rings",
        "stones",
        "to_move",
        "moves_left",
        "opening",
        "terminal",
    ]


def test_topology_edge_classes_preserve_all_d5_actions_and_rings() -> None:
    assert (MIN_RINGS, MAX_RINGS, MAX_NODES) == (4, 10, 275)
    for rings in SUPPORTED_RINGS:
        topology = get_topology(rings)
        assert topology.n == 5 * rings * (rings + 1) // 2
        assert set(topology.edge_type.tolist()) == set(range(EDGE_CLASS_COUNT))
        for left in topology.bridge:
            for right in topology.bridge:
                if left == right:
                    continue
                start = int(topology.adjacency_offsets[left])
                end = int(topology.adjacency_offsets[left + 1])
                neighbors = topology.adjacency[start:end]
                edge_types = topology.adjacency_edge_type[start:end]
                offset = int(torch.nonzero(neighbors == right)[0])
                assert int(edge_types[offset]) == EDGE_BRIDGE

        directed = {
            (int(topology.edge_index[0, edge]), int(topology.edge_index[1, edge])): int(
                topology.edge_type[edge]
            )
            for edge in range(topology.edge_index.shape[1])
        }
        for transform_index in range(10):
            transform = D5Transform.from_index(transform_index)
            permutation = topology.d5_permutation(
                transform.rotation, transform.reflected
            )
            for (source, destination), edge_class in directed.items():
                transformed_edge = (
                    int(permutation[source]),
                    int(permutation[destination]),
                )
                assert directed[transformed_edge] == edge_class


def test_schema_v3_features_are_equivariant_for_every_ring_and_d5_action() -> None:
    for rings in SUPPORTED_RINGS:
        source = live_position(rings)
        encoded = encode_position(source)
        topology = get_topology(rings)
        for transform_index in range(10):
            transform = D5Transform.from_index(transform_index)
            permutation = topology.d5_permutation(
                transform.rotation, transform.reflected
            )
            transformed = encode_position(transform_position(source, transform))
            torch.testing.assert_close(
                transformed.node_features,
                permute_nodes(encoded.node_features, permutation),
            )
            torch.testing.assert_close(
                transformed.global_features, encoded.global_features
            )
            assert torch.equal(
                transformed.legal_node_mask,
                permute_nodes(encoded.legal_node_mask, permutation),
            )


def test_terminal_semantics_are_full_board_only() -> None:
    topology = get_topology(4)
    full = DoubleStarPosition(
        rings=4,
        stones=torch.arange(topology.n, dtype=torch.int8) % 2,
        to_move=0,
        moves_left=0,
        opening=False,
        terminal=True,
    )
    encoded = encode_position(full)
    assert encoded.global_features[6].item() == 1.0
    assert not bool(encoded.legal_node_mask.any())

    with pytest.raises(ValueError, match="board-full"):
        DoubleStarPosition(
            rings=4,
            stones=torch.full((topology.n,), -1, dtype=torch.int8),
            to_move=1,
            moves_left=2,
            opening=False,
            terminal=True,
        )


def test_color_swap_is_current_player_canonical() -> None:
    source = live_position(4)
    swapped_stones = source.stones.clone()
    occupied = swapped_stones >= 0
    swapped_stones[occupied] = 1 - swapped_stones[occupied]
    swapped = DoubleStarPosition(
        rings=4,
        stones=swapped_stones,
        to_move=1,
        moves_left=source.moves_left,
        opening=False,
        terminal=False,
    )
    left = encode_position(source)
    right = encode_position(swapped)
    torch.testing.assert_close(left.node_features, right.node_features)
    torch.testing.assert_close(left.global_features, right.global_features)
