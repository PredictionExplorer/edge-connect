from dataclasses import fields

import pytest
import torch

from startrain.contracts import (
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_SCHEMA_HASH,
    RULES_CONTRACT,
    RULES_HASH,
    fnv1a64,
)
from startrain.features import (
    GLOBAL_FEATURE_DIM,
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    DoubleStarPosition,
    encode_position,
    history_sets_from_flags,
)
from startrain.features_v3 import (
    LEGACY_GLOBAL_FEATURE_DIM,
    LEGACY_NODE_FEATURE_DIM,
    encode_legacy_position,
)
from startrain.symmetry import D5Transform, permute_nodes, transform_position
from startrain.topology import (
    ANGULAR_BUCKETS,
    EDGE_BRIDGE,
    EDGE_CLASS_COUNT,
    MAX_NODES,
    MAX_RINGS,
    MIN_RINGS,
    RELATION_NODE_OFFSET,
    RELATION_NODE_TO_TOKEN,
    RELATION_PAD,
    RELATION_TOKEN_TO_NODE,
    RELATION_TOKEN_TOKEN,
    SHORTEST_PATH_CAP,
    SUPPORTED_RINGS,
    get_topology,
    relation_count,
    relation_tables,
    ring_slots,
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


def history_position(rings: int) -> DoubleStarPosition:
    """Player 0 mid-turn after a three-stone handicap opening and one reply."""

    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[[0, 1, 2]] = 0
    stones[[10, 11]] = 1
    stones[20] = 0
    current = torch.zeros(topology.n, dtype=torch.bool)
    current[20] = True
    previous = torch.zeros(topology.n, dtype=torch.bool)
    previous[[10, 11]] = True
    own_previous = torch.zeros(topology.n, dtype=torch.bool)
    own_previous[[0, 1, 2]] = True
    handicap = own_previous.clone()
    return DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=0,
        moves_left=1,
        opening=False,
        terminal=False,
        handicap=3,
        current_turn=current,
        previous_turn=previous,
        own_previous_turn=own_previous,
        handicap_stones=handicap,
        pda=2,
    )


def test_schema_v4_is_exactly_the_semantic_key_plus_context() -> None:
    assert FEATURE_SCHEMA_VERSION == 4
    assert FEATURE_SCHEMA_HASH != 0
    assert FEATURE_SCHEMA_HASH != LEGACY_FEATURE_SCHEMA_HASH
    assert (NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM) == (19, 25)
    assert (LEGACY_NODE_FEATURE_DIM, LEGACY_GLOBAL_FEATURE_DIM) == (15, 17)
    assert NODE_FEATURE_NAMES[:15] == (
        "empty",
        "current_stone",
        "opponent_stone",
        "owner_current",
        "owner_opponent",
        "owner_unclaimed",
        "alive_current",
        "alive_opponent",
        "is_peri",
        "is_quark",
        "ring_fraction",
        "arm_distance_fraction",
        "degree_fraction",
        "is_bridge",
        "legal",
    )
    assert all("pass" not in name for name in GLOBAL_FEATURE_NAMES)
    assert RULES_HASH == fnv1a64(RULES_CONTRACT)
    assert [field.name for field in fields(DoubleStarPosition)] == [
        "rings",
        "stones",
        "to_move",
        "moves_left",
        "opening",
        "terminal",
        "mode",
        "handicap",
        "pie",
        "swap_available",
        "swapped",
        "current_turn",
        "previous_turn",
        "own_previous_turn",
        "handicap_stones",
        "history_known",
        "pda",
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


def test_pairwise_relations_are_d5_invariant_and_shared_across_rings() -> None:
    assert relation_count() > RELATION_NODE_OFFSET
    assert ANGULAR_BUCKETS == 26
    seen: set[int] = set()
    for rings in SUPPORTED_RINGS:
        topology = get_topology(rings)
        relation = topology.relation_index
        assert relation.shape == (topology.n, topology.n)
        assert int(relation.min()) >= RELATION_NODE_OFFSET
        assert int(relation.max()) < relation_count()
        assert torch.equal(topology.shortest_path, topology.shortest_path.T)
        assert torch.equal(
            topology.shortest_path.diagonal(), torch.zeros(topology.n, dtype=torch.long)
        )
        # Adjacent nodes are at distance one; the bias key sees them as such.
        for edge in range(topology.edge_index.shape[1]):
            source = int(topology.edge_index[0, edge])
            destination = int(topology.edge_index[1, edge])
            assert int(topology.shortest_path[source, destination]) == 1
        assert int(topology.shortest_path.max()) >= SHORTEST_PATH_CAP or rings < 8
        for transform_index in range(10):
            transform = D5Transform.from_index(transform_index)
            permutation = topology.d5_permutation(
                transform.rotation, transform.reflected
            )
            assert torch.equal(relation, relation[permutation][:, permutation])
        seen.update(relation.unique().tolist())
    assert len(seen) == relation_count() - RELATION_NODE_OFFSET

    tables = relation_tables()
    assert tables.shape == (len(SUPPORTED_RINGS), MAX_NODES + 1, MAX_NODES + 1)
    slots = ring_slots()
    for rings in SUPPORTED_RINGS:
        table = tables[slots[rings]]
        n = get_topology(rings).n
        assert int(table[0, 0]) == RELATION_TOKEN_TOKEN
        assert bool((table[0, 1 : n + 1] == RELATION_TOKEN_TO_NODE).all())
        assert bool((table[1 : n + 1, 0] == RELATION_NODE_TO_TOKEN).all())
        assert torch.equal(
            table[1 : n + 1, 1 : n + 1], get_topology(rings).relation_index
        )
        if n < MAX_NODES:
            assert bool((table[n + 1 :, :] == RELATION_PAD).all())
            assert bool((table[:, n + 1 :] == RELATION_PAD).all())
    assert int(slots[5]) == -1


def test_schema_v4_features_are_equivariant_for_every_ring_and_d5_action() -> None:
    for rings in SUPPORTED_RINGS:
        for source in (live_position(rings), history_position(rings)):
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


def test_history_planes_and_variant_scalars_are_encoded() -> None:
    position = history_position(6)
    encoded = encode_position(position)
    planes = encoded.node_features
    assert planes[20, 15].item() == 1.0
    assert planes[20, 16].item() == 0.0
    for node in (0, 1, 2):
        assert planes[node, 16].item() == 1.0  # own previous turn
        assert planes[node, 18].item() == 1.0  # handicap stone
        assert planes[node, 17].item() == 0.0
    for node in (10, 11):
        assert planes[node, 17].item() == 1.0  # opponent previous turn
        assert planes[node, 18].item() == 0.0
    globals_ = encoded.global_features
    assert globals_[4].item() == pytest.approx(0.5)  # one of two placements left
    assert globals_[17].item() == pytest.approx(1.0)  # double
    assert globals_[18].item() == pytest.approx(3 / 9)
    assert globals_[19].item() == 0.0  # handicap phase is over
    assert globals_[20].item() == 0.0
    assert globals_[21].item() == 0.0
    assert globals_[22].item() == 0.0
    assert globals_[23].item() == 1.0
    assert globals_[24].item() == pytest.approx(2 / 3)
    assert position.segment == "handicap"
    assert position.variant_label == "handicap-3-double"
    flags = position.history_flags()
    assert int(flags[20]) == 1 and int(flags[0]) == 2 | 8 and int(flags[10]) == 4
    rebuilt = history_sets_from_flags(flags, flags.numel())
    assert torch.equal(rebuilt[0], position.current_turn)
    assert torch.equal(rebuilt[1], position.previous_turn)
    assert torch.equal(rebuilt[2], position.own_previous_turn)
    assert torch.equal(rebuilt[3], position.handicap_stones)

    # The legacy encoder ignores every variant and history field.
    legacy = encode_legacy_position(position)
    assert legacy.node_features.shape == (105, 15)
    assert legacy.global_features.shape == (17,)
    assert legacy.global_features[4].item() == pytest.approx(0.5)
    torch.testing.assert_close(legacy.node_features, planes[:, :15])
    torch.testing.assert_close(legacy.global_features[:4], globals_[:4])


def test_opening_derives_history_and_rejects_inconsistent_sets() -> None:
    topology = get_topology(4)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[[3, 4]] = 0
    opening = DoubleStarPosition(
        rings=4,
        stones=stones,
        to_move=0,
        moves_left=3,
        opening=True,
        terminal=False,
        handicap=5,
    )
    assert opening.current_turn is not None and opening.handicap_stones is not None
    assert opening.current_turn.sum().item() == 2
    assert torch.equal(opening.current_turn, opening.handicap_stones)
    encoded = encode_position(opening)
    assert encoded.global_features[19].item() == 1.0
    assert encoded.global_features[20].item() == pytest.approx(3 / 9)
    assert encoded.global_features[4].item() == pytest.approx(3 / 5)
    assert opening.pie_pending is False

    with pytest.raises(ValueError, match="opening"):
        DoubleStarPosition(
            rings=4,
            stones=stones,
            to_move=0,
            moves_left=1,
            opening=True,
            terminal=False,
        )
    bogus = torch.zeros(topology.n, dtype=torch.bool)
    bogus[40] = True
    with pytest.raises(ValueError, match="history"):
        DoubleStarPosition(
            rings=4,
            stones=stones,
            to_move=0,
            moves_left=3,
            opening=True,
            terminal=False,
            handicap=5,
            current_turn=bogus,
        )
    with pytest.raises(ValueError, match="pie"):
        DoubleStarPosition(
            rings=4,
            stones=torch.full((topology.n,), -1, dtype=torch.int8),
            to_move=0,
            moves_left=2,
            opening=True,
            terminal=False,
            handicap=2,
            pie=True,
        )
    with pytest.raises(ValueError, match="pda"):
        DoubleStarPosition(
            rings=4,
            stones=torch.full((topology.n,), -1, dtype=torch.int8),
            to_move=0,
            moves_left=1,
            opening=True,
            terminal=False,
            pda=4,
        )


def test_pie_pending_and_swap_available_positions() -> None:
    topology = get_topology(4)
    empty = torch.full((topology.n,), -1, dtype=torch.int8)
    pending = DoubleStarPosition(
        rings=4,
        stones=empty,
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
        mode="classic",
        pie=True,
    )
    assert pending.pie_pending
    assert pending.segment == "pie" and pending.variant_label == "pie-classic"
    encoded = encode_position(pending)
    assert encoded.global_features[21].item() == 1.0
    assert encoded.global_features[17].item() == pytest.approx(0.5)

    stones = empty.clone()
    stones[7] = 0
    previous = torch.zeros(topology.n, dtype=torch.bool)
    previous[7] = True
    kept = DoubleStarPosition(
        rings=4,
        stones=stones,
        to_move=1,
        moves_left=2,
        opening=False,
        terminal=False,
        pie=True,
        swap_available=True,
        previous_turn=previous,
        handicap_stones=previous,
    )
    swapped_stones = empty.clone()
    swapped_stones[7] = 1
    swapped = DoubleStarPosition(
        rings=4,
        stones=swapped_stones,
        to_move=0,
        moves_left=2,
        opening=False,
        terminal=False,
        pie=True,
        swapped=True,
        previous_turn=previous,
        handicap_stones=previous,
    )
    left = encode_position(kept)
    right = encode_position(swapped)
    torch.testing.assert_close(left.node_features, right.node_features)
    expected = left.global_features.clone()
    expected[22] = 0.0
    torch.testing.assert_close(right.global_features, expected)
    with pytest.raises(ValueError, match="swap"):
        DoubleStarPosition(
            rings=4,
            stones=stones,
            to_move=1,
            moves_left=2,
            opening=False,
            terminal=False,
            swap_available=True,
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
    with pytest.raises(ValueError, match="turn"):
        DoubleStarPosition(
            rings=4,
            stones=torch.arange(topology.n, dtype=torch.int8) % 2,
            to_move=0,
            moves_left=1,
            opening=False,
            terminal=True,
            mode="classic",
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
