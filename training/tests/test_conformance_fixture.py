import json
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from startrain.actions import extract_sample_actions, relocate_sample_actions
from startrain.contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    CONFORMANCE_SCHEMA_ID,
    EXTERNAL_FEATURE_SCHEMA_ID,
    LEGACY_RULES_HASH_WIRE,
    RULES_CANONICAL,
    RULES_HASH,
    RULES_HASH_ALGORITHM,
    RULES_HASH_HEX,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
    fnv1a64,
)
from startrain.features import DoubleStarPosition, encode_position
from startrain.scoring import ScoreResult, score_position
from startrain.symmetry import D5Transform, transform_position
from startrain.topology import MAX_NODES, SUPPORTED_RINGS, get_topology

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "testdata" / "star" / "conformance-v3.json"
)
VARIANT_GAME_IDS = (
    "rings-4-classic-board-full",
    "rings-4-handicap-5-double-board-full",
    "rings-6-handicap-9-classic-board-full",
    "rings-4-pie-double-swap-board-full",
    "rings-4-pie-double-keep-board-full",
    "rings-6-pie-classic-swap-board-full",
)


@pytest.fixture(scope="session")
def conformance() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def assert_score_matches(expected: dict, actual: ScoreResult) -> None:
    assert [
        {
            "peries": player.peries,
            "quarks": player.quarks,
            "stars": player.stars,
            "quarkPeri": player.quark_peri,
            "award": player.award,
            "total": player.total,
        }
        for player in actual.players
    ] == expected["players"]
    assert actual.node_owner.tolist() == expected["nodeOwner"]
    assert actual.alive_stone.to(torch.uint8).tolist() == expected["aliveStone"]
    assert actual.contested_peries == expected["contestedPeries"]
    assert actual.leader == expected["leader"]


def node_mask(nodes: list[int], size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[nodes] = True
    return mask


def position_from_fixture(state: dict, config: dict) -> DoubleStarPosition:
    """Rebuild the semantic key of one fixture state, including its history."""

    size = len(state["stones"])
    return DoubleStarPosition.from_sequence(
        rings=config["rings"],
        stones=state["stones"],
        to_move=state["toMove"],
        moves_left=state["movesLeft"],
        opening=state["opening"],
        terminal=state.get("terminal", state.get("over")),
        mode=config["mode"],
        handicap=config["handicap"],
        pie=config["pieRule"],
        swap_available=state.get("canSwap", False),
        swapped=state.get("swapped", False),
    ).__class__(
        rings=config["rings"],
        stones=torch.as_tensor(state["stones"], dtype=torch.int8),
        to_move=state["toMove"],
        moves_left=state["movesLeft"],
        opening=state["opening"],
        terminal=state.get("terminal", state.get("over")),
        mode=config["mode"],
        handicap=config["handicap"],
        pie=config["pieRule"],
        swap_available=state.get("canSwap", False),
        swapped=state.get("swapped", False),
        current_turn=node_mask(state["currentTurnMoves"], size),
        previous_turn=node_mask(state["previousTurnMoves"], size),
        own_previous_turn=node_mask(state["ownPreviousTurnMoves"], size),
        handicap_stones=node_mask(state["handicapStones"], size),
    )


def test_v3_rules_identifiers_and_exact_canonical_hash() -> None:
    assert RULES_SCHEMA_ID == "edgeconnect.star.rules.v3"
    assert CONFORMANCE_SCHEMA_ID == "edgeconnect.star.conformance.v3"
    assert EXTERNAL_FEATURE_SCHEMA_ID == "edgeconnect.star.model-features.external.v3"
    assert ACTION_LAYOUT_SCHEMA_ID == "edgeconnect.star.action-layout.nodes-only.v1"
    assert RULES_HASH_WIRE == "fnv1a64:a5d932b0ef8354e8"
    assert LEGACY_RULES_HASH_WIRE == "fnv1a64:2da3783519381453"
    assert f"{fnv1a64(RULES_CANONICAL):016x}" == RULES_HASH_HEX
    assert RULES_HASH == int(RULES_HASH_HEX, 16)
    assert "rings=even:{4,6,8,10};" in RULES_CANONICAL
    assert "modes={classic:turn-size-1,double:opening-1-then-2};" in RULES_CANONICAL
    assert "handicap=k-consecutive-opening-placements-by-player0" in RULES_CANONICAL
    assert "handicap-excludes-pie;" in RULES_CANONICAL
    assert "actions=atomic-place|swap;" in RULES_CANONICAL
    assert "action-wire=place(node)->node,swap->node-count;" in RULES_CANONICAL
    assert "native-action-layout=node-u-at-u;" in RULES_CANONICAL
    assert "outcome-class=loss:0,win:1;" in RULES_CANONICAL


def test_all_supported_topologies_follow_canonical_node_and_d5_layout() -> None:
    for rings in SUPPORTED_RINGS:
        topology = get_topology(rings)
        assert topology.n == 5 * rings * (rings + 1) // 2
        assert topology.n <= MAX_NODES
        assert topology.peri_count == 5 * rings
        assert topology.labels[0] == "*10"
        assert topology.labels[-1].startswith("R")
        for index in range(10):
            transform = D5Transform.from_index(index)
            mapping = topology.d5_permutation(transform.rotation, transform.reflected)
            assert sorted(mapping.tolist()) == list(range(topology.n))


def test_semantic_key_and_action_layout_have_no_reserved_action() -> None:
    names = [field.name for field in fields(DoubleStarPosition)]
    assert names[:6] == [
        "rings",
        "stones",
        "to_move",
        "moves_left",
        "opening",
        "terminal",
    ]
    assert "swap" not in names
    for sample_nodes, batch_nodes in ((50, 50), (50, 105), (105, 275)):
        native = torch.arange(sample_nodes)
        padded = relocate_sample_actions(
            native,
            sample_nodes=sample_nodes,
            batch_max_nodes=batch_nodes,
            fill_value=-999,
        )
        assert padded.shape == (batch_nodes,)
        assert torch.equal(padded[:sample_nodes], native)
        assert torch.equal(
            extract_sample_actions(
                padded,
                sample_nodes=sample_nodes,
                batch_max_nodes=batch_nodes,
            ),
            native,
        )


def test_ab_ba_pair_equivalence_has_identical_features() -> None:
    topology = get_topology(4)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[7] = 0
    stones[11] = 0
    ab = DoubleStarPosition(
        rings=4,
        stones=stones,
        to_move=1,
        moves_left=2,
        opening=False,
        terminal=False,
    )
    ba = DoubleStarPosition.from_sequence(
        rings=4,
        stones=stones.tolist(),
        to_move=1,
        moves_left=2,
        opening=False,
        terminal=False,
    )
    encoded_ab = encode_position(ab)
    encoded_ba = encode_position(ba)
    assert torch.equal(encoded_ab.node_features, encoded_ba.node_features)
    assert torch.equal(encoded_ab.global_features, encoded_ba.global_features)

    transformed = transform_position(ab, D5Transform(rotation=3, reflected=True))
    assert transformed.to_move == ab.to_move
    assert transformed.moves_left == ab.moves_left


def test_fixture_v3_identifiers_and_exact_canonical_bytes(conformance: dict) -> None:
    assert conformance["schema"] == CONFORMANCE_SCHEMA_ID
    assert conformance["schemas"] == {
        "rules": RULES_SCHEMA_ID,
        "conformance": CONFORMANCE_SCHEMA_ID,
        "modelFeatures": EXTERNAL_FEATURE_SCHEMA_ID,
        "actionLayout": ACTION_LAYOUT_SCHEMA_ID,
    }
    fixture_rules = conformance["rules"]
    assert fixture_rules["contract"]["schema"] == RULES_SCHEMA_ID
    assert fixture_rules["contract"]["board"]["supportedRings"] == list(SUPPORTED_RINGS)
    assert fixture_rules["contract"]["game"]["handicapExcludesPie"] is True
    assert fixture_rules["hashAlgorithm"] == RULES_HASH_ALGORITHM
    assert fixture_rules["hash"] == RULES_HASH_WIRE
    assert fixture_rules["canonical"] == RULES_CANONICAL
    assert f"{fnv1a64(RULES_CANONICAL):016x}" == RULES_HASH_HEX
    assert RULES_HASH == int(RULES_HASH_HEX, 16)
    assert conformance["outcomeEncoding"] == {
        "loss": 0,
        "win": 1,
        "value": "P(win)-P(loss)",
    }
    assert conformance["actionEncoding"]["swapCode"] == (
        "node count, one past the last node"
    )


def test_fixture_topology_csr_and_every_d5_vector(conformance: dict) -> None:
    assert [board["rings"] for board in conformance["boards"]] == list(SUPPORTED_RINGS)
    for expected in conformance["boards"]:
        topology = get_topology(expected["rings"])
        assert topology.n == expected["nodeCount"]
        assert topology.peri_count == expected["perimeterCount"]
        assert topology.edge_index.shape[1] // 2 == expected["edgeCount"]
        assert topology.max_degree == expected["maximumDegree"]
        assert topology.sector_of.tolist() == expected["sectorOf"]
        assert topology.ring_of.tolist() == expected["ringOf"]
        assert topology.pos_of.tolist() == expected["positionOf"]
        assert topology.is_peri.to(torch.uint8).tolist() == expected["perimeterMask"]
        assert topology.is_quark.to(torch.uint8).tolist() == expected["quarkMask"]
        assert list(topology.labels) == expected["labels"]
        assert topology.adjacency_offsets.tolist() == expected["adjacencyOffsets"]
        assert topology.adjacency.tolist() == expected["adjacency"]
        assert list(topology.bridge) == expected["bridge"]
        for node in expected["nodes"]:
            start = int(topology.adjacency_offsets[node["id"]])
            end = int(topology.adjacency_offsets[node["id"] + 1])
            assert sorted(topology.adjacency[start:end].tolist()) == node["adjacent"]
        for index, symmetry in enumerate(expected["symmetries"]):
            transform = D5Transform.from_index(index)
            mapping = topology.d5_permutation(transform.rotation, transform.reflected)
            assert transform.index == index
            assert mapping.tolist() == symmetry["map"]
            inverse = torch.empty_like(mapping)
            inverse[mapping] = torch.arange(topology.n)
            assert inverse.tolist() == symmetry["inverseMap"]


def test_fixture_scoring_vectors(conformance: dict) -> None:
    assert {vector["rings"] for vector in conformance["scores"]} == set(SUPPORTED_RINGS)
    for vector in conformance["scores"]:
        topology = get_topology(vector["rings"])
        actual = score_position(topology, vector["stones"])
        assert_score_matches(vector["expected"], actual)


def test_fixture_games_cover_every_variant_and_replay_as_semantic_keys(
    conformance: dict,
) -> None:
    games = conformance["games"]
    assert [game["id"] for game in games[: len(SUPPORTED_RINGS)]] == [
        f"rings-{rings}-board-full" for rings in SUPPORTED_RINGS
    ]
    assert (
        tuple(game["id"] for game in games[len(SUPPORTED_RINGS) :]) == VARIANT_GAME_IDS
    )
    swaps = 0
    for game in games:
        config = game["config"]
        node_count = get_topology(config["rings"]).n
        for action, code in zip(game["actions"], game["actionCodes"], strict=True):
            if action["type"] == "swap":
                swaps += 1
                assert code == node_count
            else:
                assert code == action["node"]
        # Every recorded state is a valid semantic key with a valid history.
        for state in game["states"]:
            position = position_from_fixture(state, config)
            assert position.swap_available == state["canSwap"]
            assert position.swapped == state["swapped"]
            assert position.pie_pending == (
                config["pieRule"] and state["stonesPlaced"] == 0
            )
            assert bool(position.current_turn.any()) == (
                len(state["currentTurnMoves"]) > 0
            )
        first = position_from_fixture(game["states"][0], config)
        assert first.opening and first.moves_left == config["handicap"]
    assert swaps == 2


def test_fixture_full_board_terminals_have_binary_targets(
    conformance: dict,
) -> None:
    assert len(conformance["games"]) == len(SUPPORTED_RINGS) + len(VARIANT_GAME_IDS)
    for game in conformance["games"]:
        final_state = game["states"][-1]
        position = position_from_fixture(final_state, game["config"])
        assert position.terminal
        assert not position.opening
        assert not bool(encode_position(position).legal_node_mask.any())
        assert final_state["stonesPlaced"] == len(final_state["stones"])
        assert game["terminal"]["reason"] == "board-full"
        assert position.moves_left < position.turn_size

        score = score_position(get_topology(position.rings), position.stones)
        assert_score_matches(game["terminal"]["score"], score)
        assert score.leader in (0, 1)
        terminal = game["terminal"]
        values = [1 if score.leader == player else -1 for player in (0, 1)]
        outcomes = [1 if value == 1 else 0 for value in values]
        margins = [
            score.players[0].total - score.players[1].total,
            score.players[1].total - score.players[0].total,
        ]
        assert values == terminal["valuesByPlayer"]
        assert outcomes == terminal["outcomeClassesByPlayer"]
        assert margins == terminal["scoreMarginsByPlayer"]
        assert terminal["valuePerspective"] == {
            "kind": "toMove",
            "player": position.to_move,
            "value": values[position.to_move],
            "outcomeClass": outcomes[position.to_move],
            "scoreMargin": margins[position.to_move],
        }


def test_fixture_action_layouts_are_nodes_only(conformance: dict) -> None:
    assert conformance["actionLayouts"]["schema"] == ACTION_LAYOUT_SCHEMA_ID
    for batch in conformance["actionLayouts"]["mixedBatches"]:
        maximum_nodes = batch["maximumNodes"]
        assert batch["batchActionCount"] == maximum_nodes
        for row in batch["rows"]:
            nodes = row["nodeCount"]
            native = torch.arange(nodes)
            padded = relocate_sample_actions(
                native,
                sample_nodes=nodes,
                batch_max_nodes=maximum_nodes,
                fill_value=-999,
            )
            assert row["native"]["actionCount"] == nodes
            assert row["padded"]["actionCount"] == maximum_nodes
            if row["padded"]["paddingSlots"]:
                first, last = row["padded"]["paddingSlots"]
                assert torch.equal(
                    padded[first : last + 1],
                    torch.full((last - first + 1,), -999),
                )
            assert torch.equal(
                extract_sample_actions(
                    padded,
                    sample_nodes=nodes,
                    batch_max_nodes=maximum_nodes,
                ),
                native,
            )
            for example in row["examples"]:
                assert example["action"]["type"] == "place"
                node = example["action"]["node"]
                assert example["wireCode"] == node
                assert example["nativeIndex"] == node
                assert example["paddedIndex"] == node


def test_fixture_ab_ba_semantic_keys_have_identical_model_features(
    conformance: dict,
) -> None:
    pair = conformance["pairEquivalences"][0]
    config = pair["config"]
    ab = position_from_fixture(pair["ab"]["semanticState"], config)
    ba = position_from_fixture(pair["ba"]["semanticState"], config)
    assert pair["ab"]["lastMove"] != pair["ba"]["lastMove"]
    assert torch.equal(ab.stones, ba.stones)
    assert (
        ab.to_move,
        ab.moves_left,
        ab.opening,
        ab.terminal,
    ) == (
        ba.to_move,
        ba.moves_left,
        ba.opening,
        ba.terminal,
    )
    assert torch.equal(ab.current_turn, ba.current_turn)
    encoded_ab = encode_position(ab)
    encoded_ba = encode_position(ba)
    assert torch.equal(encoded_ab.node_features, encoded_ba.node_features)
    assert torch.equal(encoded_ab.global_features, encoded_ba.global_features)
    assert torch.equal(encoded_ab.legal_node_mask, encoded_ba.legal_node_mask)


def test_fixture_swap_relabels_colors_and_encodes_identically(
    conformance: dict,
) -> None:
    vector = conformance["swapEquivalences"][0]
    config = vector["config"]
    node_count = get_topology(config["rings"]).n
    assert vector["swapCode"] == node_count
    kept = position_from_fixture(vector["kept"], config)
    swapped = position_from_fixture(vector["swapped"], config)
    assert kept.swap_available and not swapped.swap_available
    assert swapped.swapped and not kept.swapped
    assert kept.to_move == 1 and swapped.to_move == 0
    exchanged = kept.stones.clone()
    exchanged[exchanged >= 0] = 1 - exchanged[exchanged >= 0]
    assert torch.equal(exchanged, swapped.stones)
    assert torch.equal(kept.previous_turn, swapped.previous_turn)
    assert torch.equal(kept.handicap_stones, swapped.handicap_stones)
    encoded_kept = encode_position(kept)
    encoded_swapped = encode_position(swapped)
    assert torch.equal(encoded_kept.node_features, encoded_swapped.node_features)
    expected = encoded_kept.global_features.clone()
    expected[22] = 0.0
    assert torch.equal(encoded_swapped.global_features, expected)
