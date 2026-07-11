#![allow(missing_docs)]

use std::sync::{Arc, LazyLock};

use serde::Deserialize;
use star_engine::{
    ACTION_LAYOUT_SCHEMA, Action, BitBoard, Board, CONFORMANCE_SCHEMA, D5Maps, FEATURE_SCHEMA,
    GameState, Player, PlayerScore, RULES_HASH, RULES_HASH_VALUE, RULES_SCHEMA, RULES_VERSION,
    SUPPORTED_RINGS, ScoreResult, ScoringScratch, Symmetry, rules_hash, terminal_value,
};

const FIXTURE_JSON: &str = include_str!("../../../../testdata/star/conformance-v2.json");

static FIXTURE: LazyLock<ConformanceFixture> = LazyLock::new(|| {
    serde_json::from_str(FIXTURE_JSON).expect("generated conformance fixture must deserialize")
});

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConformanceFixture {
    schema: String,
    schemas: SchemasFixture,
    rules: RulesFixture,
    outcome_encoding: OutcomeEncodingFixture,
    action_encoding: ActionEncodingFixture,
    action_layouts: ActionLayoutsFixture,
    boards: Vec<BoardFixture>,
    scores: Vec<ScoreFixture>,
    games: Vec<GameFixture>,
    pair_equivalences: Vec<PairEquivalenceFixture>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SchemasFixture {
    rules: String,
    conformance: String,
    model_features: String,
    action_layout: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RulesFixture {
    version: u32,
    hash_algorithm: String,
    hash: String,
    canonical: String,
    contract: serde_json::Value,
}

#[derive(Deserialize)]
struct OutcomeEncodingFixture {
    loss: u8,
    win: u8,
    value: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ActionEncodingFixture {
    placement_code: String,
    legal_order: String,
    native_layout: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ActionLayoutsFixture {
    schema: String,
    model_feature_schema: String,
    mixed_batches: Vec<MixedBatchFixture>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MixedBatchFixture {
    rings: Vec<u8>,
    maximum_nodes: usize,
    batch_action_count: usize,
    rows: Vec<ActionLayoutRowFixture>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ActionLayoutRowFixture {
    rings: u8,
    node_count: usize,
    native: NativeLayoutFixture,
    padded: PaddedLayoutFixture,
    examples: Vec<ActionExampleFixture>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct NativeLayoutFixture {
    action_count: usize,
    placement_slots: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PaddedLayoutFixture {
    action_count: usize,
    placement_slots: Vec<usize>,
    padding_slots: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ActionExampleFixture {
    action: ActionFixture,
    wire_code: i32,
    native_index: usize,
    padded_index: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoardFixture {
    rings: u8,
    node_count: u16,
    perimeter_count: u16,
    edge_count: usize,
    maximum_degree: usize,
    sector_of: Vec<u8>,
    ring_of: Vec<u8>,
    position_of: Vec<u8>,
    perimeter_mask: Vec<u8>,
    quark_mask: Vec<u8>,
    labels: Vec<String>,
    adjacency_offsets: Vec<u16>,
    adjacency: Vec<u16>,
    bridge: Vec<u16>,
    nodes: Vec<NodeFixture>,
    symmetries: Vec<SymmetryFixture>,
}

#[derive(Deserialize)]
struct NodeFixture {
    id: u16,
    sector: u8,
    ring: u8,
    position: u8,
    label: String,
    perimeter: bool,
    quark: bool,
    adjacent: Vec<u16>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SymmetryFixture {
    id: String,
    kind: String,
    turns: u8,
    inverse_id: String,
    map: Vec<u16>,
    inverse_map: Vec<u16>,
}

#[derive(Deserialize)]
struct ScoreFixture {
    id: String,
    rings: u8,
    stones: Vec<i8>,
    expected: ScoreExpected,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScoreExpected {
    players: Vec<PlayerScoreFixture>,
    node_owner: Vec<i8>,
    alive_stone: Vec<u8>,
    contested_peries: u16,
    leader: i8,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlayerScoreFixture {
    peries: i16,
    quarks: i16,
    stars: i16,
    quark_peri: i16,
    award: i16,
    total: i16,
}

#[derive(Deserialize)]
struct GameFixture {
    id: String,
    config: GameConfigFixture,
    actions: Vec<ActionFixture>,
    #[serde(rename = "actionCodes")]
    action_codes: Vec<i32>,
    states: Vec<TraceStateFixture>,
    terminal: TerminalFixture,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GameConfigFixture {
    rings: u8,
    mode: String,
    pie_rule: bool,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ActionFixture {
    Place { node: u16 },
}

impl ActionFixture {
    fn to_native(&self) -> Action {
        match self {
            Self::Place { node } => Action::Place(*node),
        }
    }

    fn node(&self) -> u16 {
        match self {
            Self::Place { node } => *node,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TraceStateFixture {
    after_actions: usize,
    opening: bool,
    stones: Vec<i8>,
    stones_placed: u16,
    to_move: u8,
    moves_left: u8,
    mid_turn: bool,
    over: bool,
    can_swap: bool,
    swapped: bool,
    last_move: i32,
    current_turn_moves: Vec<u16>,
    turn_count: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TerminalFixture {
    reason: String,
    winner: i8,
    score: ScoreExpected,
    values_by_player: Vec<i8>,
    outcome_classes_by_player: Vec<u8>,
    score_margins_by_player: Vec<i16>,
    value_perspective: ValuePerspectiveFixture,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ValuePerspectiveFixture {
    kind: String,
    player: u8,
    value: i8,
    outcome_class: u8,
    score_margin: i16,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairEquivalenceFixture {
    config: GameConfigFixture,
    pair: PairFixture,
    ab: PairPathFixture,
    ba: PairPathFixture,
    equivalent_fields: Vec<String>,
    excluded_presentation_fields: Vec<String>,
}

#[derive(Deserialize)]
struct PairFixture {
    a: u16,
    b: u16,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairPathFixture {
    actions: Vec<ActionFixture>,
    action_codes: Vec<i32>,
    semantic_state: SemanticStateFixture,
    last_move: u16,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticStateFixture {
    rings: u8,
    stones: Vec<i8>,
    stones_placed: u16,
    to_move: u8,
    moves_left: u8,
    opening: bool,
    mid_turn: bool,
    terminal: bool,
    current_turn_moves: Vec<u16>,
    turn_count: u32,
}

#[test]
fn finalized_v2_schema_hash_and_encodings_match_runtime_constants() {
    let fixture = &*FIXTURE;
    assert_eq!(fixture.schema, CONFORMANCE_SCHEMA);
    assert_eq!(fixture.schemas.conformance, CONFORMANCE_SCHEMA);
    assert_eq!(fixture.schemas.rules, RULES_SCHEMA);
    assert_eq!(fixture.schemas.model_features, FEATURE_SCHEMA);
    assert_eq!(fixture.schemas.action_layout, ACTION_LAYOUT_SCHEMA);
    assert_eq!(fixture.rules.version, RULES_VERSION);
    assert_eq!(fixture.rules.contract["schema"], RULES_SCHEMA);
    assert_eq!(fixture.rules.contract["version"], RULES_VERSION);
    assert_eq!(fixture.rules.contract["variant"], "double-star");
    assert_eq!(fixture.rules.hash_algorithm, "fnv1a64");
    assert_eq!(fixture.rules.hash, RULES_HASH);
    assert_eq!(
        fnv1a64(fixture.rules.canonical.as_bytes()),
        RULES_HASH_VALUE
    );
    assert_eq!(rules_hash(), RULES_HASH_VALUE);
    assert_eq!(fixture.outcome_encoding.loss, 0);
    assert_eq!(fixture.outcome_encoding.win, 1);
    assert_eq!(fixture.outcome_encoding.value, "P(win)-P(loss)");
    assert_eq!(fixture.action_encoding.placement_code, "dense node id");
    assert_eq!(
        fixture.action_encoding.legal_order,
        "ascending legal placement node ids"
    );
    assert_eq!(fixture.action_encoding.native_layout, "node u at index u");
    assert_eq!(fixture.action_layouts.schema, ACTION_LAYOUT_SCHEMA);
    assert_eq!(fixture.action_layouts.model_feature_schema, FEATURE_SCHEMA);
}

#[test]
fn generated_nodes_only_action_layouts_match_wire_and_native_order() {
    let fixture = &*FIXTURE;
    assert_eq!(fixture.action_layouts.mixed_batches.len(), 1);
    let batch = &fixture.action_layouts.mixed_batches[0];
    assert_eq!(batch.rings, SUPPORTED_RINGS);
    assert_eq!(batch.batch_action_count, batch.maximum_nodes);
    assert_eq!(
        batch.rings,
        batch.rows.iter().map(|row| row.rings).collect::<Vec<_>>()
    );
    for row in &batch.rows {
        let board = Board::new(row.rings).unwrap();
        let node_count = usize::from(board.node_count());
        assert_eq!(row.node_count, node_count);
        assert_eq!(row.native.action_count, node_count);
        assert_eq!(row.native.placement_slots, [0, node_count - 1]);
        assert_eq!(row.padded.action_count, batch.maximum_nodes);
        assert_eq!(row.padded.placement_slots, [0, node_count - 1]);
        if node_count == batch.maximum_nodes {
            assert!(row.padded.padding_slots.is_empty());
        } else {
            assert_eq!(
                row.padded.padding_slots,
                [node_count, batch.maximum_nodes - 1]
            );
        }
        for example in &row.examples {
            let action = example.action.to_native();
            assert_eq!(action.code(), example.wire_code);
            assert_eq!(action.native_index(&board).unwrap(), example.native_index);
            assert_eq!(
                Action::from_native_index(example.native_index, &board).unwrap(),
                action
            );
            let Action::Place(node) = action;
            assert_eq!(usize::from(node), example.padded_index);
        }
    }
}

#[test]
fn all_generated_board_and_d5_vectors_match() {
    let fixture = &*FIXTURE;
    assert_eq!(fixture.boards.len(), SUPPORTED_RINGS.len());
    for (&expected_rings, expected) in SUPPORTED_RINGS.iter().zip(&fixture.boards) {
        assert_eq!(expected.rings, expected_rings);
        let board = Board::new(expected.rings).unwrap();
        let maps = D5Maps::new(&board);
        assert_eq!(board.node_count(), expected.node_count);
        assert_eq!(board.peri_count(), expected.perimeter_count);
        assert_eq!(board.edge_count(), expected.edge_count);
        assert_eq!(
            (0..board.node_count())
                .map(|node| board.neighbors(node).len())
                .max()
                .unwrap(),
            expected.maximum_degree
        );

        let mut offsets = Vec::with_capacity(usize::from(board.node_count()) + 1);
        let mut adjacency = Vec::new();
        offsets.push(0_u16);
        for node in 0..board.node_count() {
            let index = usize::from(node);
            assert_eq!(board.sector(node), expected.sector_of[index]);
            assert_eq!(board.ring(node), expected.ring_of[index]);
            assert_eq!(board.position(node), expected.position_of[index]);
            assert_eq!(
                u8::from(board.is_peri(node)),
                expected.perimeter_mask[index]
            );
            assert_eq!(u8::from(board.is_quark(node)), expected.quark_mask[index]);
            assert_eq!(board.label(node), expected.labels[index]);
            adjacency.extend_from_slice(board.neighbors(node));
            offsets.push(u16::try_from(adjacency.len()).unwrap());

            let expected_node = &expected.nodes[index];
            assert_eq!(expected_node.id, node);
            assert_eq!(expected_node.sector, board.sector(node));
            assert_eq!(expected_node.ring, board.ring(node));
            assert_eq!(expected_node.position, board.position(node));
            assert_eq!(expected_node.label, board.label(node));
            assert_eq!(expected_node.perimeter, board.is_peri(node));
            assert_eq!(expected_node.quark, board.is_quark(node));
            let mut sorted_neighbors = board.neighbors(node).to_vec();
            sorted_neighbors.sort_unstable();
            assert_eq!(expected_node.adjacent, sorted_neighbors);
        }
        assert_eq!(offsets, expected.adjacency_offsets);
        assert_eq!(adjacency, expected.adjacency);
        assert_eq!(board.bridge(), expected.bridge.as_slice());

        assert_eq!(expected.symmetries.len(), Symmetry::ALL.len());
        for (index, (symmetry, expected_symmetry)) in Symmetry::ALL
            .iter()
            .copied()
            .zip(&expected.symmetries)
            .enumerate()
        {
            let expected_id = if index < 5 {
                format!("r{index}")
            } else {
                format!("f{}", index - 5)
            };
            let inverse = symmetry.inverse();
            let expected_inverse_id = if inverse.is_reflection() {
                format!("f{}", inverse.arm_offset())
            } else {
                format!("r{}", inverse.arm_offset())
            };
            assert_eq!(expected_symmetry.id, expected_id);
            assert_eq!(
                expected_symmetry.kind,
                if symmetry.is_reflection() {
                    "reflection"
                } else {
                    "rotation"
                }
            );
            assert_eq!(expected_symmetry.turns, symmetry.arm_offset());
            assert_eq!(expected_symmetry.inverse_id, expected_inverse_id);
            assert_eq!(maps.map(symmetry), expected_symmetry.map);
            assert_eq!(maps.map(inverse), expected_symmetry.inverse_map);
        }
    }
}

#[test]
fn every_generated_scoring_vector_matches() {
    let fixture = &*FIXTURE;
    assert_eq!(fixture.scores.len(), 6 * SUPPORTED_RINGS.len());
    let mut scratch = ScoringScratch::default();
    for vector in &fixture.scores {
        let board = Board::new(vector.rings).unwrap();
        let stones = bitboards_from_fixture(&vector.stones);
        let actual = scratch.score(&board, stones);
        assert_score(&vector.id, &actual, &vector.expected, board.node_count());
    }
}

#[test]
fn every_generated_full_board_game_and_binary_terminal_value_matches() {
    let fixture = &*FIXTURE;
    assert_eq!(fixture.games.len(), SUPPORTED_RINGS.len());
    for (&rings, trace) in SUPPORTED_RINGS.iter().zip(&fixture.games) {
        assert_eq!(trace.config.rings, rings);
        assert_eq!(trace.config.mode, "double");
        assert!(!trace.config.pie_rule);
        assert_eq!(trace.id, format!("rings-{rings}-board-full"));
        assert_eq!(trace.actions.len(), trace.action_codes.len());
        assert_eq!(trace.states.len(), trace.actions.len() + 1);
        let board = Arc::new(Board::new(rings).unwrap());
        let mut state = GameState::new(board);

        for (step, expected_state) in trace.states.iter().enumerate() {
            assert_eq!(expected_state.after_actions, step);
            assert_trace_state(&trace.id, &state, expected_state);
            assert_legal_action_contract(&state);
            if let Some(action_fixture) = trace.actions.get(step) {
                let action = action_fixture.to_native();
                assert_eq!(action.code(), trace.action_codes[step]);
                assert!(state.is_legal(action));
                state.apply(action).unwrap();
            }
        }

        assert!(state.is_terminal());
        assert!(state.legal_actions().is_empty());
        assert_eq!(trace.terminal.reason, "board-full");
        let score = ScoringScratch::default().score_state(&state);
        assert_score(
            &trace.id,
            &score,
            &trace.terminal.score,
            state.board().node_count(),
        );
        let winner = score
            .leader
            .expect("generated full boards must have decisive winners");
        assert_eq!(trace.terminal.winner, winner as i8);
        for player in [Player::Zero, Player::One] {
            let index = player.index();
            let value = score.outcome_for(player).unwrap() as i8;
            let margin = score.players[index].total - score.players[1 - index].total;
            assert_eq!(trace.terminal.values_by_player[index], value);
            assert_eq!(
                trace.terminal.outcome_classes_by_player[index],
                outcome_class(value)
            );
            assert_eq!(trace.terminal.score_margins_by_player[index], margin);
        }
        let perspective = &trace.terminal.value_perspective;
        assert_eq!(perspective.kind, "toMove");
        assert_eq!(perspective.player, state.to_move() as u8);
        let value = terminal_value(&state).unwrap() as i8;
        let player = state.to_move().index();
        let margin = score.players[player].total - score.players[1 - player].total;
        assert!(matches!(value, -1 | 1));
        assert_eq!(perspective.value, value);
        assert_eq!(perspective.outcome_class, outcome_class(value));
        assert_eq!(perspective.score_margin, margin);
    }
}

#[test]
fn generated_ab_ba_paths_share_the_semantic_key() {
    let fixture = &*FIXTURE;
    assert_eq!(fixture.pair_equivalences.len(), 1);
    for pair in &fixture.pair_equivalences {
        assert_eq!(pair.config.mode, "double");
        assert!(!pair.config.pie_rule);
        assert_eq!(
            pair.equivalent_fields,
            [
                "rings",
                "stones",
                "stonesPlaced",
                "toMove",
                "movesLeft",
                "opening",
                "midTurn",
                "terminal",
                "currentTurnMoves",
                "turnCount",
            ]
        );
        assert_eq!(pair.excluded_presentation_fields, ["lastMove"]);
        let ab = replay_pair_path(pair.config.rings, &pair.ab);
        let ba = replay_pair_path(pair.config.rings, &pair.ba);
        assert_eq!(pair.pair.a, pair.ab.actions[1].node());
        assert_eq!(pair.pair.b, pair.ab.actions[2].node());
        assert_eq!(ab.key(), ba.key());
        assert_eq!(ab.hash64(), ba.hash64());
        assert_eq!(ab.last_move(), Some(pair.ab.last_move));
        assert_eq!(ba.last_move(), Some(pair.ba.last_move));
        assert_ne!(ab.last_move(), ba.last_move());
    }
}

fn replay_pair_path(rings: u8, path: &PairPathFixture) -> GameState {
    let board = Arc::new(Board::new(rings).unwrap());
    let mut state = GameState::new(board);
    assert_eq!(path.actions.len(), path.action_codes.len());
    for (action, code) in path.actions.iter().zip(&path.action_codes) {
        let action = action.to_native();
        assert_eq!(action.code(), *code);
        state.apply(action).unwrap();
    }
    assert_semantic_state(&state, &path.semantic_state);
    state
}

fn assert_trace_state(id: &str, state: &GameState, expected: &TraceStateFixture) {
    assert_stones(id, state, &expected.stones);
    assert_eq!(state.stones_placed(), expected.stones_placed, "{id}");
    assert_eq!(state.to_move() as u8, expected.to_move, "{id}");
    assert_eq!(state.moves_left(), expected.moves_left, "{id}");
    assert_eq!(state.is_opening(), expected.opening, "{id}");
    assert_eq!(state.is_mid_turn(), expected.mid_turn, "{id}");
    assert_eq!(state.is_terminal(), expected.over, "{id}");
    assert!(!expected.can_swap, "{id}");
    assert!(!expected.swapped, "{id}");
    assert_eq!(
        state.last_move().map_or(-1, i32::from),
        expected.last_move,
        "{id}"
    );
    assert_eq!(
        state.current_turn_moves(),
        expected.current_turn_moves,
        "{id}"
    );
    assert_eq!(state.turn_count(), expected.turn_count, "{id}");
}

fn assert_semantic_state(state: &GameState, expected: &SemanticStateFixture) {
    assert_eq!(state.board().rings(), expected.rings);
    assert_stones("pair-equivalence", state, &expected.stones);
    assert_eq!(state.stones_placed(), expected.stones_placed);
    assert_eq!(state.to_move() as u8, expected.to_move);
    assert_eq!(state.moves_left(), expected.moves_left);
    assert_eq!(state.is_opening(), expected.opening);
    assert_eq!(state.is_mid_turn(), expected.mid_turn);
    assert_eq!(state.is_terminal(), expected.terminal);
    assert_eq!(state.current_turn_moves(), expected.current_turn_moves);
    assert_eq!(state.turn_count(), expected.turn_count);
}

fn assert_stones(id: &str, state: &GameState, expected: &[i8]) {
    assert_eq!(
        expected.len(),
        usize::from(state.board().node_count()),
        "{id}"
    );
    for (node, expected_owner) in expected.iter().copied().enumerate() {
        let actual = state
            .stone_at(u16::try_from(node).unwrap())
            .map_or(-1, |player| player as i8);
        assert_eq!(actual, expected_owner, "{id}: stone {node}");
    }
}

fn assert_legal_action_contract(state: &GameState) {
    let actions = state.legal_actions().to_vec();
    if state.is_terminal() {
        assert!(actions.is_empty());
        return;
    }
    for (expected_node, action) in state.legal_actions().placements.iter().zip(&actions) {
        let Action::Place(node) = *action;
        assert_eq!(node, expected_node);
        assert!(state.stone_at(node).is_none());
        assert_eq!(action.code(), i32::from(node));
        assert_eq!(
            action.native_index(state.board()).unwrap(),
            usize::from(node)
        );
    }
    assert!(
        actions
            .windows(2)
            .all(|window| window[0].code() < window[1].code())
    );
}

fn bitboards_from_fixture(stones: &[i8]) -> [BitBoard; 2] {
    let mut bitboards = [BitBoard::empty(); 2];
    for (node, owner) in stones.iter().copied().enumerate() {
        match owner {
            -1 => {}
            0 | 1 => {
                bitboards[owner as usize].insert(u16::try_from(node).unwrap());
            }
            _ => panic!("invalid fixture stone owner {owner}"),
        }
    }
    bitboards
}

fn assert_score(id: &str, actual: &ScoreResult, expected: &ScoreExpected, node_count: u16) {
    assert_eq!(expected.players.len(), 2, "{id}");
    for (actual, expected) in actual.players.iter().zip(&expected.players) {
        assert_eq!(
            *actual,
            PlayerScore {
                peries: expected.peries,
                quarks: expected.quarks,
                stars: expected.stars,
                quark_peri: expected.quark_peri,
                award: expected.award,
                total: expected.total,
            },
            "{id}"
        );
    }
    assert_eq!(
        &actual.node_owner[..usize::from(node_count)],
        expected.node_owner,
        "{id}"
    );
    for (node, expected_alive) in expected.alive_stone.iter().copied().enumerate() {
        assert_eq!(
            u8::from(actual.alive_stones.contains(u16::try_from(node).unwrap())),
            expected_alive,
            "{id}: alive stone {node}"
        );
    }
    assert_eq!(actual.contested_peries, expected.contested_peries, "{id}");
    assert_eq!(
        actual.leader.map_or(-1, |player| player as i8),
        expected.leader,
        "{id}"
    );
}

const fn outcome_class(value: i8) -> u8 {
    match value {
        -1 => 0,
        1 => 1,
        _ => panic!("terminal value must be -1 or 1"),
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
