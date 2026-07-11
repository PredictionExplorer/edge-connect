#![allow(missing_docs)]

use std::sync::Arc;

use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use star_engine::{
    Action, BitBoard, Board, D5Maps, GameError, GameState, MAX_NODES, Player, SUPPORTED_RINGS,
    ScoreResult, ScoringScratch, StateKey, Symmetry, terminal_value,
};

#[derive(Debug, Eq, PartialEq)]
struct StateSnapshot {
    key: StateKey,
    hash: u64,
    stones_placed: u16,
    last_move: Option<u16>,
    current_turn_moves: Vec<u16>,
    turn_count: u32,
    legal_actions: Vec<Action>,
}

fn supported_rings() -> impl Strategy<Value = u8> {
    prop::sample::select(SUPPORTED_RINGS.to_vec())
}

fn snapshot(state: &GameState) -> StateSnapshot {
    StateSnapshot {
        key: state.key(),
        hash: state.hash64(),
        stones_placed: state.stones_placed(),
        last_move: state.last_move(),
        current_turn_moves: state.current_turn_moves().to_vec(),
        turn_count: state.turn_count(),
        legal_actions: state.legal_actions().to_vec(),
    }
}

fn state_from_ranks(rings: u8, ranks: &[u16]) -> GameState {
    let board = Arc::new(Board::new(rings).expect("generated ring count is supported"));
    let mut state = GameState::new(board);
    for rank in ranks {
        if state.is_terminal() {
            break;
        }
        let placements: Vec<_> = state.legal_actions().placements.iter().collect();
        let node = placements[usize::from(*rank) % placements.len()];
        state
            .apply(Action::Place(node))
            .expect("rank helper chooses an empty node");
    }
    state
}

fn assert_state_invariants(state: &GameState) -> TestCaseResult {
    let board = state.board();
    let stones = state.stones();
    let occupied = stones[0].union(stones[1]);
    let board_full = occupied.count() == board.node_count();

    prop_assert!(stones[0].intersection(stones[1]).is_empty());
    prop_assert!(occupied.difference(board.node_mask()).is_empty());
    prop_assert_eq!(state.occupied(), occupied);
    prop_assert_eq!(state.stones_placed(), occupied.count());
    prop_assert_eq!(state.is_terminal(), board_full);
    prop_assert!(state.moves_left() <= 2);

    if state.is_opening() {
        prop_assert_eq!(state.to_move(), Player::Zero);
        prop_assert_eq!(state.moves_left(), 1);
        prop_assert!(occupied.is_empty());
        prop_assert!(state.current_turn_moves().is_empty());
        prop_assert_eq!(state.turn_count(), 0);
    } else {
        let expected_turn_moves = match state.moves_left() {
            0 => 2,
            1 => 1,
            2 => 0,
            _ => unreachable!("moves_left was bounded above"),
        };
        prop_assert_eq!(state.current_turn_moves().len(), expected_turn_moves);
        prop_assert!(state.turn_count() > 0);
    }

    for (index, &node) in state.current_turn_moves().iter().enumerate() {
        prop_assert_eq!(state.stone_at(node), Some(state.to_move()));
        prop_assert!(
            !state.current_turn_moves()[..index].contains(&node),
            "current-turn placements must be unique"
        );
    }
    if let Some(last_move) = state.last_move() {
        prop_assert!(state.stone_at(last_move).is_some());
    } else {
        prop_assert!(occupied.is_empty());
    }

    let legal = state.legal_actions();
    let actions = legal.to_vec();
    let expected_placements = if state.is_terminal() {
        BitBoard::empty()
    } else {
        board.node_mask().difference(occupied)
    };
    let expected_actions: Vec<_> = expected_placements.iter().map(Action::Place).collect();
    prop_assert_eq!(legal.placements, expected_placements);
    prop_assert_eq!(legal.len(), expected_actions.len());
    prop_assert_eq!(&actions, &expected_actions);

    for window in actions.windows(2) {
        let (Action::Place(left), Action::Place(right)) = (window[0], window[1]);
        prop_assert!(left < right);
    }
    for action in actions {
        prop_assert!(state.is_legal(action));
        prop_assert_eq!(Action::from_code(action.code()), Ok(action));
        let native_index = action
            .native_index(board)
            .expect("enumerated legal actions have native indexes");
        let Action::Place(node) = action;
        prop_assert_eq!(native_index, usize::from(node));
        prop_assert_eq!(Action::from_native_index(native_index, board), Ok(action));
    }

    prop_assert!(!state.is_legal(Action::Place(board.node_count())));
    prop_assert!(matches!(
        Action::Place(board.node_count()).native_index(board),
        Err(GameError::NodeOutOfBounds(_))
    ));
    prop_assert!(matches!(
        Action::from_native_index(usize::from(board.node_count()), board),
        Err(GameError::InvalidNativeActionIndex(_))
    ));
    prop_assert!(Action::from_code(-1).is_err());

    let rebuilt = GameState::from_parts(
        state.shared_board(),
        state.stones(),
        state.to_move(),
        state.moves_left(),
        state.is_opening(),
    );
    prop_assert!(
        rebuilt.is_ok(),
        "a state reached through legal play must be importable: {rebuilt:?}"
    );
    let rebuilt = rebuilt.expect("checked above");
    prop_assert_eq!(rebuilt.key(), state.key());
    prop_assert_eq!(rebuilt.hash64(), state.hash64());

    Ok(())
}

fn assert_full_score_contract(rings: u8, score: &ScoreResult) -> TestCaseResult {
    let total = score.players[0].total + score.players[1].total;
    let margin = score.players[0].total - score.players[1].total;
    prop_assert_eq!(score.contested_peries, 0);
    prop_assert_eq!(total, i16::from(5 * rings + 1));
    prop_assert_ne!(margin, 0);
    prop_assert_ne!(margin % 2, 0);
    prop_assert!((-151..=151).contains(&margin));
    prop_assert!(score.leader.is_some());
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(192))]

    #[test]
    fn legal_play_is_replayable_and_every_transition_is_exactly_reversible(
        rings in supported_rings(),
        ranks in prop::collection::vec(any::<u16>(), 0..128),
    ) {
        let board = Arc::new(Board::new(rings).unwrap());
        let mut state = GameState::new(Arc::clone(&board));
        let mut actions = Vec::new();

        assert_state_invariants(&state)?;
        for rank in ranks {
            if state.is_terminal() {
                break;
            }
            let placements: Vec<_> = state.legal_actions().placements.iter().collect();
            let action = Action::Place(placements[usize::from(rank) % placements.len()]);
            let before = snapshot(&state);
            let (transition, undo) = state.apply_reversible(action).unwrap();
            let after = snapshot(&state);
            prop_assert_eq!(transition.action, action);
            prop_assert_eq!(transition.player_before, before.key.to_move);
            prop_assert_eq!(transition.player_after, after.key.to_move);
            prop_assert_eq!(
                transition.turn_ended,
                transition.player_before != transition.player_after
            );
            prop_assert_eq!(transition.terminal, after.key.terminal);

            state.undo(undo);
            prop_assert_eq!(snapshot(&state), before);
            let replayed_transition = state.apply(action).unwrap();
            prop_assert_eq!(replayed_transition, transition);
            prop_assert_eq!(snapshot(&state), after);
            assert_state_invariants(&state)?;
            actions.push(action);
        }

        let final_snapshot = snapshot(&state);
        let mut replay = GameState::new(board);
        for action in actions {
            prop_assert!(replay.is_legal(action));
            replay.apply(action).unwrap();
        }
        prop_assert_eq!(snapshot(&replay), final_snapshot);
    }

    #[test]
    fn completed_pair_order_has_one_semantic_key(
        rings in supported_rings(),
        opening_rank in any::<u16>(),
        first_rank in any::<u16>(),
        second_rank in any::<u16>(),
    ) {
        let board = Arc::new(Board::new(rings).unwrap());
        let opening = opening_rank % board.node_count();
        let mut base = GameState::new(board);
        base.apply(Action::Place(opening)).unwrap();

        let placements: Vec<_> = base.legal_actions().placements.iter().collect();
        let first = placements[usize::from(first_rank) % placements.len()];
        let remaining: Vec<_> = placements
            .into_iter()
            .filter(|node| *node != first)
            .collect();
        let second = remaining[usize::from(second_rank) % remaining.len()];

        let mut ab = base.clone();
        ab.apply(Action::Place(first)).unwrap();
        ab.apply(Action::Place(second)).unwrap();
        let mut ba = base;
        ba.apply(Action::Place(second)).unwrap();
        ba.apply(Action::Place(first)).unwrap();

        prop_assert_eq!(ab.key(), ba.key());
        prop_assert_eq!(ab.hash64(), ba.hash64());
        prop_assert_eq!(ab.legal_actions().to_vec(), ba.legal_actions().to_vec());
        prop_assert_eq!(
            ScoringScratch::default().score_state(&ab),
            ScoringScratch::default().score_state(&ba)
        );
        prop_assert_eq!(ab.last_move(), Some(second));
        prop_assert_eq!(ba.last_move(), Some(first));
    }

    #[test]
    fn d5_and_color_swaps_preserve_the_scoring_contract(
        rings in supported_rings(),
        ranks in prop::collection::vec(any::<u16>(), 0..80),
        symmetry_index in 0_u8..10,
        action_rank in any::<u16>(),
    ) {
        let state = state_from_ranks(rings, &ranks);
        let maps = D5Maps::new(state.board());
        let symmetry = Symmetry::from_index(symmetry_index).unwrap();
        let transformed = maps.state(symmetry, &state);
        let round_trip = maps.state(symmetry.inverse(), &transformed);

        prop_assert_eq!(snapshot(&round_trip), snapshot(&state));
        prop_assert_eq!(
            transformed.legal_actions().placements,
            maps.bitboard(symmetry, state.legal_actions().placements)
        );
        prop_assert_eq!(transformed.to_move(), state.to_move());
        prop_assert_eq!(transformed.moves_left(), state.moves_left());
        prop_assert_eq!(transformed.is_terminal(), state.is_terminal());

        let mut scratch = ScoringScratch::default();
        let original_score = scratch.score_state(&state);
        let transformed_score = scratch.score_state(&transformed);
        prop_assert_eq!(transformed_score.players, original_score.players);
        prop_assert_eq!(
            transformed_score.contested_peries,
            original_score.contested_peries
        );
        prop_assert_eq!(transformed_score.leader, original_score.leader);
        prop_assert_eq!(
            transformed_score.alive_stones,
            maps.bitboard(symmetry, original_score.alive_stones)
        );
        for node in 0..state.board().node_count() {
            prop_assert_eq!(
                transformed_score.owner(maps.node(symmetry, node)),
                original_score.owner(node)
            );
        }

        let stones = state.stones();
        let color_swapped = scratch.score(state.board(), [stones[1], stones[0]]);
        prop_assert_eq!(
            color_swapped.players,
            [original_score.players[1], original_score.players[0]]
        );
        prop_assert_eq!(
            color_swapped.contested_peries,
            original_score.contested_peries
        );
        prop_assert_eq!(color_swapped.alive_stones, original_score.alive_stones);
        prop_assert_eq!(
            color_swapped.leader,
            original_score.leader.map(Player::opponent)
        );
        for node in 0..state.board().node_count() {
            prop_assert_eq!(
                color_swapped.owner(node),
                original_score.owner(node).map(Player::opponent)
            );
        }
        prop_assert_eq!(
            original_score.players[0].award + original_score.players[1].award,
            0
        );
        prop_assert_eq!(
            original_score.players[0].peries
                + original_score.players[1].peries
                + i16::try_from(original_score.contested_peries).unwrap(),
            i16::try_from(state.board().peri_count()).unwrap()
        );

        if !state.is_terminal() {
            let legal = state.legal_actions().to_vec();
            let action = legal[usize::from(action_rank) % legal.len()];
            let mut next = state.clone();
            next.apply(action).unwrap();
            let expected = maps.state(symmetry, &next);

            let mut transformed_next = transformed;
            let mapped_action = maps.action(symmetry, action);
            prop_assert!(transformed_next.is_legal(mapped_action));
            transformed_next.apply(mapped_action).unwrap();
            prop_assert_eq!(snapshot(&transformed_next), snapshot(&expected));
        }
    }

    #[test]
    fn malformed_packed_states_are_rejected(
        rings in supported_rings(),
        node_rank in any::<u16>(),
        metadata_case in 0_u8..5,
    ) {
        let board = Arc::new(Board::new(rings).unwrap());
        let node = node_rank % board.node_count();

        let mut overlap = BitBoard::empty();
        overlap.insert(node);
        prop_assert!(matches!(
            GameState::from_parts(
                Arc::clone(&board),
                [overlap, overlap],
                Player::Zero,
                2,
                false,
            ),
            Err(GameError::OverlappingStones)
        ));

        let mut outside = BitBoard::empty();
        prop_assert!(outside.insert(board.node_count()));
        prop_assert!(matches!(
            GameState::from_parts(
                Arc::clone(&board),
                [outside, BitBoard::empty()],
                Player::Zero,
                2,
                false,
            ),
            Err(GameError::StonesOutsideBoard)
        ));

        let empty = [BitBoard::empty(); 2];
        let full = [board.node_mask(), BitBoard::empty()];
        let (stones, to_move, moves_left, opening) = match metadata_case {
            0 => (empty, Player::Zero, 3, false),
            1 => (empty, Player::Zero, 0, false),
            2 => (empty, Player::One, 1, true),
            3 => {
                let mut occupied = BitBoard::empty();
                occupied.insert(node);
                ([occupied, BitBoard::empty()], Player::Zero, 1, true)
            }
            4 => (full, Player::Zero, 2, false),
            _ => unreachable!(),
        };
        prop_assert!(matches!(
            GameState::from_parts(board, stones, to_move, moves_left, opening),
            Err(GameError::InvalidTurnMetadata)
        ));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn every_supported_ring_has_decisive_full_arbitrary_patterns(
        pattern in prop::collection::vec(any::<bool>(), MAX_NODES),
    ) {
        let mut scratch = ScoringScratch::default();
        for rings in SUPPORTED_RINGS {
            let board = Board::new(rings).unwrap();
            let mut stones = [BitBoard::empty(); 2];
            for node in 0..board.node_count() {
                stones[usize::from(pattern[usize::from(node)])].insert(node);
            }
            let score = scratch.score(&board, stones);
            assert_full_score_contract(rings, &score)?;
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn every_supported_ring_has_decisive_legal_placement_permutations(
        ordering_keys in prop::collection::vec(any::<u64>(), MAX_NODES),
    ) {
        for rings in SUPPORTED_RINGS {
            let board = Arc::new(Board::new(rings).unwrap());
            let mut order: Vec<_> = (0..board.node_count()).collect();
            order.sort_by_key(|node| (ordering_keys[usize::from(*node)], *node));
            let mut state = GameState::new(board);
            for node in order {
                prop_assert!(state.is_legal(Action::Place(node)));
                state.apply(Action::Place(node)).unwrap();
            }
            prop_assert!(state.is_terminal());
            prop_assert!(state.legal_actions().is_empty());
            let score = ScoringScratch::default().score_state(&state);
            assert_full_score_contract(rings, &score)?;
            let value = terminal_value(&state).expect("full legal play is terminal");
            prop_assert!(value == -1.0 || value == 1.0);
        }
    }
}
