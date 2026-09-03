#![allow(missing_docs)]

use std::sync::Arc;

use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use star_engine::{
    Action, BitBoard, Board, D5Maps, GameError, GameState, MAX_HANDICAP, MAX_NODES, Mode, Player,
    SUPPORTED_RINGS, ScoreResult, ScoringScratch, StateKey, StateParts, Symmetry, Variant,
    terminal_value,
};

#[derive(Debug, Eq, PartialEq)]
struct StateSnapshot {
    key: StateKey,
    hash: u64,
    stones_placed: u16,
    last_move: Option<u16>,
    current_turn_moves: Vec<u16>,
    previous_turn_moves: Vec<u16>,
    own_previous_turn_moves: Vec<u16>,
    handicap_stones: BitBoard,
    turn_count: u32,
    swap_available: bool,
    swapped: bool,
    legal_actions: Vec<Action>,
}

fn supported_rings() -> impl Strategy<Value = u8> {
    prop::sample::select(SUPPORTED_RINGS.to_vec())
}

fn variants() -> impl Strategy<Value = Variant> {
    (
        prop::sample::select(vec![Mode::Classic, Mode::Double]),
        1_u8..=MAX_HANDICAP,
        any::<bool>(),
    )
        .prop_map(|(mode, handicap, pie)| {
            // Pie requires handicap 1; fold the flag away for larger handicaps.
            Variant::new(mode, handicap, pie && handicap == 1).expect("valid variant")
        })
}

fn snapshot(state: &GameState) -> StateSnapshot {
    StateSnapshot {
        key: state.key(),
        hash: state.hash64(),
        stones_placed: state.stones_placed(),
        last_move: state.last_move(),
        current_turn_moves: state.current_turn_moves().to_vec(),
        previous_turn_moves: state.previous_turn_moves().to_vec(),
        own_previous_turn_moves: state.own_previous_turn_moves().to_vec(),
        handicap_stones: state.handicap_stones(),
        turn_count: state.turn_count(),
        swap_available: state.swap_available(),
        swapped: state.swapped(),
        legal_actions: state.legal_actions().to_vec(),
    }
}

/// Chooses a legal action from a rank; a swap is taken when the rank is odd
/// and available so both pie branches are exercised.
fn ranked_action(state: &GameState, rank: u16) -> Action {
    if state.swap_available() && rank % 2 == 1 {
        return Action::Swap;
    }
    let placements: Vec<_> = state.legal_actions().placements.iter().collect();
    Action::Place(placements[usize::from(rank) % placements.len()])
}

fn state_from_ranks(rings: u8, variant: Variant, ranks: &[u16]) -> GameState {
    let board = Arc::new(Board::new(rings).expect("generated ring count is supported"));
    let mut state = GameState::with_variant(board, variant);
    for rank in ranks {
        if state.is_terminal() {
            break;
        }
        state
            .apply(ranked_action(&state, *rank))
            .expect("rank helper chooses a legal action");
    }
    state
}

fn assert_state_invariants(state: &GameState) -> TestCaseResult {
    let board = state.board();
    let variant = state.variant();
    let stones = state.stones();
    let occupied = stones[0].union(stones[1]);
    let board_full = occupied.count() == board.node_count();
    let turn_size = variant.turn_size();

    prop_assert!(stones[0].intersection(stones[1]).is_empty());
    prop_assert!(occupied.difference(board.node_mask()).is_empty());
    prop_assert_eq!(state.occupied(), occupied);
    prop_assert_eq!(state.stones_placed(), occupied.count());
    prop_assert_eq!(state.is_terminal(), board_full);
    prop_assert!(state.moves_left() <= state.current_turn_total());
    prop_assert_eq!(state.turn_size(), turn_size);

    if state.is_opening() {
        prop_assert_eq!(state.to_move(), Player::Zero);
        prop_assert!(stones[1].is_empty());
        prop_assert!(state.moves_left() >= 1);
        prop_assert!(state.moves_left() <= variant.handicap());
        prop_assert_eq!(
            stones[0].count(),
            u16::from(variant.handicap() - state.moves_left())
        );
        prop_assert_eq!(state.current_turn_set(), stones[0]);
        prop_assert_eq!(state.handicap_stones(), stones[0]);
        prop_assert!(state.previous_turn_moves().is_empty());
        prop_assert!(state.own_previous_turn_moves().is_empty());
        prop_assert_eq!(state.turn_count(), 0);
        prop_assert!(!state.swap_available());
        prop_assert!(!state.swapped());
        prop_assert_eq!(state.is_pie_pending(), variant.pie() && occupied.is_empty());
    } else {
        prop_assert!(!state.is_pie_pending());
        if board_full {
            prop_assert!(state.moves_left() < turn_size);
        } else {
            prop_assert!(state.moves_left() >= 1);
            prop_assert!(state.moves_left() <= turn_size);
        }
        prop_assert_eq!(
            state.current_turn_moves().len(),
            usize::from(turn_size - state.moves_left())
        );
        prop_assert!(state.turn_count() > 0);
        prop_assert_eq!(
            state.handicap_stones().count(),
            u16::from(variant.handicap())
        );
        prop_assert!(state.handicap_stones().difference(occupied).is_empty());
    }
    prop_assert_eq!(
        state.is_mid_turn(),
        !state.current_turn_moves().is_empty() && state.moves_left() > 0
    );

    // Swap availability is exactly the post-opening position of a pie game.
    if state.swap_available() {
        prop_assert!(variant.pie());
        prop_assert!(!state.swapped());
        prop_assert_eq!(state.to_move(), Player::One);
        prop_assert_eq!(state.turn_count(), 1);
        prop_assert_eq!(stones[0].count(), 1);
        prop_assert!(stones[1].is_empty());
        prop_assert!(state.is_legal(Action::Swap));
    } else {
        prop_assert!(!state.is_legal(Action::Swap));
    }
    if state.swapped() {
        prop_assert!(variant.pie());
    }

    // History ownership: the current and own-previous turns belong to the
    // mover, the previous turn to the opponent, and the sets are disjoint.
    let own = stones[state.to_move().index()];
    let opponent = stones[state.to_move().opponent().index()];
    prop_assert!(state.current_turn_set().difference(own).is_empty());
    prop_assert!(state.own_previous_turn_set().difference(own).is_empty());
    prop_assert!(state.previous_turn_set().difference(opponent).is_empty());
    prop_assert!(
        state
            .current_turn_set()
            .intersection(state.own_previous_turn_set())
            .is_empty()
    );
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

    let node_count = board.node_count();
    for window in actions.windows(2) {
        prop_assert!(window[0].code(node_count) < window[1].code(node_count));
    }
    for action in actions {
        prop_assert!(state.is_legal(action));
        prop_assert_eq!(
            Action::from_code(action.code(node_count), node_count),
            Ok(action)
        );
        let native_index = action
            .native_index(board)
            .expect("enumerated legal actions have native indexes");
        let node = action.node().expect("legal actions are placements");
        prop_assert_eq!(native_index, usize::from(node));
        prop_assert_eq!(Action::from_native_index(native_index, board), Ok(action));
    }

    prop_assert!(!state.is_legal(Action::Place(node_count)));
    prop_assert!(matches!(
        Action::Place(node_count).native_index(board),
        Err(GameError::NodeOutOfBounds(_))
    ));
    prop_assert!(matches!(
        Action::from_native_index(usize::from(node_count), board),
        Err(GameError::InvalidNativeActionIndex(_))
    ));
    prop_assert!(Action::from_code(-1, node_count).is_err());
    prop_assert_eq!(
        Action::from_code(i32::from(node_count), node_count),
        Ok(Action::Swap)
    );
    prop_assert!(Action::from_code(i32::from(node_count) + 1, node_count).is_err());
    prop_assert!(matches!(
        Action::Swap.native_index(board),
        Err(GameError::SwapHasNoNativeIndex)
    ));

    let rebuilt = GameState::from_parts(
        state.shared_board(),
        StateParts {
            variant,
            stones,
            to_move: state.to_move(),
            moves_left: state.moves_left(),
            opening: state.is_opening(),
            swap_available: state.swap_available(),
            swapped: state.swapped(),
            current_turn: state.current_turn_set(),
            previous_turn: state.previous_turn_set(),
            own_previous_turn: state.own_previous_turn_set(),
            handicap_stones: state.handicap_stones(),
        },
    );
    prop_assert!(
        rebuilt.is_ok(),
        "a state reached through legal play must be importable: {rebuilt:?}"
    );
    let rebuilt = rebuilt.expect("checked above");
    prop_assert_eq!(rebuilt.key(), state.key());
    prop_assert_eq!(rebuilt.hash64(), state.hash64());
    prop_assert_eq!(rebuilt.swapped(), state.swapped());
    prop_assert_eq!(rebuilt.legal_actions(), state.legal_actions());

    // Importing without history is always accepted for a legal position and
    // differs from the exact key only in the history sets.
    let without_history = GameState::from_parts(
        state.shared_board(),
        StateParts {
            variant,
            stones,
            to_move: state.to_move(),
            moves_left: state.moves_left(),
            opening: state.is_opening(),
            swap_available: state.swap_available(),
            swapped: state.swapped(),
            current_turn: BitBoard::empty(),
            previous_turn: BitBoard::empty(),
            own_previous_turn: BitBoard::empty(),
            handicap_stones: BitBoard::empty(),
        },
    );
    prop_assert!(without_history.is_ok(), "{without_history:?}");
    let without_history = without_history.expect("checked above");
    prop_assert_eq!(without_history.stones(), stones);
    prop_assert_eq!(without_history.to_move(), state.to_move());
    prop_assert_eq!(without_history.moves_left(), state.moves_left());
    if state.is_opening() {
        prop_assert_eq!(without_history.key(), state.key());
    }

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
        variant in variants(),
        ranks in prop::collection::vec(any::<u16>(), 0..128),
    ) {
        let board = Arc::new(Board::new(rings).unwrap());
        let mut state = GameState::with_variant(Arc::clone(&board), variant);
        let mut actions = Vec::new();

        assert_state_invariants(&state)?;
        for rank in ranks {
            if state.is_terminal() {
                break;
            }
            let action = ranked_action(&state, rank);
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
        let mut replay = GameState::with_variant(board, variant);
        for action in actions {
            prop_assert!(replay.is_legal(action));
            replay.apply(action).unwrap();
        }
        prop_assert_eq!(snapshot(&replay), final_snapshot);
    }

    #[test]
    fn turn_sizes_follow_the_variant(
        rings in supported_rings(),
        variant in variants(),
        ranks in prop::collection::vec(any::<u16>(), 0..96),
    ) {
        let board = Arc::new(Board::new(rings).unwrap());
        let mut state = GameState::with_variant(board, variant);
        prop_assert_eq!(state.moves_left(), variant.handicap());
        prop_assert!(state.is_opening());
        let mut completed_turns = 0_u32;
        let mut placements_in_turn = 0_u8;
        for rank in ranks {
            if state.is_terminal() {
                break;
            }
            let action = ranked_action(&state, rank);
            let before_turn = state.turn_count();
            let expected_total = state.current_turn_total();
            state.apply(action).unwrap();
            if action == Action::Swap {
                prop_assert_eq!(state.turn_count(), before_turn + 1);
                prop_assert_eq!(state.to_move(), Player::Zero);
                prop_assert_eq!(state.moves_left(), variant.turn_size());
                completed_turns += 1;
                placements_in_turn = 0;
                continue;
            }
            placements_in_turn += 1;
            if state.turn_count() > before_turn {
                prop_assert_eq!(placements_in_turn, expected_total);
                prop_assert_eq!(
                    expected_total,
                    if before_turn == 0 { variant.handicap() } else { variant.turn_size() }
                );
                completed_turns += 1;
                placements_in_turn = 0;
            } else {
                prop_assert!(placements_in_turn < expected_total || state.is_terminal());
            }
        }
        prop_assert_eq!(state.turn_count(), completed_turns);
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
        prop_assert_ne!(ab.previous_turn_moves(), ba.previous_turn_moves());
        prop_assert_eq!(ab.previous_turn_set(), ba.previous_turn_set());
    }

    #[test]
    fn handicap_opening_order_has_one_semantic_key(
        rings in supported_rings(),
        handicap in 2_u8..=MAX_HANDICAP,
        ranks in prop::collection::vec(any::<u16>(), 9),
    ) {
        let board = Arc::new(Board::new(rings).unwrap());
        let variant = Variant::new(Mode::Double, handicap, false).unwrap();
        let mut forward = GameState::with_variant(Arc::clone(&board), variant);
        let mut nodes = Vec::new();
        for rank in ranks.iter().take(usize::from(handicap)) {
            let action = ranked_action(&forward, *rank);
            nodes.push(action.node().unwrap());
            forward.apply(action).unwrap();
        }
        prop_assert!(!forward.is_opening());
        prop_assert_eq!(forward.to_move(), Player::One);
        prop_assert_eq!(forward.turn_count(), 1);
        prop_assert_eq!(forward.handicap_stones().count(), u16::from(handicap));
        prop_assert_eq!(forward.previous_turn_set(), forward.handicap_stones());

        let mut reversed = GameState::with_variant(board, variant);
        for node in nodes.iter().rev() {
            reversed.apply(Action::Place(*node)).unwrap();
        }
        prop_assert_eq!(forward.key(), reversed.key());
        prop_assert_eq!(forward.hash64(), reversed.hash64());
    }

    #[test]
    fn pie_swap_equals_the_color_exchanged_position(
        rings in supported_rings(),
        mode in prop::sample::select(vec![Mode::Classic, Mode::Double]),
        opening_rank in any::<u16>(),
    ) {
        let board = Arc::new(Board::new(rings).unwrap());
        let variant = Variant::new(mode, 1, true).unwrap();
        let mut kept = GameState::with_variant(Arc::clone(&board), variant);
        prop_assert!(kept.is_pie_pending());
        prop_assert!(kept.key().pie_pending);
        let opening = opening_rank % board.node_count();
        kept.apply(Action::Place(opening)).unwrap();
        prop_assert!(!kept.is_pie_pending());
        prop_assert!(kept.swap_available());
        prop_assert!(kept.key().swap_available);

        let mut swapped = kept.clone();
        let transition = swapped.apply(Action::Swap).unwrap();
        prop_assert!(transition.turn_ended);
        prop_assert_eq!(transition.player_before, Player::One);
        prop_assert_eq!(transition.player_after, Player::Zero);
        prop_assert!(swapped.swapped());
        prop_assert!(!swapped.swap_available());
        prop_assert_eq!(swapped.stones(), [BitBoard::empty(), kept.stones()[0]]);
        prop_assert_eq!(swapped.to_move(), Player::Zero);
        prop_assert_eq!(swapped.moves_left(), variant.turn_size());
        prop_assert_eq!(swapped.turn_count(), 2);
        prop_assert_eq!(swapped.previous_turn_set(), kept.previous_turn_set());
        prop_assert_eq!(swapped.handicap_stones(), kept.handicap_stones());
        prop_assert_eq!(swapped.last_move(), kept.last_move());

        // The color-exchanged unswapped position has the same perspective
        // key fields: mover stones, opponent stones, history, and turn.
        let kept_key = kept.key();
        let swapped_key = swapped.key();
        prop_assert_eq!(
            [kept_key.stones[kept.to_move().index()], kept_key.stones[kept.to_move().opponent().index()]],
            [swapped_key.stones[swapped.to_move().index()], swapped_key.stones[swapped.to_move().opponent().index()]]
        );
        prop_assert_eq!(kept_key.moves_left, swapped_key.moves_left);
        prop_assert_eq!(kept_key.previous_turn, swapped_key.previous_turn);
        prop_assert_eq!(kept_key.current_turn, swapped_key.current_turn);
        prop_assert_eq!(kept_key.own_previous_turn, swapped_key.own_previous_turn);
        prop_assert!(!swapped_key.swap_available);

        // Any placement forfeits the swap.
        let mut placed = kept.clone();
        let node = placed.legal_actions().placements.iter().next().unwrap();
        placed.apply(Action::Place(node)).unwrap();
        prop_assert!(!placed.swap_available());
        prop_assert!(matches!(placed.apply(Action::Swap), Err(GameError::SwapUnavailable)));
        prop_assert!(matches!(swapped.apply(Action::Swap), Err(GameError::SwapUnavailable)));
    }

    #[test]
    fn d5_and_color_swaps_preserve_the_scoring_contract(
        rings in supported_rings(),
        variant in variants(),
        ranks in prop::collection::vec(any::<u16>(), 0..80),
        symmetry_index in 0_u8..10,
        action_rank in any::<u16>(),
    ) {
        let state = state_from_ranks(rings, variant, &ranks);
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
        prop_assert_eq!(transformed.variant(), state.variant());
        prop_assert_eq!(transformed.swap_available(), state.swap_available());
        prop_assert_eq!(transformed.swapped(), state.swapped());
        prop_assert_eq!(
            transformed.current_turn_set(),
            maps.bitboard(symmetry, state.current_turn_set())
        );
        prop_assert_eq!(
            transformed.previous_turn_set(),
            maps.bitboard(symmetry, state.previous_turn_set())
        );
        prop_assert_eq!(
            transformed.own_previous_turn_set(),
            maps.bitboard(symmetry, state.own_previous_turn_set())
        );
        prop_assert_eq!(
            transformed.handicap_stones(),
            maps.bitboard(symmetry, state.handicap_stones())
        );
        let key = state.key();
        let transformed_key = transformed.key();
        prop_assert_eq!(transformed_key.mode, key.mode);
        prop_assert_eq!(transformed_key.handicap, key.handicap);
        prop_assert_eq!(transformed_key.pie_pending, key.pie_pending);
        prop_assert_eq!(transformed_key.swap_available, key.swap_available);

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
            let action = ranked_action(&state, action_rank);
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
        metadata_case in 0_u8..9,
    ) {
        let board = Arc::new(Board::new(rings).unwrap());
        let node = node_rank % board.node_count();

        let mut overlap = BitBoard::empty();
        overlap.insert(node);
        prop_assert!(matches!(
            GameState::from_standard_parts(
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
            GameState::from_standard_parts(
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
        let mut one_stone = BitBoard::empty();
        one_stone.insert(node);
        let classic = Variant::new(Mode::Classic, 1, false).unwrap();
        let handicap_three = Variant::new(Mode::Double, 3, false).unwrap();
        let pie = Variant::new(Mode::Double, 1, true).unwrap();
        let (variant, stones, to_move, moves_left, opening, swap_available) = match metadata_case {
            0 => (Variant::STANDARD, empty, Player::Zero, 3, false, false),
            1 => (Variant::STANDARD, empty, Player::Zero, 0, false, false),
            2 => (Variant::STANDARD, empty, Player::One, 1, true, false),
            3 => (Variant::STANDARD, [one_stone, BitBoard::empty()], Player::Zero, 1, true, false),
            4 => (Variant::STANDARD, full, Player::Zero, 2, false, false),
            // Classic turns never hold two placements.
            5 => (classic, [one_stone, BitBoard::empty()], Player::One, 2, false, false),
            // A three-stone opening cannot have four placements left.
            6 => (handicap_three, empty, Player::Zero, 4, true, false),
            // The swap needs the post-opening position of a pie game.
            7 => (pie, empty, Player::Zero, 1, true, true),
            // Swap flags are meaningless without the pie rule.
            8 => (Variant::STANDARD, [one_stone, BitBoard::empty()], Player::One, 2, false, true),
            _ => unreachable!(),
        };
        let parts = StateParts {
            variant,
            stones,
            to_move,
            moves_left,
            opening,
            swap_available,
            swapped: false,
            current_turn: BitBoard::empty(),
            previous_turn: BitBoard::empty(),
            own_previous_turn: BitBoard::empty(),
            handicap_stones: BitBoard::empty(),
        };
        let rejected = GameState::from_parts(board, parts);
        prop_assert!(matches!(rejected, Err(GameError::InvalidTurnMetadata)));
    }

    #[test]
    fn inconsistent_history_is_rejected(
        rings in supported_rings(),
        ranks in prop::collection::vec(any::<u16>(), 3..40),
    ) {
        let state = state_from_ranks(rings, Variant::STANDARD, &ranks);
        prop_assume!(!state.is_terminal() && !state.is_opening());
        let key = state.key();
        // Claim an opponent stone as part of the mover's current turn.
        let opponent = key.stones[state.to_move().opponent().index()];
        prop_assume!(!opponent.is_empty());
        let mut bogus_current = BitBoard::empty();
        bogus_current.insert(opponent.iter().next().unwrap());
        let parts = StateParts {
            variant: Variant::STANDARD,
            stones: key.stones,
            to_move: key.to_move,
            moves_left: key.moves_left,
            opening: false,
            swap_available: false,
            swapped: false,
            current_turn: bogus_current,
            previous_turn: BitBoard::empty(),
            own_previous_turn: BitBoard::empty(),
            handicap_stones: BitBoard::empty(),
        };
        prop_assert!(matches!(
            GameState::from_parts(state.shared_board(), parts),
            Err(GameError::InvalidHistory)
        ));
    }
}

#[test]
fn variants_are_validated() {
    assert!(Variant::new(Mode::Double, 0, false).is_err());
    assert!(Variant::new(Mode::Double, MAX_HANDICAP + 1, false).is_err());
    assert!(matches!(
        Variant::new(Mode::Classic, 2, true),
        Err(GameError::HandicapExcludesPie)
    ));
    let standard = Variant::new(Mode::Double, 1, false).unwrap();
    assert!(standard.is_standard());
    assert_eq!(standard, Variant::STANDARD);
    assert_eq!(Variant::default(), Variant::STANDARD);
    assert!(!Variant::new(Mode::Classic, 1, false).unwrap().is_standard());
    assert!(!Variant::new(Mode::Double, 2, false).unwrap().is_standard());
    assert!(!Variant::new(Mode::Double, 1, true).unwrap().is_standard());
    assert_eq!(Mode::parse("classic"), Some(Mode::Classic));
    assert_eq!(Mode::parse("double"), Some(Mode::Double));
    assert_eq!(Mode::parse("triple"), None);
    assert_eq!(Mode::from_index(Mode::Classic.index()), Some(Mode::Classic));
    assert_eq!(Mode::from_index(Mode::Double.index()), Some(Mode::Double));
    assert_eq!(Mode::from_index(2), None);
    assert_eq!(Mode::Classic.name(), "classic");
    assert_eq!(Mode::Double.name(), "double");
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
        variant in variants(),
    ) {
        for rings in SUPPORTED_RINGS {
            let board = Arc::new(Board::new(rings).unwrap());
            let mut order: Vec<_> = (0..board.node_count()).collect();
            order.sort_by_key(|node| (ordering_keys[usize::from(*node)], *node));
            let mut state = GameState::with_variant(board, variant);
            for node in order {
                prop_assert!(state.is_legal(Action::Place(node)));
                state.apply(Action::Place(node)).unwrap();
            }
            prop_assert!(state.is_terminal());
            prop_assert!(state.legal_actions().is_empty());
            prop_assert!(!state.swap_available());
            let score = ScoringScratch::default().score_state(&state);
            assert_full_score_contract(rings, &score)?;
            let value = terminal_value(&state).expect("full legal play is terminal");
            prop_assert!(value == -1.0 || value == 1.0);
        }
    }
}
