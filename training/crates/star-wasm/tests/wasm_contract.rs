#![allow(missing_docs)]
#![cfg(target_arch = "wasm32")]

use star_wasm::{WASM_BINDINGS_ENABLED, WasmGumbel, WasmSearchTree, WasmState};
use wasm_bindgen_test::wasm_bindgen_test;

const RULES_HASH: u64 = 0xcdb3_4fb0_2be8_2843;
const EMPTY_RINGS_3_HASH: u64 = 0x0fef_680b_5604_60db;
const MID_TURN_HASH: u64 = 0x2aef_92be_2215_1bac;
const AFTER_PASS_HASH: u64 = 0xba3c_a09a_828b_f306;
const _: () = assert!(WASM_BINDINGS_ENABLED);

fn assert_ascending_nodes_then_pass(actions: &[i32]) {
    let (pass, placements) = actions
        .split_last()
        .expect("a nonterminal state always exposes pass");
    assert_eq!(*pass, -1);
    assert!(placements.iter().all(|action| *action >= 0));
    assert!(placements.windows(2).all(|pair| pair[0] < pair[1]));
}

fn assert_normalized(probabilities: &[f32]) {
    assert!(
        probabilities
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    );
    let sum: f32 = probabilities.iter().sum();
    assert!((sum - 1.0).abs() <= 5.0e-5, "probability sum was {sum}");
}

#[wasm_bindgen_test]
fn wasm_state_matches_conformance_hashes_and_action_layout() {
    assert_eq!(WasmState::rules_hash(), RULES_HASH);
    assert_eq!(WasmState::rules_hash_tag(), "fnv1a64:cdb34fb02be82843");
    assert_eq!(WasmState::rules_schema(), "edgeconnect.star.rules.v1");
    assert!(WasmState::new(2).is_err());

    let mut state = WasmState::new(3).unwrap();
    assert_eq!(state.hash64(), EMPTY_RINGS_3_HASH);
    assert_eq!(state.to_move(), 0);
    assert_eq!(state.moves_left(), 1);
    assert_eq!(state.pass_streak(), 0);
    assert!(!state.terminal());
    assert_eq!(
        state.legal_actions(),
        (0_i32..30).chain(std::iter::once(-1)).collect::<Vec<_>>()
    );
    assert_ascending_nodes_then_pass(&state.legal_actions());
    assert_eq!(state.zero_bits(), vec![0; 7]);
    assert_eq!(state.one_bits(), vec![0; 7]);

    state.apply(7).unwrap();
    state.apply(2).unwrap();
    assert_eq!(state.hash64(), MID_TURN_HASH);
    assert_eq!(state.to_move(), 1);
    assert_eq!(state.moves_left(), 1);
    assert_eq!(state.zero_bits()[0], 1 << 7);
    assert_eq!(state.one_bits()[0], 1 << 2);
    assert_eq!(
        state.legal_actions(),
        (0_i32..30)
            .filter(|node| ![2, 7].contains(node))
            .chain(std::iter::once(-1))
            .collect::<Vec<_>>()
    );
    assert_ascending_nodes_then_pass(&state.legal_actions());

    let rotated = state.transformed(3).unwrap();
    assert_eq!(rotated.score_components(), state.score_components());
    assert_eq!(
        rotated.transformed(2).unwrap().hash64(),
        MID_TURN_HASH,
        "inverse D5 rotations must recover the semantic hash"
    );
    assert!(state.transformed(10).is_err());

    let unchanged = state.hash64();
    assert!(state.apply(7).is_err());
    assert!(state.apply(-2).is_err());
    assert_eq!(state.hash64(), unchanged);

    state.apply(-1).unwrap();
    assert_eq!(state.hash64(), AFTER_PASS_HASH);
    assert_eq!(state.to_move(), 0);
    assert_eq!(state.moves_left(), 2);
    assert_eq!(state.pass_streak(), 1);
    assert_eq!(state.score_components().len(), 14);
}

#[wasm_bindgen_test]
fn wasm_search_preserves_snapshot_tokens_order_and_normalized_policy() {
    let mut state = WasmState::new(3).unwrap();
    state.apply(7).unwrap();
    let expected_actions = state.legal_actions();
    assert_ascending_nodes_then_pass(&expected_actions);

    let mut tree = WasmSearchTree::new(&state, 50.0, 1.0).unwrap();
    assert!(WasmSearchTree::new(&state, 0.0, 1.0).is_err());
    assert_eq!(tree.root_actions().unwrap(), expected_actions);
    let root_token = tree.root_token().unwrap();
    assert_eq!(tree.root_token().unwrap(), root_token);

    state.apply(2).unwrap();
    assert_eq!(
        tree.root_actions().unwrap(),
        expected_actions,
        "the search root must own an immutable state snapshot"
    );

    assert!(
        tree.initialize_root(root_token, 0.25, vec![0.0; expected_actions.len() - 1])
            .is_err()
    );
    assert_eq!(tree.root_token().unwrap(), root_token);
    let root_logits: Vec<_> = expected_actions
        .iter()
        .map(|action| -(*action as f32) / 32.0)
        .collect();
    tree.initialize_root(root_token, 0.25, root_logits).unwrap();
    assert_eq!(tree.actions(), expected_actions);
    assert_eq!(tree.visits(), vec![0; expected_actions.len()]);
    assert!(
        tree.completed_q()
            .iter()
            .all(|value| (*value - 0.25).abs() <= f32::EPSILON)
    );
    assert_normalized(&tree.policy_target());

    let forced_action = expected_actions[0];
    assert!(tree.start(forced_action).unwrap());
    let pending = tree.pending_state().unwrap();
    assert_eq!(pending.to_move(), 1);
    assert_eq!(pending.moves_left(), 1);
    let pending_actions = tree.pending_actions().unwrap();
    assert_ascending_nodes_then_pass(&pending_actions);
    let pending_token = tree.pending_token().unwrap();

    assert!(
        tree.finish(
            pending_token.wrapping_add(1),
            -0.5,
            vec![0.0; pending_actions.len()],
        )
        .is_err()
    );
    assert_eq!(tree.pending_token().unwrap(), pending_token);
    tree.finish(pending_token, -0.5, vec![0.0; pending_actions.len()])
        .unwrap();

    assert_eq!(tree.actions(), expected_actions);
    assert_eq!(tree.visits().iter().sum::<u32>(), 1);
    assert_eq!(tree.visits()[0], 1);
    assert!(tree.completed_q().iter().all(|value| value.is_finite()));
    assert_normalized(&tree.policy_target());
    assert!(tree.pending_state().is_err());

    let visits_before = tree.visits();
    assert!(tree.start(-2).is_err());
    assert_eq!(tree.visits(), visits_before);
}

#[wasm_bindgen_test]
fn wasm_gumbel_uses_the_exact_requested_budget() {
    let logits = vec![0.5, -0.25, 1.0, 0.0, -1.0];
    let completed_q = vec![0.0, 0.25, -0.25, 0.5, -0.5];
    let mut visits = vec![0_u32; logits.len()];
    let mut scheduler = WasmGumbel::new(logits, 17, 4, 50.0, 1.0, 0x5eed).unwrap();

    while !scheduler.done() {
        let candidate = scheduler.next(completed_q.clone(), visits.clone()).unwrap();
        assert!(candidate >= 0);
        let candidate = candidate as usize;
        visits[candidate] += 1;
        scheduler.record(candidate).unwrap();
    }

    assert_eq!(visits.iter().sum::<u32>(), 17);
    assert_eq!(
        scheduler.next(completed_q.clone(), visits.clone()).unwrap(),
        -1
    );
    let selected = scheduler.selected(completed_q, visits.clone()).unwrap();
    assert_eq!(visits[selected], visits.iter().copied().max().unwrap());
}
