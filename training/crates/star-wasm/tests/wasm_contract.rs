#![allow(missing_docs)]
#![cfg(target_arch = "wasm32")]

use star_wasm::{WASM_BINDINGS_ENABLED, WasmGumbel, WasmSearchTree, WasmState};
use wasm_bindgen_test::wasm_bindgen_test;

const RULES_HASH: u64 = 0x2da3_7835_1938_1453;
const _: () = assert!(WASM_BINDINGS_ENABLED);

fn assert_ascending_nodes(actions: &[u16]) {
    assert!(actions.windows(2).all(|pair| pair[0] < pair[1]));
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
fn wasm_state_matches_v2_contract_and_nodes_only_layout() {
    assert_eq!(WasmState::rules_hash(), RULES_HASH);
    assert_eq!(WasmState::rules_hash_tag(), "fnv1a64:2da3783519381453");
    assert_eq!(WasmState::rules_schema(), "edgeconnect.star.rules.v2");
    for unsupported in [0, 1, 2, 3, 5, 7, 9, 11, 12] {
        assert!(WasmState::new(unsupported).is_err());
    }

    let mut state = WasmState::new(4).unwrap();
    assert_eq!(state.to_move(), 0);
    assert_eq!(state.moves_left(), 1);
    assert!(!state.terminal());
    assert_eq!(state.legal_actions(), (0_u16..50).collect::<Vec<_>>());
    assert_ascending_nodes(&state.legal_actions());
    assert_eq!(state.zero_bits(), vec![0; 5]);
    assert_eq!(state.one_bits(), vec![0; 5]);

    state.apply(7).unwrap();
    state.apply(2).unwrap();
    let mid_turn_hash = state.hash64();
    assert_eq!(state.to_move(), 1);
    assert_eq!(state.moves_left(), 1);
    assert_eq!(state.zero_bits()[0], 1 << 7);
    assert_eq!(state.one_bits()[0], 1 << 2);
    assert_eq!(
        state.legal_actions(),
        (0_u16..50)
            .filter(|node| ![2, 7].contains(node))
            .collect::<Vec<_>>()
    );
    assert_ascending_nodes(&state.legal_actions());

    let rotated = state.transformed(3).unwrap();
    assert_eq!(rotated.score_components(), state.score_components());
    assert_eq!(
        rotated.transformed(2).unwrap().hash64(),
        mid_turn_hash,
        "inverse D5 rotations must recover the semantic hash"
    );
    assert!(state.transformed(10).is_err());

    let unchanged = state.hash64();
    assert!(state.apply(7).is_err());
    assert!(state.apply(50).is_err());
    assert_eq!(state.hash64(), unchanged);
    assert_eq!(state.score_components().len(), 14);
}

#[wasm_bindgen_test]
fn wasm_search_preserves_snapshot_tokens_order_and_normalized_policy() {
    let mut state = WasmState::new(4).unwrap();
    state.apply(7).unwrap();
    let expected_actions = state.legal_actions();
    assert_ascending_nodes(&expected_actions);

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
        .map(|action| -f32::from(*action) / 32.0)
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
    assert_ascending_nodes(&pending_actions);
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
    assert!(tree.start(50).is_err());
    assert_eq!(tree.visits(), visits_before);
}

#[wasm_bindgen_test]
fn wasm_gumbel_uses_the_exact_requested_budget() {
    let logits = vec![0.5, -0.25, 1.0, 0.0, -1.0];
    let completed_q = vec![0.0, 0.25, -0.25, 0.5, -0.5];
    let mut visits = vec![0_u32; logits.len()];
    let mut scheduler = WasmGumbel::new(logits, 17, 4, 50.0, 1.0, 0x5eed).unwrap();

    while !scheduler.done() {
        let candidate = scheduler
            .next(completed_q.clone(), visits.clone())
            .unwrap()
            .expect("unfinished scheduler has a candidate");
        visits[candidate] += 1;
        scheduler.record(candidate).unwrap();
    }

    assert_eq!(visits.iter().sum::<u32>(), 17);
    assert_eq!(
        scheduler.next(completed_q.clone(), visits.clone()).unwrap(),
        None
    );
    let selected = scheduler.selected(completed_q, visits.clone()).unwrap();
    assert_eq!(visits[selected], visits.iter().copied().max().unwrap());
}
