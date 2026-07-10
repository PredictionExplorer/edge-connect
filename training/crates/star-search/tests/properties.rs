#![allow(missing_docs)]

use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::Arc;

use proptest::prelude::*;
use star_engine::{Action, Board, GameState, MAX_RINGS, MIN_RINGS};
use star_search::{
    BatchEvaluator, Evaluation, EvaluationRequest, GumbelParameters, GumbelSequentialHalving,
    RootSearchConfig, SearchResult, gumbel_search_batch,
};

#[derive(Clone, Debug)]
struct SchedulerCase {
    logits: Vec<f32>,
    completed_q: Vec<f32>,
    budget: u32,
    max_considered: usize,
    parameters: GumbelParameters,
    seed: u64,
}

fn scheduler_cases() -> impl Strategy<Value = SchedulerCase> {
    (
        1_usize..65,
        1_u32..129,
        1_usize..81,
        any::<u64>(),
        1_u16..1001,
        1_u16..1001,
    )
        .prop_flat_map(
            |(action_count, budget, max_considered, seed, c_visit, c_scale)| {
                (
                    Just((
                        budget,
                        max_considered,
                        seed,
                        GumbelParameters {
                            c_visit: f32::from(c_visit) / 10.0,
                            c_scale: f32::from(c_scale) / 100.0,
                        },
                    )),
                    prop::collection::vec(-2000_i16..=2000, action_count),
                    prop::collection::vec(-1000_i16..=1000, action_count),
                )
            },
        )
        .prop_map(
            |((budget, max_considered, seed, parameters), logits, completed_q)| SchedulerCase {
                logits: logits
                    .into_iter()
                    .map(|value| f32::from(value) / 100.0)
                    .collect(),
                completed_q: completed_q
                    .into_iter()
                    .map(|value| f32::from(value) / 1000.0)
                    .collect(),
                budget,
                max_considered,
                parameters,
                seed,
            },
        )
}

fn placement_root(rings: u8, ranks: &[u16]) -> GameState {
    let board = Arc::new(Board::new(rings).expect("generated ring count is supported"));
    let mut state = GameState::new(board);
    for rank in ranks {
        let placements: Vec<_> = state.legal_actions().placements.iter().collect();
        let node = placements[usize::from(*rank) % placements.len()];
        state
            .apply(Action::Place(node))
            .expect("selected placement is legal");
    }
    state
}

#[derive(Default)]
struct ContractEvaluator;

impl BatchEvaluator for ContractEvaluator {
    type Error = Infallible;

    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, Self::Error> {
        Ok(requests
            .iter()
            .rev()
            .map(|request| {
                let state_hash = request.state.hash64();
                let value_bits = splitmix64(state_hash ^ 0x7661_6c75_6500_0001);
                let value = ((value_bits % 2001) as i32 - 1000) as f32 / 1000.0;
                let policy_logits = request
                    .legal_actions
                    .iter()
                    .map(|action| {
                        let action_key = match action {
                            Action::Place(node) => u64::from(*node),
                            Action::Pass => u64::MAX,
                        };
                        let bits =
                            splitmix64(state_hash ^ action_key.wrapping_mul(0x9e37_79b9_7f4a_7c15));
                        ((bits % 2001) as i32 - 1000) as f32 / 64.0
                    })
                    .collect();
                Evaluation {
                    token: request.token,
                    value,
                    policy_logits,
                }
            })
            .collect())
    }
}

fn run_search(
    root: GameState,
    simulations: u32,
    max_considered: usize,
    parameters: GumbelParameters,
    seed: u64,
) -> SearchResult {
    let config = RootSearchConfig::deterministic(simulations, max_considered, parameters, seed);
    gumbel_search_batch(vec![root], config, &mut ContractEvaluator)
        .expect("generated search configuration is valid")
        .pop()
        .expect("one root produces one result")
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn sequential_halving_is_seeded_and_consumes_every_budget_unit(case in scheduler_cases()) {
        let mut left = GumbelSequentialHalving::new(
            &case.logits,
            case.budget,
            case.max_considered,
            case.parameters,
            case.seed,
        )
        .unwrap();
        let mut right = left.clone();
        let candidates = left.candidates().to_vec();
        let candidate_set: HashSet<_> = candidates.iter().copied().collect();

        prop_assert_eq!(
            candidates.len(),
            case.logits
                .len()
                .min(case.max_considered)
                .min(case.budget as usize)
        );
        prop_assert_eq!(candidate_set.len(), candidates.len());

        let mut left_visits = vec![0_u32; case.logits.len()];
        let mut right_visits = vec![0_u32; case.logits.len()];
        for completed in 0..case.budget {
            let left_candidate = left
                .next_candidate(&case.completed_q, &left_visits)
                .unwrap()
                .unwrap();
            prop_assert_eq!(
                left.next_candidate(&case.completed_q, &left_visits)
                    .unwrap(),
                Some(left_candidate),
                "requesting twice before recording must be idempotent"
            );
            let right_candidate = right
                .next_candidate(&case.completed_q, &right_visits)
                .unwrap()
                .unwrap();

            prop_assert_eq!(left_candidate, right_candidate);
            prop_assert!(candidate_set.contains(&left_candidate));
            left_visits[left_candidate] += 1;
            right_visits[right_candidate] += 1;
            left.record_simulation(left_candidate).unwrap();
            right.record_simulation(right_candidate).unwrap();
            prop_assert_eq!(left.simulations(), completed + 1);
        }

        prop_assert!(left.is_done());
        prop_assert!(right.is_done());
        prop_assert_eq!(left.simulations(), case.budget);
        prop_assert_eq!(left_visits.iter().sum::<u32>(), case.budget);
        prop_assert_eq!(&left_visits, &right_visits);
        prop_assert_eq!(
            left.next_candidate(&case.completed_q, &left_visits)
                .unwrap(),
            None
        );
        for (index, visits) in left_visits.iter().copied().enumerate() {
            if !candidate_set.contains(&index) {
                prop_assert_eq!(visits, 0);
            }
        }

        let selected = left.selected(&case.completed_q, &left_visits).unwrap();
        let right_selected = right.selected(&case.completed_q, &right_visits).unwrap();
        let maximum_candidate_visits = candidates
            .iter()
            .map(|candidate| left_visits[*candidate])
            .max()
            .unwrap();
        prop_assert_eq!(selected, right_selected);
        prop_assert!(candidate_set.contains(&selected));
        prop_assert_eq!(left_visits[selected], maximum_candidate_visits);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn randomized_searches_are_exact_deterministic_and_policy_normalized(
        rings in MIN_RINGS..=MAX_RINGS,
        placement_ranks in prop::collection::vec(any::<u16>(), 1..20),
        simulations in 1_u32..33,
        max_considered in 1_usize..65,
        c_visit in 1_u16..1001,
        c_scale in 1_u16..1001,
        seed in any::<u64>(),
    ) {
        let root = placement_root(rings, &placement_ranks);
        prop_assert!(!root.is_terminal());
        let legal_actions = root.legal_actions().to_vec();
        let parameters = GumbelParameters {
            c_visit: f32::from(c_visit) / 10.0,
            c_scale: f32::from(c_scale) / 100.0,
        };

        let left = run_search(
            root.clone(),
            simulations,
            max_considered,
            parameters,
            seed,
        );
        let right = run_search(root, simulations, max_considered, parameters, seed);
        prop_assert_eq!(&left, &right);
        prop_assert_eq!(left.terminal_value, None);
        prop_assert_eq!(left.root_stats.len(), legal_actions.len());
        prop_assert_eq!(left.policy_target.len(), legal_actions.len());
        prop_assert_eq!(
            &left
                .root_stats
                .iter()
                .map(|stats| stats.action)
                .collect::<Vec<_>>(),
            &legal_actions
        );
        prop_assert_eq!(
            &left
                .policy_target
                .iter()
                .map(|(action, _)| *action)
                .collect::<Vec<_>>(),
            &legal_actions
        );
        prop_assert!(
            left.selected_action
                .is_some_and(|action| legal_actions.contains(&action))
        );
        prop_assert_eq!(
            left.root_stats
                .iter()
                .map(|stats| stats.visits)
                .sum::<u32>(),
            simulations
        );

        let prior_sum: f32 = left.root_stats.iter().map(|stats| stats.prior).sum();
        prop_assert!((prior_sum - 1.0).abs() <= 5.0e-5, "prior sum was {prior_sum}");
        for stats in &left.root_stats {
            prop_assert!(stats.prior.is_finite());
            prop_assert!((0.0..=1.0).contains(&stats.prior));
            prop_assert!(stats.logit.is_finite());
            prop_assert!(stats.q.is_finite());
            prop_assert!((-1.000_001..=1.000_001).contains(&stats.q));
        }

        let target_sum: f32 = left
            .policy_target
            .iter()
            .map(|(_, probability)| *probability)
            .sum();
        prop_assert!(
            (target_sum - 1.0).abs() <= 5.0e-5,
            "policy target sum was {target_sum}"
        );
        for (_, probability) in &left.policy_target {
            prop_assert!(probability.is_finite());
            prop_assert!((0.0..=1.0).contains(probability));
        }
    }
}
