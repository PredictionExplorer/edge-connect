use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use star_engine::{Action, GameState};

use crate::{
    BatchEvaluator, Evaluation, EvaluationRequest, GumbelError, GumbelParameters,
    GumbelSequentialHalving, RootActionStats, SearchError, SearchTree, SimulationStart,
};

static NONCE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Per-search randomness identity.
#[derive(Debug, Eq, PartialEq)]
pub struct SearchNonce {
    value: u64,
    deterministic: bool,
}

impl SearchNonce {
    /// Mints a process-unique, time-mixed nonce for one episode move.
    #[must_use]
    pub fn fresh() -> Self {
        let counter = NONCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos() as u64);
        Self {
            value: splitmix64(time ^ counter.rotate_left(29)),
            deterministic: false,
        }
    }

    /// Explicit reproducible mode for tests, arenas, and debugging.
    #[must_use]
    pub const fn deterministic(seed: u64) -> Self {
        Self {
            value: seed,
            deterministic: true,
        }
    }

    /// Raw nonce mixed into per-root Gumbel seeds.
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.value
    }

    /// Whether this nonce was explicitly requested as reproducible.
    #[must_use]
    pub const fn is_deterministic(&self) -> bool {
        self.deterministic
    }
}

/// Gumbel root-search controls shared by native and foreign actor loops.
#[derive(Debug, PartialEq)]
pub struct RootSearchConfig {
    /// Exact number of edge simulations after root initialization.
    pub simulations: u32,
    /// Maximum Gumbel top-k candidates.
    pub max_considered: usize,
    /// Positive Gumbel Q transformation parameters.
    pub parameters: GumbelParameters,
    /// Fresh or explicitly deterministic identity for this episode move.
    pub nonce: SearchNonce,
}

impl RootSearchConfig {
    /// Production configuration with a freshly minted nonce.
    #[must_use]
    pub fn fresh(simulations: u32, max_considered: usize, parameters: GumbelParameters) -> Self {
        Self {
            simulations,
            max_considered,
            parameters,
            nonce: SearchNonce::fresh(),
        }
    }

    /// Reproducible configuration that must be requested explicitly.
    #[must_use]
    pub const fn deterministic(
        simulations: u32,
        max_considered: usize,
        parameters: GumbelParameters,
        seed: u64,
    ) -> Self {
        Self {
            simulations,
            max_considered,
            parameters,
            nonce: SearchNonce::deterministic(seed),
        }
    }
}

impl Default for RootSearchConfig {
    fn default() -> Self {
        Self::fresh(128, 16, GumbelParameters::PAPER)
    }
}

/// Final data required by an actor and replay writer.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// Gumbel-selected atomic action, or `None` for a terminal input row.
    pub selected_action: Option<Action>,
    /// Cached exact current-player value for a terminal input row.
    pub terminal_value: Option<f32>,
    /// Visit and completed-Q diagnostics in stable legal order.
    pub root_stats: Vec<RootActionStats>,
    /// Improved completed-Q policy target in stable legal order.
    pub policy_target: Vec<(Action, f32)>,
}

/// Failure from a complete batched search run.
#[derive(Debug)]
pub enum SearchRunError<E> {
    /// Evaluator implementation failed.
    Evaluator(E),
    /// Evaluator returned one token more than once.
    DuplicateResponseToken(u64),
    /// Evaluator returned a token not present in the request batch.
    UnknownResponseToken(u64),
    /// Evaluator omitted a requested token.
    MissingResponseToken(u64),
    /// Tree protocol or inference validation failed.
    Search(SearchError),
    /// Gumbel scheduler configuration or state failed.
    Gumbel(GumbelError),
}

impl<E: fmt::Display> fmt::Display for SearchRunError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluator(error) => write!(f, "batch evaluator failed: {error}"),
            Self::DuplicateResponseToken(token) => {
                write!(f, "duplicate evaluation response token {token}")
            }
            Self::UnknownResponseToken(token) => {
                write!(f, "unknown evaluation response token {token}")
            }
            Self::MissingResponseToken(token) => {
                write!(f, "missing evaluation response token {token}")
            }
            Self::Search(error) => write!(f, "search failed: {error}"),
            Self::Gumbel(error) => write!(f, "Gumbel scheduling failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for SearchRunError<E> {}

/// Runs many roots together, crossing the evaluator boundary once per active batch round.
///
/// Terminal rows are retained in output order but never sent to the evaluator.
pub fn gumbel_search_batch<E: BatchEvaluator>(
    roots: Vec<GameState>,
    config: RootSearchConfig,
    evaluator: &mut E,
) -> Result<Vec<SearchResult>, SearchRunError<E::Error>> {
    if config.simulations == 0 {
        return Err(SearchRunError::Gumbel(GumbelError::ZeroBudget));
    }
    if config.max_considered == 0 {
        return Err(SearchRunError::Gumbel(GumbelError::ZeroCandidates));
    }
    config
        .parameters
        .validate()
        .map_err(SearchRunError::Gumbel)?;
    let mut trees: Vec<_> = roots.into_iter().map(SearchTree::new).collect();
    if trees.is_empty() {
        return Ok(Vec::new());
    }

    let active_roots: Vec<_> = trees
        .iter()
        .enumerate()
        .filter(|(_, tree)| tree.root_terminal_value().is_none())
        .map(|(index, tree)| {
            tree.root_request()
                .map(|request| (index, request))
                .map_err(SearchRunError::Search)
        })
        .collect::<Result<_, _>>()?;

    if !active_roots.is_empty() {
        let requests: Vec<_> = active_roots
            .iter()
            .map(|(_, request)| request.clone())
            .collect();
        let responses = evaluator
            .evaluate_batch(&requests)
            .map_err(SearchRunError::Evaluator)?;
        let mut matched = match_responses(&requests, responses)?;

        for (tree_index, request) in &active_roots {
            let response = matched
                .get(&request.token)
                .expect("all request tokens were matched");
            trees[*tree_index]
                .validate_root_evaluation(response)
                .map_err(SearchRunError::Search)?;
        }
        for (tree_index, request) in active_roots {
            let response = matched
                .remove(&request.token)
                .expect("validated response is still available");
            trees[tree_index]
                .initialize_root(response)
                .map_err(SearchRunError::Search)?;
        }
    }

    let mut schedulers: Vec<Option<GumbelSequentialHalving>> = trees
        .iter()
        .enumerate()
        .map(|(index, tree)| {
            if tree.root_terminal_value().is_some() {
                Ok(None)
            } else {
                GumbelSequentialHalving::new(
                    &tree.root_logits(),
                    config.simulations,
                    config.max_considered,
                    config.parameters,
                    derive_root_seed(config.nonce.value(), tree.root_state().hash64(), index),
                )
                .map(Some)
            }
        })
        .collect::<Result<_, _>>()
        .map_err(SearchRunError::Gumbel)?;

    while schedulers
        .iter()
        .flatten()
        .any(|scheduler| !scheduler.is_done())
    {
        let mut requests = Vec::new();
        let mut pending = Vec::new();

        for (tree_index, (tree, scheduler)) in trees.iter_mut().zip(&mut schedulers).enumerate() {
            let Some(scheduler) = scheduler else {
                continue;
            };
            if scheduler.is_done() {
                continue;
            }
            let completed_q = tree.root_completed_q();
            let visits = tree.root_visits();
            let candidate = scheduler
                .next_candidate(&completed_q, &visits)
                .map_err(SearchRunError::Gumbel)?
                .expect("unfinished schedulers return a candidate");
            match tree
                .start_simulation(Some(candidate), config.parameters)
                .map_err(SearchRunError::Search)?
            {
                SimulationStart::Terminal { root_edge } => {
                    debug_assert_eq!(root_edge, candidate);
                    scheduler
                        .record_simulation(root_edge)
                        .map_err(SearchRunError::Gumbel)?;
                }
                SimulationStart::NeedsEvaluation(request) => {
                    pending.push((tree_index, candidate, request.token));
                    requests.push(request);
                }
            }
        }

        if requests.is_empty() {
            continue;
        }
        let responses = evaluator
            .evaluate_batch(&requests)
            .map_err(SearchRunError::Evaluator)?;
        let mut matched = match_responses(&requests, responses)?;

        for (tree_index, _, token) in &pending {
            trees[*tree_index]
                .validate_pending_evaluation(
                    matched.get(token).expect("all pending tokens were matched"),
                )
                .map_err(SearchRunError::Search)?;
        }
        for (tree_index, candidate, token) in pending {
            let response = matched
                .remove(&token)
                .expect("validated response is still available");
            let root_edge = trees[tree_index]
                .finish_simulation(response)
                .map_err(SearchRunError::Search)?;
            debug_assert_eq!(root_edge, candidate);
            schedulers[tree_index]
                .as_mut()
                .expect("pending trees have schedulers")
                .record_simulation(root_edge)
                .map_err(SearchRunError::Gumbel)?;
        }
    }

    trees
        .iter()
        .zip(&schedulers)
        .map(|(tree, scheduler)| {
            let Some(scheduler) = scheduler else {
                return Ok(SearchResult {
                    selected_action: None,
                    terminal_value: tree.root_terminal_value(),
                    root_stats: Vec::new(),
                    policy_target: Vec::new(),
                });
            };
            let completed_q = tree.root_completed_q();
            let visits = tree.root_visits();
            let selected = scheduler
                .selected(&completed_q, &visits)
                .map_err(SearchRunError::Gumbel)?;
            let root_stats = tree.root_stats();
            Ok(SearchResult {
                selected_action: Some(root_stats[selected].action),
                terminal_value: None,
                root_stats,
                policy_target: tree.completed_q_target(config.parameters),
            })
        })
        .collect()
}

fn match_responses<E>(
    requests: &[EvaluationRequest],
    responses: Vec<Evaluation>,
) -> Result<HashMap<u64, Evaluation>, SearchRunError<E>> {
    let expected: HashSet<u64> = requests.iter().map(|request| request.token).collect();
    debug_assert_eq!(
        expected.len(),
        requests.len(),
        "request tokens must be unique"
    );
    let mut matched = HashMap::with_capacity(responses.len());
    for response in responses {
        if !expected.contains(&response.token) {
            return Err(SearchRunError::UnknownResponseToken(response.token));
        }
        let token = response.token;
        if matched.insert(token, response).is_some() {
            return Err(SearchRunError::DuplicateResponseToken(token));
        }
    }
    if let Some(missing) = requests
        .iter()
        .map(|request| request.token)
        .find(|token| !matched.contains_key(token))
    {
        return Err(SearchRunError::MissingResponseToken(missing));
    }
    Ok(matched)
}

fn derive_root_seed(nonce: u64, state_hash: u64, index: usize) -> u64 {
    splitmix64(nonce ^ state_hash.rotate_left(17) ^ (index as u64).rotate_left(41))
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;

    use star_engine::{Board, Player};

    use super::*;

    struct DeterministicEvaluator {
        batch_sizes: Vec<usize>,
    }

    impl BatchEvaluator for DeterministicEvaluator {
        type Error = Infallible;

        fn evaluate_batch(
            &mut self,
            requests: &[EvaluationRequest],
        ) -> Result<Vec<Evaluation>, Self::Error> {
            self.batch_sizes.push(requests.len());
            Ok(requests
                .iter()
                .rev()
                .map(|request| {
                    let logits = request
                        .legal_actions
                        .iter()
                        .map(|action| match action {
                            Action::Place(node) => -f32::from(*node) / 100.0,
                            Action::Pass => -10.0,
                        })
                        .collect();
                    Evaluation {
                        token: request.token,
                        value: if request.state.to_move() == Player::Zero {
                            0.25
                        } else {
                            -0.25
                        },
                        policy_logits: logits,
                    }
                })
                .collect())
        }
    }

    #[test]
    fn token_matching_rejects_duplicate_missing_and_unknown_rows() {
        let board = Arc::new(Board::new(3).unwrap());
        let requests: Vec<_> = (0..2)
            .map(|opening| {
                let mut state = GameState::new(Arc::clone(&board));
                state.apply(Action::Place(opening)).unwrap();
                SearchTree::new(state).root_request().unwrap()
            })
            .collect();
        let response = |token| Evaluation {
            token,
            value: 0.0,
            policy_logits: vec![0.0; requests[0].legal_actions.len()],
        };
        assert!(matches!(
            match_responses::<Infallible>(
                &requests,
                vec![response(requests[0].token), response(requests[0].token)]
            ),
            Err(SearchRunError::DuplicateResponseToken(_))
        ));
        assert!(matches!(
            match_responses::<Infallible>(&requests, vec![response(requests[0].token)]),
            Err(SearchRunError::MissingResponseToken(_))
        ));
        assert!(matches!(
            match_responses::<Infallible>(
                &requests,
                vec![response(requests[0].token), response(u64::MAX)]
            ),
            Err(SearchRunError::UnknownResponseToken(u64::MAX))
        ));
    }

    #[test]
    fn native_runner_batches_active_roots_and_consumes_exact_budgets() {
        let board = Arc::new(Board::new(3).unwrap());
        let mut terminal = GameState::new(Arc::clone(&board));
        terminal.apply(Action::Pass).unwrap();
        terminal.apply(Action::Pass).unwrap();
        let mut roots = vec![terminal];
        for opening in 0..4 {
            let mut state = GameState::new(Arc::clone(&board));
            state.apply(Action::Place(opening)).unwrap();
            roots.push(state);
        }
        let mut evaluator = DeterministicEvaluator {
            batch_sizes: Vec::new(),
        };
        let simulations = 13;
        let config = RootSearchConfig::deterministic(simulations, 4, GumbelParameters::PAPER, 9);
        let results = gumbel_search_batch(roots, config, &mut evaluator).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(evaluator.batch_sizes[0], 4);
        assert!(evaluator.batch_sizes.iter().all(|size| *size <= 4));
        assert!(results[0].selected_action.is_none());
        assert!(results[0].terminal_value.is_some());
        for result in &results[1..] {
            assert_ne!(result.selected_action, Some(Action::Pass));
            assert_eq!(
                result
                    .root_stats
                    .iter()
                    .map(|stats| stats.visits)
                    .sum::<u32>(),
                simulations
            );
            let target_sum: f32 = result
                .policy_target
                .iter()
                .map(|(_, probability)| probability)
                .sum();
            assert!((target_sum - 1.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn fresh_nonces_differ_and_deterministic_mode_is_explicit() {
        let left = SearchNonce::fresh();
        let right = SearchNonce::fresh();
        assert_ne!(left.value(), right.value());
        assert!(!left.is_deterministic());
        assert!(SearchNonce::deterministic(7).is_deterministic());
    }
}
