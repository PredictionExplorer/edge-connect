//! Opt-in persistent search with ordered first-visit prediction prefetching.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use star_engine::{GameState, StateKey};

use crate::{
    Evaluation, EvaluationRequest, GumbelError, GumbelParameters, GumbelSequentialHalving,
    ReuseStats, RootActionStats, SearchError, SearchResult, SearchTree, SimulationStart,
};

/// Controls for one additional search budget in a persistent session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SessionConfig {
    /// New simulations, excluding every inherited visit.
    pub simulations: u32,
    /// Maximum sampled root candidates.
    pub max_considered: usize,
    /// Gumbel value transformation parameters.
    pub parameters: GumbelParameters,
    /// Raw scheduler seed; adapters retain their existing seed derivation.
    pub seed: u64,
    /// Maximum first-pass prediction batch width, in `1..=64`.
    pub first_visit_batch_size: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            simulations: 128,
            max_considered: 16,
            parameters: GumbelParameters::PAPER,
            seed: 0,
            first_visit_batch_size: 1,
        }
    }
}

impl SessionConfig {
    /// Reject invalid budgets before any persistent state is changed.
    pub fn validate(self) -> Result<(), SessionError> {
        if self.simulations == 0 {
            return Err(GumbelError::ZeroBudget.into());
        }
        if self.max_considered == 0 {
            return Err(GumbelError::ZeroCandidates.into());
        }
        self.parameters.validate()?;
        if !(1..=64).contains(&self.first_visit_batch_size) {
            return Err(SessionError::InvalidBatchWidth(self.first_visit_batch_size));
        }
        Ok(())
    }
}

/// Protocol or search error in a persistent session.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionError {
    /// Underlying tree rejected an operation or evaluation.
    Search(SearchError),
    /// Invalid scheduler configuration or completion.
    Gumbel(GumbelError),
    /// Configured width lies outside `1..=64`.
    InvalidBatchWidth(usize),
    /// A caller supplied zero available request rows.
    ZeroRequestLimit,
    /// Result/restart was requested before the new budget completed.
    SearchIncomplete,
    /// A response token is repeated in a submitted batch.
    DuplicateResponseToken(u64),
    /// A response token was not requested.
    UnknownResponseToken(u64),
    /// A requested response is absent.
    MissingResponseToken(u64),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Search(error) => error.fmt(f),
            Self::Gumbel(error) => error.fmt(f),
            Self::InvalidBatchWidth(width) => {
                write!(f, "first-visit batch width {width} is outside 1..=64")
            }
            Self::ZeroRequestLimit => f.write_str("request limit must be positive"),
            Self::SearchIncomplete => f.write_str("the current search budget is incomplete"),
            Self::DuplicateResponseToken(token) => write!(f, "duplicate response token {token}"),
            Self::UnknownResponseToken(token) => write!(f, "unknown response token {token}"),
            Self::MissingResponseToken(token) => write!(f, "missing response token {token}"),
        }
    }
}

impl Error for SessionError {}

impl From<SearchError> for SessionError {
    fn from(error: SearchError) -> Self {
        Self::Search(error)
    }
}

impl From<GumbelError> for SessionError {
    fn from(error: GumbelError) -> Self {
        Self::Gumbel(error)
    }
}

/// Completed session result with inherited work reported separately.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionResult {
    /// Normal search result; root statistics contain only newly computed visits.
    /// Q values and the diagnostic root mean include retained evidence.
    pub search: SearchResult,
    /// New visits in stable root-action order; sum equals the requested budget.
    pub visits: Vec<u32>,
    /// Visits inherited at restart, before the new root evaluation.
    pub inherited_visits: Vec<u32>,
    /// Retained plus newly computed visits in stable root-action order.
    pub total_visits: Vec<u32>,
    /// Number of retained nodes at restart, zero for a fresh search.
    pub reused_nodes: usize,
    /// Number of retained root outgoing visits at restart.
    pub reused_visits: u32,
}

#[derive(Clone, Debug)]
struct Prediction {
    value: f32,
    logits: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
enum PendingKind {
    Prefetch,
    Leaf,
}

#[derive(Clone, Debug)]
struct PendingBatch {
    kind: PendingKind,
    requests: Vec<EvaluationRequest>,
}

/// Optional controller that batches predictions without parallel tree mutations.
///
/// First-pass root-child predictions may be requested early. Every simulation
/// still traverses and backs up in the original scheduler order. Reuse requires
/// the caller to verify unchanged model, feature/PDA, and utility context.
#[derive(Clone, Debug)]
pub struct SearchSession {
    tree: SearchTree,
    config: SessionConfig,
    scheduler: Option<GumbelSequentialHalving>,
    fresh_visits: Vec<u32>,
    inherited_visits: Vec<u32>,
    reuse: ReuseStats,
    pending: Option<PendingBatch>,
    predictions: HashMap<StateKey, Prediction>,
}

impl SearchSession {
    /// Creates a fresh, uninitialized session; terminal states need no inference.
    pub fn new(root: GameState, config: SessionConfig) -> Result<Self, SessionError> {
        config.validate()?;
        Ok(Self {
            tree: SearchTree::new(root),
            config,
            scheduler: None,
            fresh_visits: Vec::new(),
            inherited_visits: Vec::new(),
            reuse: ReuseStats {
                retained_nodes: 0,
                retained_visits: 0,
            },
            pending: None,
            predictions: HashMap::new(),
        })
    }

    /// Current exact root state.
    #[must_use]
    pub fn root_state(&self) -> &GameState {
        self.tree.root_state()
    }

    /// Current additional-budget settings.
    #[must_use]
    pub const fn config(&self) -> SessionConfig {
        self.config
    }

    /// Number of new simulations already completed.
    #[must_use]
    pub fn simulations(&self) -> u32 {
        self.scheduler
            .as_ref()
            .map_or(0, GumbelSequentialHalving::simulations)
    }

    /// Number of unique nodes currently retained in the search.
    #[must_use]
    pub fn unique_state_count(&self) -> usize {
        self.tree.unique_state_count()
    }

    /// Whether an inference batch is outstanding.
    #[must_use]
    pub fn has_pending_evaluation(&self) -> bool {
        self.pending.is_some()
    }

    /// Sets a new-simulation budget only before root initialization.
    pub fn set_simulations(&mut self, simulations: u32) -> Result<(), SessionError> {
        if self.scheduler.is_some() {
            return Err(SearchError::RootAlreadyInitialized.into());
        }
        let config = SessionConfig {
            simulations,
            ..self.config
        };
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Requests root predictions even after an exact-state reuse.
    pub fn root_request(&self) -> Result<EvaluationRequest, SessionError> {
        if self.scheduler.is_some() {
            return Err(SearchError::RootAlreadyInitialized.into());
        }
        Ok(if self.tree.is_initialized() {
            self.tree.root_refresh_request()?
        } else {
            self.tree.root_request()?
        })
    }

    /// Validates root predictions without changing retained statistics.
    pub fn validate_root_evaluation(&self, evaluation: &Evaluation) -> Result<(), SessionError> {
        if self.scheduler.is_some() {
            return Err(SearchError::RootAlreadyInitialized.into());
        }
        if self.tree.is_initialized() {
            self.tree.validate_root_refresh(evaluation)?;
        } else {
            self.tree.validate_root_evaluation(evaluation)?;
        }
        Ok(())
    }

    /// Initializes a fresh scheduler and refreshes root predictions atomically.
    pub fn initialize_root(&mut self, evaluation: Evaluation) -> Result<(), SessionError> {
        self.validate_root_evaluation(&evaluation)?;
        let scheduler = GumbelSequentialHalving::new(
            &evaluation.policy_logits,
            self.config.simulations,
            self.config.max_considered,
            self.config.parameters,
            self.config.seed,
        )?;
        if self.tree.is_initialized() {
            self.tree.refresh_root_evaluation(evaluation)?;
        } else {
            self.tree.initialize_root(evaluation)?;
        }
        self.inherited_visits = self.tree.root_visits();
        self.fresh_visits = vec![0; self.inherited_visits.len()];
        self.scheduler = Some(scheduler);
        Ok(())
    }

    /// Whether the current additional budget is complete (or root is terminal).
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.tree.root_terminal_value().is_some()
            || self
                .scheduler
                .as_ref()
                .is_some_and(GumbelSequentialHalving::is_done)
    }

    /// Requests predictions using the configured per-session batch width.
    pub fn next_requests(&mut self) -> Result<Vec<EvaluationRequest>, SessionError> {
        self.next_requests_with_limit(self.config.first_visit_batch_size)
    }

    /// Requests at most `max_requests` rows, respecting an outer batch capacity.
    pub fn next_requests_with_limit(
        &mut self,
        max_requests: usize,
    ) -> Result<Vec<EvaluationRequest>, SessionError> {
        if max_requests == 0 {
            return Err(SessionError::ZeroRequestLimit);
        }
        if self.pending.is_some() {
            return Err(SearchError::PendingEvaluation.into());
        }
        if self.is_done() {
            return Ok(Vec::new());
        }
        if self.scheduler.is_none() {
            return Err(SearchError::RootUninitialized.into());
        }
        let width = self.config.first_visit_batch_size.min(max_requests);
        loop {
            if self.is_done() {
                self.predictions.clear();
                return Ok(Vec::new());
            }
            if width > 1 && self.predictions.is_empty() {
                let requests = self.preview_first_visits(width)?;
                if !requests.is_empty() {
                    self.pending = Some(PendingBatch {
                        kind: PendingKind::Prefetch,
                        requests: requests.clone(),
                    });
                    return Ok(requests);
                }
            }
            let scheduler = self.scheduler.as_mut().expect("root initialized");
            let candidate = if let Some(candidate) = scheduler.next_scheduled_candidate() {
                candidate
            } else {
                scheduler
                    .next_candidate(&self.tree.root_completed_q(), &self.fresh_visits)?
                    .expect("unfinished scheduler returns a candidate")
            };
            match self
                .tree
                .start_simulation(Some(candidate), self.config.parameters)?
            {
                SimulationStart::Terminal { root_edge } => self.record(root_edge)?,
                SimulationStart::NeedsEvaluation(request) => {
                    if let Some(prediction) = self.predictions.remove(&request.state.key()) {
                        let edge = self.tree.finish_simulation(Evaluation {
                            token: request.token,
                            value: prediction.value,
                            policy_logits: prediction.logits,
                        })?;
                        self.record(edge)?;
                    } else {
                        self.pending = Some(PendingBatch {
                            kind: PendingKind::Leaf,
                            requests: vec![request.clone()],
                        });
                        return Ok(vec![request]);
                    }
                }
            }
        }
    }

    fn preview_first_visits(&self, width: usize) -> Result<Vec<EvaluationRequest>, SessionError> {
        let scheduler = self.scheduler.as_ref().expect("root initialized");
        let first_pass = scheduler.candidates().len() as u32;
        if scheduler.simulations() >= first_pass {
            return Ok(Vec::new());
        }
        let capacity = width.min(
            self.config
                .first_visit_batch_size
                .saturating_sub(self.predictions.len()),
        );
        if capacity == 0 {
            return Ok(Vec::new());
        }
        let mut lookahead = scheduler.clone();
        let mut requests = Vec::with_capacity(capacity);
        let mut keys = HashSet::new();
        while lookahead.simulations() < first_pass && requests.len() < capacity {
            let Some(edge) = lookahead.next_scheduled_candidate() else {
                break;
            };
            if self.fresh_visits[edge] == 0
                && let Some(request) = self.tree.preview_root_child(edge)?
            {
                let key = request.state.key();
                if !self.predictions.contains_key(&key) && keys.insert(key) {
                    requests.push(request);
                }
            }
            lookahead.record_simulation(edge)?;
        }
        Ok(requests)
    }

    /// Validates every response before caching or backing up any of the batch.
    pub fn validate_responses(&self, responses: &[Evaluation]) -> Result<(), SessionError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(SearchError::NoPendingEvaluation)?;
        let requests: HashMap<_, _> = pending
            .requests
            .iter()
            .map(|request| (request.token, request))
            .collect();
        let mut seen = HashSet::new();
        for response in responses {
            let request = requests
                .get(&response.token)
                .ok_or(SessionError::UnknownResponseToken(response.token))?;
            if !seen.insert(response.token) {
                return Err(SessionError::DuplicateResponseToken(response.token));
            }
            validate_prediction(request, response)?;
        }
        for request in &pending.requests {
            if !seen.contains(&request.token) {
                return Err(SessionError::MissingResponseToken(request.token));
            }
        }
        if matches!(pending.kind, PendingKind::Leaf) {
            self.tree.validate_pending_evaluation(&responses[0])?;
        }
        Ok(())
    }

    /// Submits a complete, possibly reordered batch. Invalid batches remain retryable.
    pub fn submit(&mut self, responses: Vec<Evaluation>) -> Result<(), SessionError> {
        self.validate_responses(&responses)?;
        let pending = self.pending.take().expect("batch validated");
        match pending.kind {
            PendingKind::Prefetch => {
                let keys: HashMap<_, _> = pending
                    .requests
                    .iter()
                    .map(|request| (request.token, request.state.key()))
                    .collect();
                for response in responses {
                    self.predictions.insert(
                        keys[&response.token],
                        Prediction {
                            value: response.value,
                            logits: response.policy_logits,
                        },
                    );
                }
                debug_assert!(self.predictions.len() <= self.config.first_visit_batch_size);
            }
            PendingKind::Leaf => {
                let edge = self.tree.finish_simulation(
                    responses
                        .into_iter()
                        .next()
                        .expect("one validated response"),
                )?;
                self.record(edge)?;
            }
        }
        Ok(())
    }

    fn record(&mut self, edge: usize) -> Result<(), SessionError> {
        self.scheduler
            .as_mut()
            .expect("root initialized")
            .record_simulation(edge)?;
        self.fresh_visits[edge] += 1;
        Ok(())
    }

    /// Cancels pending inference and speculative predictions without losing visits.
    pub fn cancel_pending(&mut self) {
        self.pending = None;
        self.predictions.clear();
        self.tree.cancel_pending();
    }

    /// Checks exact reuse eligibility after the current search completes.
    #[must_use]
    pub fn can_reuse_root(&self, target: &GameState) -> bool {
        self.is_done() && self.pending.is_none() && self.tree.can_reuse_root(target)
    }

    /// Starts a new additional budget, optionally retaining an exact descendant.
    /// Caller must establish unchanged model, feature/PDA and utility context.
    pub fn restart(
        &mut self,
        target: GameState,
        config: SessionConfig,
        allow_reuse: bool,
        max_nodes: usize,
    ) -> Result<ReuseStats, SessionError> {
        config.validate()?;
        if self.pending.is_some() {
            return Err(SearchError::PendingEvaluation.into());
        }
        if !self.is_done() {
            return Err(SessionError::SearchIncomplete);
        }
        let reuse = if allow_reuse {
            self.tree.reuse_root(target.clone(), max_nodes)?
        } else {
            None
        };
        if reuse.is_none() {
            self.tree = SearchTree::new(target);
        }
        self.reuse = reuse.unwrap_or(ReuseStats {
            retained_nodes: 0,
            retained_visits: 0,
        });
        self.config = config;
        self.scheduler = None;
        self.fresh_visits.clear();
        self.inherited_visits.clear();
        self.pending = None;
        self.predictions.clear();
        Ok(self.reuse)
    }

    /// Returns completed results; inherited visits never inflate the new budget.
    pub fn result(&self) -> Result<SessionResult, SessionError> {
        if !self.is_done() {
            return Err(SessionError::SearchIncomplete);
        }
        if let Some(value) = self.tree.root_terminal_value() {
            return Ok(SessionResult {
                search: SearchResult {
                    selected_action: None,
                    terminal_value: Some(value),
                    root_value: None,
                    selected_action_value: None,
                    root_stats: Vec::new(),
                    policy_target: Vec::new(),
                },
                visits: Vec::new(),
                inherited_visits: Vec::new(),
                total_visits: Vec::new(),
                reused_nodes: self.reuse.retained_nodes,
                reused_visits: self.reuse.retained_visits,
            });
        }
        let scheduler = self
            .scheduler
            .as_ref()
            .expect("completed nonterminal search initialized");
        let q = self.tree.root_completed_q();
        let selected = scheduler.selected(&q, &self.fresh_visits)?;
        let mut stats = self.tree.root_stats();
        let total_visits = stats.iter().map(|stat| stat.visits).collect();
        for (stat, &visits) in stats.iter_mut().zip(&self.fresh_visits) {
            stat.visits = visits;
        }
        let policy_target = fresh_policy_target(&stats, self.config.parameters);
        Ok(SessionResult {
            search: SearchResult {
                selected_action: Some(stats[selected].action),
                terminal_value: None,
                root_value: self.tree.root_value(),
                selected_action_value: Some(stats[selected].q),
                root_stats: stats,
                policy_target,
            },
            visits: self.fresh_visits.clone(),
            inherited_visits: self.inherited_visits.clone(),
            total_visits,
            reused_nodes: self.reuse.retained_nodes,
            reused_visits: self.reuse.retained_visits,
        })
    }
}

fn validate_prediction(
    request: &EvaluationRequest,
    evaluation: &Evaluation,
) -> Result<(), SearchError> {
    if evaluation.policy_logits.len() != request.legal_actions.len() {
        return Err(SearchError::PolicyLength {
            expected: request.legal_actions.len(),
            actual: evaluation.policy_logits.len(),
        });
    }
    if !evaluation.value.is_finite()
        || evaluation
            .policy_logits
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(SearchError::NonFiniteEvaluation);
    }
    if !(-1.0..=1.0).contains(&evaluation.value) {
        return Err(SearchError::ValueOutOfRange(evaluation.value));
    }
    Ok(())
}

fn fresh_policy_target(
    stats: &[RootActionStats],
    parameters: GumbelParameters,
) -> Vec<(star_engine::Action, f32)> {
    let scale = parameters.sigma_scale(stats.iter().map(|stat| stat.visits).max().unwrap_or(0));
    let logits: Vec<f32> = stats
        .iter()
        .map(|stat| stat.logit + scale * stat.q)
        .collect();
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f64> = logits
        .iter()
        .map(|logit| f64::from(*logit - maximum).exp())
        .collect();
    let sum: f64 = exp.iter().sum();
    stats
        .iter()
        .zip(exp)
        .map(|(stat, weight)| (stat.action, (weight / sum) as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;

    use star_engine::{Action, Board, Mode, Variant};

    use super::*;
    use crate::{BatchEvaluator, RootSearchConfig, gumbel_search_batch};

    fn prediction(request: &EvaluationRequest) -> Evaluation {
        let hash = request.state.hash64();
        Evaluation {
            token: request.token,
            value: (hash % 2001) as f32 / 1000.0 - 1.0,
            policy_logits: request
                .legal_actions
                .iter()
                .map(|action| {
                    let node = u64::from(action.node().unwrap());
                    ((hash.rotate_left(7) ^ node.wrapping_mul(7919)) % 201) as f32 / 100.0 - 1.0
                })
                .collect(),
        }
    }

    #[derive(Default)]
    struct Evaluator {
        states: Vec<StateKey>,
        widths: Vec<usize>,
    }

    impl BatchEvaluator for Evaluator {
        type Error = Infallible;
        fn evaluate_batch(
            &mut self,
            requests: &[EvaluationRequest],
        ) -> Result<Vec<Evaluation>, Self::Error> {
            self.states
                .extend(requests.iter().map(|request| request.state.key()));
            self.widths.push(requests.len());
            Ok(requests.iter().map(prediction).rev().collect())
        }
    }

    fn root(rings: u8, variant: Variant) -> GameState {
        GameState::with_variant(Arc::new(Board::new(rings).unwrap()), variant)
    }

    fn drive(session: &mut SearchSession, limit: usize) -> Evaluator {
        let mut evaluator = Evaluator::default();
        if !session.is_done() && session.scheduler.is_none() {
            let request = session.root_request().unwrap();
            let evaluation = evaluator.evaluate_batch(&[request]).unwrap().remove(0);
            session.initialize_root(evaluation).unwrap();
        }
        while !session.is_done() {
            let requests = session.next_requests_with_limit(limit).unwrap();
            assert!(requests.len() <= limit);
            if requests.is_empty() {
                assert!(session.is_done());
                break;
            }
            let responses = evaluator.evaluate_batch(&requests).unwrap();
            session.submit(responses).unwrap();
        }
        evaluator
    }

    fn raw_seed(nonce: u64, state: &GameState) -> u64 {
        let mut value =
            (nonce ^ state.hash64().rotate_left(17)).wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    #[test]
    fn fresh_sessions_match_legacy_values_visits_targets_and_requests_across_widths() {
        for rings in [4, 10] {
            for variant in [
                Variant::new(Mode::Classic, 1, false).unwrap(),
                Variant::new(Mode::Double, 1, false).unwrap(),
                Variant::new(Mode::Double, 3, false).unwrap(),
                Variant::new(Mode::Double, 1, true).unwrap(),
            ] {
                for simulations in [1, 5, 17, 64] {
                    let state = root(rings, variant);
                    let mut reference = Evaluator::default();
                    let expected = gumbel_search_batch(
                        vec![state.clone()],
                        RootSearchConfig::deterministic(
                            simulations,
                            16,
                            GumbelParameters::PAPER,
                            99,
                        ),
                        &mut reference,
                    )
                    .unwrap()
                    .remove(0);
                    for width in [1, 2, 8, 64] {
                        let mut session = SearchSession::new(
                            state.clone(),
                            SessionConfig {
                                simulations,
                                first_visit_batch_size: width,
                                seed: raw_seed(99, &state),
                                ..SessionConfig::default()
                            },
                        )
                        .unwrap();
                        let actual_evaluator = drive(&mut session, width);
                        let result = session.result().unwrap();
                        assert_eq!(result.search, expected);
                        assert_eq!(actual_evaluator.states, reference.states);
                        assert_eq!(result.visits.iter().sum::<u32>(), simulations);
                        assert_eq!(result.visits, result.total_visits);
                        assert!(result.inherited_visits.iter().all(|visits| *visits == 0));
                        assert_eq!(result.reused_nodes, 0);
                        assert!(actual_evaluator.widths.iter().all(|rows| *rows <= width));
                        if simulations >= 5 && width > 1 {
                            assert!(actual_evaluator.widths.iter().any(|rows| *rows > 1));
                            assert!(actual_evaluator.widths.len() < reference.widths.len());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn outer_capacity_limits_batches_without_changing_search() {
        let state = root(4, Variant::new(Mode::Double, 1, false).unwrap());
        let config = SessionConfig {
            simulations: 41,
            first_visit_batch_size: 16,
            ..SessionConfig::default()
        };
        let mut serial = SearchSession::new(state.clone(), config).unwrap();
        drive(&mut serial, 1);
        let expected = serial.result().unwrap();
        for limit in [2, 3, 5, 16, 100] {
            let mut session = SearchSession::new(state.clone(), config).unwrap();
            let evaluator = drive(&mut session, limit);
            assert_eq!(session.result().unwrap(), expected);
            assert!(evaluator.widths.iter().all(|rows| *rows <= limit.min(16)));
        }
    }

    #[test]
    fn prefetch_validation_is_atomic_and_cancellation_discards_stale_predictions() {
        let state = root(4, Variant::new(Mode::Double, 1, false).unwrap());
        let mut session = SearchSession::new(
            state.clone(),
            SessionConfig {
                simulations: 32,
                first_visit_batch_size: 4,
                ..SessionConfig::default()
            },
        )
        .unwrap();
        session
            .initialize_root(prediction(&session.root_request().unwrap()))
            .unwrap();
        let requests = session.next_requests().unwrap();
        assert_eq!(requests.len(), 4);
        assert_eq!(session.simulations(), 0);
        assert_eq!(session.unique_state_count(), 1);
        assert!(!session.tree.has_pending_evaluation());
        assert!(session.next_requests().is_err());
        assert!(
            session
                .restart(state.clone(), session.config(), true, 1000)
                .is_err()
        );
        let correct: Vec<_> = requests.iter().map(prediction).collect();
        let mut invalid_batches = vec![correct[..3].to_vec()];
        let mut duplicate = correct.clone();
        duplicate[1] = duplicate[0].clone();
        invalid_batches.push(duplicate);
        let mut unknown = correct.clone();
        unknown[1].token = 0;
        invalid_batches.push(unknown);
        let mut nonfinite = correct.clone();
        nonfinite[3].value = f32::NAN;
        invalid_batches.push(nonfinite);
        let mut wrong_length = correct.clone();
        wrong_length[3].policy_logits.pop();
        invalid_batches.push(wrong_length);
        let mut outside_range = correct.clone();
        outside_range[3].value = 2.0;
        invalid_batches.push(outside_range);
        for responses in invalid_batches {
            assert!(session.submit(responses).is_err());
            assert!(session.has_pending_evaluation());
            assert!(session.predictions.is_empty());
            assert_eq!(session.simulations(), 0);
            assert_eq!(session.unique_state_count(), 1);
        }
        session.submit(correct.clone()).unwrap();
        assert_eq!(session.simulations(), 0);
        assert_eq!(session.predictions.len(), 4);
        session.cancel_pending();
        assert!(session.predictions.is_empty());
        assert!(session.submit(correct.clone()).is_err());
        let retried = session.next_requests().unwrap();
        assert!(
            retried
                .iter()
                .zip(&requests)
                .all(|(new, old)| new.token != old.token)
        );
        assert!(session.submit(correct).is_err());
        session
            .submit(retried.iter().map(prediction).rev().collect())
            .unwrap();
        drive(&mut session, 4);
        let mut reference = SearchSession::new(state, session.config()).unwrap();
        drive(&mut reference, 1);
        assert_eq!(session.result().unwrap(), reference.result().unwrap());
    }

    #[test]
    fn cancelling_an_actual_leaf_retries_without_consuming_the_budget() {
        let state = root(4, Variant::new(Mode::Classic, 1, false).unwrap());
        let mut session = SearchSession::new(state.clone(), SessionConfig::default()).unwrap();
        session
            .initialize_root(prediction(&session.root_request().unwrap()))
            .unwrap();
        let request = session.next_requests().unwrap().remove(0);
        assert!(session.tree.has_pending_evaluation());
        session.cancel_pending();
        assert_eq!(session.simulations(), 0);
        assert!(!session.tree.has_pending_evaluation());
        let retry = session.next_requests().unwrap().remove(0);
        assert_eq!(retry.state.key(), request.state.key());
        assert_ne!(retry.token, request.token);
        assert!(session.submit(vec![prediction(&request)]).is_err());
        session.submit(vec![prediction(&retry)]).unwrap();
        drive(&mut session, 1);
        let mut reference = SearchSession::new(state, SessionConfig::default()).unwrap();
        drive(&mut reference, 1);
        assert_eq!(session.result().unwrap(), reference.result().unwrap());
    }

    #[test]
    fn first_visit_prefetch_skips_terminal_children_without_losing_simulations() {
        let mut state = root(4, Variant::new(Mode::Double, 1, false).unwrap());
        while state.legal_actions().placements.iter().count() > 1 {
            let node = state.legal_actions().placements.iter().next().unwrap();
            state.apply(Action::Place(node)).unwrap();
        }
        let mut session = SearchSession::new(
            state,
            SessionConfig {
                simulations: 13,
                first_visit_batch_size: 64,
                ..SessionConfig::default()
            },
        )
        .unwrap();
        let evaluator = drive(&mut session, 64);
        assert_eq!(evaluator.widths, vec![1]);
        assert_eq!(session.result().unwrap().visits, vec![13]);
        assert!(session.predictions.is_empty());
        assert!(!session.has_pending_evaluation());
    }

    #[test]
    fn reused_descendant_has_fresh_budget_sigma_and_mandatory_root_refresh() {
        let state = root(4, Variant::new(Mode::Double, 1, false).unwrap());
        let mut session = SearchSession::new(
            state.clone(),
            SessionConfig {
                simulations: 256,
                first_visit_batch_size: 8,
                ..SessionConfig::default()
            },
        )
        .unwrap();
        drive(&mut session, 8);
        let mut target = state;
        target
            .apply(session.result().unwrap().search.selected_action.unwrap())
            .unwrap();
        assert!(session.can_reuse_root(&target));
        let config = SessionConfig {
            simulations: 3,
            max_considered: 2,
            seed: 1,
            ..SessionConfig::default()
        };
        let reuse = session
            .restart(target.clone(), config, true, 10_000)
            .unwrap();
        assert!(reuse.retained_nodes > 1);
        assert!(reuse.retained_visits > 3);
        assert_eq!(session.simulations(), 0);
        assert!(!session.is_done());
        assert!(session.next_requests().is_err());
        let refresh = session.root_request().unwrap();
        assert_eq!(refresh.state.key(), target.key());
        let retained = session.tree.root_visits();
        session.initialize_root(prediction(&refresh)).unwrap();
        assert_eq!(session.tree.root_visits(), retained);
        assert!(session.initialize_root(prediction(&refresh)).is_err());
        drive(&mut session, 1);
        let result = session.result().unwrap();
        assert_eq!(result.visits.iter().sum::<u32>(), 3);
        assert_eq!(
            result.inherited_visits.iter().sum::<u32>(),
            reuse.retained_visits
        );
        assert_eq!(
            result.total_visits.iter().sum::<u32>(),
            reuse.retained_visits + 3
        );
        for ((&new, &old), &total) in result
            .visits
            .iter()
            .zip(&result.inherited_visits)
            .zip(&result.total_visits)
        {
            assert_eq!(total, new + old);
        }
        let scheduler = session.scheduler.as_ref().unwrap();
        let selected = scheduler
            .selected(&session.tree.root_completed_q(), &result.visits)
            .unwrap();
        assert_eq!(
            result.search.selected_action,
            Some(result.search.root_stats[selected].action)
        );
        assert_eq!(
            result.visits[selected],
            result.visits.iter().copied().max().unwrap()
        );
        let scale = config
            .parameters
            .sigma_scale(*result.visits.iter().max().unwrap());
        let positive: Vec<_> = result
            .search
            .policy_target
            .iter()
            .enumerate()
            .filter(|(_, (_, probability))| *probability > 1e-20)
            .map(|(index, (_, probability))| (index, f64::from(*probability)))
            .collect();
        assert!(positive.len() >= 2);
        let ((left, left_p), (right, right_p)) = (positive[0], positive[1]);
        let left_stat = result.search.root_stats[left];
        let right_stat = result.search.root_stats[right];
        let expected_log_ratio = f64::from(left_stat.logit + scale * left_stat.q)
            - f64::from(right_stat.logit + scale * right_stat.q);
        assert!(((left_p / right_p).ln() - expected_log_ratio).abs() < 1e-4);
        assert_ne!(
            result.search.policy_target,
            session.tree.completed_q_target(config.parameters)
        );
        assert_eq!(result.reused_nodes, reuse.retained_nodes);
        assert_eq!(result.reused_visits, reuse.retained_visits);
    }

    #[test]
    fn prefetch_matches_serial_search_with_inherited_expanded_children() {
        let state = root(4, Variant::new(Mode::Double, 1, false).unwrap());
        let mut original = SearchSession::new(
            state.clone(),
            SessionConfig {
                simulations: 128,
                ..SessionConfig::default()
            },
        )
        .unwrap();
        drive(&mut original, 1);
        let mut target = state;
        target
            .apply(original.result().unwrap().search.selected_action.unwrap())
            .unwrap();
        let mut serial = original.clone();
        serial
            .restart(
                target.clone(),
                SessionConfig {
                    simulations: 37,
                    ..SessionConfig::default()
                },
                true,
                1000,
            )
            .unwrap();
        drive(&mut serial, 1);
        let expected = serial.result().unwrap();
        for width in [2, 8, 64] {
            let mut session = original.clone();
            session
                .restart(
                    target.clone(),
                    SessionConfig {
                        simulations: 37,
                        first_visit_batch_size: width,
                        ..SessionConfig::default()
                    },
                    true,
                    1000,
                )
                .unwrap();
            drive(&mut session, width);
            assert_eq!(session.result().unwrap(), expected);
        }
        for cap in [0, 1] {
            let mut limited = original.clone();
            let reuse = limited
                .restart(target.clone(), SessionConfig::default(), true, cap)
                .unwrap();
            assert_eq!(reuse.retained_nodes, 0);
            assert_eq!(limited.root_state().key(), target.key());
            assert_eq!(limited.unique_state_count(), 1);
        }
    }

    #[test]
    fn invalid_restart_and_budget_changes_are_nonmutating_and_fresh_fallback_is_explicit() {
        let state = root(4, Variant::new(Mode::Classic, 1, false).unwrap());
        assert!(
            SearchSession::new(
                state.clone(),
                SessionConfig {
                    first_visit_batch_size: 0,
                    ..SessionConfig::default()
                }
            )
            .is_err()
        );
        assert!(
            SearchSession::new(
                state.clone(),
                SessionConfig {
                    first_visit_batch_size: 65,
                    ..SessionConfig::default()
                }
            )
            .is_err()
        );
        let mut session = SearchSession::new(state.clone(), SessionConfig::default()).unwrap();
        assert!(session.set_simulations(0).is_err());
        session.set_simulations(32).unwrap();
        assert!(session.next_requests_with_limit(0).is_err());
        assert!(session.result().is_err());
        assert!(
            session
                .restart(state.clone(), SessionConfig::default(), true, 1000)
                .is_err()
        );
        drive(&mut session, 1);
        assert!(session.set_simulations(1).is_err());
        let completed = session.result().unwrap();
        assert!(
            session
                .restart(
                    state.clone(),
                    SessionConfig {
                        simulations: 0,
                        ..SessionConfig::default()
                    },
                    true,
                    1000
                )
                .is_err()
        );
        assert_eq!(session.result().unwrap(), completed);
        let mut target = state;
        target
            .apply(completed.search.selected_action.unwrap())
            .unwrap();
        assert!(session.can_reuse_root(&target));
        let reuse = session
            .restart(target.clone(), SessionConfig::default(), false, 1000)
            .unwrap();
        assert_eq!(reuse.retained_nodes, 0);
        assert_eq!(session.unique_state_count(), 1);
        drive(&mut session, 1);
        assert!(
            session
                .result()
                .unwrap()
                .inherited_visits
                .iter()
                .all(|visits| *visits == 0)
        );
        let mut terminal = target;
        while !terminal.is_terminal() {
            let node = terminal.legal_actions().placements.iter().next().unwrap();
            terminal.apply(Action::Place(node)).unwrap();
        }
        session
            .restart(terminal, SessionConfig::default(), true, 1000)
            .unwrap();
        assert!(session.is_done());
        assert!(session.root_request().is_err());
        assert!(session.next_requests().unwrap().is_empty());
        let terminal_result = session.result().unwrap();
        assert!(terminal_result.search.terminal_value.is_some());
        assert!(terminal_result.visits.is_empty());
    }
}
