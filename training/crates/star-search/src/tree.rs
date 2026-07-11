use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use star_engine::{Action, GameError, GameState, Player, StateKey, terminal_value};

use crate::{Evaluation, EvaluationRequest, GumbelParameters};

static NEXT_EVALUATION_TOKEN: AtomicU64 = AtomicU64::new(1);

/// Search construction, inference, or protocol error.
#[derive(Clone, Debug, PartialEq)]
pub enum SearchError {
    /// A terminal state cannot be used as a search root.
    TerminalRoot,
    /// Root inference must be submitted before simulations.
    RootUninitialized,
    /// Root inference was submitted more than once.
    RootAlreadyInitialized,
    /// This tree already has one outstanding leaf.
    PendingEvaluation,
    /// No leaf is waiting for this result.
    NoPendingEvaluation,
    /// An asynchronous response used the wrong token.
    TokenMismatch {
        /// Opaque token issued with the pending request.
        expected: u64,
        /// Token submitted by the caller.
        actual: u64,
    },
    /// Policy output length does not match the legal action count.
    PolicyLength {
        /// Legal action count.
        expected: usize,
        /// Submitted logit count.
        actual: usize,
    },
    /// Value or policy output contains an invalid number.
    NonFiniteEvaluation,
    /// Value lies outside the zero-sum `[-1, 1]` contract.
    ValueOutOfRange(f32),
    /// A root edge index is invalid.
    InvalidRootEdge(usize),
    /// Gumbel constants are not finite and strictly positive.
    InvalidGumbelParameters,
    /// Applying a generated legal action failed.
    Engine(GameError),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TerminalRoot => f.write_str("cannot search a terminal root"),
            Self::RootUninitialized => f.write_str("root evaluation has not been initialized"),
            Self::RootAlreadyInitialized => f.write_str("root is already initialized"),
            Self::PendingEvaluation => f.write_str("a leaf evaluation is already pending"),
            Self::NoPendingEvaluation => f.write_str("no leaf evaluation is pending"),
            Self::TokenMismatch { expected, actual } => {
                write!(f, "leaf token mismatch: expected {expected}, got {actual}")
            }
            Self::PolicyLength { expected, actual } => {
                write!(f, "policy has {actual} logits but {expected} were expected")
            }
            Self::NonFiniteEvaluation => f.write_str("evaluation contains a non-finite number"),
            Self::ValueOutOfRange(value) => {
                write!(f, "value {value} is outside the [-1, 1] contract")
            }
            Self::InvalidRootEdge(edge) => write!(f, "invalid root edge index {edge}"),
            Self::InvalidGumbelParameters => {
                f.write_str("c_visit and c_scale must be finite and strictly positive")
            }
            Self::Engine(error) => write!(f, "engine transition failed: {error}"),
        }
    }
}

impl Error for SearchError {}

impl From<GameError> for SearchError {
    fn from(value: GameError) -> Self {
        Self::Engine(value)
    }
}

/// Outcome of starting one simulation.
#[derive(Clone, Debug)]
pub enum SimulationStart {
    /// The path reached a terminal state and was backed up immediately.
    Terminal {
        /// Root edge used by this simulation.
        root_edge: usize,
    },
    /// The path stopped at an unexpanded leaf.
    NeedsEvaluation(EvaluationRequest),
}

/// Public root statistics in stable legal-action order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootActionStats {
    /// Atomic action.
    pub action: Action,
    /// Network prior after softmax.
    pub prior: f32,
    /// Original network logit.
    pub logit: f32,
    /// Edge visit count.
    pub visits: u32,
    /// Mean return in root-player perspective, or completed Q when unvisited.
    pub q: f32,
}

#[derive(Clone, Debug)]
struct Edge {
    action: Action,
    prior: f32,
    logit: f32,
    visits: u32,
    value_sum: f64,
    child: Option<usize>,
}

#[derive(Clone, Debug)]
struct Node {
    state: GameState,
    expanded: bool,
    evaluation_value: f32,
    terminal_value: Option<f32>,
    visits: u32,
    value_sum: f64,
    edges: Vec<Edge>,
}

impl Node {
    fn new(state: GameState) -> Self {
        let cached_terminal_value = terminal_value(&state);
        Self {
            state,
            expanded: false,
            evaluation_value: 0.0,
            terminal_value: cached_terminal_value,
            visits: 0,
            value_sum: 0.0,
            edges: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct PendingSimulation {
    token: u64,
    node_path: Vec<usize>,
    edge_path: Vec<(usize, usize)>,
    root_edge: usize,
}

/// Arena-backed, exact-transposition MCTS DAG for one root.
#[derive(Clone, Debug)]
pub struct SearchTree {
    nodes: Vec<Node>,
    transpositions: HashMap<StateKey, usize>,
    pending: Option<PendingSimulation>,
    root_token: u64,
}

impl SearchTree {
    /// Creates an uninitialized tree.
    #[must_use]
    pub fn new(root: GameState) -> Self {
        let key = root.key();
        let mut transpositions = HashMap::new();
        transpositions.insert(key, 0);
        Self {
            nodes: vec![Node::new(root)],
            transpositions,
            pending: None,
            root_token: fresh_evaluation_token(),
        }
    }

    /// Root state.
    #[must_use]
    pub fn root_state(&self) -> &GameState {
        &self.nodes[0].state
    }

    /// Exact cached terminal value for a terminal root.
    #[must_use]
    pub fn root_terminal_value(&self) -> Option<f32> {
        self.nodes[0].terminal_value
    }

    /// Whether root inference has been supplied.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.nodes[0].expanded
    }

    /// Number of unique semantic states in the arena.
    #[must_use]
    pub fn unique_state_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of completed simulations.
    #[must_use]
    pub fn simulations(&self) -> u32 {
        self.nodes[0].visits
    }

    /// Whether an asynchronous leaf is outstanding.
    #[must_use]
    pub fn has_pending_evaluation(&self) -> bool {
        self.pending.is_some()
    }

    /// Initial inference request for the root.
    pub fn root_request(&self) -> Result<EvaluationRequest, SearchError> {
        if self.nodes[0].state.is_terminal() {
            return Err(SearchError::TerminalRoot);
        }
        if self.nodes[0].expanded {
            return Err(SearchError::RootAlreadyInitialized);
        }
        Ok(EvaluationRequest {
            token: self.root_token,
            state: self.nodes[0].state.clone(),
            legal_actions: self.nodes[0].state.legal_actions().to_vec(),
        })
    }

    /// Validates root inference without mutating the tree.
    pub fn validate_root_evaluation(&self, evaluation: &Evaluation) -> Result<(), SearchError> {
        if self.nodes[0].state.is_terminal() {
            return Err(SearchError::TerminalRoot);
        }
        if self.nodes[0].expanded {
            return Err(SearchError::RootAlreadyInitialized);
        }
        self.validate_token(self.root_token, evaluation.token)?;
        self.validate_evaluation(0, evaluation)
    }

    /// Supplies root inference without counting it as a simulation.
    pub fn initialize_root(&mut self, evaluation: Evaluation) -> Result<(), SearchError> {
        self.validate_root_evaluation(&evaluation)?;
        self.expand_node_unchecked(0, evaluation);
        Ok(())
    }

    /// Starts one full Gumbel AlphaZero simulation.
    ///
    /// At non-root nodes this uses the deterministic improved-policy rule
    /// `argmax(pi_improved(a) - N(a)/(1 + sum N))`.
    pub fn start_simulation(
        &mut self,
        forced_root_edge: Option<usize>,
        parameters: GumbelParameters,
    ) -> Result<SimulationStart, SearchError> {
        if !self.nodes[0].expanded {
            return Err(SearchError::RootUninitialized);
        }
        if self.pending.is_some() {
            return Err(SearchError::PendingEvaluation);
        }
        if parameters.validate().is_err() {
            return Err(SearchError::InvalidGumbelParameters);
        }
        if let Some(edge) = forced_root_edge
            && edge >= self.nodes[0].edges.len()
        {
            return Err(SearchError::InvalidRootEdge(edge));
        }

        let mut node_path = vec![0_usize];
        let mut edge_path = Vec::new();
        let mut node_id = 0_usize;
        let mut root_edge = None;

        loop {
            if let Some(value) = self.nodes[node_id].terminal_value {
                let leaf_player = self.nodes[node_id].state.to_move();
                self.backup(&node_path, &edge_path, leaf_player, value);
                return Ok(SimulationStart::Terminal {
                    root_edge: root_edge.expect("a nonterminal root has a first edge"),
                });
            }

            if !self.nodes[node_id].expanded {
                let token = fresh_evaluation_token();
                let request = EvaluationRequest {
                    token,
                    state: self.nodes[node_id].state.clone(),
                    legal_actions: self.nodes[node_id].state.legal_actions().to_vec(),
                };
                self.pending = Some(PendingSimulation {
                    token,
                    node_path,
                    edge_path,
                    root_edge: root_edge.expect("the root is already expanded"),
                });
                return Ok(SimulationStart::NeedsEvaluation(request));
            }

            let edge_id = if node_id == 0 {
                forced_root_edge.unwrap_or_else(|| self.select_improved_policy(node_id, parameters))
            } else {
                self.select_improved_policy(node_id, parameters)
            };
            if root_edge.is_none() {
                root_edge = Some(edge_id);
            }
            let child_id = self.materialize_child(node_id, edge_id)?;
            edge_path.push((node_id, edge_id));
            node_path.push(child_id);
            node_id = child_id;
        }
    }

    /// Validates the outstanding leaf response without mutation.
    pub fn validate_pending_evaluation(&self, evaluation: &Evaluation) -> Result<(), SearchError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(SearchError::NoPendingEvaluation)?;
        self.validate_token(pending.token, evaluation.token)?;
        let leaf_id = *pending
            .node_path
            .last()
            .expect("pending paths always contain the root");
        self.validate_evaluation(leaf_id, evaluation)
    }

    /// Completes the outstanding leaf and returns its root edge.
    pub fn finish_simulation(&mut self, evaluation: Evaluation) -> Result<usize, SearchError> {
        self.validate_pending_evaluation(&evaluation)?;
        let pending = self
            .pending
            .take()
            .expect("pending evaluation was checked above");
        let leaf_id = *pending
            .node_path
            .last()
            .expect("pending paths always contain the root");
        let leaf_player = self.nodes[leaf_id].state.to_move();
        let value = evaluation.value;
        self.expand_node_unchecked(leaf_id, evaluation);
        self.backup(&pending.node_path, &pending.edge_path, leaf_player, value);
        Ok(pending.root_edge)
    }

    /// Drops an outstanding simulation without changing statistics.
    pub fn cancel_pending(&mut self) {
        self.pending = None;
    }

    /// Root edge index for an action.
    #[must_use]
    pub fn root_edge(&self, action: Action) -> Option<usize> {
        self.nodes[0]
            .edges
            .iter()
            .position(|edge| edge.action == action)
    }

    /// Completed-Q statistics for every root action.
    #[must_use]
    pub fn root_stats(&self) -> Vec<RootActionStats> {
        if !self.nodes[0].expanded {
            return Vec::new();
        }
        let completed = self.completed_q(0);
        self.nodes[0]
            .edges
            .iter()
            .zip(completed)
            .map(|(edge, q)| RootActionStats {
                action: edge.action,
                prior: edge.prior,
                logit: edge.logit,
                visits: edge.visits,
                q,
            })
            .collect()
    }

    /// Completed-Q policy-improvement target over all legal root actions.
    #[must_use]
    pub fn completed_q_target(&self, parameters: GumbelParameters) -> Vec<(Action, f32)> {
        if !self.nodes[0].expanded || parameters.validate().is_err() {
            return Vec::new();
        }
        let probabilities = self.improved_policy(0, parameters);
        self.nodes[0]
            .edges
            .iter()
            .zip(probabilities)
            .map(|(edge, probability)| (edge.action, probability))
            .collect()
    }

    /// Original evaluator logits in stable root-action order.
    #[must_use]
    pub fn root_logits(&self) -> Vec<f32> {
        self.nodes[0].edges.iter().map(|edge| edge.logit).collect()
    }

    /// Completed Q values in stable root-action order.
    #[must_use]
    pub fn root_completed_q(&self) -> Vec<f32> {
        self.completed_q(0)
    }

    /// Root edge visits in stable action order.
    #[must_use]
    pub fn root_visits(&self) -> Vec<u32> {
        self.nodes[0].edges.iter().map(|edge| edge.visits).collect()
    }

    fn materialize_child(&mut self, node_id: usize, edge_id: usize) -> Result<usize, SearchError> {
        if let Some(child) = self.nodes[node_id].edges[edge_id].child {
            return Ok(child);
        }
        let action = self.nodes[node_id].edges[edge_id].action;
        let mut child_state = self.nodes[node_id].state.clone();
        child_state.apply(action)?;
        let key = child_state.key();
        let child_id = if let Some(&existing) = self.transpositions.get(&key) {
            existing
        } else {
            let new_id = self.nodes.len();
            self.nodes.push(Node::new(child_state));
            self.transpositions.insert(key, new_id);
            new_id
        };
        self.nodes[node_id].edges[edge_id].child = Some(child_id);
        Ok(child_id)
    }

    fn select_improved_policy(&self, node_id: usize, parameters: GumbelParameters) -> usize {
        let node = &self.nodes[node_id];
        let improved_policy = self.improved_policy(node_id, parameters);
        let total_visits: u32 = node.edges.iter().map(|edge| edge.visits).sum();
        let denominator = (total_visits + 1) as f32;
        let mut best = 0_usize;
        let mut best_score = f32::NEG_INFINITY;
        for (index, (edge, probability)) in node.edges.iter().zip(improved_policy).enumerate() {
            let score = probability - edge.visits as f32 / denominator;
            if score > best_score {
                best = index;
                best_score = score;
            }
        }
        best
    }

    fn improved_policy(&self, node_id: usize, parameters: GumbelParameters) -> Vec<f32> {
        let node = &self.nodes[node_id];
        let completed_q = self.completed_q(node_id);
        let max_visits = node.edges.iter().map(|edge| edge.visits).max().unwrap_or(0);
        let scale = parameters.sigma_scale(max_visits);
        let improved_logits: Vec<_> = node
            .edges
            .iter()
            .zip(completed_q)
            .map(|(edge, q)| edge.logit + scale * q)
            .collect();
        softmax(&improved_logits)
    }

    /// Appendix D mixed-value completion.
    fn completed_q(&self, node_id: usize) -> Vec<f32> {
        let node = &self.nodes[node_id];
        let total_visits: u32 = node.edges.iter().map(|edge| edge.visits).sum();
        let (prior_weighted_q, visited_prior) = node
            .edges
            .iter()
            .filter(|edge| edge.visits > 0)
            .fold((0.0_f64, 0.0_f64), |(weighted_q, prior_sum), edge| {
                let q = edge.value_sum / f64::from(edge.visits);
                (
                    weighted_q + f64::from(edge.prior) * q,
                    prior_sum + f64::from(edge.prior),
                )
            });
        let visited_estimate = if visited_prior > 0.0 {
            prior_weighted_q / visited_prior
        } else {
            f64::from(node.evaluation_value)
        };
        let mixed = ((f64::from(node.evaluation_value)
            + f64::from(total_visits) * visited_estimate)
            / f64::from(total_visits + 1)) as f32;
        node.edges
            .iter()
            .map(|edge| {
                if edge.visits == 0 {
                    mixed
                } else {
                    (edge.value_sum / f64::from(edge.visits)) as f32
                }
            })
            .collect()
    }

    fn validate_token(&self, expected: u64, actual: u64) -> Result<(), SearchError> {
        if actual == expected {
            Ok(())
        } else {
            Err(SearchError::TokenMismatch { expected, actual })
        }
    }

    fn validate_evaluation(
        &self,
        node_id: usize,
        evaluation: &Evaluation,
    ) -> Result<(), SearchError> {
        let expected = self.nodes[node_id].state.legal_actions().len();
        if evaluation.policy_logits.len() != expected {
            return Err(SearchError::PolicyLength {
                expected,
                actual: evaluation.policy_logits.len(),
            });
        }
        if !evaluation.value.is_finite()
            || evaluation
                .policy_logits
                .iter()
                .any(|logit| !logit.is_finite())
        {
            return Err(SearchError::NonFiniteEvaluation);
        }
        if !(-1.0..=1.0).contains(&evaluation.value) {
            return Err(SearchError::ValueOutOfRange(evaluation.value));
        }
        Ok(())
    }

    fn expand_node_unchecked(&mut self, node_id: usize, evaluation: Evaluation) {
        let actions = self.nodes[node_id].state.legal_actions().to_vec();
        let priors = softmax(&evaluation.policy_logits);
        self.nodes[node_id].evaluation_value = evaluation.value;
        self.nodes[node_id].edges = actions
            .into_iter()
            .zip(priors)
            .zip(evaluation.policy_logits)
            .map(|((action, prior), logit)| Edge {
                action,
                prior,
                logit,
                visits: 0,
                value_sum: 0.0,
                child: None,
            })
            .collect();
        self.nodes[node_id].expanded = true;
    }

    fn backup(
        &mut self,
        node_path: &[usize],
        edge_path: &[(usize, usize)],
        leaf_player: Player,
        leaf_value: f32,
    ) {
        for &node_id in node_path {
            let sign = if self.nodes[node_id].state.to_move() == leaf_player {
                1.0
            } else {
                -1.0
            };
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].value_sum += f64::from(sign * leaf_value);
        }
        for &(node_id, edge_id) in edge_path {
            let sign = if self.nodes[node_id].state.to_move() == leaf_player {
                1.0
            } else {
                -1.0
            };
            let edge = &mut self.nodes[node_id].edges[edge_id];
            edge.visits += 1;
            edge.value_sum += f64::from(sign * leaf_value);
        }
    }
}

fn fresh_evaluation_token() -> u64 {
    let token = NEXT_EVALUATION_TOKEN.fetch_add(1, Ordering::Relaxed);
    assert_ne!(token, 0, "evaluation token space exhausted");
    token
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f64> = values
        .iter()
        .map(|value| f64::from(*value - max).exp())
        .collect();
    let sum: f64 = exponentials.iter().sum();
    exponentials
        .into_iter()
        .map(|value| (value / sum) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use star_engine::{Board, Player};

    use super::*;

    fn evaluation(request: &EvaluationRequest, value: f32) -> Evaluation {
        Evaluation {
            token: request.token,
            value,
            policy_logits: vec![0.0; request.legal_actions.len()],
        }
    }

    fn initialize_uniform(tree: &mut SearchTree, value: f32) {
        let request = tree.root_request().unwrap();
        tree.initialize_root(evaluation(&request, value)).unwrap();
    }

    #[test]
    fn transposition_reuses_completed_pair_state() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut state = GameState::new(board);
        state.apply(Action::Place(0)).unwrap();
        let mut tree = SearchTree::new(state);
        initialize_uniform(&mut tree, 0.0);

        let a = tree.root_edge(Action::Place(1)).unwrap();
        let b = tree.root_edge(Action::Place(2)).unwrap();
        let after_a = tree.materialize_child(0, a).unwrap();
        let after_b = tree.materialize_child(0, b).unwrap();
        let request_a = EvaluationRequest {
            token: fresh_evaluation_token(),
            state: tree.nodes[after_a].state.clone(),
            legal_actions: tree.nodes[after_a].state.legal_actions().to_vec(),
        };
        let request_b = EvaluationRequest {
            token: fresh_evaluation_token(),
            state: tree.nodes[after_b].state.clone(),
            legal_actions: tree.nodes[after_b].state.legal_actions().to_vec(),
        };
        tree.expand_node_unchecked(after_a, evaluation(&request_a, 0.0));
        tree.expand_node_unchecked(after_b, evaluation(&request_b, 0.0));
        let b_after_a = tree.nodes[after_a]
            .edges
            .iter()
            .position(|edge| edge.action == Action::Place(2))
            .unwrap();
        let a_after_b = tree.nodes[after_b]
            .edges
            .iter()
            .position(|edge| edge.action == Action::Place(1))
            .unwrap();

        let pair_ab = tree.materialize_child(after_a, b_after_a).unwrap();
        let pair_ba = tree.materialize_child(after_b, a_after_b).unwrap();
        assert_eq!(pair_ab, pair_ba);
        assert_eq!(tree.nodes[pair_ab].state.to_move(), Player::Zero);
    }

    #[test]
    fn appendix_d_completion_uses_prior_weighted_visited_q() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut state = GameState::new(board);
        state.apply(Action::Place(0)).unwrap();
        let mut tree = SearchTree::new(state);
        initialize_uniform(&mut tree, 0.2);
        tree.nodes[0].edges.truncate(3);
        tree.nodes[0].edges[0].prior = 0.2;
        tree.nodes[0].edges[0].visits = 2;
        tree.nodes[0].edges[0].value_sum = 2.0;
        tree.nodes[0].edges[1].prior = 0.3;
        tree.nodes[0].edges[1].visits = 1;
        tree.nodes[0].edges[1].value_sum = -1.0;
        tree.nodes[0].edges[2].prior = 0.5;

        let completed = tree.completed_q(0);
        let visited_prior_q = (0.2 - 0.3) / 0.5;
        let expected_mixed = (0.2 + 3.0 * visited_prior_q) / 4.0;
        assert!((completed[0] - 1.0).abs() < 1.0e-6);
        assert!((completed[1] + 1.0).abs() < 1.0e-6);
        assert!((completed[2] - expected_mixed).abs() < 1.0e-6);
    }

    #[test]
    fn interior_selection_matches_improved_policy_visit_deficit() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut state = GameState::new(board);
        state.apply(Action::Place(0)).unwrap();
        let mut tree = SearchTree::new(state);
        initialize_uniform(&mut tree, 0.0);
        tree.nodes[0].edges.truncate(3);
        tree.nodes[0].edges[0].visits = 1;
        assert_eq!(tree.select_improved_policy(0, GumbelParameters::PAPER), 1);
        tree.nodes[0].edges[1].visits = 1;
        assert_eq!(tree.select_improved_policy(0, GumbelParameters::PAPER), 2);
    }

    #[test]
    fn backup_preserves_sign_for_same_player_then_flips_at_turn_boundary() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut state = GameState::new(board);
        state.apply(Action::Place(0)).unwrap();
        let mut tree = SearchTree::new(state);
        initialize_uniform(&mut tree, 0.0);

        let first = tree.root_edge(Action::Place(1)).unwrap();
        let request = match tree
            .start_simulation(Some(first), GumbelParameters::PAPER)
            .unwrap()
        {
            SimulationStart::NeedsEvaluation(request) => request,
            SimulationStart::Terminal { .. } => panic!("unexpected terminal"),
        };
        assert_eq!(request.state.to_move(), Player::One);
        tree.finish_simulation(evaluation(&request, 0.75)).unwrap();
        assert_eq!(tree.root_stats()[first].q, 0.75);

        let mut midturn = tree.root_state().clone();
        midturn.apply(Action::Place(3)).unwrap();
        let mut boundary_tree = SearchTree::new(midturn);
        initialize_uniform(&mut boundary_tree, 0.0);
        let second = boundary_tree.root_edge(Action::Place(4)).unwrap();
        let request = match boundary_tree
            .start_simulation(Some(second), GumbelParameters::PAPER)
            .unwrap()
        {
            SimulationStart::NeedsEvaluation(request) => request,
            SimulationStart::Terminal { .. } => panic!("unexpected terminal"),
        };
        assert_eq!(request.state.to_move(), Player::Zero);
        boundary_tree
            .finish_simulation(evaluation(&request, 0.75))
            .unwrap();
        assert_eq!(boundary_tree.root_stats()[second].q, -0.75);
    }

    #[test]
    fn response_token_is_mandatory() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut state = GameState::new(board);
        state.apply(Action::Place(0)).unwrap();
        let mut tree = SearchTree::new(state);
        let request = tree.root_request().unwrap();
        let mut response = evaluation(&request, 0.0);
        response.token = response.token.wrapping_add(1);
        assert!(matches!(
            tree.initialize_root(response),
            Err(SearchError::TokenMismatch { .. })
        ));
        assert!(!tree.is_initialized());
    }

    #[test]
    fn board_fill_needs_no_evaluator_and_is_cached() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut nearly_full = GameState::new(board);
        let last = nearly_full.board().node_count() - 1;
        for node in 0..last {
            nearly_full.apply(Action::Place(node)).unwrap();
        }
        let mut fill_tree = SearchTree::new(nearly_full);
        initialize_uniform(&mut fill_tree, 0.0);
        let fill = fill_tree.root_edge(Action::Place(last)).unwrap();
        assert!(matches!(
            fill_tree
                .start_simulation(Some(fill), GumbelParameters::PAPER)
                .unwrap(),
            SimulationStart::Terminal { root_edge } if root_edge == fill
        ));
    }
}
