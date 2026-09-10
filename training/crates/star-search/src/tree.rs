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
///
/// The evaluation request carries a full state clone (with its retained
/// placement history); boxing it would add an allocation per simulation on the
/// hot path, so the size difference between the variants is accepted.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
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

/// Search work retained by an exact-semantic root transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReuseStats {
    /// Unique states reachable from the new root, including that root.
    pub retained_nodes: usize,
    /// Completed outgoing-edge visits at the new root.
    pub retained_visits: u32,
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
///
/// When the root is the empty board of a pie game, every root child value `q`
/// is reported as `-|q|`: the responder will swap exactly when the opening
/// favors the opener, so the opener's exact payoff under an optimal swap is
/// `-|q|` and the best opening is the most balanced one. Deeper nodes are
/// never transformed.
#[derive(Clone, Debug)]
pub struct SearchTree {
    nodes: Vec<Node>,
    transpositions: HashMap<StateKey, usize>,
    pending: Option<PendingSimulation>,
    root_token: u64,
    pie_root_transform: bool,
    selection_scratch: Vec<f64>,
}

impl SearchTree {
    /// Creates an uninitialized tree.
    #[must_use]
    pub fn new(root: GameState) -> Self {
        let key = root.key();
        let pie_root_transform = root.is_pie_pending();
        let mut transpositions = HashMap::new();
        transpositions.insert(key, 0);
        Self {
            nodes: vec![Node::new(root)],
            transpositions,
            pending: None,
            root_token: fresh_evaluation_token(),
            pie_root_transform,
            selection_scratch: Vec::new(),
        }
    }

    /// Root state.
    #[must_use]
    pub fn root_state(&self) -> &GameState {
        &self.nodes[0].state
    }

    /// Whether root child values are reported under the optimal-swap payoff.
    #[must_use]
    pub const fn uses_pie_root_transform(&self) -> bool {
        self.pie_root_transform
    }

    /// Visit-weighted mean root value in the root player's perspective.
    ///
    /// Only backed-up child values contribute (the root's own network value is
    /// excluded), so this is the search's estimate rather than the prior. A
    /// pie-pending root reports the optimal-swap payoff. Returns `None` before
    /// any simulation completes.
    #[must_use]
    pub fn root_value(&self) -> Option<f32> {
        let node = &self.nodes[0];
        let total_visits: u32 = node.edges.iter().map(|edge| edge.visits).sum();
        if total_visits == 0 {
            return None;
        }
        let weighted: f64 = node
            .edges
            .iter()
            .filter(|edge| edge.visits > 0)
            .map(|edge| {
                let mean = edge.value_sum / f64::from(edge.visits);
                f64::from(edge.visits) * self.transform_root_q(mean)
            })
            .sum();
        Some((weighted / f64::from(total_visits)) as f32)
    }

    fn transform_root_q(&self, q: f64) -> f64 {
        if self.pie_root_transform { -q.abs() } else { q }
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

    /// Preview an unevaluated, unvisited root child without adding nodes or visits.
    ///
    /// The token belongs only to this prediction request. A later ordinary
    /// simulation issues its own token and remains the sole pending simulation.
    pub fn preview_root_child(
        &self,
        edge: usize,
    ) -> Result<Option<EvaluationRequest>, SearchError> {
        self.validate_refreshable_root()?;
        let root_edge = self.nodes[0]
            .edges
            .get(edge)
            .ok_or(SearchError::InvalidRootEdge(edge))?;
        if root_edge.visits != 0 {
            return Ok(None);
        }
        let mut state = if let Some(child) = root_edge.child {
            let child = &self.nodes[child];
            if child.expanded || child.terminal_value.is_some() {
                return Ok(None);
            }
            child.state.clone()
        } else {
            let mut state = self.nodes[0].state.clone();
            state.apply(root_edge.action)?;
            state
        };
        if state.is_terminal() {
            return Ok(None);
        }
        if let Some(&existing) = self.transpositions.get(&state.key()) {
            let child = &self.nodes[existing];
            if child.expanded || child.terminal_value.is_some() {
                return Ok(None);
            }
            state = child.state.clone();
        }
        Ok(Some(EvaluationRequest {
            token: fresh_evaluation_token(),
            legal_actions: state.legal_actions().to_vec(),
            state,
        }))
    }

    /// Whether an exact, expanded nonterminal proper descendant is reusable.
    ///
    /// This tests full semantic-key equality, including observable history;
    /// outstanding leaf evaluation makes every target ineligible.
    #[must_use]
    pub fn can_reuse_root(&self, target: &GameState) -> bool {
        self.pending.is_none()
            && !target.is_terminal()
            && self
                .transpositions
                .get(&target.key())
                .is_some_and(|&index| {
                    index != 0
                        && self.nodes[index].expanded
                        && self.nodes[index].terminal_value.is_none()
                })
    }

    /// Retain only the exact target's reachable DAG, preserving node-local values.
    ///
    /// The current root, ineligible targets and graphs larger than `max_nodes` return
    /// `None` without changing the tree. The supplied state replaces the root's
    /// equivalent stored history representation. Root expansion visits are
    /// excluded from retained simulation counts; outgoing edge statistics stay.
    pub fn reuse_root(
        &mut self,
        target: GameState,
        max_nodes: usize,
    ) -> Result<Option<ReuseStats>, SearchError> {
        if self.pending.is_some() {
            return Err(SearchError::PendingEvaluation);
        }
        if max_nodes == 0 || !self.can_reuse_root(&target) {
            return Ok(None);
        }
        let old_root = self.transpositions[&target.key()];
        let mut remap = HashMap::new();
        let mut retained = Vec::new();
        let mut stack = vec![old_root];
        while let Some(old_index) = stack.pop() {
            if remap.contains_key(&old_index) {
                continue;
            }
            if retained.len() == max_nodes {
                return Ok(None);
            }
            remap.insert(old_index, retained.len());
            retained.push(old_index);
            for edge in self.nodes[old_index].edges.iter().rev() {
                if let Some(child) = edge.child {
                    stack.push(child);
                }
            }
        }
        let mut nodes: Vec<_> = retained
            .into_iter()
            .map(|old_index| {
                let mut node = self.nodes[old_index].clone();
                for edge in &mut node.edges {
                    edge.child = edge.child.map(|child| remap[&child]);
                }
                node
            })
            .collect();
        let root = &mut nodes[0];
        root.state = target;
        root.visits = root.edges.iter().map(|edge| edge.visits).sum();
        root.value_sum = root.edges.iter().map(|edge| edge.value_sum).sum();
        let stats = ReuseStats {
            retained_nodes: nodes.len(),
            retained_visits: nodes[0].visits,
        };
        let transpositions = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.state.key(), index))
            .collect();
        let root_token = fresh_evaluation_token();
        self.pie_root_transform = nodes[0].state.is_pie_pending();
        self.nodes = nodes;
        self.transpositions = transpositions;
        self.root_token = root_token;
        self.selection_scratch = Vec::new();
        Ok(Some(stats))
    }

    fn validate_refreshable_root(&self) -> Result<(), SearchError> {
        if self.nodes[0].state.is_terminal() {
            return Err(SearchError::TerminalRoot);
        }
        if !self.nodes[0].expanded {
            return Err(SearchError::RootUninitialized);
        }
        if self.pending.is_some() {
            return Err(SearchError::PendingEvaluation);
        }
        Ok(())
    }

    /// Request fresh root predictions while retaining existing search statistics.
    pub fn root_refresh_request(&self) -> Result<EvaluationRequest, SearchError> {
        self.validate_refreshable_root()?;
        Ok(EvaluationRequest {
            token: self.root_token,
            state: self.nodes[0].state.clone(),
            legal_actions: self.nodes[0].state.legal_actions().to_vec(),
        })
    }

    /// Validate a root refresh without changing predictions or statistics.
    pub fn validate_root_refresh(&self, evaluation: &Evaluation) -> Result<(), SearchError> {
        self.validate_refreshable_root()?;
        self.validate_token(self.root_token, evaluation.token)?;
        self.validate_evaluation(0, evaluation)
    }

    /// Refresh network value, logits and priors without changing visits or children.
    pub fn refresh_root_evaluation(&mut self, evaluation: Evaluation) -> Result<(), SearchError> {
        self.validate_root_refresh(&evaluation)?;
        let priors = softmax(&evaluation.policy_logits);
        let root = &mut self.nodes[0];
        root.evaluation_value = evaluation.value;
        for ((edge, prior), logit) in root
            .edges
            .iter_mut()
            .zip(priors)
            .zip(evaluation.policy_logits)
        {
            edge.prior = prior;
            edge.logit = logit;
        }
        self.root_token = fresh_evaluation_token();
        Ok(())
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

    fn select_improved_policy(&mut self, node_id: usize, parameters: GumbelParameters) -> usize {
        if self.selection_scratch.is_empty() {
            // Shallow forced-root searches may never select inside the tree.
            // Allocate only on first use, but size for a later root selection:
            // every placement reduces the available interior action count.
            self.selection_scratch = vec![0.0; self.nodes[0].edges.len()];
        }
        let node = &self.nodes[node_id];
        let transform = node_id == 0 && self.pie_root_transform;
        let edge_q = |edge: &Edge| {
            let mean = edge.value_sum / f64::from(edge.visits);
            if transform { -mean.abs() } else { mean }
        };
        let mut total_visits = 0_u32;
        let mut max_visits = 0_u32;
        let mut prior_weighted_q = 0.0_f64;
        let mut visited_prior = 0.0_f64;
        for edge in &node.edges {
            total_visits += edge.visits;
            max_visits = max_visits.max(edge.visits);
            if edge.visits > 0 {
                prior_weighted_q += f64::from(edge.prior) * edge_q(edge);
                visited_prior += f64::from(edge.prior);
            }
        }
        let visited_estimate = if visited_prior > 0.0 {
            prior_weighted_q / visited_prior
        } else {
            f64::from(node.evaluation_value)
        };
        let mixed = ((f64::from(node.evaluation_value)
            + f64::from(total_visits) * visited_estimate)
            / f64::from(total_visits + 1)) as f32;
        let scale = parameters.sigma_scale(max_visits);
        let weights = &mut self.selection_scratch[..node.edges.len()];
        let mut max_logit = f32::NEG_INFINITY;
        for (edge, weight) in node.edges.iter().zip(weights.iter_mut()) {
            let q = if edge.visits == 0 {
                mixed
            } else {
                edge_q(edge) as f32
            };
            let logit = edge.logit + scale * q;
            // Preserve the legacy FP32 Q/logit rounding before the FP64 exp.
            *weight = f64::from(logit);
            max_logit = max_logit.max(logit);
        }
        let sum: f64 = weights
            .iter_mut()
            .map(|weight| {
                *weight = f64::from(*weight as f32 - max_logit).exp();
                *weight
            })
            .sum();
        let denominator = (total_visits + 1) as f32;
        let mut best = 0_usize;
        let mut best_score = f32::NEG_INFINITY;
        for (index, (edge, weight)) in node.edges.iter().zip(weights.iter()).enumerate() {
            let probability = (*weight / sum) as f32;
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
    ///
    /// At a pie-pending root the visited child means enter as `-|q|`; the
    /// root's own network value already estimates the optimal-swap payoff, so
    /// the mixed estimate for unvisited children needs no further transform.
    fn completed_q(&self, node_id: usize) -> Vec<f32> {
        let node = &self.nodes[node_id];
        let transform = node_id == 0 && self.pie_root_transform;
        let edge_q = |edge: &Edge| {
            let mean = edge.value_sum / f64::from(edge.visits);
            if transform { -mean.abs() } else { mean }
        };
        let total_visits: u32 = node.edges.iter().map(|edge| edge.visits).sum();
        let (prior_weighted_q, visited_prior) = node
            .edges
            .iter()
            .filter(|edge| edge.visits > 0)
            .fold((0.0_f64, 0.0_f64), |(weighted_q, prior_sum), edge| {
                (
                    weighted_q + f64::from(edge.prior) * edge_q(edge),
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
                    edge_q(edge) as f32
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

    use star_engine::{BitBoard, Board, Mode, Player, StateParts, Variant};

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

    fn evaluate_one_leaf(tree: &mut SearchTree, edge: usize, value: f32) -> EvaluationRequest {
        let request = match tree
            .start_simulation(Some(edge), GumbelParameters::PAPER)
            .unwrap()
        {
            SimulationStart::NeedsEvaluation(request) => request,
            SimulationStart::Terminal { .. } => panic!("expected a nonterminal leaf"),
        };
        tree.finish_simulation(evaluation(&request, value)).unwrap();
        request
    }

    fn expand_child(tree: &mut SearchTree, parent: usize, action: Action) -> usize {
        let edge = tree.nodes[parent]
            .edges
            .iter()
            .position(|edge| edge.action == action)
            .unwrap();
        let child = tree.materialize_child(parent, edge).unwrap();
        if !tree.nodes[child].expanded {
            tree.expand_node_unchecked(
                child,
                Evaluation {
                    token: 0,
                    value: 0.2,
                    policy_logits: vec![0.0; tree.nodes[child].state.legal_actions().len()],
                },
            );
        }
        child
    }

    #[test]
    fn root_child_preview_does_not_materialize_or_back_up_a_simulation() {
        let root = GameState::new(Arc::new(Board::new(4).unwrap()));
        let mut tree = SearchTree::new(root);
        assert!(matches!(
            tree.preview_root_child(0),
            Err(SearchError::RootUninitialized)
        ));
        initialize_uniform(&mut tree, 0.0);
        let before = format!("{tree:?}");
        let preview = tree.preview_root_child(0).unwrap().unwrap();
        let repeated = tree.preview_root_child(0).unwrap().unwrap();
        assert_ne!(preview.token, repeated.token);
        assert_eq!(preview.state.key(), repeated.state.key());
        assert_eq!(format!("{tree:?}"), before);
        assert!(matches!(
            tree.preview_root_child(usize::MAX),
            Err(SearchError::InvalidRootEdge(_))
        ));

        let pending = match tree
            .start_simulation(Some(0), GumbelParameters::PAPER)
            .unwrap()
        {
            SimulationStart::NeedsEvaluation(request) => request,
            SimulationStart::Terminal { .. } => panic!("unexpected terminal"),
        };
        assert_eq!(pending.state.key(), preview.state.key());
        assert_eq!(pending.legal_actions, preview.legal_actions);
        assert_ne!(pending.token, preview.token);
        assert!(matches!(
            tree.finish_simulation(evaluation(&preview, 0.3)),
            Err(SearchError::TokenMismatch { .. })
        ));
        tree.cancel_pending();
        assert!(tree.preview_root_child(0).unwrap().is_some());
        evaluate_one_leaf(&mut tree, 0, 0.3);
        assert_eq!(tree.simulations(), 1);
        assert!(tree.preview_root_child(0).unwrap().is_none());

        // An expanded transposition is skipped even when this edge has no
        // linked child yet (as can happen after reaching it by another path).
        let other = expand_child(&mut tree, 0, Action::Place(1));
        let edge = tree.root_edge(Action::Place(1)).unwrap();
        tree.nodes[0].edges[edge].child = None;
        assert!(tree.nodes[other].expanded);
        assert!(tree.preview_root_child(edge).unwrap().is_none());
        assert!(tree.nodes[0].edges[edge].child.is_none());
    }

    #[test]
    fn preview_skips_terminal_children_and_all_new_apis_reject_pending_work() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut almost_full = GameState::new(Arc::clone(&board));
        for node in 0..board.node_count() - 1 {
            almost_full.apply(Action::Place(node)).unwrap();
        }
        let mut terminal_child = SearchTree::new(almost_full);
        initialize_uniform(&mut terminal_child, 0.0);
        assert!(terminal_child.preview_root_child(0).unwrap().is_none());
        assert_eq!(terminal_child.unique_state_count(), 1);

        let mut tree = SearchTree::new(GameState::new(board));
        assert!(matches!(
            tree.root_refresh_request(),
            Err(SearchError::RootUninitialized)
        ));
        initialize_uniform(&mut tree, 0.0);
        let refresh = tree.root_refresh_request().unwrap();
        let target = tree.root_state().clone();
        tree.start_simulation(Some(0), GumbelParameters::PAPER)
            .unwrap();
        let before = format!("{tree:?}");
        assert!(!tree.can_reuse_root(&target));
        assert!(matches!(
            tree.reuse_root(target, 100),
            Err(SearchError::PendingEvaluation)
        ));
        assert!(matches!(
            tree.preview_root_child(1),
            Err(SearchError::PendingEvaluation)
        ));
        assert!(matches!(
            tree.root_refresh_request(),
            Err(SearchError::PendingEvaluation)
        ));
        assert!(matches!(
            tree.validate_root_refresh(&evaluation(&refresh, 0.0)),
            Err(SearchError::PendingEvaluation)
        ));
        assert!(matches!(
            tree.refresh_root_evaluation(evaluation(&refresh, 0.0)),
            Err(SearchError::PendingEvaluation)
        ));
        assert_eq!(format!("{tree:?}"), before);
    }

    #[test]
    fn root_reuse_prunes_and_remaps_a_shared_dag_atomically() {
        let mut root = GameState::new(Arc::new(Board::new(4).unwrap()));
        root.apply(Action::Place(0)).unwrap();
        let mut tree = SearchTree::new(root);
        initialize_uniform(&mut tree, 0.0);
        let a = expand_child(&mut tree, 0, Action::Place(1));
        let b = expand_child(&mut tree, 0, Action::Place(2));
        let pair = expand_child(&mut tree, a, Action::Place(2));
        assert_eq!(pair, expand_child(&mut tree, b, Action::Place(1)));
        let c = expand_child(&mut tree, pair, Action::Place(3));
        let d = expand_child(&mut tree, pair, Action::Place(4));
        let shared = expand_child(&mut tree, c, Action::Place(4));
        assert_eq!(shared, expand_child(&mut tree, d, Action::Place(3)));
        expand_child(&mut tree, 0, Action::Place(5));
        tree.nodes[pair].edges[0].visits = 3;
        tree.nodes[pair].edges[0].value_sum = 1.5;
        tree.nodes[pair].edges[1].visits = 2;
        tree.nodes[pair].edges[1].value_sum = -0.5;
        tree.nodes[pair].visits = 6; // Includes this node's original expansion.
        tree.nodes[pair].value_sum = 1.2;
        tree.select_improved_policy(0, GumbelParameters::PAPER);
        let target = tree.nodes[pair].state.clone();
        let token = tree.root_refresh_request().unwrap().token;
        let shared_key = tree.nodes[shared].state.key();
        let before = format!("{tree:?}");
        assert!(tree.reuse_root(target.clone(), 0).unwrap().is_none());
        assert!(tree.reuse_root(target.clone(), 3).unwrap().is_none());
        assert_eq!(format!("{tree:?}"), before);

        assert_eq!(
            tree.reuse_root(target.clone(), 4).unwrap(),
            Some(ReuseStats {
                retained_nodes: 4,
                retained_visits: 5,
            })
        );
        assert_eq!(tree.root_state().key(), target.key());
        assert_eq!(tree.simulations(), 5);
        assert_eq!(tree.nodes[0].value_sum, 1.0);
        assert_eq!(tree.selection_scratch.capacity(), 0);
        assert_ne!(tree.root_refresh_request().unwrap().token, token);
        assert_eq!(tree.transpositions.len(), 4);
        let remapped_shared = tree.transpositions[&shared_key];
        let incoming_shared = tree
            .nodes
            .iter()
            .flat_map(|node| &node.edges)
            .filter(|edge| edge.child == Some(remapped_shared))
            .count();
        assert_eq!(incoming_shared, 2);
        assert!(
            tree.nodes
                .iter()
                .flat_map(|node| &node.edges)
                .all(|edge| edge.child.is_none_or(|child| child < 4))
        );
    }

    #[test]
    fn root_reuse_requires_exact_history_but_preserves_supplied_order() {
        let mut root = GameState::new(Arc::new(Board::new(4).unwrap()));
        root.apply(Action::Place(0)).unwrap();
        let mut tree = SearchTree::new(root);
        initialize_uniform(&mut tree, 0.0);
        let a = expand_child(&mut tree, 0, Action::Place(1));
        let pair = expand_child(&mut tree, a, Action::Place(2));
        let target = tree.nodes[pair].state.clone();
        let key = target.key();
        let unknown_history = GameState::from_parts(
            target.shared_board(),
            StateParts {
                variant: target.variant(),
                stones: target.stones(),
                to_move: target.to_move(),
                moves_left: target.moves_left(),
                opening: target.is_opening(),
                swap_available: target.swap_available(),
                swapped: target.swapped(),
                current_turn: key.current_turn,
                previous_turn: BitBoard::empty(),
                own_previous_turn: key.own_previous_turn,
                handicap_stones: key.handicap_stones,
            },
        )
        .unwrap();
        let before = format!("{tree:?}");
        assert!(!tree.can_reuse_root(&unknown_history));
        assert!(tree.reuse_root(unknown_history, 100).unwrap().is_none());
        assert_eq!(format!("{tree:?}"), before);
        let supplied = target.with_ordered_history(&[], &[2, 1], &[0]).unwrap();
        assert!(tree.can_reuse_root(&supplied));
        assert_eq!(
            tree.reuse_root(supplied, 100)
                .unwrap()
                .unwrap()
                .retained_nodes,
            1
        );
        assert_eq!(tree.root_state().previous_turn_moves(), &[2, 1]);
        assert_eq!(tree.simulations(), 0);
    }

    #[test]
    fn root_reuse_rejects_same_root_and_unsearched_pie_swap_without_mutation() {
        for pie in [false, true] {
            let root = GameState::with_variant(
                Arc::new(Board::new(4).unwrap()),
                Variant::new(Mode::Double, 1, pie).unwrap(),
            );
            let mut tree = SearchTree::new(root);
            initialize_uniform(&mut tree, 0.0);
            let child = evaluate_one_leaf(&mut tree, 0, -0.75).state;
            let original = tree.root_state().clone();
            let before = format!("{tree:?}");
            assert!(!tree.can_reuse_root(&original));
            assert!(tree.reuse_root(original, 100).unwrap().is_none());
            assert_eq!(format!("{tree:?}"), before);
            if pie {
                let mut swapped = child;
                swapped.apply(Action::Swap).unwrap();
                assert!(!tree.can_reuse_root(&swapped));
                assert!(tree.reuse_root(swapped, 100).unwrap().is_none());
                assert_eq!(format!("{tree:?}"), before);
            }
        }
    }

    #[test]
    fn root_reuse_keeps_terminal_and_unexpanded_descendants_but_cannot_root_them() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut root = GameState::new(Arc::clone(&board));
        let a = board.node_count() - 3;
        for action in 0..a {
            root.apply(Action::Place(action)).unwrap();
        }
        let mut tree = SearchTree::new(root);
        initialize_uniform(&mut tree, 0.0);
        let target_index = expand_child(&mut tree, 0, Action::Place(a));
        let branch = expand_child(&mut tree, target_index, Action::Place(a + 1));
        let terminal = tree.materialize_child(branch, 0).unwrap();
        let unexpanded_edge = tree.nodes[target_index]
            .edges
            .iter()
            .position(|edge| edge.action == Action::Place(a + 2))
            .unwrap();
        let unexpanded = tree
            .materialize_child(target_index, unexpanded_edge)
            .unwrap();
        assert!(tree.nodes[terminal].terminal_value.is_some());
        assert!(!tree.nodes[unexpanded].expanded);
        for index in [terminal, unexpanded] {
            let target = tree.nodes[index].state.clone();
            let before = format!("{tree:?}");
            assert!(!tree.can_reuse_root(&target));
            assert!(tree.reuse_root(target, 100).unwrap().is_none());
            assert_eq!(format!("{tree:?}"), before);
        }
        let mut terminal_root = SearchTree::new(tree.nodes[terminal].state.clone());
        assert!(matches!(
            terminal_root.root_refresh_request(),
            Err(SearchError::TerminalRoot)
        ));
        assert!(matches!(
            terminal_root.preview_root_child(0),
            Err(SearchError::TerminalRoot)
        ));
        assert!(
            terminal_root
                .reuse_root(terminal_root.root_state().clone(), 100)
                .unwrap()
                .is_none()
        );

        let target = tree.nodes[target_index].state.clone();
        assert_eq!(
            tree.reuse_root(target, 4).unwrap().unwrap().retained_nodes,
            4
        );
        assert_eq!(
            tree.nodes
                .iter()
                .filter(|node| node.terminal_value.is_some())
                .count(),
            1
        );
        assert_eq!(tree.nodes.iter().filter(|node| !node.expanded).count(), 2);
        let exact = tree.root_edge(Action::Place(a + 1)).unwrap();
        assert!(matches!(
            tree.start_simulation(Some(exact), GumbelParameters::PAPER)
                .unwrap(),
            SimulationStart::Terminal { .. }
        ));
        let neural = tree.root_edge(Action::Place(a + 2)).unwrap();
        assert!(matches!(
            tree.start_simulation(Some(neural), GumbelParameters::PAPER)
                .unwrap(),
            SimulationStart::NeedsEvaluation(_)
        ));
    }

    #[test]
    fn root_reuse_and_continuation_keep_values_in_each_players_perspective() {
        for (mode, handicap, pie, prefix) in [
            (Mode::Classic, 1, false, 1),
            (Mode::Double, 1, false, 1),
            (Mode::Classic, 4, false, 0),
            (Mode::Double, 4, false, 0),
            (Mode::Classic, 4, false, 3),
            (Mode::Double, 4, false, 3),
            (Mode::Classic, 1, true, 0),
            (Mode::Double, 1, true, 0),
        ] {
            let mut root = GameState::with_variant(
                Arc::new(Board::new(4).unwrap()),
                Variant::new(mode, handicap, pie).unwrap(),
            );
            for action in 0..prefix {
                root.apply(Action::Place(action)).unwrap();
            }
            let mut tree = SearchTree::new(root);
            initialize_uniform(&mut tree, 0.0);
            let target = evaluate_one_leaf(&mut tree, 0, 0.2).state;
            let leaf = evaluate_one_leaf(&mut tree, 0, 0.75).state;
            let expected = if target.to_move() == leaf.to_move() {
                0.75
            } else {
                -0.75
            };
            assert_eq!(
                tree.reuse_root(target, 100)
                    .unwrap()
                    .unwrap()
                    .retained_visits,
                1
            );
            assert_eq!(tree.root_stats()[0].q, expected);
            assert_eq!(tree.root_value(), Some(expected));
            assert!(!tree.uses_pie_root_transform());
            let next = evaluate_one_leaf(&mut tree, 0, 0.5);
            let added = if tree.root_state().to_move() == next.state.to_move() {
                0.5
            } else {
                -0.5
            };
            assert_eq!(tree.root_stats()[0].q, (expected + added) / 2.0);
            assert_eq!(tree.simulations(), 2);
        }
    }

    #[test]
    fn root_refresh_validates_atomically_and_preserves_reused_search_statistics() {
        let mut tree = SearchTree::new(GameState::new(Arc::new(Board::new(4).unwrap())));
        initialize_uniform(&mut tree, 0.0);
        let stale = tree.root_refresh_request().unwrap();
        let target = evaluate_one_leaf(&mut tree, 0, 0.2).state;
        evaluate_one_leaf(&mut tree, 0, 0.75);
        tree.reuse_root(target, 100).unwrap().unwrap();
        assert!(matches!(
            tree.validate_root_refresh(&evaluation(&stale, 0.0)),
            Err(SearchError::TokenMismatch { .. })
        ));
        let request = tree.root_refresh_request().unwrap();
        let valid = Evaluation {
            token: request.token,
            value: -0.6,
            policy_logits: (0..request.legal_actions.len())
                .map(|index| index as f32 * 0.1)
                .collect(),
        };
        let before = format!("{tree:?}");
        for case in 0..5 {
            let mut invalid = valid.clone();
            match case {
                0 => invalid.token = stale.token,
                1 => {
                    invalid.policy_logits.pop();
                }
                2 => invalid.value = f32::NAN,
                3 => invalid.policy_logits[0] = f32::INFINITY,
                _ => invalid.value = 1.1,
            }
            assert!(tree.refresh_root_evaluation(invalid).is_err());
            assert_eq!(format!("{tree:?}"), before);
        }
        let visits = tree.root_visits();
        let sums: Vec<_> = tree.nodes[0]
            .edges
            .iter()
            .map(|edge| edge.value_sum.to_bits())
            .collect();
        let children: Vec<_> = tree.nodes[0].edges.iter().map(|edge| edge.child).collect();
        let nodes = tree.unique_state_count();
        let root_value = tree.root_value();
        let priors = softmax(&valid.policy_logits);
        tree.validate_root_refresh(&valid).unwrap();
        tree.refresh_root_evaluation(valid.clone()).unwrap();
        assert_eq!(tree.nodes[0].evaluation_value, -0.6);
        assert_eq!(tree.root_logits(), valid.policy_logits);
        assert_eq!(tree.root_visits(), visits);
        assert_eq!(
            tree.nodes[0]
                .edges
                .iter()
                .map(|edge| edge.value_sum.to_bits())
                .collect::<Vec<_>>(),
            sums
        );
        assert_eq!(
            tree.nodes[0]
                .edges
                .iter()
                .map(|edge| edge.child)
                .collect::<Vec<_>>(),
            children
        );
        assert_eq!(
            tree.nodes[0]
                .edges
                .iter()
                .map(|edge| edge.prior)
                .collect::<Vec<_>>(),
            priors
        );
        assert_eq!(tree.unique_state_count(), nodes);
        assert_eq!(tree.root_value(), root_value);
        assert!(matches!(
            tree.validate_root_refresh(&valid),
            Err(SearchError::TokenMismatch { .. })
        ));
        assert!(matches!(
            tree.root_request(),
            Err(SearchError::RootAlreadyInitialized)
        ));
        assert!(matches!(
            tree.initialize_root(evaluation(&request, 0.0)),
            Err(SearchError::RootAlreadyInitialized)
        ));
    }

    #[test]
    fn forced_root_leaf_needs_no_selection_scratch_allocation() {
        let root = GameState::new(Arc::new(Board::new(10).unwrap()));
        let mut tree = SearchTree::new(root);
        assert_eq!(tree.selection_scratch.capacity(), 0);
        initialize_uniform(&mut tree, 0.0);
        let request = match tree
            .start_simulation(Some(0), GumbelParameters::PAPER)
            .unwrap()
        {
            SimulationStart::NeedsEvaluation(request) => request,
            SimulationStart::Terminal { .. } => panic!("unexpected terminal"),
        };
        tree.finish_simulation(evaluation(&request, 0.0)).unwrap();
        assert_eq!(tree.selection_scratch.capacity(), 0);

        // Revisiting the expanded child needs interior selection. Reserving
        // root size on that first use also supports later unforced root moves.
        tree.start_simulation(Some(0), GumbelParameters::PAPER)
            .unwrap();
        assert_eq!(tree.selection_scratch.len(), tree.nodes[0].edges.len());
        let pointer = tree.selection_scratch.as_ptr();
        tree.cancel_pending();
        tree.start_simulation(None, GumbelParameters::PAPER)
            .unwrap();
        assert_eq!(tree.selection_scratch.as_ptr(), pointer);
    }

    /// Frozen allocation-based selection math from before scratch reuse.
    fn legacy_selection(
        node: &Node,
        transform: bool,
        parameters: GumbelParameters,
    ) -> (Vec<f32>, usize) {
        let edge_q = |edge: &Edge| {
            let mean = edge.value_sum / f64::from(edge.visits);
            if transform { -mean.abs() } else { mean }
        };
        let total_visits: u32 = node.edges.iter().map(|edge| edge.visits).sum();
        let (weighted_q, visited_prior) = node.edges.iter().filter(|edge| edge.visits > 0).fold(
            (0.0_f64, 0.0_f64),
            |(weighted_q, prior_sum), edge| {
                (
                    weighted_q + f64::from(edge.prior) * edge_q(edge),
                    prior_sum + f64::from(edge.prior),
                )
            },
        );
        let visited_estimate = if visited_prior > 0.0 {
            weighted_q / visited_prior
        } else {
            f64::from(node.evaluation_value)
        };
        let mixed = ((f64::from(node.evaluation_value)
            + f64::from(total_visits) * visited_estimate)
            / f64::from(total_visits + 1)) as f32;
        let completed_q: Vec<_> = node
            .edges
            .iter()
            .map(|edge| {
                if edge.visits == 0 {
                    mixed
                } else {
                    edge_q(edge) as f32
                }
            })
            .collect();
        let max_visits = node.edges.iter().map(|edge| edge.visits).max().unwrap_or(0);
        let scale = parameters.sigma_scale(max_visits);
        let logits: Vec<_> = node
            .edges
            .iter()
            .zip(completed_q)
            .map(|(edge, q)| edge.logit + scale * q)
            .collect();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f64> = logits
            .iter()
            .map(|value| f64::from(*value - max).exp())
            .collect();
        let sum: f64 = exponentials.iter().sum();
        let probabilities: Vec<f32> = exponentials
            .into_iter()
            .map(|value| (value / sum) as f32)
            .collect();
        let denominator = (total_visits + 1) as f32;
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (index, (edge, probability)) in node.edges.iter().zip(&probabilities).enumerate() {
            let score = probability - edge.visits as f32 / denominator;
            if score > best_score {
                best = index;
                best_score = score;
            }
        }
        (probabilities, best)
    }

    #[test]
    fn scratch_selection_matches_legacy_bits_for_all_variants_and_statistics() {
        let variants = [
            (Mode::Classic, 1, false),
            (Mode::Double, 1, false),
            (Mode::Classic, 9, false),
            (Mode::Double, 9, false),
            (Mode::Classic, 1, true),
            (Mode::Double, 1, true),
        ];
        let parameters = [
            GumbelParameters::PAPER,
            GumbelParameters {
                c_visit: 0.25,
                c_scale: 0.1,
            },
            GumbelParameters {
                c_visit: 100.0,
                c_scale: 10.0,
            },
        ];
        let mut random_state = 0x483d_382d_193b_0771_u64;
        let mut random = || {
            random_state = random_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (random_state >> 32) as u32
        };
        for rings in [4, 6, 8, 10] {
            let board = Arc::new(Board::new(rings).unwrap());
            for (mode, handicap, pie) in variants {
                let mut tree = SearchTree::new(GameState::with_variant(
                    Arc::clone(&board),
                    Variant::new(mode, handicap, pie).unwrap(),
                ));
                initialize_uniform(&mut tree, 0.0);
                let child = tree.materialize_child(0, 0).unwrap();
                tree.expand_node_unchecked(
                    child,
                    Evaluation {
                        token: 0,
                        value: 0.0,
                        policy_logits: vec![0.0; tree.nodes[child].state.legal_actions().len()],
                    },
                );
                assert_eq!(tree.selection_scratch.capacity(), 0);
                let mut scratch_pointer = None;
                // Allocate from an interior selection first, then exercise the
                // larger root with the same allocation.
                for node_id in [child, 0] {
                    let full_edges = tree.nodes[node_id].edges.clone();
                    for pattern in 0..32 {
                        let node = &mut tree.nodes[node_id];
                        node.edges = full_edges.clone();
                        let count = if matches!(pattern, 0 | 1 | 4) {
                            full_edges.len()
                        } else {
                            [1, 2, 3, full_edges.len()][pattern % 4]
                        };
                        node.edges.truncate(count);
                        node.evaluation_value = (random() % 2001) as f32 / 1000.0 - 1.0;
                        for (index, edge) in node.edges.iter_mut().enumerate() {
                            edge.visits =
                                [0, 0, 1, 2, 7, 31, 1000, 1_000_000][(random() % 8) as usize];
                            edge.logit = (random() % 3201) as f32 / 8.0 - 200.0;
                            edge.value_sum = (f64::from(random() % 20001) / 10000.0 - 1.0)
                                * f64::from(edge.visits);
                            match pattern {
                                0 | 1 => {
                                    edge.visits = pattern as u32;
                                    edge.logit = 0.0;
                                    edge.value_sum = 0.0;
                                }
                                2 => {
                                    edge.visits = u32::from(index % 2 == 0);
                                    edge.value_sum = 0.25 * f64::from(edge.visits);
                                }
                                3 => edge.logit = if index % 2 == 0 { -1000.0 } else { 1000.0 },
                                4 => {
                                    edge.visits = 1;
                                    edge.logit = -0.0;
                                    edge.value_sum = -0.0;
                                }
                                _ => {}
                            }
                        }
                        let priors =
                            softmax(&node.edges.iter().map(|edge| edge.logit).collect::<Vec<_>>());
                        for (edge, prior) in node.edges.iter_mut().zip(priors) {
                            // Exercise the zero visited-prior fallback as well
                            // as ordinary, very skewed and underflowed priors.
                            edge.prior = if pattern == 2 { 0.0 } else { prior };
                        }
                        for parameter in parameters {
                            let (expected_probabilities, expected_edge) = legacy_selection(
                                &tree.nodes[node_id],
                                node_id == 0 && pie,
                                parameter,
                            );
                            let actual_edge = tree.select_improved_policy(node_id, parameter);
                            let weights = &tree.selection_scratch[..expected_probabilities.len()];
                            let sum: f64 = weights.iter().sum();
                            let actual_bits: Vec<_> = weights
                                .iter()
                                .map(|weight| ((*weight / sum) as f32).to_bits())
                                .collect();
                            let expected_bits: Vec<_> = expected_probabilities
                                .iter()
                                .map(|probability| probability.to_bits())
                                .collect();
                            assert_eq!(
                                actual_bits, expected_bits,
                                "rings={rings} mode={mode:?} handicap={handicap} pie={pie} node={node_id} pattern={pattern}"
                            );
                            assert_eq!(actual_edge, expected_edge);
                            let pointer = tree.selection_scratch.as_ptr();
                            assert_eq!(pointer, *scratch_pointer.get_or_insert(pointer));
                            if pattern <= 1 {
                                assert_eq!(actual_edge, 0);
                            }
                        }
                    }
                }
            }
        }
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
    fn handicap_and_classic_backups_flip_only_at_turn_boundaries() {
        let board = Arc::new(Board::new(4).unwrap());
        // Handicap 3: the root is mid-opening with two placements left, so a
        // child evaluated for player 0 backs up with its sign preserved.
        let handicap = Variant::new(Mode::Double, 3, false).unwrap();
        let mut opening = GameState::with_variant(Arc::clone(&board), handicap);
        opening.apply(Action::Place(0)).unwrap();
        let mut tree = SearchTree::new(opening);
        initialize_uniform(&mut tree, 0.0);
        let edge = tree.root_edge(Action::Place(1)).unwrap();
        let request = match tree
            .start_simulation(Some(edge), GumbelParameters::PAPER)
            .unwrap()
        {
            SimulationStart::NeedsEvaluation(request) => request,
            SimulationStart::Terminal { .. } => panic!("unexpected terminal"),
        };
        assert_eq!(request.state.to_move(), Player::Zero);
        assert!(request.state.is_opening());
        tree.finish_simulation(evaluation(&request, 0.5)).unwrap();
        assert_eq!(tree.root_stats()[edge].q, 0.5);
        assert!(!tree.uses_pie_root_transform());

        // Classic: every placement ends the turn, so every child flips.
        let classic = Variant::new(Mode::Classic, 1, false).unwrap();
        let mut state = GameState::with_variant(board, classic);
        state.apply(Action::Place(0)).unwrap();
        let mut tree = SearchTree::new(state);
        initialize_uniform(&mut tree, 0.0);
        let edge = tree.root_edge(Action::Place(1)).unwrap();
        let request = match tree
            .start_simulation(Some(edge), GumbelParameters::PAPER)
            .unwrap()
        {
            SimulationStart::NeedsEvaluation(request) => request,
            SimulationStart::Terminal { .. } => panic!("unexpected terminal"),
        };
        assert_eq!(request.state.to_move(), Player::Zero);
        tree.finish_simulation(evaluation(&request, 0.5)).unwrap();
        assert_eq!(tree.root_stats()[edge].q, -0.5);
        assert_eq!(tree.root_value(), Some(-0.5));
    }

    #[test]
    fn pie_pending_root_reports_the_optimal_swap_payoff() {
        let board = Arc::new(Board::new(4).unwrap());
        let pie = Variant::new(Mode::Double, 1, true).unwrap();
        let root = GameState::with_variant(Arc::clone(&board), pie);
        assert!(root.is_pie_pending());
        let mut tree = SearchTree::new(root);
        assert!(tree.uses_pie_root_transform());
        assert_eq!(tree.root_value(), None);
        initialize_uniform(&mut tree, -0.1);

        // Opening 0 looks great for the opener (+0.8 for the responder means
        // -0.8 for the opener before the swap); opening 1 is balanced.
        for (node, responder_value) in [(0_u16, -0.8_f32), (1, 0.05)] {
            let edge = tree.root_edge(Action::Place(node)).unwrap();
            let request = match tree
                .start_simulation(Some(edge), GumbelParameters::PAPER)
                .unwrap()
            {
                SimulationStart::NeedsEvaluation(request) => request,
                SimulationStart::Terminal { .. } => panic!("unexpected terminal"),
            };
            assert_eq!(request.state.to_move(), Player::One);
            assert!(request.state.swap_available());
            tree.finish_simulation(evaluation(&request, responder_value))
                .unwrap();
        }
        let stats = tree.root_stats();
        let strong = tree.root_edge(Action::Place(0)).unwrap();
        let balanced = tree.root_edge(Action::Place(1)).unwrap();
        // Raw backup would give +0.8; the responder swaps, so the opener gets -0.8.
        assert!((stats[strong].q + 0.8).abs() < 1.0e-6);
        assert!((stats[balanced].q + 0.05).abs() < 1.0e-6);
        assert!(stats[balanced].q > stats[strong].q);
        assert_eq!(tree.root_value(), Some(-0.425));
        let target = tree.completed_q_target(GumbelParameters::PAPER);
        assert!(target[balanced].1 > target[strong].1);

        // Unvisited edges receive the mixed estimate built from transformed
        // values and the root's own (already optimal-swap) network value.
        let completed = tree.completed_q(0);
        let unvisited = (0..completed.len())
            .find(|edge| tree.nodes[0].edges[*edge].visits == 0)
            .unwrap();
        let prior = tree.nodes[0].edges[strong].prior;
        let visited_prior_q = (prior * -0.8 + prior * -0.05) / (2.0 * prior);
        let expected_mixed = (-0.1 + 2.0 * visited_prior_q) / 3.0;
        assert!((completed[unvisited] - expected_mixed).abs() < 1.0e-5);

        // Deeper nodes are untouched: the responder's root is a normal root.
        let mut responder = tree.root_state().clone();
        responder.apply(Action::Place(0)).unwrap();
        let responder_tree = SearchTree::new(responder);
        assert!(!responder_tree.uses_pie_root_transform());
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
