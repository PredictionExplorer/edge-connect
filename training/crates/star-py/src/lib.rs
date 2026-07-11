//! Coarse PyO3 interfaces for high-throughput actor and inference loops.
//!
//! Search inference uses flat token-addressed batches. CPU-heavy state,
//! scoring, and search work detaches from the Python interpreter and shares
//! one Rayon pool. Set `RAYON_NUM_THREADS` per actor process, or call
//! `configure_rayon_threads` before creating the first batch. No method creates
//! a nested pool and no single search tree is parallelized.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyBytesMethods};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use star_engine::{
    Action, BITBOARD_WORDS, BitBoard, Board, D5Maps, GameState, Player, RULES_HASH, RULES_SCHEMA,
    ScoreResult, ScoringScratch, Symmetry, rules_hash,
};
use star_search::{
    Evaluation, EvaluationRequest, GumbelParameters, GumbelSequentialHalving, RootSearchConfig,
    SearchTree, SimulationStart,
};

const NODE_FEATURE_DIM: usize = 15;
const GLOBAL_FEATURE_DIM: usize = 17;
const SCORE_COMPONENT_DIM: usize = 14;
const SEMANTIC_METADATA_DIM: usize = 4;
const FEATURE_SCHEMA_VERSION: u8 = 3;
const FEATURE_SCHEMA_HASH: u64 = 0x6b5b_00f6_38e9_c16b;
const SCORE_MARGIN_SUPPORT: i16 = 151;

#[pyclass(name = "StateData", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyStateData {
    rings: u8,
    node_count: u16,
    batch_size: usize,
    zero_bits: Vec<u64>,
    one_bits: Vec<u64>,
    legal_bits: Vec<u64>,
    hashes: Vec<u64>,
    stones_placed: Vec<u16>,
    to_move: Vec<u8>,
    moves_left: Vec<u8>,
    opening: Vec<bool>,
    mid_turn: Vec<bool>,
    terminal: Vec<bool>,
}

struct PackedStateRow {
    zero_bits: [u64; BITBOARD_WORDS],
    one_bits: [u64; BITBOARD_WORDS],
    legal_bits: [u64; BITBOARD_WORDS],
    hash: u64,
    stones_placed: u16,
    to_move: u8,
    moves_left: u8,
    opening: bool,
    mid_turn: bool,
    terminal: bool,
}

#[pymethods]
impl PyStateData {
    #[getter]
    const fn rings(&self) -> u8 {
        self.rings
    }

    #[getter]
    const fn node_count(&self) -> u16 {
        self.node_count
    }

    #[getter]
    const fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[getter]
    fn zero_bits(&self) -> Vec<u64> {
        self.zero_bits.clone()
    }

    #[getter]
    fn one_bits(&self) -> Vec<u64> {
        self.one_bits.clone()
    }

    #[getter]
    fn legal_bits(&self) -> Vec<u64> {
        self.legal_bits.clone()
    }

    #[getter]
    fn hashes(&self) -> Vec<u64> {
        self.hashes.clone()
    }

    #[getter]
    fn stones_placed(&self) -> Vec<u16> {
        self.stones_placed.clone()
    }

    #[getter]
    fn to_move(&self) -> Vec<u8> {
        self.to_move.clone()
    }

    #[getter]
    fn moves_left(&self) -> Vec<u8> {
        self.moves_left.clone()
    }

    #[getter]
    fn opening(&self) -> Vec<bool> {
        self.opening.clone()
    }

    #[getter]
    fn mid_turn(&self) -> Vec<bool> {
        self.mid_turn.clone()
    }

    #[getter]
    fn terminal(&self) -> Vec<bool> {
        self.terminal.clone()
    }

    /// Builds the schema-v3 features and score buffers without Python unpacking.
    fn feature_data(&self, py: Python<'_>) -> PyResult<PyFeatureData> {
        let board = Arc::new(Board::new(self.rings).map_err(value_error)?);
        let zero_bits = self.zero_bits.clone();
        let one_bits = self.one_bits.clone();
        let to_move = self.to_move.clone();
        let moves_left = self.moves_left.clone();
        let opening = self.opening.clone();
        py.detach(move || {
            let states =
                decode_semantic_states(board, zero_bits, one_bits, to_move, moves_left, opening)
                    .map_err(PyValueError::new_err)?;
            Ok(pack_feature_states(&states))
        })
    }
}

#[pyclass(name = "TrajectoryData", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyTrajectoryData {
    batch_size: usize,
    last_move: Vec<i32>,
    current_turn_offsets: Vec<usize>,
    current_turn_moves: Vec<u16>,
    turn_count: Vec<u32>,
}

#[pymethods]
impl PyTrajectoryData {
    #[getter]
    const fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[getter]
    fn last_move(&self) -> Vec<i32> {
        self.last_move.clone()
    }

    /// CSR offsets into `current_turn_moves`.
    #[getter]
    fn current_turn_offsets(&self) -> Vec<usize> {
        self.current_turn_offsets.clone()
    }

    #[getter]
    fn current_turn_moves(&self) -> Vec<u16> {
        self.current_turn_moves.clone()
    }

    #[getter]
    fn turn_count(&self) -> Vec<u32> {
        self.turn_count.clone()
    }
}

#[pyclass(name = "ScoreData", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyScoreData {
    batch_size: usize,
    node_count: u16,
    components: Vec<i32>,
    node_owner: Vec<i8>,
    alive_bits: Vec<u64>,
    winner: Vec<i8>,
    terminal_value: Vec<f32>,
    outcome_class: Vec<u8>,
    score_margin: Vec<i16>,
}

#[pymethods]
impl PyScoreData {
    #[getter]
    const fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[getter]
    const fn node_count(&self) -> u16 {
        self.node_count
    }

    /// Fourteen integers per row: six components per player, contested, leader.
    #[getter]
    fn components(&self) -> Vec<i32> {
        self.components.clone()
    }

    /// Flattened owner rows (`-1`, `0`, or `1`) with `node_count` columns.
    #[getter]
    fn node_owner(&self) -> Vec<i8> {
        self.node_owner.clone()
    }

    /// Five alive-stone words per row.
    #[getter]
    fn alive_bits(&self) -> Vec<u64> {
        self.alive_bits.clone()
    }

    /// Static leader (`-1`, `0`, or `1`) for each row.
    #[getter]
    fn winner(&self) -> Vec<i8> {
        self.winner.clone()
    }

    /// Current-player terminal value; nonterminal rows contain zero.
    #[getter]
    fn terminal_value(&self) -> Vec<f32> {
        self.terminal_value.clone()
    }

    /// Binary outcome class (`loss=0`, `win=1`); nonterminal rows contain `255`.
    #[getter]
    fn outcome_class(&self) -> Vec<u8> {
        self.outcome_class.clone()
    }

    /// Current-player conventional score margin.
    #[getter]
    fn score_margin(&self) -> Vec<i16> {
        self.score_margin.clone()
    }
}

struct FeatureBuffers {
    rings: Vec<u8>,
    node_features: Vec<u8>,
    global_features: Vec<u8>,
    node_mask: Vec<u8>,
    legal_action_mask: Vec<u8>,
    score_components: Vec<u8>,
    node_owner: Vec<u8>,
    alive_stones: Vec<u8>,
}

/// Contiguous schema-v3 model features plus the exact native score annotations.
#[pyclass(name = "FeatureData", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyFeatureData {
    batch_size: usize,
    max_nodes: usize,
    buffers: Arc<FeatureBuffers>,
}

#[pymethods]
impl PyFeatureData {
    #[getter]
    const fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[getter]
    const fn max_nodes(&self) -> usize {
        self.max_nodes
    }

    #[getter]
    const fn node_feature_dim(&self) -> usize {
        NODE_FEATURE_DIM
    }

    #[getter]
    const fn global_feature_dim(&self) -> usize {
        GLOBAL_FEATURE_DIM
    }

    #[getter]
    const fn score_component_dim(&self) -> usize {
        SCORE_COMPONENT_DIM
    }

    #[getter]
    const fn feature_schema_version(&self) -> u8 {
        FEATURE_SCHEMA_VERSION
    }

    #[getter]
    const fn feature_schema_hash(&self) -> u64 {
        FEATURE_SCHEMA_HASH
    }

    /// Native-endian `u8[batch_size]`.
    #[getter]
    fn rings<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.rings)
    }

    /// Native-endian `float32[batch_size, max_nodes, 15]`.
    #[getter]
    fn node_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.node_features)
    }

    /// Native-endian `float32[batch_size, 17]`.
    #[getter]
    fn global_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.global_features)
    }

    /// `uint8[batch_size, max_nodes]`, containing only zero and one.
    #[getter]
    fn node_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.node_mask)
    }

    /// `uint8[batch_size, max_nodes]` in the nodes-only layout.
    #[getter]
    fn legal_action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.legal_action_mask)
    }

    /// Native-endian `int32[batch_size, 14]` in `ScoreData.components` order.
    #[getter]
    fn score_components<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.score_components)
    }

    /// `int8[batch_size, max_nodes]`; padded nodes and unowned nodes are `-1`.
    #[getter]
    fn node_owner<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.node_owner)
    }

    /// `uint8[batch_size, max_nodes]`, containing only zero and one.
    #[getter]
    fn alive_stones<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.alive_stones)
    }
}

struct FeatureState {
    board: Arc<Board>,
    stones: [BitBoard; 2],
    to_move: Player,
    moves_left: u8,
    opening: bool,
    terminal: bool,
}

struct PackedFeatureRow {
    rings: u8,
    node_count: usize,
    node_features: Vec<f32>,
    global_features: [f32; GLOBAL_FEATURE_DIM],
    legal_nodes: Vec<u8>,
    score_components: [i32; SCORE_COMPONENT_DIM],
    node_owner: Vec<i8>,
    alive_stones: Vec<u8>,
}

/// Mutable homogeneous environment batch.
#[pyclass(name = "StateBatch")]
struct PyStateBatch {
    board: Arc<Board>,
    states: Vec<GameState>,
}

#[pymethods]
impl PyStateBatch {
    #[new]
    fn new(rings: u8, batch_size: usize) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive"));
        }
        let board = Arc::new(Board::new(rings).map_err(value_error)?);
        let states = (0..batch_size)
            .into_par_iter()
            .map(|_| GameState::new(Arc::clone(&board)))
            .collect();
        Ok(Self { board, states })
    }

    /// Constructs a batch from packed semantic (history-free) state fields.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_semantic(
        py: Python<'_>,
        rings: u8,
        zero_bits: Vec<u64>,
        one_bits: Vec<u64>,
        to_move: Vec<u8>,
        moves_left: Vec<u8>,
        opening: Vec<bool>,
    ) -> PyResult<Self> {
        let board = Arc::new(Board::new(rings).map_err(value_error)?);
        let shared = Arc::clone(&board);
        let states = py
            .detach(move || {
                decode_semantic_states(shared, zero_bits, one_bits, to_move, moves_left, opening)
            })
            .map_err(PyValueError::new_err)?;
        if states.is_empty() {
            return Err(PyValueError::new_err(
                "semantic import must contain at least one row",
            ));
        }
        Ok(Self { board, states })
    }

    fn __len__(&self) -> usize {
        self.states.len()
    }

    #[getter]
    fn rings(&self) -> u8 {
        self.board.rings()
    }

    #[getter]
    fn node_count(&self) -> u16 {
        self.board.node_count()
    }

    /// Resets all rows to the one-stone opening state.
    fn reset(&mut self, py: Python<'_>) {
        py.detach(|| {
            self.states.par_iter_mut().for_each(|state| {
                *state = GameState::new(Arc::clone(&self.board));
            });
        });
    }

    /// Resets only terminal rows, atomically.
    fn reset_many(&mut self, py: Python<'_>, indices: Vec<usize>) -> PyResult<()> {
        let replacements = py
            .detach(|| prepare_terminal_resets(&self.board, &self.states, &indices))
            .map_err(PyValueError::new_err)?;
        for (index, replacement) in replacements {
            self.states[index] = replacement;
        }
        Ok(())
    }

    /// Applies a transaction of indexed atomic actions.
    fn apply_many(
        &mut self,
        py: Python<'_>,
        indices: Vec<usize>,
        actions: Vec<i32>,
    ) -> PyResult<()> {
        let replacements = py
            .detach(|| prepare_applied_rows(&self.states, indices, actions))
            .map_err(PyValueError::new_err)?;
        for (index, replacement) in replacements {
            self.states[index] = replacement;
        }
        Ok(())
    }

    /// Atomically replaces selected rows from packed semantic state fields.
    #[allow(clippy::too_many_arguments)]
    fn replace_semantic(
        &mut self,
        py: Python<'_>,
        indices: Vec<usize>,
        zero_bits: Vec<u64>,
        one_bits: Vec<u64>,
        to_move: Vec<u8>,
        moves_left: Vec<u8>,
        opening: Vec<bool>,
    ) -> PyResult<()> {
        let board = Arc::clone(&self.board);
        let replacements = py
            .detach(|| {
                decode_semantic_states(board, zero_bits, one_bits, to_move, moves_left, opening)
            })
            .map_err(PyValueError::new_err)?;
        let replacements = py
            .detach(|| replace_rows(&self.states, &indices, replacements))
            .map_err(PyValueError::new_err)?;
        for (index, replacement) in replacements {
            self.states[index] = replacement;
        }
        Ok(())
    }

    /// Packed state metadata and fixed bitboards.
    fn data(&self, py: Python<'_>) -> PyStateData {
        py.detach(|| pack_states(&self.states))
    }

    /// Presentation metadata for replay/trajectory persistence, kept separate
    /// from evaluator state features.
    fn trajectory_data(&self, py: Python<'_>) -> PyTrajectoryData {
        py.detach(|| pack_trajectory_data(&self.states))
    }

    /// Exact static score, ownership, and alive-star annotations for all rows.
    fn score_data(&self, py: Python<'_>) -> PyScoreData {
        py.detach(|| score_states(&self.states, self.board.node_count()))
    }

    /// Contiguous schema-v3 model features and exact score annotations.
    fn feature_data(&self, py: Python<'_>) -> PyFeatureData {
        py.detach(|| pack_feature_states(&self.states))
    }

    /// Applies one D5 augmentation to every row.
    fn transformed(&self, py: Python<'_>, symmetry: u8) -> PyResult<Self> {
        let symmetry = Symmetry::from_index(symmetry)
            .ok_or_else(|| PyValueError::new_err("symmetry must be in 0..10"))?;
        let board = Arc::clone(&self.board);
        let states = py.detach(|| {
            let maps = D5Maps::new(&board);
            self.states
                .par_iter()
                .map(|state| maps.state(symmetry, state))
                .collect()
        });
        Ok(Self { board, states })
    }
}

#[pyclass(name = "EvalBatch", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyEvalBatch {
    tree_indices: Vec<usize>,
    tokens: Vec<u64>,
    states: PyStateData,
    features: PyFeatureData,
    legal_offsets: Vec<usize>,
    legal_actions: Vec<i32>,
}

#[pymethods]
impl PyEvalBatch {
    fn __len__(&self) -> usize {
        self.tree_indices.len()
    }

    #[getter]
    fn tree_indices(&self) -> Vec<usize> {
        self.tree_indices.clone()
    }

    #[getter]
    fn tokens(&self) -> Vec<u64> {
        self.tokens.clone()
    }

    #[getter]
    fn states(&self) -> PyStateData {
        self.states.clone()
    }

    /// Precomputed contiguous features for the request states.
    #[getter]
    fn features(&self) -> PyFeatureData {
        self.features.clone()
    }

    /// CSR offsets into `legal_actions`.
    #[getter]
    fn legal_offsets(&self) -> Vec<usize> {
        self.legal_offsets.clone()
    }

    /// Flattened legal node ids in ascending order per row.
    #[getter]
    fn legal_actions(&self) -> Vec<i32> {
        self.legal_actions.clone()
    }
}

#[derive(Clone, Debug)]
struct PendingRow {
    tree_index: usize,
    candidate: usize,
    token: u64,
    legal_count: usize,
}

struct PackedSearchRow {
    selected_action: i32,
    terminal: bool,
    terminal_value: f32,
    actions: Vec<i32>,
    visits: Vec<u32>,
    q_values: Vec<f32>,
    policy_target: Vec<f32>,
}

#[pyclass(name = "SearchResults", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PySearchResults {
    selected_actions: Vec<i32>,
    terminal: Vec<bool>,
    terminal_values: Vec<f32>,
    action_offsets: Vec<usize>,
    actions: Vec<i32>,
    visits: Vec<u32>,
    q_values: Vec<f32>,
    policy_target: Vec<f32>,
}

#[pymethods]
impl PySearchResults {
    /// Selected action per row; terminal rows use `-2`.
    #[getter]
    fn selected_actions(&self) -> Vec<i32> {
        self.selected_actions.clone()
    }

    #[getter]
    fn terminal(&self) -> Vec<bool> {
        self.terminal.clone()
    }

    #[getter]
    fn terminal_values(&self) -> Vec<f32> {
        self.terminal_values.clone()
    }

    #[getter]
    fn action_offsets(&self) -> Vec<usize> {
        self.action_offsets.clone()
    }

    #[getter]
    fn actions(&self) -> Vec<i32> {
        self.actions.clone()
    }

    #[getter]
    fn visits(&self) -> Vec<u32> {
        self.visits.clone()
    }

    #[getter]
    fn q_values(&self) -> Vec<f32> {
        self.q_values.clone()
    }

    #[getter]
    fn policy_target(&self) -> Vec<f32> {
        self.policy_target.clone()
    }
}

/// Ask/tell Gumbel MCTS over a full actor batch.
#[pyclass(name = "SearchBatch")]
struct PySearchBatch {
    trees: Vec<SearchTree>,
    schedulers: Option<Vec<Option<GumbelSequentialHalving>>>,
    config: RootSearchConfig,
    pending: Vec<PendingRow>,
}

#[pymethods]
impl PySearchBatch {
    #[new]
    #[pyo3(signature = (
        states,
        simulations=128,
        max_considered=16,
        c_visit=50.0,
        c_scale=1.0,
        deterministic_seed=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        states: PyRef<'_, PyStateBatch>,
        simulations: u32,
        max_considered: usize,
        c_visit: f32,
        c_scale: f32,
        deterministic_seed: Option<u64>,
    ) -> PyResult<Self> {
        if simulations == 0 {
            return Err(PyValueError::new_err("simulations must be positive"));
        }
        if max_considered == 0 {
            return Err(PyValueError::new_err("max_considered must be positive"));
        }
        let parameters = GumbelParameters { c_visit, c_scale };
        parameters.validate().map_err(value_error)?;
        let roots = states.states.clone();
        let trees = py.detach(|| roots.into_iter().map(SearchTree::new).collect());
        let config = deterministic_seed.map_or_else(
            || RootSearchConfig::fresh(simulations, max_considered, parameters),
            |seed| RootSearchConfig::deterministic(simulations, max_considered, parameters, seed),
        );
        Ok(Self {
            trees,
            schedulers: None,
            config,
            pending: Vec::new(),
        })
    }

    fn __len__(&self) -> usize {
        self.trees.len()
    }

    /// One inference row per active root; terminal roots are omitted.
    fn root_requests(&self, py: Python<'_>) -> PyResult<PyEvalBatch> {
        py.detach(|| {
            if self.schedulers.is_some() {
                return Err(PyRuntimeError::new_err(
                    "root evaluations were already submitted",
                ));
            }
            let active = active_root_requests(&self.trees).map_err(PyValueError::new_err)?;
            let (indices, requests): (Vec<_>, Vec<_>) = active.into_iter().unzip();
            Ok(pack_requests(indices, requests))
        })
    }

    /// Initializes active roots from flat token-addressed response buffers.
    fn initialize_roots(
        &mut self,
        py: Python<'_>,
        tokens: Vec<u64>,
        values: Vec<f32>,
        policy_offsets: Vec<usize>,
        policy_logits: Vec<f32>,
    ) -> PyResult<()> {
        py.detach(|| {
            if self.schedulers.is_some() {
                return Err(PyRuntimeError::new_err(
                    "root evaluations were already submitted",
                ));
            }
            let responses = unpack_evaluations(tokens, values, policy_offsets, policy_logits)?;
            let active = active_root_requests(&self.trees).map_err(PyValueError::new_err)?;
            let expected: Vec<_> = active.iter().map(|(_, request)| request.clone()).collect();
            let mut matched = match_evaluations(&expected, responses)?;
            let jobs: Vec<_> = active
                .into_iter()
                .map(|(tree_index, request)| {
                    (
                        tree_index,
                        matched
                            .remove(&request.token)
                            .expect("all responses were matched"),
                    )
                })
                .collect();
            let initialized: Result<Vec<_>, String> = jobs
                .into_par_iter()
                .map(|(tree_index, evaluation)| {
                    let mut tree = self.trees[tree_index].clone();
                    tree.initialize_root(evaluation)
                        .map_err(|error| error.to_string())?;
                    let scheduler = GumbelSequentialHalving::new(
                        &tree.root_logits(),
                        self.config.simulations,
                        self.config.max_considered,
                        self.config.parameters,
                        derive_root_seed(
                            self.config.nonce.value(),
                            tree.root_state().hash64(),
                            tree_index,
                        ),
                    )
                    .map_err(|error| error.to_string())?;
                    Ok((tree_index, tree, scheduler))
                })
                .collect();
            let initialized = initialized.map_err(PyValueError::new_err)?;
            let mut schedulers = vec![None; self.trees.len()];
            for (tree_index, tree, scheduler) in initialized {
                self.trees[tree_index] = tree;
                schedulers[tree_index] = Some(scheduler);
            }
            self.schedulers = Some(schedulers);
            Ok(())
        })
    }

    /// Selects at most one leaf per active tree and returns one packed batch.
    fn next_requests(&mut self, py: Python<'_>) -> PyResult<PyEvalBatch> {
        py.detach(|| {
            if !self.pending.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "submit the outstanding leaf batch first",
                ));
            }
            let schedulers = self
                .schedulers
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("initialize roots first"))?;
            let rows: Result<Vec<_>, String> = self
                .trees
                .par_iter_mut()
                .zip(schedulers.par_iter_mut())
                .enumerate()
                .map(|(tree_index, (tree, scheduler))| {
                    let Some(scheduler) = scheduler else {
                        return Ok(None);
                    };
                    while !scheduler.is_done() {
                        let candidate = scheduler
                            .next_candidate(&tree.root_completed_q(), &tree.root_visits())
                            .map_err(|error| error.to_string())?
                            .expect("unfinished scheduler returns a candidate");
                        match tree
                            .start_simulation(Some(candidate), self.config.parameters)
                            .map_err(|error| error.to_string())?
                        {
                            SimulationStart::Terminal { root_edge } => {
                                scheduler
                                    .record_simulation(root_edge)
                                    .map_err(|error| error.to_string())?;
                            }
                            SimulationStart::NeedsEvaluation(request) => {
                                let pending = PendingRow {
                                    tree_index,
                                    candidate,
                                    token: request.token,
                                    legal_count: request.legal_actions.len(),
                                };
                                return Ok(Some((pending, request)));
                            }
                        }
                    }
                    Ok(None)
                })
                .collect();
            let rows = rows.map_err(PyValueError::new_err)?;
            let mut requests = Vec::with_capacity(rows.len());
            let mut tree_indices = Vec::with_capacity(rows.len());
            for (pending, request) in rows.into_iter().flatten() {
                tree_indices.push(pending.tree_index);
                self.pending.push(pending);
                requests.push(request);
            }
            Ok(pack_requests(tree_indices, requests))
        })
    }

    /// Backs up a complete flat response batch, matched exclusively by token.
    fn submit(
        &mut self,
        py: Python<'_>,
        tokens: Vec<u64>,
        values: Vec<f32>,
        policy_offsets: Vec<usize>,
        policy_logits: Vec<f32>,
    ) -> PyResult<()> {
        py.detach(|| {
            if self.pending.is_empty() {
                return Err(PyRuntimeError::new_err("no leaf batch is pending"));
            }
            let responses = unpack_evaluations(tokens, values, policy_offsets, policy_logits)?;
            let expected_tokens: Vec<_> = self.pending.iter().map(|row| row.token).collect();
            let mut matched = match_token_set(&expected_tokens, responses)?;
            let validation: Result<Vec<_>, String> = self
                .pending
                .par_iter()
                .map(|row| {
                    let response = matched.get(&row.token).expect("all responses were matched");
                    validate_evaluation_native(response, row.legal_count)?;
                    self.trees[row.tree_index]
                        .validate_pending_evaluation(response)
                        .map_err(|error| error.to_string())
                })
                .collect();
            validation.map_err(PyValueError::new_err)?;

            let schedulers = self
                .schedulers
                .as_mut()
                .expect("pending leaves require initialized schedulers");
            let mut slots = vec![None; self.trees.len()];
            for row in &self.pending {
                let tree_index = row.tree_index;
                slots[tree_index] = Some((
                    row.clone(),
                    matched
                        .remove(&row.token)
                        .expect("validated response remains available"),
                ));
            }
            self.trees
                .par_iter_mut()
                .zip(schedulers.par_iter_mut())
                .zip(slots.into_par_iter())
                .for_each(|((tree, scheduler), slot)| {
                    let Some((row, response)) = slot else {
                        return;
                    };
                    let root_edge = tree
                        .finish_simulation(response)
                        .expect("parallel response was prevalidated");
                    debug_assert_eq!(root_edge, row.candidate);
                    scheduler
                        .as_mut()
                        .expect("pending tree has a scheduler")
                        .record_simulation(root_edge)
                        .expect("scheduler candidate was selected by this tree");
                });
            self.pending.clear();
            Ok(())
        })
    }

    /// Whether every active root consumed its exact simulation budget.
    fn is_done(&self) -> bool {
        self.schedulers.as_ref().is_some_and(|schedulers| {
            schedulers
                .iter()
                .flatten()
                .all(GumbelSequentialHalving::is_done)
        }) && self.pending.is_empty()
    }

    /// Flattened final root statistics and completed-Q targets.
    fn results(&self, py: Python<'_>) -> PyResult<PySearchResults> {
        py.detach(|| {
            if !self.is_done() {
                return Err(PyRuntimeError::new_err("search is not complete"));
            }
            let schedulers = self
                .schedulers
                .as_ref()
                .expect("completed search has schedulers");
            pack_search_results(&self.trees, schedulers, self.config.parameters)
                .map_err(PyValueError::new_err)
        })
    }
}

fn pack_search_results(
    trees: &[SearchTree],
    schedulers: &[Option<GumbelSequentialHalving>],
    parameters: GumbelParameters,
) -> Result<PySearchResults, String> {
    let rows: Result<Vec<_>, String> = trees
        .par_iter()
        .zip(schedulers.par_iter())
        .map(|(tree, scheduler)| {
            let Some(scheduler) = scheduler else {
                return Ok(PackedSearchRow {
                    selected_action: -2,
                    terminal: true,
                    terminal_value: tree
                        .root_terminal_value()
                        .expect("inactive roots are terminal"),
                    actions: Vec::new(),
                    visits: Vec::new(),
                    q_values: Vec::new(),
                    policy_target: Vec::new(),
                });
            };
            let stats = tree.root_stats();
            let selected = scheduler
                .selected(&tree.root_completed_q(), &tree.root_visits())
                .map_err(|error| error.to_string())?;
            Ok(PackedSearchRow {
                selected_action: stats[selected].action.code(),
                terminal: false,
                terminal_value: 0.0,
                actions: stats.iter().map(|row| row.action.code()).collect(),
                visits: stats.iter().map(|row| row.visits).collect(),
                q_values: stats.iter().map(|row| row.q).collect(),
                policy_target: tree
                    .completed_q_target(parameters)
                    .into_iter()
                    .map(|(_, probability)| probability)
                    .collect(),
            })
        })
        .collect();
    let rows = rows?;
    let action_count: usize = rows.iter().map(|row| row.actions.len()).sum();
    let mut selected_actions = Vec::with_capacity(rows.len());
    let mut terminal = Vec::with_capacity(rows.len());
    let mut terminal_values = Vec::with_capacity(rows.len());
    let mut action_offsets = Vec::with_capacity(rows.len() + 1);
    let mut actions = Vec::with_capacity(action_count);
    let mut visits = Vec::with_capacity(action_count);
    let mut q_values = Vec::with_capacity(action_count);
    let mut policy_target = Vec::with_capacity(action_count);
    action_offsets.push(0);
    for row in rows {
        selected_actions.push(row.selected_action);
        terminal.push(row.terminal);
        terminal_values.push(row.terminal_value);
        actions.extend(row.actions);
        visits.extend(row.visits);
        q_values.extend(row.q_values);
        policy_target.extend(row.policy_target);
        action_offsets.push(actions.len());
    }
    Ok(PySearchResults {
        selected_actions,
        terminal,
        terminal_values,
        action_offsets,
        actions,
        visits,
        q_values,
        policy_target,
    })
}

fn score_states(states: &[GameState], node_count: u16) -> PyScoreData {
    let scores: Vec<_> = states
        .par_iter()
        .map_init(ScoringScratch::default, |scratch, state| {
            scratch.score_state(state)
        })
        .collect();
    let mut components = Vec::with_capacity(states.len() * 14);
    let mut node_owner = Vec::with_capacity(states.len() * usize::from(node_count));
    let mut alive_bits = Vec::with_capacity(states.len() * BITBOARD_WORDS);
    let mut winner = Vec::with_capacity(states.len());
    let mut terminal_values = Vec::with_capacity(states.len());
    let mut outcome_classes = Vec::with_capacity(states.len());
    let mut score_margins = Vec::with_capacity(states.len());
    for (state, score) in states.iter().zip(&scores) {
        for player in score.players {
            components.extend([
                i32::from(player.peries),
                i32::from(player.quarks),
                i32::from(player.stars),
                i32::from(player.quark_peri),
                i32::from(player.award),
                i32::from(player.total),
            ]);
        }
        components.push(i32::from(score.contested_peries));
        components.push(score.leader.map_or(-1, |player| player as i32));
        node_owner.extend_from_slice(&score.node_owner[..usize::from(node_count)]);
        alive_bits.extend(score.alive_stones.words());
        let leader = if state.is_terminal() {
            Some(
                score
                    .leader
                    .expect("a full Double *Star board must have a decisive winner"),
            )
        } else {
            score.leader
        };
        winner.push(leader.map_or(-1, |player| player as i8));
        let player = state.to_move().index();
        score_margins.push(score.players[player].total - score.players[1 - player].total);
        if state.is_terminal() {
            let value = score
                .outcome_for(state.to_move())
                .expect("a full Double *Star board must have a decisive winner");
            terminal_values.push(value);
            outcome_classes.push(outcome_class(value));
        } else {
            terminal_values.push(0.0);
            outcome_classes.push(u8::MAX);
        }
    }
    PyScoreData {
        batch_size: states.len(),
        node_count,
        components,
        node_owner,
        alive_bits,
        winner,
        terminal_value: terminal_values,
        outcome_class: outcome_classes,
        score_margin: score_margins,
    }
}

fn pack_feature_states(states: &[GameState]) -> PyFeatureData {
    let rows = states
        .par_iter()
        .map_init(ScoringScratch::default, |scratch, state| {
            let score = scratch.score_state(state);
            pack_feature_row(
                state.board(),
                state.stones(),
                state.to_move(),
                state.moves_left(),
                state.is_opening(),
                state.is_terminal(),
                score,
            )
        })
        .collect();
    pack_feature_rows(rows)
}

fn pack_semantic_feature_states(states: &[FeatureState]) -> PyFeatureData {
    let rows = states
        .par_iter()
        .map_init(ScoringScratch::default, |scratch, state| {
            let score = scratch.score(&state.board, state.stones);
            pack_feature_row(
                &state.board,
                state.stones,
                state.to_move,
                state.moves_left,
                state.opening,
                state.terminal,
                score,
            )
        })
        .collect();
    pack_feature_rows(rows)
}

#[allow(clippy::too_many_arguments)]
fn pack_feature_row(
    board: &Board,
    stones: [BitBoard; 2],
    to_move: Player,
    moves_left: u8,
    opening: bool,
    terminal: bool,
    score: ScoreResult,
) -> PackedFeatureRow {
    let node_count = usize::from(board.node_count());
    let max_degree = (0..board.node_count())
        .map(|node| board.neighbors(node).len())
        .max()
        .expect("supported boards are nonempty");
    let current = to_move.index();
    let opponent = 1 - current;
    let mut node_features = Vec::with_capacity(node_count * NODE_FEATURE_DIM);
    let mut legal_nodes = Vec::with_capacity(node_count);
    let mut node_owner = Vec::with_capacity(node_count);
    let mut alive_stones = Vec::with_capacity(node_count);
    for node_index in 0..node_count {
        let node = u16::try_from(node_index).expect("board nodes fit in u16");
        let current_stone = stones[current].contains(node);
        let opponent_stone = stones[opponent].contains(node);
        let empty = !current_stone && !opponent_stone;
        let owner = score.node_owner[node_index];
        let alive = score.alive_stones.contains(node);
        let ring = board.ring(node);
        let position = board.position(node);
        let arm_distance = position.min(ring - position);
        let legal = empty && !terminal;
        node_features.extend([
            binary_feature(empty),
            binary_feature(current_stone),
            binary_feature(opponent_stone),
            binary_feature(owner == current as i8),
            binary_feature(owner == opponent as i8),
            binary_feature(owner == -1),
            binary_feature(alive && current_stone),
            binary_feature(alive && opponent_stone),
            binary_feature(board.is_peri(node)),
            binary_feature(board.is_quark(node)),
            f32::from(ring) / f32::from(board.rings()),
            f32::from(arm_distance) / f32::from(ring),
            board.neighbors(node).len() as f32 / max_degree as f32,
            binary_feature(ring == 1),
            binary_feature(legal),
        ]);
        legal_nodes.push(u8::from(legal));
        node_owner.push(owner);
        alive_stones.push(u8::from(alive));
    }

    let occupied = stones[0].count() + stones[1].count();
    let current_count = stones[current].count();
    let opponent_count = stones[opponent].count();
    let current_score = score.players[current];
    let opponent_score = score.players[opponent];
    let score_scale = f64::from(SCORE_MARGIN_SUPPORT);
    let star_scale = (f64::from(board.peri_count()) / 2.0).max(1.0);
    let global_features = [
        (f64::from(board.rings()) / 10.0) as f32,
        (f64::from(occupied) / f64::from(board.node_count())) as f32,
        (f64::from(current_count) / f64::from(board.node_count())) as f32,
        (f64::from(opponent_count) / f64::from(board.node_count())) as f32,
        (f64::from(moves_left) / 2.0) as f32,
        binary_feature(opening),
        binary_feature(terminal),
        (f64::from(current_score.total) / score_scale) as f32,
        (f64::from(opponent_score.total) / score_scale) as f32,
        (f64::from(current_score.total - opponent_score.total) / score_scale) as f32,
        (f64::from(current_score.peries) / f64::from(board.peri_count())) as f32,
        (f64::from(opponent_score.peries) / f64::from(board.peri_count())) as f32,
        (f64::from(current_score.quarks) / 5.0) as f32,
        (f64::from(opponent_score.quarks) / 5.0) as f32,
        (f64::from(current_score.stars) / star_scale) as f32,
        (f64::from(opponent_score.stars) / star_scale) as f32,
        (f64::from(score.contested_peries) / f64::from(board.peri_count())) as f32,
    ];
    let score_components = packed_score_components(&score);
    PackedFeatureRow {
        rings: board.rings(),
        node_count,
        node_features,
        global_features,
        legal_nodes,
        score_components,
        node_owner,
        alive_stones,
    }
}

fn pack_feature_rows(rows: Vec<PackedFeatureRow>) -> PyFeatureData {
    let batch_size = rows.len();
    let max_nodes = rows.iter().map(|row| row.node_count).max().unwrap_or(0);
    let mut rings = Vec::with_capacity(batch_size);
    let mut node_features = vec![0.0_f32; batch_size * max_nodes * NODE_FEATURE_DIM];
    let mut global_features = Vec::with_capacity(batch_size * GLOBAL_FEATURE_DIM);
    let mut node_mask = vec![0_u8; batch_size * max_nodes];
    let mut legal_action_mask = vec![0_u8; batch_size * max_nodes];
    let mut score_components = Vec::with_capacity(batch_size * SCORE_COMPONENT_DIM);
    let mut node_owner = vec![-1_i8; batch_size * max_nodes];
    let mut alive_stones = vec![0_u8; batch_size * max_nodes];
    for (row_index, row) in rows.into_iter().enumerate() {
        rings.push(row.rings);
        let feature_start = row_index * max_nodes * NODE_FEATURE_DIM;
        let feature_end = feature_start + row.node_count * NODE_FEATURE_DIM;
        node_features[feature_start..feature_end].copy_from_slice(&row.node_features);
        global_features.extend_from_slice(&row.global_features);
        let node_start = row_index * max_nodes;
        let node_end = node_start + row.node_count;
        node_mask[node_start..node_end].fill(1);
        node_owner[node_start..node_end].copy_from_slice(&row.node_owner);
        alive_stones[node_start..node_end].copy_from_slice(&row.alive_stones);
        let action_start = row_index * max_nodes;
        legal_action_mask[action_start..action_start + row.node_count]
            .copy_from_slice(&row.legal_nodes);
        score_components.extend_from_slice(&row.score_components);
    }
    PyFeatureData {
        batch_size,
        max_nodes,
        buffers: Arc::new(FeatureBuffers {
            rings,
            node_features: f32_buffer(&node_features),
            global_features: f32_buffer(&global_features),
            node_mask,
            legal_action_mask,
            score_components: i32_buffer(&score_components),
            node_owner: i8_buffer(&node_owner),
            alive_stones,
        }),
    }
}

const fn binary_feature(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

fn packed_score_components(score: &ScoreResult) -> [i32; SCORE_COMPONENT_DIM] {
    let zero = score.players[0];
    let one = score.players[1];
    [
        i32::from(zero.peries),
        i32::from(zero.quarks),
        i32::from(zero.stars),
        i32::from(zero.quark_peri),
        i32::from(zero.award),
        i32::from(zero.total),
        i32::from(one.peries),
        i32::from(one.quarks),
        i32::from(one.stars),
        i32::from(one.quark_peri),
        i32::from(one.award),
        i32::from(one.total),
        i32::from(score.contested_peries),
        score.leader.map_or(-1, |player| player as i32),
    ]
}

fn f32_buffer(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn i32_buffer(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn i8_buffer(values: &[i8]) -> Vec<u8> {
    values.iter().map(|value| value.to_ne_bytes()[0]).collect()
}

fn outcome_class(value: f32) -> u8 {
    match value {
        1.0 => 1,
        -1.0 => 0,
        _ => panic!("terminal value must be exactly -1 or 1"),
    }
}

fn pack_states(states: &[GameState]) -> PyStateData {
    let rings = states.first().map_or(0, |state| state.board().rings());
    let node_count = states.first().map_or(0, |state| state.board().node_count());
    let rows: Vec<_> = states
        .par_iter()
        .map(|state| {
            let legal = state.legal_actions();
            PackedStateRow {
                zero_bits: state.stones_for(Player::Zero).words(),
                one_bits: state.stones_for(Player::One).words(),
                legal_bits: legal.placements.words(),
                hash: state.hash64(),
                stones_placed: state.stones_placed(),
                to_move: state.to_move() as u8,
                moves_left: state.moves_left(),
                opening: state.is_opening(),
                mid_turn: state.is_mid_turn(),
                terminal: state.is_terminal(),
            }
        })
        .collect();
    let mut zero_bits = Vec::with_capacity(states.len() * BITBOARD_WORDS);
    let mut one_bits = Vec::with_capacity(states.len() * BITBOARD_WORDS);
    let mut legal_bits = Vec::with_capacity(states.len() * BITBOARD_WORDS);
    let mut hashes = Vec::with_capacity(states.len());
    let mut stones_placed = Vec::with_capacity(states.len());
    let mut to_move = Vec::with_capacity(states.len());
    let mut moves_left = Vec::with_capacity(states.len());
    let mut opening = Vec::with_capacity(states.len());
    let mut mid_turn = Vec::with_capacity(states.len());
    let mut terminal = Vec::with_capacity(states.len());
    for row in rows {
        zero_bits.extend(row.zero_bits);
        one_bits.extend(row.one_bits);
        legal_bits.extend(row.legal_bits);
        hashes.push(row.hash);
        stones_placed.push(row.stones_placed);
        to_move.push(row.to_move);
        moves_left.push(row.moves_left);
        opening.push(row.opening);
        mid_turn.push(row.mid_turn);
        terminal.push(row.terminal);
    }
    PyStateData {
        rings,
        node_count,
        batch_size: states.len(),
        zero_bits,
        one_bits,
        legal_bits,
        hashes,
        stones_placed,
        to_move,
        moves_left,
        opening,
        mid_turn,
        terminal,
    }
}

fn pack_trajectory_data(states: &[GameState]) -> PyTrajectoryData {
    let mut last_move = Vec::with_capacity(states.len());
    let mut current_turn_offsets = Vec::with_capacity(states.len() + 1);
    let mut current_turn_moves = Vec::with_capacity(states.len() * 2);
    let mut turn_count = Vec::with_capacity(states.len());
    current_turn_offsets.push(0);
    for state in states {
        last_move.push(state.last_move().map_or(-1, i32::from));
        current_turn_moves.extend_from_slice(state.current_turn_moves());
        current_turn_offsets.push(current_turn_moves.len());
        turn_count.push(state.turn_count());
    }
    PyTrajectoryData {
        batch_size: states.len(),
        last_move,
        current_turn_offsets,
        current_turn_moves,
        turn_count,
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_semantic_states(
    board: Arc<Board>,
    zero_bits: Vec<u64>,
    one_bits: Vec<u64>,
    to_move: Vec<u8>,
    moves_left: Vec<u8>,
    opening: Vec<bool>,
) -> Result<Vec<GameState>, String> {
    let rows = to_move.len();
    if zero_bits.len() != rows * BITBOARD_WORDS
        || one_bits.len() != rows * BITBOARD_WORDS
        || moves_left.len() != rows
        || opening.len() != rows
    {
        return Err(format!("semantic buffers disagree on row count {rows}"));
    }

    (0..rows)
        .into_par_iter()
        .map(|row| {
            let word_start = row * BITBOARD_WORDS;
            let word_end = word_start + BITBOARD_WORDS;
            let mut zero_words = [0_u64; BITBOARD_WORDS];
            let mut one_words = [0_u64; BITBOARD_WORDS];
            zero_words.copy_from_slice(&zero_bits[word_start..word_end]);
            one_words.copy_from_slice(&one_bits[word_start..word_end]);
            let player = match to_move[row] {
                0 => Player::Zero,
                1 => Player::One,
                value => {
                    return Err(format!("row {row} has invalid to_move value {value}"));
                }
            };
            GameState::from_parts(
                Arc::clone(&board),
                [
                    BitBoard::from_words(zero_words),
                    BitBoard::from_words(one_words),
                ],
                player,
                moves_left[row],
                opening[row],
            )
            .map_err(|error| format!("row {row}: {error}"))
        })
        .collect()
}

fn decode_semantic_feature_states(
    rings: &[u8],
    metadata: &[u8],
    stone_values: &[u8],
) -> Result<Vec<FeatureState>, String> {
    let rows = rings.len();
    if rows == 0 {
        return Err("semantic feature batch must contain at least one row".to_owned());
    }
    if metadata.len() != rows * SEMANTIC_METADATA_DIM {
        return Err(format!(
            "semantic metadata must contain {} bytes for {rows} rows",
            rows * SEMANTIC_METADATA_DIM
        ));
    }
    let mut boards: HashMap<u8, Arc<Board>> = HashMap::new();
    let mut states = Vec::with_capacity(rows);
    let mut stone_offset = 0;
    for (row, &ring_count) in rings.iter().enumerate() {
        let board = if let Some(board) = boards.get(&ring_count) {
            Arc::clone(board)
        } else {
            let board = Arc::new(Board::new(ring_count).map_err(|error| {
                format!("row {row} has invalid ring count {ring_count}: {error}")
            })?);
            boards.insert(ring_count, Arc::clone(&board));
            board
        };
        let node_count = usize::from(board.node_count());
        let stone_end = stone_offset + node_count;
        if stone_end > stone_values.len() {
            return Err(format!(
                "semantic stones end inside row {row}: expected {node_count} node values"
            ));
        }
        let mut stones = [BitBoard::empty(); 2];
        for (node_index, &value) in stone_values[stone_offset..stone_end].iter().enumerate() {
            let node = u16::try_from(node_index).expect("board nodes fit in u16");
            match i8::from_ne_bytes([value]) {
                -1 => {}
                0 => {
                    stones[0].insert(node);
                }
                1 => {
                    stones[1].insert(node);
                }
                invalid => {
                    return Err(format!(
                        "row {row} node {node_index} has invalid stone value {invalid}"
                    ));
                }
            }
        }
        stone_offset = stone_end;

        let base = row * SEMANTIC_METADATA_DIM;
        let to_move = match metadata[base] {
            0 => Player::Zero,
            1 => Player::One,
            value => return Err(format!("row {row} has invalid to_move value {value}")),
        };
        let moves_left = metadata[base + 1];
        let opening = semantic_bool("opening", row, metadata[base + 2])?;
        let terminal = semantic_bool("terminal", row, metadata[base + 3])?;
        validate_feature_semantics(row, &board, stones, to_move, moves_left, opening, terminal)?;
        states.push(FeatureState {
            board,
            stones,
            to_move,
            moves_left,
            opening,
            terminal,
        });
    }
    if stone_offset != stone_values.len() {
        return Err(format!(
            "semantic stones contain {} trailing node values",
            stone_values.len() - stone_offset
        ));
    }
    Ok(states)
}

#[allow(clippy::too_many_arguments)]
fn validate_feature_semantics(
    row: usize,
    board: &Board,
    stones: [BitBoard; 2],
    to_move: Player,
    moves_left: u8,
    opening: bool,
    terminal: bool,
) -> Result<(), String> {
    let occupied = stones[0].count() + stones[1].count();
    let board_full = occupied == board.node_count();
    if moves_left > 2 {
        return Err(format!("row {row} has moves_left outside 0..2"));
    }
    if terminal != board_full {
        return Err(format!("row {row} terminal must equal board-full"));
    }
    if moves_left == 0 && !board_full {
        return Err(format!(
            "row {row} uses moves_left == 0 on a non-full board"
        ));
    }
    if board_full && moves_left > 1 {
        return Err(format!(
            "row {row} full board retains more than one placement"
        ));
    }
    if opening && (to_move != Player::Zero || moves_left != 1 || occupied != 0 || terminal) {
        return Err(format!("row {row} has invalid opening metadata"));
    }
    Ok(())
}

fn semantic_bool(name: &str, row: usize, value: u8) -> Result<bool, String> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(format!("row {row} has invalid {name} value {value}")),
    }
}

fn prepare_terminal_resets(
    board: &Arc<Board>,
    states: &[GameState],
    indices: &[usize],
) -> Result<Vec<(usize, GameState)>, String> {
    validate_row_indices(states.len(), indices)?;
    for &index in indices {
        if !states[index].is_terminal() {
            return Err(format!("state row {index} is not terminal"));
        }
    }
    Ok(indices
        .par_iter()
        .map(|&index| (index, GameState::new(Arc::clone(board))))
        .collect())
}

fn prepare_applied_rows(
    states: &[GameState],
    indices: Vec<usize>,
    action_codes: Vec<i32>,
) -> Result<Vec<(usize, GameState)>, String> {
    if indices.len() != action_codes.len() {
        return Err("indices and actions must have equal lengths".to_owned());
    }
    let mut job_by_index = HashMap::with_capacity(indices.len());
    let mut jobs: Vec<(usize, Vec<Action>)> = Vec::new();
    for (index, action_code) in indices.into_iter().zip(action_codes) {
        if index >= states.len() {
            return Err(format!("state index {index} is out of range"));
        }
        let action = Action::from_code(action_code).map_err(|error| error.to_string())?;
        let job = if let Some(&job) = job_by_index.get(&index) {
            job
        } else {
            let job = jobs.len();
            jobs.push((index, Vec::new()));
            job_by_index.insert(index, job);
            job
        };
        jobs[job].1.push(action);
    }
    jobs.into_par_iter()
        .map(|(index, actions)| {
            let mut state = states[index].clone();
            for action in actions {
                state.apply(action).map_err(|error| error.to_string())?;
            }
            Ok((index, state))
        })
        .collect()
}

fn replace_rows(
    states: &[GameState],
    indices: &[usize],
    replacements: Vec<GameState>,
) -> Result<Vec<(usize, GameState)>, String> {
    if replacements.len() != indices.len() {
        return Err(format!(
            "received {} replacement rows for {} indices",
            replacements.len(),
            indices.len()
        ));
    }
    validate_row_indices(states.len(), indices)?;
    Ok(indices.iter().copied().zip(replacements).collect())
}

fn validate_row_indices(row_count: usize, indices: &[usize]) -> Result<(), String> {
    let mut unique = HashSet::with_capacity(indices.len());
    for &index in indices {
        if index >= row_count {
            return Err(format!("state index {index} is out of range"));
        }
        if !unique.insert(index) {
            return Err(format!("state index {index} is duplicated"));
        }
    }
    Ok(())
}

fn active_root_requests(trees: &[SearchTree]) -> Result<Vec<(usize, EvaluationRequest)>, String> {
    let rows: Result<Vec<_>, String> = trees
        .par_iter()
        .enumerate()
        .map(|(index, tree)| {
            if tree.root_terminal_value().is_some() {
                Ok(None)
            } else {
                tree.root_request()
                    .map(|request| Some((index, request)))
                    .map_err(|error| error.to_string())
            }
        })
        .collect();
    Ok(rows?.into_iter().flatten().collect())
}

fn pack_requests(tree_indices: Vec<usize>, requests: Vec<EvaluationRequest>) -> PyEvalBatch {
    let tokens = requests.iter().map(|request| request.token).collect();
    let request_states = requests
        .iter()
        .map(|request| request.state.clone())
        .collect::<Vec<_>>();
    let states = pack_states(&request_states);
    let features = pack_feature_states(&request_states);
    let mut legal_offsets = Vec::with_capacity(requests.len() + 1);
    let mut legal_actions = Vec::new();
    legal_offsets.push(0);
    for request in requests {
        legal_actions.extend(request.legal_actions.into_iter().map(Action::code));
        legal_offsets.push(legal_actions.len());
    }
    PyEvalBatch {
        tree_indices,
        tokens,
        states,
        features,
        legal_offsets,
        legal_actions,
    }
}

fn unpack_evaluations(
    tokens: Vec<u64>,
    values: Vec<f32>,
    offsets: Vec<usize>,
    logits: Vec<f32>,
) -> PyResult<Vec<Evaluation>> {
    if tokens.len() != values.len() {
        return Err(PyValueError::new_err(
            "tokens and values must have equal lengths",
        ));
    }
    if offsets.len() != tokens.len() + 1
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&logits.len())
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(PyValueError::new_err(
            "policy_offsets must be monotonic CSR offsets covering policy_logits",
        ));
    }
    Ok(tokens
        .into_iter()
        .zip(values)
        .enumerate()
        .map(|(row, (token, value))| Evaluation {
            token,
            value,
            policy_logits: logits[offsets[row]..offsets[row + 1]].to_vec(),
        })
        .collect())
}

fn match_evaluations(
    requests: &[EvaluationRequest],
    responses: Vec<Evaluation>,
) -> PyResult<HashMap<u64, Evaluation>> {
    match_token_set(
        &requests
            .iter()
            .map(|request| request.token)
            .collect::<Vec<_>>(),
        responses,
    )
}

fn match_token_set(
    expected_tokens: &[u64],
    responses: Vec<Evaluation>,
) -> PyResult<HashMap<u64, Evaluation>> {
    let expected: HashSet<_> = expected_tokens.iter().copied().collect();
    if expected.len() != expected_tokens.len() {
        return Err(PyRuntimeError::new_err(
            "internal request tokens are not unique",
        ));
    }
    let mut matched = HashMap::with_capacity(responses.len());
    for response in responses {
        if !expected.contains(&response.token) {
            return Err(PyValueError::new_err(format!(
                "unknown evaluation token {}",
                response.token
            )));
        }
        let token = response.token;
        if matched.insert(token, response).is_some() {
            return Err(PyValueError::new_err(format!(
                "duplicate evaluation token {token}"
            )));
        }
    }
    if let Some(missing) = expected_tokens
        .iter()
        .find(|token| !matched.contains_key(token))
    {
        return Err(PyValueError::new_err(format!(
            "missing evaluation token {missing}"
        )));
    }
    Ok(matched)
}

fn validate_evaluation_native(evaluation: &Evaluation, expected: usize) -> Result<(), String> {
    if evaluation.policy_logits.len() != expected {
        return Err(format!(
            "expected {expected} policy logits, got {}",
            evaluation.policy_logits.len()
        ));
    }
    if !evaluation.value.is_finite()
        || evaluation
            .policy_logits
            .iter()
            .any(|logit| !logit.is_finite())
    {
        return Err("value and policy logits must be finite".to_owned());
    }
    if !(-1.0..=1.0).contains(&evaluation.value) {
        return Err("value must be in [-1, 1]".to_owned());
    }
    Ok(())
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

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Encodes a heterogeneous semantic-key batch from three contiguous byte buffers.
///
/// `rings` is `uint8[B]`; `metadata` is `uint8[B, 4]` in
/// `(to_move, moves_left, opening, terminal)` order; and `stones`
/// is the concatenation of each row's native node-order `int8` stones.
#[pyfunction]
fn encode_semantic_features(
    py: Python<'_>,
    rings: &Bound<'_, PyBytes>,
    metadata: &Bound<'_, PyBytes>,
    stones: &Bound<'_, PyBytes>,
) -> PyResult<PyFeatureData> {
    let rings = rings.as_bytes().to_vec();
    let metadata = metadata.as_bytes().to_vec();
    let stones = stones.as_bytes().to_vec();
    py.detach(move || {
        let states = decode_semantic_feature_states(&rings, &metadata, &stones)
            .map_err(PyValueError::new_err)?;
        Ok(pack_semantic_feature_states(&states))
    })
}

#[pyfunction]
fn native_rules_hash() -> u64 {
    rules_hash()
}

#[pyfunction]
fn native_rules_hash_tag() -> &'static str {
    RULES_HASH
}

#[pyfunction]
fn native_rules_schema() -> &'static str {
    RULES_SCHEMA
}

/// Configures the process-wide actor pool before its first use.
#[pyfunction]
fn configure_rayon_threads(threads: usize) -> PyResult<()> {
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be positive"));
    }
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("star-actor-{index}"))
        .build_global()
        .map_err(|error| {
            PyRuntimeError::new_err(format!(
                "Rayon pool is already initialized or unavailable: {error}"
            ))
        })
}

#[pyfunction]
fn rayon_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Native extension module.
#[pymodule]
fn star_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStateData>()?;
    module.add_class::<PyTrajectoryData>()?;
    module.add_class::<PyScoreData>()?;
    module.add_class::<PyFeatureData>()?;
    module.add_class::<PyStateBatch>()?;
    module.add_class::<PyEvalBatch>()?;
    module.add_class::<PySearchResults>()?;
    module.add_class::<PySearchBatch>()?;
    module.add_function(wrap_pyfunction!(native_rules_hash, module)?)?;
    module.add_function(wrap_pyfunction!(native_rules_hash_tag, module)?)?;
    module.add_function(wrap_pyfunction!(native_rules_schema, module)?)?;
    module.add_function(wrap_pyfunction!(encode_semantic_features, module)?)?;
    module.add_function(wrap_pyfunction!(configure_rayon_threads, module)?)?;
    module.add_function(wrap_pyfunction!(rayon_num_threads, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct ParallelSnapshot {
        hashes: Vec<u64>,
        legal_bits: Vec<u64>,
        score_components: Vec<i32>,
        node_owner: Vec<i8>,
        alive_bits: Vec<u64>,
        root_rows: Vec<usize>,
        selected_actions: Vec<i32>,
        action_offsets: Vec<usize>,
        actions: Vec<i32>,
        visits: Vec<u32>,
        q_values: Vec<f32>,
        policy_target: Vec<f32>,
    }

    #[test]
    fn flat_response_buffers_preserve_tokens_and_rows() {
        let rows = unpack_evaluations(
            vec![9, 3],
            vec![0.25, -0.5],
            vec![0, 2, 5],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        assert_eq!(rows[0].token, 9);
        assert_eq!(rows[0].policy_logits, [1.0, 2.0]);
        assert_eq!(rows[1].token, 3);
        assert_eq!(rows[1].policy_logits, [3.0, 4.0, 5.0]);
    }

    #[test]
    fn selected_terminal_rows_reset_independently_and_transactionally() {
        let board = Arc::new(Board::new(4).unwrap());
        let terminal = full_state(Arc::clone(&board));
        let mut active = GameState::new(Arc::clone(&board));
        active.apply(Action::Place(0)).unwrap();
        let full = full_state(Arc::clone(&board));
        let states = vec![terminal, active, full];
        let active_key = states[1].key();
        let full_key = states[2].key();

        let mut reset = states.clone();
        commit_rows(
            &mut reset,
            prepare_terminal_resets(&board, &states, &[0]).unwrap(),
        );
        assert!(reset[0].is_opening());
        assert!(!reset[0].is_terminal());
        assert_eq!(reset[1].key(), active_key);
        assert_eq!(reset[2].key(), full_key);

        assert!(prepare_terminal_resets(&board, &states, &[1]).is_err());
        assert!(prepare_terminal_resets(&board, &states, &[0, 0]).is_err());
        assert_eq!(states[1].key(), active_key);
    }

    #[test]
    fn semantic_import_round_trips_terminal_residual_states() {
        for rings in [4, 6] {
            let board = Arc::new(Board::new(rings).unwrap());
            let full = full_state(Arc::clone(&board));
            let imported = semantic_round_trip(&full);
            assert_eq!(imported.key(), full.key());
            assert_eq!(imported.hash64(), full.hash64());
            assert_eq!(imported.moves_left(), if rings == 4 { 1 } else { 0 });
            assert!(imported.is_terminal());
            assert_eq!(
                ScoringScratch::default().score_state(&imported).players,
                ScoringScratch::default().score_state(&full).players
            );
        }
    }

    #[test]
    fn imported_and_reset_rows_are_immediately_searchable() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut imported_source = GameState::new(Arc::clone(&board));
        imported_source.apply(Action::Place(7)).unwrap();
        imported_source.apply(Action::Place(8)).unwrap();
        let packed = pack_states(&[imported_source]);
        let replacements = decode_semantic_states(
            Arc::clone(&board),
            packed.zero_bits,
            packed.one_bits,
            packed.to_move,
            packed.moves_left,
            packed.opening,
        )
        .unwrap();

        let terminal = full_state(Arc::clone(&board));
        let states = vec![terminal, GameState::new(Arc::clone(&board))];
        let mut replaced = states.clone();
        commit_rows(
            &mut replaced,
            replace_rows(&states, &[1], replacements).unwrap(),
        );
        let imported_request = SearchTree::new(replaced[1].clone()).root_request().unwrap();
        assert!(!imported_request.legal_actions.is_empty());

        let mut reset = replaced.clone();
        commit_rows(
            &mut reset,
            prepare_terminal_resets(&board, &replaced, &[0]).unwrap(),
        );
        let reset_request = SearchTree::new(reset[0].clone()).root_request().unwrap();
        assert_eq!(
            reset_request.legal_actions.len(),
            usize::from(board.node_count())
        );
    }

    #[test]
    fn semantic_import_rejects_malformed_rows() {
        let board = Arc::new(Board::new(4).unwrap());
        let empty_words = vec![0_u64; BITBOARD_WORDS];
        assert!(
            decode_semantic_states(
                Arc::clone(&board),
                empty_words.clone(),
                empty_words.clone(),
                vec![2],
                vec![1],
                vec![true],
            )
            .is_err()
        );
        assert!(
            decode_semantic_states(
                Arc::clone(&board),
                vec![1, 0, 0, 0, 0],
                vec![1, 0, 0, 0, 0],
                vec![0],
                vec![2],
                vec![false],
            )
            .is_err()
        );
        assert!(
            decode_semantic_states(
                board,
                empty_words.clone(),
                empty_words,
                vec![1],
                vec![0],
                vec![false],
            )
            .is_err()
        );
    }

    #[test]
    fn feature_v3_uses_four_field_semantics_and_nodes_only_masks() {
        let rings = [4_u8, 10];
        let metadata = [0_u8, 1, 1, 0, 0, 1, 1, 0];
        let stones = vec![u8::MAX; 50 + 275];
        let states = decode_semantic_feature_states(&rings, &metadata, &stones).unwrap();
        let features = pack_semantic_feature_states(&states);

        assert_eq!(features.batch_size, 2);
        assert_eq!(features.max_nodes, 275);
        assert_eq!(NODE_FEATURE_DIM, 15);
        assert_eq!(GLOBAL_FEATURE_DIM, 17);
        assert_eq!(FEATURE_SCHEMA_VERSION, 3);
        assert_eq!(FEATURE_SCHEMA_HASH, 0x6b5b_00f6_38e9_c16b);
        assert_eq!(features.buffers.legal_action_mask.len(), 2 * 275);
        assert!(
            features.buffers.legal_action_mask[..50]
                .iter()
                .all(|bit| *bit == 1)
        );
        assert!(
            features.buffers.legal_action_mask[50..275]
                .iter()
                .all(|bit| *bit == 0)
        );
        assert!(
            features.buffers.legal_action_mask[275..]
                .iter()
                .all(|bit| *bit == 1)
        );

        let globals: Vec<_> = features
            .buffers
            .global_features
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(globals.len(), 2 * GLOBAL_FEATURE_DIM);
        assert!((globals[0] - 0.4).abs() <= f32::EPSILON);
        assert_eq!(globals[4], 0.5);
        assert_eq!(globals[5], 1.0);
        assert_eq!(globals[6], 0.0);
        assert_eq!(globals[GLOBAL_FEATURE_DIM], 1.0);
    }

    #[test]
    fn static_ties_remain_optional_but_action_minus_one_is_rejected() {
        let board = Arc::new(Board::new(4).unwrap());
        let state = GameState::new(board);
        let score = ScoringScratch::default().score_state(&state);
        assert_eq!(score.leader, None);
        assert_eq!(score.outcome_for(Player::Zero), None);
        let score_data = score_states(std::slice::from_ref(&state), state.board().node_count());
        assert_eq!(score_data.winner, [-1]);
        assert_eq!(score_data.terminal_value, [0.0]);
        assert_eq!(score_data.outcome_class, [u8::MAX]);
        assert!(prepare_applied_rows(&[state], vec![0], vec![-1]).is_err());
    }

    #[test]
    fn trajectory_metadata_exposes_terminal_value_and_residuals() {
        let board = Arc::new(Board::new(4).unwrap());
        let full = full_state(board);
        let state_data = pack_states(std::slice::from_ref(&full));
        assert_eq!(state_data.stones_placed, [50]);
        assert_eq!(state_data.moves_left, [1]);
        assert_eq!(state_data.mid_turn, [true]);
        let trajectory_data = pack_trajectory_data(std::slice::from_ref(&full));
        assert_eq!(trajectory_data.current_turn_offsets, [0, 1]);
        assert_eq!(trajectory_data.current_turn_moves.len(), 1);
        assert_eq!(trajectory_data.turn_count, [25]);

        let score_data = score_states(&[full], 50);
        let value = score_data.terminal_value[0];
        assert!(value == -1.0 || value == 1.0);
        assert_eq!(score_data.outcome_class, [if value == 1.0 { 1 } else { 0 }]);
        assert!(matches!(score_data.winner.as_slice(), [0 | 1]));
        assert_ne!(score_data.score_margin[0], 0);
        assert_ne!(score_data.score_margin[0] % 2, 0);
    }

    #[test]
    fn one_and_many_threads_produce_identical_actor_results() {
        Python::initialize();
        let single = parallel_snapshot(1);
        let many = parallel_snapshot(4);
        assert_eq!(single, many);
    }

    fn semantic_round_trip(state: &GameState) -> GameState {
        let packed = pack_states(std::slice::from_ref(state));
        decode_semantic_states(
            state.shared_board(),
            packed.zero_bits,
            packed.one_bits,
            packed.to_move,
            packed.moves_left,
            packed.opening,
        )
        .unwrap()
        .pop()
        .unwrap()
    }

    fn full_state(board: Arc<Board>) -> GameState {
        let mut state = GameState::new(board);
        for node in 0..state.board().node_count() {
            state.apply(Action::Place(node)).unwrap();
        }
        state
    }

    fn parallel_snapshot(threads: usize) -> ParallelSnapshot {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            Python::attach(|py| {
                let board = Arc::new(Board::new(4).unwrap());
                let mut batch = PyStateBatch {
                    board: Arc::clone(&board),
                    states: (0..32)
                        .map(|_| GameState::new(Arc::clone(&board)))
                        .collect(),
                };
                let row_indices: Vec<_> = (0..batch.states.len()).collect();
                let opening_actions: Vec<_> = row_indices
                    .iter()
                    .map(|index| (index % usize::from(board.node_count())) as i32)
                    .collect();
                batch
                    .apply_many(py, row_indices.clone(), opening_actions)
                    .unwrap();
                let second_actions: Vec<_> = row_indices
                    .iter()
                    .map(|index| ((index + 1) % usize::from(board.node_count())) as i32)
                    .collect();
                batch.apply_many(py, row_indices, second_actions).unwrap();

                let transformed = batch.transformed(py, 7).unwrap();
                let state_data = transformed.data(py);
                let score_data = transformed.score_data(py);

                let mut search = PySearchBatch {
                    trees: transformed
                        .states
                        .iter()
                        .cloned()
                        .map(SearchTree::new)
                        .collect(),
                    schedulers: None,
                    config: RootSearchConfig::deterministic(9, 4, GumbelParameters::PAPER, 0x5eed),
                    pending: Vec::new(),
                };
                let roots = search.root_requests(py).unwrap();
                let root_rows = roots.tree_indices.clone();
                submit_uniform_roots(&mut search, py, roots);
                while !search.is_done() {
                    let leaves = search.next_requests(py).unwrap();
                    if !leaves.tokens.is_empty() {
                        submit_uniform_leaves(&mut search, py, leaves);
                    }
                }
                let results = search.results(py).unwrap();
                ParallelSnapshot {
                    hashes: state_data.hashes,
                    legal_bits: state_data.legal_bits,
                    score_components: score_data.components,
                    node_owner: score_data.node_owner,
                    alive_bits: score_data.alive_bits,
                    root_rows,
                    selected_actions: results.selected_actions,
                    action_offsets: results.action_offsets,
                    actions: results.actions,
                    visits: results.visits,
                    q_values: results.q_values,
                    policy_target: results.policy_target,
                }
            })
        })
    }

    fn submit_uniform_roots(search: &mut PySearchBatch, py: Python<'_>, requests: PyEvalBatch) {
        let values = vec![0.0; requests.tokens.len()];
        let logits = vec![0.0; requests.legal_actions.len()];
        search
            .initialize_roots(py, requests.tokens, values, requests.legal_offsets, logits)
            .unwrap();
    }

    fn submit_uniform_leaves(search: &mut PySearchBatch, py: Python<'_>, requests: PyEvalBatch) {
        let values = vec![0.0; requests.tokens.len()];
        let logits = vec![0.0; requests.legal_actions.len()];
        search
            .submit(py, requests.tokens, values, requests.legal_offsets, logits)
            .unwrap();
    }

    fn commit_rows(states: &mut [GameState], replacements: Vec<(usize, GameState)>) {
        for (index, replacement) in replacements {
            states[index] = replacement;
        }
    }
}
