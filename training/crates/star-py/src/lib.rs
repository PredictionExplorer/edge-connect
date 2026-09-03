//! Coarse PyO3 interfaces for high-throughput actor and inference loops.
//!
//! Search inference uses flat token-addressed batches. CPU-heavy state,
//! scoring, and search work detaches from the Python interpreter and shares
//! one Rayon pool. Set `RAYON_NUM_THREADS` per actor process, or call
//! `configure_rayon_threads` before creating the first batch. No method creates
//! a nested pool and no single search tree is parallelized.
//!
//! Every batch row carries its own rule variant (mode, handicap, pie) and the
//! retained placement history. Feature schema v4 is the production encoding;
//! the previous lineage's schema v3 remains available for teacher inference
//! during lineage transfer and cross-schema arenas.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyBytesMethods};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use star_engine::{
    Action, BITBOARD_WORDS, BitBoard, Board, D5Maps, GameState, MAX_HANDICAP, Mode, Player,
    RULES_HASH, RULES_SCHEMA, ScoreResult, ScoringScratch, StateParts, Symmetry, Variant,
    rules_hash, score_completion_bounds,
};
use star_search::{
    Evaluation, EvaluationRequest, GumbelParameters, GumbelSequentialHalving, RootSearchConfig,
    SearchTree, SimulationStart,
};

/// Schema v4 node planes.
const NODE_FEATURE_DIM: usize = 19;
/// Schema v4 global scalars.
const GLOBAL_FEATURE_DIM: usize = 25;
/// Schema v3 node planes (previous lineage).
const LEGACY_NODE_FEATURE_DIM: usize = 15;
/// Schema v3 global scalars (previous lineage).
const LEGACY_GLOBAL_FEATURE_DIM: usize = 17;
const SCORE_COMPONENT_DIM: usize = 14;
/// Schema v4 semantic metadata columns:
/// `(to_move, moves_left, opening, terminal, mode, handicap, pie,
/// swap_available, swapped, history_known, pda)`.
const SEMANTIC_METADATA_DIM: usize = 11;
/// Schema v3 semantic metadata columns: `(to_move, moves_left, opening, terminal)`.
const LEGACY_SEMANTIC_METADATA_DIM: usize = 4;
const FEATURE_SCHEMA_VERSION: u8 = 4;
const FEATURE_SCHEMA_HASH: u64 = 0xcb0e_1e89_a6ce_3540;
const LEGACY_FEATURE_SCHEMA_VERSION: u8 = 3;
const LEGACY_FEATURE_SCHEMA_HASH: u64 = 0x6b5b_00f6_38e9_c16b;
const SCORE_MARGIN_SUPPORT: i16 = 151;
/// Largest playout-doubling advantage magnitude accepted as a network input.
const MAX_PLAYOUT_DOUBLING_ADVANTAGE: i8 = 3;

/// Bit flags of the per-node `history_flags` byte.
const HISTORY_CURRENT_TURN: u8 = 1;
const HISTORY_OWN_PREVIOUS_TURN: u8 = 2;
const HISTORY_OPPONENT_PREVIOUS_TURN: u8 = 4;
const HISTORY_HANDICAP_STONE: u8 = 8;

/// Evaluation-time context that is not part of the game state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct FeatureContext {
    /// Playout-doubling advantage of the side to move, in `-3..=3`.
    pda: i8,
    /// Whether the retained placement history is known for this row.
    history_known: bool,
}

impl FeatureContext {
    const fn known(pda: i8) -> Self {
        Self {
            pda,
            history_known: true,
        }
    }
}

#[pyclass(name = "StateData", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyStateData {
    rings: u8,
    node_count: u16,
    batch_size: usize,
    zero_bits: Vec<u64>,
    one_bits: Vec<u64>,
    legal_bits: Vec<u64>,
    current_turn_bits: Vec<u64>,
    previous_turn_bits: Vec<u64>,
    own_previous_turn_bits: Vec<u64>,
    handicap_bits: Vec<u64>,
    hashes: Vec<u64>,
    stones_placed: Vec<u16>,
    to_move: Vec<u8>,
    moves_left: Vec<u8>,
    opening: Vec<bool>,
    mid_turn: Vec<bool>,
    terminal: Vec<bool>,
    mode: Vec<u8>,
    handicap: Vec<u8>,
    pie: Vec<bool>,
    pie_pending: Vec<bool>,
    swap_available: Vec<bool>,
    swapped: Vec<bool>,
    turn_size: Vec<u8>,
    current_turn_total: Vec<u8>,
    turn_count: Vec<u32>,
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

    /// Placements of the unfinished current turn, five words per row.
    #[getter]
    fn current_turn_bits(&self) -> Vec<u64> {
        self.current_turn_bits.clone()
    }

    /// The most recently completed turn (the opponent's), five words per row.
    #[getter]
    fn previous_turn_bits(&self) -> Vec<u64> {
        self.previous_turn_bits.clone()
    }

    /// The completed turn before that (the mover's), five words per row.
    #[getter]
    fn own_previous_turn_bits(&self) -> Vec<u64> {
        self.own_previous_turn_bits.clone()
    }

    /// Stones placed during the opening phase, five words per row.
    #[getter]
    fn handicap_bits(&self) -> Vec<u64> {
        self.handicap_bits.clone()
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

    /// Turn protocol per row: `0` classic, `1` double.
    #[getter]
    fn mode(&self) -> Vec<u8> {
        self.mode.clone()
    }

    #[getter]
    fn handicap(&self) -> Vec<u8> {
        self.handicap.clone()
    }

    #[getter]
    fn pie(&self) -> Vec<bool> {
        self.pie.clone()
    }

    #[getter]
    fn pie_pending(&self) -> Vec<bool> {
        self.pie_pending.clone()
    }

    #[getter]
    fn swap_available(&self) -> Vec<bool> {
        self.swap_available.clone()
    }

    #[getter]
    fn swapped(&self) -> Vec<bool> {
        self.swapped.clone()
    }

    #[getter]
    fn turn_size(&self) -> Vec<u8> {
        self.turn_size.clone()
    }

    /// Placements of the turn in progress: handicap during the opening.
    #[getter]
    fn current_turn_total(&self) -> Vec<u8> {
        self.current_turn_total.clone()
    }

    #[getter]
    fn turn_count(&self) -> Vec<u32> {
        self.turn_count.clone()
    }

    /// Rebuilds the model features without Python unpacking.
    ///
    /// `schema_version` selects the production v4 encoding or the previous
    /// lineage's v3 encoding. `pda` supplies one playout-doubling advantage
    /// per row for the side to move (v4 only).
    #[pyo3(signature = (schema_version=FEATURE_SCHEMA_VERSION, pda=None, history_known=true))]
    fn feature_data(
        &self,
        py: Python<'_>,
        schema_version: u8,
        pda: Option<Vec<i8>>,
        history_known: bool,
    ) -> PyResult<PyFeatureData> {
        let board = Arc::new(Board::new(self.rings).map_err(value_error)?);
        let rows = self.semantic_rows();
        let contexts = feature_contexts(self.batch_size, pda, history_known)?;
        py.detach(move || {
            let states = decode_semantic_states(board, rows).map_err(PyValueError::new_err)?;
            pack_feature_states(&states, &contexts, schema_version).map_err(PyValueError::new_err)
        })
    }
}

impl PyStateData {
    fn semantic_rows(&self) -> SemanticRows {
        SemanticRows {
            zero_bits: self.zero_bits.clone(),
            one_bits: self.one_bits.clone(),
            to_move: self.to_move.clone(),
            moves_left: self.moves_left.clone(),
            opening: self.opening.clone(),
            mode: Some(self.mode.clone()),
            handicap: Some(self.handicap.clone()),
            pie: Some(self.pie.clone()),
            swap_available: Some(self.swap_available.clone()),
            swapped: Some(self.swapped.clone()),
            current_turn_bits: Some(self.current_turn_bits.clone()),
            previous_turn_bits: Some(self.previous_turn_bits.clone()),
            own_previous_turn_bits: Some(self.own_previous_turn_bits.clone()),
            handicap_bits: Some(self.handicap_bits.clone()),
        }
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

/// Rows finalized from a proven winner to the loser-filled proof board.
#[pyclass(name = "ClinchData", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyClinchData {
    batch_size: usize,
    clinched: Vec<bool>,
    winner: Vec<i8>,
    empty_nodes: Vec<u16>,
    last_move: Vec<i32>,
    turn_count: Vec<u32>,
}

#[pymethods]
impl PyClinchData {
    #[getter]
    const fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[getter]
    fn clinched(&self) -> Vec<bool> {
        self.clinched.clone()
    }

    /// Guaranteed winner (`-1`, `0`, or `1`) for each row.
    #[getter]
    fn winner(&self) -> Vec<i8> {
        self.winner.clone()
    }

    /// Number of source-position empties assigned to the loser.
    #[getter]
    fn empty_nodes(&self) -> Vec<u16> {
        self.empty_nodes.clone()
    }

    /// Last real self-play move before synthetic completion.
    #[getter]
    fn last_move(&self) -> Vec<i32> {
        self.last_move.clone()
    }

    /// Real self-play turn count before synthetic completion.
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

/// Contiguous model features plus the exact native score annotations.
#[pyclass(name = "FeatureData", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyFeatureData {
    batch_size: usize,
    max_nodes: usize,
    schema_version: u8,
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
        node_feature_dim(self.schema_version)
    }

    #[getter]
    const fn global_feature_dim(&self) -> usize {
        global_feature_dim(self.schema_version)
    }

    #[getter]
    const fn score_component_dim(&self) -> usize {
        SCORE_COMPONENT_DIM
    }

    #[getter]
    const fn feature_schema_version(&self) -> u8 {
        self.schema_version
    }

    #[getter]
    const fn feature_schema_hash(&self) -> u64 {
        feature_schema_hash(self.schema_version)
    }

    /// Native-endian `u8[batch_size]`.
    #[getter]
    fn rings<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.rings)
    }

    /// Native-endian `float32[batch_size, max_nodes, node_feature_dim]`.
    #[getter]
    fn node_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.buffers.node_features)
    }

    /// Native-endian `float32[batch_size, global_feature_dim]`.
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

const fn node_feature_dim(schema_version: u8) -> usize {
    if schema_version == LEGACY_FEATURE_SCHEMA_VERSION {
        LEGACY_NODE_FEATURE_DIM
    } else {
        NODE_FEATURE_DIM
    }
}

const fn global_feature_dim(schema_version: u8) -> usize {
    if schema_version == LEGACY_FEATURE_SCHEMA_VERSION {
        LEGACY_GLOBAL_FEATURE_DIM
    } else {
        GLOBAL_FEATURE_DIM
    }
}

const fn feature_schema_hash(schema_version: u8) -> u64 {
    if schema_version == LEGACY_FEATURE_SCHEMA_VERSION {
        LEGACY_FEATURE_SCHEMA_HASH
    } else {
        FEATURE_SCHEMA_HASH
    }
}

fn validate_schema_version(schema_version: u8) -> Result<(), String> {
    if schema_version == FEATURE_SCHEMA_VERSION || schema_version == LEGACY_FEATURE_SCHEMA_VERSION {
        Ok(())
    } else {
        Err(format!(
            "feature schema_version must be {FEATURE_SCHEMA_VERSION} or \
             {LEGACY_FEATURE_SCHEMA_VERSION}, got {schema_version}"
        ))
    }
}

struct PackedFeatureRow {
    rings: u8,
    node_count: usize,
    node_features: Vec<f32>,
    global_features: Vec<f32>,
    legal_nodes: Vec<u8>,
    score_components: [i32; SCORE_COMPONENT_DIM],
    node_owner: Vec<i8>,
    alive_stones: Vec<u8>,
}

struct PreparedClinchRow {
    index: usize,
    replacement: GameState,
    winner: Player,
    empty_nodes: u16,
    last_move: i32,
    turn_count: u32,
}

/// Packed semantic rows shared by every importer.
struct SemanticRows {
    zero_bits: Vec<u64>,
    one_bits: Vec<u64>,
    to_move: Vec<u8>,
    moves_left: Vec<u8>,
    opening: Vec<bool>,
    mode: Option<Vec<u8>>,
    handicap: Option<Vec<u8>>,
    pie: Option<Vec<bool>>,
    swap_available: Option<Vec<bool>>,
    swapped: Option<Vec<bool>>,
    current_turn_bits: Option<Vec<u64>>,
    previous_turn_bits: Option<Vec<u64>>,
    own_previous_turn_bits: Option<Vec<u64>>,
    handicap_bits: Option<Vec<u64>>,
}

impl SemanticRows {
    fn row_count(&self) -> usize {
        self.to_move.len()
    }
}

fn parse_variant(mode: u8, handicap: u8, pie: bool) -> Result<Variant, String> {
    let mode = Mode::from_index(mode).ok_or_else(|| format!("invalid mode index {mode}"))?;
    Variant::new(mode, handicap, pie).map_err(|error| error.to_string())
}

fn variants_from_rows(
    rows: usize,
    mode: Option<Vec<u8>>,
    handicap: Option<Vec<u8>>,
    pie: Option<Vec<bool>>,
) -> Result<Vec<Variant>, String> {
    let mode = mode.unwrap_or_else(|| vec![Mode::Double.index(); rows]);
    let handicap = handicap.unwrap_or_else(|| vec![1; rows]);
    let pie = pie.unwrap_or_else(|| vec![false; rows]);
    if mode.len() != rows || handicap.len() != rows || pie.len() != rows {
        return Err(format!("variant buffers disagree on row count {rows}"));
    }
    (0..rows)
        .map(|row| {
            parse_variant(mode[row], handicap[row], pie[row])
                .map_err(|error| format!("row {row}: {error}"))
        })
        .collect()
}

/// Mutable homogeneous-board environment batch; every row may hold a variant.
#[pyclass(name = "StateBatch")]
struct PyStateBatch {
    board: Arc<Board>,
    states: Vec<GameState>,
}

#[pymethods]
impl PyStateBatch {
    /// Creates `batch_size` empty games of one variant.
    #[new]
    #[pyo3(signature = (rings, batch_size, mode="double", handicap=1, pie=false))]
    fn new(rings: u8, batch_size: usize, mode: &str, handicap: u8, pie: bool) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive"));
        }
        let board = Arc::new(Board::new(rings).map_err(value_error)?);
        let mode = Mode::parse(mode)
            .ok_or_else(|| PyValueError::new_err("mode must be 'classic' or 'double'"))?;
        let variant = Variant::new(mode, handicap, pie).map_err(value_error)?;
        let states = (0..batch_size)
            .into_par_iter()
            .map(|_| GameState::with_variant(Arc::clone(&board), variant))
            .collect();
        Ok(Self { board, states })
    }

    /// Constructs a batch from packed semantic state fields.
    ///
    /// Variant and history buffers are optional; omitted variants default to
    /// the standard game and omitted history to unknown (empty) sets.
    #[staticmethod]
    #[pyo3(signature = (
        rings,
        zero_bits,
        one_bits,
        to_move,
        moves_left,
        opening,
        mode=None,
        handicap=None,
        pie=None,
        swap_available=None,
        swapped=None,
        current_turn_bits=None,
        previous_turn_bits=None,
        own_previous_turn_bits=None,
        handicap_bits=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_semantic(
        py: Python<'_>,
        rings: u8,
        zero_bits: Vec<u64>,
        one_bits: Vec<u64>,
        to_move: Vec<u8>,
        moves_left: Vec<u8>,
        opening: Vec<bool>,
        mode: Option<Vec<u8>>,
        handicap: Option<Vec<u8>>,
        pie: Option<Vec<bool>>,
        swap_available: Option<Vec<bool>>,
        swapped: Option<Vec<bool>>,
        current_turn_bits: Option<Vec<u64>>,
        previous_turn_bits: Option<Vec<u64>>,
        own_previous_turn_bits: Option<Vec<u64>>,
        handicap_bits: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        let board = Arc::new(Board::new(rings).map_err(value_error)?);
        let shared = Arc::clone(&board);
        let rows = SemanticRows {
            zero_bits,
            one_bits,
            to_move,
            moves_left,
            opening,
            mode,
            handicap,
            pie,
            swap_available,
            swapped,
            current_turn_bits,
            previous_turn_bits,
            own_previous_turn_bits,
            handicap_bits,
        };
        let states = py
            .detach(move || decode_semantic_states(shared, rows))
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

    /// Resets all rows to the empty opening state of their own variant.
    fn reset(&mut self, py: Python<'_>) {
        py.detach(|| {
            self.states.par_iter_mut().for_each(|state| {
                *state = GameState::with_variant(Arc::clone(&self.board), state.variant());
            });
        });
    }

    /// Resets only terminal rows, atomically, keeping each row's variant.
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
    ///
    /// Placements use their node id; the pie swap uses `node_count`.
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
    #[pyo3(signature = (
        indices,
        zero_bits,
        one_bits,
        to_move,
        moves_left,
        opening,
        mode=None,
        handicap=None,
        pie=None,
        swap_available=None,
        swapped=None,
        current_turn_bits=None,
        previous_turn_bits=None,
        own_previous_turn_bits=None,
        handicap_bits=None
    ))]
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
        mode: Option<Vec<u8>>,
        handicap: Option<Vec<u8>>,
        pie: Option<Vec<bool>>,
        swap_available: Option<Vec<bool>>,
        swapped: Option<Vec<bool>>,
        current_turn_bits: Option<Vec<u64>>,
        previous_turn_bits: Option<Vec<u64>>,
        own_previous_turn_bits: Option<Vec<u64>>,
        handicap_bits: Option<Vec<u64>>,
    ) -> PyResult<()> {
        let board = Arc::clone(&self.board);
        let rows = SemanticRows {
            zero_bits,
            one_bits,
            to_move,
            moves_left,
            opening,
            mode,
            handicap,
            pie,
            swap_available,
            swapped,
            current_turn_bits,
            previous_turn_bits,
            own_previous_turn_bits,
            handicap_bits,
        };
        let replacements = py
            .detach(move || decode_semantic_states(board, rows))
            .map_err(PyValueError::new_err)?;
        let replacements = py
            .detach(|| replace_rows(&self.states, &indices, replacements))
            .map_err(PyValueError::new_err)?;
        for (index, replacement) in replacements {
            self.states[index] = replacement;
        }
        Ok(())
    }

    /// Finalizes proven live rows to the full proof board that favors the loser.
    fn complete_clinches(&mut self, py: Python<'_>) -> PyResult<PyClinchData> {
        let prepared: Result<Vec<_>, String> = py.detach(|| {
            self.states
                .par_iter()
                .enumerate()
                .map(|(index, state)| {
                    if state.is_terminal() {
                        return Ok(None);
                    }
                    let bounds = score_completion_bounds(state.board(), state.stones());
                    let Some(winner) = bounds.guaranteed_winner else {
                        return Ok(None);
                    };
                    let scenario = bounds
                        .loser_filled_scenario()
                        .expect("a guaranteed winner has a loser-filled scenario");
                    debug_assert_eq!(scenario.score.leader, Some(winner));
                    // The synthetic proof board keeps the variant and the
                    // history known at the clinch; the turn residual is zero
                    // placements left, as a final completed turn would leave.
                    let replacement = GameState::from_parts(
                        state.shared_board(),
                        StateParts {
                            variant: state.variant(),
                            stones: scenario.stones,
                            to_move: state.to_move(),
                            moves_left: 0,
                            opening: false,
                            swap_available: false,
                            swapped: state.swapped(),
                            current_turn: BitBoard::empty(),
                            previous_turn: BitBoard::empty(),
                            own_previous_turn: BitBoard::empty(),
                            handicap_stones: state.handicap_stones(),
                        },
                    )
                    .map_err(|error| error.to_string())?;
                    Ok(Some(PreparedClinchRow {
                        index,
                        replacement,
                        winner,
                        empty_nodes: bounds.empty_nodes,
                        last_move: state.last_move().map_or(-1, i32::from),
                        turn_count: state.turn_count(),
                    }))
                })
                .collect()
        });
        let prepared = prepared.map_err(PyValueError::new_err)?;
        let batch_size = self.states.len();
        let mut output = PyClinchData {
            batch_size,
            clinched: vec![false; batch_size],
            winner: vec![-1; batch_size],
            empty_nodes: vec![0; batch_size],
            last_move: vec![-1; batch_size],
            turn_count: vec![0; batch_size],
        };
        for row in prepared.into_iter().flatten() {
            output.clinched[row.index] = true;
            output.winner[row.index] = row.winner as i8;
            output.empty_nodes[row.index] = row.empty_nodes;
            output.last_move[row.index] = row.last_move;
            output.turn_count[row.index] = row.turn_count;
            self.states[row.index] = row.replacement;
        }
        Ok(output)
    }

    /// Packed state metadata, fixed bitboards, variants, and history sets.
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

    /// Contiguous model features and exact score annotations.
    ///
    /// `pda` supplies one playout-doubling advantage per row for the side to
    /// move. `schema_version` selects the production v4 encoding or the
    /// previous lineage's v3 encoding.
    #[pyo3(signature = (pda=None, schema_version=FEATURE_SCHEMA_VERSION, history_known=true))]
    fn feature_data(
        &self,
        py: Python<'_>,
        pda: Option<Vec<i8>>,
        schema_version: u8,
        history_known: bool,
    ) -> PyResult<PyFeatureData> {
        let contexts = feature_contexts(self.states.len(), pda, history_known)?;
        py.detach(|| {
            pack_feature_states(&self.states, &contexts, schema_version)
                .map_err(PyValueError::new_err)
        })
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

fn feature_contexts(
    rows: usize,
    pda: Option<Vec<i8>>,
    history_known: bool,
) -> PyResult<Vec<FeatureContext>> {
    match pda {
        None => Ok(vec![
            FeatureContext {
                pda: 0,
                history_known,
            };
            rows
        ]),
        Some(values) => {
            if values.len() != rows {
                return Err(PyValueError::new_err(format!(
                    "pda must contain one value per row ({rows})"
                )));
            }
            values
                .into_iter()
                .map(|pda| {
                    validate_pda(pda).map_err(PyValueError::new_err)?;
                    Ok(FeatureContext { pda, history_known })
                })
                .collect()
        }
    }
}

fn validate_pda(pda: i8) -> Result<(), String> {
    if pda.abs() > MAX_PLAYOUT_DOUBLING_ADVANTAGE {
        Err(format!(
            "playout doubling advantage must be in -{MAX_PLAYOUT_DOUBLING_ADVANTAGE}..={MAX_PLAYOUT_DOUBLING_ADVANTAGE}, got {pda}"
        ))
    } else {
        Ok(())
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
    pda: Vec<i8>,
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

    /// Precomputed contiguous schema-v4 features for the request states.
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

    /// Playout-doubling advantage of the side to move, one per row.
    #[getter]
    fn pda(&self) -> Vec<i8> {
        self.pda.clone()
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
    root_value: f32,
    actions: Vec<i32>,
    visits: Vec<u32>,
    q_values: Vec<f32>,
    priors: Vec<f32>,
    policy_target: Vec<f32>,
}

#[pyclass(name = "SearchResults", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PySearchResults {
    selected_actions: Vec<i32>,
    terminal: Vec<bool>,
    terminal_values: Vec<f32>,
    root_values: Vec<f32>,
    action_offsets: Vec<usize>,
    actions: Vec<i32>,
    visits: Vec<u32>,
    q_values: Vec<f32>,
    priors: Vec<f32>,
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

    /// Visit-weighted search value for the root player; terminal rows use `0`.
    ///
    /// A pie-pending root reports the opener's optimal-swap payoff. The game
    /// driver takes the pie swap when the responder's root value is negative.
    #[getter]
    fn root_values(&self) -> Vec<f32> {
        self.root_values.clone()
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
    fn priors(&self) -> Vec<f32> {
        self.priors.clone()
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
    budgets: Vec<u32>,
    pda_by_seat: Vec<[i8; 2]>,
    pending: Vec<PendingRow>,
}

#[pymethods]
impl PySearchBatch {
    /// Creates one tree per row.
    ///
    /// `simulations_per_root` overrides the shared budget row by row (for a
    /// playout-doubling advantage). `pda_by_seat` gives `(seat 0, seat 1)`
    /// advantages per row; every leaf evaluated for a side receives that
    /// side's advantage as a network input.
    #[new]
    #[pyo3(signature = (
        states,
        simulations=128,
        max_considered=16,
        c_visit=50.0,
        c_scale=1.0,
        deterministic_seed=None,
        simulations_per_root=None,
        pda_by_seat=None
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
        simulations_per_root: Option<Vec<u32>>,
        pda_by_seat: Option<Vec<(i8, i8)>>,
    ) -> PyResult<Self> {
        if simulations == 0 {
            return Err(PyValueError::new_err("simulations must be positive"));
        }
        if max_considered == 0 {
            return Err(PyValueError::new_err("max_considered must be positive"));
        }
        let parameters = GumbelParameters { c_visit, c_scale };
        parameters.validate().map_err(value_error)?;
        let rows = states.states.len();
        let budgets = match simulations_per_root {
            None => vec![simulations; rows],
            Some(budgets) => {
                if budgets.len() != rows {
                    return Err(PyValueError::new_err(
                        "simulations_per_root must contain one budget per row",
                    ));
                }
                if budgets.contains(&0) {
                    return Err(PyValueError::new_err(
                        "simulations_per_root entries must be positive",
                    ));
                }
                budgets
            }
        };
        let pda_by_seat = match pda_by_seat {
            None => vec![[0, 0]; rows],
            Some(values) => {
                if values.len() != rows {
                    return Err(PyValueError::new_err(
                        "pda_by_seat must contain one (seat 0, seat 1) pair per row",
                    ));
                }
                values
                    .into_iter()
                    .map(|(zero, one)| {
                        validate_pda(zero).map_err(PyValueError::new_err)?;
                        validate_pda(one).map_err(PyValueError::new_err)?;
                        Ok([zero, one])
                    })
                    .collect::<PyResult<Vec<_>>>()?
            }
        };
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
            budgets,
            pda_by_seat,
            pending: Vec::new(),
        })
    }

    fn __len__(&self) -> usize {
        self.trees.len()
    }

    /// Exact simulation budget per row.
    #[getter]
    fn budgets(&self) -> Vec<u32> {
        self.budgets.clone()
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
            Ok(pack_requests(indices, requests, &self.pda_by_seat))
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
                        self.budgets[tree_index],
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
            Ok(pack_requests(tree_indices, requests, &self.pda_by_seat))
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
            let node_count = tree.root_state().board().node_count();
            let Some(scheduler) = scheduler else {
                return Ok(PackedSearchRow {
                    selected_action: -2,
                    terminal: true,
                    terminal_value: tree
                        .root_terminal_value()
                        .expect("inactive roots are terminal"),
                    root_value: 0.0,
                    actions: Vec::new(),
                    visits: Vec::new(),
                    q_values: Vec::new(),
                    priors: Vec::new(),
                    policy_target: Vec::new(),
                });
            };
            let stats = tree.root_stats();
            let selected = scheduler
                .selected(&tree.root_completed_q(), &tree.root_visits())
                .map_err(|error| error.to_string())?;
            Ok(PackedSearchRow {
                selected_action: stats[selected].action.code(node_count),
                terminal: false,
                terminal_value: 0.0,
                root_value: tree.root_value().unwrap_or(0.0),
                actions: stats
                    .iter()
                    .map(|row| row.action.code(node_count))
                    .collect(),
                visits: stats.iter().map(|row| row.visits).collect(),
                q_values: stats.iter().map(|row| row.q).collect(),
                priors: stats.iter().map(|row| row.prior).collect(),
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
    let mut root_values = Vec::with_capacity(rows.len());
    let mut action_offsets = Vec::with_capacity(rows.len() + 1);
    let mut actions = Vec::with_capacity(action_count);
    let mut visits = Vec::with_capacity(action_count);
    let mut q_values = Vec::with_capacity(action_count);
    let mut priors = Vec::with_capacity(action_count);
    let mut policy_target = Vec::with_capacity(action_count);
    action_offsets.push(0);
    for row in rows {
        selected_actions.push(row.selected_action);
        terminal.push(row.terminal);
        terminal_values.push(row.terminal_value);
        root_values.push(row.root_value);
        actions.extend(row.actions);
        visits.extend(row.visits);
        q_values.extend(row.q_values);
        priors.extend(row.priors);
        policy_target.extend(row.policy_target);
        action_offsets.push(actions.len());
    }
    Ok(PySearchResults {
        selected_actions,
        terminal,
        terminal_values,
        root_values,
        action_offsets,
        actions,
        visits,
        q_values,
        priors,
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
                    .expect("a full *Star board must have a decisive winner"),
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
                .expect("a full *Star board must have a decisive winner");
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

fn pack_feature_states(
    states: &[GameState],
    contexts: &[FeatureContext],
    schema_version: u8,
) -> Result<PyFeatureData, String> {
    validate_schema_version(schema_version)?;
    debug_assert_eq!(states.len(), contexts.len());
    let rows = states
        .par_iter()
        .zip(contexts.par_iter())
        .map_init(ScoringScratch::default, |scratch, (state, context)| {
            let score = scratch.score_state(state);
            if schema_version == LEGACY_FEATURE_SCHEMA_VERSION {
                pack_legacy_feature_row(state, &score)
            } else {
                pack_feature_row(state, *context, &score)
            }
        })
        .collect();
    Ok(pack_feature_rows(rows, schema_version))
}

fn max_degree(board: &Board) -> usize {
    (0..board.node_count())
        .map(|node| board.neighbors(node).len())
        .max()
        .expect("supported boards are nonempty")
}

/// Shared node planes of both schemas plus the schema-v3 global scalars.
fn legacy_node_planes(
    board: &Board,
    stones: [BitBoard; 2],
    current: usize,
    terminal: bool,
    score: &ScoreResult,
    node: u16,
) -> ([f32; LEGACY_NODE_FEATURE_DIM], bool, i8, bool) {
    let node_index = usize::from(node);
    let opponent = 1 - current;
    let max_degree = max_degree(board);
    let current_stone = stones[current].contains(node);
    let opponent_stone = stones[opponent].contains(node);
    let empty = !current_stone && !opponent_stone;
    let owner = score.node_owner[node_index];
    let alive = score.alive_stones.contains(node);
    let ring = board.ring(node);
    let position = board.position(node);
    let arm_distance = position.min(ring - position);
    let legal = empty && !terminal;
    let planes = [
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
    ];
    (planes, legal, owner, alive)
}

fn legacy_global_scalars(
    board: &Board,
    stones: [BitBoard; 2],
    current: usize,
    moves_left_fraction: f32,
    opening: bool,
    terminal: bool,
    score: &ScoreResult,
) -> [f32; LEGACY_GLOBAL_FEATURE_DIM] {
    let opponent = 1 - current;
    let occupied = stones[0].count() + stones[1].count();
    let current_count = stones[current].count();
    let opponent_count = stones[opponent].count();
    let current_score = score.players[current];
    let opponent_score = score.players[opponent];
    let score_scale = f64::from(SCORE_MARGIN_SUPPORT);
    let star_scale = (f64::from(board.peri_count()) / 2.0).max(1.0);
    [
        (f64::from(board.rings()) / 10.0) as f32,
        (f64::from(occupied) / f64::from(board.node_count())) as f32,
        (f64::from(current_count) / f64::from(board.node_count())) as f32,
        (f64::from(opponent_count) / f64::from(board.node_count())) as f32,
        moves_left_fraction,
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
    ]
}

/// Schema v3: the previous lineage's fifteen node planes and seventeen scalars.
fn pack_legacy_feature_row(state: &GameState, score: &ScoreResult) -> PackedFeatureRow {
    let board = state.board();
    let stones = state.stones();
    let current = state.to_move().index();
    let terminal = state.is_terminal();
    let node_count = usize::from(board.node_count());
    let mut node_features = Vec::with_capacity(node_count * LEGACY_NODE_FEATURE_DIM);
    let mut legal_nodes = Vec::with_capacity(node_count);
    let mut node_owner = Vec::with_capacity(node_count);
    let mut alive_stones = Vec::with_capacity(node_count);
    for node in 0..board.node_count() {
        let (planes, legal, owner, alive) =
            legacy_node_planes(board, stones, current, terminal, score, node);
        node_features.extend(planes);
        legal_nodes.push(u8::from(legal));
        node_owner.push(owner);
        alive_stones.push(u8::from(alive));
    }
    let global_features = legacy_global_scalars(
        board,
        stones,
        current,
        (f64::from(state.moves_left()) / 2.0) as f32,
        state.is_opening(),
        terminal,
        score,
    );
    PackedFeatureRow {
        rings: board.rings(),
        node_count,
        node_features,
        global_features: global_features.to_vec(),
        legal_nodes,
        score_components: packed_score_components(score),
        node_owner,
        alive_stones,
    }
}

/// Schema v4: nineteen node planes and twenty-five scalars.
fn pack_feature_row(
    state: &GameState,
    context: FeatureContext,
    score: &ScoreResult,
) -> PackedFeatureRow {
    let board = state.board();
    let stones = state.stones();
    let current = state.to_move().index();
    let terminal = state.is_terminal();
    let node_count = usize::from(board.node_count());
    let current_turn = state.current_turn_set();
    let own_previous = state.own_previous_turn_set();
    let opponent_previous = state.previous_turn_set();
    let handicap_stones = state.handicap_stones();
    let mut node_features = Vec::with_capacity(node_count * NODE_FEATURE_DIM);
    let mut legal_nodes = Vec::with_capacity(node_count);
    let mut node_owner = Vec::with_capacity(node_count);
    let mut alive_stones = Vec::with_capacity(node_count);
    for node in 0..board.node_count() {
        let (planes, legal, owner, alive) =
            legacy_node_planes(board, stones, current, terminal, score, node);
        node_features.extend(planes);
        node_features.extend([
            binary_feature(current_turn.contains(node)),
            binary_feature(own_previous.contains(node)),
            binary_feature(opponent_previous.contains(node)),
            binary_feature(handicap_stones.contains(node)),
        ]);
        legal_nodes.push(u8::from(legal));
        node_owner.push(owner);
        alive_stones.push(u8::from(alive));
    }
    let variant = state.variant();
    let opening = state.is_opening();
    let turn_total = state.current_turn_total().max(1);
    let mut global_features = legacy_global_scalars(
        board,
        stones,
        current,
        (f64::from(state.moves_left()) / f64::from(turn_total)) as f32,
        opening,
        terminal,
        score,
    )
    .to_vec();
    global_features.extend([
        f32::from(variant.turn_size()) / 2.0,
        f32::from(variant.handicap()) / f32::from(MAX_HANDICAP),
        binary_feature(opening && variant.handicap() >= 2),
        if opening {
            f32::from(state.moves_left()) / f32::from(MAX_HANDICAP)
        } else {
            0.0
        },
        binary_feature(state.is_pie_pending()),
        binary_feature(state.swap_available()),
        binary_feature(context.history_known),
        f32::from(context.pda) / f32::from(MAX_PLAYOUT_DOUBLING_ADVANTAGE),
    ]);
    debug_assert_eq!(global_features.len(), GLOBAL_FEATURE_DIM);
    PackedFeatureRow {
        rings: board.rings(),
        node_count,
        node_features,
        global_features,
        legal_nodes,
        score_components: packed_score_components(score),
        node_owner,
        alive_stones,
    }
}

fn pack_feature_rows(rows: Vec<PackedFeatureRow>, schema_version: u8) -> PyFeatureData {
    let node_dim = node_feature_dim(schema_version);
    let global_dim = global_feature_dim(schema_version);
    let batch_size = rows.len();
    let max_nodes = rows.iter().map(|row| row.node_count).max().unwrap_or(0);
    let mut rings = Vec::with_capacity(batch_size);
    let mut node_features = vec![0.0_f32; batch_size * max_nodes * node_dim];
    let mut global_features = Vec::with_capacity(batch_size * global_dim);
    let mut node_mask = vec![0_u8; batch_size * max_nodes];
    let mut legal_action_mask = vec![0_u8; batch_size * max_nodes];
    let mut score_components = Vec::with_capacity(batch_size * SCORE_COMPONENT_DIM);
    let mut node_owner = vec![-1_i8; batch_size * max_nodes];
    let mut alive_stones = vec![0_u8; batch_size * max_nodes];
    for (row_index, row) in rows.into_iter().enumerate() {
        rings.push(row.rings);
        let feature_start = row_index * max_nodes * node_dim;
        let feature_end = feature_start + row.node_count * node_dim;
        node_features[feature_start..feature_end].copy_from_slice(&row.node_features);
        debug_assert_eq!(row.global_features.len(), global_dim);
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
        schema_version,
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
    let batch_size = states.len();
    let mut data = PyStateData {
        rings,
        node_count,
        batch_size,
        zero_bits: Vec::with_capacity(batch_size * BITBOARD_WORDS),
        one_bits: Vec::with_capacity(batch_size * BITBOARD_WORDS),
        legal_bits: Vec::with_capacity(batch_size * BITBOARD_WORDS),
        current_turn_bits: Vec::with_capacity(batch_size * BITBOARD_WORDS),
        previous_turn_bits: Vec::with_capacity(batch_size * BITBOARD_WORDS),
        own_previous_turn_bits: Vec::with_capacity(batch_size * BITBOARD_WORDS),
        handicap_bits: Vec::with_capacity(batch_size * BITBOARD_WORDS),
        hashes: Vec::with_capacity(batch_size),
        stones_placed: Vec::with_capacity(batch_size),
        to_move: Vec::with_capacity(batch_size),
        moves_left: Vec::with_capacity(batch_size),
        opening: Vec::with_capacity(batch_size),
        mid_turn: Vec::with_capacity(batch_size),
        terminal: Vec::with_capacity(batch_size),
        mode: Vec::with_capacity(batch_size),
        handicap: Vec::with_capacity(batch_size),
        pie: Vec::with_capacity(batch_size),
        pie_pending: Vec::with_capacity(batch_size),
        swap_available: Vec::with_capacity(batch_size),
        swapped: Vec::with_capacity(batch_size),
        turn_size: Vec::with_capacity(batch_size),
        current_turn_total: Vec::with_capacity(batch_size),
        turn_count: Vec::with_capacity(batch_size),
    };
    let hashes: Vec<_> = states.par_iter().map(GameState::hash64).collect();
    for (state, hash) in states.iter().zip(hashes) {
        let variant = state.variant();
        data.zero_bits
            .extend(state.stones_for(Player::Zero).words());
        data.one_bits.extend(state.stones_for(Player::One).words());
        data.legal_bits
            .extend(state.legal_actions().placements.words());
        data.current_turn_bits
            .extend(state.current_turn_set().words());
        data.previous_turn_bits
            .extend(state.previous_turn_set().words());
        data.own_previous_turn_bits
            .extend(state.own_previous_turn_set().words());
        data.handicap_bits.extend(state.handicap_stones().words());
        data.hashes.push(hash);
        data.stones_placed.push(state.stones_placed());
        data.to_move.push(state.to_move() as u8);
        data.moves_left.push(state.moves_left());
        data.opening.push(state.is_opening());
        data.mid_turn.push(state.is_mid_turn());
        data.terminal.push(state.is_terminal());
        data.mode.push(variant.mode().index());
        data.handicap.push(variant.handicap());
        data.pie.push(variant.pie());
        data.pie_pending.push(state.is_pie_pending());
        data.swap_available.push(state.swap_available());
        data.swapped.push(state.swapped());
        data.turn_size.push(variant.turn_size());
        data.current_turn_total.push(state.current_turn_total());
        data.turn_count.push(state.turn_count());
    }
    data
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

fn bitboard_at(words: Option<&[u64]>, row: usize) -> BitBoard {
    match words {
        None => BitBoard::empty(),
        Some(words) => {
            let mut buffer = [0_u64; BITBOARD_WORDS];
            buffer.copy_from_slice(&words[row * BITBOARD_WORDS..(row + 1) * BITBOARD_WORDS]);
            BitBoard::from_words(buffer)
        }
    }
}

fn decode_semantic_states(board: Arc<Board>, rows: SemanticRows) -> Result<Vec<GameState>, String> {
    let count = rows.row_count();
    let word_len =
        |words: Option<&Vec<u64>>| words.is_none_or(|w| w.len() == count * BITBOARD_WORDS);
    let flag_len = |flags: Option<&Vec<bool>>| flags.is_none_or(|f| f.len() == count);
    if rows.zero_bits.len() != count * BITBOARD_WORDS
        || rows.one_bits.len() != count * BITBOARD_WORDS
        || rows.moves_left.len() != count
        || rows.opening.len() != count
        || !flag_len(rows.swap_available.as_ref())
        || !flag_len(rows.swapped.as_ref())
        || !word_len(rows.current_turn_bits.as_ref())
        || !word_len(rows.previous_turn_bits.as_ref())
        || !word_len(rows.own_previous_turn_bits.as_ref())
        || !word_len(rows.handicap_bits.as_ref())
    {
        return Err(format!("semantic buffers disagree on row count {count}"));
    }
    let variants = variants_from_rows(count, rows.mode, rows.handicap, rows.pie)?;
    let SemanticRows {
        zero_bits,
        one_bits,
        to_move,
        moves_left,
        opening,
        swap_available,
        swapped,
        current_turn_bits,
        previous_turn_bits,
        own_previous_turn_bits,
        handicap_bits,
        ..
    } = rows;

    (0..count)
        .into_par_iter()
        .map(|row| {
            let player = match to_move[row] {
                0 => Player::Zero,
                1 => Player::One,
                value => {
                    return Err(format!("row {row} has invalid to_move value {value}"));
                }
            };
            GameState::from_parts(
                Arc::clone(&board),
                StateParts {
                    variant: variants[row],
                    stones: [
                        bitboard_at(Some(&zero_bits), row),
                        bitboard_at(Some(&one_bits), row),
                    ],
                    to_move: player,
                    moves_left: moves_left[row],
                    opening: opening[row],
                    swap_available: swap_available.as_ref().is_some_and(|flags| flags[row]),
                    swapped: swapped.as_ref().is_some_and(|flags| flags[row]),
                    current_turn: bitboard_at(current_turn_bits.as_deref(), row),
                    previous_turn: bitboard_at(previous_turn_bits.as_deref(), row),
                    own_previous_turn: bitboard_at(own_previous_turn_bits.as_deref(), row),
                    handicap_stones: bitboard_at(handicap_bits.as_deref(), row),
                },
            )
            .map_err(|error| format!("row {row}: {error}"))
        })
        .collect()
}

/// One decoded feature row of `encode_semantic_features`.
struct SemanticFeatureRow {
    state: GameState,
    context: FeatureContext,
}

fn decode_semantic_feature_rows(
    rings: &[u8],
    metadata: &[u8],
    stone_values: &[u8],
    history_flags: Option<&[u8]>,
    schema_version: u8,
) -> Result<Vec<SemanticFeatureRow>, String> {
    validate_schema_version(schema_version)?;
    let rows = rings.len();
    if rows == 0 {
        return Err("semantic feature batch must contain at least one row".to_owned());
    }
    let metadata_dim = if schema_version == LEGACY_FEATURE_SCHEMA_VERSION {
        LEGACY_SEMANTIC_METADATA_DIM
    } else {
        SEMANTIC_METADATA_DIM
    };
    if metadata.len() != rows * metadata_dim {
        return Err(format!(
            "semantic metadata must contain {} bytes for {rows} rows",
            rows * metadata_dim
        ));
    }
    if let Some(flags) = history_flags
        && flags.len() != stone_values.len()
    {
        return Err("history_flags must align with the stones buffer".to_owned());
    }
    let mut boards: HashMap<u8, Arc<Board>> = HashMap::new();
    let mut decoded = Vec::with_capacity(rows);
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
        let mut current_turn = BitBoard::empty();
        let mut own_previous_turn = BitBoard::empty();
        let mut previous_turn = BitBoard::empty();
        let mut handicap_stones = BitBoard::empty();
        if let Some(flags) = history_flags {
            for (node_index, &flag) in flags[stone_offset..stone_end].iter().enumerate() {
                if flag
                    & !(HISTORY_CURRENT_TURN
                        | HISTORY_OWN_PREVIOUS_TURN
                        | HISTORY_OPPONENT_PREVIOUS_TURN
                        | HISTORY_HANDICAP_STONE)
                    != 0
                {
                    return Err(format!(
                        "row {row} node {node_index} has invalid history flags {flag}"
                    ));
                }
                let node = u16::try_from(node_index).expect("board nodes fit in u16");
                if flag & HISTORY_CURRENT_TURN != 0 {
                    current_turn.insert(node);
                }
                if flag & HISTORY_OWN_PREVIOUS_TURN != 0 {
                    own_previous_turn.insert(node);
                }
                if flag & HISTORY_OPPONENT_PREVIOUS_TURN != 0 {
                    previous_turn.insert(node);
                }
                if flag & HISTORY_HANDICAP_STONE != 0 {
                    handicap_stones.insert(node);
                }
            }
        }
        stone_offset = stone_end;

        let base = row * metadata_dim;
        let to_move = match metadata[base] {
            0 => Player::Zero,
            1 => Player::One,
            value => return Err(format!("row {row} has invalid to_move value {value}")),
        };
        let moves_left = metadata[base + 1];
        let opening = semantic_bool("opening", row, metadata[base + 2])?;
        let terminal = semantic_bool("terminal", row, metadata[base + 3])?;
        let (variant, swap_available, swapped, context) =
            if schema_version == LEGACY_FEATURE_SCHEMA_VERSION {
                (Variant::STANDARD, false, false, FeatureContext::default())
            } else {
                let variant = parse_variant(
                    metadata[base + 4],
                    metadata[base + 5],
                    semantic_bool("pie", row, metadata[base + 6])?,
                )
                .map_err(|error| format!("row {row}: {error}"))?;
                let swap_available = semantic_bool("swap_available", row, metadata[base + 7])?;
                let swapped = semantic_bool("swapped", row, metadata[base + 8])?;
                let history_known = semantic_bool("history_known", row, metadata[base + 9])?;
                let pda = i8::from_ne_bytes([metadata[base + 10]]);
                validate_pda(pda).map_err(|error| format!("row {row}: {error}"))?;
                (
                    variant,
                    swap_available,
                    swapped,
                    FeatureContext { pda, history_known },
                )
            };
        let state = GameState::from_parts(
            Arc::clone(&board),
            StateParts {
                variant,
                stones,
                to_move,
                moves_left,
                opening,
                swap_available,
                swapped,
                current_turn,
                previous_turn,
                own_previous_turn,
                handicap_stones,
            },
        )
        .map_err(|error| format!("row {row}: {error}"))?;
        if state.is_terminal() != terminal {
            return Err(format!("row {row} terminal must equal board-full"));
        }
        decoded.push(SemanticFeatureRow { state, context });
    }
    if stone_offset != stone_values.len() {
        return Err(format!(
            "semantic stones contain {} trailing node values",
            stone_values.len() - stone_offset
        ));
    }
    Ok(decoded)
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
        .map(|&index| {
            (
                index,
                GameState::with_variant(Arc::clone(board), states[index].variant()),
            )
        })
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
        let node_count = states[index].board().node_count();
        let action =
            Action::from_code(action_code, node_count).map_err(|error| error.to_string())?;
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

fn pack_requests(
    tree_indices: Vec<usize>,
    requests: Vec<EvaluationRequest>,
    pda_by_seat: &[[i8; 2]],
) -> PyEvalBatch {
    let tokens = requests.iter().map(|request| request.token).collect();
    let request_states = requests
        .iter()
        .map(|request| request.state.clone())
        .collect::<Vec<_>>();
    let pda: Vec<i8> = tree_indices
        .iter()
        .zip(&request_states)
        .map(|(tree_index, state)| pda_by_seat[*tree_index][state.to_move().index()])
        .collect();
    let contexts: Vec<_> = pda.iter().map(|pda| FeatureContext::known(*pda)).collect();
    let states = pack_states(&request_states);
    let features = pack_feature_states(&request_states, &contexts, FEATURE_SCHEMA_VERSION)
        .expect("the production schema version is always valid");
    let mut legal_offsets = Vec::with_capacity(requests.len() + 1);
    let mut legal_actions = Vec::new();
    legal_offsets.push(0);
    for request in requests {
        let node_count = request.state.board().node_count();
        legal_actions.extend(
            request
                .legal_actions
                .into_iter()
                .map(|action| action.code(node_count)),
        );
        legal_offsets.push(legal_actions.len());
    }
    PyEvalBatch {
        tree_indices,
        tokens,
        states,
        features,
        legal_offsets,
        legal_actions,
        pda,
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

/// Encodes a heterogeneous semantic-key batch from contiguous byte buffers.
///
/// `rings` is `uint8[B]`; `metadata` is `uint8[B, 11]` in
/// `(to_move, moves_left, opening, terminal, mode, handicap, pie,
/// swap_available, swapped, history_known, pda)` order, where `pda` is a
/// two's-complement `int8`; `stones` is the concatenation of each row's native
/// node-order `int8` stones; and the optional `history_flags` buffer aligns
/// with `stones` and carries the bits `1` current turn, `2` own previous turn,
/// `4` opponent previous turn, `8` handicap stone.
///
/// With `schema_version=3` the metadata has the four legacy columns and the
/// previous lineage's fifteen-plane encoding is produced.
#[pyfunction]
#[pyo3(signature = (rings, metadata, stones, history_flags=None, schema_version=FEATURE_SCHEMA_VERSION))]
fn encode_semantic_features(
    py: Python<'_>,
    rings: &Bound<'_, PyBytes>,
    metadata: &Bound<'_, PyBytes>,
    stones: &Bound<'_, PyBytes>,
    history_flags: Option<&Bound<'_, PyBytes>>,
    schema_version: u8,
) -> PyResult<PyFeatureData> {
    let rings = rings.as_bytes().to_vec();
    let metadata = metadata.as_bytes().to_vec();
    let stones = stones.as_bytes().to_vec();
    let history_flags = history_flags.map(|flags| flags.as_bytes().to_vec());
    py.detach(move || {
        let rows = decode_semantic_feature_rows(
            &rings,
            &metadata,
            &stones,
            history_flags.as_deref(),
            schema_version,
        )
        .map_err(PyValueError::new_err)?;
        let states: Vec<_> = rows.iter().map(|row| row.state.clone()).collect();
        let contexts: Vec<_> = rows.iter().map(|row| row.context).collect();
        pack_feature_states(&states, &contexts, schema_version).map_err(PyValueError::new_err)
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

/// Production feature schema version.
#[pyfunction]
const fn native_feature_schema_version() -> u8 {
    FEATURE_SCHEMA_VERSION
}

/// Production feature schema hash.
#[pyfunction]
const fn native_feature_schema_hash() -> u64 {
    FEATURE_SCHEMA_HASH
}

/// Previous lineage's feature schema hash, available for teacher inference.
#[pyfunction]
const fn native_legacy_feature_schema_hash() -> u64 {
    LEGACY_FEATURE_SCHEMA_HASH
}

/// Largest supported handicap.
#[pyfunction]
const fn native_max_handicap() -> u8 {
    MAX_HANDICAP
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
    module.add_class::<PyClinchData>()?;
    module.add_class::<PyScoreData>()?;
    module.add_class::<PyFeatureData>()?;
    module.add_class::<PyStateBatch>()?;
    module.add_class::<PyEvalBatch>()?;
    module.add_class::<PySearchResults>()?;
    module.add_class::<PySearchBatch>()?;
    module.add_function(wrap_pyfunction!(native_rules_hash, module)?)?;
    module.add_function(wrap_pyfunction!(native_rules_hash_tag, module)?)?;
    module.add_function(wrap_pyfunction!(native_rules_schema, module)?)?;
    module.add_function(wrap_pyfunction!(native_feature_schema_version, module)?)?;
    module.add_function(wrap_pyfunction!(native_feature_schema_hash, module)?)?;
    module.add_function(wrap_pyfunction!(native_legacy_feature_schema_hash, module)?)?;
    module.add_function(wrap_pyfunction!(native_max_handicap, module)?)?;
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
        root_values: Vec<f32>,
        action_offsets: Vec<usize>,
        actions: Vec<i32>,
        visits: Vec<u32>,
        q_values: Vec<f32>,
        policy_target: Vec<f32>,
    }

    fn standard_rows(packed: &PyStateData) -> SemanticRows {
        SemanticRows {
            zero_bits: packed.zero_bits.clone(),
            one_bits: packed.one_bits.clone(),
            to_move: packed.to_move.clone(),
            moves_left: packed.moves_left.clone(),
            opening: packed.opening.clone(),
            mode: None,
            handicap: None,
            pie: None,
            swap_available: None,
            swapped: None,
            current_turn_bits: None,
            previous_turn_bits: None,
            own_previous_turn_bits: None,
            handicap_bits: None,
        }
    }

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
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
    fn selected_terminal_rows_reset_independently_and_keep_their_variant() {
        let board = Arc::new(Board::new(4).unwrap());
        let classic = Variant::new(Mode::Classic, 1, false).unwrap();
        let mut terminal = GameState::with_variant(Arc::clone(&board), classic);
        for node in 0..board.node_count() {
            terminal.apply(Action::Place(node)).unwrap();
        }
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
        assert_eq!(reset[0].variant(), classic);
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
    fn semantic_import_round_trips_variants_and_history() {
        let board = Arc::new(Board::new(4).unwrap());
        let variants = [
            Variant::new(Mode::Classic, 1, false).unwrap(),
            Variant::new(Mode::Double, 4, false).unwrap(),
            Variant::new(Mode::Double, 1, true).unwrap(),
            Variant::new(Mode::Classic, 1, true).unwrap(),
        ];
        let mut states = Vec::new();
        for (index, variant) in variants.iter().enumerate() {
            let mut state = GameState::with_variant(Arc::clone(&board), *variant);
            for node in 0..(7 + index as u16) {
                if state.swap_available() && index % 2 == 0 {
                    state.apply(Action::Swap).unwrap();
                }
                state.apply(Action::Place(node)).unwrap();
            }
            states.push(state);
        }
        let packed = pack_states(&states);
        assert_eq!(packed.mode, [0, 1, 1, 0]);
        assert_eq!(packed.handicap, [1, 4, 1, 1]);
        assert_eq!(packed.pie, [false, false, true, true]);
        assert!(packed.swapped[2]);
        assert!(!packed.swapped[3]);
        let imported = decode_semantic_states(Arc::clone(&board), packed.semantic_rows()).unwrap();
        for (state, imported) in states.iter().zip(&imported) {
            assert_eq!(imported.key(), state.key());
            assert_eq!(imported.hash64(), state.hash64());
            assert_eq!(imported.swapped(), state.swapped());
            assert_eq!(imported.variant(), state.variant());
        }

        // Importing without history keeps the position but loses the sets.
        let without =
            decode_semantic_states(board, standard_rows(&pack_states(&states[..1]))).unwrap();
        assert_eq!(without[0].stones(), states[0].stones());
        assert_ne!(without[0].variant(), states[0].variant());
    }

    #[test]
    fn imported_and_reset_rows_are_immediately_searchable() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut imported_source = GameState::new(Arc::clone(&board));
        imported_source.apply(Action::Place(7)).unwrap();
        imported_source.apply(Action::Place(8)).unwrap();
        let packed = pack_states(&[imported_source]);
        let replacements =
            decode_semantic_states(Arc::clone(&board), standard_rows(&packed)).unwrap();

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
        let rows = |zero: Vec<u64>, one: Vec<u64>, to_move: u8, moves_left: u8, opening: bool| {
            SemanticRows {
                zero_bits: zero,
                one_bits: one,
                to_move: vec![to_move],
                moves_left: vec![moves_left],
                opening: vec![opening],
                mode: None,
                handicap: None,
                pie: None,
                swap_available: None,
                swapped: None,
                current_turn_bits: None,
                previous_turn_bits: None,
                own_previous_turn_bits: None,
                handicap_bits: None,
            }
        };
        assert!(
            decode_semantic_states(
                Arc::clone(&board),
                rows(empty_words.clone(), empty_words.clone(), 2, 1, true),
            )
            .is_err()
        );
        assert!(
            decode_semantic_states(
                Arc::clone(&board),
                rows(vec![1, 0, 0, 0, 0], vec![1, 0, 0, 0, 0], 0, 2, false),
            )
            .is_err()
        );
        assert!(
            decode_semantic_states(
                Arc::clone(&board),
                rows(empty_words.clone(), empty_words.clone(), 1, 0, false),
            )
            .is_err()
        );
        // Handicap games cannot use the pie rule.
        let mut invalid = rows(empty_words.clone(), empty_words, 0, 3, true);
        invalid.handicap = Some(vec![3]);
        invalid.pie = Some(vec![true]);
        assert!(decode_semantic_states(board, invalid).is_err());
    }

    #[test]
    fn feature_v4_encodes_variants_history_and_context() {
        let rings = [4_u8, 10];
        // Row 0: classic, handicap 1, pie pending on an empty board, pda +2.
        // Row 1: standard double game, empty board, unknown history, pda 0.
        let metadata = [
            0_u8, 1, 1, 0, 0, 1, 1, 0, 0, 1, 2, //
            0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
        ];
        let stones = vec![u8::MAX; 50 + 275];
        let rows =
            decode_semantic_feature_rows(&rings, &metadata, &stones, None, FEATURE_SCHEMA_VERSION)
                .unwrap();
        assert!(rows[0].state.is_pie_pending());
        assert_eq!(rows[0].context, FeatureContext::known(2));
        assert!(!rows[1].context.history_known);
        let states: Vec<_> = rows.iter().map(|row| row.state.clone()).collect();
        let contexts: Vec<_> = rows.iter().map(|row| row.context).collect();
        let features = pack_feature_states(&states, &contexts, FEATURE_SCHEMA_VERSION).unwrap();

        assert_eq!(features.batch_size, 2);
        assert_eq!(features.max_nodes, 275);
        assert_eq!(features.schema_version, 4);
        assert_eq!(features.node_feature_dim(), 19);
        assert_eq!(features.global_feature_dim(), 25);
        assert_eq!(features.feature_schema_hash(), 0xcb0e_1e89_a6ce_3540);
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

        let globals = f32s(&features.buffers.global_features);
        assert_eq!(globals.len(), 2 * GLOBAL_FEATURE_DIM);
        assert!((globals[0] - 0.4).abs() <= f32::EPSILON);
        assert_eq!(globals[4], 1.0); // one placement left of a one-stone opening
        assert_eq!(globals[5], 1.0); // opening
        assert_eq!(globals[6], 0.0); // terminal
        assert_eq!(globals[17], 0.5); // classic turn size
        assert!((globals[18] - 1.0 / 9.0).abs() <= 1.0e-6);
        assert_eq!(globals[19], 0.0); // no handicap phase for k = 1
        assert!((globals[20] - 1.0 / 9.0).abs() <= 1.0e-6);
        assert_eq!(globals[21], 1.0); // pie pending
        assert_eq!(globals[22], 0.0); // swap not available yet
        assert_eq!(globals[23], 1.0); // history known
        assert!((globals[24] - 2.0 / 3.0).abs() <= 1.0e-6);
        let second = &globals[GLOBAL_FEATURE_DIM..];
        assert_eq!(second[0], 1.0);
        assert_eq!(second[17], 1.0); // double turn size
        assert_eq!(second[21], 0.0);
        assert_eq!(second[23], 0.0); // history unknown
        assert_eq!(second[24], 0.0);
    }

    #[test]
    fn feature_v4_history_planes_follow_the_game() {
        let board = Arc::new(Board::new(4).unwrap());
        let handicap = Variant::new(Mode::Double, 3, false).unwrap();
        let mut state = GameState::with_variant(Arc::clone(&board), handicap);
        for node in [0_u16, 1, 2] {
            state.apply(Action::Place(node)).unwrap();
        }
        // Player 1's turn: one of two placed.
        state.apply(Action::Place(10)).unwrap();
        let score = ScoringScratch::default().score_state(&state);
        let row = pack_feature_row(&state, FeatureContext::known(-1), &score);
        let plane = |node: usize, plane: usize| row.node_features[node * NODE_FEATURE_DIM + plane];
        // Node 10 was placed this turn by the mover (player 1).
        assert_eq!(plane(10, 15), 1.0);
        assert_eq!(plane(10, 16), 0.0);
        assert_eq!(plane(10, 17), 0.0);
        assert_eq!(plane(10, 18), 0.0);
        // Nodes 0..3 are the opponent's previous turn and the handicap stones.
        for node in 0..3 {
            assert_eq!(plane(node, 15), 0.0);
            assert_eq!(plane(node, 16), 0.0);
            assert_eq!(plane(node, 17), 1.0);
            assert_eq!(plane(node, 18), 1.0);
        }
        assert_eq!(plane(20, 17), 0.0);
        assert_eq!(row.global_features[4], 0.5); // one of two placements left
        assert_eq!(row.global_features[18], 3.0 / 9.0);
        assert_eq!(row.global_features[19], 0.0); // opening is over
        assert_eq!(row.global_features[20], 0.0);
        assert!((row.global_features[24] + 1.0 / 3.0).abs() <= 1.0e-6);

        // Mid-opening: handicap phase is on and the remaining count is exposed.
        let mut opening = GameState::with_variant(board, handicap);
        opening.apply(Action::Place(5)).unwrap();
        let score = ScoringScratch::default().score_state(&opening);
        let row = pack_feature_row(&opening, FeatureContext::known(0), &score);
        assert_eq!(row.global_features[19], 1.0);
        assert!((row.global_features[20] - 2.0 / 9.0).abs() <= 1.0e-6);
        assert!((row.global_features[4] - 2.0 / 3.0).abs() <= 1.0e-6);
        assert_eq!(row.node_features[5 * NODE_FEATURE_DIM + 15], 1.0);
        assert_eq!(row.node_features[5 * NODE_FEATURE_DIM + 18], 1.0);
    }

    #[test]
    fn swapped_and_unswapped_positions_encode_identically() {
        let board = Arc::new(Board::new(6).unwrap());
        let pie = Variant::new(Mode::Double, 1, true).unwrap();
        let mut kept = GameState::with_variant(board, pie);
        kept.apply(Action::Place(17)).unwrap();
        let mut swapped = kept.clone();
        swapped.apply(Action::Swap).unwrap();
        // The pre-swap position advertises the swap; the post-swap position
        // does not. Everything else, including the history planes, matches.
        let mut scratch = ScoringScratch::default();
        let kept_row =
            pack_feature_row(&kept, FeatureContext::known(0), &scratch.score_state(&kept));
        let swapped_row = pack_feature_row(
            &swapped,
            FeatureContext::known(0),
            &scratch.score_state(&swapped),
        );
        assert_eq!(kept_row.node_features, swapped_row.node_features);
        assert_eq!(kept_row.legal_nodes, swapped_row.legal_nodes);
        let mut expected = kept_row.global_features.clone();
        expected[22] = 0.0;
        assert_eq!(swapped_row.global_features, expected);
        assert_eq!(kept_row.global_features[22], 1.0);
    }

    #[test]
    fn legacy_feature_v3_is_still_produced_from_the_four_field_metadata() {
        let rings = [4_u8, 10];
        let metadata = [0_u8, 1, 1, 0, 0, 1, 1, 0];
        let stones = vec![u8::MAX; 50 + 275];
        let rows = decode_semantic_feature_rows(
            &rings,
            &metadata,
            &stones,
            None,
            LEGACY_FEATURE_SCHEMA_VERSION,
        )
        .unwrap();
        let states: Vec<_> = rows.iter().map(|row| row.state.clone()).collect();
        let contexts: Vec<_> = rows.iter().map(|row| row.context).collect();
        let features =
            pack_feature_states(&states, &contexts, LEGACY_FEATURE_SCHEMA_VERSION).unwrap();
        assert_eq!(features.schema_version, 3);
        assert_eq!(features.node_feature_dim(), 15);
        assert_eq!(features.global_feature_dim(), 17);
        assert_eq!(features.feature_schema_hash(), 0x6b5b_00f6_38e9_c16b);
        assert_eq!(
            features.buffers.node_features.len(),
            2 * 275 * 15 * std::mem::size_of::<f32>()
        );
        let globals = f32s(&features.buffers.global_features);
        assert_eq!(globals.len(), 2 * LEGACY_GLOBAL_FEATURE_DIM);
        assert!((globals[0] - 0.4).abs() <= f32::EPSILON);
        assert_eq!(globals[4], 0.5);
        assert_eq!(globals[5], 1.0);
        assert_eq!(globals[6], 0.0);
        assert_eq!(globals[LEGACY_GLOBAL_FEATURE_DIM], 1.0);

        // The four-field metadata is rejected under the production schema
        // and the eleven-field metadata under the legacy schema.
        assert!(
            decode_semantic_feature_rows(&rings, &metadata, &stones, None, FEATURE_SCHEMA_VERSION)
                .is_err()
        );
        assert!(decode_semantic_feature_rows(&rings, &metadata, &stones, None, 2).is_err());
    }

    #[test]
    fn history_flags_rebuild_the_retained_sets() {
        let board = Arc::new(Board::new(4).unwrap());
        let mut state = GameState::new(Arc::clone(&board));
        for node in [3_u16, 8, 9, 20] {
            state.apply(Action::Place(node)).unwrap();
        }
        // Player 1 to move? No: 3 (opening), 8+9 (player 1), 20 (player 0 mid-turn).
        assert_eq!(state.to_move(), Player::Zero);
        let mut stones = vec![u8::MAX; 50];
        let mut flags = vec![0_u8; 50];
        for node in 0..50_u16 {
            if let Some(player) = state.stone_at(node) {
                stones[usize::from(node)] = player as u8;
            }
        }
        flags[20] = HISTORY_CURRENT_TURN;
        flags[8] = HISTORY_OPPONENT_PREVIOUS_TURN;
        flags[9] = HISTORY_OPPONENT_PREVIOUS_TURN;
        flags[3] = HISTORY_OWN_PREVIOUS_TURN | HISTORY_HANDICAP_STONE;
        let metadata = [0_u8, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0];
        let rows = decode_semantic_feature_rows(
            &[4],
            &metadata,
            &stones,
            Some(&flags),
            FEATURE_SCHEMA_VERSION,
        )
        .unwrap();
        assert_eq!(rows[0].state.key(), state.key());

        // A flag on an opponent stone claimed as the mover's turn is rejected.
        flags[8] = HISTORY_CURRENT_TURN;
        assert!(
            decode_semantic_feature_rows(
                &[4],
                &metadata,
                &stones,
                Some(&flags),
                FEATURE_SCHEMA_VERSION,
            )
            .is_err()
        );
        flags[8] = 0x10;
        assert!(
            decode_semantic_feature_rows(
                &[4],
                &metadata,
                &stones,
                Some(&flags),
                FEATURE_SCHEMA_VERSION,
            )
            .is_err()
        );
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
    fn swap_code_applies_only_when_available() {
        let board = Arc::new(Board::new(4).unwrap());
        let pie = Variant::new(Mode::Double, 1, true).unwrap();
        let state = GameState::with_variant(Arc::clone(&board), pie);
        let swap_code = i32::from(board.node_count());
        assert!(
            prepare_applied_rows(std::slice::from_ref(&state), vec![0], vec![swap_code]).is_err()
        );
        let opened = prepare_applied_rows(&[state], vec![0], vec![7]).unwrap();
        assert!(opened[0].1.swap_available());
        let states: Vec<_> = opened.into_iter().map(|(_, state)| state).collect();
        let swapped = prepare_applied_rows(&states, vec![0], vec![swap_code]).unwrap();
        assert!(swapped[0].1.swapped());
        assert_eq!(swapped[0].1.to_move(), Player::Zero);
        assert!(prepare_applied_rows(&states, vec![0], vec![swap_code + 1]).is_err());
    }

    #[test]
    fn trajectory_metadata_exposes_terminal_value_and_residuals() {
        let board = Arc::new(Board::new(4).unwrap());
        let full = full_state(board);
        let state_data = pack_states(std::slice::from_ref(&full));
        assert_eq!(state_data.stones_placed, [50]);
        assert_eq!(state_data.moves_left, [1]);
        assert_eq!(state_data.mid_turn, [true]);
        assert_eq!(state_data.turn_size, [2]);
        assert_eq!(state_data.current_turn_total, [2]);
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
    fn mixed_batch_completes_only_clinched_rows_to_the_loser_filled_board() {
        Python::initialize();
        Python::attach(|py| {
            let board = Arc::new(Board::new(4).unwrap());
            let active = GameState::new(Arc::clone(&board));
            let mut clinched = GameState::new(Arc::clone(&board));
            for node in 0..49 {
                clinched.apply(Action::Place(node)).unwrap();
            }
            let full = full_state(Arc::clone(&board));
            let full_key = full.key();
            let mut batch = PyStateBatch {
                board,
                states: vec![active, clinched, full],
            };

            let result = batch.complete_clinches(py).unwrap();

            assert_eq!(result.clinched, [false, true, false]);
            assert_eq!(result.winner, [-1, 1, -1]);
            assert_eq!(result.empty_nodes, [0, 1, 0]);
            assert_eq!(result.last_move, [-1, 48, -1]);
            assert_eq!(result.turn_count, [0, 25, 0]);
            assert!(!batch.states[0].is_terminal());
            assert!(batch.states[1].is_terminal());
            assert!(batch.states[1].stones_for(Player::Zero).contains(49));
            assert_eq!(
                ScoringScratch::default()
                    .score_state(&batch.states[1])
                    .leader,
                Some(Player::One)
            );
            assert_eq!(batch.states[2].key(), full_key);
        });
    }

    #[test]
    fn one_and_many_threads_produce_identical_actor_results() {
        Python::initialize();
        let single = parallel_snapshot(1);
        let many = parallel_snapshot(4);
        assert_eq!(single, many);
    }

    #[test]
    fn per_root_budgets_and_seat_advantages_flow_through_the_batch() {
        Python::initialize();
        Python::attach(|py| {
            let board = Arc::new(Board::new(4).unwrap());
            let mut states = Vec::new();
            for opening in 0..3_u16 {
                let mut state = GameState::new(Arc::clone(&board));
                state.apply(Action::Place(opening)).unwrap();
                states.push(state);
            }
            let batch = PyStateBatch {
                board: Arc::clone(&board),
                states,
            };
            let mut search = PySearchBatch {
                trees: batch.states.iter().cloned().map(SearchTree::new).collect(),
                schedulers: None,
                config: RootSearchConfig::deterministic(6, 4, GumbelParameters::PAPER, 0x77),
                budgets: vec![6, 12, 3],
                pda_by_seat: vec![[0, 0], [2, -2], [-1, 1]],
                pending: Vec::new(),
            };
            let roots = search.root_requests(py).unwrap();
            // Every root has player 1 to move: seat-1 advantages apply.
            assert_eq!(roots.pda, [0, -2, 1]);
            let root_globals = f32s(&roots.features.buffers.global_features);
            assert!((root_globals[GLOBAL_FEATURE_DIM + 24] + 2.0 / 3.0).abs() <= 1.0e-6);
            submit_uniform_roots(&mut search, py, roots);
            while !search.is_done() {
                let leaves = search.next_requests(py).unwrap();
                if !leaves.tokens.is_empty() {
                    for (tree_index, to_move) in
                        leaves.tree_indices.iter().zip(&leaves.states.to_move)
                    {
                        let expected = search.pda_by_seat[*tree_index][usize::from(*to_move)];
                        let row = leaves
                            .tree_indices
                            .iter()
                            .position(|index| index == tree_index)
                            .unwrap();
                        assert_eq!(leaves.pda[row], expected);
                    }
                    submit_uniform_leaves(&mut search, py, leaves);
                }
            }
            let results = search.results(py).unwrap();
            let visits: Vec<u32> = (0..3)
                .map(|row| {
                    results.visits[results.action_offsets[row]..results.action_offsets[row + 1]]
                        .iter()
                        .sum()
                })
                .collect();
            assert_eq!(visits, [6, 12, 3]);
            assert_eq!(results.root_values.len(), 3);
            assert!(results.root_values.iter().all(|value| value.abs() <= 1.0));
        });
    }

    fn semantic_round_trip(state: &GameState) -> GameState {
        let packed = pack_states(std::slice::from_ref(state));
        decode_semantic_states(state.shared_board(), packed.semantic_rows())
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
                    budgets: vec![9; transformed.states.len()],
                    pda_by_seat: vec![[0, 0]; transformed.states.len()],
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
                    root_values: results.root_values,
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
