use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::{BitBoard, Board, NodeId};

/// Largest supported handicap: consecutive opening placements by player 0.
pub const MAX_HANDICAP: u8 = 9;
/// Capacity of every retained placement list: `max(handicap, turn size)`.
pub const MAX_TURN_PLACEMENTS: usize = MAX_HANDICAP as usize;

/// One of the two fixed stone colors.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Player {
    /// Opening player.
    Zero = 0,
    /// Second player.
    One = 1,
}

impl Player {
    /// Numeric player index.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The other player.
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::Zero => Self::One,
            Self::One => Self::Zero,
        }
    }
}

/// Turn protocol after the opening turn.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Mode {
    /// Classic *Star: one placement every turn.
    Classic = 0,
    /// Double *Star: two placements per turn after the opening.
    Double = 1,
}

impl Mode {
    /// Placements per completed non-opening turn.
    #[must_use]
    pub const fn turn_size(self) -> u8 {
        match self {
            Self::Classic => 1,
            Self::Double => 2,
        }
    }

    /// Stable numeric index used by foreign interfaces.
    #[must_use]
    pub const fn index(self) -> u8 {
        self as u8
    }

    /// Decodes the stable numeric index.
    #[must_use]
    pub const fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::Classic),
            1 => Some(Self::Double),
            _ => None,
        }
    }

    /// Canonical lowercase name shared with the web client.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Double => "double",
        }
    }

    /// Parses the canonical lowercase name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "classic" => Some(Self::Classic),
            "double" => Some(Self::Double),
            _ => None,
        }
    }
}

/// Rule variant of one game: turn protocol, handicap, and pie availability.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Variant {
    mode: Mode,
    handicap: u8,
    pie: bool,
}

impl Variant {
    /// Standard Double *Star: one opening stone, two per turn, no pie.
    pub const STANDARD: Self = Self {
        mode: Mode::Double,
        handicap: 1,
        pie: false,
    };

    /// Validates a variant. Handicap is `1..=9`; pie requires handicap 1.
    pub const fn new(mode: Mode, handicap: u8, pie: bool) -> Result<Self, GameError> {
        if handicap == 0 || handicap > MAX_HANDICAP {
            return Err(GameError::InvalidHandicap(handicap));
        }
        if pie && handicap != 1 {
            return Err(GameError::HandicapExcludesPie);
        }
        Ok(Self {
            mode,
            handicap,
            pie,
        })
    }

    /// Turn protocol.
    #[must_use]
    pub const fn mode(self) -> Mode {
        self.mode
    }

    /// Consecutive opening placements by player 0.
    #[must_use]
    pub const fn handicap(self) -> u8 {
        self.handicap
    }

    /// Whether player 1 may swap after the opening turn.
    #[must_use]
    pub const fn pie(self) -> bool {
        self.pie
    }

    /// Placements per completed non-opening turn.
    #[must_use]
    pub const fn turn_size(self) -> u8 {
        self.mode.turn_size()
    }

    /// Whether this is the standard no-pie Double *Star game.
    #[must_use]
    pub const fn is_standard(self) -> bool {
        matches!(self.mode, Mode::Double) && self.handicap == 1 && !self.pie
    }
}

impl Default for Variant {
    fn default() -> Self {
        Self::STANDARD
    }
}

/// A single search/game transition.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Action {
    /// Place one stone at the dense node id.
    Place(NodeId),
    /// Pie-rule swap: player 1 takes the opening stone; player 0 moves next.
    Swap,
}

impl Action {
    /// Stable integer encoding used by foreign interfaces.
    ///
    /// Placements encode as their node id; the swap encodes as `node_count`,
    /// one past the last node, so the nodes-only layout is preserved.
    #[must_use]
    pub const fn code(self, node_count: u16) -> i32 {
        match self {
            Self::Place(node) => node as i32,
            Self::Swap => node_count as i32,
        }
    }

    /// Decodes a foreign-interface action code for a board of `node_count`.
    pub fn from_code(code: i32, node_count: u16) -> Result<Self, GameError> {
        match NodeId::try_from(code) {
            Ok(node) if node < node_count => Ok(Self::Place(node)),
            Ok(node) if node == node_count => Ok(Self::Swap),
            _ => Err(GameError::InvalidActionCode(code)),
        }
    }

    /// Native model index: node `u` maps exactly to `u`. The swap has none.
    pub fn native_index(self, board: &Board) -> Result<usize, GameError> {
        match self {
            Self::Place(node) if node < board.node_count() => Ok(usize::from(node)),
            Self::Place(node) => Err(GameError::NodeOutOfBounds(node)),
            Self::Swap => Err(GameError::SwapHasNoNativeIndex),
        }
    }

    /// Decodes the native nodes-only model layout.
    pub fn from_native_index(index: usize, board: &Board) -> Result<Self, GameError> {
        if index < usize::from(board.node_count()) {
            Ok(Self::Place(
                NodeId::try_from(index).expect("board node ids fit in u16"),
            ))
        } else {
            Err(GameError::InvalidNativeActionIndex(index))
        }
    }

    /// The placed node, if this is a placement.
    #[must_use]
    pub const fn node(self) -> Option<NodeId> {
        match self {
            Self::Place(node) => Some(node),
            Self::Swap => None,
        }
    }
}

/// Placement mask for all legal atomic placement actions.
///
/// The swap is never part of this mask: it is a value decision taken by the
/// game driver, never a policy output. Query [`GameState::swap_available`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegalActions {
    /// Empty nodes on which a stone may be placed.
    pub placements: BitBoard,
}

impl LegalActions {
    /// Number of legal atomic placements.
    #[must_use]
    pub fn len(self) -> usize {
        usize::from(self.placements.count())
    }

    /// Whether no legal placement exists.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.placements.is_empty()
    }

    /// Materializes placements in ascending node-id order.
    #[must_use]
    pub fn to_vec(self) -> Vec<Action> {
        self.placements.iter().map(Action::Place).collect()
    }
}

/// Semantic state key used by transposition tables and feature encoders.
///
/// Every field the network may observe is part of the key. Placement sets are
/// order-free, so `{a, b}` and `{b, a}` within one turn share a key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StateKey {
    /// Board size.
    pub rings: u8,
    /// Fixed player bitboards.
    pub stones: [BitBoard; 2],
    /// Player who performs the next atomic action.
    pub to_move: Player,
    /// Placements still available in the current turn.
    pub moves_left: u8,
    /// Whether the opening turn is active.
    pub opening: bool,
    /// Terminal marker, included defensively even though it is derivable.
    pub terminal: bool,
    /// Turn protocol.
    pub mode: Mode,
    /// Consecutive opening placements by player 0.
    pub handicap: u8,
    /// Empty board of a pie game: the opener must expect an optimal swap.
    pub pie_pending: bool,
    /// Whether player 1 may swap right now.
    pub swap_available: bool,
    /// Placements made so far in the unfinished current turn.
    pub current_turn: BitBoard,
    /// The most recently completed turn (the opponent's).
    pub previous_turn: BitBoard,
    /// The completed turn before that (the current player's).
    pub own_previous_turn: BitBoard,
    /// Stones placed during the opening phase.
    pub handicap_stones: BitBoard,
}

/// Errors from state construction or transitions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GameError {
    /// No action may follow a terminal state.
    GameOver,
    /// A placement references a node outside this board.
    NodeOutOfBounds(NodeId),
    /// A placement references an occupied node.
    Occupied(NodeId),
    /// Foreign action code is neither a node id nor the swap code.
    InvalidActionCode(i32),
    /// Native action index is outside the nodes-only layout.
    InvalidNativeActionIndex(usize),
    /// The swap is not a node and has no slot in the nodes-only layout.
    SwapHasNoNativeIndex,
    /// The swap is only legal immediately after the opening turn of a pie game.
    SwapUnavailable,
    /// Handicap must be in `1..=9`.
    InvalidHandicap(u8),
    /// A handicap game cannot also use the pie rule.
    HandicapExcludesPie,
    /// Imported bitboards overlap.
    OverlappingStones,
    /// Imported bitboards contain nodes outside this board.
    StonesOutsideBoard,
    /// Imported turn metadata cannot occur under this variant.
    InvalidTurnMetadata,
    /// Imported placement history disagrees with the stones or the turn.
    InvalidHistory,
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GameOver => f.write_str("the game is already over"),
            Self::NodeOutOfBounds(node) => write!(f, "node {node} is outside the board"),
            Self::Occupied(node) => write!(f, "node {node} is occupied"),
            Self::InvalidActionCode(code) => write!(f, "invalid action code {code}"),
            Self::InvalidNativeActionIndex(index) => {
                write!(f, "invalid native action index {index}")
            }
            Self::SwapHasNoNativeIndex => f.write_str("the swap has no native action index"),
            Self::SwapUnavailable => f.write_str("the pie swap is not available"),
            Self::InvalidHandicap(handicap) => {
                write!(f, "handicap must be in 1..=9, got {handicap}")
            }
            Self::HandicapExcludesPie => f.write_str("handicap games cannot use the pie rule"),
            Self::OverlappingStones => f.write_str("player bitboards overlap"),
            Self::StonesOutsideBoard => f.write_str("player bitboards contain off-board nodes"),
            Self::InvalidTurnMetadata => f.write_str("invalid *Star turn metadata"),
            Self::InvalidHistory => f.write_str("invalid placement history"),
        }
    }
}

impl Error for GameError {}

/// Metadata describing one successfully applied atomic action.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Transition {
    /// Applied action.
    pub action: Action,
    /// Acting player.
    pub player_before: Player,
    /// Player after the transition.
    pub player_after: Player,
    /// Whether control changed players.
    pub turn_ended: bool,
    /// Whether this action ended the game.
    pub terminal: bool,
}

/// Fixed-capacity ordered placement list.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Placements {
    nodes: [NodeId; MAX_TURN_PLACEMENTS],
    len: u8,
    set: BitBoard,
}

impl Placements {
    const fn empty() -> Self {
        Self {
            nodes: [0; MAX_TURN_PLACEMENTS],
            len: 0,
            set: BitBoard::empty(),
        }
    }

    fn push(&mut self, node: NodeId) {
        debug_assert!(usize::from(self.len) < MAX_TURN_PLACEMENTS);
        self.nodes[usize::from(self.len)] = node;
        self.len += 1;
        self.set.insert(node);
    }

    fn as_slice(&self) -> &[NodeId] {
        &self.nodes[..usize::from(self.len)]
    }

    fn from_slice(nodes: &[NodeId]) -> Self {
        let mut placements = Self::empty();
        for &node in nodes {
            placements.push(node);
        }
        placements
    }

    fn from_set(set: BitBoard) -> Self {
        let mut placements = Self::empty();
        for node in set {
            placements.push(node);
        }
        placements
    }

    fn map(&self, map: impl Fn(NodeId) -> NodeId) -> Self {
        let mut mapped = Self::empty();
        for &node in self.as_slice() {
            mapped.push(map(node));
        }
        mapped
    }
}

/// Exact reversible snapshot for one mutable transition.
#[derive(Clone, Copy, Debug)]
pub struct Undo {
    stones: [BitBoard; 2],
    to_move: Player,
    moves_left: u8,
    opening: bool,
    terminal: bool,
    swap_available: bool,
    swapped: bool,
    stones_placed: u16,
    last_move: Option<NodeId>,
    current_turn: Placements,
    previous_turn: Placements,
    own_previous_turn: Placements,
    handicap_stones: BitBoard,
    turn_count: u32,
}

/// Semantic fields needed to rebuild a state without replaying a game.
///
/// History sets may be empty when the importer does not know them; the state
/// then behaves as if no placement history were retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateParts {
    /// Rule variant.
    pub variant: Variant,
    /// Fixed player bitboards.
    pub stones: [BitBoard; 2],
    /// Player who performs the next atomic action.
    pub to_move: Player,
    /// Placements still available in the current turn.
    pub moves_left: u8,
    /// Whether the opening turn is active.
    pub opening: bool,
    /// Whether player 1 may swap right now.
    pub swap_available: bool,
    /// Whether the pie swap was taken earlier in this game.
    pub swapped: bool,
    /// Placements made so far in the unfinished current turn.
    pub current_turn: BitBoard,
    /// The most recently completed turn (the opponent's).
    pub previous_turn: BitBoard,
    /// The completed turn before that (the current player's).
    pub own_previous_turn: BitBoard,
    /// Stones placed during the opening phase.
    pub handicap_stones: BitBoard,
}

impl StateParts {
    /// Standard-variant parts with no retained history.
    #[must_use]
    pub const fn standard(
        stones: [BitBoard; 2],
        to_move: Player,
        moves_left: u8,
        opening: bool,
    ) -> Self {
        Self {
            variant: Variant::STANDARD,
            stones,
            to_move,
            moves_left,
            opening,
            swap_available: false,
            swapped: false,
            current_turn: BitBoard::empty(),
            previous_turn: BitBoard::empty(),
            own_previous_turn: BitBoard::empty(),
            handicap_stones: BitBoard::empty(),
        }
    }
}

/// Complete *Star state for every supported rule variant.
#[derive(Clone, Debug)]
pub struct GameState {
    board: Arc<Board>,
    variant: Variant,
    stones: [BitBoard; 2],
    to_move: Player,
    moves_left: u8,
    opening: bool,
    terminal: bool,
    swap_available: bool,
    swapped: bool,
    stones_placed: u16,
    last_move: Option<NodeId>,
    current_turn: Placements,
    previous_turn: Placements,
    own_previous_turn: Placements,
    handicap_stones: BitBoard,
    turn_count: u32,
}

impl GameState {
    /// Creates the empty standard Double *Star state for a prebuilt board.
    #[must_use]
    pub fn new(board: Arc<Board>) -> Self {
        Self::with_variant(board, Variant::STANDARD)
    }

    /// Creates the empty state of one rule variant.
    #[must_use]
    pub fn with_variant(board: Arc<Board>, variant: Variant) -> Self {
        Self {
            board,
            variant,
            stones: [BitBoard::empty(); 2],
            to_move: Player::Zero,
            moves_left: variant.handicap(),
            opening: true,
            terminal: false,
            swap_available: false,
            swapped: false,
            stones_placed: 0,
            last_move: None,
            current_turn: Placements::empty(),
            previous_turn: Placements::empty(),
            own_previous_turn: Placements::empty(),
            handicap_stones: BitBoard::empty(),
            turn_count: 0,
        }
    }

    /// Reconstructs a standard-variant state without placement history.
    pub fn from_standard_parts(
        board: Arc<Board>,
        stones: [BitBoard; 2],
        to_move: Player,
        moves_left: u8,
        opening: bool,
    ) -> Result<Self, GameError> {
        Self::from_parts(
            board,
            StateParts::standard(stones, to_move, moves_left, opening),
        )
    }

    /// Reconstructs a semantically complete state from packed data.
    pub fn from_parts(board: Arc<Board>, parts: StateParts) -> Result<Self, GameError> {
        let StateParts {
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
        } = parts;
        if !stones[0].intersection(stones[1]).is_empty() {
            return Err(GameError::OverlappingStones);
        }
        let board_mask = board.node_mask();
        if !stones[0].difference(board_mask).is_empty()
            || !stones[1].difference(board_mask).is_empty()
        {
            return Err(GameError::StonesOutsideBoard);
        }
        let occupied = stones[0].union(stones[1]);
        let stones_placed = occupied.count();
        let board_full = stones_placed == board.node_count();
        let terminal = board_full;
        let turn_size = variant.turn_size();
        let handicap = variant.handicap();

        if opening {
            if to_move != Player::Zero
                || !stones[1].is_empty()
                || moves_left == 0
                || moves_left > handicap
                || stones[0].count() != u16::from(handicap - moves_left)
                || swap_available
                || swapped
                || terminal
            {
                return Err(GameError::InvalidTurnMetadata);
            }
        } else if board_full {
            if moves_left >= turn_size {
                return Err(GameError::InvalidTurnMetadata);
            }
        } else if moves_left == 0 || moves_left > turn_size {
            return Err(GameError::InvalidTurnMetadata);
        }
        if (swap_available || swapped) && !variant.pie() {
            return Err(GameError::InvalidTurnMetadata);
        }
        if swap_available
            && (swapped
                || opening
                || to_move != Player::One
                || moves_left != turn_size
                || stones[0].count() != 1
                || !stones[1].is_empty())
        {
            return Err(GameError::InvalidTurnMetadata);
        }

        // History sets: containment, disjointness, and current-turn size.
        // During the opening the history is derivable (every stone is player
        // 0's current turn and a handicap stone), so it is filled in when the
        // importer omitted it and must agree when it was supplied.
        let own = stones[to_move.index()];
        let opponent = stones[to_move.opponent().index()];
        let (current_turn, handicap_stones) = if opening {
            if (!current_turn.is_empty() && current_turn != stones[0])
                || (!handicap_stones.is_empty() && handicap_stones != stones[0])
                || !previous_turn.is_empty()
                || !own_previous_turn.is_empty()
            {
                return Err(GameError::InvalidHistory);
            }
            (stones[0], stones[0])
        } else {
            (current_turn, handicap_stones)
        };
        let expected_current = if opening {
            u16::from(handicap - moves_left)
        } else {
            u16::from(turn_size - moves_left)
        };
        if !current_turn.difference(own).is_empty()
            || !own_previous_turn.difference(own).is_empty()
            || !previous_turn.difference(opponent).is_empty()
            || !handicap_stones.difference(occupied).is_empty()
            || !current_turn.intersection(own_previous_turn).is_empty()
            || current_turn.count() > MAX_TURN_PLACEMENTS as u16
            || previous_turn.count() > MAX_TURN_PLACEMENTS as u16
            || own_previous_turn.count() > MAX_TURN_PLACEMENTS as u16
            || (!current_turn.is_empty() && current_turn.count() != expected_current)
        {
            return Err(GameError::InvalidHistory);
        }
        let current_turn = Placements::from_set(current_turn);
        let last_move = current_turn.as_slice().last().copied();

        Ok(Self {
            board,
            variant,
            stones,
            to_move,
            moves_left,
            opening,
            terminal,
            swap_available,
            swapped,
            stones_placed,
            last_move,
            current_turn,
            previous_turn: Placements::from_set(previous_turn),
            own_previous_turn: Placements::from_set(own_previous_turn),
            handicap_stones,
            turn_count: u32::from(!opening),
        })
    }

    /// Immutable board topology.
    #[must_use]
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Shared immutable board topology.
    #[must_use]
    pub fn shared_board(&self) -> Arc<Board> {
        Arc::clone(&self.board)
    }

    /// Rule variant of this game.
    #[must_use]
    pub const fn variant(&self) -> Variant {
        self.variant
    }

    /// Turn protocol.
    #[must_use]
    pub const fn mode(&self) -> Mode {
        self.variant.mode()
    }

    /// Consecutive opening placements by player 0.
    #[must_use]
    pub const fn handicap(&self) -> u8 {
        self.variant.handicap()
    }

    /// Whether the pie rule is in effect for this game.
    #[must_use]
    pub const fn pie(&self) -> bool {
        self.variant.pie()
    }

    /// Placements per completed non-opening turn.
    #[must_use]
    pub const fn turn_size(&self) -> u8 {
        self.variant.turn_size()
    }

    /// Total placements of the turn in progress: handicap during the opening.
    #[must_use]
    pub const fn current_turn_total(&self) -> u8 {
        if self.opening {
            self.variant.handicap()
        } else {
            self.variant.turn_size()
        }
    }

    /// Both fixed player bitboards.
    #[must_use]
    pub const fn stones(&self) -> [BitBoard; 2] {
        self.stones
    }

    /// Bitboard for one player.
    #[must_use]
    pub const fn stones_for(&self, player: Player) -> BitBoard {
        self.stones[player.index()]
    }

    /// Occupied-node mask.
    #[must_use]
    pub fn occupied(&self) -> BitBoard {
        self.stones[0].union(self.stones[1])
    }

    /// Stone owner at a node.
    #[must_use]
    pub fn stone_at(&self, node: NodeId) -> Option<Player> {
        if self.stones[0].contains(node) {
            Some(Player::Zero)
        } else if self.stones[1].contains(node) {
            Some(Player::One)
        } else {
            None
        }
    }

    /// Player who takes the next atomic action.
    #[must_use]
    pub const fn to_move(&self) -> Player {
        self.to_move
    }

    /// Placements remaining in this turn.
    #[must_use]
    pub const fn moves_left(&self) -> u8 {
        self.moves_left
    }

    /// Whether the opening turn is active.
    #[must_use]
    pub const fn is_opening(&self) -> bool {
        self.opening
    }

    /// Empty board of a pie game, before the opener's stone.
    #[must_use]
    pub const fn is_pie_pending(&self) -> bool {
        self.variant.pie() && self.opening && self.stones_placed == 0
    }

    /// Whether player 1 may swap right now.
    #[must_use]
    pub const fn swap_available(&self) -> bool {
        self.swap_available
    }

    /// Whether the pie swap was taken earlier in this game.
    #[must_use]
    pub const fn swapped(&self) -> bool {
        self.swapped
    }

    /// Whether no further action is legal.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// Number of placed stones.
    #[must_use]
    pub const fn stones_placed(&self) -> u16 {
        self.stones_placed
    }

    /// Most recent placement, if tracked by this state.
    #[must_use]
    pub const fn last_move(&self) -> Option<NodeId> {
        self.last_move
    }

    /// Placements retained in the current unfinished turn, in order.
    #[must_use]
    pub fn current_turn_moves(&self) -> &[NodeId] {
        self.current_turn.as_slice()
    }

    /// The most recently completed turn's placements, in order.
    #[must_use]
    pub fn previous_turn_moves(&self) -> &[NodeId] {
        self.previous_turn.as_slice()
    }

    /// The completed turn before the previous one, in order.
    #[must_use]
    pub fn own_previous_turn_moves(&self) -> &[NodeId] {
        self.own_previous_turn.as_slice()
    }

    /// Placements of the current unfinished turn as a set.
    #[must_use]
    pub const fn current_turn_set(&self) -> BitBoard {
        self.current_turn.set
    }

    /// The most recently completed turn as a set.
    #[must_use]
    pub const fn previous_turn_set(&self) -> BitBoard {
        self.previous_turn.set
    }

    /// The completed turn before the previous one as a set.
    #[must_use]
    pub const fn own_previous_turn_set(&self) -> BitBoard {
        self.own_previous_turn.set
    }

    /// Stones placed during the opening phase.
    #[must_use]
    pub const fn handicap_stones(&self) -> BitBoard {
        self.handicap_stones
    }

    /// Whether at least one placement has been made and another remains.
    #[must_use]
    pub const fn is_mid_turn(&self) -> bool {
        self.current_turn.len > 0 && self.moves_left > 0
    }

    /// Number of completed turns.
    #[must_use]
    pub const fn turn_count(&self) -> u32 {
        self.turn_count
    }

    /// Semantic key for exact transposition reuse and feature encoding.
    #[must_use]
    pub fn key(&self) -> StateKey {
        StateKey {
            rings: self.board.rings(),
            stones: self.stones,
            to_move: self.to_move,
            moves_left: self.moves_left,
            opening: self.opening,
            terminal: self.terminal,
            mode: self.variant.mode(),
            handicap: self.variant.handicap(),
            pie_pending: self.is_pie_pending(),
            swap_available: self.swap_available,
            current_turn: self.current_turn.set,
            previous_turn: self.previous_turn.set,
            own_previous_turn: self.own_previous_turn.set,
            handicap_stones: self.handicap_stones,
        }
    }

    /// Stable deterministic Zobrist-style hash of [`Self::key`].
    #[must_use]
    pub fn hash64(&self) -> u64 {
        let mut hash = splitmix64(0xd0ab_1e5a_7a12_0000 ^ u64::from(self.board.rings()));
        for player in [Player::Zero, Player::One] {
            for node in self.stones_for(player) {
                let index = (player as u64) * 448 + u64::from(node);
                hash ^= splitmix64(0x51a7_e000_0000_0000 ^ index);
            }
        }
        for (salt, set) in [
            (0x5a00_0000_0000_0000_u64, self.current_turn.set),
            (0x5b00_0000_0000_0000_u64, self.previous_turn.set),
            (0x5c00_0000_0000_0000_u64, self.own_previous_turn.set),
            (0x5d00_0000_0000_0000_u64, self.handicap_stones),
        ] {
            for node in set {
                hash ^= splitmix64(salt ^ u64::from(node));
            }
        }
        hash ^= splitmix64(0x7000_0000_0000_0000 ^ self.to_move as u64);
        hash ^= splitmix64(0x7100_0000_0000_0000 ^ u64::from(self.moves_left));
        hash ^= splitmix64(0x7200_0000_0000_0000 ^ u64::from(self.opening));
        hash ^= splitmix64(0x7400_0000_0000_0000 ^ u64::from(self.terminal));
        hash ^= splitmix64(0x7500_0000_0000_0000 ^ u64::from(self.variant.mode().index()));
        hash ^= splitmix64(0x7600_0000_0000_0000 ^ u64::from(self.variant.handicap()));
        hash ^= splitmix64(0x7700_0000_0000_0000 ^ u64::from(self.is_pie_pending()));
        hash ^= splitmix64(0x7800_0000_0000_0000 ^ u64::from(self.swap_available));
        hash
    }

    /// Applies one node relabeling to every spatial field.
    ///
    /// Turn metadata, the variant, and swap flags are preserved; every stone
    /// set, placement list, and the last move are mapped through `map`.
    #[must_use]
    pub fn map_nodes(&self, map: impl Fn(NodeId) -> NodeId) -> Self {
        let map_set = |set: BitBoard| {
            let mut mapped = BitBoard::empty();
            for node in set {
                mapped.insert(map(node));
            }
            mapped
        };
        let mut transformed = self.clone();
        transformed.stones = [map_set(self.stones[0]), map_set(self.stones[1])];
        transformed.last_move = self.last_move.map(&map);
        transformed.current_turn = self.current_turn.map(&map);
        transformed.previous_turn = self.previous_turn.map(&map);
        transformed.own_previous_turn = self.own_previous_turn.map(&map);
        transformed.handicap_stones = map_set(self.handicap_stones);
        transformed
    }

    /// Legal atomic placements. The swap is reported by [`Self::swap_available`].
    #[must_use]
    pub fn legal_actions(&self) -> LegalActions {
        if self.terminal {
            LegalActions {
                placements: BitBoard::empty(),
            }
        } else {
            LegalActions {
                placements: self.board.node_mask().difference(self.occupied()),
            }
        }
    }

    /// Tests one action without mutating the state.
    #[must_use]
    pub fn is_legal(&self, action: Action) -> bool {
        if self.terminal {
            return false;
        }
        match action {
            Action::Place(node) => {
                node < self.board.node_count() && !self.occupied().contains(node)
            }
            Action::Swap => self.swap_available,
        }
    }

    /// Applies one atomic action.
    pub fn apply(&mut self, action: Action) -> Result<Transition, GameError> {
        self.apply_internal(action)
    }

    /// Applies one action and returns an exact undo snapshot.
    pub fn apply_reversible(&mut self, action: Action) -> Result<(Transition, Undo), GameError> {
        let undo = Undo {
            stones: self.stones,
            to_move: self.to_move,
            moves_left: self.moves_left,
            opening: self.opening,
            terminal: self.terminal,
            swap_available: self.swap_available,
            swapped: self.swapped,
            stones_placed: self.stones_placed,
            last_move: self.last_move,
            current_turn: self.current_turn,
            previous_turn: self.previous_turn,
            own_previous_turn: self.own_previous_turn,
            handicap_stones: self.handicap_stones,
            turn_count: self.turn_count,
        };
        let transition = self.apply_internal(action)?;
        Ok((transition, undo))
    }

    /// Restores a snapshot produced by [`Self::apply_reversible`].
    pub fn undo(&mut self, undo: Undo) {
        self.stones = undo.stones;
        self.to_move = undo.to_move;
        self.moves_left = undo.moves_left;
        self.opening = undo.opening;
        self.terminal = undo.terminal;
        self.swap_available = undo.swap_available;
        self.swapped = undo.swapped;
        self.stones_placed = undo.stones_placed;
        self.last_move = undo.last_move;
        self.current_turn = undo.current_turn;
        self.previous_turn = undo.previous_turn;
        self.own_previous_turn = undo.own_previous_turn;
        self.handicap_stones = undo.handicap_stones;
        self.turn_count = undo.turn_count;
    }

    fn apply_internal(&mut self, action: Action) -> Result<Transition, GameError> {
        if self.terminal {
            return Err(GameError::GameOver);
        }
        let player_before = self.to_move;
        match action {
            Action::Swap => {
                if !self.swap_available {
                    return Err(GameError::SwapUnavailable);
                }
                // Player 1 takes the opener's color: every stone becomes
                // player 1's and player 0 moves next with a full turn. The
                // opening stone remains the most recent completed turn so the
                // encoded inputs equal the unswapped position with colors
                // exchanged.
                self.stones = [BitBoard::empty(), self.stones[0].union(self.stones[1])];
                self.swapped = true;
                self.swap_available = false;
                self.to_move = Player::Zero;
                self.turn_count += 1;
                self.moves_left = self.variant.turn_size();
                self.current_turn = Placements::empty();
            }
            Action::Place(node) => {
                if node >= self.board.node_count() {
                    return Err(GameError::NodeOutOfBounds(node));
                }
                if self.occupied().contains(node) {
                    return Err(GameError::Occupied(node));
                }
                self.stones[self.to_move.index()].insert(node);
                self.stones_placed += 1;
                self.last_move = Some(node);
                self.current_turn.push(node);
                if self.opening {
                    self.handicap_stones.insert(node);
                }
                self.moves_left -= 1;
                self.swap_available = false;

                if self.stones_placed == self.board.node_count() {
                    self.terminal = true;
                } else if self.moves_left == 0 {
                    self.end_turn();
                }
            }
        }

        Ok(Transition {
            action,
            player_before,
            player_after: self.to_move,
            turn_ended: self.to_move != player_before,
            terminal: self.terminal,
        })
    }

    fn end_turn(&mut self) {
        self.to_move = self.to_move.opponent();
        self.turn_count += 1;
        self.own_previous_turn = self.previous_turn;
        self.previous_turn = self.current_turn;
        self.current_turn = Placements::empty();
        self.moves_left = self.variant.turn_size();
        self.opening = false;
        self.swap_available = self.variant.pie() && self.turn_count == 1 && !self.swapped;
    }
}

impl GameState {
    /// Rebuilds the retained history lists from ordered slices.
    ///
    /// Used by importers that know placement order; sets alone lose it.
    pub fn with_ordered_history(
        mut self,
        current_turn: &[NodeId],
        previous_turn: &[NodeId],
        own_previous_turn: &[NodeId],
    ) -> Result<Self, GameError> {
        let as_set = |nodes: &[NodeId]| {
            let mut set = BitBoard::empty();
            for &node in nodes {
                set.insert(node);
            }
            set
        };
        if as_set(current_turn) != self.current_turn.set
            || as_set(previous_turn) != self.previous_turn.set
            || as_set(own_previous_turn) != self.own_previous_turn.set
        {
            return Err(GameError::InvalidHistory);
        }
        self.current_turn = Placements::from_slice(current_turn);
        self.previous_turn = Placements::from_slice(previous_turn);
        self.own_previous_turn = Placements::from_slice(own_previous_turn);
        self.last_move = current_turn
            .last()
            .or(previous_turn.last())
            .copied()
            .or(self.last_move);
        Ok(self)
    }
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
