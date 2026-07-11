use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::{BitBoard, Board, NodeId};

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

/// A single search/game transition.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Action {
    /// Place one stone at the dense node id.
    Place(NodeId),
}

impl Action {
    /// Stable integer encoding used by foreign interfaces.
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::Place(node) => node as i32,
        }
    }

    /// Decodes a foreign-interface action code.
    pub fn from_code(code: i32) -> Result<Self, GameError> {
        if let Ok(node) = NodeId::try_from(code) {
            Ok(Self::Place(node))
        } else {
            Err(GameError::InvalidActionCode(code))
        }
    }

    /// Native model index: node `u` maps exactly to `u`.
    pub fn native_index(self, board: &Board) -> Result<usize, GameError> {
        match self {
            Self::Place(node) if node < board.node_count() => Ok(usize::from(node)),
            Self::Place(node) => Err(GameError::NodeOutOfBounds(node)),
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
}

/// Placement mask for all legal atomic actions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegalActions {
    /// Empty nodes on which a stone may be placed.
    pub placements: BitBoard,
}

impl LegalActions {
    /// Number of legal atomic actions.
    #[must_use]
    pub fn len(self) -> usize {
        usize::from(self.placements.count())
    }

    /// Whether no legal action exists.
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

/// Semantic state key used by transposition tables.
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
    /// Whether the special one-stone opening turn is active.
    pub opening: bool,
    /// Terminal marker, included defensively even though it is derivable.
    pub terminal: bool,
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
    /// Foreign action code is not a node id.
    InvalidActionCode(i32),
    /// Native action index is outside the nodes-only layout.
    InvalidNativeActionIndex(usize),
    /// Imported bitboards overlap.
    OverlappingStones,
    /// Imported bitboards contain nodes outside this board.
    StonesOutsideBoard,
    /// Imported turn metadata cannot occur in Double *Star.
    InvalidTurnMetadata,
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
            Self::OverlappingStones => f.write_str("player bitboards overlap"),
            Self::StonesOutsideBoard => f.write_str("player bitboards contain off-board nodes"),
            Self::InvalidTurnMetadata => f.write_str("invalid Double *Star turn metadata"),
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

/// Exact reversible snapshot for one mutable transition.
#[derive(Clone, Copy, Debug)]
pub struct Undo {
    stones: [BitBoard; 2],
    to_move: Player,
    moves_left: u8,
    opening: bool,
    terminal: bool,
    stones_placed: u16,
    last_move: Option<NodeId>,
    current_turn_moves: [NodeId; 2],
    current_turn_len: u8,
    turn_count: u32,
}

/// Complete Double *Star state. Pie and classic mode are intentionally absent.
#[derive(Clone, Debug)]
pub struct GameState {
    board: Arc<Board>,
    stones: [BitBoard; 2],
    to_move: Player,
    moves_left: u8,
    opening: bool,
    terminal: bool,
    stones_placed: u16,
    last_move: Option<NodeId>,
    current_turn_moves: [NodeId; 2],
    current_turn_len: u8,
    turn_count: u32,
}

impl GameState {
    /// Creates the empty state for a prebuilt board.
    #[must_use]
    pub fn new(board: Arc<Board>) -> Self {
        Self {
            board,
            stones: [BitBoard::empty(); 2],
            to_move: Player::Zero,
            moves_left: 1,
            opening: true,
            terminal: false,
            stones_placed: 0,
            last_move: None,
            current_turn_moves: [0; 2],
            current_turn_len: 0,
            turn_count: 0,
        }
    }

    /// Reconstructs a semantically complete state from packed data.
    pub fn from_parts(
        board: Arc<Board>,
        stones: [BitBoard; 2],
        to_move: Player,
        moves_left: u8,
        opening: bool,
    ) -> Result<Self, GameError> {
        if !stones[0].intersection(stones[1]).is_empty() {
            return Err(GameError::OverlappingStones);
        }
        let board_mask = board.node_mask();
        if !stones[0].difference(board_mask).is_empty()
            || !stones[1].difference(board_mask).is_empty()
        {
            return Err(GameError::StonesOutsideBoard);
        }
        let stones_placed = stones[0].count() + stones[1].count();
        let board_full = stones_placed == board.node_count();
        let terminal = board_full;
        if moves_left > 2
            || (moves_left == 0 && !board_full)
            || (board_full && moves_left > 1)
            || (opening
                && (to_move != Player::Zero
                    || moves_left != 1
                    || !stones[0].is_empty()
                    || !stones[1].is_empty()))
        {
            return Err(GameError::InvalidTurnMetadata);
        }

        Ok(Self {
            board,
            stones,
            to_move,
            moves_left,
            opening,
            terminal,
            stones_placed,
            last_move: None,
            current_turn_moves: [0; 2],
            current_turn_len: 0,
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

    /// Whether the opening one-stone turn is active.
    #[must_use]
    pub const fn is_opening(&self) -> bool {
        self.opening
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

    /// Placements retained in the current unfinished turn.
    #[must_use]
    pub fn current_turn_moves(&self) -> &[NodeId] {
        &self.current_turn_moves[..usize::from(self.current_turn_len)]
    }

    /// Whether at least one placement has been made and another remains.
    #[must_use]
    pub const fn is_mid_turn(&self) -> bool {
        !self.opening && self.moves_left == 1
    }

    /// Number of completed turns.
    #[must_use]
    pub const fn turn_count(&self) -> u32 {
        self.turn_count
    }

    /// Semantic key for exact transposition reuse.
    #[must_use]
    pub fn key(&self) -> StateKey {
        StateKey {
            rings: self.board.rings(),
            stones: self.stones,
            to_move: self.to_move,
            moves_left: self.moves_left,
            opening: self.opening,
            terminal: self.terminal,
        }
    }

    /// Stable deterministic Zobrist-style hash.
    #[must_use]
    pub fn hash64(&self) -> u64 {
        let mut hash = splitmix64(0xd0ab_1e5a_7a12_0000 ^ u64::from(self.board.rings()));
        for player in [Player::Zero, Player::One] {
            for node in self.stones_for(player) {
                let index = (player as u64) * 448 + u64::from(node);
                hash ^= splitmix64(0x51a7_e000_0000_0000 ^ index);
            }
        }
        hash ^= splitmix64(0x7000_0000_0000_0000 ^ self.to_move as u64);
        hash ^= splitmix64(0x7100_0000_0000_0000 ^ u64::from(self.moves_left));
        hash ^= splitmix64(0x7200_0000_0000_0000 ^ u64::from(self.opening));
        hash ^= splitmix64(0x7400_0000_0000_0000 ^ u64::from(self.terminal));
        hash
    }

    /// Replaces only spatial fields while retaining protocol metadata.
    pub(crate) fn with_transformed_spatial(
        &self,
        stones: [BitBoard; 2],
        last_move: Option<NodeId>,
        current_turn_moves: &[NodeId],
    ) -> Self {
        let mut transformed = self.clone();
        transformed.stones = stones;
        transformed.last_move = last_move;
        transformed.current_turn_len = current_turn_moves.len() as u8;
        transformed.current_turn_moves = [0; 2];
        transformed.current_turn_moves[..current_turn_moves.len()]
            .copy_from_slice(current_turn_moves);
        transformed
    }

    /// Legal atomic actions.
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
        let Action::Place(node) = action;
        node < self.board.node_count() && !self.occupied().contains(node)
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
            stones_placed: self.stones_placed,
            last_move: self.last_move,
            current_turn_moves: self.current_turn_moves,
            current_turn_len: self.current_turn_len,
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
        self.stones_placed = undo.stones_placed;
        self.last_move = undo.last_move;
        self.current_turn_moves = undo.current_turn_moves;
        self.current_turn_len = undo.current_turn_len;
        self.turn_count = undo.turn_count;
    }

    fn apply_internal(&mut self, action: Action) -> Result<Transition, GameError> {
        if self.terminal {
            return Err(GameError::GameOver);
        }
        let player_before = self.to_move;
        let Action::Place(node) = action;
        if node >= self.board.node_count() {
            return Err(GameError::NodeOutOfBounds(node));
        }
        if self.occupied().contains(node) {
            return Err(GameError::Occupied(node));
        }
        self.stones[self.to_move.index()].insert(node);
        self.stones_placed += 1;
        self.last_move = Some(node);
        self.current_turn_moves[usize::from(self.current_turn_len)] = node;
        self.current_turn_len += 1;
        self.moves_left -= 1;

        if self.stones_placed == self.board.node_count() {
            self.terminal = true;
        } else if self.moves_left == 0 {
            self.end_turn();
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
        self.moves_left = 2;
        self.opening = false;
        self.current_turn_len = 0;
    }
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
