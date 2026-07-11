use crate::{Action, BitBoard, Board, GameState, NodeId};

/// Number of rotations and reflections in the pentagon's D5 group.
pub const D5_ORDER: usize = 10;

/// One deterministic D5 symmetry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Symmetry(u8);

impl Symmetry {
    /// All five rotations followed by all five reflections.
    pub const ALL: [Self; D5_ORDER] = [
        Self(0),
        Self(1),
        Self(2),
        Self(3),
        Self(4),
        Self(5),
        Self(6),
        Self(7),
        Self(8),
        Self(9),
    ];

    /// Builds a symmetry from its stable index.
    #[must_use]
    pub const fn from_index(index: u8) -> Option<Self> {
        if index < D5_ORDER as u8 {
            Some(Self(index))
        } else {
            None
        }
    }

    /// Stable index in `0..10`.
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Whether this transform reverses orientation.
    #[must_use]
    pub const fn is_reflection(self) -> bool {
        self.0 >= 5
    }

    /// Arm offset in `0..5`.
    #[must_use]
    pub const fn arm_offset(self) -> u8 {
        self.0 % 5
    }

    /// Group inverse.
    #[must_use]
    pub const fn inverse(self) -> Self {
        if self.is_reflection() {
            self
        } else {
            Self((5 - self.arm_offset()) % 5)
        }
    }
}

/// Precomputed dense maps for all ten D5 transforms.
#[derive(Clone, Debug)]
pub struct D5Maps {
    maps: [Vec<NodeId>; D5_ORDER],
}

impl D5Maps {
    /// Computes maps in stable [`Symmetry::ALL`] order.
    #[must_use]
    pub fn new(board: &Board) -> Self {
        Self {
            maps: core::array::from_fn(|index| {
                let symmetry = Symmetry::ALL[index];
                (0..board.node_count())
                    .map(|node| map_node(board, node, symmetry))
                    .collect()
            }),
        }
    }

    /// Dense old-id to transformed-id map.
    #[must_use]
    pub fn map(&self, symmetry: Symmetry) -> &[NodeId] {
        &self.maps[usize::from(symmetry.index())]
    }

    /// Transforms one node.
    #[must_use]
    pub fn node(&self, symmetry: Symmetry, node: NodeId) -> NodeId {
        self.maps[usize::from(symmetry.index())][usize::from(node)]
    }

    /// Transforms one fixed bitboard.
    #[must_use]
    pub fn bitboard(&self, symmetry: Symmetry, source: BitBoard) -> BitBoard {
        let mut transformed = BitBoard::empty();
        for node in source {
            transformed.insert(self.node(symmetry, node));
        }
        transformed
    }

    /// Transforms one atomic action.
    #[must_use]
    pub fn action(&self, symmetry: Symmetry, action: Action) -> Action {
        match action {
            Action::Place(node) => Action::Place(self.node(symmetry, node)),
        }
    }

    /// Transforms stones while preserving all turn semantics.
    #[must_use]
    pub fn state(&self, symmetry: Symmetry, source: &GameState) -> GameState {
        let stones = [
            self.bitboard(symmetry, source.stones()[0]),
            self.bitboard(symmetry, source.stones()[1]),
        ];
        let last_move = source.last_move().map(|node| self.node(symmetry, node));
        let current_turn_moves: Vec<_> = source
            .current_turn_moves()
            .iter()
            .map(|node| self.node(symmetry, *node))
            .collect();
        source.with_transformed_spatial(stones, last_move, &current_turn_moves)
    }
}

fn map_node(board: &Board, node: NodeId, symmetry: Symmetry) -> NodeId {
    let ring = board.ring(node);
    let circumference = 5_i32 * i32::from(ring);
    let linear = i32::from(board.sector(node)) * i32::from(ring) + i32::from(board.position(node));
    let arm_shift = i32::from(symmetry.arm_offset()) * i32::from(ring);
    let mapped = if symmetry.is_reflection() {
        (arm_shift - linear).rem_euclid(circumference)
    } else {
        (linear + arm_shift).rem_euclid(circumference)
    };
    let sector = (mapped / i32::from(ring)) as u8;
    let position = (mapped % i32::from(ring)) as u8;
    board
        .index(sector, ring, position)
        .expect("D5 coordinates remain on the same ring")
}
