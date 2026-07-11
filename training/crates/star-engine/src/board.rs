use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use crate::{BitBoard, NodeId, SUPPORTED_RINGS};

/// Sector symbols used by the official `Nxy` notation.
pub const SECTOR_CHARS: [char; 5] = ['*', 'S', 'T', 'A', 'R'];

/// Errors returned while constructing or addressing a board.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BoardError {
    /// The ring count is outside the supported range.
    InvalidRingCount(u8),
    /// A textual node label is unknown on this board.
    UnknownLabel(String),
}

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRingCount(rings) => {
                write!(f, "rings must be one of 4, 6, 8, or 10, got {rings}")
            }
            Self::UnknownLabel(label) => write!(f, "unknown node label: {label}"),
        }
    }
}

impl Error for BoardError {}

/// Immutable pentagonal board topology.
#[derive(Clone, Debug)]
pub struct Board {
    rings: u8,
    node_count: u16,
    peri_count: u16,
    sector_of: Vec<u8>,
    ring_of: Vec<u8>,
    position_of: Vec<u8>,
    peries: BitBoard,
    quarks: BitBoard,
    labels: Vec<String>,
    label_to_id: HashMap<String, NodeId>,
    adjacency_offsets: Vec<u16>,
    adjacency: Vec<NodeId>,
    bridge: [NodeId; 5],
    topology_hash: u64,
}

impl Board {
    /// Constructs one of the supported 4, 6, 8, or 10 ring boards.
    pub fn new(rings: u8) -> Result<Self, BoardError> {
        if !SUPPORTED_RINGS.contains(&rings) {
            return Err(BoardError::InvalidRingCount(rings));
        }

        let node_count = ring_start(rings + 1);
        let n = usize::from(node_count);
        let mut sector_of = vec![0; n];
        let mut ring_of = vec![0; n];
        let mut position_of = vec![0; n];
        let mut peries = BitBoard::empty();
        let mut quarks = BitBoard::empty();
        let mut labels = vec![String::new(); n];
        let mut label_to_id = HashMap::with_capacity(n);

        for ring in 1..=rings {
            for sector in 0..5 {
                for position in 0..ring {
                    let node = index_unchecked(sector, ring, position);
                    let node_index = usize::from(node);
                    sector_of[node_index] = sector;
                    ring_of[node_index] = ring;
                    position_of[node_index] = position;
                    if ring == rings {
                        peries.insert(node);
                        if position == 0 {
                            quarks.insert(node);
                        }
                    }
                    let ring_label = if ring == 10 {
                        "0".to_owned()
                    } else {
                        ring.to_string()
                    };
                    let label = format!(
                        "{}{ring_label}{position}",
                        SECTOR_CHARS[usize::from(sector)]
                    );
                    labels[node_index].clone_from(&label);
                    label_to_id.insert(label, node);
                }
            }
        }

        let mut edge_keys = HashSet::new();
        let mut edges = Vec::new();
        let mut add_edge = |a: NodeId, b: NodeId| {
            let edge = if a < b { (a, b) } else { (b, a) };
            if edge_keys.insert(edge) {
                edges.push(edge);
            }
        };

        for ring in 1..=rings {
            for sector in 0..5 {
                for position in 0..ring {
                    let node = index_unchecked(sector, ring, position);
                    let clockwise = if position < ring - 1 {
                        index_unchecked(sector, ring, position + 1)
                    } else {
                        index_unchecked((sector + 1) % 5, ring, 0)
                    };
                    add_edge(node, clockwise);

                    if ring >= 2 {
                        if position <= ring - 2 {
                            add_edge(node, index_unchecked(sector, ring - 1, position));
                        }
                        if position >= 1 {
                            add_edge(node, index_unchecked(sector, ring - 1, position - 1));
                        }
                        if position == ring - 1 {
                            add_edge(node, index_unchecked((sector + 1) % 5, ring - 1, 0));
                        }
                    }
                }
            }
        }

        let bridge = core::array::from_fn(|sector| index_unchecked(sector as u8, 1, 0));
        for left in 0..bridge.len() {
            for right in (left + 1)..bridge.len() {
                add_edge(bridge[left], bridge[right]);
            }
        }

        let mut degrees = vec![0_u16; n];
        for &(left, right) in &edges {
            degrees[usize::from(left)] += 1;
            degrees[usize::from(right)] += 1;
        }
        let mut adjacency_offsets = vec![0_u16; n + 1];
        for node in 0..n {
            adjacency_offsets[node + 1] = adjacency_offsets[node] + degrees[node];
        }
        let mut adjacency = vec![0; usize::from(adjacency_offsets[n])];
        let mut cursor = adjacency_offsets[..n].to_vec();
        for &(left, right) in &edges {
            let left_cursor = &mut cursor[usize::from(left)];
            adjacency[usize::from(*left_cursor)] = right;
            *left_cursor += 1;
            let right_cursor = &mut cursor[usize::from(right)];
            adjacency[usize::from(*right_cursor)] = left;
            *right_cursor += 1;
        }
        let topology_hash = topology_hash(
            rings,
            node_count,
            &sector_of,
            &ring_of,
            &position_of,
            &adjacency_offsets,
            &adjacency,
        );

        Ok(Self {
            rings,
            node_count,
            peri_count: 5 * u16::from(rings),
            sector_of,
            ring_of,
            position_of,
            peries,
            quarks,
            labels,
            label_to_id,
            adjacency_offsets,
            adjacency,
            bridge,
            topology_hash,
        })
    }

    /// Number of concentric rings.
    #[must_use]
    pub const fn rings(&self) -> u8 {
        self.rings
    }

    /// Number of playable nodes.
    #[must_use]
    pub const fn node_count(&self) -> u16 {
        self.node_count
    }

    /// Number of perimeter nodes.
    #[must_use]
    pub const fn peri_count(&self) -> u16 {
        self.peri_count
    }

    /// Number of undirected topology edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.adjacency.len() / 2
    }

    /// Dense id for a valid `(sector, ring, position)` coordinate.
    #[must_use]
    pub fn index(&self, sector: u8, ring: u8, position: u8) -> Option<NodeId> {
        (sector < 5 && (1..=self.rings).contains(&ring) && position < ring)
            .then(|| index_unchecked(sector, ring, position))
    }

    /// Sector containing a node.
    #[must_use]
    pub fn sector(&self, node: NodeId) -> u8 {
        self.sector_of[usize::from(node)]
    }

    /// Ring containing a node.
    #[must_use]
    pub fn ring(&self, node: NodeId) -> u8 {
        self.ring_of[usize::from(node)]
    }

    /// Tangential position of a node within its sector.
    #[must_use]
    pub fn position(&self, node: NodeId) -> u8 {
        self.position_of[usize::from(node)]
    }

    /// Tests whether a node is on the perimeter.
    #[must_use]
    pub fn is_peri(&self, node: NodeId) -> bool {
        self.peries.contains(node)
    }

    /// Tests whether a node is one of the five corner quarks.
    #[must_use]
    pub fn is_quark(&self, node: NodeId) -> bool {
        self.quarks.contains(node)
    }

    /// Fixed mask of all playable nodes.
    #[must_use]
    pub fn node_mask(&self) -> BitBoard {
        BitBoard::board_mask(usize::from(self.node_count))
    }

    /// Fixed perimeter mask.
    #[must_use]
    pub const fn peri_mask(&self) -> BitBoard {
        self.peries
    }

    /// Fixed quark mask.
    #[must_use]
    pub const fn quark_mask(&self) -> BitBoard {
        self.quarks
    }

    /// Sorted neighbors of a node.
    #[must_use]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let node = usize::from(node);
        let start = usize::from(self.adjacency_offsets[node]);
        let end = usize::from(self.adjacency_offsets[node + 1]);
        &self.adjacency[start..end]
    }

    /// Official label for a node.
    #[must_use]
    pub fn label(&self, node: NodeId) -> &str {
        &self.labels[usize::from(node)]
    }

    /// Parses an official label on this board.
    pub fn parse_label(&self, label: &str) -> Result<NodeId, BoardError> {
        self.label_to_id
            .get(label)
            .copied()
            .ok_or_else(|| BoardError::UnknownLabel(label.to_owned()))
    }

    /// Ring-one nodes joined by the non-playable central bridge.
    #[must_use]
    pub const fn bridge(&self) -> [NodeId; 5] {
        self.bridge
    }

    /// Perimeter ids in clockwise cycle order.
    #[must_use]
    pub fn perimeter_cycle(&self) -> Vec<NodeId> {
        let mut nodes = Vec::with_capacity(usize::from(self.peri_count));
        for sector in 0..5 {
            for position in 0..self.rings {
                nodes.push(index_unchecked(sector, self.rings, position));
            }
        }
        nodes
    }

    /// Nodes on one radial arm, from ring one to the corner.
    #[must_use]
    pub fn arm_path(&self, sector: u8) -> Option<Vec<NodeId>> {
        (sector < 5).then(|| {
            (1..=self.rings)
                .map(|ring| index_unchecked(sector, ring, 0))
                .collect()
        })
    }

    /// Stable hash of this generated topology.
    #[must_use]
    pub const fn topology_hash(&self) -> u64 {
        self.topology_hash
    }
}

const fn ring_start(ring: u8) -> u16 {
    5 * (ring as u16) * ((ring - 1) as u16) / 2
}

const fn index_unchecked(sector: u8, ring: u8, position: u8) -> NodeId {
    ring_start(ring) + (sector as u16) * (ring as u16) + (position as u16)
}

fn topology_hash(
    rings: u8,
    node_count: u16,
    sectors: &[u8],
    ring_of: &[u8],
    positions: &[u8],
    offsets: &[u16],
    adjacency: &[NodeId],
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in [rings, node_count as u8, (node_count >> 8) as u8] {
        hash = fnv_byte(hash, byte);
    }
    for values in [sectors, ring_of, positions] {
        for &byte in values {
            hash = fnv_byte(hash, byte);
        }
    }
    for &value in offsets.iter().chain(adjacency) {
        hash = fnv_byte(hash, value as u8);
        hash = fnv_byte(hash, (value >> 8) as u8);
    }
    hash
}

const fn fnv_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ (byte as u64)).wrapping_mul(0x0000_0100_0000_01b3)
}
