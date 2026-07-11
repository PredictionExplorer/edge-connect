use crate::{BitBoard, Board, GameState, MAX_NODES, NodeId, Player};

const NO_REGION: i16 = -1;
const NO_BORDER: i8 = -2;
const MIXED_BORDER: i8 = -1;

/// Static score components for one player.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlayerScore {
    /// Perimeter nodes occupied by stars or owned as territory.
    pub peries: i16,
    /// Owned corner peries.
    pub quarks: i16,
    /// Number of live connected stars.
    pub stars: i16,
    /// One point for owning at least three quarks.
    pub quark_peri: i16,
    /// Twice the opponent-star minus own-star count.
    pub award: i16,
    /// Conventional total score.
    pub total: i16,
}

/// Authoritative static score and node ownership.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScoreResult {
    /// Scores in fixed player order.
    pub players: [PlayerScore; 2],
    /// `-1` for contested/unowned, otherwise the player index.
    pub node_owner: [i8; MAX_NODES],
    /// Stones belonging to groups that occupy at least two peries.
    pub alive_stones: BitBoard,
    /// Peries owned by neither player.
    pub contested_peries: u16,
    /// Player ahead after the quark tie-break, or `None` for a dead tie.
    pub leader: Option<Player>,
}

impl ScoreResult {
    /// Ownership of one in-range node.
    #[must_use]
    pub fn owner(&self, node: NodeId) -> Option<Player> {
        match self.node_owner[usize::from(node)] {
            0 => Some(Player::Zero),
            1 => Some(Player::One),
            _ => None,
        }
    }

    /// Decisive zero-sum result from one player's perspective.
    ///
    /// Arbitrary static positions may be tied and therefore return `None`.
    #[must_use]
    pub fn outcome_for(&self, player: Player) -> Option<f32> {
        self.leader
            .map(|leader| if leader == player { 1.0 } else { -1.0 })
    }
}

/// Reusable, allocation-free scoring workspace.
#[derive(Clone)]
pub struct ScoringScratch {
    parent: [NodeId; MAX_NODES],
    occupied_peries: [u16; MAX_NODES],
    region_of: [i16; MAX_NODES],
    stack: [NodeId; MAX_NODES],
    region_color: [i8; MAX_NODES],
}

impl Default for ScoringScratch {
    fn default() -> Self {
        Self {
            parent: [0; MAX_NODES],
            occupied_peries: [0; MAX_NODES],
            region_of: [NO_REGION; MAX_NODES],
            stack: [0; MAX_NODES],
            region_color: [NO_BORDER; MAX_NODES],
        }
    }
}

impl ScoringScratch {
    /// Scores a complete state without modifying it.
    pub fn score_state(&mut self, state: &GameState) -> ScoreResult {
        self.score(state.board(), state.stones())
    }

    /// Scores two non-overlapping player bitboards on a board.
    pub fn score(&mut self, board: &Board, stones: [BitBoard; 2]) -> ScoreResult {
        let n = usize::from(board.node_count());
        for node in 0..n {
            self.parent[node] = node as NodeId;
            self.occupied_peries[node] = 0;
            self.region_of[node] = NO_REGION;
            self.region_color[node] = NO_BORDER;
        }

        for node_index in 0..n {
            let node = node_index as NodeId;
            let Some(color) = stone_owner(stones, node) else {
                continue;
            };
            for &neighbor in board.neighbors(node) {
                if neighbor > node && stone_owner(stones, neighbor) == Some(color) {
                    let left_root = self.find(node);
                    let right_root = self.find(neighbor);
                    if left_root != right_root {
                        self.parent[usize::from(right_root)] = left_root;
                    }
                }
            }
        }

        for node in board.peri_mask() {
            if stone_owner(stones, node).is_some() {
                let root = self.find(node);
                self.occupied_peries[usize::from(root)] += 1;
            }
        }

        let mut alive_stones = BitBoard::empty();
        for node_index in 0..n {
            let node = node_index as NodeId;
            if stone_owner(stones, node).is_some() {
                let root = self.find(node);
                if self.occupied_peries[usize::from(root)] >= 2 {
                    alive_stones.insert(node);
                }
            }
        }

        let mut region_count = 0_usize;
        for start_index in 0..n {
            let start = start_index as NodeId;
            if alive_stones.contains(start) || self.region_of[start_index] != NO_REGION {
                continue;
            }

            let region_id = region_count as i16;
            region_count += 1;
            let mut color = NO_BORDER;
            let mut top = 0_usize;
            self.stack[top] = start;
            top += 1;
            self.region_of[start_index] = region_id;

            while top > 0 {
                top -= 1;
                let node = self.stack[top];
                for &neighbor in board.neighbors(node) {
                    if alive_stones.contains(neighbor) {
                        let neighbor_color = stone_owner(stones, neighbor)
                            .expect("alive nodes always contain a stone")
                            as i8;
                        color = if color == NO_BORDER {
                            neighbor_color
                        } else if color == neighbor_color {
                            color
                        } else {
                            MIXED_BORDER
                        };
                    } else {
                        let neighbor_index = usize::from(neighbor);
                        if self.region_of[neighbor_index] == NO_REGION {
                            self.region_of[neighbor_index] = region_id;
                            self.stack[top] = neighbor;
                            top += 1;
                        }
                    }
                }
            }
            self.region_color[usize::try_from(region_id).expect("region id is non-negative")] =
                color;
        }

        let mut node_owner = [-1_i8; MAX_NODES];
        let mut peries = [0_i16; 2];
        let mut quarks = [0_i16; 2];
        let mut stars = [0_i16; 2];
        let mut contested_peries = 0_u16;

        for (node_index, owner_slot) in node_owner.iter_mut().enumerate().take(n) {
            let node = node_index as NodeId;
            let owner = if alive_stones.contains(node) {
                let player = stone_owner(stones, node).expect("alive nodes always contain a stone");
                if self.find(node) == node {
                    stars[player.index()] += 1;
                }
                player as i8
            } else {
                let region = self.region_of[node_index];
                self.region_color[usize::try_from(region).expect("territory has a region")]
            };

            if owner == 0 || owner == 1 {
                *owner_slot = owner;
                if board.is_peri(node) {
                    let player = owner as usize;
                    peries[player] += 1;
                    if board.is_quark(node) {
                        quarks[player] += 1;
                    }
                }
            } else if board.is_peri(node) {
                contested_peries += 1;
            }
        }

        let players = core::array::from_fn(|player| {
            let quark_peri = i16::from(quarks[player] >= 3);
            let award = 2 * (stars[1 - player] - stars[player]);
            PlayerScore {
                peries: peries[player],
                quarks: quarks[player],
                stars: stars[player],
                quark_peri,
                award,
                total: peries[player] + quark_peri + award,
            }
        });
        let leader = if players[0].total != players[1].total {
            Some(if players[0].total > players[1].total {
                Player::Zero
            } else {
                Player::One
            })
        } else if players[0].quarks != players[1].quarks {
            Some(if players[0].quarks > players[1].quarks {
                Player::Zero
            } else {
                Player::One
            })
        } else {
            None
        };

        ScoreResult {
            players,
            node_owner,
            alive_stones,
            contested_peries,
            leader,
        }
    }

    fn find(&mut self, node: NodeId) -> NodeId {
        let mut root = node;
        while self.parent[usize::from(root)] != root {
            let parent = self.parent[usize::from(root)];
            let grandparent = self.parent[usize::from(parent)];
            self.parent[usize::from(root)] = grandparent;
            root = grandparent;
        }
        root
    }
}

/// Convenience scoring entry point for callers that do not retain scratch.
#[must_use]
pub fn score_state(state: &GameState) -> ScoreResult {
    ScoringScratch::default().score_state(state)
}

/// Terminal zero-sum value from the state's current-player perspective.
#[must_use]
pub fn terminal_value(state: &GameState) -> Option<f32> {
    state.is_terminal().then(|| {
        score_state(state)
            .outcome_for(state.to_move())
            .expect("a full Double *Star board must have a decisive winner")
    })
}

fn stone_owner(stones: [BitBoard; 2], node: NodeId) -> Option<Player> {
    if stones[0].contains(node) {
        Some(Player::Zero)
    } else if stones[1].contains(node) {
        Some(Player::One)
    } else {
        None
    }
}
