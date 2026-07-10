use core::fmt;

use crate::{MAX_NODES, NodeId};

/// Number of machine words reserved for every board mask.
pub const BITBOARD_WORDS: usize = 7;

/// A fixed-width bitboard large enough for every supported board.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct BitBoard(pub(crate) [u64; BITBOARD_WORDS]);

impl BitBoard {
    /// Returns an empty mask.
    #[must_use]
    pub const fn empty() -> Self {
        Self([0; BITBOARD_WORDS])
    }

    /// Returns a mask containing node ids `0..node_count`.
    #[must_use]
    pub fn board_mask(node_count: usize) -> Self {
        debug_assert!(node_count <= MAX_NODES);
        let mut words = [u64::MAX; BITBOARD_WORDS];
        let full_words = node_count / u64::BITS as usize;
        let tail_bits = node_count % u64::BITS as usize;

        if full_words < BITBOARD_WORDS {
            words[full_words] = if tail_bits == 0 {
                0
            } else {
                (1_u64 << tail_bits) - 1
            };
            words[(full_words + 1)..].fill(0);
        }
        Self(words)
    }

    /// Builds a mask from its stable seven-word representation.
    #[must_use]
    pub const fn from_words(words: [u64; BITBOARD_WORDS]) -> Self {
        Self(words)
    }

    /// Exposes the stable seven-word representation.
    #[must_use]
    pub const fn words(self) -> [u64; BITBOARD_WORDS] {
        self.0
    }

    /// Tests whether a node is present.
    #[must_use]
    pub fn contains(self, node: NodeId) -> bool {
        let node = usize::from(node);
        if node >= BITBOARD_WORDS * 64 {
            return false;
        }
        (self.0[node / 64] & (1_u64 << (node % 64))) != 0
    }

    /// Inserts a node and returns whether it was newly inserted.
    pub fn insert(&mut self, node: NodeId) -> bool {
        let node = usize::from(node);
        if node >= BITBOARD_WORDS * 64 {
            return false;
        }
        let bit = 1_u64 << (node % 64);
        let word = &mut self.0[node / 64];
        let was_absent = (*word & bit) == 0;
        *word |= bit;
        was_absent
    }

    /// Removes a node and returns whether it was present.
    pub fn remove(&mut self, node: NodeId) -> bool {
        let node = usize::from(node);
        if node >= BITBOARD_WORDS * 64 {
            return false;
        }
        let bit = 1_u64 << (node % 64);
        let word = &mut self.0[node / 64];
        let was_present = (*word & bit) != 0;
        *word &= !bit;
        was_present
    }

    /// Returns the number of set bits.
    #[must_use]
    pub fn count(self) -> u16 {
        self.0.iter().map(|word| word.count_ones() as u16).sum()
    }

    /// Returns whether the mask is empty.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    /// Returns the union of two masks.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] | other.0[i]))
    }

    /// Returns the intersection of two masks.
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] & other.0[i]))
    }

    /// Returns the bits in `self` that are absent from `other`.
    #[must_use]
    pub fn difference(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] & !other.0[i]))
    }

    /// Iterates set node ids in ascending order.
    pub fn iter(self) -> BitIter {
        BitIter {
            words: self.0,
            word_index: 0,
        }
    }
}

impl fmt::Debug for BitBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries((*self).iter()).finish()
    }
}

/// Ascending iterator over a [`BitBoard`].
pub struct BitIter {
    words: [u64; BITBOARD_WORDS],
    word_index: usize,
}

impl Iterator for BitIter {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        while self.word_index < BITBOARD_WORDS {
            let word = &mut self.words[self.word_index];
            if *word != 0 {
                let bit = word.trailing_zeros() as usize;
                *word &= *word - 1;
                let node = self.word_index * 64 + bit;
                return Some(node as NodeId);
            }
            self.word_index += 1;
        }
        None
    }
}

impl IntoIterator for BitBoard {
    type Item = NodeId;
    type IntoIter = BitIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
