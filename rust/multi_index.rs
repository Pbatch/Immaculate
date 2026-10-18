//! Multi-index hashing (Norouzi, Punjani and Fleet, 2012), finding every 256-bit hash within
//! some Hamming distance of a query without comparing the query against them all.
//!
//! Each hash is split into 16 chunks of 16 bits. If two hashes differ in at most `threshold`
//! bits, some chunk differs in at most `threshold / 16` bits, by the pigeonhole principle. So
//! looking up each of the query's chunks, and every value within `threshold / 16` bits of
//! it, finds every match. The candidates found are then checked against the full hash, so
//! the results are exactly those of comparing every pair.

use crate::pdq::{Bits, distance};

/// Number of chunks each hash is split into.
const CHUNKS: usize = 16;
/// `CHUNKS` as a `u32`, for dividing thresholds.
#[expect(clippy::cast_possible_truncation, reason = "there are 16 chunks")]
const CHUNKS_U32: u32 = CHUNKS as u32;
/// Number of values a 16-bit chunk can take.
const CHUNK_VALUES: usize = 1 << 16;

/// The hashes, with each chunk's positions bucketed by the chunk's value.
pub struct MultiIndex {
    hashes: Vec<Bits>,
    threshold: u32,
    /// Every 16-bit mask with at most `threshold / 16` bits set, to look up each chunk value
    /// within that many bits of the query's.
    masks: Vec<u16>,
    /// For each chunk, where each value's bucket starts in `positions`, plus the end.
    starts: Vec<u32>,
    /// For each chunk, the positions of the hashes, ascending within each bucket.
    positions: Vec<u32>,
}

impl MultiIndex {
    /// Indexes `hashes`, to find those within `threshold` bits of a query. Returns `None` if
    /// there are too many hashes to number with a `u32`.
    pub fn new(hashes: Vec<Bits>, threshold: u32) -> Option<Self> {
        let len = u32::try_from(hashes.len()).ok()?;
        let radius = threshold / CHUNKS_U32;
        let masks = (0..=u16::MAX)
            .filter(|mask| mask.count_ones() <= radius)
            .collect();

        // A counting sort of the positions by each chunk's value. Positions are visited in
        // order, so each bucket is ascending.
        let mut starts = vec![0; CHUNKS * (CHUNK_VALUES + 1)];
        let mut positions = vec![0; CHUNKS * hashes.len()];
        for (chunk, (starts, positions)) in starts
            .as_chunks_mut::<{ CHUNK_VALUES + 1 }>()
            .0
            .iter_mut()
            .zip(positions.chunks_exact_mut(hashes.len().max(1)))
            .enumerate()
        {
            for bits in &hashes {
                starts[chunk_value(bits, chunk) + 1] += 1;
            }
            for value in 0..CHUNK_VALUES {
                starts[value + 1] += starts[value];
            }
            let mut next = starts.to_vec();
            for (bits, position) in hashes.iter().zip(0..len) {
                let slot = &mut next[chunk_value(bits, chunk)];
                positions[*slot as usize] = position;
                *slot += 1;
            }
        }

        Some(Self {
            hashes,
            threshold,
            masks,
            starts,
            positions,
        })
    }

    /// Calls `found` with the position of each hash from `first` on that is within the
    /// threshold of `query`, possibly more than once per hash.
    pub fn find(&self, query: &Bits, first: usize, mut found: impl FnMut(usize)) {
        if self.hashes.is_empty() {
            return;
        }
        for (chunk, (starts, positions)) in self
            .starts
            .as_chunks::<{ CHUNK_VALUES + 1 }>()
            .0
            .iter()
            .zip(self.positions.chunks_exact(self.hashes.len()))
            .enumerate()
        {
            let value = chunk_value(query, chunk);
            for &mask in &self.masks {
                let probe = value ^ usize::from(mask);
                let bucket = &positions[starts[probe] as usize..starts[probe + 1] as usize];
                let later = bucket.partition_point(|&position| (position as usize) < first);
                for &position in &bucket[later..] {
                    let position = position as usize;
                    if distance(query, &self.hashes[position]) <= self.threshold {
                        found(position);
                    }
                }
            }
        }
    }
}

/// The value of the hash's `chunk`th 16-bit chunk.
fn chunk_value(bits: &Bits, chunk: usize) -> usize {
    #[expect(clippy::cast_possible_truncation, reason = "keeps the chunk's 16 bits")]
    let value = (bits[chunk / 4] >> (16 * (chunk % 4))) as u16;
    usize::from(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::XorShift;

    /// `count` random hashes, each followed by copies with up to `max_flips` random bits
    /// flipped, so many pairs lie near every threshold.
    fn test_hashes(count: usize, max_flips: u64, rng: &mut XorShift) -> Vec<Bits> {
        let mut hashes = Vec::new();
        for _ in 0..count {
            let base = rng.bits();
            hashes.push(base);
            for _ in 0..3 {
                let mut copy = base;
                for _ in 0..rng.next_u64() % (max_flips + 1) {
                    let bit = rng.next_u64() % 256;
                    copy[(bit / 64) as usize] ^= 1 << (bit % 64);
                }
                hashes.push(copy);
            }
        }
        hashes
    }

    #[test]
    fn finds_exactly_the_hashes_within_the_threshold() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        let hashes = test_hashes(100, 80, &mut rng);
        let queries = test_hashes(20, 80, &mut rng);
        for threshold in [0, 1, 15, 16, 31, 32, 47, 70] {
            let index = MultiIndex::new(hashes.clone(), threshold).expect("few hashes");
            for query in queries.iter().chain(&hashes) {
                for first in [0, 1, 150, hashes.len()] {
                    let mut found = Vec::new();
                    index.find(query, first, |position| found.push(position));
                    found.sort_unstable();
                    found.dedup();
                    let expected: Vec<usize> = (first..hashes.len())
                        .filter(|&position| distance(query, &hashes[position]) <= threshold)
                        .collect();
                    assert_eq!(found, expected, "threshold {threshold}, first {first}");
                }
            }
        }
    }

    #[test]
    fn empty_index_finds_nothing() {
        let index = MultiIndex::new(Vec::new(), 31).expect("no hashes");
        let mut found = Vec::new();
        index.find(&Bits::default(), 0, |position| found.push(position));
        assert_eq!(found, Vec::<usize>::new());
    }
}
