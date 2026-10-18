//! PDQ, Meta's perceptual hash for photos, following the reference implementation at
//! <https://github.com/facebook/ThreatExchange/tree/main/pdq>.
//!
//! Images match when their 256-bit hashes differ in at most `threshold` bits, comparing every
//! rotation and flip of one image against the other. Matches chain: if A matches B and B
//! matches C, all three are one set.

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Write;
use std::rc::Rc;
use std::sync::LazyLock;

use image::DynamicImage;
use rayon::prelude::*;

use crate::perceptual_hash::PerceptualHash;
use crate::rgb8;

/// Side length images are filtered and decimated to before the DCT.
const BUFFER_SIZE: usize = 64;
/// Side length of the block of DCT coefficients, giving `DCT_SIZE²` bits.
const DCT_SIZE: usize = 16;
/// Number of times the box filter is applied along rows then columns.
const JAROSZ_PASSES: usize = 2;
/// Meta's recommended match threshold, out of 256 bits.
pub const DEFAULT_THRESHOLD: u32 = 31;

/// A 256-bit hash. Bit `k` is bit `k % 64` of word `k / 64`.
pub type Bits = [u64; 4];
/// An image's hash in each of the 8 rotations and flips, starting with the image as it is.
pub type Orientations = [Bits; 8];

/// 16x64 DCT-II matrix, skipping the DC row: `sqrt(2/64) cos(π/128 (i+1)(2j+1))`.
static DCT_MATRIX: LazyLock<[[f32; BUFFER_SIZE]; DCT_SIZE]> = LazyLock::new(|| {
    let scale = (2.0 / 64.0_f64).sqrt();
    let mut matrix = [[0.0; BUFFER_SIZE]; DCT_SIZE];
    for (row, i) in matrix.iter_mut().zip(1_u32..) {
        for (value, j) in row.iter_mut().zip(0_u32..) {
            let angle = PI / 128.0 * f64::from(i) * f64::from(2 * j + 1);
            // The reference implementation also stores the matrix as floats
            #[expect(clippy::cast_possible_truncation, reason = "values are in [-1, 1]")]
            let value_f32 = (scale * angle.cos()) as f32;
            *value = value_f32;
        }
    }
    matrix
});

/// PDQ, matching hashes within `threshold` bits in any rotation or flip.
pub struct Pdq {
    /// Most bits two hashes can differ in and still match.
    pub threshold: u32,
}

impl Default for Pdq {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
        }
    }
}

impl PerceptualHash for Pdq {
    type Hash = Orientations;

    /// Returns `None` if the image is empty.
    fn hash(&self, image: &DynamicImage) -> Option<Orientations> {
        let coefficients = dct_coefficients(image)?;
        // The DCT of a flipped or transposed image is the original DCT with some signs flipped
        // and/or transposed, so the other orientations don't need hashing from scratch. Each
        // coefficient's row (column) frequency is i + 1 (j + 1); flipping vertically
        // (horizontally) negates the coefficients with odd row (column) frequency.
        let mut orientations = [Bits::default(); 8];
        for (orientation, bits) in orientations.iter_mut().enumerate() {
            let transpose = orientation & 4 != 0;
            let flip_vertical = orientation & 2 != 0;
            let flip_horizontal = orientation & 1 != 0;
            let mut oriented = [[0.0; DCT_SIZE]; DCT_SIZE];
            for (i, row) in oriented.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    let (u, v) = if transpose { (j, i) } else { (i, j) };
                    let odd_row = u % 2 == 0;
                    let odd_column = v % 2 == 0;
                    let negate = (flip_vertical && odd_row) != (flip_horizontal && odd_column);
                    let coefficient = coefficients[u][v];
                    *value = if negate { -coefficient } else { coefficient };
                }
            }
            *bits = to_bits(&oriented);
        }
        Some(orientations)
    }

    fn group(&self, hashes: &[(usize, &Orientations)]) -> Vec<Vec<usize>> {
        let matches: Vec<(usize, usize)> = (0..hashes.len())
            .into_par_iter()
            .flat_map_iter(|a| {
                let orientations = hashes[a].1;
                (a + 1..hashes.len())
                    .filter(move |&b| {
                        let other = &hashes[b].1[0];
                        orientations
                            .iter()
                            .any(|bits| distance(bits, other) <= self.threshold)
                    })
                    .map(move |b| (a, b))
            })
            .collect();

        let mut sets = UnionFind::new(hashes.len());
        for (a, b) in matches {
            sets.union(a, b);
        }
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); hashes.len()];
        for (position, &(index, _)) in hashes.iter().enumerate() {
            groups[sets.find(position)].push(index);
        }
        // Positions are visited in ascending index order, so each group is already sorted and
        // the groups are ordered by their first index once empty ones are removed.
        let mut groups: Vec<Vec<usize>> =
            groups.into_iter().filter(|group| group.len() > 1).collect();
        groups.sort_unstable_by_key(|group| group[0]);
        groups
    }
}

/// Number of bits that differ.
pub fn distance(a: &Bits, b: &Bits) -> u32 {
    a.iter().zip(b).map(|(a, b)| (a ^ b).count_ones()).sum()
}

/// Formats a hash as 64 hex digits, as Meta's tools do.
pub fn to_hex(bits: &Bits) -> String {
    // Meta stores the hash as 16-bit words with bit k in word k / 16, printed last word first
    (0..16)
        .rev()
        .fold(String::with_capacity(64), |mut hex, word| {
            let value = (bits[word / 4] >> (16 * (word % 4))) & 0xffff;
            write!(hex, "{value:04x}").unwrap_or_else(|_| unreachable!("writing to a String"));
            hex
        })
}

/// The 16x16 low-frequency DCT coefficients of the image's luma, after blurring it with a
/// box filter and decimating it to 64x64, or `None` if the image is empty.
fn dct_coefficients(image: &DynamicImage) -> Option<[[f32; DCT_SIZE]; DCT_SIZE]> {
    let decimated = decimated_luma(image)?;

    // B = D A Dᵀ
    let dct = &*DCT_MATRIX;
    let mut partial = [[0.0f32; BUFFER_SIZE]; DCT_SIZE];
    for (i, row) in partial.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = (0..BUFFER_SIZE).map(|k| dct[i][k] * decimated[k][j]).sum();
        }
    }
    let mut coefficients = [[0.0f32; DCT_SIZE]; DCT_SIZE];
    for (i, row) in coefficients.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = (0..BUFFER_SIZE).map(|k| partial[i][k] * dct[j][k]).sum();
        }
    }
    Some(coefficients)
}

/// The image's luma, blurred and decimated to 64x64, or `None` if the image is empty.
///
/// The reference implementation converts the whole image to luma, blurs it with
/// `JAROSZ_PASSES` box filters along rows and columns, then samples a 64x64 grid. That is
/// linear and separable, so each sample is a weighted sum of luma along each axis
/// ([`axis_taps`]). Applying the weights directly takes one pass over the image and never
/// stores it as floats. The result differs from the reference only by float rounding.
fn decimated_luma(image: &DynamicImage) -> Option<[[f32; BUFFER_SIZE]; BUFFER_SIZE]> {
    let rgb = rgb8(image);
    let columns = usize::try_from(rgb.width()).ok()?;
    let rows = usize::try_from(rgb.height()).ok()?;
    if rows == 0 || columns == 0 {
        return None;
    }
    let column_taps = cached_axis_taps(columns);
    let row_taps = cached_axis_taps(rows);
    let mut row_used = vec![false; rows];
    for tap in row_taps.iter() {
        row_used[tap.start..tap.start + tap.weights.len()].fill(true);
    }

    // Each used row, reduced to the 64 sampled columns
    let mut sampled_rows = vec![[0.0f32; BUFFER_SIZE]; rows];
    let mut luma = vec![0.0f32; columns];
    for ((pixels, sampled), _) in rgb
        .rows()
        .zip(&mut sampled_rows)
        .zip(&row_used)
        .filter(|(_, used)| **used)
    {
        for (value, pixel) in luma.iter_mut().zip(pixels) {
            let [r, g, b] = pixel.0.map(f32::from);
            // Separate multiplies and adds, as in the reference implementation. mul_add
            // would also be slow: release builds target CPUs without FMA instructions.
            #[expect(clippy::suboptimal_flops, reason = "see above")]
            let weighted = 0.299 * r + 0.587 * g + 0.114 * b;
            *value = weighted;
        }
        for (value, tap) in sampled.iter_mut().zip(column_taps.iter()) {
            *value = tap.apply(&luma);
        }
    }

    let mut decimated = [[0.0f32; BUFFER_SIZE]; BUFFER_SIZE];
    #[expect(clippy::suboptimal_flops, reason = "mul_add is slow without FMA")]
    for (out, tap) in decimated.iter_mut().zip(row_taps.iter()) {
        for (&weight, sampled) in tap.weights.iter().zip(&sampled_rows[tap.start..]) {
            for (value, &sample) in out.iter_mut().zip(sampled) {
                *value += weight * sample;
            }
        }
    }
    Some(decimated)
}

/// The weights one decimated sample gives a run of consecutive pixels along an axis.
struct Tap {
    start: usize,
    weights: Vec<f32>,
}

/// Running sums [`Tap::apply`] keeps, which the compiler can add in parallel with SIMD.
/// It won't split a single float sum itself, since that changes how the sum rounds.
const TAP_LANES: usize = 8;

impl Tap {
    fn apply(&self, values: &[f32]) -> f32 {
        let values = &values[self.start..self.start + self.weights.len()];
        let (weight_chunks, weight_remainder) = self.weights.as_chunks::<TAP_LANES>();
        let (value_chunks, value_remainder) = values.as_chunks::<TAP_LANES>();
        let mut sums = [0.0f32; TAP_LANES];
        for (weights, values) in weight_chunks.iter().zip(value_chunks) {
            for ((sum, weight), value) in sums.iter_mut().zip(weights).zip(values) {
                *sum += weight * value;
            }
        }
        let remainder: f32 = weight_remainder
            .iter()
            .zip(value_remainder)
            .map(|(weight, value)| weight * value)
            .sum();
        sums.iter().sum::<f32>() + remainder
    }
}

/// Most image side lengths whose taps are cached, bounding the cache's memory.
const MAX_CACHED_TAPS: usize = 64;

thread_local! {
    /// Each thread's [`axis_taps`] by side length, since most folders have only a few image
    /// sizes.
    static TAPS: RefCell<HashMap<usize, Rc<[Tap]>>> = RefCell::default();
}

/// [`axis_taps`] for `length`, computed once per side length on each thread.
fn cached_axis_taps(length: usize) -> Rc<[Tap]> {
    TAPS.with_borrow_mut(|cache| {
        if cache.len() >= MAX_CACHED_TAPS && !cache.contains_key(&length) {
            cache.clear();
        }
        Rc::clone(
            cache
                .entry(length)
                .or_insert_with(|| axis_taps(length).into()),
        )
    })
}

/// The weights each of the 64 samples along an axis of `length` pixels gives each pixel,
/// after `JAROSZ_PASSES` box filters.
///
/// Starting from the sampled pixel, each pass spreads every weight evenly over the inputs
/// the box filter averages for that output, i.e. applies the transposed filter.
fn axis_taps(length: usize) -> Vec<Tap> {
    let window = window_size(length);
    (0..BUFFER_SIZE)
        .map(|i| {
            let mut weights = vec![0.0f64; length];
            weights[sample_index(i, length)] = 1.0;
            let (mut first, mut last) = (sample_index(i, length), sample_index(i, length));
            for _ in 0..JAROSZ_PASSES {
                let mut spread = vec![0.0f64; length];
                let (mut new_first, mut new_last) = (length, 0);
                for (output, &weight) in weights.iter().enumerate().take(last + 1).skip(first) {
                    let (low, high) = box_range(output, length, window);
                    #[expect(clippy::cast_precision_loss, reason = "windows are small")]
                    let share = weight / (high - low + 1) as f64;
                    for spread_weight in &mut spread[low..=high] {
                        *spread_weight += share;
                    }
                    new_first = new_first.min(low);
                    new_last = new_last.max(high);
                }
                (weights, first, last) = (spread, new_first, new_last);
            }
            #[expect(clippy::cast_possible_truncation, reason = "weights are in [0, 1]")]
            let weights = weights[first..=last].iter().map(|&w| w as f32).collect();
            Tap {
                start: first,
                weights,
            }
        })
        .collect()
}

/// Box filter window size for shrinking `dimension` to 64, rounded up.
const fn window_size(dimension: usize) -> usize {
    dimension.div_ceil(2 * BUFFER_SIZE)
}

/// The first and last inputs the reference box filter averages for `output`: a `window`-wide
/// window roughly centered on it, cut off at the edges.
const fn box_range(output: usize, length: usize, window: usize) -> (usize, usize) {
    let end = output + window / 2 + 1;
    let low = end.saturating_sub(window);
    let high = if end < length { end - 1 } else { length - 1 };
    (low, high)
}

/// The pixel sampled for the `i`th of 64 decimated values: (i + 0.5) * length / 64,
/// rounded down.
const fn sample_index(i: usize, length: usize) -> usize {
    (2 * i + 1) * length / (2 * BUFFER_SIZE)
}

/// Sets each bit where the coefficient is above the median, in row-major order.
fn to_bits(coefficients: &[[f32; DCT_SIZE]; DCT_SIZE]) -> Bits {
    let mut values = [0.0f32; DCT_SIZE * DCT_SIZE];
    for (value, &coefficient) in values.iter_mut().zip(coefficients.iter().flatten()) {
        *value = coefficient;
    }
    // The reference implementation's Torben median picks the lower of the two middle values
    let middle = values.len() / 2 - 1;
    let (_, &mut median, _) = values.select_nth_unstable_by(middle, f32::total_cmp);

    let mut bits = Bits::default();
    for (k, &coefficient) in coefficients.iter().flatten().enumerate() {
        if coefficient > median {
            bits[k / 64] |= 1 << (k % 64);
        }
    }
    bits
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, mut item: usize) -> usize {
        while self.parent[item] != item {
            self.parent[item] = self.parent[self.parent[item]];
            item = self.parent[item];
        }
        item
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        self.parent[a.max(b)] = a.min(b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference implementation's box filter: the mean of a `window`-wide window around
    /// each value, shrinking at the edges.
    fn box_filter(input: &[f32], window: usize) -> Vec<f32> {
        let half_window = window / 2 + 1;
        let mut output = Vec::with_capacity(input.len());
        let mut sum = 0.0;
        let mut count = 0.0;
        let mut left = 0;
        for (right, &value) in input.iter().enumerate() {
            sum += value;
            if right >= window {
                sum -= input[left];
                left += 1;
            } else {
                count += 1.0;
            }
            if right + 1 >= half_window {
                output.push(sum / count);
            }
        }
        while output.len() < input.len() {
            sum -= input[left];
            left += 1;
            count -= 1.0;
            output.push(sum / count);
        }
        output
    }

    #[test]
    fn taps_match_repeated_box_filters() {
        for length in [1, 2, 7, 63, 64, 65, 129, 480, 640, 1001, 2879] {
            #[expect(clippy::cast_precision_loss, reason = "small test values")]
            let values: Vec<f32> = (0..length).map(|i| ((i * 7919) % 256) as f32).collect();
            let mut filtered = values.clone();
            for _ in 0..JAROSZ_PASSES {
                filtered = box_filter(&filtered, window_size(length));
            }
            for (i, tap) in axis_taps(length).iter().enumerate() {
                let expected = filtered[sample_index(i, length)];
                let actual = tap.apply(&values);
                assert!(
                    (expected - actual).abs() <= 1e-3 * expected.abs().max(1.0),
                    "length {length}, sample {i}: {expected} != {actual}"
                );
            }
        }
    }
}
