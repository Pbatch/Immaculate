//! pHash, as in the `imagehash` library. Images match when their hashes are identical.

use std::cell::RefCell;
use std::convert::Infallible;
use std::f64::consts::PI;
use std::sync::LazyLock;

use image::DynamicImage;

use crate::perceptual_hash::PerceptualHash;
use crate::side_cache::SideCache;
use crate::{group_by_key, rgb8};

/// Side length of the DCT block, giving `HASH_SIZE²` bits.
const HASH_SIZE: usize = 8;
/// Side length images are shrunk to before the DCT, as in `imagehash.phash`.
const IMAGE_SIZE: usize = HASH_SIZE * 4;
/// `IMAGE_SIZE` as a `u32`, for converting to floats.
#[expect(clippy::cast_possible_truncation, reason = "the image size is 32")]
const IMAGE_SIZE_U32: u32 = IMAGE_SIZE as u32;
/// Fractional bits in Pillow's fixed-point weights for resampling 8-bit images.
const PRECISION_BITS: u32 = 32 - 8 - 2;
/// Lanczos3's support, in input pixels at scale 1.
const LANCZOS_SUPPORT: f64 = 3.0;

/// 64-bit pHash, matching identical hashes.
pub struct PHash;

/// DCT-II basis for the lowest frequencies. Scipy's unnormalized DCT also multiplies by 2,
/// which doesn't change which coefficients are above the median.
static DCT_BASIS: LazyLock<[[f64; IMAGE_SIZE]; HASH_SIZE]> = LazyLock::new(|| {
    let mut basis = [[0.0; IMAGE_SIZE]; HASH_SIZE];
    for (row, k) in basis.iter_mut().zip(0_u32..) {
        for (value, n) in row.iter_mut().zip(0_u32..) {
            let n = f64::from(n);
            *value = (PI * f64::from(k) * 2.0f64.mul_add(n, 1.0)
                / (2.0 * f64::from(IMAGE_SIZE_U32)))
            .cos();
        }
    }
    basis
});

/// Working memory for [`shrink_luma`]: one image row of grayscale, the rows its horizontal
/// pass produces, and resize weights by image dimension.
#[derive(Default)]
struct Scratch {
    luma: Vec<i16>,
    narrow_rows: Vec<[u8; IMAGE_SIZE]>,
    kernels: SideCache<u32, Kernel>,
}

thread_local! {
    /// Each thread's [`Scratch`], reused from one image to the next.
    static SCRATCH: RefCell<Scratch> = RefCell::default();
}

impl PerceptualHash for PHash {
    type Hash = u64;
    type Error = Infallible;

    /// Computes a 64-bit perceptual hash, following `imagehash.phash`.
    ///
    /// The image is converted to grayscale and shrunk to 32x32. The hash's bits record
    /// whether each of the 8x8 lowest-frequency DCT coefficients is above their median,
    /// in row-major order with the first coefficient as the most significant bit.
    ///
    /// Returns `None` if the image can't be resized, e.g. because it is empty.
    fn hash(&self, image: &DynamicImage) -> Result<Option<u64>, Infallible> {
        let Some(small) = SCRATCH.with_borrow_mut(|scratch| shrink_luma(image, scratch)) else {
            return Ok(None);
        };
        let basis = &*DCT_BASIS;

        // Separable 2D DCT: along columns (axis 0), then along rows (axis 1).
        let mut columns = [[0.0; IMAGE_SIZE]; HASH_SIZE];
        for (u, out_row) in columns.iter_mut().enumerate() {
            for (x, out) in out_row.iter_mut().enumerate() {
                *out = (0..IMAGE_SIZE)
                    .map(|y| basis[u][y] * f64::from(small[y][x]))
                    .sum();
            }
        }
        let mut coefficients = [0.0; HASH_SIZE * HASH_SIZE];
        for u in 0..HASH_SIZE {
            for v in 0..HASH_SIZE {
                coefficients[u * HASH_SIZE + v] =
                    (0..IMAGE_SIZE).map(|x| basis[v][x] * columns[u][x]).sum();
            }
        }

        let mut sorted = coefficients;
        sorted.sort_unstable_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        let median = f64::midpoint(sorted[mid - 1], sorted[mid]);

        Ok(Some(coefficients.iter().fold(0, |hash, &coefficient| {
            (hash << 1) | u64::from(coefficient > median)
        })))
    }

    fn group(&self, hashes: &[(usize, &u64)]) -> Vec<Vec<usize>> {
        group_by_key(hashes.iter().map(|&(index, &hash)| (index, hash)))
    }
}

/// Converts the image to grayscale and shrinks it to 32x32, exactly as Pillow's
/// `convert("L").resize((32, 32), Image.LANCZOS)` does, or returns `None` if it is empty.
///
/// This ports Pillow's `Resample.c`: fixed-point weights, a horizontal pass over only the
/// rows the vertical pass reads, rounded to 8 bits, then a vertical pass.
fn shrink_luma(
    image: &DynamicImage,
    scratch: &mut Scratch,
) -> Option<[[u8; IMAGE_SIZE]; IMAGE_SIZE]> {
    let rgb = rgb8(image);
    if rgb.width() == 0 || rgb.height() == 0 {
        return None;
    }
    let column_kernels = scratch.kernels.get(rgb.width(), || kernels(rgb.width()));
    let row_kernels = scratch.kernels.get(rgb.height(), || kernels(rgb.height()));
    let first_row = row_kernels[0].start;
    let last_row = row_kernels[IMAGE_SIZE - 1].end();

    scratch.narrow_rows.clear();
    for pixels in rgb.rows().skip(first_row).take(last_row - first_row) {
        scratch.luma.clear();
        scratch.luma.extend(pixels.map(|pixel| {
            let [r, g, b] = pixel.0.map(u32::from);
            // Pillow's convert("L") (ITU-R 601-2 luma)
            let luma = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
            i16::try_from(luma).unwrap_or(i16::MAX)
        }));
        let mut narrow = [0; IMAGE_SIZE];
        for (value, kernel) in narrow.iter_mut().zip(column_kernels.iter()) {
            *value = kernel.apply_split(&scratch.luma[kernel.start..kernel.end()]);
        }
        scratch.narrow_rows.push(narrow);
    }

    let mut small = [[0; IMAGE_SIZE]; IMAGE_SIZE];
    for (row, kernel) in small.iter_mut().zip(row_kernels.iter()) {
        let rows = &scratch.narrow_rows[kernel.start - first_row..];
        for (column, value) in row.iter_mut().enumerate() {
            *value = kernel.apply(rows.iter().map(|narrow| narrow[column]));
        }
    }
    Some(small)
}

/// Pillow's fixed-point weights for one output pixel: `weights[k]` applies to input
/// pixel `start + k`.
struct Kernel {
    start: usize,
    weights: Vec<i32>,
    /// Each weight's high bits, `weight >> SPLIT_BITS`, and low bits, so that sums of
    /// products fit 16-bit multiplies, which baseline x86-64 can vectorize, unlike 32-bit
    /// ones. The sum is still exact.
    high: Vec<i16>,
    low: Vec<i16>,
}

/// Low bits of each weight kept in `Kernel::low`.
const SPLIT_BITS: u32 = 11;

impl Kernel {
    const fn end(&self) -> usize {
        self.start + self.weights.len()
    }

    /// The weighted sum of `inputs`, rounded and clamped to 8 bits as Pillow does.
    fn apply(&self, inputs: impl Iterator<Item = u8>) -> u8 {
        let sum = self
            .weights
            .iter()
            .zip(inputs)
            .fold(0, |sum, (&weight, input)| sum + i32::from(input) * weight);
        round_to_u8(sum)
    }

    /// Like `apply`, for one row of 8-bit values stored as `i16`, using the split weights.
    fn apply_split(&self, inputs: &[i16]) -> u8 {
        let dot = |weights: &[i16]| -> i32 {
            weights
                .iter()
                .zip(inputs)
                .map(|(&weight, &input)| i32::from(weight) * i32::from(input))
                .sum()
        };
        round_to_u8((dot(&self.high) << SPLIT_BITS) + dot(&self.low))
    }
}

/// Rounds a fixed-point weighted sum to 8 bits, clamping like Pillow's `clip8`.
fn round_to_u8(sum: i32) -> u8 {
    let sum = sum + (1 << (PRECISION_BITS - 1));
    if sum >= 1 << PRECISION_BITS << 8 {
        u8::MAX
    } else if sum <= 0 {
        0
    } else {
        u8::try_from(sum >> PRECISION_BITS).unwrap_or(u8::MAX)
    }
}

/// Pillow's Lanczos weights for shrinking `input` pixels to 32, following its
/// `precompute_coeffs` and `normalize_coeffs_8bpc`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "these truncate like Pillow's (int) casts, and the bounds are clamped to the image"
)]
fn kernels(input: u32) -> Vec<Kernel> {
    let scale = f64::from(input) / f64::from(IMAGE_SIZE_U32);
    let filter_scale = scale.max(1.0);
    let support = LANCZOS_SUPPORT * filter_scale;
    let reciprocal = 1.0 / filter_scale;
    (0..IMAGE_SIZE_U32)
        .map(|output| {
            let center = (f64::from(output) + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as u32;
            let end = ((center + support + 0.5) as i64).min(i64::from(input)) as u32;
            let weights: Vec<f64> = (start..end)
                .map(|x| lanczos((f64::from(x) - center + 0.5) * reciprocal))
                .collect();
            let total: f64 = weights.iter().sum();
            let fixed: Vec<i32> = weights
                .iter()
                .map(|&weight| {
                    let weight = if total == 0.0 { weight } else { weight / total };
                    let scaled = weight * f64::from(1_u32 << PRECISION_BITS);
                    (if weight < 0.0 {
                        scaled - 0.5
                    } else {
                        scaled + 0.5
                    }) as i32
                })
                .collect();
            let high = fixed
                .iter()
                .map(|&w: &i32| (w >> SPLIT_BITS) as i16)
                .collect();
            let low = fixed
                .iter()
                .map(|&w: &i32| (w & ((1 << SPLIT_BITS) - 1)) as i16)
                .collect();
            Kernel {
                start: start as usize,
                weights: fixed,
                high,
                low,
            }
        })
        .collect()
}

/// Lanczos3: a sinc windowed by a wider sinc, as in Pillow.
fn lanczos(x: f64) -> f64 {
    if (-LANCZOS_SUPPORT..LANCZOS_SUPPORT).contains(&x) {
        sinc(x) * sinc(x / LANCZOS_SUPPORT)
    } else {
        0.0
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x = x * PI;
        x.sin() / x
    }
}
