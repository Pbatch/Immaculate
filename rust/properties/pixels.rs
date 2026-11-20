//! The one pass over an image's pixels that the property checks need, collecting brightness
//! counts, channel histograms, grayscale moments and a grayscale thumbnail.

use image::GrayImage;

use super::stats::{Moments, laplacian_spread};

/// Interleaved sets of histograms that [`Histograms`] counts in.
const LANES: usize = 4;
/// Longest side images are shrunk to before measuring blur.
const BLUR_SIZE: u32 = 256;
/// Thumbnails whose grayscale values' standard deviation is below this are too flat to be
/// blurry: they have low information instead.
const FLAT_SPREAD: f64 = 2.0;

/// Weights of squared red, green and blue in perceived brightness, in thousandths:
/// brightness is `sqrt(0.241 R² + 0.691 G² + 0.068 B²) / 255`, as in `CleanVision`.
const BRIGHTNESS_WEIGHTS: [u32; 3] = [241, 691, 68];

/// Weighted squared brightness levels, `1000 (255 brightness)²` compared with
/// `241 R² + 691 G² + 68 B²`, at or above which a pixel counts as bright, and at or below
/// which a pixel counts as not light.
#[derive(Clone, Copy)]
pub struct Thresholds {
    pub bright: u32,
    pub not_light: u32,
}

/// The weighted squared brightness level of `brightness`, from 0 to 1, as a float.
fn level(brightness: f64) -> f64 {
    let value = 255.0 * brightness;
    1000.0 * value * value
}

/// The lowest weighted squared brightness that is at least `brightness`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "float to int casts saturate, which is the intent for out of range thresholds"
)]
pub fn bright_level(brightness: f64) -> u32 {
    level(brightness).ceil() as u32
}

/// The highest weighted squared brightness that is at most `brightness`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "float to int casts saturate, which is the intent for out of range thresholds"
)]
pub fn not_light_level(brightness: f64) -> u32 {
    level(brightness).floor() as u32
}

/// What one pass over an image's pixels finds.
pub struct PixelStats {
    /// Pixels at least as bright as [`Thresholds::bright`].
    pub bright: u64,
    /// Pixels at most as bright as [`Thresholds::not_light`].
    pub not_light: u64,
    /// Histograms of each channel's values; only the first is used for grayscale images.
    histograms: [[u64; 256]; 3],
    /// Channels per pixel: 1 or 3.
    channels: usize,
    /// The largest difference between a pixel's channels.
    pub spread: u8,
    /// The sum of the pixels' grayscale values.
    luma_sum: u64,
    /// The sum of the squares of the pixels' grayscale values.
    luma_squares: u64,
    /// The image shrunk to at most [`BLUR_SIZE`] on its longer side, in grayscale, by
    /// averaging the pixels that fall in each of its pixels.
    thumbnail: GrayImage,
}

impl PixelStats {
    /// Measures `pixels`, `width` by `height` pixels of `CHANNELS` bytes each: gray, or red,
    /// green and blue.
    pub fn measure<const CHANNELS: usize>(
        pixels: &[u8],
        width: u32,
        height: u32,
        thresholds: Thresholds,
    ) -> Self {
        let longest = width.max(height);
        let (small_width, small_height) = if longest > BLUR_SIZE {
            (
                scale(width, BLUR_SIZE, longest).max(1),
                scale(height, BLUR_SIZE, longest).max(1),
            )
        } else {
            (width, height)
        };
        let mut sums = vec![0_u64; small_width as usize * small_height as usize];

        let mut bright = 0;
        let mut not_light = 0;
        let mut histograms = Histograms::<CHANNELS>::new();
        let mut spread = 0;
        let mut luma_sum = 0;
        let mut luma_squares = 0;
        let mut luma_row = vec![0_u8; width as usize];
        // Each image column's grayscale sum over the rows of the current thumbnail row
        let mut column_sums = vec![0_u32; width as usize];
        // The first image column of each thumbnail column, and the end of the last
        let bounds: Vec<usize> = (0..=small_width)
            .map(|small| first(small, width, small_width) as usize)
            .collect();
        for (y, row) in (0..height).zip(pixels.chunks_exact(CHANNELS * width as usize)) {
            let row = row.as_chunks::<CHANNELS>().0;
            // Branch-free arithmetic per pixel, which the compiler vectorizes
            let (row_bright, row_not_light, row_spread) =
                brightness_row(row, &mut luma_row, thresholds);
            bright += u64::from(row_bright);
            not_light += u64::from(row_not_light);
            spread = spread.max(row_spread);
            histograms.count(row);
            // Summed in u32, which vectorizes better than u64: 65536 squares of up to 255²
            // fit in u32
            for (sums, lumas) in column_sums
                .chunks_mut(1 << 16)
                .zip(luma_row.chunks(1 << 16))
            {
                let (mut chunk_sum, mut chunk_squares) = (0_u32, 0_u32);
                for (sum, &luma) in sums.iter_mut().zip(lumas) {
                    let luma = u32::from(luma);
                    *sum += luma;
                    chunk_sum += luma;
                    chunk_squares += luma * luma;
                }
                luma_sum += u64::from(chunk_sum);
                luma_squares += u64::from(chunk_squares);
            }
            // Fold the column sums into the thumbnail row on its last image row
            let small_y = scale(y, small_height, height);
            if y + 1 == height || scale(y + 1, small_height, height) != small_y {
                let small_row = small_y as usize * small_width as usize;
                for (sum, bounds) in sums[small_row..].iter_mut().zip(bounds.windows(2)) {
                    *sum = column_sums[bounds[0]..bounds[1]]
                        .iter()
                        .map(|&sum| u64::from(sum))
                        .sum();
                }
                column_sums.fill(0);
            }
        }

        // Every thumbnail pixel covers the same number of image pixels, give or take one row
        // or column, so they are counted rather than tallied
        let mut thumbnail = GrayImage::new(small_width, small_height);
        for (x, y, value) in thumbnail.enumerate_pixels_mut() {
            let count = |small: u32, small_side: u32, side: u32| {
                // The image pixels `i` with `i * small_side / side == small`
                u64::from(first(small + 1, side, small_side) - first(small, side, small_side))
            };
            let pixels = count(x, small_width, width) * count(y, small_height, height);
            let sum = sums[y as usize * small_width as usize + x as usize];
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the rounded average of 8-bit values fits in 8 bits"
            )]
            let average = ((2 * sum + pixels) / (2 * pixels)) as u8;
            value.0 = [average];
        }

        Self {
            bright,
            not_light,
            histograms: histograms.totals(),
            channels: CHANNELS,
            spread,
            luma_sum,
            luma_squares,
            thumbnail,
        }
    }

    /// The RMS contrast, from 0 to 1, of an image of `pixels` pixels: the standard deviation
    /// of its grayscale values divided by 255.
    pub fn contrast(&self, pixels: f64) -> f64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "rounding sums beyond 2^53 changes the contrast negligibly"
        )]
        let (sum, squares) = (self.luma_sum as f64, self.luma_squares as f64);
        let mean = sum / pixels;
        mean.mul_add(-mean, squares / pixels).max(0.0).sqrt() / 255.0
    }

    /// The entropy, in bits, of the channels' histograms joined into one, as Pillow's
    /// `Image.entropy` measures it.
    pub fn entropy(&self) -> f64 {
        let histograms = &self.histograms[..self.channels];
        let total: u64 = histograms.iter().flatten().sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "pixel counts are far below 2^53"
        )]
        let total = total as f64;
        -histograms
            .iter()
            .flatten()
            .filter(|&&count| count > 0)
            .map(|&count| {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "pixel counts are far below 2^53"
                )]
                let probability = count as f64 / total;
                probability * probability.log2()
            })
            .sum::<f64>()
    }

    /// Whether the image's sharpness is below `sharpness`. Thumbnails too small to have an
    /// inside, or too flat to have edges, aren't blurry.
    pub fn is_blurry(&self, sharpness: f64) -> bool {
        let mut values = Moments::default();
        for pixel in self.thumbnail.pixels() {
            values.add(f64::from(pixel.0[0]));
        }
        laplacian_spread(&self.thumbnail)
            .is_some_and(|spread| values.std() >= FLAT_SPREAD && spread < sharpness)
    }
}

/// `value * to / from`, rounded down, for `value` at most `from`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the result is at most `to`, a u32"
)]
const fn scale(value: u32, to: u32, from: u32) -> u32 {
    (value as u64 * to as u64 / from as u64) as u32
}

/// The first of `0..side` that [`scale`] maps to `small` or beyond, out of `small_side`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the result is at most `side`, a u32"
)]
const fn first(small: u32, side: u32, small_side: u32) -> u32 {
    (small as u64 * side as u64).div_ceil(small_side as u64) as u32
}

/// A pixel's grayscale value, as Pillow's `convert("L")` computes it (ITU-R 601-2 luma).
const fn luma_value(r: u32, g: u32, b: u32) -> u32 {
    (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16
}

/// Histograms of each channel's values, counted in [`LANES`] interleaved sets.
///
/// Neighbouring pixels often have the same values, and incrementing the same count twice in a
/// row waits for the first increment to finish, so consecutive pixels are counted in
/// different sets, which are added up at the end.
struct Histograms<const CHANNELS: usize> {
    /// Counts by lane, channel and value. A lane counts at most a quarter of an image's
    /// pixels, and images of 2^34 pixels or more can't be decoded, so the counts fit in u32.
    lanes: Box<[[[u32; 256]; CHANNELS]; LANES]>,
}

impl<const CHANNELS: usize> Histograms<CHANNELS> {
    fn new() -> Self {
        Self {
            lanes: Box::new([[[0; 256]; CHANNELS]; LANES]),
        }
    }

    /// Counts the channel values of a row of pixels.
    fn count(&mut self, row: &[[u8; CHANNELS]]) {
        let (groups, rest) = row.as_chunks::<LANES>();
        for group in groups {
            for (lane, pixel) in self.lanes.iter_mut().zip(group) {
                for (histogram, &value) in lane.iter_mut().zip(pixel) {
                    histogram[usize::from(value)] += 1;
                }
            }
        }
        for (lane, pixel) in self.lanes.iter_mut().zip(rest) {
            for (histogram, &value) in lane.iter_mut().zip(pixel) {
                histogram[usize::from(value)] += 1;
            }
        }
    }

    /// The total counts, by channel and value, with unused channels left at 0.
    fn totals(&self) -> [[u64; 256]; 3] {
        let mut totals = [[0; 256]; 3];
        for lane in &*self.lanes {
            for (total, histogram) in totals.iter_mut().zip(lane) {
                for (total, &count) in total.iter_mut().zip(histogram) {
                    *total += u64::from(count);
                }
            }
        }
        totals
    }
}

/// Measures one row of `CHANNELS`-byte pixels, gray or red, green and blue: returns how many
/// are at least [`Thresholds::bright`] and at most [`Thresholds::not_light`], and the largest
/// difference between a pixel's channels, and fills `luma` with their grayscale values.
fn brightness_row<const CHANNELS: usize>(
    row: &[[u8; CHANNELS]],
    luma: &mut [u8],
    thresholds: Thresholds,
) -> (u32, u32, u8) {
    let mut bright = 0_u32;
    let mut not_light = 0_u32;
    let mut spread = 0_u8;
    for (pixel, luma) in row.iter().zip(luma) {
        let [r, g, b] = if CHANNELS == 1 {
            [pixel[0]; 3]
        } else {
            [pixel[0], pixel[1], pixel[2]]
        };
        let [wr, wg, wb] = [r, g, b].map(u32::from);
        let squared = BRIGHTNESS_WEIGHTS[0] * wr * wr
            + BRIGHTNESS_WEIGHTS[1] * wg * wg
            + BRIGHTNESS_WEIGHTS[2] * wb * wb;
        bright += u32::from(squared >= thresholds.bright);
        not_light += u32::from(squared <= thresholds.not_light);
        spread = spread.max(r.max(g).max(b) - r.min(g).min(b));
        #[expect(clippy::cast_possible_truncation, reason = "luma is at most 255")]
        let value = luma_value(wr, wg, wb) as u8;
        *luma = value;
    }
    (bright, not_light, spread)
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma};

    use super::*;

    /// A gray value's weighted squared brightness; the weights sum to 1000.
    const fn gray_squared(value: u32) -> u32 {
        1000 * value * value
    }

    #[test]
    fn brightness_levels_bracket_the_threshold() {
        // A gray of 81.6 is exactly 0.32 bright
        assert!(gray_squared(82) >= bright_level(0.32));
        assert!(gray_squared(81) < bright_level(0.32));
        assert!(gray_squared(242) <= not_light_level(0.95));
        assert!(gray_squared(243) > not_light_level(0.95));
        assert_eq!(bright_level(2.0), gray_squared(255) * 4);
    }

    #[test]
    fn grayscale_entropy_uses_one_histogram() {
        // Two equally common values: one bit, however many channels the image has
        let half = |x: u32| if x.is_multiple_of(2) { 0 } else { 255 };
        let gray = GrayImage::from_fn(8, 8, |x, _| Luma([half(x)]));
        let stats = PixelStats::measure::<1>(
            gray.as_raw(),
            8,
            8,
            Thresholds {
                bright: 0,
                not_light: 0,
            },
        );
        assert!((stats.entropy() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn thumbnail_averages_every_pixel_once() {
        // 600 by 20 shrinks to 256 by 8, unevenly; the average of a constant is the constant
        let stats = PixelStats::measure::<3>(
            &vec![77; 600 * 20 * 3],
            600,
            20,
            Thresholds {
                bright: 0,
                not_light: 0,
            },
        );
        assert_eq!(stats.thumbnail.dimensions(), (256, 8));
        assert!(stats.thumbnail.pixels().all(|pixel| pixel.0 == [77]));
    }
}
