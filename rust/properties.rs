//! Checks of each image's own properties, following `CleanVision`'s image property issues:
//! dark, light, blurry, low information, odd aspect ratio, odd size and grayscale, plus low
//! contrast, which `CleanVision` doesn't check.
//!
//! Most checks use the same measures and default thresholds as `CleanVision`, though images
//! are decoded by the image crate rather than Pillow. Two differ: blur is measured as the
//! spread of a thumbnail's Laplacian, which `CleanVision`'s edge score, dominated by the
//! image's border, doesn't do reliably, and grayscale also covers color images whose pixels
//! are all gray.

mod pixels;
mod settings;
mod stats;

use std::borrow::Cow;

use image::{DynamicImage, GrayImage};

use self::pixels::{PixelStats, Thresholds, bright_level, not_light_level};
pub use self::settings::{
    Blurry, Dark, Grayscale, Light, LowContrast, LowInformation, OddAspectRatio, OddSize,
};
use self::stats::percentile;
use crate::images::DecodedImage;

/// Which property checks to run, and their settings.
#[derive(Clone, Copy, Default)]
pub struct Settings {
    /// The dark check, if it runs.
    pub dark: Option<Dark>,
    /// The light check, if it runs.
    pub light: Option<Light>,
    /// The blurry check, if it runs.
    pub blurry: Option<Blurry>,
    /// The low information check, if it runs.
    pub low_information: Option<LowInformation>,
    /// The odd aspect ratio check, if it runs.
    pub odd_aspect_ratio: Option<OddAspectRatio>,
    /// The odd size check, if it runs. It compares images, so [`odd_sizes`] runs it.
    pub odd_size: Option<OddSize>,
    /// The grayscale check, if it runs.
    pub grayscale: Option<Grayscale>,
    /// The low contrast check, if it runs.
    pub low_contrast: Option<LowContrast>,
}

/// One image's results from the checks that run, all `false` for checks that don't.
#[derive(Clone, Copy, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per check, each with its own name"
)]
pub struct Measurement {
    /// Whether the image is dark.
    pub dark: bool,
    /// Whether the image is light.
    pub light: bool,
    /// Whether the image is blurry.
    pub blurry: bool,
    /// Whether the image has low information.
    pub low_information: bool,
    /// Whether the image has an odd aspect ratio.
    pub odd_aspect_ratio: bool,
    /// Whether the image is grayscale.
    pub grayscale: bool,
    /// Whether the image has low contrast.
    pub low_contrast: bool,
    /// The square root of the image's area, for [`odd_sizes`].
    pub side: f64,
}

impl Settings {
    /// Whether any check runs that needs each image's pixels, not just its dimensions.
    const fn needs_pixels(&self) -> bool {
        self.dark.is_some()
            || self.light.is_some()
            || self.blurry.is_some()
            || self.low_information.is_some()
            || self.grayscale.is_some()
            || self.low_contrast.is_some()
    }

    /// Runs the checks of one image on its own. `image` must not be empty.
    pub fn measure(&self, decoded: &DecodedImage) -> Measurement {
        let image = decoded.image();
        let (width, height) = (image.width(), image.height());
        let mut measurement = Measurement {
            side: f64::from(width).sqrt() * f64::from(height).sqrt(),
            ..Measurement::default()
        };
        if let Some(odd) = self.odd_aspect_ratio {
            let (short, long) = (width.min(height), width.max(height));
            measurement.odd_aspect_ratio = f64::from(short) / f64::from(long) < odd.ratio;
        }
        if !self.needs_pixels() {
            return measurement;
        }

        let thresholds = Thresholds {
            bright: self
                .dark
                .map_or(u32::MAX, |dark| bright_level(dark.brightness)),
            not_light: self
                .light
                .map_or(0, |light| not_light_level(light.brightness)),
        };
        let stats = if image.color().has_color() {
            PixelStats::measure::<3>(decoded.rgb().as_raw(), width, height, thresholds)
        } else {
            PixelStats::measure::<1>(luma8(image).as_raw(), width, height, thresholds)
        };
        let pixels = f64::from(width) * f64::from(height);
        #[expect(
            clippy::cast_precision_loss,
            reason = "pixel counts are far below 2^53"
        )]
        let fraction = |count: u64| count as f64 / pixels;

        measurement.dark = self
            .dark
            .is_some_and(|dark| fraction(stats.bright) < dark.fraction);
        measurement.light = self
            .light
            .is_some_and(|light| fraction(stats.not_light) < light.fraction);
        measurement.low_information = self
            .low_information
            .is_some_and(|low| stats.entropy() < low.entropy);
        measurement.grayscale = self
            .grayscale
            .is_some_and(|gray| stats.spread <= gray.tolerance);
        measurement.blurry = self
            .blurry
            .is_some_and(|blurry| stats.is_blurry(blurry.sharpness));
        measurement.low_contrast = self
            .low_contrast
            .is_some_and(|low| stats.contrast(pixels) < low.contrast);
        measurement
    }
}

/// The image as 8-bit grayscale, borrowed when it already is.
fn luma8(image: &DynamicImage) -> Cow<'_, GrayImage> {
    image
        .as_luma8()
        .map_or_else(|| Cow::Owned(image.to_luma8()), Cow::Borrowed)
}

/// Whether each image has an odd size: whether its [`Measurement::side`], one of `sides`,
/// is more than `iqr_factor` interquartile ranges below the first quartile or above the
/// third, with quartiles interpolated linearly as numpy's `percentile` does. When every
/// image's side is in the interquartile range, an image only has an odd size if its side
/// differs from the quartiles.
pub fn odd_sizes(sides: &[f64], odd: OddSize) -> Vec<bool> {
    let mut sorted = sides.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let (Some(q1), Some(q3)) = (percentile(&sorted, 0.25), percentile(&sorted, 0.75)) else {
        return Vec::new();
    };
    let iqr = q3 - q1;
    let (low, high) = (
        odd.iqr_factor.mul_add(-iqr, q1),
        odd.iqr_factor.mul_add(iqr, q3),
    );
    sides
        .iter()
        .map(|&side| side < low || side > high)
        .collect()
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma, Rgb, RgbImage};

    use super::*;

    const ALL: Settings = Settings {
        dark: Some(Dark {
            brightness: 0.32,
            fraction: 0.01,
        }),
        light: Some(Light {
            brightness: 0.95,
            fraction: 0.05,
        }),
        blurry: Some(Blurry { sharpness: 5.0 }),
        low_information: Some(LowInformation { entropy: 3.0 }),
        odd_aspect_ratio: Some(OddAspectRatio { ratio: 0.35 }),
        odd_size: Some(OddSize { iqr_factor: 3.0 }),
        grayscale: Some(Grayscale { tolerance: 2 }),
        low_contrast: Some(LowContrast { contrast: 0.05 }),
    };

    fn rgb(width: u32, height: u32, mut pixel: impl FnMut(u32, u32) -> [u8; 3]) -> DecodedImage {
        let image = RgbImage::from_fn(width, height, |x, y| Rgb(pixel(x, y)));
        DecodedImage::new(DynamicImage::ImageRgb8(image))
    }

    #[test]
    fn black_is_dark_and_white_is_light() {
        let black = ALL.measure(&rgb(64, 48, |_, _| [0, 0, 0]));
        assert!(black.dark && !black.light && black.low_information && black.grayscale);
        let white = ALL.measure(&rgb(64, 48, |_, _| [255, 255, 255]));
        assert!(!white.dark && white.light);
    }

    #[test]
    fn noise_is_sharp_and_informative() {
        let mut random = crate::test_util::XorShift(1);
        let noise = ALL.measure(&rgb(200, 150, |_, _| {
            let bits = random.next_u64().to_le_bytes();
            [bits[0], bits[1], bits[2]]
        }));
        assert!(!noise.blurry && !noise.low_information && !noise.grayscale);
        assert!(!noise.dark && !noise.light && !noise.low_contrast);
    }

    #[test]
    fn contrast_is_the_spread_of_grayscale_values() {
        // Half the pixels 0 and half 2v: a standard deviation of v
        let halves = |value: u8| {
            let image = GrayImage::from_fn(8, 8, |x, _| Luma([if x < 4 { 0 } else { value }]));
            ALL.measure(&DecodedImage::new(DynamicImage::ImageLuma8(image)))
                .low_contrast
        };
        // 0.05 * 255 = 12.75, so a spread of 12.5 is low and 13 isn't
        assert!(halves(25));
        assert!(!halves(26));
        assert!(ALL.measure(&rgb(4, 4, |_, _| [200, 10, 90])).low_contrast);
    }

    #[test]
    fn smooth_gradient_is_blurry() {
        #[expect(clippy::cast_possible_truncation, reason = "values are below 256")]
        let gradient = rgb(256, 192, |x, y| [x as u8, (y + 32) as u8, 128]);
        assert!(ALL.measure(&gradient).blurry);
    }

    #[test]
    fn gray_pixels_are_grayscale_within_tolerance() {
        let gray = DecodedImage::new(DynamicImage::ImageLuma8(GrayImage::from_pixel(
            4,
            4,
            Luma([100]),
        )));
        assert!(ALL.measure(&gray).grayscale);
        assert!(ALL.measure(&rgb(4, 4, |_, _| [100, 102, 101])).grayscale);
        assert!(!ALL.measure(&rgb(4, 4, |_, _| [100, 103, 101])).grayscale);
    }

    #[test]
    fn aspect_ratio_compares_short_side_to_long() {
        assert!(
            ALL.measure(&rgb(100, 34, |_, _| [9, 99, 199]))
                .odd_aspect_ratio
        );
        assert!(
            !ALL.measure(&rgb(35, 100, |_, _| [9, 99, 199]))
                .odd_aspect_ratio
        );
    }

    #[test]
    fn odd_sizes_are_outside_the_widened_quartiles() {
        let odd = OddSize { iqr_factor: 3.0 };
        let sides = [100.0, 100.0, 100.0, 120.0, 1000.0];
        // q1 = 100 and q3 = 120, so sides from 40 to 180 are normal
        assert_eq!(odd_sizes(&sides, odd), [false, false, false, false, true]);
        assert_eq!(
            odd_sizes(&[5.0, 5.0, 5.0, 6.0, 5.0, 5.0], odd),
            [false, false, false, true, false, false]
        );
        assert_eq!(odd_sizes(&[7.0], odd), [false]);
        assert_eq!(odd_sizes(&[], odd), Vec::<bool>::new());
    }
}
