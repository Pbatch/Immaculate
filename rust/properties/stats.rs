//! Statistics the checks share: standard deviations, the Laplacian and percentiles.

use image::GrayImage;

/// The standard deviation of the image's Laplacian, 4 times each inner pixel less its 4
/// nearest neighbours, or `None` if the image has no inner pixels.
pub fn laplacian_spread(image: &GrayImage) -> Option<f64> {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 {
        return None;
    }
    let value = |x: u32, y: u32| i32::from(image.get_pixel(x, y).0[0]);
    let mut moments = Moments::default();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let laplacian = 4 * value(x, y)
                - value(x - 1, y)
                - value(x + 1, y)
                - value(x, y - 1)
                - value(x, y + 1);
            moments.add(f64::from(laplacian));
        }
    }
    Some(moments.std())
}

/// Running sums for a population standard deviation.
#[derive(Default)]
pub struct Moments {
    count: f64,
    sum: f64,
    sum_squares: f64,
}

impl Moments {
    pub fn add(&mut self, value: f64) {
        self.count += 1.0;
        self.sum += value;
        self.sum_squares = value.mul_add(value, self.sum_squares);
    }

    pub fn std(&self) -> f64 {
        if self.count == 0.0 {
            return 0.0;
        }
        let mean = self.sum / self.count;
        mean.mul_add(-mean, self.sum_squares / self.count)
            .max(0.0)
            .sqrt()
    }
}

/// The `fraction` percentile of `sorted`, interpolating linearly between the values either
/// side of it, or `None` if `sorted` is empty.
pub fn percentile(sorted: &[f64], fraction: f64) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "image counts are far below 2^53"
    )]
    let position = fraction * last as f64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the position is between 0 and the last index"
    )]
    let below = position.floor() as usize;
    let above = (below + 1).min(last);
    #[expect(
        clippy::cast_precision_loss,
        reason = "image counts are far below 2^53"
    )]
    let weight = position - below as f64;
    Some((sorted[above] - sorted[below]).mul_add(weight, sorted[below]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates_like_numpy() {
        let sorted = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(percentile(&sorted, 0.25), Some(1.75));
        assert_eq!(percentile(&sorted, 0.75), Some(5.0));
        assert_eq!(percentile(&[3.0], 0.75), Some(3.0));
    }
}
