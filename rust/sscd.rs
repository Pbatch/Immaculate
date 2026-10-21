//! SSCD, Meta's self-supervised copy detection model, following
//! <https://github.com/facebookresearch/sscd-copy-detection>.
//!
//! Each image is embedded as an L2-normalised 512-dimensional vector by `sscd_disc_mixup`,
//! quantised to int8 and run with ONNX Runtime. Images match when their embeddings' cosine
//! similarity is at least `threshold`. Matches chain: if A matches B and B matches C, all three
//! are one set.
//!
//! Embeddings are stored as int8, each with its own scale, and similarities are computed from
//! those. That is four times less memory to read for every pair than f32, so grouping large
//! sets is several times faster. Similarities are within about 0.003 of f32 ones, so the odd
//! pair right at the threshold can land on the other side of it.

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use image::imageops::{self, FilterType};
use image::{DynamicImage, RgbImage};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::TensorRef;

use crate::perceptual_hash::{PerceptualHash, group_matching};
use crate::rgb8;

/// Length images' shorter side is resized to, as SSCD's README recommends.
const SHORT_SIDE: u32 = 288;
/// The threshold SSCD's README gives for `sscd_disc_mixup`: pairs at or above it are copies
/// with 90% precision on DISC2021.
pub const DEFAULT_THRESHOLD: f32 = 0.75;

/// The ONNX Runtime environment, made on first use. ONNX Runtime allows one per process, even
/// after it is dropped, so every [`Sscd`] shares it.
static ENVIRONMENT: Mutex<Option<Environment>> = Mutex::new(None);

/// SSCD, matching embeddings with cosine similarity of at least `threshold`.
pub struct Sscd {
    /// `Session::run` needs exclusive access, so images are embedded one at a time. Each run
    /// uses every thread, and the other threads decode images in the meantime.
    session: Mutex<Session>,
    /// Lowest cosine similarity at which two embeddings match.
    pub threshold: f32,
}

impl Sscd {
    /// Loads ONNX Runtime from `library` (its `onnxruntime` shared library), then the model
    /// from `model`, an ONNX model made by `dev/quantize_sscd_onnx.py` (SSCD) or
    /// `dev/export_mobilenetv4_onnx.py` (the distilled `MobileNetV4` student). It takes 8-bit RGB
    /// pixels already resized for SSCD, as a (1, height, width, 3) tensor, and normalises them
    /// itself.
    ///
    /// ONNX Runtime is loaded once per process; later calls use the library first loaded.
    pub fn new(library: &Path, model: &Path, threads: usize) -> Result<Self, String> {
        let environment = environment(library)?;
        let session = Session::builder(&environment)
            .map_err(|err| err.to_string())?
            .with_intra_threads(threads)
            .map_err(|err| err.to_string())?
            // Idle threads would otherwise busy-wait for the next run, taking CPU from the
            // threads decoding images
            .with_intra_op_spinning(false)
            .map_err(|err| err.to_string())?
            .commit_from_file(model)
            .map_err(|err| err.to_string())?;
        Ok(Self {
            session: Mutex::new(session),
            threshold: DEFAULT_THRESHOLD,
        })
    }
}

/// The process's ONNX Runtime environment, loading ONNX Runtime from `library` and making the
/// environment if this is the first call to succeed.
fn environment(library: &Path) -> Result<Environment, String> {
    // Held while the environment is made, so two threads can't both try to make one
    let mut shared = ENVIRONMENT.lock().unwrap_or_else(PoisonError::into_inner);
    let environment = if let Some(environment) = &*shared {
        environment.clone()
    } else {
        let created = ort::init_from(library)
            .map_err(|err| err.to_string())?
            .with_name("immaculate")
            .build()
            .map_err(|err| err.to_string())?;
        *shared = Some(created.clone());
        created
    };
    drop(shared);
    Ok(environment)
}

/// An embedding quantised to int8: each value is `values[i] * scale`.
pub struct Embedding {
    values: Vec<i8>,
    scale: f32,
}

impl Embedding {
    /// Quantises `embedding`, scaling its largest magnitude to 127.
    fn new(embedding: &[f32]) -> Self {
        let max = embedding
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        let scale = max / 127.0;
        let values = embedding
            .iter()
            .map(|value| {
                if scale > 0.0 {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "values round to within -127..=127"
                    )]
                    let quantised = (value / scale).round() as i8;
                    quantised
                } else {
                    0
                }
            })
            .collect();
        Self { values, scale }
    }

    /// Cosine similarity with `other`, as both are L2-normalised.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the dot product of 512 values is at most 512 * 127^2, exact in an f32"
    )]
    fn similarity(&self, other: &Self) -> f32 {
        dot_i8(&self.values, &other.values) as f32 * self.scale * other.scale
    }
}

impl PerceptualHash for Sscd {
    type Hash = Embedding;
    type Error = ort::Error;

    /// Returns `None` if the image is empty, and an error if the model fails to run.
    fn hash(&self, image: &DynamicImage) -> Result<Option<Embedding>, ort::Error> {
        let Some(resized) = resize(image) else {
            return Ok(None);
        };
        let shape = [1, resized.height() as usize, resized.width() as usize, 3];
        let input = TensorRef::from_array_view((shape, resized.as_raw().as_slice()))?;

        // Only poisoned if another thread panicked, which fails the whole search anyway
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs = session.run(ort::inputs![input])?;
        let (_, embedding) = outputs[0].try_extract_tensor::<f32>()?;
        let embedding = Embedding::new(embedding);
        drop(outputs);
        drop(session);
        Ok(Some(embedding))
    }

    fn group(&self, hashes: &[(usize, &Embedding)]) -> Vec<Vec<usize>> {
        group_matching(hashes, |a, b| a.similarity(b) >= self.threshold)
    }
}

/// The image as RGB with its shorter side resized to [`SHORT_SIDE`], as SSCD's README resizes
/// images, or `None` if it is empty.
fn resize(image: &DynamicImage) -> Option<RgbImage> {
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 {
        return None;
    }
    // As torchvision's Resize does, the longer side is scaled in proportion and truncated
    let scale = |long: u32, short: u32| {
        u32::try_from(u64::from(SHORT_SIDE) * u64::from(long) / u64::from(short)).ok()
    };
    let (new_width, new_height) = if width <= height {
        (SHORT_SIDE, scale(height, width)?)
    } else {
        (scale(width, height)?, SHORT_SIDE)
    };
    // Triangle is bilinear filtering, widened when shrinking, as PIL's bilinear resize is
    Some(imageops::resize(
        &*rgb8(image),
        new_width,
        new_height,
        FilterType::Triangle,
    ))
}

/// Dot product of int8 vectors.
///
/// Summed in 16 lanes of i16 products, which the compiler vectorises as multiply-adds.
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut lanes = [0_i32; 16];
    let ((a_chunks, a_tail), (b_chunks, b_tail)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    let tail: i32 = a_tail
        .iter()
        .zip(b_tail)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum();
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..16 {
            lanes[lane] += i32::from(i16::from(a[lane]) * i16::from(b[lane]));
        }
    }
    lanes.iter().sum::<i32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit vector of `len` values.
    fn unit(len: u16, value: impl Fn(f32) -> f32) -> Vec<f32> {
        let vector: Vec<f32> = (0..len).map(|i| value(f32::from(i))).collect();
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        vector.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn dot_i8_matches_naive_sum() {
        let a: Vec<i8> = (-127..=127).cycle().take(515).collect();
        let b: Vec<i8> = (-127..=127).rev().step_by(7).cycle().take(515).collect();
        let naive: i32 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| i32::from(x) * i32::from(y))
            .sum();
        assert_eq!(dot_i8(&a, &b), naive);
    }

    #[test]
    fn similarity_is_close_to_f32_cosine() {
        let a = unit(512, f32::sin);
        let b = unit(512, |i| 0.3_f32.mul_add(i.cos(), (i * 1.01).sin()));
        let exact: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let quantised = Embedding::new(&a).similarity(&Embedding::new(&b));
        assert!((quantised - exact).abs() < 0.01, "{quantised} vs {exact}");
        assert!((Embedding::new(&a).similarity(&Embedding::new(&a)) - 1.0).abs() < 0.01);
    }

    #[test]
    fn zero_embedding_matches_nothing() {
        let zero = Embedding::new(&[0.0; 512]);
        assert!(zero.similarity(&Embedding::new(&unit(512, f32::sin))).abs() < f32::EPSILON);
    }

    #[test]
    fn resize_resizes_shorter_side_to_288() {
        let landscape = DynamicImage::new_rgb8(640, 480);
        let size = resize(&landscape).map(|resized| resized.dimensions());
        assert_eq!(size, Some((384, 288)));

        let portrait = DynamicImage::new_rgb8(100, 333);
        let size = resize(&portrait).map(|resized| resized.dimensions());
        // 288 * 333 / 100 = 959.04, truncated
        assert_eq!(size, Some((288, 959)));
    }

    #[test]
    fn resize_rejects_empty_images() {
        assert!(resize(&DynamicImage::new_rgb8(0, 10)).is_none());
    }
}
