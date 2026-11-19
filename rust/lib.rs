//! Finding issues in a set of images, following `CleanVision`'s algorithms:
//!
//! - Exact duplicates have identical decoded pixel data, so the same image saved with
//!   different metadata or in a different lossless format still matches.
//! - Near duplicates have matching perceptual hashes, using one of the methods in
//!   `NearMethod`, or matching SSCD embeddings. When exact duplicates are found too,
//!   near-duplicate sets identical to an exact set are dropped.
//! - Dark, light, blurry, low information, odd aspect ratio, odd size, grayscale and low
//!   contrast images are found from each image's properties; see the `properties` module.
//! - Truncated images, whose files end early, are found from the files' structure; see
//!   the `truncation` module.
//! - Images that cannot be decoded, or are empty, are skipped.

mod error;
mod images;
mod near;
mod progress;
mod properties;
#[cfg(test)]
mod test_util;
mod truncation;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

use crate::error::DedupError;
use crate::images::{
    DecodedImage, IMAGE_EXTENSIONS, Images, decode_image, first_identical_files, load_image,
    pixel_hash,
};
use crate::near::{NoHash, PHash, Pdq, PerceptualHash, Sscd, pdq, sscd};
use crate::progress::{Progress, run_with_progress};
use crate::properties::{
    Blurry, Dark, Grayscale, Light, LowContrast, LowInformation, Measurement, OddAspectRatio,
    OddSize,
};

/// The near-duplicate methods' names, as `find_issues` takes them.
const METHODS: &[&str] = &["pdq", "phash", "sscd", "sscd_distilled"];

/// How near duplicates are found, ready to run.
enum NearMethod {
    /// Near duplicates aren't looked for.
    None,
    PHash,
    Pdq,
    Sscd(Sscd),
}

/// Which issues to look for, besides near duplicates.
struct Checks {
    /// Whether to look for exact duplicates.
    exact: bool,
    /// Which image property checks to run.
    properties: properties::Settings,
    /// Whether to look for truncated files.
    truncated: bool,
}

/// What [`analyse_image`] finds about one image.
struct ImageResults<H> {
    /// The pixel hash, if exact duplicates are looked for.
    pixels: Option<u128>,
    perceptual: H,
    properties: Measurement,
    /// Whether the file is truncated, if truncated files are looked for.
    truncated: bool,
}

/// Hashes and measures the image at `path`, or returns `None` if it can't be decoded or is
/// empty.
fn analyse_image<P: PerceptualHash>(
    path: &Path,
    method: &P,
    checks: &Checks,
) -> Result<Option<ImageResults<P::Hash>>, DedupError> {
    let Ok(bytes) = fs::read(path) else {
        return Ok(None);
    };
    let truncated = checks.truncated && truncation::is_truncated(&bytes);
    let image = decode_image(path, &bytes);
    drop(bytes);
    let Some(image) = image else {
        return Ok(None);
    };
    if image.width() == 0 || image.height() == 0 {
        return Ok(None);
    }
    // Converted to 8-bit RGB at most once, for the method and the checks to share
    let image = DecodedImage::new(image);
    let perceptual = method
        .hash(&image)
        .map_err(|err| DedupError::Hash(path.to_owned(), err.to_string()))?;
    Ok(perceptual.map(|perceptual| ImageResults {
        pixels: checks.exact.then(|| pixel_hash(image.image())),
        perceptual,
        properties: checks.properties.measure(&image),
        truncated,
    }))
}

/// Groups indices that share a key into sets of two or more.
///
/// Groups are ordered by their first index, and indices within a group are ascending.
pub(crate) fn group_by_key<K: Hash + Eq>(
    keys: impl Iterator<Item = (usize, K)>,
) -> Vec<Vec<usize>> {
    let mut by_key: HashMap<K, Vec<usize>> = HashMap::new();
    for (index, key) in keys {
        by_key.entry(key).or_default().push(index);
    }
    let mut groups: Vec<Vec<usize>> = by_key
        .into_values()
        .filter(|group| group.len() > 1)
        .map(|mut group| {
            group.sort_unstable();
            group
        })
        .collect();
    groups.sort_unstable_by_key(|group| group[0]);
    groups
}

/// Sets of duplicate images' paths.
type PathSets = Vec<Vec<PathBuf>>;

/// Everything a search finds, each list sorted by path. Checks that didn't run find nothing.
struct Issues {
    exact: PathSets,
    near: PathSets,
    skipped: Vec<PathBuf>,
    dark: Vec<PathBuf>,
    light: Vec<PathBuf>,
    blurry: Vec<PathBuf>,
    low_information: Vec<PathBuf>,
    odd_aspect_ratio: Vec<PathBuf>,
    odd_size: Vec<PathBuf>,
    grayscale: Vec<PathBuf>,
    low_contrast: Vec<PathBuf>,
    truncated: Vec<PathBuf>,
}

/// The issues tuple `find_issues` returns, in `immaculate.Issues`' field order.
type IssuesTuple = (
    PathSets,
    PathSets,
    Vec<PathBuf>,
    Vec<PathBuf>,
    Vec<PathBuf>,
    Vec<PathBuf>,
    Vec<PathBuf>,
    Vec<PathBuf>,
    Vec<PathBuf>,
    Vec<PathBuf>,
    Vec<PathBuf>,
    Vec<PathBuf>,
);

impl From<Issues> for IssuesTuple {
    fn from(issues: Issues) -> Self {
        (
            issues.exact,
            issues.near,
            issues.skipped,
            issues.dark,
            issues.light,
            issues.blurry,
            issues.low_information,
            issues.odd_aspect_ratio,
            issues.odd_size,
            issues.grayscale,
            issues.low_contrast,
            issues.truncated,
        )
    }
}

fn find_image_issues<P: PerceptualHash>(
    images: Images,
    method: &P,
    checks: &Checks,
    progress: &Progress,
) -> Result<Issues, DedupError> {
    let images = images.list(&progress.cancelled)?;
    // Only this function sets it, once
    let _ = progress.total.set(images.len());
    // Byte-identical files are decoded and analysed once, through the first of them
    let firsts = first_identical_files(&images, &progress.cancelled)?;
    let mut copies = vec![0; images.len()];
    for &first in &firsts {
        copies[first] += 1;
    }
    // Only set for the first of each set of byte-identical files
    let results: Vec<Option<ImageResults<P::Hash>>> = images
        .par_iter()
        .zip(&copies)
        .map(|(path, &copies)| {
            if copies == 0 {
                return Ok(None);
            }
            if progress.cancelled.load(Ordering::Relaxed) {
                return Err(DedupError::Cancelled);
            }
            let results = analyse_image(path, method, checks);
            progress.done.fetch_add(copies, Ordering::Relaxed);
            results
        })
        .collect::<Result<_, DedupError>>()?;
    let image_results = |index: usize| results[firsts[index]].as_ref();
    let loaded: Vec<(usize, &ImageResults<P::Hash>)> = (0..images.len())
        .filter_map(|index| image_results(index).map(|results| (index, results)))
        .collect();

    let exact = group_by_key(
        loaded
            .iter()
            .filter_map(|&(index, results)| results.pixels.map(|pixels| (index, pixels))),
    );
    let perceptual: Vec<(usize, &P::Hash)> = loaded
        .iter()
        .map(|&(index, results)| (index, &results.perceptual))
        .collect();
    let mut near = method.group(&perceptual);
    let exact_sets: HashSet<&Vec<usize>> = exact.iter().collect();
    near.retain(|group| !exact_sets.contains(group));

    let to_paths = |groups: Vec<Vec<usize>>| -> Vec<Vec<PathBuf>> {
        groups
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|index| images[index].clone())
                    .collect()
            })
            .collect()
    };
    let flagged = |flag: fn(&Measurement) -> bool| -> Vec<PathBuf> {
        loaded
            .iter()
            .filter(|(_, results)| flag(&results.properties))
            .map(|&(index, _)| images[index].clone())
            .collect()
    };
    let odd_size = checks.properties.odd_size.map_or_else(Vec::new, |odd| {
        let sides: Vec<f64> = loaded
            .iter()
            .map(|(_, results)| results.properties.side)
            .collect();
        loaded
            .iter()
            .zip(properties::odd_sizes(&sides, odd))
            .filter(|&(_, odd)| odd)
            .map(|(&(index, _), _)| images[index].clone())
            .collect()
    });
    let skipped = images
        .iter()
        .enumerate()
        .filter(|&(index, _)| image_results(index).is_none())
        .map(|(_, path)| path.clone())
        .collect();
    Ok(Issues {
        exact: to_paths(exact),
        near: to_paths(near),
        skipped,
        dark: flagged(|measurement| measurement.dark),
        light: flagged(|measurement| measurement.light),
        blurry: flagged(|measurement| measurement.blurry),
        low_information: flagged(|measurement| measurement.low_information),
        odd_aspect_ratio: flagged(|measurement| measurement.odd_aspect_ratio),
        odd_size,
        grayscale: flagged(|measurement| measurement.grayscale),
        low_contrast: flagged(|measurement| measurement.low_contrast),
        truncated: loaded
            .iter()
            .filter(|(_, results)| results.truncated)
            .map(|&(index, _)| images[index].clone())
            .collect(),
    })
}

/// Find issues in `input_dir`, a folder or a list of image files.
///
/// Exact duplicates are looked for if `exact` is true, truncated files if `truncated` is
/// true, and near duplicates if `near` is one
/// of [`METHODS`]. `"sscd"` and `"sscd_distilled"` need `model`, an SSCD-style ONNX model
/// (SSCD itself, or the `MobileNetV4` student distilled from it), and `onnxruntime`, the path
/// to ONNX Runtime's `onnxruntime` library to run it with. Each image property check runs if
/// its settings are given: any object with the attributes of the matching struct in
/// [`properties`]. `progress`, if given, is called as `progress(done, total)`; see
/// [`run_with_progress`].
///
/// Returns the fields of `immaculate.Issues` as a tuple. See `immaculate.find_issues`.
#[pyfunction]
#[pyo3(signature = (
    input_dir,
    *,
    exact = true,
    near = None,
    model = None,
    onnxruntime = None,
    dark = None,
    light = None,
    blurry = None,
    low_information = None,
    odd_aspect_ratio = None,
    odd_size = None,
    grayscale = None,
    low_contrast = None,
    truncated = false,
    progress = None,
))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "pyo3 extracts arguments as owned values"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "one keyword argument per check, as Python passes them"
)]
fn find_issues(
    py: Python<'_>,
    input_dir: Bound<'_, PyAny>,
    exact: bool,
    near: Option<&str>,
    model: Option<PathBuf>,
    onnxruntime: Option<PathBuf>,
    dark: Option<Dark>,
    light: Option<Light>,
    blurry: Option<Blurry>,
    low_information: Option<LowInformation>,
    odd_aspect_ratio: Option<OddAspectRatio>,
    odd_size: Option<OddSize>,
    grayscale: Option<Grayscale>,
    low_contrast: Option<LowContrast>,
    truncated: bool,
    progress: Option<Bound<'_, PyAny>>,
) -> PyResult<IssuesTuple> {
    let images = Images::extract(&input_dir)?;
    let method = match (near, model, onnxruntime) {
        (None, None, None) => NearMethod::None,
        (Some("phash"), None, None) => NearMethod::PHash,
        (Some("pdq"), None, None) => NearMethod::Pdq,
        (Some("sscd" | "sscd_distilled"), Some(model), Some(onnxruntime)) => NearMethod::Sscd(
            py.detach(|| Sscd::new(&onnxruntime, &model, rayon::current_num_threads()))
                .map_err(|err| {
                    PyRuntimeError::new_err(format!(
                        "Couldn't load SSCD model {}: {err}",
                        model.to_string_lossy()
                    ))
                })?,
        ),
        (Some(name @ ("sscd" | "sscd_distilled")), None, _) => {
            return Err(PyValueError::new_err(format!(
                "method {name:?} needs a model path"
            )));
        }
        (Some(name @ ("sscd" | "sscd_distilled")), _, None) => {
            return Err(PyValueError::new_err(format!(
                "method {name:?} needs ONNX Runtime's library path"
            )));
        }
        (None | Some("phash" | "pdq"), _, _) => {
            return Err(PyValueError::new_err(
                "model and onnxruntime are only used by methods \"sscd\" and \"sscd_distilled\"",
            ));
        }
        (Some(name), _, _) => {
            return Err(PyValueError::new_err(format!(
                "Unknown method {name:?}, expected one of {METHODS:?}"
            )));
        }
    };
    let checks = Checks {
        exact,
        properties: properties::Settings {
            dark,
            light,
            blurry,
            low_information,
            odd_aspect_ratio,
            odd_size,
            grayscale,
            low_contrast,
        },
        truncated,
    };
    let issues = run_with_progress(py, progress.as_ref(), |progress| match &method {
        NearMethod::None => find_image_issues(images, &NoHash, &checks, progress),
        NearMethod::PHash => find_image_issues(images, &PHash, &checks, progress),
        NearMethod::Pdq => find_image_issues(images, &Pdq::default(), &checks, progress),
        NearMethod::Sscd(sscd) => find_image_issues(images, sscd, &checks, progress),
    })?;
    Ok(issues.into())
}

/// Compute an image's 64-bit pHash, or `None` if it can't be decoded.
#[pyfunction(name = "perceptual_hash")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "pyo3 extracts arguments as owned values"
)]
fn py_perceptual_hash(py: Python<'_>, path: PathBuf) -> Option<u64> {
    py.detach(|| {
        let Ok(hash) = PHash.hash(&DecodedImage::new(load_image(&path)?));
        hash
    })
}

/// Compute an image's PDQ hash as 64 hex digits, or `None` if it can't be decoded.
#[pyfunction(name = "pdq_hash")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "pyo3 extracts arguments as owned values"
)]
fn py_pdq_hash(py: Python<'_>, path: PathBuf) -> Option<String> {
    py.detach(|| {
        let Ok(orientations) = Pdq::default().hash(&DecodedImage::new(load_image(&path)?));
        Some(pdq::to_hex(&orientations?[0]))
    })
}

#[pymodule(name = "_core")]
fn core_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IMAGE_EXTENSIONS", PyTuple::new(m.py(), IMAGE_EXTENSIONS)?)?;
    m.add("METHODS", PyTuple::new(m.py(), METHODS)?)?;
    m.add("PDQ_THRESHOLD", pdq::DEFAULT_THRESHOLD)?;
    m.add("SSCD_THRESHOLD", sscd::DEFAULT_THRESHOLD)?;
    m.add_function(wrap_pyfunction!(find_issues, m)?)?;
    m.add_function(wrap_pyfunction!(py_perceptual_hash, m)?)?;
    m.add_function(wrap_pyfunction!(py_pdq_hash, m)?)
}
