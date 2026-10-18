//! Image deduplication, following `CleanVision`'s algorithm:
//!
//! - Exact duplicates have identical decoded pixel data, so the same image saved with
//!   different metadata or in a different lossless format still matches.
//! - Near duplicates have matching perceptual hashes, using one of the methods in
//!   [`Method`]. Near-duplicate sets identical to an exact set are dropped.
//! - Images that cannot be decoded are skipped.

mod pdq;
mod perceptual_hash;
mod phash;

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::Hash;
use std::io;
use std::path::{Path, PathBuf};

use image::{DynamicImage, ImageReader, RgbImage};
use pyo3::exceptions::{PyNotADirectoryError, PyOSError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use walkdir::WalkDir;
use xxhash_rust::xxh3::Xxh3Default;

use crate::pdq::Pdq;
use crate::perceptual_hash::PerceptualHash;
use crate::phash::PHash;

/// File extensions treated as images, compared case-insensitively.
const IMAGE_EXTENSIONS: &[&str] = &["bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"];

/// How near duplicates are found.
#[derive(Clone, Copy)]
enum Method {
    PHash,
    Pdq,
}

impl Method {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "phash" => Ok(Self::PHash),
            "pdq" => Ok(Self::Pdq),
            _ => Err(PyValueError::new_err(format!(
                "Unknown method {name:?}, expected \"phash\" or \"pdq\""
            ))),
        }
    }
}

enum DedupError {
    Io(PathBuf, io::Error),
    NotADirectory(PathBuf),
}

impl From<DedupError> for PyErr {
    fn from(err: DedupError) -> Self {
        match err {
            DedupError::Io(path, err) => {
                let filename = path.to_string_lossy().into_owned();
                match err.raw_os_error() {
                    // OSError(errno, strerror, filename) maps errno to the right subclass,
                    // e.g. FileNotFoundError or PermissionError.
                    Some(errno) => PyOSError::new_err((errno, err.to_string(), filename)),
                    None => PyOSError::new_err(format!("{filename}: {err}")),
                }
            }
            DedupError::NotADirectory(path) => PyNotADirectoryError::new_err(format!(
                "Not a directory: {}",
                path.to_string_lossy()
            )),
        }
    }
}

impl From<walkdir::Error> for DedupError {
    fn from(err: walkdir::Error) -> Self {
        let path = err.path().map(Path::to_owned).unwrap_or_default();
        let err = err
            .into_io_error()
            .unwrap_or_else(|| io::Error::other("filesystem loop detected"));
        Self::Io(path, err)
    }
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            IMAGE_EXTENSIONS
                .iter()
                .any(|known| known.eq_ignore_ascii_case(ext))
        })
}

/// Recursively lists image files under `input_dir`, sorted by path.
///
/// Symlinks are not followed.
fn find_images(input_dir: &Path) -> Result<Vec<PathBuf>, DedupError> {
    let metadata =
        fs::metadata(input_dir).map_err(|err| DedupError::Io(input_dir.to_owned(), err))?;
    if !metadata.is_dir() {
        return Err(DedupError::NotADirectory(input_dir.to_owned()));
    }

    let mut images = Vec::new();
    for entry in WalkDir::new(input_dir).sort_by_file_name() {
        let entry = entry?;
        if entry.file_type().is_file() && is_image(entry.path()) {
            images.push(entry.into_path());
        }
    }
    Ok(images)
}

/// The image as 8-bit RGB, borrowed when it already is, which most decoded images are,
/// rather than copied.
pub(crate) fn rgb8(image: &DynamicImage) -> Cow<'_, RgbImage> {
    image
        .as_rgb8()
        .map_or_else(|| Cow::Owned(image.to_rgb8()), Cow::Borrowed)
}

fn load_image(path: &Path) -> Option<DynamicImage> {
    ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .decode()
        .ok()
}

/// Hashes the decoded pixel data with 128-bit xxh3.
///
/// xxh3 isn't cryptographic, but identifying duplicates only needs accidental collisions to
/// be negligible, which 128 bits make them, and it is several times faster than a
/// cryptographic hash.
///
/// Unlike `CleanVision`, which hashes only the raw pixel bytes, this also hashes the
/// dimensions and pixel layout so that e.g. a 2x8 and a 4x4 image with the same bytes
/// don't count as duplicates.
fn pixel_hash(image: &DynamicImage) -> u128 {
    let color = image.color();
    let mut hasher = Xxh3Default::new();
    hasher.update(&image.width().to_le_bytes());
    hasher.update(&image.height().to_le_bytes());
    hasher.update(&[color.channel_count(), color.bytes_per_pixel()]);
    hasher.update(image.as_bytes());
    hasher.digest128()
}

struct ImageHashes<H> {
    pixels: u128,
    perceptual: H,
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

struct Duplicates {
    exact: Vec<Vec<PathBuf>>,
    near: Vec<Vec<PathBuf>>,
    skipped: Vec<PathBuf>,
}

fn find_duplicate_images<P: PerceptualHash>(
    input_dir: &Path,
    method: &P,
) -> Result<Duplicates, DedupError> {
    let images = find_images(input_dir)?;
    let hashes: Vec<Option<ImageHashes<P::Hash>>> = images
        .par_iter()
        .map(|path| {
            let image = load_image(path)?;
            Some(ImageHashes {
                pixels: pixel_hash(&image),
                perceptual: method.hash(&image)?,
            })
        })
        .collect();

    let loaded = || {
        hashes
            .iter()
            .enumerate()
            .filter_map(|(index, hashes)| hashes.as_ref().map(|hashes| (index, hashes)))
    };
    let exact = group_by_key(loaded().map(|(index, hashes)| (index, hashes.pixels)));
    let perceptual: Vec<(usize, &P::Hash)> = loaded()
        .map(|(index, hashes)| (index, &hashes.perceptual))
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
    let skipped = hashes
        .iter()
        .zip(&images)
        .filter(|(hashes, _)| hashes.is_none())
        .map(|(_, path)| path.clone())
        .collect();
    Ok(Duplicates {
        exact: to_paths(exact),
        near: to_paths(near),
        skipped,
    })
}

/// Find exact and near-duplicate images in a folder, finding near duplicates with `method`
/// ("pdq" or "phash").
///
/// Returns `(exact, near, skipped)`. See `immaculate.find_duplicates`.
#[pyfunction]
#[pyo3(signature = (input_dir, method = "pdq"))]
#[allow(clippy::type_complexity)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "pyo3 extracts arguments as owned values"
)]
fn find_duplicates(
    py: Python<'_>,
    input_dir: PathBuf,
    method: &str,
) -> PyResult<(Vec<Vec<PathBuf>>, Vec<Vec<PathBuf>>, Vec<PathBuf>)> {
    let method = Method::parse(method)?;
    let duplicates = py.detach(|| match method {
        Method::PHash => find_duplicate_images(&input_dir, &PHash),
        Method::Pdq => find_duplicate_images(&input_dir, &Pdq::default()),
    })?;
    Ok((duplicates.exact, duplicates.near, duplicates.skipped))
}

/// Compute an image's 64-bit pHash, or `None` if it can't be decoded.
#[pyfunction(name = "perceptual_hash")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "pyo3 extracts arguments as owned values"
)]
fn py_perceptual_hash(py: Python<'_>, path: PathBuf) -> Option<u64> {
    py.detach(|| load_image(&path).and_then(|image| PHash.hash(&image)))
}

/// Compute an image's PDQ hash as 64 hex digits, or `None` if it can't be decoded.
#[pyfunction(name = "pdq_hash")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "pyo3 extracts arguments as owned values"
)]
fn py_pdq_hash(py: Python<'_>, path: PathBuf) -> Option<String> {
    py.detach(|| {
        let orientations = Pdq::default().hash(&load_image(&path)?)?;
        Some(pdq::to_hex(&orientations[0]))
    })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(py_perceptual_hash, m)?)?;
    m.add_function(wrap_pyfunction!(py_pdq_hash, m)?)
}
