//! Image deduplication, following `CleanVision`'s algorithm:
//!
//! - Exact duplicates have identical decoded pixel data, so the same image saved with
//!   different metadata or in a different lossless format still matches.
//! - Near duplicates have matching perceptual hashes, using one of the methods in
//!   [`NearMethod`], or matching SSCD embeddings. Near-duplicate sets identical to an exact
//!   set are dropped.
//! - Images that cannot be decoded, or are empty, are skipped.

mod multi_index;
mod pdq;
mod perceptual_hash;
mod phash;
mod side_cache;
mod sscd;
#[cfg(test)]
mod test_util;

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::Hash;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use image::{DynamicImage, ImageReader, RgbImage};
use pyo3::exceptions::{PyNotADirectoryError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use walkdir::WalkDir;
use xxhash_rust::xxh3::{Xxh3Default, xxh3_128};

use crate::pdq::Pdq;
use crate::perceptual_hash::PerceptualHash;
use crate::phash::PHash;
use crate::sscd::Sscd;

/// File extensions treated as images, compared case-insensitively.
const IMAGE_EXTENSIONS: &[&str] = &["bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"];
/// The near-duplicate methods' names, as `find_duplicates` takes them.
const METHODS: &[&str] = &["pdq", "phash", "sscd", "sscd_distilled"];
/// How often a search reports progress and checks for Ctrl-C.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// How near duplicates are found, ready to run.
enum NearMethod {
    PHash,
    Pdq,
    Sscd(Sscd),
}

enum DedupError {
    Io(PathBuf, io::Error),
    NotADirectory(PathBuf),
    /// The near-duplicate method failed on this image in a way that would fail for any image.
    Hash(PathBuf, String),
    /// The search was stopped early through [`Progress::cancelled`].
    Cancelled,
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
            DedupError::Hash(path, err) => PyRuntimeError::new_err(format!(
                "Couldn't find near duplicates of {}: {err}",
                path.to_string_lossy()
            )),
            DedupError::Cancelled => PyRuntimeError::new_err("The search was cancelled"),
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
/// Symlinks are not followed. Stops early if `cancelled` is set.
fn find_images(input_dir: &Path, cancelled: &AtomicBool) -> Result<Vec<PathBuf>, DedupError> {
    let metadata =
        fs::metadata(input_dir).map_err(|err| DedupError::Io(input_dir.to_owned(), err))?;
    if !metadata.is_dir() {
        return Err(DedupError::NotADirectory(input_dir.to_owned()));
    }

    let mut images = Vec::new();
    for entry in WalkDir::new(input_dir).sort_by_file_name() {
        if cancelled.load(Ordering::Relaxed) {
            return Err(DedupError::Cancelled);
        }
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

/// For each image, the index of the first image whose file has the same contents, which is
/// its own index if no earlier file does.
///
/// Only files whose size another file shares are read, and their contents compared by 128-bit
/// xxh3, so a folder without byte-identical copies costs little more than a `stat` per file.
/// A file that can't be read or measured is treated as unique, to be skipped when it fails to
/// decode. Stops early if `cancelled` is set.
fn first_identical_files(
    images: &[PathBuf],
    cancelled: &AtomicBool,
) -> Result<Vec<usize>, DedupError> {
    let sizes: Vec<Option<u64>> = images
        .par_iter()
        .map(|path| fs::metadata(path).ok().map(|metadata| metadata.len()))
        .collect();
    let mut files_by_size: HashMap<u64, usize> = HashMap::new();
    for &size in sizes.iter().flatten() {
        *files_by_size.entry(size).or_default() += 1;
    }
    let contents: Vec<Option<(usize, u128)>> = images
        .par_iter()
        .zip(&sizes)
        .map(|(path, size)| {
            if cancelled.load(Ordering::Relaxed) {
                return Err(DedupError::Cancelled);
            }
            let shared = size.is_some_and(|size| files_by_size[&size] > 1);
            // The length is read again in case the file changed since it was measured
            Ok(shared
                .then(|| fs::read(path).ok())
                .flatten()
                .map(|bytes| (bytes.len(), xxh3_128(&bytes))))
        })
        .collect::<Result<_, DedupError>>()?;

    let mut first_with_contents: HashMap<(usize, u128), usize> = HashMap::new();
    Ok(contents
        .into_iter()
        .enumerate()
        .map(|(index, key)| {
            key.map_or(index, |key| {
                *first_with_contents.entry(key).or_insert(index)
            })
        })
        .collect())
}

struct ImageHashes<H> {
    pixels: u128,
    perceptual: H,
}

/// Hashes the image at `path`, or returns `None` if it can't be decoded or is empty.
fn hash_image<P: PerceptualHash>(
    path: &Path,
    method: &P,
) -> Result<Option<ImageHashes<P::Hash>>, DedupError> {
    let Some(image) = load_image(path) else {
        return Ok(None);
    };
    let perceptual = method
        .hash(&image)
        .map_err(|err| DedupError::Hash(path.to_owned(), err.to_string()))?;
    Ok(perceptual.map(|perceptual| ImageHashes {
        pixels: pixel_hash(&image),
        perceptual,
    }))
}

/// How far a search has got, shared between the threads doing it and the one reporting it.
#[derive(Default)]
struct Progress {
    /// Images found, set once the folder has been listed.
    total: OnceLock<usize>,
    /// Images hashed or skipped so far.
    done: AtomicUsize,
    /// Set to stop the search early, which then fails with [`DedupError::Cancelled`].
    cancelled: AtomicBool,
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

struct Duplicates {
    exact: PathSets,
    near: PathSets,
    skipped: Vec<PathBuf>,
}

fn find_duplicate_images<P: PerceptualHash>(
    input_dir: &Path,
    method: &P,
    progress: &Progress,
) -> Result<Duplicates, DedupError> {
    let images = find_images(input_dir, &progress.cancelled)?;
    // Only this function sets it, once
    let _ = progress.total.set(images.len());
    // Byte-identical files are decoded and hashed once, through the first of them
    let firsts = first_identical_files(&images, &progress.cancelled)?;
    let mut copies = vec![0; images.len()];
    for &first in &firsts {
        copies[first] += 1;
    }
    // Only set for the first of each set of byte-identical files
    let hashes: Vec<Option<ImageHashes<P::Hash>>> = images
        .par_iter()
        .zip(&copies)
        .map(|(path, &copies)| {
            if copies == 0 {
                return Ok(None);
            }
            if progress.cancelled.load(Ordering::Relaxed) {
                return Err(DedupError::Cancelled);
            }
            let hashes = hash_image(path, method);
            progress.done.fetch_add(copies, Ordering::Relaxed);
            hashes
        })
        .collect::<Result<_, DedupError>>()?;
    let image_hashes = |index: usize| hashes[firsts[index]].as_ref();

    let loaded =
        || (0..images.len()).filter_map(|index| image_hashes(index).map(|hashes| (index, hashes)));
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
    let skipped = images
        .iter()
        .enumerate()
        .filter(|&(index, _)| image_hashes(index).is_none())
        .map(|(_, path)| path.clone())
        .collect();
    Ok(Duplicates {
        exact: to_paths(exact),
        near: to_paths(near),
        skipped,
    })
}

/// Runs `search` on another thread, so that this one, attached to Python, can call
/// `callback(done, total)` every [`PROGRESS_INTERVAL`] while images are hashed, and once more
/// when the search succeeds. Only calls it when `done` has changed.
///
/// If the callback raises, or Ctrl-C is pressed, the search is cancelled and that exception
/// is raised.
fn run_with_progress<T: Send>(
    py: Python<'_>,
    callback: Option<&Bound<'_, PyAny>>,
    search: impl FnOnce(&Progress) -> Result<T, DedupError> + Send,
) -> PyResult<T> {
    let progress = Progress::default();
    let finished = AtomicBool::new(false);
    let caller = thread::current();
    let mut reported = None;
    let mut report = || -> PyResult<()> {
        let (Some(callback), Some(&total)) = (callback, progress.total.get()) else {
            return Ok(());
        };
        let done = progress.done.load(Ordering::Relaxed);
        if reported != Some(done) {
            callback.call1((done, total))?;
            reported = Some(done);
        }
        Ok(())
    };

    thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let result = search(&progress);
            finished.store(true, Ordering::Release);
            caller.unpark();
            result
        });
        let mut error = None;
        while !finished.load(Ordering::Acquire) {
            py.detach(|| thread::park_timeout(PROGRESS_INTERVAL));
            if error.is_none()
                && let Err(err) = py.check_signals().and_then(|()| report())
            {
                progress.cancelled.store(true, Ordering::Relaxed);
                error = Some(err);
            }
        }
        let result = worker
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        if let Some(err) = error {
            return Err(err);
        }
        let value = result?;
        report()?;
        Ok(value)
    })
}

/// Find exact and near-duplicate images in a folder, finding near duplicates with `method`
/// (`"pdq"`, `"phash"`, `"sscd"` or `"sscd_distilled"`). `"sscd"` and `"sscd_distilled"` need
/// `model`, an SSCD-style ONNX model (SSCD itself, or the `MobileNetV4` student distilled from
/// it), and `onnxruntime`, the path to ONNX Runtime's `onnxruntime` library to run it with.
/// `progress`, if given, is called as `progress(done, total)`; see `run_with_progress`.
///
/// Returns `(exact, near, skipped)`. See `immaculate.find_duplicates`.
#[pyfunction]
#[pyo3(signature = (input_dir, method = "pdq", model = None, onnxruntime = None, progress = None))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "pyo3 extracts arguments as owned values"
)]
fn find_duplicates(
    py: Python<'_>,
    input_dir: PathBuf,
    method: &str,
    model: Option<PathBuf>,
    onnxruntime: Option<PathBuf>,
    progress: Option<Bound<'_, PyAny>>,
) -> PyResult<(PathSets, PathSets, Vec<PathBuf>)> {
    let method = match (method, model, onnxruntime) {
        ("phash", None, None) => NearMethod::PHash,
        ("pdq", None, None) => NearMethod::Pdq,
        ("sscd" | "sscd_distilled", Some(model), Some(onnxruntime)) => NearMethod::Sscd(
            py.detach(|| Sscd::new(&onnxruntime, &model, rayon::current_num_threads()))
                .map_err(|err| {
                    PyRuntimeError::new_err(format!(
                        "Couldn't load SSCD model {}: {err}",
                        model.to_string_lossy()
                    ))
                })?,
        ),
        (name @ ("sscd" | "sscd_distilled"), None, _) => {
            return Err(PyValueError::new_err(format!(
                "method {name:?} needs a model path"
            )));
        }
        (name @ ("sscd" | "sscd_distilled"), _, None) => {
            return Err(PyValueError::new_err(format!(
                "method {name:?} needs ONNX Runtime's library path"
            )));
        }
        ("phash" | "pdq", _, _) => {
            return Err(PyValueError::new_err(
                "model and onnxruntime are only used by methods \"sscd\" and \"sscd_distilled\"",
            ));
        }
        (name, _, _) => {
            return Err(PyValueError::new_err(format!(
                "Unknown method {name:?}, expected one of {METHODS:?}"
            )));
        }
    };
    let duplicates = run_with_progress(py, progress.as_ref(), |progress| match &method {
        NearMethod::PHash => find_duplicate_images(&input_dir, &PHash, progress),
        NearMethod::Pdq => find_duplicate_images(&input_dir, &Pdq::default(), progress),
        NearMethod::Sscd(sscd) => find_duplicate_images(&input_dir, sscd, progress),
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
    py.detach(|| {
        let Ok(hash) = PHash.hash(&load_image(&path)?);
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
        let Ok(orientations) = Pdq::default().hash(&load_image(&path)?);
        Some(pdq::to_hex(&orientations?[0]))
    })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IMAGE_EXTENSIONS", PyTuple::new(m.py(), IMAGE_EXTENSIONS)?)?;
    m.add("METHODS", PyTuple::new(m.py(), METHODS)?)?;
    m.add("PDQ_THRESHOLD", pdq::DEFAULT_THRESHOLD)?;
    m.add("SSCD_THRESHOLD", sscd::DEFAULT_THRESHOLD)?;
    m.add_function(wrap_pyfunction!(find_duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(py_perceptual_hash, m)?)?;
    m.add_function(wrap_pyfunction!(py_pdq_hash, m)?)
}
