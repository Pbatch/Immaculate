//! Finding image files, reading them, and spotting files with identical contents.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use image::{DynamicImage, ImageFormat, ImageReader, RgbImage};
use pyo3::prelude::*;
use rayon::prelude::*;
use walkdir::WalkDir;
use xxhash_rust::xxh3::{Xxh3Default, xxh3_128};

use crate::error::DedupError;

/// File extensions treated as images, compared case-insensitively.
pub const IMAGE_EXTENSIONS: &[&str] = &["bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"];

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            IMAGE_EXTENSIONS
                .iter()
                .any(|known| known.eq_ignore_ascii_case(ext))
        })
}

/// What to search: a folder, or a list of image files.
pub enum Images {
    Folder(PathBuf),
    Files(Vec<PathBuf>),
}

impl Images {
    /// A folder from a `str` or `os.PathLike`, or the files from any other iterable of them.
    pub fn extract(images: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(folder) = images.extract::<PathBuf>() {
            return Ok(Self::Folder(folder));
        }
        let files = images
            .try_iter()?
            .map(|path| path?.extract::<PathBuf>())
            .collect::<PyResult<_>>()?;
        Ok(Self::Files(files))
    }

    /// The images to search, sorted by path: the image files under a folder, or the listed
    /// files without repeats.
    pub fn list(self, cancelled: &AtomicBool) -> Result<Vec<PathBuf>, DedupError> {
        match self {
            Self::Folder(folder) => find_images(&folder, cancelled),
            Self::Files(mut files) => {
                files.sort_unstable();
                files.dedup();
                Ok(files)
            }
        }
    }
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

/// A decoded image, and its pixels as 8-bit RGB, converted at most once and only if needed.
///
/// The near-duplicate methods and the property checks all read 8-bit RGB, so sharing one
/// conversion saves converting images decoded in another format, such as RGBA or palette
/// PNGs, once for each of them.
pub struct DecodedImage {
    image: DynamicImage,
    rgb: OnceCell<RgbImage>,
}

impl DecodedImage {
    pub const fn new(image: DynamicImage) -> Self {
        Self {
            image,
            rgb: OnceCell::new(),
        }
    }

    /// The image as decoded.
    pub const fn image(&self) -> &DynamicImage {
        &self.image
    }

    /// The image as 8-bit RGB: borrowed when it decoded as that, which most images do, and
    /// otherwise converted the first time it's asked for.
    pub fn rgb(&self) -> &RgbImage {
        self.image
            .as_rgb8()
            .unwrap_or_else(|| self.rgb.get_or_init(|| self.image.to_rgb8()))
    }
}

/// Decodes the image file at `path`, or returns `None` if it can't be read or decoded.
pub fn load_image(path: &Path) -> Option<DynamicImage> {
    decode_image(path, &fs::read(path).ok()?)
}

/// Decodes `bytes`, the contents of the image file at `path`, or returns `None` if they
/// can't be decoded. The format is guessed from the contents, or else from `path`'s
/// extension, as `ImageReader::open` does.
pub fn decode_image(path: &Path, bytes: &[u8]) -> Option<DynamicImage> {
    let mut reader = ImageReader::new(Cursor::new(bytes));
    if let Ok(format) = ImageFormat::from_path(path) {
        reader.set_format(format);
    }
    reader.with_guessed_format().ok()?.decode().ok()
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
pub fn pixel_hash(image: &DynamicImage) -> u128 {
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
pub fn first_identical_files(
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
