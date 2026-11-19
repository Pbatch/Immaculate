//! Why a search failed.

use std::io;
use std::path::{Path, PathBuf};

use pyo3::PyErr;
use pyo3::exceptions::{PyNotADirectoryError, PyOSError, PyRuntimeError};

pub enum DedupError {
    Io(PathBuf, io::Error),
    NotADirectory(PathBuf),
    /// The near-duplicate method failed on this image in a way that would fail for any image.
    Hash(PathBuf, String),
    /// The search was stopped early through [`Progress::cancelled`](crate::progress::Progress::cancelled).
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
        let path = err.path().map_or_default(Path::to_owned);
        let err = err
            .into_io_error()
            .unwrap_or_else(|| io::Error::other("filesystem loop detected"));
        Self::Io(path, err)
    }
}
