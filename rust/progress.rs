//! Reporting a search's progress to Python, and stopping it on Ctrl-C.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use pyo3::prelude::*;

use crate::error::DedupError;

/// How often a search reports progress and checks for Ctrl-C.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// How far a search has got, shared between the threads doing it and the one reporting it.
#[derive(Default)]
pub struct Progress {
    /// Images found, set once the folder has been listed.
    pub total: OnceLock<usize>,
    /// Images hashed or skipped so far.
    pub done: AtomicUsize,
    /// Set to stop the search early, which then fails with [`DedupError::Cancelled`].
    pub cancelled: AtomicBool,
}

/// Runs `search` on another thread, so that this one, attached to Python, can call
/// `callback(done, total)` every [`PROGRESS_INTERVAL`] while images are hashed, and once more
/// when the search succeeds. Only calls it when `done` has changed.
///
/// If the callback raises, or Ctrl-C is pressed, the search is cancelled and that exception
/// is raised.
pub fn run_with_progress<T: Send>(
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
