import importlib.util
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Literal, NamedTuple

from immaculate import _core

type Method = Literal["pdq", "phash", "sscd", "sscd_distilled"]
"""How near duplicates are found; see `find_duplicates`."""

SSCD_METHODS = ("sscd", "sscd_distilled")
"""The methods that need the `sscd` extra, which installs ONNX Runtime and their models."""


class Duplicates(NamedTuple):
    exact: list[list[Path]]
    """Sets of images with identical pixel data."""
    near: list[list[Path]]
    """Sets of near-duplicate images, as matched by `method`, excluding sets already in
    `exact`."""
    skipped: list[Path]
    """Images that could not be decoded or are empty."""


def find_duplicates(
    input_dir: str | PathLike[str],
    method: Method = "pdq",
    *,
    progress: Callable[[int, int], object] | None = None,
) -> Duplicates:
    """Find exact and near-duplicate images in a folder.

    Searches `input_dir` recursively for image files (bmp, gif, jpeg, jpg, png, tif, tiff,
    webp). Each set holds two or more paths, sorted, and sets are ordered by their first
    path. Images with no duplicate are omitted.

    `method` chooses how near duplicates are found:

    - `"pdq"` (default): images whose 256-bit PDQ hashes (Meta's perceptual hash) differ in
      at most 31 bits, in any rotation by a multiple of 90° or flip. Catches resized,
      recompressed, rotated and flipped copies. Matches chain, so if A matches B and B
      matches C, all three are one set.
    - `"phash"`: images with identical 64-bit perceptual hashes, as in the `imagehash`
      library. Catches resized and recompressed copies.
    - `"sscd"`: images whose embeddings from Meta's SSCD copy detection model
      have cosine similarity of at least 0.75. Catches heavier edits than hashes, such as
      crops and overlays, but is much slower. The model, quantised to int8, and ONNX Runtime,
      which runs it, come with the `sscd` extra: `pip install "immaculate[sscd]"`. Matches
      chain, as with `"pdq"`.
    - `"sscd_distilled"`: as `"sscd"`, with the same 0.75 threshold, but using a small
      MobileNetV4 model trained to reproduce SSCD's embeddings. About 4x faster than
      `"sscd"`, at some cost in recall.

    `progress`, if given, is called as `progress(done, total)` about every 0.1 seconds while
    images are decoded and hashed, where `done` of the `total` images found are finished,
    and once more when the search succeeds. It is only called when `done` has changed. If it
    raises, the search stops and the exception is raised. Pressing Ctrl-C also stops it.

    Raises `FileNotFoundError` or `NotADirectoryError` if `input_dir` is not a folder,
    `OSError` if part of it cannot be read, `ValueError` for an unknown `method`,
    `ModuleNotFoundError` if `"sscd"` or `"sscd_distilled"` is used without the `sscd` extra,
    and `RuntimeError` if ONNX Runtime can't load the model.
    """
    if method in SSCD_METHODS:
        return Duplicates(
            *_core.find_duplicates(
                input_dir,
                method,
                _sscd_model(method),
                _onnxruntime_library(method),
                progress=progress,
            )
        )
    return Duplicates(*_core.find_duplicates(input_dir, method, progress=progress))


def _sscd_model(method: str) -> Path:
    """The model `method` runs, from the `immaculate-onnx` package."""
    try:
        import immaculate_onnx  # noqa: PLC0415 - only installed with the sscd extra
    except ModuleNotFoundError:
        raise ModuleNotFoundError(_missing_sscd_extra(method)) from None
    return immaculate_onnx.MODELS[method]


def _onnxruntime_library(method: str = "sscd") -> Path:
    """The C library in the `onnxruntime` package, found without importing the package."""
    spec = importlib.util.find_spec("onnxruntime")
    for folder in (spec.submodule_search_locations or []) if spec else []:
        # libonnxruntime.so.<version> on Linux, libonnxruntime.<version>.dylib on macOS, and
        # onnxruntime.dll on Windows
        libraries = sorted(
            path
            for path in (Path(folder) / "capi").glob("*onnxruntime.*")
            if ".so" in path.name or path.suffix in {".dylib", ".dll"}
        )
        if libraries:
            return libraries[0]
    raise ModuleNotFoundError(_missing_sscd_extra(method))


def _missing_sscd_extra(method: str) -> str:
    return (
        f'method "{method}" needs the sscd extra, which is missing; install it with: '
        'pip install "immaculate[sscd]"'
    )
