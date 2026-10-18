import importlib.util
from os import PathLike
from pathlib import Path
from typing import Literal, NamedTuple

from immaculate import _core

type Method = Literal["pdq", "phash", "sscd"]
"""How near duplicates are found; see `find_duplicates`."""

SSCD_MODEL = Path(__file__).parent / "models" / "sscd.onnx"
"""SSCD's `sscd_disc_mixup` model quantised to int8 for ONNX Runtime. Meta's weights are MIT
licensed; see `SSCD_LICENSE` in the same folder."""


class Duplicates(NamedTuple):
    exact: list[list[Path]]
    """Sets of images with identical pixel data."""
    near: list[list[Path]]
    """Sets of near-duplicate images, as matched by `method`, excluding sets already in
    `exact`."""
    skipped: list[Path]
    """Images that could not be decoded or are empty."""


def find_duplicates(input_dir: str | PathLike[str], method: Method = "pdq") -> Duplicates:
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
      crops and overlays, but is much slower. The model ships with Immaculate, quantised to
      int8, and is run with ONNX Runtime. Matches chain, as with `"pdq"`.

    Raises `FileNotFoundError` or `NotADirectoryError` if `input_dir` is not a folder,
    `OSError` if part of it cannot be read, `ValueError` for an unknown `method`,
    `ModuleNotFoundError` if `"sscd"` is used but ONNX Runtime, one of Immaculate's
    dependencies, is missing, and `RuntimeError` if ONNX Runtime can't load SSCD's model.
    """
    if method == "sscd":
        return Duplicates(
            *_core.find_duplicates(input_dir, method, SSCD_MODEL, _onnxruntime_library())
        )
    return Duplicates(*_core.find_duplicates(input_dir, method))


def _onnxruntime_library() -> Path:
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
    msg = (
        'method "sscd" needs ONNX Runtime, which is missing; install it with: '
        "pip install onnxruntime"
    )
    raise ModuleNotFoundError(msg)
