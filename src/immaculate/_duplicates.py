from os import PathLike
from pathlib import Path
from typing import Literal, NamedTuple

from immaculate import _core

type Method = Literal["pdq", "phash"]
"""How near duplicates are found; see `find_duplicates`."""


class Duplicates(NamedTuple):
    exact: list[list[Path]]
    """Sets of images with identical pixel data."""
    near: list[list[Path]]
    """Sets of images with matching perceptual hashes, excluding sets already in `exact`."""
    skipped: list[Path]
    """Images that could not be decoded."""


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

    Raises `FileNotFoundError` or `NotADirectoryError` if `input_dir` is not a folder,
    `OSError` if part of it cannot be read, and `ValueError` for an unknown `method`.
    """
    return Duplicates(*_core.find_duplicates(input_dir, method))
