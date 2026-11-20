from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path

from immaculate._checks import (
    Blurry,
    Dark,
    Grayscale,
    Light,
    LowContrast,
    LowInformation,
    Method,
    OddAspectRatio,
    OddSize,
)

IMAGE_EXTENSIONS: tuple[str, ...]
METHODS: tuple[Method, ...]
PDQ_THRESHOLD: int
SSCD_THRESHOLD: float

def find_issues(
    input_dir: str | PathLike[str] | Iterable[str | PathLike[str]],
    *,
    exact: bool = True,
    near: Method | None = None,
    model: str | PathLike[str] | None = None,
    onnxruntime: str | PathLike[str] | None = None,
    dark: Dark | None = None,
    light: Light | None = None,
    blurry: Blurry | None = None,
    low_information: LowInformation | None = None,
    odd_aspect_ratio: OddAspectRatio | None = None,
    odd_size: OddSize | None = None,
    grayscale: Grayscale | None = None,
    low_contrast: LowContrast | None = None,
    truncated: bool = False,
    progress: Callable[[int, int], object] | None = None,
) -> tuple[
    list[list[Path]],
    list[list[Path]],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
]: ...
def perceptual_hash(path: str | PathLike[str]) -> int | None: ...
def pdq_hash(path: str | PathLike[str]) -> str | None: ...
