from os import PathLike
from pathlib import Path

from immaculate._duplicates import Method

IMAGE_EXTENSIONS: tuple[str, ...]
METHODS: tuple[Method, ...]
PDQ_THRESHOLD: int
SSCD_THRESHOLD: float

def find_duplicates(
    input_dir: str | PathLike[str],
    method: Method = "pdq",
    model: str | PathLike[str] | None = None,
    onnxruntime: str | PathLike[str] | None = None,
) -> tuple[list[list[Path]], list[list[Path]], list[Path]]: ...
def perceptual_hash(path: str | PathLike[str]) -> int | None: ...
def pdq_hash(path: str | PathLike[str]) -> str | None: ...
