from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

type DuplicateSets = set[frozenset[Path]]

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class Duplicates(NamedTuple):
    exact: DuplicateSets | None
    """Exact duplicate sets, or None when the library doesn't separate them from `near`."""
    near: DuplicateSets


class Dedup(ABC):
    """A library that finds duplicate images in a folder, using at most `threads` CPU cores."""

    name: str

    def __init__(self, threads: int) -> None:
        self.threads = threads

    @abstractmethod
    def find_duplicates(self, images_dir: Path) -> Duplicates:
        """Return the duplicate sets found in `images_dir`."""


def to_sets(groups: Iterable[Iterable[str | Path]]) -> DuplicateSets:
    return {frozenset(Path(path).resolve() for path in group) for group in groups}
