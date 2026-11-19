from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple, Protocol

type DuplicateSets = set[frozenset[Path]]
type Images = set[Path]
"""Images a check found, as resolved paths."""
type Found = "Duplicates | Images"
"""What a check found: duplicate sets for the duplicates check, images for any other."""

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

SSCD_ONNX_MODEL = Path(__file__).parents[1] / "data" / "models" / "sscd_disc_mixup.onnx"
"""SSCD as ONNX, made by dev/export_sscd_onnx.py."""

CHECKS = (
    "duplicates",
    "dark",
    "light",
    "blurry",
    "low_information",
    "odd_aspect_ratio",
    "odd_size",
    "grayscale",
    "low_contrast",
    "truncated",
    "corrupt",
)
"""Every check a `Checker` can implement, each as its method `find_<check>`."""


class Duplicates(NamedTuple):
    exact: DuplicateSets | None
    """Exact duplicate sets, or None when the library doesn't separate them from `near`."""
    near: DuplicateSets


class Checker(Protocol):
    """A library that finds issues in a folder of images, using at most `threads` CPU cores.

    Each check is a method, `find_<check>` for each of `CHECKS`, which a library implements
    by subclassing this and overriding it. The checks a library doesn't have raise
    `NotImplementedError`; `implements` tells which it has without running them.
    """

    name: str

    def __init__(self, threads: int) -> None:
        self.threads = threads

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        """The duplicate sets in `images_dir`."""
        raise NotImplementedError

    def find_dark(self, images_dir: Path) -> Images:
        """The dark images in `images_dir`."""
        raise NotImplementedError

    def find_light(self, images_dir: Path) -> Images:
        """The light images in `images_dir`."""
        raise NotImplementedError

    def find_blurry(self, images_dir: Path) -> Images:
        """The blurry images in `images_dir`."""
        raise NotImplementedError

    def find_low_information(self, images_dir: Path) -> Images:
        """The images with little information, such as blank ones, in `images_dir`."""
        raise NotImplementedError

    def find_odd_aspect_ratio(self, images_dir: Path) -> Images:
        """The unusually long or tall images in `images_dir`."""
        raise NotImplementedError

    def find_odd_size(self, images_dir: Path) -> Images:
        """The images in `images_dir` much larger or smaller than the others."""
        raise NotImplementedError

    def find_grayscale(self, images_dir: Path) -> Images:
        """The grayscale images in `images_dir`."""
        raise NotImplementedError

    def find_low_contrast(self, images_dir: Path) -> Images:
        """The low-contrast images in `images_dir`."""
        raise NotImplementedError

    def find_truncated(self, images_dir: Path) -> Images:
        """The images in `images_dir` whose files end early."""
        raise NotImplementedError

    def find_corrupt(self, images_dir: Path) -> Images:
        """The images in `images_dir` that can't be decoded at all."""
        raise NotImplementedError

    def find_issues(self, images_dir: Path, checks: list[str]) -> dict[str, Found]:
        """What each of `checks`, which the library must implement, finds in `images_dir`,
        run together as the library's users would run them. By default each runs in turn;
        libraries that can run several checks in one pass override this."""
        return {check: getattr(self, method(check))(images_dir) for check in checks}


def method(check: str) -> str:
    """The name of the `Checker` method that runs `check`."""
    if check not in CHECKS:
        msg = f"Unknown check {check!r}, expected one of {CHECKS}"
        raise ValueError(msg)
    return f"find_{check}"


def implements(checker: type[Checker], check: str) -> bool:
    """Whether `checker` overrides `check`'s method, rather than leaving it unimplemented."""
    name = method(check)
    return getattr(checker, name) is not getattr(Checker, name)


def to_sets(groups: Iterable[Iterable[str | Path]]) -> DuplicateSets:
    return {frozenset(Path(path).resolve() for path in group) for group in groups}


def to_images(paths: Iterable[str | Path]) -> Images:
    return {Path(path).resolve() for path in paths}
