"""The checks `find_issues` can run, and their settings."""

from dataclasses import dataclass
from typing import ClassVar, Literal, get_args

type Method = Literal["pdq", "phash", "sscd", "sscd_distilled"]
"""How near duplicates are found; see `NearDuplicates`."""

type IssueName = Literal[
    "exact_duplicates",
    "near_duplicates",
    "dark",
    "light",
    "blurry",
    "low_information",
    "odd_aspect_ratio",
    "odd_size",
    "grayscale",
    "low_contrast",
    "truncated",
]
"""A check's name, as in its `name`, the fields of `Issues` and the command line."""


def _check_range(check: object, field: str, low: float, high: float | None = None) -> None:
    """Raise `ValueError` unless the check's `field` is from `low` to `high`, inclusive."""
    value = getattr(check, field)
    if value < low or (high is not None and value > high):
        bounds = f"from {low} to {high}" if high is not None else f"at least {low}"
        message = f"{type(check).__name__}.{field} must be {bounds}, not {value}"
        raise ValueError(message)


@dataclass(frozen=True)
class ExactDuplicates:
    """Find sets of images with identical pixel data, so the same image saved with different
    metadata or in a different lossless format still matches."""

    name: ClassVar[IssueName] = "exact_duplicates"


@dataclass(frozen=True)
class NearDuplicates:
    """Find sets of near-duplicate images, as matched by `method`:

    - `"pdq"` (default): images whose 256-bit PDQ hashes (Meta's perceptual hash) differ in
      at most 31 bits, in any rotation by a multiple of 90° or flip. Catches resized,
      recompressed, rotated and flipped copies. Matches chain, so if A matches B and B
      matches C, all three are one set.
    - `"phash"`: images with identical 64-bit perceptual hashes, as in the `imagehash`
      library. Catches resized and recompressed copies.
    - `"sscd"`: images whose embeddings from Meta's SSCD copy detection model have cosine
      similarity of at least 0.75. Catches heavier edits than hashes, such as crops and
      overlays, but is much slower. The model, quantised to int8, and ONNX Runtime, which
      runs it, come with the `sscd` extra: `pip install "immaculate[sscd]"`. Matches chain,
      as with `"pdq"`.
    - `"sscd_distilled"`: as `"sscd"`, with the same 0.75 threshold, but using a small
      MobileNetV4 model trained to reproduce SSCD's embeddings. About 4x faster than
      `"sscd"`, at some cost in recall.

    When exact duplicates are found too, near-duplicate sets identical to an exact set are
    left out.
    """

    name: ClassVar[IssueName] = "near_duplicates"
    method: Method = "pdq"

    def __post_init__(self) -> None:
        if self.method not in get_args(Method.__value__):
            message = (
                f"Unknown method {self.method!r}, expected one of {get_args(Method.__value__)}"
            )
            raise ValueError(message)


@dataclass(frozen=True)
class Dark:
    """Find dark images: those where fewer than `fraction` of the pixels are at least
    `brightness` bright.

    A pixel's brightness, from 0 to 1, is `sqrt(0.241 R² + 0.691 G² + 0.068 B²) / 255`.
    """

    name: ClassVar[IssueName] = "dark"
    brightness: float = 0.32
    fraction: float = 0.01

    def __post_init__(self) -> None:
        _check_range(self, "brightness", 0, 1)
        _check_range(self, "fraction", 0, 1)


@dataclass(frozen=True)
class Light:
    """Find light images: those where fewer than `fraction` of the pixels are at most
    `brightness` bright, measured as in `Dark`."""

    name: ClassVar[IssueName] = "light"
    brightness: float = 0.95
    fraction: float = 0.05

    def __post_init__(self) -> None:
        _check_range(self, "brightness", 0, 1)
        _check_range(self, "fraction", 0, 1)


@dataclass(frozen=True)
class Blurry:
    """Find blurry images: those whose sharpness is below `sharpness`.

    Sharpness is the standard deviation of the Laplacian (4 times each pixel less its 4
    neighbours) of the image in grayscale, shrunk to at most 256 pixels on its longer side by
    averaging. The default flags images blurred by a Gaussian of about 2 of those pixels, but
    also some smooth, hazy scenes. Images too flat to have edges, which have low information
    instead, and images under 3 pixels wide or tall are never blurry.
    """

    name: ClassVar[IssueName] = "blurry"
    sharpness: float = 5.0

    def __post_init__(self) -> None:
        _check_range(self, "sharpness", 0)


@dataclass(frozen=True)
class LowInformation:
    """Find images with low information: those whose color histogram has an entropy below
    `entropy` bits.

    The histogram is the red, green and blue channels' 256-bin histograms joined into one,
    or the single gray channel's for grayscale images, as Pillow's `Image.entropy` measures.
    """

    name: ClassVar[IssueName] = "low_information"
    entropy: float = 3.0

    def __post_init__(self) -> None:
        _check_range(self, "entropy", 0)


@dataclass(frozen=True)
class OddAspectRatio:
    """Find images with an odd aspect ratio: those whose shorter side divided by their longer
    side is below `ratio`."""

    name: ClassVar[IssueName] = "odd_aspect_ratio"
    ratio: float = 0.35

    def __post_init__(self) -> None:
        _check_range(self, "ratio", 0, 1)


@dataclass(frozen=True)
class OddSize:
    """Find images with an odd size compared to the others: those whose size, the square root
    of the area, is more than `iqr_factor` interquartile ranges below the first quartile of
    all the images' sizes, or above the third. If at least half the images have the same
    size, any image of a different size is odd.
    """

    name: ClassVar[IssueName] = "odd_size"
    iqr_factor: float = 3.0

    def __post_init__(self) -> None:
        _check_range(self, "iqr_factor", 0)


@dataclass(frozen=True)
class Grayscale:
    """Find grayscale images: those stored as grayscale, or whose pixels' red, green and blue
    values all differ by at most `tolerance`."""

    name: ClassVar[IssueName] = "grayscale"
    tolerance: int = 2

    def __post_init__(self) -> None:
        _check_range(self, "tolerance", 0, 255)


@dataclass(frozen=True)
class LowContrast:
    """Find images with low contrast, such as washed-out or foggy ones: those whose RMS
    contrast, the standard deviation of their grayscale values divided by 255, is below
    `contrast`.

    Grayscale values are computed as by Pillow's `convert("L")`.
    """

    name: ClassVar[IssueName] = "low_contrast"
    contrast: float = 0.05

    def __post_init__(self) -> None:
        _check_range(self, "contrast", 0, 1)


@dataclass(frozen=True)
class Truncated:
    """Find truncated images, such as interrupted downloads: those whose file ends before
    its end-of-image marker (JPEG), its `IEND` chunk (PNG) or the length its header gives
    (WebP). Data after the end, such as a motion photo's video, doesn't matter.

    Truncated JPEGs still decode, with whatever is missing filled in gray, so they are
    checked like other images too. Truncated files in other formats usually fail to decode,
    so they are skipped instead.
    """

    name: ClassVar[IssueName] = "truncated"


type Check = (
    ExactDuplicates
    | NearDuplicates
    | Dark
    | Light
    | Blurry
    | LowInformation
    | OddAspectRatio
    | OddSize
    | Grayscale
    | LowContrast
    | Truncated
)
"""A check `find_issues` can run, with its settings."""

CHECK_TYPES: tuple[type[Check], ...] = get_args(Check.__value__)
"""Every check's class, in the order `find_issues` lists them."""

CHECKS: dict[str, type[Check]] = {check.name: check for check in CHECK_TYPES}
"""Each check's class, by name."""
