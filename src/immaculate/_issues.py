import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from os import PathLike
from pathlib import Path

from immaculate import _core
from immaculate._checks import (
    CHECK_TYPES,
    CHECKS,
    Blurry,
    Check,
    Dark,
    Grayscale,
    IssueName,
    Light,
    LowContrast,
    LowInformation,
    NearDuplicates,
    OddAspectRatio,
    OddSize,
)

SSCD_METHODS = ("sscd", "sscd_distilled")
"""The methods that need the `sscd` extra, which installs ONNX Runtime and their models."""


def _check_of[C: Check](checks: dict[str, Check], kind: type[C]) -> C | None:
    """The check of type `kind` in `checks`, if there is one."""
    check = checks.get(kind.name)
    return check if isinstance(check, kind) else None


@dataclass(frozen=True, kw_only=True, slots=True)
class Issues:
    """What `find_issues` found, with a field named after each check. A check's field is
    `None` if it didn't run, and otherwise holds paths, sorted."""

    exact_duplicates: list[list[Path]] | None
    """Sets of images with identical pixel data; see `ExactDuplicates`."""
    near_duplicates: list[list[Path]] | None
    """Sets of near-duplicate images; see `NearDuplicates`."""
    dark: list[Path] | None
    """Dark images; see `Dark`."""
    light: list[Path] | None
    """Light images; see `Light`."""
    blurry: list[Path] | None
    """Blurry images; see `Blurry`."""
    low_information: list[Path] | None
    """Images with little information; see `LowInformation`."""
    odd_aspect_ratio: list[Path] | None
    """Images with an odd aspect ratio; see `OddAspectRatio`."""
    odd_size: list[Path] | None
    """Images with an odd size compared to the others; see `OddSize`."""
    grayscale: list[Path] | None
    """Grayscale images; see `Grayscale`."""
    low_contrast: list[Path] | None
    """Images with low contrast; see `LowContrast`."""
    truncated: list[Path] | None
    """Images whose files end early; see `Truncated`."""
    skipped: list[Path]
    """Images that could not be read or decoded, or are empty, whatever checks ran. No other
    field includes them."""

    def to_dict(self) -> dict[str, list[str] | list[list[str]]]:
        """The fields of the checks that ran, and `skipped`, with paths as strings, ready
        for `json.dumps`."""
        output: dict[str, list[str] | list[list[str]]] = {}
        for field in fields(self):
            found = getattr(self, field.name)
            if found is None:
                continue
            if field.name in SET_ISSUES:
                output[field.name] = [[str(path) for path in images] for images in found]
            else:
                output[field.name] = [str(path) for path in found]
        return output


SET_ISSUES = ("exact_duplicates", "near_duplicates")
"""The issues whose fields hold sets of images rather than images."""


def resolve_checks(issues: Iterable[Check] | None) -> dict[str, Check]:
    """The checks to run, by name, from `find_issues`' `issues` argument.

    Raises `ValueError` for a check given twice, and `TypeError` for anything that isn't a
    check.
    """
    if issues is None:
        return {name: check() for name, check in CHECKS.items()}
    checks: dict[str, Check] = {}
    for check in issues:
        if not isinstance(check, CHECK_TYPES):
            message = f"Expected a check, such as Dark(), not {check!r}"
            raise TypeError(message)
        if check.name in checks:
            message = f"{check.name} is given more than once"
            raise ValueError(message)
        checks[check.name] = check
    return checks


def find_issues(
    input_dir: str | PathLike[str] | Iterable[str | PathLike[str]],
    issues: Iterable[Check] | None = None,
    *,
    progress: Callable[[int, int], object] | None = None,
) -> Issues:
    """Find duplicates and other issues in a folder of images or a list of image files.

    `input_dir` is either a folder, searched recursively for image files (bmp, gif, jpeg, jpg,
    png, tif, tiff, webp), or an iterable of image file paths, which are used as given
    whatever their extension; a path listed twice counts once.

    `issues` lists the checks to run, each with its settings, such as `Dark()` or
    `Dark(brightness=0.2)`. By default every check runs with its default settings:

    - `ExactDuplicates`: images with identical pixels.
    - `NearDuplicates`: near-duplicate images, such as resized or flipped copies.
    - `Dark` and `Light`: images that are almost all dark or light.
    - `Blurry`: blurry images.
    - `LowInformation`: images with few distinct colors, such as blank images.
    - `OddAspectRatio`: unusually long or tall images.
    - `OddSize`: images much larger or smaller than the others.
    - `Grayscale`: grayscale images.
    - `LowContrast`: washed-out, foggy or otherwise flat-looking images.
    - `Truncated`: images whose files end early, such as interrupted downloads.

    Each image is decoded once, however many checks run. The `Issues` returned has a field
    for each check, named after it, which is `None` if the check didn't run. Duplicate sets
    hold two or more paths, sorted, and are ordered by their first path.

    `progress`, if given, is called as `progress(done, total)` about every 0.1 seconds while
    images are checked, where `done` of the `total` images found are finished, and once more
    when the search succeeds. It is only called when `done` has changed. If it raises, the
    search stops and the exception is raised. Pressing Ctrl-C also stops it.

    Raises `FileNotFoundError` or `NotADirectoryError` if `input_dir` is a path but not a
    folder, `OSError` if part of the folder cannot be read (listed files that cannot be read
    are skipped instead), `ValueError` for a check given twice, `TypeError` if `issues`
    holds anything that isn't a check, `ModuleNotFoundError` if the `"sscd"` or
    `"sscd_distilled"` method is used without the `sscd` extra, and
    `RuntimeError` if ONNX Runtime can't load the model.
    """
    checks = resolve_checks(issues)
    near = _check_of(checks, NearDuplicates)
    method = near.method if near else None
    model, onnxruntime = (
        (_sscd_model(method), _onnxruntime_library(method))
        if method in SSCD_METHODS
        else (None, None)
    )
    (
        exact_duplicates,
        near_duplicates,
        skipped,
        dark,
        light,
        blurry,
        low_information,
        odd_aspect_ratio,
        odd_size,
        grayscale,
        low_contrast,
        truncated,
    ) = _core.find_issues(
        input_dir,
        exact="exact_duplicates" in checks,
        near=method,
        model=model,
        onnxruntime=onnxruntime,
        dark=_check_of(checks, Dark),
        light=_check_of(checks, Light),
        blurry=_check_of(checks, Blurry),
        low_information=_check_of(checks, LowInformation),
        odd_aspect_ratio=_check_of(checks, OddAspectRatio),
        odd_size=_check_of(checks, OddSize),
        grayscale=_check_of(checks, Grayscale),
        low_contrast=_check_of(checks, LowContrast),
        truncated="truncated" in checks,
        progress=progress,
    )

    def ran[T](name: IssueName, found: T) -> T | None:
        return found if name in checks else None

    return Issues(
        exact_duplicates=ran("exact_duplicates", exact_duplicates),
        near_duplicates=ran("near_duplicates", near_duplicates),
        dark=ran("dark", dark),
        light=ran("light", light),
        blurry=ran("blurry", blurry),
        low_information=ran("low_information", low_information),
        odd_aspect_ratio=ran("odd_aspect_ratio", odd_aspect_ratio),
        odd_size=ran("odd_size", odd_size),
        grayscale=ran("grayscale", grayscale),
        low_contrast=ran("low_contrast", low_contrast),
        truncated=ran("truncated", truncated),
        skipped=skipped,
    )


def _sscd_model(method: str) -> Path:
    """The model `method` runs, from the `immaculate-onnx` package."""
    try:
        import immaculate_onnx  # noqa: PLC0415 - only installed with the sscd extra
    except ModuleNotFoundError:
        raise ModuleNotFoundError(_missing_sscd_extra(method)) from None
    return immaculate_onnx.MODELS[method]


def _onnxruntime_library(method: str) -> Path:
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
