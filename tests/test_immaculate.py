import dataclasses
import importlib.metadata
import importlib.util
import io
import json
import os
import random
import re
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple, get_args

import pytest
from immaculate_onnx import SSCD_DISTILLED_MODEL, SSCD_MODEL
from PIL import Image, ImageDraw, ImageFilter, ImageOps

import immaculate
from immaculate import (
    Blurry,
    Check,
    Dark,
    ExactDuplicates,
    Grayscale,
    IssueName,
    Issues,
    Light,
    LowContrast,
    LowInformation,
    Method,
    NearDuplicates,
    OddAspectRatio,
    OddSize,
    Truncated,
    _core,
    find_issues,
)
from immaculate.__main__ import cli
from immaculate.__main__ import main as cli_main
from immaculate._checks import CHECKS
from immaculate._issues import _onnxruntime_library

METHODS: tuple[Method, ...] = get_args(Method.__value__)
HASH_METHODS = tuple(method for method in METHODS if method not in {"sscd", "sscd_distilled"})
"""SSCD (and its distilled student) scores `make_image`'s smooth, textureless images as copies
of each other (0.65-0.83 in Meta's own PyTorch model too), so tests that need them distinct use
only the hashes."""


class Duplicates(NamedTuple):
    exact: list[list[Path]]
    near: list[list[Path]]
    skipped: list[Path]


def find_duplicates(
    input_dir: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    method: Method = "pdq",
    *,
    progress: Callable[[int, int], object] | None = None,
) -> Duplicates:
    """`find_issues` looking only for duplicates, which most tests are about."""
    issues = find_issues(input_dir, [ExactDuplicates(), NearDuplicates(method)], progress=progress)
    assert issues.exact_duplicates is not None
    assert issues.near_duplicates is not None
    return Duplicates(issues.exact_duplicates, issues.near_duplicates, issues.skipped)


def make_image(seed: int, size: tuple[int, int] = (256, 192)) -> Image.Image:
    """Draw a deterministic, smooth image whose structure depends on `seed`."""
    image = Image.new("RGB", size, (40 * seed % 256, 90, 160))
    draw = ImageDraw.Draw(image)
    width, height = size
    for i in range(4):
        offset = (seed * 37 + i * 53) % 100
        box = (
            width * offset // 200,
            height * i // 5,
            width * (offset + 100) // 200,
            height * (i + 2) // 5,
        )
        draw.ellipse(box, fill=((seed * 70 + i * 60) % 256, (i * 90) % 256, (seed * 30) % 256))
    return image.filter(ImageFilter.GaussianBlur(4))


def save(image: Image.Image, path: Path, image_format: str | None = None, **params: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, image_format, **params)
    return path


# Hashes recorded from the current implementation, to catch any change in output, e.g. from
# optimizing it. They also match the reference implementations bit for bit: imagehash.phash
# for pHash, and Meta's pdqhash package for PDQ. The images cover box filter windows wider
# than 1 (large), odd and non-square sizes, images smaller than the 32x32 and 64x64 grids
# the hashes shrink to (tiny), and non-RGB modes.
HASH_IMAGES: dict[str, Callable[[], Image.Image]] = {
    "rgb-1": lambda: make_image(1),
    "rgb-2": lambda: make_image(2),
    "rgb-3": lambda: make_image(3),
    "odd-tall": lambda: make_image(4, (97, 301)),
    "large": lambda: make_image(5, (1000, 700)),
    "tiny": lambda: make_image(6, (10, 7)),
    "grayscale": lambda: make_image(7).convert("L"),
    "rgba": lambda: make_image(8).convert("RGBA"),
    "palette": lambda: make_image(9).convert("P"),
}
EXPECTED_HASHES = [
    (
        "rgb-1",
        0x9F3034C6C33CE1B3,
        "72e1e0d82c792659d8c3b1a3e78df0797274c3e68dc31f1ec3e1e1b13e066c06",
    ),
    (
        "rgb-2",
        0xE4C93E36912C99CD,
        "f4cc9d966cd837c632c65a38b0e45b3661b960d92b4c6c9a9344c836cd3f97c9",
    ),
    (
        "rgb-3",
        0xE51E3BC0CC6B349C,
        "b393c66335811331db0eac4eb5b49f33d999c09c4237396bc299f48116ee47bc",
    ),
    (
        "odd-tall",
        0xB7647193DA02B6C6,
        "db31966cf669971c1d361966580dc86c7e61f4f1a1b6eb60292dace4c2c79293",
    ),
    (
        "large",
        0xB0CB6EB4932CA3C9,
        "1a404196165c5266cd638dda5a716996aedc9dc9e962be1ee964b59626bb4a69",
    ),
    (
        "tiny",
        0xFEA35888B8CA3A35,
        "46ef92e2b910ed159e7083f7c66364dba3d39b243c0c9a25399c7c0866cf92ea",
    ),
    (
        "grayscale",
        0x924D6D968C6B33CC,
        "ccc6d643392c698ce69396335acb64ccb7244e9370e6a96b469896b4295b6dd9",
    ),
    (
        "rgba",
        0xC24E3DE0CC033B9F,
        "a74e1be4fa01ee23207ea19ebfd06447c2091e7c4deee5e014330083fede3ab9",
    ),
    (
        "palette",
        0xEEB132C689766991,
        "66cbcdb96a746cd981c313a6a64cd0b9e2648fc41dcb3a3795c843b16e36dc46",
    ),
]


@pytest.mark.parametrize(("name", "phash", "pdq"), EXPECTED_HASHES)
def test_hashes_are_unchanged(tmp_path: Path, name: str, phash: int, pdq: str) -> None:
    path = save(HASH_IMAGES[name](), tmp_path / f"{name}.png")

    assert _core.perceptual_hash(path) == phash
    assert _core.pdq_hash(path) == pdq


def test_version() -> None:
    assert immaculate.__version__ == "0.4.0"


def test_exact_duplicates_match_on_pixels_not_file_bytes(tmp_path: Path) -> None:
    image = make_image(1)
    png = save(image, tmp_path / "a.png")
    bmp = save(image, tmp_path / "b.bmp")
    nested = save(image, tmp_path / "nested" / "c.png", compress_level=1)
    save(make_image(2), tmp_path / "other.png")

    assert png.read_bytes() != nested.read_bytes()
    assert find_duplicates(tmp_path) == Duplicates(exact=[[png, bmp, nested]], near=[], skipped=[])


@pytest.mark.parametrize("method", METHODS)
def test_byte_identical_copies_share_results(tmp_path: Path, method: Method) -> None:
    image = make_image(1)
    original = save(image, tmp_path / "a.png")
    copy = tmp_path / "b.png"
    copy.write_bytes(original.read_bytes())
    pixel_copy = save(image, tmp_path / "c.png", compress_level=1)
    recompressed = save(image, tmp_path / "d.jpg", quality=90)
    corrupt = tmp_path / "e.png"
    corrupt.write_bytes(b"not an image")
    corrupt_copy = tmp_path / "f.png"
    corrupt_copy.write_bytes(b"not an image")

    result = find_duplicates(tmp_path, method)

    assert result.exact == [[original, copy, pixel_copy]]
    assert result.near == [[original, copy, pixel_copy, recompressed]]
    assert result.skipped == [corrupt, corrupt_copy]


@pytest.mark.parametrize("method", HASH_METHODS)
def test_near_duplicates(tmp_path: Path, method: Method) -> None:
    image = make_image(1)
    original = save(image, tmp_path / "a.png")
    resized = save(image.resize((128, 96)), tmp_path / "b.png")
    recompressed = save(image, tmp_path / "c.jpg", quality=90)
    save(make_image(2), tmp_path / "other.png")

    result = find_duplicates(tmp_path, method)

    assert result.exact == []
    assert result.near == [[original, resized, recompressed]]


def test_pdq_matches_rotations_and_flips(tmp_path: Path) -> None:
    image = make_image(1)
    original = save(image, tmp_path / "a.png")
    mirrored = save(ImageOps.mirror(image), tmp_path / "b.png")
    flipped = save(ImageOps.flip(image), tmp_path / "c.png")
    rotated = save(image.transpose(Image.Transpose.ROTATE_90), tmp_path / "d.png")
    transposed = save(image.transpose(Image.Transpose.TRANSPOSE), tmp_path / "e.png")
    save(make_image(2), tmp_path / "other.png")

    assert find_duplicates(tmp_path).near == [[original, mirrored, flipped, rotated, transposed]]
    assert find_duplicates(tmp_path, "phash").near == []


@pytest.mark.parametrize("method", METHODS)
def test_near_set_identical_to_exact_set_is_dropped(tmp_path: Path, method: Method) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(image, tmp_path / "b.png")

    assert find_duplicates(tmp_path, method) == Duplicates(exact=[[a, b]], near=[], skipped=[])


@pytest.mark.parametrize("method", HASH_METHODS)
def test_distinct_images_are_not_duplicates(tmp_path: Path, method: Method) -> None:
    for seed in range(5):
        save(make_image(seed), tmp_path / f"{seed}.png")

    assert find_duplicates(tmp_path, method) == Duplicates(exact=[], near=[], skipped=[])


def test_unknown_method_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        find_duplicates(tmp_path, "ahash")  # ty: ignore[invalid-argument-type]


def test_method_type_matches_core() -> None:
    assert get_args(Method.__value__) == _core.METHODS


def test_docstrings_match_core() -> None:
    doc = " ".join((find_issues.__doc__ or "").split())
    assert f"({', '.join(_core.IMAGE_EXTENSIONS)})" in doc
    doc = " ".join((NearDuplicates.__doc__ or "").split())
    assert f"at most {_core.PDQ_THRESHOLD} bits" in doc
    assert f"at least {_core.SSCD_THRESHOLD:g}" in doc


def test_sscd_matches_crops(tmp_path: Path) -> None:
    # Unlike make_image's smooth images, fractals have the texture SSCD tells apart: the crop
    # scores 0.92 and the other region 0.25 or less, against a threshold of 0.75
    image = Image.effect_mandelbrot((256, 192), (-2.0, -1.2, 0.8, 1.2), 100).convert("RGB")
    original = save(image, tmp_path / "a.png")
    cropped = save(image.crop((32, 24, 224, 168)), tmp_path / "b.png")
    other = Image.effect_mandelbrot((256, 192), (-0.75, 0.05, -0.7, 0.12), 200)
    save(other.convert("RGB"), tmp_path / "other.png")

    for method in ("sscd", "sscd_distilled"):
        assert find_duplicates(tmp_path, method) == Duplicates(
            exact=[], near=[[original, cropped]], skipped=[]
        )
    assert find_duplicates(tmp_path).near == []


def test_sscd_models_are_packaged() -> None:
    assert SSCD_MODEL.is_file()
    assert SSCD_DISTILLED_MODEL.is_file()
    assert (SSCD_MODEL.parent / "SSCD_LICENSE").is_file()


@pytest.mark.parametrize("method", ["sscd", "sscd_distilled"])
def test_sscd_needs_onnxruntime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: Method
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    with pytest.raises(ModuleNotFoundError, match=re.escape('pip install "immaculate[sscd]"')):
        find_duplicates(tmp_path, method)


@pytest.mark.parametrize("method", ["sscd", "sscd_distilled"])
def test_sscd_needs_its_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: Method
) -> None:
    # None in sys.modules makes importing the package raise ModuleNotFoundError
    monkeypatch.setitem(sys.modules, "immaculate_onnx", None)
    with pytest.raises(ModuleNotFoundError, match=re.escape('pip install "immaculate[sscd]"')):
        find_duplicates(tmp_path, method)


def test_core_needs_a_model_for_sscd(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a model"):
        _core.find_issues(tmp_path, near="sscd", onnxruntime=_onnxruntime_library("sscd"))


def test_core_model_is_only_for_sscd(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only used by"):
        _core.find_issues(tmp_path, near="pdq", model=SSCD_MODEL)


def test_unloadable_model_raises(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="Couldn't load SSCD model"):
        _core.find_issues(
            tmp_path, near="sscd", model=model, onnxruntime=_onnxruntime_library("sscd")
        )


def test_skips_undecodable_images(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    save(make_image(1), tmp_path / "a.png")

    assert find_duplicates(tmp_path).skipped == [corrupt]


def test_ignores_non_image_files(tmp_path: Path) -> None:
    image = make_image(1)
    save(image, tmp_path / "a.png")
    save(image, tmp_path / "b.txt", "PNG")

    assert find_duplicates(tmp_path) == Duplicates(exact=[], near=[], skipped=[])


def test_extension_match_is_case_insensitive(tmp_path: Path) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.PNG")
    b = save(image, tmp_path / "b.Png")

    assert find_duplicates(tmp_path).exact == [[a, b]]


def test_accepts_str_path(tmp_path: Path) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(image, tmp_path / "b.png")

    assert find_duplicates(str(tmp_path)).exact == [[a, b]]


def test_list_of_files(tmp_path: Path) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(image, tmp_path / "sub" / "b.bmp")
    unlisted = save(image, tmp_path / "unlisted.png")
    no_extension = save(image, tmp_path / "c.data", "PNG")
    missing = tmp_path / "missing.png"

    result = find_duplicates([str(no_extension), b, a, a, missing])

    assert unlisted not in result.exact[0]
    assert result == Duplicates(exact=[[a, no_extension, b]], near=[], skipped=[missing])


def test_iterable_of_files(tmp_path: Path) -> None:
    image = make_image(1)
    paths = [save(image, tmp_path / f"{name}.png") for name in "ab"]

    assert find_duplicates(path for path in paths).exact == [paths]


def test_empty_list() -> None:
    calls: list[tuple[int, int]] = []
    result = find_duplicates([], progress=lambda done, total: calls.append((done, total)))
    assert result == Duplicates(exact=[], near=[], skipped=[])
    assert calls == [(0, 0)]


@pytest.mark.parametrize("method", METHODS)
def test_progress_ends_with_every_image_done(tmp_path: Path, method: Method) -> None:
    for seed in range(3):
        save(make_image(seed), tmp_path / f"{seed}.png")
    (tmp_path / "corrupt.png").write_bytes(b"not an image")
    (tmp_path / "copy.png").write_bytes((tmp_path / "0.png").read_bytes())
    images = 5
    calls: list[tuple[int, int]] = []

    find_duplicates(tmp_path, method, progress=lambda done, total: calls.append((done, total)))
    assert calls[-1] == (images, images)
    assert all(total == images for _, total in calls)
    assert [done for done, _ in calls] == sorted({done for done, _ in calls})


def test_progress_on_empty_folder(tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []
    find_duplicates(tmp_path, progress=lambda done, total: calls.append((done, total)))
    assert calls == [(0, 0)]


def test_progress_exception_is_raised(tmp_path: Path) -> None:
    save(make_image(1), tmp_path / "a.png")

    def fail(_done: int, _total: int) -> None:
        msg = "stop"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="stop"):
        find_duplicates(tmp_path, progress=fail)


def test_empty_folder(tmp_path: Path) -> None:
    assert find_duplicates(tmp_path) == Duplicates(exact=[], near=[], skipped=[])


def test_missing_folder_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError) as exc_info:
        find_duplicates(missing)

    assert exc_info.value.filename == str(missing)


def test_file_instead_of_folder_raises(tmp_path: Path) -> None:
    file = save(make_image(1), tmp_path / "a.png")

    with pytest.raises(NotADirectoryError):
        find_duplicates(file)


DUPLICATE_CHECKS = [ExactDuplicates(), NearDuplicates()]


def test_cli_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(image, tmp_path / "b.png")
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")

    assert cli_main(tmp_path, as_json=True, checks=DUPLICATE_CHECKS) == 0
    assert json.loads(capsys.readouterr().out) == {
        "exact_duplicates": [[str(a), str(b)]],
        "near_duplicates": [],
        "skipped": [str(corrupt)],
    }


def test_cli_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(image, tmp_path / "b.png")

    assert cli_main(tmp_path, as_json=False, checks=DUPLICATE_CHECKS) == 0
    assert capsys.readouterr().out == (
        f"Exact duplicates: 1 set\n\n  {a}\n  {b}\n\nNear duplicates: 0 sets\n"
    )


def test_cli_method(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(ImageOps.mirror(image), tmp_path / "b.png")

    assert cli_main(tmp_path, as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["near_duplicates"] == [[str(a), str(b)]]
    assert cli_main(tmp_path, as_json=True, checks=[NearDuplicates("phash")]) == 0
    assert json.loads(capsys.readouterr().out)["near_duplicates"] == []


def test_cli_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    save(make_image(1), tmp_path / "a.png")
    save(make_image(2), tmp_path / "b.png")

    assert cli_main(tmp_path, as_json=True, show_progress=True) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["exact_duplicates"] == []
    assert captured.err.endswith(f"\r[{'#' * 30}] 2/2 images\n")


def test_cli_missing_folder_with_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli_main(tmp_path / "missing", as_json=False, show_progress=True) == 1
    assert capsys.readouterr().err.startswith("immaculate: error:")


def test_cli_parses_arguments(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(ImageOps.mirror(image), tmp_path / "b.png")

    with pytest.raises(SystemExit) as exited:
        cli([str(tmp_path), "--json", "--method", "phash"])
    assert exited.value.code == 0
    assert json.loads(capsys.readouterr().out)["near_duplicates"] == []
    with pytest.raises(SystemExit) as exited:
        cli([str(tmp_path), "--json"])
    assert exited.value.code == 0
    assert json.loads(capsys.readouterr().out)["near_duplicates"] == [[str(a), str(b)]]


def test_cli_rejects_unknown_method(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        cli([str(tmp_path), "--method", "ahash"])
    assert exited.value.code == 2  # noqa: PLR2004 - argparse's usage error code
    assert "usage: immaculate" in capsys.readouterr().err


def test_command_is_installed() -> None:
    (command,) = importlib.metadata.entry_points(group="console_scripts", name="immaculate")
    assert command.load() is cli


def test_cli_closed_pipe(tmp_path: Path) -> None:
    image = make_image(1)
    save(image, tmp_path / "a.png")
    save(image, tmp_path / "b.png")
    read, write = os.pipe()
    os.close(read)

    with os.fdopen(write, "wb") as stdout:
        command = [sys.executable, "-m", "immaculate", str(tmp_path), "--json"]
        # The command is built from this interpreter and a temporary folder
        result = subprocess.run(command, stdout=stdout, stderr=subprocess.PIPE, check=False)  # noqa: S603
    assert result.returncode == 141  # noqa: PLR2004 - 128 + SIGPIPE, like other tools
    assert result.stderr == b""


def test_cli_missing_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(tmp_path / "missing", as_json=False) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("immaculate: error:")


def noise(seed: int, size: tuple[int, int] = (256, 192)) -> Image.Image:
    """A deterministic image of random pixels: sharp, varied and colorful."""
    width, height = size
    data = random.Random(seed).randbytes(width * height * 3)  # noqa: S311 - test data
    return Image.frombytes("RGB", size, data)


def property_images(folder: Path) -> dict[str, Path]:
    """One image with each image property issue, and two without any, in `folder`."""
    return {
        "normal": save(noise(1), folder / "normal.png"),
        "normal2": save(noise(2), folder / "normal2.png"),
        "dark": save(noise(3).point(lambda value: value // 5), folder / "dark.png"),
        "light": save(noise(4).point(lambda value: 245 + value % 11), folder / "light.png"),
        "blurry": save(make_image(1), folder / "blurry.png"),
        "low_information": save(
            Image.new("RGB", (256, 192), (200, 120, 60)), folder / "low_information.png"
        ),
        # The same area as the others, so its size isn't odd
        "odd_aspect_ratio": save(noise(5, (768, 64)), folder / "odd_aspect_ratio.png"),
        "odd_size": save(noise(6, (1024, 768)), folder / "odd_size.png"),
        "grayscale": save(noise(7).convert("L"), folder / "grayscale.png"),
        "gray_rgb": save(noise(8).convert("L").convert("RGB"), folder / "gray_rgb.png"),
        "low_contrast": save(
            noise(9).point(lambda value: 110 + value % 30), folder / "low_contrast.png"
        ),
    }


PROPERTY_ISSUES: tuple[IssueName, ...] = (
    "dark",
    "light",
    "blurry",
    "low_information",
    "odd_aspect_ratio",
    "odd_size",
    "grayscale",
    "low_contrast",
)


def test_property_checks_find_their_images(tmp_path: Path) -> None:
    images = property_images(tmp_path)

    issues = find_issues(tmp_path)

    expected = {name: [images[name]] for name in PROPERTY_ISSUES}
    expected["grayscale"] = sorted([images["grayscale"], images["gray_rgb"]])
    # Mostly dark, mostly light and blank images have low contrast too
    expected["low_contrast"] = sorted(
        images[name] for name in ("low_contrast", "dark", "light", "low_information")
    )
    assert {name: getattr(issues, name) for name in PROPERTY_ISSUES} == expected
    assert (issues.exact_duplicates, issues.near_duplicates, issues.skipped) == ([], [], [])


def test_issues_runs_only_the_checks_given(tmp_path: Path) -> None:
    images = property_images(tmp_path)
    copy = save(noise(3).point(lambda value: value // 5), tmp_path / "dark_copy.png")

    issues = find_issues(tmp_path, [Dark()])

    # The checks that didn't run found None, not nothing
    assert issues == Issues(
        exact_duplicates=None,
        near_duplicates=None,
        dark=[images["dark"], copy],
        light=None,
        blurry=None,
        low_information=None,
        odd_aspect_ratio=None,
        odd_size=None,
        grayscale=None,
        low_contrast=None,
        truncated=None,
        skipped=[],
    )


def test_near_duplicates_alone_keep_identical_sets(tmp_path: Path) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(image, tmp_path / "b.png")

    issues = find_issues(tmp_path, [NearDuplicates()])

    assert (issues.exact_duplicates, issues.near_duplicates) == (None, [[a, b]])


def test_settings_change_what_is_found(tmp_path: Path) -> None:
    images = property_images(tmp_path)

    def found(check: Check) -> list[Path]:
        """The images `check`, run alone, finds."""
        return getattr(find_issues(tmp_path, [check]), check.name)

    # 256 by 192 is 0.75, which only a ratio of 0.9 calls odd
    assert images["normal"] in found(OddAspectRatio(0.9))
    # The dark image's pixels are at most 0.2 bright, and most are brighter than 0.1
    assert found(Dark(brightness=0.1)) == []
    assert images["normal"] in found(Light(brightness=0.9, fraction=1))
    assert found(Blurry(sharpness=0)) == []
    assert found(LowInformation(entropy=0)) == []
    assert images["normal"] in found(Grayscale(tolerance=255))
    assert found(LowContrast(contrast=0)) == []
    assert images["normal"] in found(LowContrast(contrast=1))


def test_odd_size_widens_with_iqr_factor(tmp_path: Path) -> None:
    # Sides 100 to 130, so the quartiles are 110 and 130: 400 is 13.5 IQRs above the third
    paths = [
        save(noise(side, (side, side)), tmp_path / f"{side}.png")
        for side in (100, 110, 120, 130, 400)
    ]

    assert find_issues(paths, [OddSize()]).odd_size == [paths[-1]]
    assert find_issues(paths, [OddSize(iqr_factor=13)]).odd_size == [paths[-1]]
    assert find_issues(paths, [OddSize(iqr_factor=14)]).odd_size == []


@pytest.mark.parametrize("name", ["rgb-1", "large", "grayscale", "rgba", "palette"])
def test_contrast_matches_numpy(tmp_path: Path, name: str) -> None:
    image = HASH_IMAGES[name]()
    path = save(image, tmp_path / f"{name}.png")
    contrast = statistics.pstdev(image.convert("L").tobytes()) / 255

    assert find_issues([path], [LowContrast(contrast + 1e-9)]).low_contrast == [path]
    assert find_issues([path], [LowContrast(contrast - 1e-9)]).low_contrast == []


@pytest.mark.parametrize("name", ["rgb-1", "large", "grayscale"])
def test_entropy_matches_pillow(tmp_path: Path, name: str) -> None:
    image = HASH_IMAGES[name]()
    path = save(image, tmp_path / f"{name}.png")
    entropy = image.entropy()

    assert find_issues([path], [LowInformation(entropy + 1e-9)]).low_information == [path]
    assert find_issues([path], [LowInformation(entropy - 1e-9)]).low_information == []


def test_skipped_images_have_no_other_issues(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")

    issues = find_issues(tmp_path)

    assert issues.skipped == [corrupt, empty]
    assert all(getattr(issues, name) == [] for name in CHECKS)


def test_truncated_files_are_found(tmp_path: Path) -> None:
    def encoded(image_format: str) -> bytes:
        output = io.BytesIO()
        noise(1).save(output, image_format)
        return output.getvalue()

    jpeg, png, webp = encoded("JPEG"), encoded("PNG"), encoded("WEBP")
    whole = tmp_path / "whole.jpg"
    whole.write_bytes(jpeg)
    # Data after the end, as motion photos append their video
    motion = tmp_path / "motion.jpg"
    motion.write_bytes(jpeg + b"\xff\xd8 video")
    half_jpeg = tmp_path / "half.jpg"
    half_jpeg.write_bytes(jpeg[: len(jpeg) // 2])
    no_end_jpeg = tmp_path / "no_end.jpg"
    no_end_jpeg.write_bytes(jpeg[:-2])
    short_webp = tmp_path / "short.webp"
    short_webp.write_bytes(webp[:-2])
    # Truncated PNGs fail to decode, so are skipped instead
    half_png = tmp_path / "half.png"
    half_png.write_bytes(png[: len(png) // 2])

    issues = find_issues(tmp_path, [Truncated(), Dark()])

    assert issues.truncated == [half_jpeg, no_end_jpeg, short_webp]
    assert issues.skipped == [half_png]
    # The JPEG's missing half is filled in gray, but it is still checked
    assert issues.dark == []
    assert find_issues(tmp_path, [Dark()]).truncated is None


@pytest.mark.parametrize(
    ("check", "message"),
    [
        (lambda: Dark(brightness=2), "Dark.brightness must be from 0 to 1, not 2"),
        (lambda: Light(fraction=-0.5), "Light.fraction must be from 0 to 1"),
        (lambda: Blurry(sharpness=-1), "Blurry.sharpness must be at least 0"),
        (lambda: Grayscale(tolerance=256), "Grayscale.tolerance must be from 0 to 255"),
        (lambda: OddAspectRatio(ratio=1.5), "OddAspectRatio.ratio must be from 0 to 1"),
        (lambda: LowContrast(contrast=-0.1), "LowContrast.contrast must be from 0 to 1"),
        (lambda: NearDuplicates("ahash"), "Unknown method"),  # ty: ignore[invalid-argument-type]
    ],
)
def test_settings_out_of_range_raise(check: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        check()


@pytest.mark.parametrize(
    "issues",
    [
        [Dark(), Dark()],
        [Dark(), Dark(brightness=0.2)],
        [Dark(brightness=0.2), Blurry(), Dark()],
    ],
)
def test_check_given_twice_raises(tmp_path: Path, issues: list[Check]) -> None:
    with pytest.raises(ValueError, match="dark is given more than once"):
        find_issues(tmp_path, issues)


def test_invalid_issues_raise(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Expected a check, such as Dark\\(\\), not 'dark'"):
        find_issues(tmp_path, ["dark"])  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Expected a check"):
        find_issues(tmp_path, "dark")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Expected a check"):
        find_issues(tmp_path, [Dark])  # ty: ignore[invalid-argument-type]


def test_checks_cover_every_issue() -> None:
    names = get_args(IssueName.__value__)
    assert tuple(CHECKS) == names
    assert [field.name for field in dataclasses.fields(Issues)] == [*names, "skipped"]


def test_issues_to_dict_has_the_checks_that_ran(tmp_path: Path) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(image, tmp_path / "b.png")
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")

    issues = find_issues(tmp_path, [ExactDuplicates(), Dark()])

    assert issues.to_dict() == {
        "exact_duplicates": [[str(a), str(b)]],
        "dark": [],
        "skipped": [str(corrupt)],
    }
    json.dumps(issues.to_dict())


def test_issues_are_frozen_and_not_a_tuple(tmp_path: Path) -> None:
    issues = find_issues(tmp_path, [Dark()])

    with pytest.raises(dataclasses.FrozenInstanceError):
        issues.dark = []  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        Issues(None, None)  # ty: ignore[missing-argument, too-many-positional-arguments]
    assert not isinstance(issues, tuple)


def test_cli_issues_and_settings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    images = property_images(tmp_path)

    with pytest.raises(SystemExit) as exited:
        cli([str(tmp_path), "--json", "--issues", "dark", "odd_aspect_ratio"])
    assert exited.value.code == 0
    assert json.loads(capsys.readouterr().out) == {
        "dark": [str(images["dark"])],
        "odd_aspect_ratio": [str(images["odd_aspect_ratio"])],
        "skipped": [],
    }
    with pytest.raises(SystemExit) as exited:
        cli([str(tmp_path), "--json", "--issues", "dark", "--set", "dark.brightness=0.1"])
    assert exited.value.code == 0
    assert json.loads(capsys.readouterr().out)["dark"] == []


def test_cli_text_lists_images(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    images = property_images(tmp_path)

    assert cli_main(tmp_path, as_json=False, checks=[Dark(), OddSize()]) == 0
    assert capsys.readouterr().out == (
        f"Dark: 1 image\n  {images['dark']}\n\nOdd size: 1 image\n  {images['odd_size']}\n"
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--set", "dark.brightness"], "expected CHECK.FIELD=VALUE"),
        (["--set", "darkness.brightness=1"], "expected CHECK.FIELD=VALUE"),
        (["--set", "dark.shade=1"], "dark has no setting 'shade'"),
        (["--set", "dark.brightness=bright"], "expected float"),
        (["--set", "dark.brightness=2"], "Dark.brightness must be from 0 to 1"),
        (["--issues", "dark", "--set", "light.brightness=0.5"], "isn't one of the --issues"),
        (["--issues", "dark", "--method", "phash"], "isn't one of the --issues"),
        (["--set", "dark.fraction=0.1", "--set", "dark.fraction=0.2"], "more than once"),
    ],
)
def test_cli_rejects_bad_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], arguments: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as exited:
        cli([str(tmp_path), *arguments])
    assert exited.value.code == 2  # noqa: PLR2004 - argparse's usage error code
    assert message in capsys.readouterr().err
