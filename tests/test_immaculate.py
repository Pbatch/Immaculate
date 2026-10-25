import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest
from immaculate_onnx import SSCD_DISTILLED_MODEL, SSCD_MODEL
from PIL import Image, ImageDraw, ImageFilter, ImageOps

import immaculate
from immaculate import Duplicates, Method, _core, find_duplicates
from immaculate.__main__ import cli
from immaculate.__main__ import main as cli_main
from immaculate._duplicates import _onnxruntime_library

METHODS: tuple[Method, ...] = get_args(Method.__value__)
HASH_METHODS = tuple(method for method in METHODS if method not in {"sscd", "sscd_distilled"})
"""SSCD (and its distilled student) scores `make_image`'s smooth, textureless images as copies
of each other (0.65-0.83 in Meta's own PyTorch model too), so tests that need them distinct use
only the hashes."""


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
    assert immaculate.__version__ == "0.2.0"


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


def test_docstring_matches_core() -> None:
    doc = " ".join((find_duplicates.__doc__ or "").split())
    assert f"({', '.join(_core.IMAGE_EXTENSIONS)})" in doc
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
        _core.find_duplicates(tmp_path, "sscd", None, _onnxruntime_library())


def test_core_model_is_only_for_sscd(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only used by"):
        _core.find_duplicates(tmp_path, "pdq", SSCD_MODEL)


def test_unloadable_model_raises(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="Couldn't load SSCD model"):
        _core.find_duplicates(tmp_path, "sscd", model, _onnxruntime_library())


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


def test_cli_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(image, tmp_path / "b.png")
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")

    assert cli_main(tmp_path, as_json=True) == 0
    assert json.loads(capsys.readouterr().out) == {
        "exact": [[str(a), str(b)]],
        "near": [],
        "skipped": [str(corrupt)],
    }


def test_cli_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(image, tmp_path / "b.png")

    assert cli_main(tmp_path, as_json=False) == 0
    assert capsys.readouterr().out == (
        f"Exact duplicates: 1 set\n\n  {a}\n  {b}\n\nNear duplicates: 0 sets\n"
    )


def test_cli_method(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = make_image(1)
    a = save(image, tmp_path / "a.png")
    b = save(ImageOps.mirror(image), tmp_path / "b.png")

    assert cli_main(tmp_path, as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["near"] == [[str(a), str(b)]]
    assert cli_main(tmp_path, as_json=True, method="phash") == 0
    assert json.loads(capsys.readouterr().out)["near"] == []


def test_cli_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    save(make_image(1), tmp_path / "a.png")
    save(make_image(2), tmp_path / "b.png")

    assert cli_main(tmp_path, as_json=True, show_progress=True) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["exact"] == []
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
    assert json.loads(capsys.readouterr().out)["near"] == []
    with pytest.raises(SystemExit) as exited:
        cli([str(tmp_path), "--json"])
    assert exited.value.code == 0
    assert json.loads(capsys.readouterr().out)["near"] == [[str(a), str(b)]]


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
