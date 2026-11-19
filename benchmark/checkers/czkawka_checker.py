import hashlib
import json
import platform
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from checkers.base_checker import IMAGE_EXTENSIONS, Checker, Duplicates, Images, to_images, to_sets

VERSION = "12.0.2"
RELEASE_URL = f"https://github.com/qarmin/czkawka/releases/download/{VERSION}/{{asset}}"
ASSETS = {
    ("Linux", "x86_64"): (
        "linux_czkawka_cli_x86_64",
        "61e7e8ac3f42338957ddc0739df653b9df6ff2958cab3b4cce14663b39c40843",
    ),
    ("Linux", "aarch64"): (
        "linux_czkawka_cli_arm64",
        "3a65408ca30036cf1cd993112bde6288e13c870842d12131dee8b4e435325c38",
    ),
    ("Darwin", "arm64"): (
        "mac_czkawka_cli_arm64",
        "3362df5776b209b6365482768bc960e5a853f1b554787954c4a4c64e90bc2c75",
    ),
    ("Windows", "AMD64"): (
        "windows_czkawka_cli.exe",
        "eb7c2009d2dd49cf5202acbd65370caece0a3f819b9515e727cedb1ec088ed1a",
    ),
}
"""czkawka's prebuilt command-line binaries for each (system, machine), with their SHA-256
digests as GitHub lists them."""
TOOLS_DIR = Path(__file__).parents[1] / "data" / "tools" / f"czkawka-{VERSION}"


class CzkawkaChecker(Checker):
    """czkawka's similar image search and broken file check, run with its command-line tool at
    its default settings. Similar images are Gradient hashes of 16x16 bits, resized with the
    Nearest filter, matching at a difference of at most 5; broken files are checked as images
    only.

    With `dihedral`, images are also compared flipped and rotated by multiples of 90°
    (`--geometric-invariance mirror-flip-rotate90`). Overrides only what would skew the
    benchmark: its 16 KiB minimum file size, its hash cache (which makes repeat runs faster),
    and its extension list, which lacks some of the benchmark's. czkawka doesn't separate
    exact and near duplicates, so every set is reported as near. The binary is downloaded to
    benchmark/data/tools on first use. Raises ImportError on platforms without one.
    """

    def __init__(self, threads: int, *, dihedral: bool = False) -> None:
        super().__init__(threads)
        self.dihedral = dihedral
        self.name = f"czkawka {VERSION}{' dihedral' if dihedral else ''}"

    def run(self, tool: str, images_dir: Path, options: list[str]) -> Any:  # noqa: ANN401 - JSON
        """Run czkawka's `tool` on `images_dir` with `options`, returning the results it saves."""
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results.json"
            command = [
                str(self.binary()),
                tool,
                "--directories",
                str(images_dir.resolve()),
                "--thread-number",
                str(self.threads),
                "--disable-cache",
                "--allowed-extensions",
                ",".join(extension.lstrip(".") for extension in sorted(IMAGE_EXTENSIONS)),
                "--compact-file-to-save",
                str(results),
                # It exits with an error code when it finds anything otherwise
                "--ignore-error-code-on-found",
                "--do-not-print-results",
                "--do-not-print-messages",
                *options,
            ]
            subprocess.run(command, check=True)  # noqa: S603 - the binary's path is fixed
            return json.loads(results.read_text())

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        options = ["--minimal-file-size", "1"]
        if self.dihedral:
            options += ["--geometric-invariance", "mirror-flip-rotate90"]
        groups = self.run("image", images_dir, options)
        return Duplicates(
            exact=None, near=to_sets([entry["path"] for entry in group] for group in groups)
        )

    def find_corrupt(self, images_dir: Path) -> Images:
        # The broken files tool, checking images only
        entries = self.run("broken", images_dir, ["--checked-types", "IMAGE"])
        return to_images(entry["path"] for entry in entries)

    @staticmethod
    def binary() -> Path:
        """The command-line binary, downloaded and checked against its digest if missing."""
        asset = ASSETS.get((platform.system(), platform.machine()))
        if asset is None:
            msg = f"czkawka has no binary for {platform.system()} {platform.machine()}"
            raise ImportError(msg)
        name, digest = asset
        path = TOOLS_DIR / ("czkawka_cli.exe" if name.endswith(".exe") else "czkawka_cli")
        if not path.exists():
            with urllib.request.urlopen(RELEASE_URL.format(asset=name)) as response:  # noqa: S310
                data = response.read()
            if hashlib.sha256(data).hexdigest() != digest:
                msg = f"{name} doesn't match its SHA-256 digest"
                raise RuntimeError(msg)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".part")
            partial.write_bytes(data)
            partial.chmod(0o755)
            partial.replace(path)
        return path
