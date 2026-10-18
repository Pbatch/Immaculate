import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from loguru import logger

type DuplicatePairs = set[frozenset[Path]]


@dataclass(frozen=True)
class GroundTruth:
    duplicates: DuplicatePairs
    """Pairs of images that are duplicates."""
    ignored: DuplicatePairs = field(default_factory=set)
    """Ambiguous pairs, which count neither for nor against a library."""
    complete: bool = True
    """Whether every duplicate pair is labelled.

    If not, pairs a library finds that aren't labelled may still be duplicates, so precision
    can't be measured.
    """


class Dataset(ABC):
    """A benchmark image dataset, stored in its own folder under the data directory."""

    key: str
    """Folder name and command-line name."""
    name: str

    def prepare(self, data_dir: Path) -> Path:
        """Download or build the dataset's images if needed, returning their folder."""
        images_dir = data_dir / self.key / "images"
        if not images_dir.is_dir():
            logger.info(f"Preparing {self.name} ...")
            partial = images_dir.with_name(f"{images_dir.name}.partial")
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            self.build(partial, data_dir)
            partial.rename(images_dir)
        return images_dir

    @abstractmethod
    def build(self, images_dir: Path, data_dir: Path) -> None:
        """Write the dataset's images into the empty folder `images_dir`."""

    def ground_truth(self, images_dir: Path) -> GroundTruth | None:  # noqa: ARG002
        """Labelled duplicates for the images in `images_dir`, if the dataset has them."""
        return None


def fetch(url: str, path: Path) -> Path:
    """Download `url` to `path` unless it's already there."""
    if not path.exists():
        logger.info(f"Downloading {url} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.partial")
        urllib.request.urlretrieve(url, partial)  # noqa: S310 - fixed https URLs
        partial.rename(path)
    return path


def extract(archive: Path, destination: Path, subdir: str = "") -> None:
    """Extract the files under `subdir` in a zip or tar archive directly into `destination`."""
    logger.info(f"Extracting {archive} ...")
    prefix = f"{subdir}/" if subdir else ""
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zip_file:
                # ZipFile.extract strips absolute paths and ".." components
                members = [name for name in zip_file.namelist() if name.startswith(prefix)]
                zip_file.extractall(staging, members)  # noqa: S202
        else:
            with tarfile.open(archive) as tar:
                members = [member for member in tar if member.name.startswith(prefix)]
                tar.extractall(staging, members, filter="data")
        for path in (Path(staging) / subdir).iterdir():
            path.rename(destination / path.name)


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link `source` to `destination` to save space, copying where links aren't supported."""
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copyfile(source, destination)


def group_pairs(paths: Iterable[Path], group_of: Callable[[Path], Hashable]) -> DuplicatePairs:
    """Every pair of images in the same group."""
    groups: defaultdict[Hashable, list[Path]] = defaultdict(list)
    for path in paths:
        groups[group_of(path)].append(path.resolve())
    return {frozenset(pair) for group in groups.values() for pair in combinations(group, 2)}
