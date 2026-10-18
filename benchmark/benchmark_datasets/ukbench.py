import random
from collections.abc import Callable
from pathlib import Path

from benchmark_datasets.base_dataset import (
    Dataset,
    GroundTruth,
    extract,
    fetch,
    group_pairs,
    link_or_copy,
)
from PIL import Image, ImageOps


class UKBench(Dataset):
    """10,200 photos of 2,550 objects, each taken four times from different angles (~1.6 GB).

    The four photos of an object count as duplicates of each other.
    """

    key = "ukbench"
    name = "UKBench"
    url = "https://archive.org/download/ukbench/ukbench.zip"

    def build(self, images_dir: Path, data_dir: Path) -> None:
        archive = fetch(self.url, data_dir / self.key / "ukbench.zip")
        extract(archive, images_dir, "full")
        archive.unlink()

    def ground_truth(self, images_dir: Path) -> GroundTruth:
        return GroundTruth(group_pairs(images_dir.iterdir(), ukbench_object))

    @staticmethod
    def image(images_dir: Path, obj: int) -> Path:
        """The first photo of object number `obj`."""
        return images_dir / f"ukbench{obj * 4:05d}.jpg"


def ukbench_object(path: Path) -> int:
    """UKBench images come in consecutive groups of four photos of the same object."""
    return int(path.stem.removeprefix("ukbench")) // 4


def derived_group(path: Path) -> str:
    """Images in datasets derived from UKBench are named `<group>_<variant>.jpg`."""
    return path.stem.split("_")[0]


def random_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = min(560, image.width), min(420, image.height)
    left = rng.randint(0, image.width - width)
    top = rng.randint(0, image.height - height)
    return image.crop((left, top, left + width, top + height))


TRANSFORMS: dict[str, Callable[[Image.Image, random.Random], Image.Image]] = {
    "crop": random_crop,
    "hflip": lambda image, _: ImageOps.mirror(image),
    "vflip": lambda image, _: ImageOps.flip(image),
    "rotate": lambda image, _: image.rotate(25),
    "square": lambda image, _: image.resize((image.height, image.height)),
}


class UKBenchTransformed(Dataset):
    """imagededup's transformed benchmark, built from UKBench (~1.3 GB).

    The first photo of each of the first 1,800 objects, plus five edited copies: a random
    560x420 crop, horizontal and vertical flips, a 25° rotation, and a resize to a square.
    Each photo and its copies count as duplicates of each other.
    """

    key = "ukbench-transformed"
    name = "UKBench transformed (imagededup)"
    objects = 1800

    def build(self, images_dir: Path, data_dir: Path) -> None:
        source_dir = UKBench().prepare(data_dir)
        for obj in range(self.objects):
            source = UKBench.image(source_dir, obj)
            link_or_copy(source, images_dir / f"{obj:04d}_original.jpg")
            # Seeded per object so the dataset is the same every time it's built
            rng = random.Random(obj)  # noqa: S311 - not used for security
            with Image.open(source) as opened:
                image = opened.convert("RGB")
            for variant, transform in TRANSFORMS.items():
                transform(image, rng).save(images_dir / f"{obj:04d}_{variant}.jpg", quality=95)

    def ground_truth(self, images_dir: Path) -> GroundTruth:
        return GroundTruth(group_pairs(images_dir.iterdir(), derived_group))


class UKBenchExact(Dataset):
    """imagededup's exact-duplicate benchmark, built from UKBench.

    The first photo of each of the 2,550 objects, plus a byte-identical copy of it.
    """

    key = "ukbench-exact"
    name = "UKBench exact copies (imagededup)"
    objects = 2550

    def build(self, images_dir: Path, data_dir: Path) -> None:
        source_dir = UKBench().prepare(data_dir)
        for obj in range(self.objects):
            source = UKBench.image(source_dir, obj)
            for copy in ("a", "b"):
                link_or_copy(source, images_dir / f"{obj:04d}_{copy}.jpg")

    def ground_truth(self, images_dir: Path) -> GroundTruth:
        return GroundTruth(group_pairs(images_dir.iterdir(), derived_group))
