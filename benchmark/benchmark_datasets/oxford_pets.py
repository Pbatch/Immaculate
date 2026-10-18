from pathlib import Path

from benchmark_datasets.base_dataset import Dataset, extract, fetch


class OxfordPets(Dataset):
    """Photos of 37 cat and dog breeds, with no duplicate labels (~800 MB)."""

    key = "oxford-pets"
    name = "Oxford-IIIT Pet"
    url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

    def build(self, images_dir: Path, data_dir: Path) -> None:
        archive = fetch(self.url, data_dir / self.key / "images.tar.gz")
        extract(archive, images_dir, "images")
        archive.unlink()
