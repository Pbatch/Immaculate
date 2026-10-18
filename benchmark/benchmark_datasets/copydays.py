from pathlib import Path

from benchmark_datasets.base_dataset import Dataset, GroundTruth, extract, fetch, group_pairs


class Copydays(Dataset):
    """INRIA Copydays: 157 holiday photos and 229 heavily edited copies of them (~280 MB).

    The edits include printing and scanning, blurring and painting over. Files are named
    `BBBBQQ.jpg`, where `BBBB` identifies the original photo and `QQ` is `00` for the
    original itself. Copydays also has JPEG and crop sets, but those are only hosted on
    INRIA's server, which no longer responds; this uses Meta's mirror of the other two.
    """

    key = "copydays"
    name = "INRIA Copydays (original + strong)"
    urls = (
        "https://dl.fbaipublicfiles.com/vissl/datasets/copydays_original.tar.gz",
        "https://dl.fbaipublicfiles.com/vissl/datasets/copydays_strong.tar.gz",
    )

    def build(self, images_dir: Path, data_dir: Path) -> None:
        for url in self.urls:
            archive = fetch(url, data_dir / self.key / url.rsplit("/", 1)[-1])
            extract(archive, images_dir)
            archive.unlink()

    def ground_truth(self, images_dir: Path) -> GroundTruth:
        return GroundTruth(group_pairs(images_dir.iterdir(), lambda path: path.stem[:4]))
