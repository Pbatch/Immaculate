import csv
import io
from pathlib import Path

import pyarrow.parquet as pq
from benchmark_datasets.base_dataset import Dataset, DuplicatePairs, GroundTruth, fetch
from PIL import Image

# Lists of duplicates found by Barz & Denzler, "Do We Train on Test Data? Purging CIFAR of
# Near-Duplicates" (2020), pinned to a commit so the labels can't change underneath us
CIFAIR_META_URL = (
    "https://raw.githubusercontent.com/cvjena/cifair/70917827fcd2bbd52997c0f797fd686d23e66a05/meta"
)
BATCH_SIZE = 10_000

# ciFAIR's manual judgments: exact duplicate, near duplicate (same camera shot, different
# post-processing), and "very similar" (different shots that are hard to tell apart)
DUPLICATE_JUDGMENTS = {"0", "1"}
AMBIGUOUS_JUDGMENT = "2"


class CIFAR(Dataset):
    """A CIFAR dataset's 60,000 32x32 images, labelled with ciFAIR's list of duplicates.

    Images are named `train_<index>.png` and `test_<index>.png`, using their index in the
    original dataset, which ciFAIR's labels refer to. ciFAIR only checked each test image
    against its nearest neighbours in the training and test sets, so the labels are
    incomplete: a pair a library finds that isn't listed may still be a duplicate. Pairs
    ciFAIR judged "very similar" but not duplicates are ignored.

    The images come from Hugging Face's Parquet mirror, pinned to a revision, because the
    original host serves them at around 50 KB/s.
    """

    repo: str
    revision: str
    config: str
    train_batch_order: tuple[int, ...] | None = None
    """The original 10,000-image training batch held in each block of the mirror's training
    split, when the mirror stores them in a different order."""
    cifair_name: str

    def original_train_index(self, index: int) -> int:
        if self.train_batch_order is None:
            return index
        block, offset = divmod(index, BATCH_SIZE)
        return self.train_batch_order[block] * BATCH_SIZE + offset

    def build(self, images_dir: Path, data_dir: Path) -> None:
        base_url = f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}"
        for split in ("train", "test"):
            url = f"{base_url}/{self.config}/{split}-00000-of-00001.parquet"
            images = pq.read_table(fetch(url, data_dir / self.key / f"{split}.parquet")).column(0)
            for index, cell in enumerate(images.to_pylist()):
                original = self.original_train_index(index) if split == "train" else index
                with Image.open(io.BytesIO(cell["bytes"])) as image:
                    image.save(images_dir / f"{split}_{original:05d}.png")

    def ground_truth(self, images_dir: Path) -> GroundTruth:
        duplicates: DuplicatePairs = set()
        ignored: DuplicatePairs = set()
        # The first file pairs test images with training images, the second with test images
        for suffix, other_split in (("", "train"), ("_test", "test")):
            name = f"duplicates_{self.cifair_name}{suffix}.csv"
            labels = fetch(f"{CIFAIR_META_URL}/{name}", images_dir.parent / name)
            with labels.open(newline="") as file:
                for row in csv.DictReader(file):
                    pair = frozenset(
                        (
                            (images_dir / f"test_{int(row['TestID']):05d}.png").resolve(),
                            (images_dir / f"{other_split}_{int(row['TrainID']):05d}.png").resolve(),
                        )
                    )
                    if row["Judgment"] in DUPLICATE_JUDGMENTS:
                        duplicates.add(pair)
                    elif row["Judgment"] == AMBIGUOUS_JUDGMENT:
                        ignored.add(pair)
        return GroundTruth(duplicates, ignored, complete=False)


class CIFAR10(CIFAR):
    key = "cifar10"
    name = "CIFAR-10 (ciFAIR labels)"
    repo = "uoft-cs/cifar10"
    revision = "0b2714987fa478483af9968de7c934580d0bb9a2"
    config = "plain_text"
    # The mirror holds data_batch_4, 3, 2, 5 then 1, found by matching ciFAIR's labelled
    # duplicates to their nearest images in the mirror
    train_batch_order = (3, 2, 1, 4, 0)
    cifair_name = "cifar10"


class CIFAR100(CIFAR):
    key = "cifar100"
    name = "CIFAR-100 (ciFAIR labels)"
    repo = "uoft-cs/cifar100"
    revision = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
    config = "cifar100"
    cifair_name = "cifar100"
