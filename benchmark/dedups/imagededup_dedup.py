from pathlib import Path
from typing import Literal

from dedups.base_dedup import Dedup, Duplicates


class ImagededupDedup(Dedup):
    """Runs one of imagededup's methods with its default threshold.

    imagededup reports each image's duplicates rather than sets of duplicates, so every
    duplicate pair is returned as a set of two. It doesn't separate exact and near
    duplicates, so every pair is reported as near.
    """

    def __init__(self, threads: int, method: Literal["phash", "cnn"]) -> None:
        super().__init__(threads)
        self.method = method
        self.name = f"imagededup {'PHash' if method == 'phash' else 'CNN'}"

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        # Imported here because imagededup loads PyTorch, which is slow to import
        from imagededup.methods import CNN, PHash  # noqa: PLC0415

        workers = self.threads
        if self.method == "phash":
            found = PHash(verbose=False).find_duplicates(
                image_dir=images_dir,
                recursive=True,
                num_enc_workers=workers,
                num_dist_workers=workers,
            )
        else:
            found = CNN(verbose=False).find_duplicates(
                image_dir=images_dir,
                recursive=True,
                num_enc_workers=workers,
                num_sim_workers=workers,
            )
        pairs = {
            frozenset(((images_dir / image).resolve(), (images_dir / duplicate).resolve()))
            for image, duplicates in found.items()
            for duplicate in duplicates
        }
        return Duplicates(exact=None, near=pairs)
