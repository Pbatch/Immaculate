from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dedups.base_dedup import IMAGE_EXTENSIONS, Dedup, Duplicates, to_sets
from loguru import logger
from PIL import Image


def hash_image(path: Path) -> str | None:
    """The image's 64-bit pHash as hex, or None if it can't be decoded."""
    import imagehash  # noqa: PLC0415 - imported in the worker process

    try:
        with Image.open(path) as image:
            return str(imagehash.phash(image))
    except OSError:
        return None


class ImagehashDedup(Dedup):
    """The imagehash library's pHash, grouping images with identical hashes.

    imagehash only hashes, so images are hashed in `threads` processes and grouped here, as
    Immaculate's "phash" method groups them. Every set is reported as near.
    Raises ImportError when imagehash isn't installed.
    """

    name = "imagehash pHash"

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        import imagehash  # noqa: F401, PLC0415 - raise ImportError before starting workers

        paths = sorted(
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        with ProcessPoolExecutor(self.threads) as executor:
            hashes = list(executor.map(hash_image, paths, chunksize=64))
        skipped = sum(hash_ is None for hash_ in hashes)
        if skipped:
            logger.warning(f"imagehash skipped {skipped} undecodable images")

        groups: dict[str, list[Path]] = {}
        for path, hash_ in zip(paths, hashes, strict=True):
            if hash_ is not None:
                groups.setdefault(hash_, []).append(path)
        return Duplicates(exact=None, near=to_sets(g for g in groups.values() if len(g) > 1))
