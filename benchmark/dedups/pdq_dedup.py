from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from dedups.base_dedup import IMAGE_EXTENSIONS, Dedup, Duplicates, to_sets
from loguru import logger
from PIL import Image

DIHEDRAL_COUNT = 8
"""pdqhash.compute_dihedral returns the original first, then 7 rotated or flipped versions."""
CHUNK_ROWS = 2048
"""Rows of the distance matrix computed at a time, to bound memory."""


def hash_image(path: Path, *, dihedral: bool) -> np.ndarray | None:
    """The image's 256-bit PDQ hashes as rows of 0/1 bytes, one row per orientation."""
    import pdqhash  # noqa: PLC0415 - imported in the worker process

    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
    except OSError:
        return None
    if dihedral:
        hashes, _ = pdqhash.compute_dihedral(pixels)
    else:
        hash_, _ = pdqhash.compute(pixels)
        hashes = [hash_]
    return np.asarray(hashes, dtype=np.uint8)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


class PDQDedup(Dedup):
    """Meta's PDQ hash, linking images whose hashes differ in at most `threshold` of 256 bits.

    Duplicate sets are the connected components of those links, so they can chain. With
    `dihedral`, each image is also compared in all 8 rotations and flips. PDQ doesn't separate
    exact and near duplicates, so every set is reported as near.
    Raises ImportError when pdqhash isn't installed.
    """

    def __init__(self, threads: int, threshold: int = 31, *, dihedral: bool = False) -> None:
        super().__init__(threads)
        self.threshold = threshold
        self.dihedral = dihedral
        self.name = f"PDQ{' dihedral' if dihedral else ''} <={threshold}"

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        import pdqhash  # noqa: F401, PLC0415 - raise ImportError before starting workers

        paths = sorted(
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        with ProcessPoolExecutor(self.threads) as executor:
            hashed = list(
                executor.map(
                    _hash_image_dihedral if self.dihedral else _hash_image_plain,
                    paths,
                    chunksize=64,
                )
            )
        skipped = sum(hashes is None for hashes in hashed)
        if skipped:
            logger.warning(f"PDQ skipped {skipped} undecodable images")
        paths = [path for path, hashes in zip(paths, hashed, strict=True) if hashes is not None]
        hashes = np.stack([hashes for hashes in hashed if hashes is not None]).astype(np.float32)

        sets = UnionFind(len(paths))
        for first, second in self.close_pairs(hashes):
            sets.union(first, second)
        groups: dict[int, list[Path]] = {}
        for index, path in enumerate(paths):
            groups.setdefault(sets.find(index), []).append(path)
        return Duplicates(exact=None, near=to_sets(g for g in groups.values() if len(g) > 1))

    def close_pairs(self, hashes: np.ndarray) -> list[tuple[int, int]]:
        """Index pairs within the threshold. `hashes` is (images, orientations, 256) of 0/1.

        Hamming distance between 0/1 vectors a and b is |a| + |b| - 2 a.b, so a whole block of
        distances is one matrix product. Every orientation of the query is compared against the
        original orientation of every other image.
        """
        originals = hashes[:, 0]
        original_counts = originals.sum(axis=1)
        pairs = []
        for start in range(0, len(hashes), CHUNK_ROWS):
            block = hashes[start : start + CHUNK_ROWS]
            distances = np.full((len(block), len(hashes)), np.inf, dtype=np.float32)
            for orientation in range(block.shape[1]):
                queries = block[:, orientation]
                np.minimum(
                    distances,
                    queries.sum(axis=1)[:, None]
                    + original_counts[None, :]
                    - 2 * queries @ originals.T,
                    out=distances,
                )
            rows, columns = np.nonzero(distances <= self.threshold)
            rows += start
            keep = rows != columns
            pairs.extend(zip(rows[keep].tolist(), columns[keep].tolist(), strict=True))
        return pairs


def _hash_image_plain(path: Path) -> np.ndarray | None:
    return hash_image(path, dihedral=False)


def _hash_image_dihedral(path: Path) -> np.ndarray | None:
    return hash_image(path, dihedral=True)
