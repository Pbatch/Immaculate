from pathlib import Path

from dedups.base_dedup import Dedup, Duplicates, to_sets
from loguru import logger

import immaculate


class ImmaculateDedup(Dedup):
    """Immaculate has no thread setting; the runner limits it with RAYON_NUM_THREADS."""

    name = "Immaculate"

    def __init__(self, threads: int, method: immaculate.Method = "pdq") -> None:
        super().__init__(threads)
        self.method = method
        if method == "phash":
            self.name = "Immaculate pHash"
        elif method == "sscd":
            self.name = "Immaculate SSCD"

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        result = immaculate.find_duplicates(images_dir, self.method)
        if result.skipped:
            logger.warning(f"{self.name} skipped {len(result.skipped)} undecodable images")
        return Duplicates(exact=to_sets(result.exact), near=to_sets(result.near))
