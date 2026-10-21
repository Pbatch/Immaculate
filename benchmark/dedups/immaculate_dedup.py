from pathlib import Path

from dedups.base_dedup import Dedup, Duplicates, to_sets
from loguru import logger

import immaculate


class ImmaculateDedup(Dedup):
    """Immaculate has no thread setting; the runner limits it with RAYON_NUM_THREADS."""

    name = "Immaculate"

    def __init__(
        self, threads: int, method: immaculate.Method = "pdq", sscd_model: str | None = None
    ) -> None:
        """`sscd_model`, a path relative to the repository's root, replaces the SSCD model
        Immaculate ships, such as a student exported by dev/export_mobilenetv4_onnx.py."""
        super().__init__(threads)
        self.method = method
        if method == "phash":
            self.name = "Immaculate pHash"
        elif method == "sscd":
            self.name = "Immaculate SSCD"
        elif method == "sscd_distilled":
            self.name = "Immaculate SSCD distilled"
        if sscd_model is not None:
            model = Path(__file__).parents[2] / sscd_model
            if not model.exists():
                msg = f"{model} doesn't exist; export it with dev/export_mobilenetv4_onnx.py"
                raise FileNotFoundError(msg)
            immaculate._duplicates.SSCD_MODEL = model  # noqa: SLF001 - no public setting
            self.name = f"Immaculate SSCD ({model.stem})"

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        result = immaculate.find_duplicates(images_dir, self.method)
        if result.skipped:
            logger.warning(f"{self.name} skipped {len(result.skipped)} undecodable images")
        return Duplicates(exact=to_sets(result.exact), near=to_sets(result.near))
