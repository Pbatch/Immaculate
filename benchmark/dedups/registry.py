"""Which environment and class runs each deduplicator.

Only uses the standard library, so the benchmark can look deduplicators up without importing
the libraries they wrap, which are only installed in their own environments.
"""

import importlib
from dataclasses import dataclass, field
from typing import Any

from dedups.base_dedup import Dedup


@dataclass(frozen=True)
class DedupSpec:
    env: str
    """Folder under benchmark/envs holding the uv project the deduplicator runs in."""
    module: str
    cls: str
    packages: tuple[str, ...]
    """Distributions whose versions are recorded with each run."""
    kwargs: dict[str, Any] = field(default_factory=dict)

    def create(self, threads: int) -> Dedup:
        """Import the deduplicator's module and construct it; raises ImportError if unavailable."""
        dedup_class = getattr(importlib.import_module(self.module), self.cls)
        return dedup_class(threads, **self.kwargs)


DEDUPS: dict[str, DedupSpec] = {
    "cleanvision": DedupSpec(
        "cleanvision", "dedups.cleanvision_dedup", "CleanVisionDedup", ("cleanvision",)
    ),
    "fastdup": DedupSpec("fastdup", "dedups.fastdup_dedup", "FastdupDedup", ("fastdup",)),
    "imagededup-phash": DedupSpec(
        "imagededup",
        "dedups.imagededup_dedup",
        "ImagededupDedup",
        ("imagededup",),
        {"method": "phash"},
    ),
    "imagededup-cnn": DedupSpec(
        "imagededup",
        "dedups.imagededup_dedup",
        "ImagededupDedup",
        ("imagededup", "torch"),
        {"method": "cnn"},
    ),
    "immaculate": DedupSpec(
        "immaculate", "dedups.immaculate_dedup", "ImmaculateDedup", ("immaculate",)
    ),
    "immaculate-phash": DedupSpec(
        "immaculate",
        "dedups.immaculate_dedup",
        "ImmaculateDedup",
        ("immaculate",),
        {"method": "phash"},
    ),
    "pdq": DedupSpec("pdq", "dedups.pdq_dedup", "PDQDedup", ("pdqhash",)),
    "pdq-dihedral": DedupSpec(
        "pdq", "dedups.pdq_dedup", "PDQDedup", ("pdqhash",), {"dihedral": True}
    ),
    "sscd-disc-mixup": DedupSpec(
        "sscd", "dedups.sscd_dedup", "SSCDDedup", ("torch",), {"model": "sscd_disc_mixup"}
    ),
}
