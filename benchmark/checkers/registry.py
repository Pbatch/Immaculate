"""Which environment and class runs each checker.

Only uses the standard library, so the benchmark can look checkers up without importing the
libraries they wrap, which are only installed in their own environments.
"""

import importlib
from dataclasses import dataclass, field
from typing import Any

from checkers.base_checker import Checker


@dataclass(frozen=True)
class CheckerSpec:
    env: str
    """Folder under benchmark/envs holding the uv project the checker runs in."""
    module: str
    cls: str
    packages: tuple[str, ...]
    """Distributions whose versions are recorded with each run."""
    kwargs: dict[str, Any] = field(default_factory=dict)

    def checker_class(self) -> type[Checker]:
        """Import the checker's module and return its class; raises ImportError if unavailable."""
        return getattr(importlib.import_module(self.module), self.cls)

    def create(self, threads: int) -> Checker:
        """Import the checker's module and construct it; raises ImportError if unavailable."""
        return self.checker_class()(threads, **self.kwargs)


CHECKERS: dict[str, CheckerSpec] = {
    "cleanvision": CheckerSpec(
        "cleanvision", "checkers.cleanvision_checker", "CleanVisionChecker", ("cleanvision",)
    ),
    # czkawka is a binary rather than a Python package, so its version is in its name
    "czkawka": CheckerSpec("czkawka", "checkers.czkawka_checker", "CzkawkaChecker", ()),
    "czkawka-dihedral": CheckerSpec(
        "czkawka", "checkers.czkawka_checker", "CzkawkaChecker", (), {"dihedral": True}
    ),
    "fastdup": CheckerSpec("fastdup", "checkers.fastdup_checker", "FastdupChecker", ("fastdup",)),
    "imagededup-phash": CheckerSpec(
        "imagededup",
        "checkers.imagededup_checker",
        "ImagededupChecker",
        ("imagededup",),
        {"method": "phash"},
    ),
    "imagededup-cnn": CheckerSpec(
        "imagededup",
        "checkers.imagededup_checker",
        "ImagededupChecker",
        ("imagededup", "torch"),
        {"method": "cnn"},
    ),
    "imagehash": CheckerSpec(
        "imagehash", "checkers.imagehash_checker", "ImagehashChecker", ("imagehash",)
    ),
    "immaculate": CheckerSpec(
        "immaculate", "checkers.immaculate_checker", "ImmaculateChecker", ("immaculate",)
    ),
    "immaculate-phash": CheckerSpec(
        "immaculate",
        "checkers.immaculate_checker",
        "ImmaculateChecker",
        ("immaculate",),
        {"method": "phash"},
    ),
    "immaculate-sscd": CheckerSpec(
        "immaculate",
        "checkers.immaculate_checker",
        "ImmaculateChecker",
        ("immaculate",),
        {"method": "sscd"},
    ),
    "immaculate-sscd-distilled": CheckerSpec(
        "immaculate",
        "checkers.immaculate_checker",
        "ImmaculateChecker",
        ("immaculate",),
        {"method": "sscd_distilled"},
    ),
    "immaculate-sscd-student": CheckerSpec(
        "immaculate",
        "checkers.immaculate_checker",
        "ImmaculateChecker",
        ("immaculate",),
        {"method": "sscd", "sscd_model": "dev/models/conv-s.onnx"},
    ),
    "pdq": CheckerSpec("pdq", "checkers.pdq_checker", "PDQChecker", ("pdqhash",)),
    "pdq-dihedral": CheckerSpec(
        "pdq", "checkers.pdq_checker", "PDQChecker", ("pdqhash",), {"dihedral": True}
    ),
    "sscd-disc-mixup": CheckerSpec(
        "sscd",
        "checkers.sscd_checker",
        "SSCDChecker",
        ("torch",),
        # On the GPU for now, to save time; the other libraries all run on the CPU
        {"model": "sscd_disc_mixup", "device": "cuda"},
    ),
}
