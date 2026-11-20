"""Immaculate."""

from immaculate._checks import (
    Blurry,
    Check,
    Dark,
    ExactDuplicates,
    Grayscale,
    IssueName,
    Light,
    LowContrast,
    LowInformation,
    Method,
    NearDuplicates,
    OddAspectRatio,
    OddSize,
    Truncated,
)
from immaculate._issues import Issues, find_issues

__all__ = [
    "Blurry",
    "Check",
    "Dark",
    "ExactDuplicates",
    "Grayscale",
    "IssueName",
    "Issues",
    "Light",
    "LowContrast",
    "LowInformation",
    "Method",
    "NearDuplicates",
    "OddAspectRatio",
    "OddSize",
    "Truncated",
    "find_issues",
]
__version__ = "0.4.0"
