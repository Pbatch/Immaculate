"""Find exact and near-duplicate images in a folder.

python -m immaculate photos/
python -m immaculate photos/ --json
python -m immaculate photos/ --method sscd
"""

import argparse
import json
import sys
from pathlib import Path
from typing import get_args

from immaculate import Duplicates, Method, _core, find_duplicates


def format_text(duplicates: Duplicates) -> str:
    """One section per kind of result, with a blank line between duplicate sets."""
    sections = []
    for title, sets in (
        ("Exact duplicates", duplicates.exact),
        ("Near duplicates", duplicates.near),
    ):
        lines = [f"{title}: {len(sets)} set{'' if len(sets) == 1 else 's'}"]
        for duplicate_set in sets:
            lines.append("")
            lines.extend(f"  {path}" for path in duplicate_set)
        sections.append("\n".join(lines))
    if duplicates.skipped:
        lines = [f"Skipped (could not decode or empty): {len(duplicates.skipped)}"]
        lines.extend(f"  {path}" for path in duplicates.skipped)
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def format_json(duplicates: Duplicates) -> str:
    return json.dumps(
        {
            "exact": [[str(path) for path in duplicate_set] for duplicate_set in duplicates.exact],
            "near": [[str(path) for path in duplicate_set] for duplicate_set in duplicates.near],
            "skipped": [str(path) for path in duplicates.skipped],
        },
        indent=2,
    )


def main(input_dir: Path, *, as_json: bool, method: Method = "pdq") -> int:
    try:
        duplicates = find_duplicates(input_dir, method)
    except OSError as error:
        sys.stderr.write(f"immaculate: error: {error}\n")
        return 1
    sys.stdout.write((format_json(duplicates) if as_json else format_text(duplicates)) + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m immaculate",
        description=(
            "Find exact and near-duplicate images in a folder and its subfolders "
            f"({', '.join(_core.IMAGE_EXTENSIONS)}). Exact duplicates have identical "
            "pixels; near duplicates are found by --method. Nothing is modified."
        ),
    )
    parser.add_argument("input_dir", type=Path, help="folder to search")
    parser.add_argument(
        "--json",
        action="store_true",
        help='print {"exact": [[path, ...]], "near": [[path, ...]], "skipped": [path, ...]}',
    )
    parser.add_argument(
        "--method",
        choices=get_args(Method.__value__),
        default="pdq",
        help=(
            f"how to find near duplicates: pdq (PDQ hashes within {_core.PDQ_THRESHOLD} bits, "
            "also matching rotations and flips), phash (identical pHash) or sscd (Meta's SSCD "
            f"copy detection model, with embeddings at least {_core.SSCD_THRESHOLD:g} cosine "
            "similar; also catches crops and overlays, but is much slower) (default: pdq)"
        ),
    )
    args = parser.parse_args()
    sys.exit(main(args.input_dir, as_json=args.json, method=args.method))
