"""Find exact and near-duplicate images in a folder.

immaculate photos/
immaculate photos/ --json
immaculate photos/ --method sscd

`python -m immaculate` works the same.
"""

import argparse
import errno
import json
import os
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


class ProgressBar:
    """A progress bar on stderr, redrawn in place on each call."""

    WIDTH = 30

    def __init__(self) -> None:
        self.drawn = False

    def __call__(self, done: int, total: int) -> None:
        filled = self.WIDTH * done // total if total else self.WIDTH
        bar = "#" * filled + "-" * (self.WIDTH - filled)
        sys.stderr.write(f"\r[{bar}] {done}/{total} images")
        sys.stderr.flush()
        self.drawn = True

    def close(self) -> None:
        """End the bar's line, if it was drawn, so later output starts on a new one."""
        if self.drawn:
            sys.stderr.write("\n")


def main(
    input_dir: Path, *, as_json: bool, method: Method = "pdq", show_progress: bool = False
) -> int:
    progress = ProgressBar() if show_progress else None
    try:
        duplicates = find_duplicates(input_dir, method, progress=progress)
    except OSError as error:
        sys.stderr.write(f"immaculate: error: {error}\n")
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if progress:
            progress.close()
    try:
        sys.stdout.write((format_json(duplicates) if as_json else format_text(duplicates)) + "\n")
        sys.stdout.flush()
    except OSError as error:
        if not is_closed_pipe(error):
            raise
        # The reader closed the pipe early, as `head` does. Point stdout at devnull so Python
        # doesn't fail again flushing it at exit, and exit as if killed by SIGPIPE, like
        # other command-line tools.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 141
    return 0


def is_closed_pipe(error: OSError) -> bool:
    """Whether `error` came from writing to a pipe whose reader has closed it, which Windows
    reports as EINVAL rather than EPIPE."""
    return isinstance(error, BrokenPipeError) or (
        sys.platform == "win32" and error.errno == errno.EINVAL
    )


def cli(argv: list[str] | None = None) -> None:
    """The `immaculate` command: parses `argv` (default: the command line) and exits with
    `main`'s exit code."""
    # Python 3.12's argparse names the program "__main__.py" when run with -m
    prog = "python -m immaculate" if Path(sys.argv[0]).name == "__main__.py" else "immaculate"
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Find exact and near-duplicate images in a folder and its subfolders "
            f"({', '.join(_core.IMAGE_EXTENSIONS)}). Exact duplicates have identical "
            "pixels; near duplicates are found by --method. Nothing is modified. Shows a "
            "progress bar on stderr when it is a terminal."
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
            "also matching rotations and flips), phash (identical pHash), sscd (Meta's SSCD "
            f"copy detection model, with embeddings at least {_core.SSCD_THRESHOLD:g} cosine "
            "similar; also catches crops and overlays, but is much slower) or sscd_distilled "
            "(a small model distilled from SSCD; about 4x faster, at some cost in recall) "
            "(default: pdq)"
        ),
    )
    args = parser.parse_args(argv)
    sys.exit(
        main(
            args.input_dir,
            as_json=args.json,
            method=args.method,
            show_progress=sys.stderr.isatty(),
        )
    )


if __name__ == "__main__":
    cli()
