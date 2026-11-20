"""Find duplicates and other issues in a folder of images.

immaculate photos/
immaculate photos/ --json
immaculate photos/ --issues near_duplicates dark --method sscd
immaculate photos/ --set dark.brightness=0.25 --set odd_aspect_ratio.ratio=0.5

`python -m immaculate` works the same.
"""

import argparse
import dataclasses
import errno
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, get_args

from immaculate import Check, IssueName, Issues, _core, find_issues
from immaculate._checks import CHECKS

ISSUE_NAMES: tuple[IssueName, ...] = get_args(IssueName.__value__)

SET_TITLES = {"exact_duplicates": "Exact duplicates", "near_duplicates": "Near duplicates"}
"""Titles of the issues that find sets of images, by name."""

IMAGE_TITLES: dict[str, str] = {
    "dark": "Dark",
    "light": "Light",
    "blurry": "Blurry",
    "low_information": "Low information",
    "odd_aspect_ratio": "Odd aspect ratio",
    "odd_size": "Odd size",
    "grayscale": "Grayscale",
    "low_contrast": "Low contrast",
    "truncated": "Truncated",
}
"""Titles of the issues that find single images, by name."""


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def format_text(issues: Issues) -> str:
    """One section per check that ran, with a blank line between duplicate sets, and the
    skipped images, if any."""
    sections = []
    for name, found in issues.to_dict().items():
        if name in SET_TITLES:
            lines = [f"{SET_TITLES[name]}: {plural(len(found), 'set')}"]
            for duplicate_set in found:
                lines.append("")
                lines.extend(f"  {path}" for path in duplicate_set)
        elif name != "skipped":
            lines = [f"{IMAGE_TITLES[name]}: {plural(len(found), 'image')}"]
            lines.extend(f"  {path}" for path in found)
        elif found:
            lines = [f"Skipped (could not decode or empty): {len(found)}"]
            lines.extend(f"  {path}" for path in found)
        else:
            continue
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


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
    input_dir: Path,
    *,
    as_json: bool,
    checks: Iterable[Check] | None = None,
    show_progress: bool = False,
) -> int:
    checks = list(checks) if checks is not None else [check() for check in CHECKS.values()]
    progress = ProgressBar() if show_progress else None
    try:
        issues = find_issues(input_dir, checks, progress=progress)
    except OSError as error:
        sys.stderr.write(f"immaculate: error: {error}\n")
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if progress:
            progress.close()
    try:
        output = json.dumps(issues.to_dict(), indent=2) if as_json else format_text(issues)
        sys.stdout.write(output + "\n")
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


def setting_help() -> str:
    """Every `check.field` that --set takes, with its default."""
    return ", ".join(
        f"{name}.{field.name} ({field.default})"
        for name, check in CHECKS.items()
        for field in dataclasses.fields(check)
    )


def build_checks(
    parser: argparse.ArgumentParser,
    names: list[IssueName],
    settings: list[str],
    method: str | None,
) -> list[Check]:
    """The checks called `names`, with `settings`, each `check.field=value`, and the near
    duplicate `method`, if given. Exits through `parser` if any of them are invalid."""
    values: dict[IssueName, dict[str, Any]] = {name: {} for name in names}
    if method is not None:
        settings = [f"near_duplicates.method={method}", *settings]
    for setting in settings:
        target, equals, value = setting.partition("=")
        name, dot, field_name = target.partition(".")
        if not (equals and dot) or name not in CHECKS:
            parser.error(f"--set {setting}: expected CHECK.FIELD=VALUE, with CHECK an issue")
        fields = {field.name: field for field in dataclasses.fields(CHECKS[name])}
        if field_name not in fields:
            parser.error(f"--set {setting}: {name} has no setting {field_name!r}")
        if name not in values:
            parser.error(f"--set {setting}: {name} isn't one of the --issues")
        if field_name in values[name]:
            parser.error(f"--set {setting}: {name}.{field_name} is set more than once")
        convert = type(fields[field_name].default)
        try:
            values[name][field_name] = convert(value)
        except ValueError:
            parser.error(f"--set {setting}: expected {convert.__name__} for {name}.{field_name}")
    checks = []
    for name, fields in values.items():
        try:
            checks.append(CHECKS[name](**fields))
        except ValueError as error:
            parser.error(str(error))
    return checks


def cli(argv: list[str] | None = None) -> None:
    """The `immaculate` command: parses `argv` (default: the command line) and exits with
    `main`'s exit code."""
    # Python 3.12's argparse names the program "__main__.py" when run with -m
    prog = "python -m immaculate" if Path(sys.argv[0]).name == "__main__.py" else "immaculate"
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Find duplicates and other issues in a folder of images and its subfolders "
            f"({', '.join(_core.IMAGE_EXTENSIONS)}). Nothing is modified. Shows a "
            "progress bar on stderr when it is a terminal."
        ),
    )
    parser.add_argument("input_dir", type=Path, help="folder to search")
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            'print {"exact_duplicates": [[path, ...]], "near_duplicates": [[path, ...]], '
            '"dark": [path, ...], ..., "skipped": [path, ...]}, with a key for each issue'
        ),
    )
    parser.add_argument(
        "--issues",
        nargs="+",
        choices=ISSUE_NAMES,
        default=list(ISSUE_NAMES),
        metavar="ISSUE",
        help=f"issues to look for, from {', '.join(ISSUE_NAMES)} (default: all of them)",
    )
    parser.add_argument(
        "--method",
        choices=_core.METHODS,
        help=(
            f"how to find near duplicates: pdq (PDQ hashes within {_core.PDQ_THRESHOLD} bits, "
            "also matching rotations and flips), phash (identical pHash), sscd (Meta's SSCD "
            f"copy detection model, with embeddings at least {_core.SSCD_THRESHOLD:g} cosine "
            "similar; also catches crops and overlays, but is much slower) or sscd_distilled "
            "(a small model distilled from SSCD; about 4x faster, at some cost in recall) "
            "(default: pdq); the same as --set near_duplicates.method=METHOD"
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="CHECK.FIELD=VALUE",
        dest="settings",
        help=f"change a check's setting; can be repeated. Settings and defaults: {setting_help()}",
    )
    args = parser.parse_args(argv)
    names = list(dict.fromkeys(args.issues))
    checks = build_checks(parser, names, args.settings, args.method)
    sys.exit(
        main(
            args.input_dir,
            as_json=args.json,
            checks=checks,
            show_progress=sys.stderr.isatty(),
        )
    )


if __name__ == "__main__":
    cli()
