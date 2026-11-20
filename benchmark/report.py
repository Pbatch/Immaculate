"""Combining each library's saved runs of a check, comparing the libraries, and reporting them
as Markdown tables and CSV rows."""

import csv
import json
import statistics
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple

from benchmark_datasets.base_dataset import DuplicatePairs, GroundTruth
from checkers.base_checker import Duplicates, DuplicateSets, Images
from loguru import logger

TIMING_FIELDS = [
    "checker",
    "runs",
    "seconds",
    "seconds_min",
    "seconds_max",
    "peak_memory_mib",
    "memory_method",
]
DUPLICATE_FIELDS = [
    *TIMING_FIELDS,
    "exact_sets",
    "near_sets",
    "pairs",
    "labelled_pairs",
    "correct_pairs",
    "precision",
    "recall",
]
"""The CSV columns of the duplicates check."""
IMAGE_FIELDS = [*TIMING_FIELDS, "images"]
TOGETHER_FIELDS = [*TIMING_FIELDS, "checks", "images_flagged"]
"""The CSV columns of several checks run together."""
"""The CSV columns of every other check, which finds single images."""


class Result(NamedTuple):
    """One library's successful runs of one check."""

    name: str
    seconds: list[float]
    peak_bytes: list[int]
    memory_method: str
    duplicates: Duplicates | None
    """What the duplicates check found, from the first run."""
    images: Images | None
    """What any other check found, from the first run."""
    flagged: Images | None = None
    """Every image that any of several checks run together found, from the first run."""


def from_records(images_dir: Path, records: list[dict]) -> Result:
    """Combine one library's successful runs, whose paths are relative to `images_dir`.
    What was found comes from the first run."""

    def resolve(paths: list[str]) -> frozenset[Path]:
        return frozenset((images_dir / path).resolve() for path in paths)

    def to_sets(groups: list[list[str]]) -> DuplicateSets:
        return {resolve(group) for group in groups}

    first = records[0]
    found_keys = ("exact", "near", "images", "found")
    if any(
        [record.get(key) for key in found_keys] != [first.get(key) for key in found_keys]
        for record in records[1:]
    ):
        logger.warning(f"{first['name']} found different results in different runs")
    duplicates = images = flagged = None
    if "found" in first:
        flagged = set()
        for found in first["found"].values():
            if "images" in found:
                groups = [found["images"]]
            else:
                groups = [*(found["exact"] or []), *found["near"]]
            for group in groups:
                flagged |= resolve(group)
    elif "images" in first:
        images = set(resolve(first["images"]))
    else:
        duplicates = Duplicates(
            exact=None if first["exact"] is None else to_sets(first["exact"]),
            near=to_sets(first["near"]),
        )
    return Result(
        first["name"],
        [record["seconds"] for record in records],
        [record["peak_memory_bytes"] for record in records],
        first["memory_method"],
        duplicates,
        images,
        flagged,
    )


def load_results(keys: list[str], images_dir: Path, raw_dir: Path) -> dict[str, Result]:
    """Each library's saved runs in `raw_dir`, combined, skipping libraries with none that
    succeeded."""
    results: dict[str, Result] = {}
    for key in keys:
        records = [json.loads(path.read_text()) for path in raw_files(raw_dir, key)]
        if not records:
            logger.warning(f"No saved runs for {key} in {raw_dir}")
            continue
        for record in records:
            if record["status"] == "unavailable":
                logger.warning(f"{key} is not installed for this platform, skipping it")
                break
            if record["status"] != "ok":
                logger.error(f"{key} {record['status']}: {record.get('error', '')}")
        ok = [record for record in records if record["status"] == "ok"]
        if ok:
            results[key] = from_records(images_dir, ok)
    return results


def raw_files(raw_dir: Path, key: str) -> list[Path]:
    """Saved runs of `key`, named `<key>-<repeat>.json`. Matching exactly matters because keys
    can be prefixes of each other, like pdq and pdq-dihedral."""
    return sorted(path for path in raw_dir.glob("*.json") if path.stem.rsplit("-", 1)[0] == key)


def to_pairs(duplicates: Duplicates) -> DuplicatePairs:
    """Every pair of images that share a duplicate set, exact or near."""
    return {
        frozenset(pair)
        for group_sets in (duplicates.exact or set(), duplicates.near)
        for group in group_sets
        for pair in combinations(group, 2)
    }


class Score(NamedTuple):
    counted: int
    """Pairs found, excluding those the ground truth ignores."""
    correct: int
    precision: float | None
    """None when the ground truth is incomplete or nothing was found."""
    recall: float


def score(truth: GroundTruth, found: DuplicatePairs) -> Score:
    counted = {pair for pair in found if truth.counts(pair)}
    correct = len(counted & truth.duplicates)
    recall = correct / len(truth.duplicates) if truth.duplicates else 0.0
    precision = correct / len(counted) if truth.complete and counted else None
    return Score(len(counted), correct, precision, recall)


def compare_sets(kind: str, a: str, a_sets: DuplicateSets, b: str, b_sets: DuplicateSets) -> None:
    logger.info(f"{kind} duplicates: {a} vs {b}")
    width = max(len(a), len(b)) + 1
    for name, sets in [(a, a_sets), (b, b_sets)]:
        images = sum(len(group) for group in sets)
        logger.info(f"  {name + ':':<{width}} {len(sets)} sets, {images} images")
    logger.info(f"  Identical sets: {len(a_sets & b_sets)}")
    for name, sets in [(a, a_sets - b_sets), (b, b_sets - a_sets)]:
        if sets:
            logger.info(f"  Only in {name} ({len(sets)}):")
            for group in sorted(sets, key=sorted):
                logger.info(f"    {', '.join(sorted(path.name for path in group))}")


def compare_pairs(a: str, a_pairs: DuplicatePairs, b: str, b_pairs: DuplicatePairs) -> None:
    """Compare duplicate pairs, for libraries that don't separate exact and near duplicates."""
    logger.info(f"Duplicate pairs (exact and near combined): {a} vs {b}")
    logger.info(
        f"  {a}: {len(a_pairs)} pairs, {b}: {len(b_pairs)} pairs, "
        f"{len(a_pairs & b_pairs)} found by both"
    )


def compare_images(check: str, a: str, a_images: Images, b: str, b_images: Images) -> None:
    logger.info(f"{check}: {a} vs {b}")
    logger.info(
        f"  {a}: {len(a_images)} images, {b}: {len(b_images)} images, "
        f"{len(a_images & b_images)} found by both"
    )
    for name, images in [(a, a_images - b_images), (b, b_images - a_images)]:
        if images:
            names = ", ".join(sorted(path.name for path in images))
            logger.info(f"  Only in {name} ({len(images)}): {names}")


def compare_ground_truth(truth: GroundTruth, scores: dict[str, Score]) -> None:
    """Log each library's score against the labelled duplicates."""
    logger.info(f"Against ground truth: {len(truth.duplicates)} labelled duplicate pairs")
    width = max(len(name) for name in scores) + 1
    for name, (counted, correct, precision, recall) in scores.items():
        if truth.complete:
            shown = "n/a" if precision is None else f"{precision:.1%}"
            logger.info(
                f"  {name + ':':<{width}} {counted} pairs, {correct} correct, "
                f"precision {shown}, recall {recall:.1%}"
            )
        else:
            logger.info(
                f"  {name + ':':<{width}} {correct} labelled duplicates found "
                f"(recall {recall:.1%}), {counted - correct} unlabelled pairs"
            )


def timing_row(result: Result) -> dict[str, Any]:
    return {
        "checker": result.name,
        "runs": len(result.seconds),
        "seconds": f"{statistics.median(result.seconds):.3f}",
        "seconds_min": f"{min(result.seconds):.3f}",
        "seconds_max": f"{max(result.seconds):.3f}",
        "peak_memory_mib": f"{statistics.median(result.peak_bytes) / 2**20:.1f}",
        "memory_method": result.memory_method,
    }


def percent(fraction: float) -> str:
    return "100%" if fraction == 1 else f"{fraction:.1%}"


def print_table(
    title: str,
    results: list[Result],
    columns: dict[str, Callable[[Result], str]],
    best: dict[str, Callable[[Result], float]],
) -> None:
    """Print `results` as a Markdown table, fastest first, with a column for the time, the peak
    memory and each of `columns`. The least time and memory, and the highest of each of `best`,
    are bold unless every library ties."""
    lowest: dict[str, Callable[[Result], float]] = {
        "Time": lambda result: -statistics.median(result.seconds),
        "Peak memory": lambda result: -statistics.median(result.peak_bytes),
    }
    columns = {
        "Time": lambda result: f"{statistics.median(result.seconds):.2f} s",
        "Peak memory": lambda result: f"{statistics.median(result.peak_bytes) / 2**20:,.0f} MiB",
        **columns,
    }
    best = lowest | best
    results = sorted(results, key=lambda result: statistics.median(result.seconds))
    top = {name: max(map(key, results)) for name, key in best.items()}
    # Only bolded when it stands out, not when every library ties
    stands_out = {name: len(set(map(key, results))) > 1 for name, key in best.items()}
    print(f"\n### {title}\n")
    print(f"| | {' | '.join(columns)} |")
    print(f"|---|{'---:|' * len(columns)}")
    for result in results:
        cells = []
        for name, cell in columns.items():
            shown = cell(result)
            if name in best and stands_out[name] and best[name](result) == top[name]:
                shown = f"**{shown}**"
            cells.append(shown)
        print(f"| {result.name} | {' | '.join(cells)} |")


def report_duplicates(
    title: str, results: dict[str, Result], truth: GroundTruth | None
) -> list[dict[str, Any]]:
    """Compare the libraries' duplicates, score them against `truth` if there is one, and
    print them as a table, returning their CSV rows."""
    named = {result.name: result for result in results.values()}
    for (a, a_result), (b, b_result) in combinations(named.items(), 2):
        a_found = a_result.duplicates or Duplicates(None, set())
        b_found = b_result.duplicates or Duplicates(None, set())
        if a_found.exact is not None and b_found.exact is not None:
            compare_sets("Exact", a, a_found.exact, b, b_found.exact)
            compare_sets("Near", a, a_found.near, b, b_found.near)
        else:
            compare_pairs(a, to_pairs(a_found), b, to_pairs(b_found))

    def pairs(result: Result) -> DuplicatePairs:
        return to_pairs(result.duplicates or Duplicates(None, set()))

    scores: dict[str, Score] = {}
    columns: dict[str, Callable[[Result], str]] = {
        "Pairs found": lambda result: f"{len(pairs(result)):,}"
    }
    best: dict[str, Callable[[Result], float]] = {}
    if truth is not None:
        scores = {name: score(truth, pairs(result)) for name, result in named.items()}
        compare_ground_truth(truth, scores)
        columns["Precision"] = lambda result: (
            "n/a" if (p := scores[result.name].precision) is None else percent(p)
        )
        columns["Recall"] = lambda result: percent(scores[result.name].recall)
        best["Recall"] = lambda result: scores[result.name].recall
    print_table(title, list(named.values()), columns, best)

    rows = []
    for result in named.values():
        duplicates = result.duplicates or Duplicates(None, set())
        row = timing_row(result) | {
            "exact_sets": "" if duplicates.exact is None else len(duplicates.exact),
            "near_sets": len(duplicates.near),
            "pairs": len(pairs(result)),
        }
        if truth is not None:
            precision = scores[result.name].precision
            row |= {
                "labelled_pairs": len(truth.duplicates),
                "correct_pairs": scores[result.name].correct,
                "precision": "" if precision is None else f"{precision:.4f}",
                "recall": f"{scores[result.name].recall:.4f}",
            }
        rows.append(row)
    return rows


def report_images(title: str, check: str, results: dict[str, Result]) -> list[dict[str, Any]]:
    """Compare the images the libraries found, and print them as a table, returning their CSV
    rows."""
    named = {result.name: result.images or set() for result in results.values()}
    for (a, a_images), (b, b_images) in combinations(named.items(), 2):
        compare_images(check, a, a_images, b, b_images)
    columns: dict[str, Callable[[Result], str]] = {
        "Images found": lambda result: f"{len(result.images or ()):,}"
    }
    print_table(title, list(results.values()), columns, {})
    return [
        timing_row(result) | {"images": len(result.images or ())} for result in results.values()
    ]


def report_together(
    title: str, checks: list[str], results: dict[str, Result]
) -> list[dict[str, Any]]:
    """Print several checks run together as a table, returning their CSV rows."""
    columns: dict[str, Callable[[Result], str]] = {
        "Images flagged": lambda result: f"{len(result.flagged or ()):,}"
    }
    print_table(title, list(results.values()), columns, {})
    return [
        timing_row(result)
        | {"checks": " ".join(checks), "images_flagged": len(result.flagged or ())}
        for result in results.values()
    ]


def save_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved results to {path}")
