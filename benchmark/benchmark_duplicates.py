"""Compare duplicate image detection libraries on benchmark datasets.

Downloads or builds each dataset on first run, then times each library's duplicate detection
and reports where their results agree, along with each library's peak memory. Datasets with
labelled duplicates are also scored for precision and recall. Only Immaculate runs, on UKBench
transformed, by default; choose others with --dedup and --dataset.

Each library runs in isolation: in its own uv environment under benchmark/envs, in a fresh
process started by runner.py, and on Linux with systemd in its own cgroup, so peak memory covers
every process it starts. All libraries get the same thread budget, are warmed up on a few images
first so one-off work like downloading model weights isn't timed, and run in shuffled order.
Each run's raw results are saved as JSON, and each dataset's summary to a CSV file, one row per
library. fastdup only runs on CPython 3.12 on Linux x86_64 or macOS arm64, and is skipped
elsewhere.

    uv run benchmark/benchmark_duplicates.py --dataset copydays --repeats 3 --dedup all
"""

import argparse
import csv
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple

from benchmark_datasets.base_dataset import Dataset, DuplicatePairs, GroundTruth
from benchmark_datasets.cifar import CIFAR10, CIFAR100
from benchmark_datasets.copydays import Copydays
from benchmark_datasets.disc import DISC2021
from benchmark_datasets.ukbench import UKBench, UKBenchExact, UKBenchTransformed
from dedups.base_dedup import Duplicates, DuplicateSets
from dedups.registry import DEDUPS
from loguru import logger

BENCHMARK_DIR = Path(__file__).parent
ENVS_DIR = BENCHMARK_DIR / "envs"
RUNNER = BENCHMARK_DIR / "runner.py"
DEFAULT_DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results"
DEFAULT_TIMEOUT = 3600
WARMUP_IMAGES = 8

THREAD_VARIABLES = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

CSV_FIELDS = [
    "dedup",
    "runs",
    "seconds",
    "seconds_min",
    "seconds_max",
    "peak_memory_mib",
    "memory_method",
    "exact_sets",
    "near_sets",
    "pairs",
    "labelled_pairs",
    "correct_pairs",
    "precision",
    "recall",
]

DATASETS: dict[str, Dataset] = {
    dataset.key: dataset
    for dataset in (
        UKBench(),
        UKBenchTransformed(),
        UKBenchExact(),
        Copydays(),
        CIFAR10(),
        CIFAR100(),
        DISC2021(),
    )
}


def warm_file_cache(images_dir: Path) -> int:
    """Read every file once so no library pays for cold disk reads."""
    count = 0
    for path in images_dir.rglob("*"):
        if path.is_file():
            path.read_bytes()
            count += 1
    return count


def to_pairs(duplicates: Duplicates) -> DuplicatePairs:
    """Every pair of images that share a duplicate set, exact or near."""
    return {
        frozenset(pair)
        for group_sets in (duplicates.exact or set(), duplicates.near)
        for group in group_sets
        for pair in combinations(group, 2)
    }


class Launcher:
    """Starts runner.py for a deduplicator in its environment, with the same limits every time."""

    def __init__(self, threads: int, timeout: float, memory_limit: str | None) -> None:
        self.threads = threads
        self.timeout = timeout
        self.pythons: dict[str, Path] = {}
        self.env = os.environ | {name: str(threads) for name in THREAD_VARIABLES}
        # fastdup reports to Sentry unless this is set
        self.env["SENTRY_OPT_OUT"] = "1"
        self.cgroup = self.cgroup_prefix(memory_limit)
        if not self.cgroup:
            logger.warning(
                "systemd-run isn't available, so peak memory only covers each library's largest "
                "single process"
            )

    @staticmethod
    def cgroup_prefix(memory_limit: str | None) -> list[str]:
        """A command prefix that runs a command in a new cgroup, or [] when that's unavailable."""
        if sys.platform != "linux" or shutil.which("systemd-run") is None:
            return []
        prefix = ["systemd-run", "--user", "--scope", "--quiet", "--collect"]
        prefix += ["-p", "MemoryAccounting=yes"]
        if memory_limit is not None:
            prefix += ["-p", f"MemoryMax={memory_limit}", "-p", "MemorySwapMax=0"]
        probe = subprocess.run([*prefix, "true"], capture_output=True, check=False)
        return prefix if probe.returncode == 0 else []

    def python(self, key: str) -> Path:
        """The interpreter of the deduplicator's environment, syncing it first if needed."""
        env = DEDUPS[key].env
        if env not in self.pythons:
            project = ENVS_DIR / env
            logger.info(f"Syncing the {env} environment ...")
            subprocess.run(["uv", "sync", "--project", str(project), "--quiet"], check=True)
            self.pythons[env] = project / ".venv" / "bin" / "python"
        return self.pythons[env]

    def launch(self, key: str, images_dir: Path, out: Path, *, measured: bool = True) -> dict:
        """Run the deduplicator and return its record, killing it if it exceeds the timeout."""
        out.unlink(missing_ok=True)
        command = [str(self.python(key)), str(RUNNER), "--dedup", key]
        command += ["--images", str(images_dir), "--out", str(out), "--threads", str(self.threads)]
        if measured and self.cgroup:
            command = [*self.cgroup, *command, "--cgroup"]
        start = time.perf_counter()
        # A new session, so a timeout can kill the library's worker processes too
        process = subprocess.Popen(command, env=self.env, start_new_session=True)
        try:
            returncode = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"dedup": key, "name": key, "status": "timeout"}
        wall_seconds = time.perf_counter() - start
        if returncode != 0 or not out.exists():
            # A negative code is a signal, e.g. SIGKILL from reaching the memory limit
            error = f"runner exited with code {returncode}"
            return {"dedup": key, "name": key, "status": "error", "error": error}
        record = json.loads(out.read_text())
        record["wall_seconds"] = wall_seconds
        out.write_text(json.dumps(record))
        return record


class Result(NamedTuple):
    name: str
    duplicates: Duplicates
    seconds: list[float]
    peak_bytes: list[int]
    memory_method: str


def from_records(images_dir: Path, records: list[dict]) -> Result:
    """Combine one deduplicator's successful runs. Duplicates come from the first run."""

    def to_sets(groups: list[list[str]]) -> DuplicateSets:
        return {frozenset((images_dir / path).resolve() for path in group) for group in groups}

    first = records[0]
    duplicates = Duplicates(
        exact=None if first["exact"] is None else to_sets(first["exact"]),
        near=to_sets(first["near"]),
    )
    if any((r["exact"], r["near"]) != (first["exact"], first["near"]) for r in records[1:]):
        logger.warning(f"{first['name']} found different duplicates in different runs")
    return Result(
        first["name"],
        duplicates,
        [record["seconds"] for record in records],
        [record["peak_memory_bytes"] for record in records],
        first["memory_method"],
    )


def warm_up(launcher: Launcher, keys: list[str], images_dir: Path) -> None:
    """Run each deduplicator once on a few images, untimed, so one-off work isn't measured."""
    sample = sorted(path for path in images_dir.iterdir() if path.is_file())[:WARMUP_IMAGES]
    with tempfile.TemporaryDirectory(prefix="warmup-") as warmup_dir:
        for path in sample:
            shutil.copyfile(path, Path(warmup_dir) / path.name)
        for key in keys:
            logger.info(f"Warming up {key} ...")
            record = launcher.launch(
                key, Path(warmup_dir), Path(warmup_dir) / "result.json", measured=False
            )
            if record["status"] != "ok":
                logger.warning(f"Warming up {key} failed: {record['status']}")


def run_all(
    launcher: Launcher, keys: list[str], images_dir: Path, raw_dir: Path, repeats: int
) -> None:
    """Run every deduplicator `repeats` times in shuffled order, saving each run's record."""
    runs = [(key, repeat) for key in keys for repeat in range(1, repeats + 1)]
    random.Random(0).shuffle(runs)  # noqa: S311 - not used for security
    for index, (key, repeat) in enumerate(runs, 1):
        logger.info(f"Running {key} ({index}/{len(runs)}) ...")
        out = raw_dir / f"{key}-{repeat}.json"
        record = launcher.launch(key, images_dir, out)
        if record["status"] != "ok" and not out.exists():
            out.write_text(json.dumps(record))


def raw_files(raw_dir: Path, key: str) -> list[Path]:
    """Saved runs of `key`, named `<key>-<repeat>.json`. Matching exactly matters because keys
    can be prefixes of each other, like pdq and pdq-dihedral."""
    return sorted(path for path in raw_dir.glob("*.json") if path.stem.rsplit("-", 1)[0] == key)


def load_results(keys: list[str], images_dir: Path, raw_dir: Path) -> dict[str, Result]:
    results: dict[str, Result] = {}
    for key in keys:
        records = [json.loads(path.read_text()) for path in raw_files(raw_dir, key)]
        if not records:
            logger.warning(f"No saved runs for {key}")
            continue
        ok = [record for record in records if record["status"] == "ok"]
        for record in records:
            if record["status"] == "unavailable":
                logger.warning(f"{key} is not installed for this platform, skipping it")
                break
            if record["status"] != "ok":
                logger.error(f"{key} {record['status']}: {record.get('error', '')}")
        if ok:
            results[key] = from_records(images_dir, ok)
    return results


def image_count(sets: DuplicateSets) -> int:
    return sum(len(group) for group in sets)


def summarize(results: dict[str, Result]) -> None:
    """Log each deduplicator's median time, relative to Immaculate when it ran, its median peak
    memory, and what it found."""
    logger.info("Results (medians)")
    width = max(len(result.name) for result in results.values()) + 1
    baseline = results.get("immaculate")
    for key, result in results.items():
        seconds = statistics.median(result.seconds)
        timing = f"{seconds:.2f} s"
        if len(result.seconds) > 1:
            timing += f" ({min(result.seconds):.2f}-{max(result.seconds):.2f})"
        if baseline is not None and key != "immaculate":
            timing += f", {seconds / statistics.median(baseline.seconds):.1f}x Immaculate"
        memory = f"{statistics.median(result.peak_bytes) / 2**20:.0f} MiB peak"
        if result.memory_method != "cgroup":
            memory += " (largest process only)"
        duplicates = result.duplicates
        if duplicates.exact is None:
            found = f"{len(duplicates.near)} sets (exact and near combined)"
        else:
            found = f"{len(duplicates.exact)} exact sets, {len(duplicates.near)} near sets"
        logger.info(f"  {result.name + ':':<{width}} {timing}, {memory}, {found}")


def compare_sets(kind: str, a: str, a_sets: DuplicateSets, b: str, b_sets: DuplicateSets) -> None:
    logger.info(f"{kind} duplicates: {a} vs {b}")
    width = max(len(a), len(b)) + 1
    for name, sets in [(a, a_sets), (b, b_sets)]:
        logger.info(f"  {name + ':':<{width}} {len(sets)} sets, {image_count(sets)} images")
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


def compare_ground_truth(truth: GroundTruth, scores: dict[str, Score]) -> None:
    """Log each deduplicator's score against the labelled duplicates."""
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


def csv_rows(
    results: dict[str, Result],
    truth: GroundTruth | None,
    scores: dict[str, Score],
) -> list[dict[str, Any]]:
    """One row per deduplicator; columns that don't apply are left empty."""
    rows = []
    for result in results.values():
        duplicates = result.duplicates
        row: dict[str, Any] = {
            "dedup": result.name,
            "runs": len(result.seconds),
            "seconds": f"{statistics.median(result.seconds):.3f}",
            "seconds_min": f"{min(result.seconds):.3f}",
            "seconds_max": f"{max(result.seconds):.3f}",
            "peak_memory_mib": f"{statistics.median(result.peak_bytes) / 2**20:.1f}",
            "memory_method": result.memory_method,
            "exact_sets": "" if duplicates.exact is None else len(duplicates.exact),
            "near_sets": len(duplicates.near),
            "pairs": len(to_pairs(duplicates)),
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


def report(dataset: Dataset, results: dict[str, Result], images_dir: Path) -> list[dict[str, Any]]:
    """Log how the deduplicators compare on `dataset`, returning its CSV rows."""
    summarize(results)
    named = {result.name: result.duplicates for result in results.values()}
    for (a, a_result), (b, b_result) in combinations(named.items(), 2):
        if a_result.exact is not None and b_result.exact is not None:
            compare_sets("Exact", a, a_result.exact, b, b_result.exact)
            compare_sets("Near", a, a_result.near, b, b_result.near)
        else:
            compare_pairs(a, to_pairs(a_result), b, to_pairs(b_result))

    truth = dataset.ground_truth(images_dir)
    scores: dict[str, Score] = {}
    if truth is not None:
        scores = {name: score(truth, to_pairs(duplicates)) for name, duplicates in named.items()}
        compare_ground_truth(truth, scores)
    return csv_rows(results, truth, scores)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved results to {path}")


def expand_all(keys: list[str], every: dict[str, Any]) -> list[str]:
    """`keys`, or every key in `every` if one of them is "all"."""
    return list(every) if "all" in keys else keys


def main(  # noqa: PLR0913 - one parameter per command-line option
    datasets: list[str],
    dedup_keys: list[str],
    data_dir: Path,
    output_dir: Path,
    *,
    repeats: int,
    threads: int,
    timeout: float,
    memory_limit: str | None,
    score_only: bool,
) -> None:
    keys = list(dict.fromkeys(dedup_keys))
    launcher = None if score_only else Launcher(threads, timeout, memory_limit)
    warmed_up = False
    for key in datasets:
        dataset = DATASETS[key]
        logger.info(f"=== {dataset.name} ===")
        images_dir = dataset.prepare(data_dir)
        raw_dir = output_dir / "raw" / key
        if launcher is not None:
            if not warmed_up:
                warm_up(launcher, keys, images_dir)
                warmed_up = True
            logger.info(f"Warming file cache: {warm_file_cache(images_dir)} files in {images_dir}")
            for stale in raw_dir.glob("*.json"):
                if stale.stem.rsplit("-", 1)[0] in keys:
                    stale.unlink()
            raw_dir.mkdir(parents=True, exist_ok=True)
            run_all(launcher, keys, images_dir, raw_dir, repeats)
        results = load_results(keys, images_dir, raw_dir)
        if results:  # keep earlier results when every library failed
            save_csv(output_dir / f"{key}.csv", report(dataset, results, images_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset",
        dest="datasets",
        nargs="+",
        action="extend",
        choices=[*DATASETS, "all"],
        help="datasets to benchmark, or all (default: ukbench-transformed)",
    )
    parser.add_argument(
        "--dedup",
        dest="dedups",
        nargs="+",
        action="extend",
        choices=[*DEDUPS, "all"],
        help="deduplicators to run, or all (default: immaculate)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="times to run each deduplicator (default: 1)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads or worker processes each deduplicator may use (default: all CPUs)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds before a run is killed (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--memory-limit",
        help="kill runs that use more memory than this, e.g. 16G (Linux with systemd only)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="don't run anything; re-score the raw results saved by an earlier run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"where to download datasets (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "directory to save results to: <dataset>.csv summaries and raw/<dataset>/*.json "
            f"runs, overwriting earlier results (default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    args = parser.parse_args()
    main(
        expand_all(args.datasets or [UKBenchTransformed.key], DATASETS),
        expand_all(args.dedups or ["immaculate"], DEDUPS),
        args.data_dir,
        args.output_dir,
        repeats=args.repeats,
        threads=args.threads,
        timeout=args.timeout,
        memory_limit=args.memory_limit,
        score_only=args.score_only,
    )
