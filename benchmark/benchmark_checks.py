"""Compare image checking libraries, check by check, on benchmark datasets.

Downloads or builds each dataset on first run, then runs each `--check` of each `--checker` on
it, or on its first `--images` images by name, and prints one Markdown table per check: each
library's median time and peak memory, and what it found. Duplicates are also scored for
precision and recall on datasets with labelled duplicates. The libraries' results are compared
in the log.

Each run is a fresh process started by runner.py in the library's own uv environment under
benchmark/envs, and on Linux with systemd in its own cgroup, so peak memory covers every process
it starts. Peak memory is anonymous memory, leaving out page cache, which the kernel can drop.
All libraries get the same thread budget, are warmed up on a few images first so one-off work
like downloading model weights isn't timed, and run in shuffled order. A library that doesn't
implement a check gets a warning, and that check is skipped for it. With --together, the checks
instead run together, in one call per library as its users would run them, and only libraries
that implement every one of them run. Each run's raw results are
saved as JSON under results/raw/<check>/<dataset>, and each check's summary to
results/<check>/<dataset>.csv, one row per library. fastdup only runs on CPython 3.12 on Linux
x86_64 or macOS arm64, and is skipped elsewhere.

    uv run benchmark/benchmark_checks.py --dataset copydays --check duplicates dark \\
        --checker immaculate cleanvision --repeats 3
"""

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

from benchmark_datasets.base_dataset import Dataset
from benchmark_datasets.cifar import CIFAR10, CIFAR100
from benchmark_datasets.copydays import Copydays
from benchmark_datasets.disc import DISC2021
from benchmark_datasets.ukbench import UKBench, UKBenchExact, UKBenchTransformed
from checkers.base_checker import CHECKS, IMAGE_EXTENSIONS
from checkers.registry import CHECKERS
from launcher import Launcher
from loguru import logger
from report import (
    DUPLICATE_FIELDS,
    IMAGE_FIELDS,
    TOGETHER_FIELDS,
    load_results,
    raw_files,
    report_duplicates,
    report_images,
    report_together,
    save_csv,
)

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results"
DEFAULT_TIMEOUT = 3600
WARMUP_IMAGES = 8

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


def image_paths(images_dir: Path) -> list[Path]:
    """The images under `images_dir`, sorted by path."""
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_images(images_dir: Path, paths: list[Path], to: Path) -> None:
    """Copy `paths`, under `images_dir`, to the same paths under `to`."""
    for path in paths:
        copy = to / path.relative_to(images_dir)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)


def warm_file_cache(paths: list[Path]) -> None:
    """Read every image once so no library pays for cold disk reads."""
    for path in paths:
        path.read_bytes()


def implemented_jobs(
    launcher: Launcher, keys: list[str], jobs: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Which of `jobs`, each some checks to run together, each library can run: those whose
    checks it all implements. Warns about the checks a library doesn't implement."""
    implemented = {}
    for key in keys:
        has = set(launcher.checks(key))
        implemented[key] = [job for job, checks in jobs.items() if set(checks) <= has]
        for job, checks in jobs.items():
            for check in checks:
                if check not in has:
                    skipped = "it is skipped" if job == check else f"it doesn't run {job}"
                    logger.warning(f"{key} doesn't implement {check}, so {skipped}")
    return implemented


def warm_up(
    launcher: Launcher,
    implemented: dict[str, list[str]],
    jobs: dict[str, list[str]],
    images_dir: Path,
) -> None:
    """Run each library once on a few images, untimed, so one-off work isn't measured."""
    with tempfile.TemporaryDirectory(prefix="warmup-") as temp:
        warmup_dir = Path(temp) / "images"
        copy_images(images_dir, image_paths(images_dir)[:WARMUP_IMAGES], warmup_dir)
        for key, names in implemented.items():
            if not names:
                continue
            logger.info(f"Warming up {key} ...")
            out = Path(temp) / "result.json"
            record = launcher.launch(key, warmup_dir, out, checks=jobs[names[0]], measured=False)
            if record["status"] != "ok":
                logger.warning(f"Warming up {key} failed: {record['status']}")


def run_all(  # noqa: PLR0913 - the run's settings
    launcher: Launcher,
    implemented: dict[str, list[str]],
    jobs: dict[str, list[str]],
    images_dir: Path,
    *,
    raw_dirs: dict[str, Path],
    repeats: int,
) -> None:
    """Run each library's jobs `repeats` times in shuffled order, saving each run's record
    to `raw_dirs[job]`."""
    runs = [
        (check, key, repeat)
        for key, checks in implemented.items()
        for check in checks
        for repeat in range(1, repeats + 1)
    ]
    random.Random(0).shuffle(runs)  # noqa: S311 - not used for security
    for index, (check, key, repeat) in enumerate(runs, 1):
        logger.info(f"Running {key} {check} ({index}/{len(runs)}) ...")
        out = raw_dirs[check] / f"{key}-{repeat}.json"
        record = launcher.launch(key, images_dir, out, checks=jobs[check])
        if record["status"] != "ok" and not out.exists():
            out.write_text(json.dumps(record))


def report(  # noqa: PLR0913 - the report's settings
    dataset: Dataset,
    images_dir: Path,
    paths: list[Path],
    *,
    jobs: dict[str, list[str]],
    keys: dict[str, list[str]],
    output_dir: Path,
) -> None:
    """Print and save how the libraries compare on each job, from their saved runs of it:
    those of `keys[job]`. Runs
    record images relative to the folder they ran on, which has the same names as the
    dataset's `images_dir`, so they are resolved against it."""
    truth = dataset.ground_truth(images_dir)
    if truth is not None:
        truth = truth.within({path.resolve() for path in paths})
    for check, checks in jobs.items():
        raw_dir = output_dir / "raw" / check / dataset.key
        results = load_results(keys[check], images_dir, raw_dir)
        if not results:  # keep earlier results when every library failed
            continue
        title = f"{dataset.name}: {check} ({len(paths):,} images)"
        if check == "together":
            title = f"{dataset.name}: {', '.join(checks)} together ({len(paths):,} images)"
            rows = report_together(title, checks, results)
            fields = TOGETHER_FIELDS
        elif check == "duplicates":
            rows = report_duplicates(title, results, truth)
            fields = DUPLICATE_FIELDS
        else:
            rows = report_images(title, check, results)
            fields = IMAGE_FIELDS
        save_csv(output_dir / check / f"{dataset.key}.csv", fields, rows)


def benchmark(  # noqa: PLR0913 - the run's settings
    launcher: Launcher,
    dataset: Dataset,
    images_dir: Path,
    paths: list[Path],
    *,
    implemented: dict[str, list[str]],
    jobs: dict[str, list[str]],
    output_dir: Path,
    repeats: int,
) -> None:
    """Run the libraries on `paths`, the dataset's images to check, saving their records."""
    raw_dirs = {}
    for check in {check for checks in implemented.values() for check in checks}:
        raw_dirs[check] = output_dir / "raw" / check / dataset.key
        raw_dirs[check].mkdir(parents=True, exist_ok=True)
        for key, checks in implemented.items():
            if check in checks:
                for stale in raw_files(raw_dirs[check], key):
                    stale.unlink()
    with tempfile.TemporaryDirectory(prefix="images-") as temp:
        if len(paths) < len(image_paths(images_dir)):
            run_dir = Path(temp) / "images"
            copy_images(images_dir, paths, run_dir)
        else:
            run_dir = images_dir
            warm_file_cache(paths)
        run_all(launcher, implemented, jobs, run_dir, raw_dirs=raw_dirs, repeats=repeats)


def main(  # noqa: PLR0913 - one parameter per command-line option
    datasets: list[str],
    checks: list[str],
    keys: list[str],
    data_dir: Path,
    output_dir: Path,
    *,
    images: int | None,
    repeats: int,
    threads: int,
    timeout: float,
    memory_limit: str | None,
    score_only: bool,
    together: bool,
) -> None:
    # Each job is some checks to run together: each check alone, or all of them at once
    jobs = {"together": checks} if together else {check: [check] for check in checks}
    launcher = None if score_only else Launcher(threads, timeout, memory_limit)
    implemented = None if launcher is None else implemented_jobs(launcher, keys, jobs)
    for key in datasets:
        dataset = DATASETS[key]
        logger.info(f"=== {dataset.name} ===")
        images_dir = dataset.prepare(data_dir)
        paths = image_paths(images_dir)[:images]
        if launcher is not None and implemented is not None:
            if key == datasets[0]:
                warm_up(launcher, implemented, jobs, images_dir)
            benchmark(
                launcher,
                dataset,
                images_dir,
                paths,
                implemented=implemented,
                jobs=jobs,
                output_dir=output_dir,
                repeats=repeats,
            )
        # Libraries that can't run a job have no runs of it to report
        job_keys = {
            job: [key for key in keys if implemented is None or job in implemented[key]]
            for job in jobs
        }
        report(dataset, images_dir, paths, jobs=jobs, keys=job_keys, output_dir=output_dir)


def unique(keys: list[str] | None, every: list[str], default: list[str]) -> list[str]:
    """`keys` without repeats, `default` if there are none, or `every` if one is "all"."""
    keys = keys or default
    return list(every) if "all" in keys else list(dict.fromkeys(keys))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset",
        dest="datasets",
        nargs="+",
        action="extend",
        choices=[*DATASETS, "all"],
        help="datasets to benchmark, or all (default: copydays)",
    )
    parser.add_argument(
        "--check",
        dest="checks",
        nargs="+",
        action="extend",
        choices=[*CHECKS, "all"],
        help="checks to run, or all (default: all)",
    )
    parser.add_argument(
        "--checker",
        dest="checkers",
        nargs="+",
        action="extend",
        choices=[*CHECKERS, "all"],
        help="libraries to compare, or all (default: immaculate)",
    )
    parser.add_argument(
        "--images", type=int, help="only check each dataset's first N images, by path"
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="times to run each check (default: 1)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads or worker processes each library may use (default: all CPUs)",
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
        "--together",
        action="store_true",
        help="run the checks together, in one call per library, rather than one at a time",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="don't run anything; report the raw results saved by an earlier run",
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
            "directory to save results to: <check>/<dataset>.csv summaries and "
            "raw/<check>/<dataset>/*.json runs, overwriting earlier results "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    args = parser.parse_args()
    main(
        unique(args.datasets, list(DATASETS), [Copydays.key]),
        unique(args.checks, list(CHECKS), list(CHECKS)),
        unique(args.checkers, list(CHECKERS), ["immaculate"]),
        args.data_dir,
        args.output_dir,
        images=args.images,
        repeats=args.repeats,
        threads=args.threads,
        timeout=args.timeout,
        memory_limit=args.memory_limit,
        score_only=args.score_only,
        together=args.together,
    )
