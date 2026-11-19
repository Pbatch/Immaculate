"""Run one check of one library on one folder and save what it found, with timing and memory,
as JSON.

benchmark_duplicates.py and benchmark_checks.py start this in the library's own environment, so
it only imports that one library. On Linux with systemd it also runs in its own cgroup, whose
anonymous memory covers every process the library starts. With --list-checks, it prints the
checks the library implements as JSON instead.

    python benchmark/runner.py --checker immaculate --check dark --images DIR --out result.json \
        --threads 16
"""

import argparse
import contextlib
import importlib.metadata
import json
import platform
import resource
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Self

from checkers.base_checker import CHECKS, DuplicateSets, Images, implements, method
from checkers.registry import CHECKERS

CGROUP_ROOT = Path("/sys/fs/cgroup")
ANON_INTERVAL = 0.01
"""Seconds between samples of the cgroup's anonymous memory."""


def cgroup_dir() -> Path | None:
    """This process's cgroup v2 directory, or `None` if it isn't in one."""
    with contextlib.suppress(OSError):
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return CGROUP_ROOT / line.removeprefix("0::").lstrip("/")
    return None


def cgroup_anon(cgroup: Path) -> int:
    """Anonymous memory of every process in `cgroup` now, in bytes."""
    for line in (cgroup / "memory.stat").read_text().splitlines():
        name, value = line.split()
        if name == "anon":
            return int(value)
    msg = f"no anon in {cgroup / 'memory.stat'}"
    raise ValueError(msg)


class AnonPeak:
    """Samples a cgroup's anonymous memory on a thread, keeping the largest.

    Unlike its memory.peak, this leaves out page cache, which the kernel drops under pressure
    and charges to the cgroup when an image is read again after being evicted, so it grows
    with how long a run takes rather than with what the library holds. A spike shorter than
    `ANON_INTERVAL` can be missed. Peak RSS isn't used as a floor, as it counts mapped library
    files, which are page cache too.
    """

    def __init__(self, cgroup: Path) -> None:
        self.cgroup = cgroup
        self.peak = cgroup_anon(cgroup)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self) -> None:
        while not self.stop.wait(ANON_INTERVAL):
            self.peak = max(self.peak, cgroup_anon(self.cgroup))

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, cgroup_anon(self.cgroup))


def maxrss_peak() -> int:
    """The larger of this process's peak and its largest finished child's, in bytes.

    Worker processes run at the same time aren't added together, so this underestimates
    libraries that use several.
    """
    scale = 1 if sys.platform == "darwin" else 1024  # Linux reports KiB
    usage = max(
        resource.getrusage(who).ru_maxrss
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    )
    return usage * scale


def relative(images_dir: Path, sets: DuplicateSets) -> list[list[str]]:
    return sorted(sorted(str(path.relative_to(images_dir)) for path in group) for group in sets)


def relative_images(images_dir: Path, images: Images) -> list[str]:
    return sorted(str(path.relative_to(images_dir)) for path in images)


def versions(packages: tuple[str, ...]) -> dict[str, str]:
    found = {}
    for package in packages:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            found[package] = importlib.metadata.version(package)
    return found


def found_record(images_dir: Path, check: str, found: Any) -> dict[str, Any]:  # noqa: ANN401
    """What `check` found, with paths relative to `images_dir`, for the record."""
    if check == "duplicates":
        return {
            "exact": None if found.exact is None else relative(images_dir, found.exact),
            "near": relative(images_dir, found.near),
        }
    return {"images": relative_images(images_dir, found)}


def run(key: str, checks: list[str], images_dir: Path, threads: int) -> dict[str, Any]:
    """Run `checks` of the checker `key` on `images_dir`: alone if there's one, or together,
    through the checker's `find_issues`, if there are several."""
    spec = CHECKERS[key]
    record: dict[str, Any] = {
        "checker": key,
        "checks": checks,
        "name": key,
        "threads": threads,
        "python": platform.python_version(),
        "versions": versions(spec.packages),
    }
    start = time.perf_counter()
    try:
        checker = spec.create(threads)
        record["name"] = checker.name
        if not all(implements(type(checker), check) for check in checks):
            return record | {"status": "unsupported"}
        if len(checks) == 1:
            found = {checks[0]: getattr(checker, method(checks[0]))(images_dir)}
        else:
            found = checker.find_issues(images_dir, checks)
    except ImportError as error:
        record |= {"status": "unavailable", "error": str(error)}
    except Exception:  # noqa: BLE001 - reported to the benchmark rather than crashing it
        record |= {"status": "error", "error": traceback.format_exc()}
    else:
        record |= {"status": "ok", "seconds": time.perf_counter() - start}
        if len(checks) == 1:
            record |= found_record(images_dir, checks[0], found[checks[0]])
        else:
            record["found"] = {
                check: found_record(images_dir, check, found[check]) for check in checks
            }
    return record


def list_checks(key: str) -> list[str]:
    """The checks the checker `key` implements, without running any."""
    checker_class = CHECKERS[key].checker_class()
    return [check for check in CHECKS if implements(checker_class, check)]


def main(  # noqa: PLR0913 - one parameter per command-line option
    checker: str, checks: list[str], images_dir: Path, out: Path, *, threads: int, cgroup: bool
) -> None:
    group = cgroup_dir() if cgroup else None
    if group is None:
        record = run(checker, checks, images_dir.resolve(), threads)
        record |= {"peak_memory_bytes": maxrss_peak(), "memory_method": "maxrss"}
    else:
        # Stopped before writing the JSON, so serializing the results isn't counted
        with AnonPeak(group) as anon:
            record = run(checker, checks, images_dir.resolve(), threads)
        record |= {
            "peak_memory_bytes": anon.peak,
            "memory_method": "cgroup-anon",
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checker", required=True, choices=CHECKERS)
    parser.add_argument(
        "--check",
        dest="checks",
        nargs="+",
        choices=CHECKS,
        default=["duplicates"],
        help="the check to run, or several to run together",
    )
    parser.add_argument("--list-checks", action="store_true", help="print the checks it has")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--cgroup", action="store_true", help="measure memory with this process's cgroup"
    )
    args = parser.parse_args()
    if args.list_checks:
        print(json.dumps(list_checks(args.checker)))
    elif args.images is None or args.out is None or args.threads is None:
        parser.error("--images, --out and --threads are required unless --list-checks is given")
    else:
        main(
            args.checker,
            args.checks,
            args.images,
            args.out,
            threads=args.threads,
            cgroup=args.cgroup,
        )
