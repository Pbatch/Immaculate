"""Run one deduplicator on one folder and save what it found, with timing and memory, as JSON.

benchmark_duplicates.py starts this in the deduplicator's own environment, so it only imports
that one library. On Linux with systemd it also runs in its own cgroup, whose anonymous memory
covers every process the library starts.

    python benchmark/runner.py --dedup immaculate --images DIR --out result.json --threads 16
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

from dedups.base_dedup import DuplicateSets
from dedups.registry import DEDUPS

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


def versions(packages: tuple[str, ...]) -> dict[str, str]:
    found = {}
    for package in packages:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            found[package] = importlib.metadata.version(package)
    return found


def run(key: str, images_dir: Path, threads: int) -> dict[str, Any]:
    spec = DEDUPS[key]
    record: dict[str, Any] = {
        "dedup": key,
        "name": key,
        "threads": threads,
        "python": platform.python_version(),
        "versions": versions(spec.packages),
    }
    start = time.perf_counter()
    try:
        dedup = spec.create(threads)
        record["name"] = dedup.name
        duplicates = dedup.find_duplicates(images_dir)
    except ImportError as error:
        record |= {"status": "unavailable", "error": str(error)}
    except Exception:  # noqa: BLE001 - reported to the benchmark rather than crashing it
        record |= {"status": "error", "error": traceback.format_exc()}
    else:
        record |= {
            "status": "ok",
            "seconds": time.perf_counter() - start,
            "exact": None if duplicates.exact is None else relative(images_dir, duplicates.exact),
            "near": relative(images_dir, duplicates.near),
        }
    return record


def main(dedup: str, images_dir: Path, out: Path, *, threads: int, cgroup: bool) -> None:
    group = cgroup_dir() if cgroup else None
    if group is None:
        record = run(dedup, images_dir.resolve(), threads)
        record |= {"peak_memory_bytes": maxrss_peak(), "memory_method": "maxrss"}
    else:
        # Stopped before writing the JSON, so serializing the results isn't counted
        with AnonPeak(group) as anon:
            record = run(dedup, images_dir.resolve(), threads)
        record |= {
            "peak_memory_bytes": anon.peak,
            "memory_method": "cgroup-anon",
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dedup", required=True, choices=DEDUPS)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--cgroup", action="store_true", help="measure memory with this process's cgroup"
    )
    args = parser.parse_args()
    main(args.dedup, args.images, args.out, threads=args.threads, cgroup=args.cgroup)
