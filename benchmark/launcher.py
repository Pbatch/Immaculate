"""Starting runner.py for one library at a time, each in its own environment and cgroup."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from checkers.registry import CHECKERS
from loguru import logger

BENCHMARK_DIR = Path(__file__).parent
ENVS_DIR = BENCHMARK_DIR / "envs"
RUNNER = BENCHMARK_DIR / "runner.py"

THREAD_VARIABLES = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
"""Variables that set how many threads libraries use, all set to the same budget."""


class Launcher:
    """Starts runner.py for a checker in its environment, with the same limits every time."""

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
        """The interpreter of the checker's environment, syncing it first if needed."""
        env = CHECKERS[key].env
        if env not in self.pythons:
            project = ENVS_DIR / env
            logger.info(f"Syncing the {env} environment ...")
            subprocess.run(["uv", "sync", "--project", str(project), "--quiet"], check=True)
            self.pythons[env] = project / ".venv" / "bin" / "python"
        return self.pythons[env]

    def checks(self, key: str) -> list[str]:
        """The checks the checker `key` implements."""
        command = [str(self.python(key)), str(RUNNER), "--checker", key, "--list-checks"]
        output = subprocess.run(command, env=self.env, capture_output=True, text=True, check=True)
        return json.loads(output.stdout)

    def launch(
        self,
        key: str,
        images_dir: Path,
        out: Path,
        *,
        checks: list[str] | None = None,
        measured: bool = True,
    ) -> dict:
        """Run the checker's `checks`, by default just duplicates, and return its record,
        killing it if it exceeds the timeout. Several checks run together."""
        out.unlink(missing_ok=True)
        checks = checks or ["duplicates"]
        command = [str(self.python(key)), str(RUNNER), "--checker", key, "--check", *checks]
        command += ["--images", str(images_dir), "--out", str(out), "--threads", str(self.threads)]
        if measured and self.cgroup:
            command = [*self.cgroup, *command, "--cgroup"]
        start = time.perf_counter()
        # A new session, so a timeout can kill the library's worker processes too
        # The library's output goes to stderr, keeping stdout for the benchmark's tables
        process = subprocess.Popen(command, env=self.env, start_new_session=True, stdout=sys.stderr)
        try:
            returncode = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"checker": key, "name": key, "status": "timeout"}
        wall_seconds = time.perf_counter() - start
        if returncode != 0 or not out.exists():
            # A negative code is a signal, e.g. SIGKILL from reaching the memory limit
            error = f"runner exited with code {returncode}"
            return {"checker": key, "name": key, "status": "error", "error": error}
        record = json.loads(out.read_text())
        record["wall_seconds"] = wall_seconds
        out.write_text(json.dumps(record))
        return record
