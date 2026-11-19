import inspect
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from checkers.base_checker import Checker, Duplicates, Images, to_images, to_sets


class FastdupChecker(Checker):
    """Runs fastdup with default settings.

    Duplicate sets are its connected components; fastdup doesn't separate exact and near
    duplicates, so every set is reported as near. Dark, light and blurry images are those its
    `summary` counts, from its image statistics with that method's default thresholds: a mean
    gray level below 4 or above 253 out of 255, or a blur score below 150. Corrupt images are
    its invalid instances. Every check runs fastdup in full, as it has no way to compute only
    some of its results. Raises ImportError when fastdup isn't installed for this platform.
    """

    name = "fastdup"

    def run[T](self, images_dir: Path, result: Callable[[Any], T]) -> T:
        """Run fastdup on `images_dir`, returning `result` of the fastdup object."""
        # fastdup is imported here because it isn't available on every platform, and so that
        # its Sentry telemetry is disabled before it loads
        os.environ["SENTRY_OPT_OUT"] = "1"
        import fastdup  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="fastdup-") as work_dir:
            fd = fastdup.create(input_dir=images_dir, work_dir=work_dir)
            fd.run(overwrite=True, verbose=False, num_threads=self.threads)
            return result(fd)

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        components, _ = self.run(images_dir, lambda fd: fd.connected_components())
        # With no similar images, fastdup returns a table without the usual columns
        if "filename" not in components.columns:
            return Duplicates(exact=None, near=set())
        groups = components.groupby("component_id")["filename"].apply(list)
        return Duplicates(exact=None, near=to_sets(group for group in groups if len(group) > 1))

    def find_stats(
        self, images_dir: Path, flagged: Callable[[Any, dict[str, float]], Any]
    ) -> Images:
        """The images whose statistics `flagged(stats, thresholds)` picks, where `thresholds`
        are the defaults of fastdup's `summary`, by parameter name."""

        def result(fd: Any) -> Images:  # noqa: ANN401 - fastdup's object, untyped
            parameters = inspect.signature(fd.summary).parameters.items()
            thresholds = {
                name: parameter.default
                for name, parameter in parameters
                if name.endswith("_threshold")
            }
            stats = fd.img_stats()
            if stats.empty:
                return set()
            return to_images(stats.filename[flagged(stats, thresholds)])

        return self.run(images_dir, result)

    def find_dark(self, images_dir: Path) -> Images:
        return self.find_stats(
            images_dir, lambda stats, limits: stats["mean"] < limits["darkness_threshold"]
        )

    def find_light(self, images_dir: Path) -> Images:
        return self.find_stats(
            images_dir, lambda stats, limits: stats["mean"] > limits["brightness_threshold"]
        )

    def find_blurry(self, images_dir: Path) -> Images:
        return self.find_stats(
            images_dir, lambda stats, limits: stats["blur"] < limits["blur_threshold"]
        )

    def find_corrupt(self, images_dir: Path) -> Images:
        return self.run(images_dir, lambda fd: to_images(fd.invalid_instances().filename))
