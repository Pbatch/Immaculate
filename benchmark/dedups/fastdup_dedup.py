import os
import tempfile
from pathlib import Path

from dedups.base_dedup import Dedup, Duplicates, to_sets


class FastdupDedup(Dedup):
    """Runs fastdup with default settings, using its connected components as duplicate sets.

    fastdup doesn't separate exact and near duplicates, so every set is reported as near.
    Raises ImportError when fastdup isn't installed for this platform.
    """

    name = "fastdup"

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        # fastdup is imported here because it isn't available on every platform, and so that
        # its Sentry telemetry is disabled before it loads
        os.environ["SENTRY_OPT_OUT"] = "1"
        import fastdup  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="fastdup-") as work_dir:
            fd = fastdup.create(input_dir=images_dir, work_dir=work_dir)
            fd.run(overwrite=True, verbose=False, num_threads=self.threads)
            components, _ = fd.connected_components()
        # With no similar images, fastdup returns a table without the usual columns
        if "filename" not in components.columns:
            return Duplicates(exact=None, near=set())
        groups = components.groupby("component_id")["filename"].apply(list)
        return Duplicates(exact=None, near=to_sets(group for group in groups if len(group) > 1))
