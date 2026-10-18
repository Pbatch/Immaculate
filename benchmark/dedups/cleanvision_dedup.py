from pathlib import Path

from cleanvision import Imagelab
from dedups.base_dedup import Dedup, Duplicates, to_sets


class CleanVisionDedup(Dedup):
    name = "CleanVision"

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        imagelab = Imagelab(data_path=str(images_dir), verbose=False)
        imagelab.find_issues(
            issue_types={"exact_duplicates": {}, "near_duplicates": {}},
            n_jobs=self.threads,
            verbose=False,
        )
        return Duplicates(
            exact=to_sets(imagelab.info["exact_duplicates"]["sets"]),
            near=to_sets(imagelab.info["near_duplicates"]["sets"]),
        )
