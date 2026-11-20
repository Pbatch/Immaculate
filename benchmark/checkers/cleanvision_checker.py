from pathlib import Path

from checkers.base_checker import Checker, Duplicates, Found, Images, to_images, to_sets
from cleanvision import Imagelab


class CleanVisionChecker(Checker):
    """CleanVision has every check but low contrast and truncated, under the same names."""

    name = "CleanVision"

    def run(self, images_dir: Path, issues: list[str]) -> Imagelab:
        """An `Imagelab` that has run CleanVision's checks `issues` on `images_dir`."""
        imagelab = Imagelab(data_path=str(images_dir), verbose=False)
        imagelab.find_issues(
            issue_types={issue: {} for issue in issues}, n_jobs=self.threads, verbose=False
        )
        return imagelab

    def find_images(self, images_dir: Path, issue: str) -> Images:
        """The images that CleanVision's check `issue`, run alone, finds in `images_dir`."""
        issues = self.run(images_dir, [issue]).issues
        return to_images(issues.index[issues[f"is_{issue}_issue"]])

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        imagelab = self.run(images_dir, ["exact_duplicates", "near_duplicates"])
        return Duplicates(
            exact=to_sets(imagelab.info["exact_duplicates"]["sets"]),
            near=to_sets(imagelab.info["near_duplicates"]["sets"]),
        )

    def find_dark(self, images_dir: Path) -> Images:
        return self.find_images(images_dir, "dark")

    def find_light(self, images_dir: Path) -> Images:
        return self.find_images(images_dir, "light")

    def find_blurry(self, images_dir: Path) -> Images:
        return self.find_images(images_dir, "blurry")

    def find_low_information(self, images_dir: Path) -> Images:
        return self.find_images(images_dir, "low_information")

    def find_odd_aspect_ratio(self, images_dir: Path) -> Images:
        return self.find_images(images_dir, "odd_aspect_ratio")

    def find_odd_size(self, images_dir: Path) -> Images:
        return self.find_images(images_dir, "odd_size")

    def find_grayscale(self, images_dir: Path) -> Images:
        return self.find_images(images_dir, "grayscale")

    def find_issues(self, images_dir: Path, checks: list[str]) -> dict[str, Found]:
        # One Imagelab run, as CleanVision runs its default checks
        issues = [check for check in checks if check != "duplicates"]
        if "duplicates" in checks:
            issues += ["exact_duplicates", "near_duplicates"]
        imagelab = self.run(images_dir, issues)
        found: dict[str, Found] = {}
        for check in checks:
            if check == "duplicates":
                found[check] = Duplicates(
                    exact=to_sets(imagelab.info["exact_duplicates"]["sets"]),
                    near=to_sets(imagelab.info["near_duplicates"]["sets"]),
                )
            else:
                column = imagelab.issues[f"is_{check}_issue"]
                found[check] = to_images(imagelab.issues.index[column])
        return found
