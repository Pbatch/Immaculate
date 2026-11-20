from pathlib import Path

import immaculate_onnx
from checkers.base_checker import Checker, Duplicates, Found, Images, to_images, to_sets
from loguru import logger

import immaculate
from immaculate._checks import CHECKS


class ImmaculateChecker(Checker):
    """Immaculate has no thread setting; the runner limits it with RAYON_NUM_THREADS."""

    name = "Immaculate"

    def __init__(
        self, threads: int, method: immaculate.Method = "pdq", sscd_model: str | None = None
    ) -> None:
        """`sscd_model`, a path relative to the repository's root, replaces the SSCD model
        Immaculate's sscd extra installs, such as a student exported by
        dev/export_mobilenetv4_onnx.py."""
        super().__init__(threads)
        self.method = method
        if method == "phash":
            self.name = "Immaculate pHash"
        elif method == "sscd":
            self.name = "Immaculate SSCD"
        elif method == "sscd_distilled":
            self.name = "Immaculate SSCD distilled"
        if sscd_model is not None:
            model = Path(__file__).parents[2] / sscd_model
            if not model.exists():
                msg = f"{model} doesn't exist; export it with dev/export_mobilenetv4_onnx.py"
                raise FileNotFoundError(msg)
            immaculate_onnx.MODELS["sscd"] = model
            self.name = f"Immaculate SSCD ({model.stem})"

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        result = immaculate.find_issues(
            images_dir,
            [immaculate.ExactDuplicates(), immaculate.NearDuplicates(method=self.method)],
        )
        if result.skipped:
            logger.warning(f"{self.name} skipped {len(result.skipped)} undecodable images")
        return Duplicates(
            exact=to_sets(result.exact_duplicates), near=to_sets(result.near_duplicates)
        )

    def find_images(self, images_dir: Path, issue: immaculate.IssueName) -> Images:
        """The images that Immaculate's check `issue`, run alone, finds in `images_dir`."""
        result = immaculate.find_issues(images_dir, [CHECKS[issue]()])
        if result.skipped:
            logger.warning(f"{self.name} skipped {len(result.skipped)} undecodable images")
        return to_images(getattr(result, issue))

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

    def find_low_contrast(self, images_dir: Path) -> Images:
        return self.find_images(images_dir, "low_contrast")

    def find_truncated(self, images_dir: Path) -> Images:
        return self.find_images(images_dir, "truncated")

    def find_corrupt(self, images_dir: Path) -> Images:
        # Immaculate skips the images it can't decode, or that are empty, whatever checks run
        return to_images(immaculate.find_issues(images_dir, []).skipped)

    def find_issues(self, images_dir: Path, checks: list[str]) -> dict[str, Found]:
        # One call, which decodes each image once for every check
        issues: list[immaculate.Check] = []
        for check in checks:
            if check == "duplicates":
                issues += [
                    immaculate.ExactDuplicates(),
                    immaculate.NearDuplicates(method=self.method),
                ]
            elif check != "corrupt":  # Corrupt images are the ones skipped, whatever runs
                issues.append(CHECKS[check]())
        result = immaculate.find_issues(images_dir, issues)
        found: dict[str, Found] = {}
        for check in checks:
            if check == "duplicates":
                found[check] = Duplicates(
                    exact=to_sets(result.exact_duplicates), near=to_sets(result.near_duplicates)
                )
            elif check == "corrupt":
                found[check] = to_images(result.skipped)
            else:
                found[check] = to_images(getattr(result, check))
        return found
