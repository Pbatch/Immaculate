from pathlib import Path

import numpy as np
from dedups.base_dedup import IMAGE_EXTENSIONS, Dedup, Duplicates
from loguru import logger
from PIL import Image

MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/{model}.torchscript.pt"
CHUNK_ROWS = 2048
"""Rows of the similarity matrix computed at a time, to bound memory."""


class SSCDDedup(Dedup):
    """Meta's SSCD copy detection model, pairing images whose embeddings have cosine similarity
    of at least `threshold`.

    Follows SSCD's README: the small edge resized to 288, ImageNet normalisation, and 0.75 as
    the threshold for `sscd_disc_mixup` (90% precision on DISC). The model is downloaded to
    PyTorch's hub cache on first use, and runs on `device`, a PyTorch device such as "cuda".
    Like imagededup, every duplicate pair is returned as a set of two, all as near duplicates.
    """

    def __init__(
        self,
        threads: int,
        model: str = "sscd_disc_mixup",
        threshold: float = 0.75,
        device: str = "cpu",
    ) -> None:
        super().__init__(threads)
        self.model = model
        self.threshold = threshold
        self.device = device
        self.name = f"SSCD {model.removeprefix('sscd_')} >={threshold}"
        if device != "cpu":
            self.name += f" ({device.upper()})"

    def find_duplicates(self, images_dir: Path) -> Duplicates:
        # Imported here because PyTorch is slow to import
        import torch  # noqa: PLC0415

        torch.set_num_threads(self.threads)
        model = torch.jit.load(self.model_path(), map_location=self.device).eval()
        paths = sorted(
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        # Images keep their aspect ratio, so they're decoded in parallel but embedded one at a time
        loader = torch.utils.data.DataLoader(
            ImageFolder(paths), batch_size=None, num_workers=self.threads
        )
        embedded: list[Path] = []
        embeddings = []
        with torch.inference_mode():
            for path, tensor in zip(paths, loader, strict=True):
                if tensor is None:
                    continue
                embedded.append(path)
                embeddings.append(model(tensor.unsqueeze(0).to(self.device))[0].cpu().numpy())
        if skipped := len(paths) - len(embedded):
            logger.warning(f"SSCD skipped {skipped} undecodable images")
        if not embeddings:
            return Duplicates(exact=None, near=set())

        pairs = self.close_pairs(np.stack(embeddings))
        return Duplicates(
            exact=None,
            near={frozenset((embedded[a].resolve(), embedded[b].resolve())) for a, b in pairs},
        )

    def model_path(self) -> Path:
        import torch  # noqa: PLC0415

        path = Path(torch.hub.get_dir()) / "checkpoints" / f"{self.model}.torchscript.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(MODEL_URL.format(model=self.model), str(path))
        return path

    def close_pairs(self, embeddings: np.ndarray) -> list[tuple[int, int]]:
        """Index pairs (a, b), a < b, at or above the threshold. SSCD's embeddings are already
        L2 normalised, so a dot product is the cosine similarity."""
        pairs = []
        for start in range(0, len(embeddings), CHUNK_ROWS):
            similarities = embeddings[start : start + CHUNK_ROWS] @ embeddings.T
            rows, columns = np.nonzero(similarities >= self.threshold)
            rows += start
            keep = rows < columns
            pairs.extend(zip(rows[keep].tolist(), columns[keep].tolist(), strict=True))
        return pairs


class ImageFolder:
    """Decodes and preprocesses images for SSCD, giving None for any that can't be decoded."""

    def __init__(self, paths: list[Path]) -> None:
        from torchvision import transforms  # noqa: PLC0415

        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(288),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):  # noqa: ANN204 - a tensor, or None
        try:
            with Image.open(self.paths[index]) as image:
                return self.transform(image.convert("RGB"))
        except OSError:
            return None
