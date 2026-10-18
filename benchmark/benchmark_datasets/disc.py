import csv
import io
import struct
import urllib.request
import zipfile
import zlib
from collections.abc import Buffer
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from benchmark_datasets.base_dataset import Dataset, GroundTruth, extract, fetch, group_pairs
from loguru import logger

BASE_URL = "https://dl.fbaipublicfiles.com/image_similarity_challenge/public"
REFERENCE_ZIPS = 20
REFERENCES_PER_ZIP = 50_000
DOWNLOAD_THREADS = 16
LOCAL_HEADER = struct.Struct("<4s22xHH")
"""A zip local file header's signature, file name length and extra field length."""
LOCAL_HEADER_SLACK = 1024
"""Bytes to fetch beyond a member's central directory sizes, in case its local extra field is
longer."""


def read_range(url: str, start: int, length: int) -> bytes:
    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - fixed https URLs
    with urllib.request.urlopen(request) as response:  # noqa: S310
        return response.read()


class RemoteFile(io.RawIOBase):
    """A file on an HTTP server that supports range requests, read without downloading it all."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.position = 0
        request = urllib.request.Request(url, method="HEAD")  # noqa: S310 - fixed https URLs
        with urllib.request.urlopen(request) as response:  # noqa: S310
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer: Buffer) -> int:
        view = memoryview(buffer).cast("B")
        length = min(len(view), self.size - self.position)
        if length <= 0:
            return 0
        data = read_range(self.url, self.position, length)
        view[: len(data)] = data
        self.position += len(data)
        return len(data)


def read_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """One member of a remote zip, fetched with a single range request."""
    start = 30 + len(info.filename.encode())
    chunk = read_range(
        url, info.header_offset, start + len(info.extra) + LOCAL_HEADER_SLACK + info.compress_size
    )
    signature, name_length, extra_length = LOCAL_HEADER.unpack_from(chunk)
    if signature != b"PK\x03\x04":
        msg = f"bad local header for {info.filename} in {url}"
        raise zipfile.BadZipFile(msg)
    data_start = 30 + name_length + extra_length
    data = chunk[data_start : data_start + info.compress_size]
    if len(data) != info.compress_size:
        msg = f"{info.filename} in {url} has a local extra field over {LOCAL_HEADER_SLACK} bytes"
        raise zipfile.BadZipFile(msg)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -zlib.MAX_WBITS)
    elif info.compress_type != zipfile.ZIP_STORED:
        msg = f"{info.filename} in {url} uses unsupported compression {info.compress_type}"
        raise zipfile.BadZipFile(msg)
    if zlib.crc32(data) != info.CRC:
        msg = f"CRC mismatch for {info.filename} in {url}"
        raise zipfile.BadZipFile(msg)
    return data


def save_member(url: str, images_dir: Path, info: zipfile.ZipInfo) -> None:
    """Save one member of a remote zip into `images_dir` under its base name."""
    (images_dir / Path(info.filename).name).write_bytes(read_member(url, info))


class DISC2021(Dataset):
    """Meta's DISC2021 development queries and the reference images they copy (~9 GB).

    The 50,000 development queries are edited copies of images (crops, overlays, filters,
    screenshots, blending and more), 10,000 of them of an image in DISC2021's 1,000,000-image
    reference set and the rest of other images, as distractors. This uses every query plus
    only the 10,000 references they copy, fetched from the reference zips with range requests
    rather than downloading all 175 GB of them. Images keep their DISC2021 names,
    `Q<number>.jpg` for queries and `R<number>.jpg` for references. As in DISC2021, only
    query-reference pairs are scored: pairs of two queries or two references aren't labelled,
    so they're ignored.
    """

    key = "disc2021"
    name = "DISC2021 (dev queries + matched references)"

    def build(self, images_dir: Path, data_dir: Path) -> None:
        dataset_dir = data_dir / self.key
        references = sorted(set(self.matches(dataset_dir).values()))
        archive = fetch(f"{BASE_URL}/dev_queries.zip", dataset_dir / "dev_queries.zip")
        extract(archive, images_dir, "images/queries")
        archive.unlink()

        by_zip: dict[int, list[str]] = {}
        for reference in references:
            by_zip.setdefault(int(reference[1:]) // REFERENCES_PER_ZIP, []).append(reference)
        with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
            for part in range(REFERENCE_ZIPS):
                url = f"{BASE_URL}/references_{part}.zip"
                logger.info(f"Fetching {len(by_zip.get(part, []))} images from {url} ...")
                with zipfile.ZipFile(io.BufferedReader(RemoteFile(url), 1 << 20)) as archive:
                    infos = [
                        archive.getinfo(f"images/references/{reference}.jpg")
                        for reference in by_zip.get(part, [])
                    ]
                list(pool.map(partial(save_member, url, images_dir), infos))

    def matches(self, dataset_dir: Path) -> dict[str, str]:
        """The reference each copying query copies, by name without extension."""
        labels = fetch(f"{BASE_URL}/dev_ground_truth.csv", dataset_dir / "dev_ground_truth.csv")
        with labels.open(newline="") as file:
            return dict(csv.reader(file))

    def ground_truth(self, images_dir: Path) -> GroundTruth:
        matches = self.matches(images_dir.parent)
        return GroundTruth(
            group_pairs(images_dir.iterdir(), lambda path: matches.get(path.stem, path.stem)),
            labelled=is_query_reference,
        )


def is_query_reference(pair: frozenset[Path]) -> bool:
    """Whether a pair is one query and one reference, the only pairs DISC2021 labels."""
    return sorted(path.name[0] for path in pair) == ["Q", "R"]
