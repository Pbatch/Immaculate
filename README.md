# Immaculate

Find exact and near-duplicate images in a folder, fast.

## Installation

```bash
pip install immaculate
```

## Usage

```python
from immaculate import find_duplicates

result = find_duplicates("photos/")  # searches subfolders too
result.exact  # sets of images with identical pixels
result.near  # sets of near-duplicate images, such as resized or flipped copies
result.skipped  # images that couldn't be decoded or are empty

result = find_duplicates("photos/", method="sscd")  # also catch crops and other heavier edits
```

`method` chooses how near duplicates are found:

| Method | Catches | Speed |
|---|---|---|
| `"pdq"` (default) | Resized, recompressed, flipped and 90°-rotated copies | Fast |
| `"phash"` | Resized and recompressed copies | Fast |
| `"sscd"` | Also crops, rotations by other angles and overlays, using Meta's SSCD copy detection model | About 70x slower than `"pdq"` |

Or from the command line:

```bash
python -m immaculate photos/                # readable summary
python -m immaculate photos/ --json         # {"exact": [[path, ...], ...], "near": [[path, ...], ...], "skipped": [path, ...]}
python -m immaculate photos/ --method sscd  # pdq (default), phash or sscd
```

Nothing is moved or deleted. Errors (missing or unreadable folder) go to stderr with exit code 1.

## Benchmark

### UKBench

1,800 photos from [UKBench](https://archive.org/details/ukbench), each with five edited copies (a crop, horizontal and vertical flips, a 25° rotation and a resize to a square), as in [imagededup's benchmark](https://idealo.github.io/imagededup/user_guide/benchmarks/). Recall is out of the 27,000 labelled pairs.

| | Time | Peak memory | Pairs found | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Immaculate (`pdq`) | **2.67 s** | **56 MiB** | 10,782 | 100% | 39.9% |
| czkawka | 3.79 s | 64 MiB | 31 | 100% | 0.1% |
| imagehash (`pHash`) | 4.11 s | 492 MiB | 1,693 | 100% | 6.3% |
| czkawka (rotations and flips) | 20.74 s | 120 MiB | 4,359 | 100% | 16.1% |
| CleanVision | 25.57 s | 547 MiB | 1,693 | 100% | 6.3% |
| fastdup | 28.92 s | 337 MiB | 9,443 | 100% | 35.0% |
| imagededup (`CNN`) | 64.00 s | 2,257 MiB | 20,620 | 98.8% | 75.5% |
| Immaculate (`sscd`) | 183.93 s | 188 MiB | 23,196 | 100% | **85.9%** |

```bash
uv run benchmark/benchmark_duplicates.py --dataset ukbench-transformed --dedup immaculate immaculate-sscd imagededup-cnn cleanvision fastdup imagehash czkawka czkawka-dihedral
```

### DISC2021

50,000 development queries from Meta's [DISC2021](https://arxiv.org/abs/2106.09672) ([data](https://sites.google.com/view/isc2021/dataset)), plus the 10,000 reference photos that 10,000 of them copy with heavy edits (crops, overlays, filters, screenshots and blends). The other 40,000 queries are distractors. Recall is out of the 10,000 labelled pairs.

| | Time | Peak memory | Pairs found | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Immaculate (`pdq`) | **29.13 s** | **149 MiB** | 1,731 | 100% | **17.3%** |
| imagehash (`pHash`) | 34.38 s | 639 MiB | 316 | 100% | 3.2% |
| czkawka | 43.94 s | 198 MiB | 89 | 96.6% | 0.9% |

```bash
uv run benchmark/benchmark_duplicates.py --dataset disc2021 --dedup immaculate czkawka imagehash
```
