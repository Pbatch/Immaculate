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
result = find_duplicates("photos/", progress=lambda done, total: print(f"{done}/{total}"))
```

Or from the command line:

```bash
immaculate photos/                # readable summary
immaculate photos/ --json         # {"exact": [[path, ...], ...], "near": [[path, ...], ...], "skipped": [path, ...]}
immaculate photos/ --method sscd  # pdq (default), phash, sscd or sscd_distilled
```

## Methods

`method` (or `--method`) chooses how near duplicates are found:

| Method | Catches |
|---|---|
| `"pdq"` (default) | Resized, recompressed, flipped and 90°-rotated copies |
| `"phash"` | Resized and recompressed copies |
| `"sscd_distilled"` | As `"sscd"`, using a small MobileNetV4 model distilled from SSCD; finds somewhat fewer |
| `"sscd"` | Also crops, rotations by other angles and overlays, using Meta's SSCD copy detection model |

## Benchmark

### UKBench (11k images, 27k pairs)

1,800 photos from [UKBench](https://archive.org/details/ukbench), each with five edited copies (a crop, horizontal and vertical flips, a 25° rotation and a resize to a square), as in [imagededup's benchmark](https://idealo.github.io/imagededup/user_guide/benchmarks/).

| | Time | Peak memory | Pairs found | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Immaculate (`pdq`) | **2.35 s** | 59 MiB | 10,782 | 100% | 39.9% |
| Immaculate (`phash`) | 2.61 s | **54 MiB** | 1,686 | 100% | 6.2% |
| czkawka | 3.79 s | 64 MiB | 31 | 100% | 0.1% |
| imagehash (`pHash`) | 4.11 s | 492 MiB | 1,693 | 100% | 6.3% |
| czkawka (rotations and flips) | 20.74 s | 120 MiB | 4,359 | 100% | 16.1% |
| CleanVision | 25.57 s | 547 MiB | 1,693 | 100% | 6.3% |
| fastdup | 28.92 s | 337 MiB | 9,443 | 100% | 35.0% |
| Immaculate (`sscd_distilled`) | 42.72 s | 148 MiB | 19,886 | 100% | 73.6% |
| imagededup (`CNN`) | 64.00 s | 2,257 MiB | 20,620 | 98.8% | 75.5% |
| Immaculate (`sscd`) | 183.93 s | 188 MiB | 23,196 | 100% | **85.9%** |

```bash
uv run benchmark/benchmark_duplicates.py --dataset ukbench-transformed --dedup immaculate immaculate-phash immaculate-sscd immaculate-sscd-distilled imagededup-cnn cleanvision fastdup imagehash czkawka czkawka-dihedral
```

### DISC2021 (60k images, 10k pairs)

50,000 development queries from Meta's [DISC2021](https://arxiv.org/abs/2106.09672) ([data](https://sites.google.com/view/isc2021/dataset)), plus the 10,000 reference photos that 10,000 of them copy with heavy edits (crops, overlays, filters, screenshots and blends).

| | Time | Peak memory | Pairs found | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Immaculate (`pdq`) | **29.13 s** | **149 MiB** | 1,731 | 100% | 17.3% |
| imagehash (`pHash`) | 34.38 s | 639 MiB | 316 | 100% | 3.2% |
| czkawka | 43.94 s | 198 MiB | 89 | 96.6% | 0.9% |
| Immaculate (`sscd_distilled`) | 290.74 s | 1,672 MiB | 1,987 | 99.9% | **19.9%** |

```bash
uv run benchmark/benchmark_duplicates.py --dataset disc2021 --dedup immaculate immaculate-sscd-distilled czkawka imagehash
```
