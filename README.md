# Immaculate

[![PyPI](https://img.shields.io/pypi/v/immaculate)](https://pypi.org/project/immaculate/)
[![Python](https://img.shields.io/pypi/pyversions/immaculate)](https://pypi.org/project/immaculate/)
[![CI](https://github.com/Pbatch/Immaculate/actions/workflows/ci.yml/badge.svg)](https://github.com/Pbatch/Immaculate/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-pbatch.github.io-blue)](https://pbatch.github.io/Immaculate/)

Find exact and near-duplicate images in a folder, fast.

## Installation

```bash
pip install immaculate
pip install "immaculate[sscd]"  # adds ONNX Runtime and the models for the sscd and sscd_distilled methods
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
| Immaculate (`phash`) | **2.69 s** | **50 MiB** | 1,686 | 100% | 6.2% |
| Immaculate (`pdq`) | 2.78 s | 56 MiB | 10,782 | 100% | 39.9% |
| czkawka | 3.72 s | 61 MiB | 31 | 100% | 0.1% |
| imagehash (`pHash`) | 4.06 s | 470 MiB | 1,693 | 100% | 6.3% |
| czkawka (rotations and flips) | 23.49 s | 116 MiB | 4,359 | 100% | 16.1% |
| CleanVision | 26.03 s | 520 MiB | 1,693 | 100% | 6.3% |
| fastdup | 32.26 s | 279 MiB | 9,443 | 100% | 35.0% |
| Immaculate (`sscd_distilled`) | 36.71 s | 141 MiB | 19,886 | 100% | 73.6% |
| imagededup (`CNN`) | 65.00 s | 1,414 MiB | 20,620 | 98.8% | 75.5% |
| Immaculate (`sscd`) | 172.51 s | 189 MiB | 23,196 | 100% | **85.9%** |

```bash
uv run benchmark/benchmark_duplicates.py --dataset ukbench-transformed --dedup immaculate immaculate-phash immaculate-sscd immaculate-sscd-distilled imagededup-cnn cleanvision fastdup imagehash czkawka czkawka-dihedral
```

### DISC2021 (60k images, 10k pairs)

50,000 development queries from Meta's [DISC2021](https://arxiv.org/abs/2106.09672) ([data](https://sites.google.com/view/isc2021/dataset)), plus the 10,000 reference photos that 10,000 of them copy with heavy edits (crops, overlays, filters, screenshots and blends).

| | Time | Peak memory | Pairs found | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Immaculate (`pdq`) | **23.48 s** | **146 MiB** | 1,731 | 100% | 17.3% |
| imagehash (`pHash`) | 35.89 s | 620 MiB | 316 | 100% | 3.2% |
| czkawka | 38.64 s | 194 MiB | 89 | 96.6% | 0.9% |
| Immaculate (`sscd_distilled`) | 243.23 s | 488 MiB | 1,987 | 99.9% | **19.9%** |

```bash
uv run benchmark/benchmark_duplicates.py --dataset disc2021 --dedup immaculate immaculate-sscd-distilled czkawka imagehash
```
