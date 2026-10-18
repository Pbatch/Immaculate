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
result.near  # sets of visually identical images (matching perceptual hashes)
result.skipped  # images that couldn't be decoded
```

Or from the command line:

```bash
python -m immaculate photos/         # readable summary
python -m immaculate photos/ --json  # {"exact": [[path, ...], ...], "near": [[path, ...], ...], "skipped": [path, ...]}
```

### Near-duplicate methods

Choose how near duplicates are found with `find_duplicates(..., method=...)` or `--method`:

- `pdq` (default): Meta's 256-bit [PDQ](https://github.com/facebook/ThreatExchange/tree/main/pdq) hashes within 31 bits, in any 90° rotation or flip. Catches resized, recompressed, rotated and flipped copies.
- `phash`: identical 64-bit perceptual hashes, as in `imagehash`. Catches resized and recompressed copies.

Nothing is moved or deleted. Errors (missing or unreadable folder) go to stderr with exit code 1.

## Benchmark

On INRIA Copydays (157 photos plus 229 heavily edited copies: printed and scanned, blurred, painted over), against imagededup's CNN method, which found the most duplicates of the open source libraries tested (CleanVision, fastdup, imagededup, pdqhash):

| | Time | Peak memory | Pairs found | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Immaculate 0.1.0 (`pdq`) | **0.79 s** | **341 MiB** | 2 | 100% | 0.6% |
| imagededup 0.3.3 (`CNN`) | 7.44 s | 1,195 MiB | 29 | 100% | 9.2% |

Immaculate is 9x faster and uses a third of the memory. Copydays' edits are much heavier than resizing or recompression, so hash-based methods find few of them, and the CNN finds more. Recall is out of 314 labelled duplicate pairs. Times are the mean of 3 runs on an AMD Ryzen 7 5800H with all 16 threads. To reproduce, see [benchmark_duplicates.py](benchmark/benchmark_duplicates.py):

```bash
uv run benchmark/benchmark_duplicates.py --dataset copydays --repeats 3 --dedup immaculate --dedup imagededup-cnn
```

## License

Proprietary. You may install and use this package, but you may not copy, modify, or redistribute its source code. See [LICENSE](LICENSE).
