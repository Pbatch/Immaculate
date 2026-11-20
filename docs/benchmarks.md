# Benchmarks

## Compared with CleanVision

Every check CleanVision has (exact and near duplicates, dark, light, blurry, low information, odd aspect ratio, odd size and grayscale), run together in one call per library, as each runs by default, on the first 250 images, by name, of [INRIA Copydays](https://thoth.inrialpes.fr/~jegou/data.php.html#copydays) ([data](https://dl.fbaipublicfiles.com/vissl/datasets/copydays_original.tar.gz), [edited copies](https://dl.fbaipublicfiles.com/vissl/datasets/copydays_strong.tar.gz)).

| | Time | Peak memory | Images flagged |
|---|---:|---:|---:|
| Immaculate | **0.73 s** | **387 MiB** | 12 |
| CleanVision | 14.12 s | 2,974 MiB | 12 |

```bash
uv run benchmark/benchmark_checks.py --check duplicates dark light blurry low_information odd_aspect_ratio odd_size grayscale --checker immaculate cleanvision --images 250 --together
```

## Near duplicates

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
uv run benchmark/benchmark_checks.py --dataset ukbench-transformed --check duplicates --checker immaculate immaculate-phash immaculate-sscd immaculate-sscd-distilled imagededup-cnn cleanvision fastdup imagehash czkawka czkawka-dihedral
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
uv run benchmark/benchmark_checks.py --dataset disc2021 --check duplicates --checker immaculate immaculate-sscd-distilled czkawka imagehash
```

## Image checks

Each check runs on its own on the first 250 images, by name, of [INRIA Copydays](https://thoth.inrialpes.fr/~jegou/data.php.html#copydays) ([data](https://dl.fbaipublicfiles.com/vissl/datasets/copydays_original.tar.gz), [edited copies](https://dl.fbaipublicfiles.com/vissl/datasets/copydays_strong.tar.gz)): holiday photos of about 2 megapixels and heavily edited copies of them.

### Duplicates

| | Time | Peak memory | Pairs found | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Immaculate | **0.56 s** | **379 MiB** | 1 | 100% | **0.5%** |
| czkawka 12.0.2 | 0.95 s | 485 MiB | 0 | n/a | 0.0% |
| CleanVision | 2.82 s | 1,404 MiB | 1 | 100% | **0.5%** |
| fastdup | 3.44 s | 432 MiB | 0 | n/a | 0.0% |

```bash
uv run benchmark/benchmark_checks.py --check duplicates --checker immaculate cleanvision fastdup czkawka --images 250
```

### Dark

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | **0.70 s** | **356 MiB** | 0 |
| fastdup | 3.41 s | 436 MiB | 0 |
| CleanVision | 12.31 s | 3,364 MiB | 0 |

```bash
uv run benchmark/benchmark_checks.py --check dark --checker immaculate cleanvision fastdup --images 250
```

### Light

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | **0.72 s** | **337 MiB** | 0 |
| fastdup | 3.47 s | 437 MiB | 0 |
| CleanVision | 12.46 s | 3,334 MiB | 0 |

```bash
uv run benchmark/benchmark_checks.py --check light --checker immaculate cleanvision fastdup --images 250
```

### Blurry

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | **0.65 s** | **389 MiB** | 7 |
| CleanVision | 1.82 s | 622 MiB | 7 |
| fastdup | 3.44 s | 427 MiB | 19 |

```bash
uv run benchmark/benchmark_checks.py --check blurry --checker immaculate cleanvision fastdup --images 250
```

### Low information

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | **0.64 s** | **354 MiB** | 0 |
| CleanVision | 1.54 s | 581 MiB | 0 |

```bash
uv run benchmark/benchmark_checks.py --check low_information --checker immaculate cleanvision --images 250
```

### Odd aspect ratio

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | **0.50 s** | **360 MiB** | 3 |
| CleanVision | 1.47 s | 576 MiB | 3 |

```bash
uv run benchmark/benchmark_checks.py --check odd_aspect_ratio --checker immaculate cleanvision --images 250
```

### Odd size

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | **0.48 s** | **340 MiB** | 0 |
| CleanVision | 1.47 s | 571 MiB | 0 |

```bash
uv run benchmark/benchmark_checks.py --check odd_size --checker immaculate cleanvision --images 250
```

### Grayscale

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | **0.67 s** | **371 MiB** | 0 |
| CleanVision | 1.47 s | 531 MiB | 0 |

```bash
uv run benchmark/benchmark_checks.py --check grayscale --checker immaculate cleanvision --images 250
```

### Low contrast

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | 0.64 s | 373 MiB | 0 |

```bash
uv run benchmark/benchmark_checks.py --check low_contrast --checker immaculate --images 250
```

### Truncated

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | 0.49 s | 376 MiB | 0 |

```bash
uv run benchmark/benchmark_checks.py --check truncated --checker immaculate --images 250
```

### Corrupt

| | Time | Peak memory | Images found |
|---|---:|---:|---:|
| Immaculate | **0.47 s** | 370 MiB | 0 |
| czkawka 12.0.2 | 0.53 s | **322 MiB** | 0 |
| fastdup | 3.48 s | 431 MiB | 0 |

```bash
uv run benchmark/benchmark_checks.py --check corrupt --checker immaculate fastdup czkawka --images 250
```
