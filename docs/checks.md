# Checks

Each check is a class whose fields are its settings. Pass checks to `find_issues` with their settings:

```python
from immaculate import Blurry, Dark, NearDuplicates, find_issues

issues = find_issues("photos/", [Dark(brightness=0.25), NearDuplicates(method="sscd"), Blurry()])
```

On the command line, change a setting with `--set CHECK.FIELD=VALUE`, which can be repeated. The command line uses each check's name, given under its heading below, such as `dark` for `Dark`. The check must be one of the `--issues`, which by default are all of them:

```console
$ immaculate photos/ --set dark.brightness=0.25 --set odd_aspect_ratio.ratio=0.5
$ immaculate photos/ --issues near_duplicates --method sscd
```

`--method METHOD` is the same as `--set near_duplicates.method=METHOD`. A setting outside its valid range raises `ValueError`, or exits with an error on the command line.

## Exact duplicates

`ExactDuplicates`, or `exact_duplicates` on the command line

Finds sets of images with identical pixel data, so the same image saved with different metadata or in a different lossless format still matches.

No settings.

## Near duplicates

`NearDuplicates`, or `near_duplicates` on the command line

Finds sets of near-duplicate images. When exact duplicates are found too, near-duplicate sets identical to an exact set are left out.

| Setting | Default | Values |
|---|---|---|
| `method` | `"pdq"` | `"pdq"`, `"phash"`, `"sscd"`, `"sscd_distilled"` |

| Method | Matches | Catches |
|---|---|---|
| `"pdq"` | 256-bit PDQ hashes (Meta's perceptual hash) differing in at most 31 bits, in any rotation by a multiple of 90° or flip | Resized, recompressed, rotated and flipped copies |
| `"phash"` | Identical 64-bit perceptual hashes, as in the `imagehash` library | Resized and recompressed copies |
| `"sscd"` | Embeddings from Meta's SSCD copy detection model with cosine similarity of at least 0.75 | Heavier edits than hashes, such as crops and overlays, but much slower |
| `"sscd_distilled"` | As `"sscd"`, but with a small MobileNetV4 model trained to reproduce SSCD's embeddings | As `"sscd"`, about 4x faster, at some cost in recall |

With `"pdq"`, `"sscd"` and `"sscd_distilled"`, matches chain: if A matches B and B matches C, all three are one set. The SSCD methods need the `sscd` extra, which installs ONNX Runtime and the models: `pip install "immaculate[sscd]"`.

The thresholds aren't settings. See the [benchmarks](benchmarks.md) for how the methods compare.

## Dark

`Dark`, or `dark` on the command line

Finds images where fewer than `fraction` of the pixels are at least `brightness` bright. A pixel's brightness, from 0 to 1, is `sqrt(0.241 R² + 0.691 G² + 0.068 B²) / 255`.

| Setting | Default | Range | Raise it to flag |
|---|---|---|---|
| `brightness` | 0.32 | 0 to 1 | More images: a pixel must be brighter to count as bright |
| `fraction` | 0.01 | 0 to 1 | More images: more of the pixels must be bright |

## Light

`Light`, or `light` on the command line

Finds images where fewer than `fraction` of the pixels are at most `brightness` bright, with brightness measured as for [dark](#dark) images.

| Setting | Default | Range | Raise it to flag |
|---|---|---|---|
| `brightness` | 0.95 | 0 to 1 | Fewer images: more pixels count as not light |
| `fraction` | 0.05 | 0 to 1 | More images: more of the pixels must be not light |

## Blurry

`Blurry`, or `blurry` on the command line

Finds images whose sharpness is below `sharpness`. Sharpness is the standard deviation of the Laplacian (4 times each pixel less its 4 neighbours) of the image in grayscale, shrunk to at most 256 pixels on its longer side by averaging.

| Setting | Default | Range | Raise it to flag |
|---|---|---|---|
| `sharpness` | 5.0 | At least 0 | More images |

The default flags images blurred by a Gaussian of about 2 of those pixels, but also some smooth, hazy scenes. Images too flat to have edges, which have [low information](#low-information) instead, and images under 3 pixels wide or tall are never blurry.

## Low information

`LowInformation`, or `low_information` on the command line

Finds images, such as blank ones, whose color histogram has an entropy below `entropy` bits. The histogram is the red, green and blue channels' 256-bin histograms joined into one, or the single gray channel's for grayscale images, as Pillow's `Image.entropy` measures.

| Setting | Default | Range | Raise it to flag |
|---|---|---|---|
| `entropy` | 3.0 | At least 0 | More images |

The entropy is at most log₂ 768 ≈ 9.6 bits for color images and 8 bits for grayscale ones.

## Odd aspect ratio

`OddAspectRatio`, or `odd_aspect_ratio` on the command line

Finds unusually long or tall images: those whose shorter side divided by their longer side is below `ratio`.

| Setting | Default | Range | Raise it to flag |
|---|---|---|---|
| `ratio` | 0.35 | 0 to 1 | More images; 1 flags every image that isn't square |

## Odd size

`OddSize`, or `odd_size` on the command line

Finds images much larger or smaller than the others. An image's size is the square root of its area, and it is odd if it is more than `iqr_factor` interquartile ranges below the first quartile of all the images' sizes, or above the third. If at least half the images have the same size, any image of a different size is odd.

| Setting | Default | Range | Raise it to flag |
|---|---|---|---|
| `iqr_factor` | 3.0 | At least 0 | Fewer images; 0 flags every image outside the quartiles |

## Grayscale

`Grayscale`, or `grayscale` on the command line

Finds images stored as grayscale, or whose pixels' red, green and blue values all differ by at most `tolerance`.

| Setting | Default | Range | Raise it to flag |
|---|---|---|---|
| `tolerance` | 2 | Whole numbers from 0 to 255 | More images: color files with slightly tinted pixels count as gray |

## Low contrast

`LowContrast`, or `low_contrast` on the command line

Finds washed-out, foggy or otherwise flat-looking images: those whose RMS contrast, the standard deviation of their grayscale values divided by 255, is below `contrast`. Grayscale values are computed as by Pillow's `convert("L")`.

| Setting | Default | Range | Raise it to flag |
|---|---|---|---|
| `contrast` | 0.05 | 0 to 1 | More images |

The contrast is at most 0.5, for an image half black and half white.

## Truncated

`Truncated`, or `truncated` on the command line

Finds images whose files end early, such as interrupted downloads: those whose file ends before its end-of-image marker (JPEG), its `IEND` chunk (PNG) or the length its header gives (WebP). Data after the end, such as a motion photo's video, doesn't matter.

No settings.

Truncated JPEGs still decode, with whatever is missing filled in gray, so they are checked like other images too. Truncated files in other formats usually fail to decode, so they are skipped instead.
