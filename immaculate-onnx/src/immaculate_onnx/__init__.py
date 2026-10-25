"""The models for Immaculate's "sscd" and "sscd_distilled" methods."""

from pathlib import Path

SSCD_MODEL = Path(__file__).parent / "sscd.onnx"
"""SSCD's `sscd_disc_mixup` model quantised to int8 for ONNX Runtime. Meta's weights are MIT
licensed; see `SSCD_LICENSE` in the same folder."""
SSCD_DISTILLED_MODEL = Path(__file__).parent / "sscd_distilled.onnx"
"""A MobileNetV4 Conv-S student trained to reproduce SSCD's embeddings, with fp16 weights and
fp32 maths, made by `dev/export_mobilenetv4_onnx.py`. Its trunk starts from timm's
ImageNet-trained weights (Apache 2.0), and it was distilled from SSCD (`SSCD_LICENSE`) on
Conceptual Captions 3M images."""
MODELS: dict[str, Path] = {"sscd": SSCD_MODEL, "sscd_distilled": SSCD_DISTILLED_MODEL}
"""The model each method runs."""
