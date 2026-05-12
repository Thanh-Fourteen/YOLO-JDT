"""Vendored Ultralytics 8.4.48 building blocks for YOLO11.

Source modules:
    conv.py  ← ultralytics/nn/modules/conv.py
    block.py ← ultralytics/nn/modules/block.py
    head.py  ← ultralytics/nn/modules/head.py + utils/tal.py + utils/utils.py

See `VENDORED.md` in this directory for the per-symbol mapping.
"""
from third_party.ultralytics_extract.block import (
    C2PSA, C2f, C3, C3k, C3k2, DFL, SPPF, Attention, Bottleneck, PSABlock,
)
from third_party.ultralytics_extract.conv import Concat, Conv, DWConv, autopad
from third_party.ultralytics_extract.head import (
    Detect, bias_init_with_prob, dist2bbox, make_anchors,
)

__all__ = [
    "autopad", "Conv", "DWConv", "Concat",
    "DFL", "SPPF", "Bottleneck", "C2f", "C3", "C3k", "C3k2",
    "Attention", "PSABlock", "C2PSA",
    "Detect", "make_anchors", "dist2bbox", "bias_init_with_prob",
]
