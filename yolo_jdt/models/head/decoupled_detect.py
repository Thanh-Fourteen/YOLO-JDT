"""YOLO11 decoupled Detect head wrapper.

This module re-exports the vendored `Detect` from
`third_party/ultralytics_extract/head.py` and pins the construction
parameters used by YOLO11 (reg_max=16, no end2end, legacy=False).

Forward signature: `forward(features_list) -> raw_outputs`. In training
mode, raw_outputs is a list of per-level concatenated (box_dfl + cls)
feature maps. In eval mode, raw_outputs is `(decoded[B,84,A], raw_per_level)`.

Stride must be set on the head before its first inference call (the
inference path uses stride to convert anchor-grid coordinates into
absolute box pixels). For YOLO11 input 640: stride = (8, 16, 32).
"""
from __future__ import annotations

import torch

from third_party.ultralytics_extract import Detect

__all__ = ["DecoupledDetect"]


class DecoupledDetect(Detect):
    """YOLO11 detection head with stride preset for the standard 640×640 input."""

    def __init__(self, nc: int = 80, ch: tuple = (), reg_max: int = 16,
                 strides: tuple[float, ...] = (8.0, 16.0, 32.0)):
        super().__init__(nc=nc, reg_max=reg_max, ch=ch)
        if len(strides) != self.nl:
            raise ValueError(f"strides len {len(strides)} != nl {self.nl}")
        self.stride = torch.tensor(strides, dtype=torch.float)
        # Bias init only after stride is populated (per Ultralytics convention).
        self.bias_init()
