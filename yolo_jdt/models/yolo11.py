"""End-to-end YOLO11 model assembly: backbone + neck + head.

This wrapper composes `YOLO11Backbone`, `YOLO11PANet`, `DecoupledDetect` into
a single nn.Module with the same forward semantics as Ultralytics' YOLO11.

Forward signature:
    train mode: returns the 3 raw per-level outputs (list of [B, 4*reg_max+nc, H, W])
    eval mode:  returns (decoded[B, 4+nc, A], raw_per_level)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from yolo_jdt.models.backbone.yolo11 import YOLO11Backbone
from yolo_jdt.models.head.decoupled_detect import DecoupledDetect
from yolo_jdt.models.neck.panet import YOLO11PANet


class YOLO11(nn.Module):
    """Standalone YOLO11 detection model. Compatible with Ultralytics state_dict
    via `yolo_jdt/weights/loader.py`."""

    def __init__(self, scale: str = "s", nc: int = 80, reg_max: int = 16,
                 strides: tuple[float, ...] = (8.0, 16.0, 32.0)):
        super().__init__()
        self.scale = scale
        self.nc = nc
        self.backbone = YOLO11Backbone(scale)
        self.neck = YOLO11PANet(scale, in_channels=self.backbone.out_channels)
        self.head = DecoupledDetect(nc=nc, ch=self.neck.out_channels,
                                    reg_max=reg_max, strides=strides)

    def forward(self, x: torch.Tensor):
        p3, p4, p5 = self.backbone(x)
        feat16, feat19, feat22 = self.neck(p3, p4, p5)
        return self.head([feat16, feat19, feat22])
