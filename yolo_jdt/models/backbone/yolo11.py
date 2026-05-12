"""YOLO11 backbone (layers 0-10 in `cfg/models/11/yolo11.yaml`).

Forward signature: `forward(image) -> (P3, P4, P5)`. P3 has stride 8,
P4 stride 16, P5 stride 32. P5 is the C2PSA output (last backbone layer).

Architecture (yaml lines 0-10) — channels shown are for scale `s`:
    0: Conv  (3 → 32, k=3, s=2)               P1/2
    1: Conv  (32 → 64, k=3, s=2)              P2/4
    2: C3k2  (64 → 128, n=1, c3k=False)
    3: Conv  (128 → 128, k=3, s=2)            P3/8
    4: C3k2  (128 → 256, n=1, c3k=False)      → output P3
    5: Conv  (256 → 256, k=3, s=2)            P4/16
    6: C3k2  (256 → 256, n=1, c3k=True)       → output P4
    7: Conv  (256 → 512, k=3, s=2)            P5/32
    8: C3k2  (512 → 512, n=1, c3k=True)
    9: SPPF  (512 → 512, k=5)
   10: C2PSA (512 → 512, n=1)                 → output P5

State_dict mapping uses Ultralytics' flat layer index `model.{i}.{...}` —
see `yolo_jdt/weights/loader.py` for the rename rule.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from third_party.ultralytics_extract import C2PSA, C3k2, Conv, SPPF
from yolo_jdt.models._scale import ScaleConfig, scale_channels, scale_params, scale_repeats


class YOLO11Backbone(nn.Module):
    """Backbone-only forward producing (P3, P4, P5) feature maps."""

    def __init__(self, scale: str = "s"):
        super().__init__()
        cfg = scale_params(scale)
        self.scale = scale
        self.cfg = cfg
        # Layer 0..10 channel widths, after width+max_channels scaling
        c0 = scale_channels(64, cfg)        # layer 0
        c1 = scale_channels(128, cfg)       # layer 1
        c2 = scale_channels(256, cfg)       # layer 2 (C3k2)
        c3 = scale_channels(256, cfg)       # layer 3 (Conv)
        c4 = scale_channels(512, cfg)       # layer 4 (C3k2) → P3
        c5 = scale_channels(512, cfg)       # layer 5 (Conv)
        c6 = scale_channels(512, cfg)       # layer 6 (C3k2) → P4
        c7 = scale_channels(1024, cfg)      # layer 7 (Conv)
        c8 = scale_channels(1024, cfg)      # layer 8 (C3k2)
        c9 = scale_channels(1024, cfg)      # layer 9 (SPPF)
        c10 = scale_channels(1024, cfg)     # layer 10 (C2PSA) → P5
        n = scale_repeats(2, cfg)           # depth-scaled repeat count (yaml uses n=2)
        c3k_force = cfg.force_c3k           # m/l/x override

        self.layer0  = Conv(3,   c0, k=3, s=2)
        self.layer1  = Conv(c0,  c1, k=3, s=2)
        self.layer2  = C3k2(c1,  c2, n=n, c3k=c3k_force or False, e=0.25)
        self.layer3  = Conv(c2,  c3, k=3, s=2)
        self.layer4  = C3k2(c3,  c4, n=n, c3k=c3k_force or False, e=0.25)
        self.layer5  = Conv(c4,  c5, k=3, s=2)
        self.layer6  = C3k2(c5,  c6, n=n, c3k=True)            # yaml True → True everywhere
        self.layer7  = Conv(c6,  c7, k=3, s=2)
        self.layer8  = C3k2(c7,  c8, n=n, c3k=True)
        self.layer9  = SPPF(c8,  c9, k=5)
        self.layer10 = C2PSA(c9, c10, n=n)

        self.out_channels = (c4, c6, c10)   # P3, P4, P5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        p3 = x
        x = self.layer5(x)
        x = self.layer6(x)
        p4 = x
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        p5 = x
        return p3, p4, p5
