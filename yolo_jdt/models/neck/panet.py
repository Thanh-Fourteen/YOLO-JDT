"""YOLO11 PANet neck (layers 11-22 in `cfg/models/11/yolo11.yaml`).

Forward signature: `forward(P3, P4, P5) -> (P3', P4', P5')`. Strides
preserved (8, 16, 32). The neck does top-down upsampling + concatenation
with backbone features (PAN), then bottom-up downsampling for the final
two scales (FPN-PAN style).

Architecture (yaml lines 11-22) for scale `s`:
   11: Upsample(P5, ×2)
   12: Concat (upsampled, P4)
   13: C3k2 (cv4 channels = 256, c3k=False)
   14: Upsample(layer 13, ×2)
   15: Concat (upsampled, P3)
   16: C3k2 (cv5 channels = 128, c3k=False)            → output P3'
   17: Conv (down, stride 2)
   18: Concat (layer 17, layer 13)
   19: C3k2 (cv6 channels = 256, c3k=False)            → output P4'
   20: Conv (down, stride 2)
   21: Concat (layer 20, layer 10/P5)
   22: C3k2 (cv7 channels = 512, c3k=True)             → output P5'

For scales m/l/x, c3k=True is forced for ALL C3k2 in this neck.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from third_party.ultralytics_extract import C3k2, Conv
from yolo_jdt.models._scale import scale_channels, scale_params, scale_repeats


class YOLO11PANet(nn.Module):
    """PANet neck producing 3 scales of decoupled features."""

    def __init__(self, scale: str = "s",
                 in_channels: tuple[int, int, int] = None):
        super().__init__()
        cfg = scale_params(scale)
        self.scale = scale
        # Channel widths (yaml-arg → after scaling)
        c13 = scale_channels(512, cfg)
        c16 = scale_channels(256, cfg)
        c19 = scale_channels(512, cfg)
        c22 = scale_channels(1024, cfg)
        n = scale_repeats(2, cfg)
        c3k_n = cfg.force_c3k  # False for n/s, True for m/l/x

        # Backbone output channels — must match backbone.out_channels
        if in_channels is None:
            # Default: query backbone for the same scale
            from yolo_jdt.models.backbone.yolo11 import YOLO11Backbone
            in_channels = YOLO11Backbone(scale).out_channels
        c_p3, c_p4, c_p5 = in_channels

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down: P5 → P4' intermediate
        self.layer13 = C3k2(c_p5 + c_p4, c13, n=n, c3k=c3k_n)
        # Top-down: → P3'
        self.layer16 = C3k2(c13 + c_p3, c16, n=n, c3k=c3k_n)
        # Bottom-up: P3' → P4' final
        self.layer17 = Conv(c16, c16, k=3, s=2)
        self.layer19 = C3k2(c16 + c13, c19, n=n, c3k=c3k_n)
        # Bottom-up: P4' → P5' final
        self.layer20 = Conv(c19, c19, k=3, s=2)
        self.layer22 = C3k2(c19 + c_p5, c22, n=n, c3k=True)  # yaml True → True everywhere

        self.out_channels = (c16, c19, c22)  # P3', P4', P5'

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        # Top-down pathway
        x = self.up(p5)
        x = torch.cat((x, p4), dim=1)
        feat13 = self.layer13(x)

        x = self.up(feat13)
        x = torch.cat((x, p3), dim=1)
        feat16 = self.layer16(x)  # P3' (small)

        # Bottom-up pathway
        x = self.layer17(feat16)
        x = torch.cat((x, feat13), dim=1)
        feat19 = self.layer19(x)  # P4' (medium)

        x = self.layer20(feat19)
        x = torch.cat((x, p5), dim=1)
        feat22 = self.layer22(x)  # P5' (large)

        return feat16, feat19, feat22
