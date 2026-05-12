# Vendored from Ultralytics 8.4.48 (https://github.com/ultralytics/ultralytics)
# Source: ultralytics/nn/modules/conv.py
# Licensed under AGPL-3.0. Original copyright (c) Ultralytics.
# Modifications:
#   - Removed runtime dependency on ultralytics.utils.* (no imports outside torch / stdlib).
#   - Kept only the building blocks used by YOLO11 detection (Conv, DWConv, Concat, autopad).
#   - Class/attribute names preserved verbatim — state_dict mapping in `weights/loader.py`
#     depends on identical naming.
"""Standalone copies of YOLO11 convolution blocks."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["autopad", "Conv", "DWConv", "Concat"]


def autopad(k, p=None, d=1):
    """'same' padding for given kernel size, dilation."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv + BN + SiLU. Default activation is SiLU."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # Ultralytics overrides PyTorch BN defaults via initialize_weights() in
        # ultralytics/utils/torch_utils.py — eps=1e-3, momentum=0.03. Applied here
        # at construction so eval-mode parity holds without an extra init pass.
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depth-wise convolution (groups = gcd(c1, c2))."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Concat(nn.Module):
    """Concatenate a list of tensors along a given channel dimension."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        return torch.cat(x, self.d)
