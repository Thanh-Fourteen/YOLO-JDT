"""YOLO11 joint detection + ReID head.

Extends `DecoupledDetect` (which has `cv2` for box-DFL and `cv3` for cls)
with a third branch `cv4` per FPN level that produces an L2-normalized
128-d ReID embedding per anchor. Architecture per level (per user spec):

    Conv3x3(c_in → 256) → BN → SiLU → Conv3x3(256 → 128) → L2-normalize

Forward semantics:
    Train mode: returns (raw_det_per_level, reid_per_level) — 2-tuple.
        raw_det_per_level: list of [B, 4*reg_max+nc, H, W] (same as parent)
        reid_per_level:   list of [B, 128, H, W], L2-normalized along ch dim
    Eval mode:  returns (decoded[B, 4+nc, A], raw_det_per_level, reid_per_level)
        — backward-compatible with callers that do `decoded, _ = model(x)`
        because Python tuple unpacking with `_` collapses the trailing items.

Weight loading: cv4 has no Ultralytics counterpart. Use the partial-load
fallback in `DetLitModule._load_pretrained_filter_nc` (or its successor
in this Phase 4 work) to load box+cls from pretrained while leaving cv4
freshly initialized.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.ultralytics_extract.conv import Conv
from yolo_jdt.models.head.decoupled_detect import DecoupledDetect

__all__ = ["JointHead"]


class JointHead(DecoupledDetect):
    """DecoupledDetect + per-level ReID branch (cv4).

    Args:
        nc: detection class count.
        ch: per-level input channel tuple (P3', P4', P5').
        reg_max: DFL distribution length.
        strides: per-level strides matching backbone (e.g. (8, 16, 32)).
        reid_dim: ReID embedding dimensionality (default 128).
        reid_hidden: width of the cv4 hidden Conv (default 256).
    """

    def __init__(self, nc: int = 1, ch: tuple = (), reg_max: int = 16,
                 strides: tuple[float, ...] = (8.0, 16.0, 32.0),
                 reid_dim: int = 128, reid_hidden: int = 256):
        super().__init__(nc=nc, ch=ch, reg_max=reg_max, strides=strides)
        self.reid_dim = reid_dim
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(c, reid_hidden, 3),                       # Conv3x3 + BN + SiLU
                nn.Conv2d(reid_hidden, reid_dim, 3, padding=1, bias=True),
            )
            for c in ch
        )

    def forward(self, x: list[torch.Tensor]):
        # Compute ReID branch BEFORE the parent's cv2/cv3 mutate x[i] in place.
        reid = []
        for i in range(self.nl):
            emb = self.cv4[i](x[i])                            # [B, reid_dim, H, W]
            emb = F.normalize(emb, dim=1, p=2, eps=1e-6)       # unit-norm along channel
            reid.append(emb)

        # Detection branches (parent's exact computation, kept inline so we can
        # tack `reid` onto the eval-mode tuple cleanly).
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x, reid
        decoded, raw = self._inference(x)
        return decoded, raw, reid
