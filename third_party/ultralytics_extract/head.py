# Vendored from Ultralytics 8.4.48 (https://github.com/ultralytics/ultralytics)
# Source: ultralytics/nn/modules/head.py
# Licensed under AGPL-3.0. Original copyright (c) Ultralytics.
# Modifications:
#   - Removed runtime dependency on ultralytics.utils.* — inlined make_anchors,
#     dist2bbox, bias_init_with_prob (small helpers).
#   - Stripped end2end / one2one branch (YOLO11 detection does not use it; YOLO26 will
#     get its own head module in third_party/yolo26_extract/).
#   - Stripped postprocess() / get_topk_index() / fuse() — we run NMS / postproc in
#     our own pipeline (yolo_jdt/ utilities).
#   - Kept attribute names (cv2, cv3, dfl, no, nc, reg_max, nl, stride, anchors,
#     strides, shape, dynamic, export) verbatim for state_dict compatibility.
"""Standalone YOLO11 Detect head."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from third_party.ultralytics_extract.block import DFL
from third_party.ultralytics_extract.conv import Conv, DWConv

__all__ = ["Detect", "make_anchors", "dist2bbox", "bias_init_with_prob"]


def make_anchors(feats, strides, grid_cell_offset: float = 0.5):
    """Generate (anchor_points, stride_tensor) for each feature level.

    Args:
        feats: list of feature tensors (BCHW), one per level.
        strides: per-level stride tensor.
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh: bool = True, dim: int = -1):
    """Decode (l, t, r, b) distance encoding into a bbox."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Logit init (inverse sigmoid) for classification bias."""
    return float(-math.log((1 - prior_prob) / prior_prob))


class Detect(nn.Module):
    """YOLO11 decoupled detection head. Outputs raw box-regression (DFL) + class logits per
    feature level. Inference path decodes via DFL + dist2bbox; training returns raw preds.

    Note: this implementation drops Ultralytics' optional end2end/one2one branches and
    its post-processing. Those will be re-added in `yolo_jdt/models/head/joint_head.py`
    as part of the JDT extension; this is the minimal vendored detection-only head.
    """

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False  # YOLO11 m/l/x set this False (DWConv variant of cv3)

    def __init__(self, nc: int = 80, reg_max: int = 16, ch: tuple = ()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: list[torch.Tensor]):
        """If training: return list of per-level concatenated raw outputs (box_dfl + cls_logits).
        If inference: decode boxes via DFL + dist2bbox and return (decoded, raw_per_level).
        """
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        return self._inference(x)

    def _inference(self, x: list[torch.Tensor]):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect biases. Requires self.stride populated."""
        for i, (a, b) in enumerate(zip(self.cv2, self.cv3)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
