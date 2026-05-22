"""YOLO-JDT full model assembly: Backbone + Neck + TAGate + JointHead + TrackOffsetHead.

This is the core contribution model. It extends _YOLO11WithJointHead from Phase 4
with a TAGate cross-attention temporal branch.

Step 5.DE pivot (2026-05-22): TAGate no longer feeds the ReID branch. Research
(Kumar ICLR'22 "fine-tuning distorts pretrained features" + the DanceTrack
oracle study) showed that perturbing a frozen embedding head's input corrupts
JDE-quality embeddings — every v1–v9 run degraded HOTA below baseline. The fix:

  * Detection (cv2/cv3) AND ReID (cv4) read the ORIGINAL neck features — both
    branches are bit-exact frozen JDE, no temporal perturbation possible.
  * TAGate-enhanced features feed a NEW TrackOffsetHead that predicts per-anchor
    inter-frame motion (offset to t-1). This is a purely-additive task — there is
    no pretrained embedding to corrupt. The associator consumes the offset as a
    learned motion cue.

Forward signature (ONNX-friendly — no Python state):

    Train mode:
        raw_det_per_level, reid_per_level, offset_out, features_to_cache
        = model(image_t, cached_features_prev)

    Eval mode:
        decoded, raw_det_per_level, reid_per_level, offset_out, features_to_cache
        = model(image_t, cached_features_prev)

Where:
    cached_features_prev : list[Tensor]   — one Tensor per cached FPN level,
                           in ascending order (P3, P4, P5 subset per cache_levels).
                           Pass zeros tensors at t=0 (use zero_cache() helper).
    features_to_cache    : list[Tensor]   — neck outputs BEFORE TAGate; pass back
                           as cached_features_prev on the next call.
    offset_out           : list[Tensor]   — per-level [B, 2, H, W] (Δx, Δy) raw
                           offsets in input-canvas-normalised coords (3 levels).

cache_levels controls which FPN levels receive TAGate processing:
    "P5"         — only P5' (default; cheapest)
    "P4+P5"      — P4' and P5'
    "P3+P4+P5"   — all three levels

TAGate strides + default spatial sizes for 640×640 input:
    P3'  stride  8  → 80×80    (in_channels=128 for YOLO11s)
    P4'  stride 16  → 40×40    (in_channels=256 for YOLO11s)
    P5'  stride 32  → 20×20    (in_channels=512 for YOLO11s)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from yolo_jdt.models.backbone.yolo11 import YOLO11Backbone
from yolo_jdt.models.head.joint_head import JointHead
from yolo_jdt.models.head.offset_head import TrackOffsetHead
from yolo_jdt.models.neck.panet import YOLO11PANet
from yolo_jdt.models.tagate.module import TAGate

__all__ = ["YOLO_JDT"]

_CACHE_LEVEL_MAP: dict[str, list[int]] = {
    "P5":       [2],
    "P4+P5":    [1, 2],
    "P3+P4+P5": [0, 1, 2],
}
# Strides per FPN level index [P3', P4', P5']
_FPN_STRIDES = [8, 16, 32]


class YOLO_JDT(nn.Module):
    """YOLO11 + TAGate + JointHead (detection + ReID, no offset in Phase 5).

    Args:
        scale:            YOLO11 scale letter ('n'/'s'/'m'/'l'/'x').
        nc:               detection class count (1 = person-only).
        reg_max:          DFL distribution bins (default 16).
        strides:          per-level anchor strides matching backbone.
        reid_dim:         ReID embedding dim (default 128).
        reid_hidden:      cv4 hidden width (default 256).
        cache_levels:     which FPN levels to cache + gate ('P5', 'P4+P5', 'P3+P4+P5').
        tagate_num_layers: N gated cross-attention layers per cached level (1–4).
        tagate_num_heads: attention heads per layer (default 8).
        tagate_ffn_ratio: FFN width multiplier (default 2).
        tagate_gate_init: accepted for config/ckpt compatibility; unused in the
                          zero-init proj_out design.
        offset_hidden:    hidden Conv width of the TrackOffsetHead (default 256).
        img_size:         spatial input resolution used to pre-size default spatial dims
                          in CrossAttentionBlock's pos-enc buffer (default 640).
    """

    def __init__(
        self,
        scale: str = "s",
        nc: int = 1,
        reg_max: int = 16,
        strides: tuple[float, ...] = (8.0, 16.0, 32.0),
        reid_dim: int = 128,
        reid_hidden: int = 256,
        cache_levels: str = "P5",
        tagate_num_layers: int = 2,
        tagate_num_heads: int = 8,
        tagate_ffn_ratio: int = 2,
        tagate_gate_init: float = 0.0,
        offset_hidden: int = 256,
        img_size: int = 640,
    ):
        super().__init__()
        if cache_levels not in _CACHE_LEVEL_MAP:
            raise ValueError(
                f"cache_levels must be one of {list(_CACHE_LEVEL_MAP)}, got {cache_levels!r}"
            )

        self.cache_levels = cache_levels
        self._level_ids: list[int] = _CACHE_LEVEL_MAP[cache_levels]

        # ---- Backbone + Neck (same as _YOLO11WithJointHead) ---------------
        self.backbone = YOLO11Backbone(scale)
        self.neck = YOLO11PANet(scale, in_channels=self.backbone.out_channels)
        self.head = JointHead(
            nc=nc, ch=self.neck.out_channels,
            reg_max=reg_max, strides=strides,
            reid_dim=reid_dim, reid_hidden=reid_hidden,
        )

        # ---- TAGate: one module per cached FPN level ----------------------
        neck_ch = self.neck.out_channels   # (P3_ch, P4_ch, P5_ch)
        self.tagates = nn.ModuleList([
            TAGate(
                in_channels=neck_ch[i],
                num_layers=tagate_num_layers,
                num_heads=tagate_num_heads,
                ffn_ratio=tagate_ffn_ratio,
                gate_init=tagate_gate_init,
            )
            for i in self._level_ids
        ])

        # ---- TrackOffsetHead: per-anchor inter-frame motion (3 FPN levels) ----
        # Reads TAGate-enhanced features; purely additive — never touches the
        # frozen JDE detection/ReID branches. See offset_head.py / Step 5.DE pivot.
        self.offset_head = TrackOffsetHead(ch=neck_ch, hidden=offset_hidden)

        # Convenience: store channel counts for the cached levels
        self._cache_channels: list[int] = [neck_ch[i] for i in self._level_ids]
        self._img_size = img_size

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def zero_cache(
        self, batch_size: int = 1, img_h: int | None = None, img_w: int | None = None,
        device=None, dtype=torch.float32,
    ) -> list[Tensor]:
        """Return zero-filled cached_features_prev for t=0 initialization.

        The spatial dims are derived from (img_h, img_w) (default: self._img_size).
        """
        H = img_h or self._img_size
        W = img_w or self._img_size
        return [
            torch.zeros(batch_size, self._cache_channels[j],
                        H // _FPN_STRIDES[i], W // _FPN_STRIDES[i],
                        device=device, dtype=dtype)
            for j, i in enumerate(self._level_ids)
        ]

    @property
    def num_cached_levels(self) -> int:
        return len(self._level_ids)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image_t: Tensor,
        cached_features_prev: list[Tensor],
    ) -> tuple:
        """
        Args:
            image_t:              [B, 3, H, W]
            cached_features_prev: list[Tensor], one per cached level.
                                  Use zero_cache() for t=0.
        Returns (train mode):
            (raw_det_per_level, reid_per_level, offset_out, features_to_cache)

        Returns (eval mode):
            (decoded, raw_det_per_level, reid_per_level, offset_out, features_to_cache)

        offset_out is a list of per-level [B, 2, H, W] (Δx, Δy) raw offsets.

        features_to_cache is a list[Tensor] of neck outputs BEFORE TAGate — pass
        back as cached_features_prev on the next frame call.
        """
        # --- Backbone + Neck ---
        p3, p4, p5 = self.backbone(image_t)
        neck_feats: list[Tensor] = list(self.neck(p3, p4, p5))  # [P3', P4', P5']

        # --- Cache = raw neck outputs (before TAGate) ---
        features_to_cache: list[Tensor] = [neck_feats[i] for i in self._level_ids]

        # --- Apply TAGate → temporally-enhanced features for the offset head ---
        # tagate_feats is a shallow copy of neck_feats with only the cached
        # levels replaced by TAGate cross-attention output. It feeds ONLY the
        # TrackOffsetHead. neck_feats itself is untouched.
        tagate_feats = list(neck_feats)
        for j, i in enumerate(self._level_ids):
            tagate_feats[i] = self.tagates[j](
                neck_feats[i],              # F_t  (current frame)
                cached_features_prev[j],    # F_prev (caller-supplied cache)
            )

        # --- Joint Head: detection (cv2/cv3) AND ReID (cv4) read the ORIGINAL
        # neck features → both branches are bit-exact frozen JDE. TAGate cannot
        # perturb them. (Step 5.DE pivot — see module docstring.)
        head_out = self.head(neck_feats)

        # --- TrackOffsetHead reads the TAGate-enhanced features ---
        offset_out = self.offset_head(tagate_feats)

        if self.training:
            raw_det, reid = head_out
            return raw_det, reid, offset_out, features_to_cache
        decoded, raw_det, reid = head_out
        return decoded, raw_det, reid, offset_out, features_to_cache
