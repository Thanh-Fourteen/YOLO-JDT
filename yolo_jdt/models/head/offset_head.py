"""Track-offset head — per-anchor inter-frame motion regression for YOLO-JDT.

Step 5.DE pivot (2026-05-22). Research (Kumar ICLR'22 + DanceTrack oracle study)
showed that recycling temporal features into the ReID branch is structurally
unsound: it perturbs the input distribution of a frozen embedding head and
corrupts JDE-quality embeddings (v1–v9 all degraded HOTA below baseline). The
fix re-targets TAGate to feed a *new* task instead of ReID.

`TrackOffsetHead` predicts, for each anchor at frame t, the displacement of the
object centre back to frame t-1:

    offset = centre_{t-1} − centre_t          (input-canvas-normalised [0,1] coords)

It consumes TAGate-enhanced features — the cross-frame correspondence TAGate
computes against the cached t-1 neck features is read out here as an explicit
motion vector (cf. CenterTrack offset head, MATR query-motion prediction).
Detection (cv2/cv3) and ReID (cv4) never see these features, so this head is a
*purely-additive new task*: it cannot perturb the frozen JDE detector or its
embedding space. The associator consumes the offset as a learned motion cue
(competing with / complementing the Kalman constant-velocity prior).

Per level:  Conv3x3(c → hidden) → BN → SiLU → Conv1x1(hidden → 2).
The final 1×1 conv is zero-initialised so the head outputs offset 0 at init;
paired with TAGate's zero-init `proj_out`, the whole temporal branch starts as
an exact no-op and training can only *add* signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from third_party.ultralytics_extract.conv import Conv

__all__ = ["TrackOffsetHead"]


class TrackOffsetHead(nn.Module):
    """Per-FPN-level (Δx, Δy) inter-frame motion head.

    Args:
        ch:     per-level input channel tuple (P3', P4', P5').
        hidden: hidden Conv width (default 256).
    """

    def __init__(self, ch: tuple = (), hidden: int = 256):
        super().__init__()
        self.nl = len(ch)
        self.cv_off = nn.ModuleList(
            nn.Sequential(
                Conv(c, hidden, 3),                       # Conv3x3 + BN + SiLU
                nn.Conv2d(hidden, 2, 1, bias=True),       # Conv1x1 → (Δx, Δy)
            )
            for c in ch
        )
        # Zero-init the final conv → offset == 0 at init (no-motion prior).
        for seq in self.cv_off:
            nn.init.zeros_(seq[-1].weight)
            nn.init.zeros_(seq[-1].bias)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Args:
            x: per-level features (TAGate-enhanced), one Tensor [B, C, H, W] per level.
        Returns:
            list of [B, 2, H, W] — raw (Δx, Δy) offsets per anchor, per level.
        """
        return [self.cv_off[i](x[i]) for i in range(self.nl)]
