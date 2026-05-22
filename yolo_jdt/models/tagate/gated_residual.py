"""Zero-init cross-attention residual layer for YOLO-JDT TAGate.

    x = F_t + CrossAttn(F_t, F_prev)        # single residual, no gate, no FFN

Design history (Step 5.DE, 8 failed variants → root-cause diagnostic):

The original Flamingo-style design used a learnable `tanh(gate)` scalar plus a
dense (FFN) sublayer.  Two structural problems were proven empirically:

  1. The scalar gate is *un-trainable* on a small fine-tune: its gradient is
     `⟨∂L/∂x, attn_delta⟩` which is ∝ the (tiny) attention-delta magnitude, so
     it received ≈1e-8 gradient and never moved in 25 epochs (v1–v8).
  2. The FFN sublayer is non-temporal (`ffn(LN(x))`, no F_prev) and undertrained
     on 2.6k MOT pairs → it injects noise into the ReID features.

A controlled diagnostic (JDE weights + identity TAGate, run through the full
JDT tracker/eval) reproduced the JDE baseline EXACTLY (HOTA 0.5600, IDs 453).
That proved the pipeline is correct and the *only* failure mode is the training
degrading JDE-quality features.  The fix:

  * Remove the scalar gate and the FFN entirely.
  * `CrossAttentionBlock.proj_out` is zero-initialised (ControlNet/LoRA trick),
    so the block returns an EXACT-zero delta at init → `x = F_t` →  TAGate is a
    provable identity (det AND reid == JDE bit-for-bit → hard HOTA≥JDE floor).
  * `proj_out` is a full [C,C] matrix, so unlike the scalar gate it gets O(C)
    healthy gradient terms and trains away from zero normally.
  * The rest of the JDE model (backbone+neck+JointHead incl. cv4) and the ReID
    classifier are frozen by the LightningModule, so the embedding space stays
    anchored at JDE quality and TAGate can only *add* temporal consistency.

The constructor still accepts `ffn_ratio` / `gate_init` for call-site
compatibility; they are unused (kept so existing configs/tests don't break).
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from yolo_jdt.models.tagate.cross_attn import CrossAttentionBlock

__all__ = ["GatedResidual"]


class GatedResidual(nn.Module):
    """Single zero-init cross-attention residual: `x = F_t + CrossAttn(F_t, F_prev)`.

    Args:
        in_channels: spatial feature channel count.
        num_heads:   attention heads (default 8).
        ffn_ratio:   accepted but unused (FFN removed — kept for call compat).
        gate_init:   accepted but unused (scalar gate removed — kept for compat).
    """

    def __init__(self, in_channels: int, num_heads: int = 8, ffn_ratio: int = 2,
                 gate_init: float = 0.0):
        super().__init__()
        # proj_out is zero-init inside CrossAttentionBlock → delta == 0 at init.
        self.attn = CrossAttentionBlock(in_channels, num_heads=num_heads)
        # Train-time-only hook kept for LightningModule compatibility (no-op).
        self.stage_a_alpha: float | None = None

    def forward(self, F_t: Tensor, F_prev: Tensor) -> Tensor:
        """
        Args:
            F_t:    [B, C, H, W]  current-frame features
            F_prev: [B, C, H, W]  cached previous-frame features (INPUT tensor)
        Returns:
            [B, C, H, W]  F_t + zero-init cross-attention delta
        """
        return F_t + self.attn(F_t, F_prev)
