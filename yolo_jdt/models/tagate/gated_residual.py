"""Gated residual: F'_t = F_t + α · CrossAttn(F_t, F_prev).

α = sigmoid(gate), gate is a learnable scalar parameter.
Initialized at gate = -2  →  α = sigmoid(-2) ≈ 0.12.

This near-zero init means the model starts close to the Phase 4 JDE baseline
(no temporal modulation), then gradually learns how much temporal context
to inject as training proceeds.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from yolo_jdt.models.tagate.cross_attn import CrossAttentionBlock

__all__ = ["GatedResidual"]


class GatedResidual(nn.Module):
    """Single gated temporal attention layer.

    Output:  F'_t = F_t + sigmoid(gate) · CrossAttn(F_t, F_prev)

    Args:
        in_channels: spatial feature channel count.
        num_heads:   attention heads (default 8).
        ffn_ratio:   FFN hidden / in_channels ratio (default 2).
    """

    def __init__(self, in_channels: int, num_heads: int = 8, ffn_ratio: int = 2):
        super().__init__()
        self.attn = CrossAttentionBlock(in_channels, num_heads=num_heads,
                                        ffn_ratio=ffn_ratio)
        # Scalar gate — initialized to -2 so α ≈ 0.12 at the start of training
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, F_t: Tensor, F_prev: Tensor) -> Tensor:
        """
        Args:
            F_t:    [B, C, H, W]  current frame features
            F_prev: [B, C, H, W]  cached previous frame features (INPUT, not self.cache)
        Returns:
            [B, C, H, W]
        """
        return F_t + torch.sigmoid(self.gate) * self.attn(F_t, F_prev)
