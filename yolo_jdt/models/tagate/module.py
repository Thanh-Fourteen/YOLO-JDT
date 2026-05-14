"""TAGate: N-layer Temporal Attention Gate.

Stacks N GatedResidual layers to progressively refine current-frame features
with temporal context from the cached previous frame.

CRITICAL — ONNX/production constraint:
    forward(F_t, F_prev) -> Tensor
    F_prev is an INPUT tensor passed by the caller — it is NOT stored in self.cache.
    This keeps the module stateless and fully exportable to ONNX/TensorRT.
    The caller is responsible for maintaining the feature cache between frames.

Layer stacking semantics:
    Each layer takes the *updated* F_t from the previous layer but always uses
    the *original* F_prev (neck output from t-1) as the K/V source.
    This prevents temporal feature drift and keeps the t-1 reference clean.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from yolo_jdt.models.tagate.gated_residual import GatedResidual

__all__ = ["TAGate"]


class TAGate(nn.Module):
    """Temporal Attention Gate — N stacked GatedResidual layers.

    Args:
        in_channels: feature map channels (P5=512/P4=256/P3=128 for YOLO11s).
        num_layers:  number of gated cross-attention layers (1–4, default 2).
        num_heads:   attention heads per layer (default 8).
        ffn_ratio:   FFN width multiplier (default 2; keeps params ≈ 1–2M per layer).
    """

    def __init__(self, in_channels: int, num_layers: int = 2,
                 num_heads: int = 8, ffn_ratio: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedResidual(in_channels, num_heads=num_heads, ffn_ratio=ffn_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, F_t: Tensor, F_prev: Tensor) -> Tensor:
        """Refine F_t with temporal context from F_prev.

        Args:
            F_t:    [B, C, H, W]  current-frame neck features (will be refined)
            F_prev: [B, C, H, W]  cached neck features from t-1 (K/V reference,
                                  stays fixed across all N layers)
        Returns:
            [B, C, H, W]  temporally-enhanced features
        """
        for layer in self.layers:
            # F_prev is the SAME cached t-1 tensor for every layer.
            # F_t accumulates refinements across layers.
            F_t = layer(F_t, F_prev)
        return F_t
