"""Pre-LN multi-head cross-attention block with 2D sinusoidal positional encoding.

Q = F_t (current frame), K/V = F_prev (cached previous frame).
Uses F.scaled_dot_product_attention — no flash-attn dependency (SDPA auto-selects
FlashAttention-3 backend on Blackwell sm_120 if available).

Architecture per block:
    1. Pre-LN cross-attention: F_t' = F_t + CrossAttn(norm_q(F_t), norm_kv(F_prev))
       - 2D sinusoidal pos encoding added to Q and K (not V)
    2. Pre-LN FFN:             F_t'' = F_t' + FFN(norm_ffn(F_t'))
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["CrossAttentionBlock"]


def _make_2d_sinusoidal(H: int, W: int, dim: int) -> Tensor:
    """Sinusoidal 2D pos encoding: [1, H*W, dim].

    First dim//2 channels encode the Y (row) position; last dim//2 encode X (col).
    Each half uses the standard 1D sin/cos schedule with base 10000.
    """
    assert dim % 2 == 0, f"dim must be even for 2D sinusoidal enc, got {dim}"
    d = dim // 2  # channels per spatial axis

    pos_y = torch.arange(H, dtype=torch.float32).unsqueeze(1)   # [H, 1]
    pos_x = torch.arange(W, dtype=torch.float32).unsqueeze(1)   # [W, 1]
    # Frequencies: exp(-2k * log(10000) / d) for k=0,1,...,d//2-1
    freq = torch.exp(
        torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d)
    )  # [d//2]

    pe_y = torch.zeros(H, d)
    pe_y[:, 0::2] = torch.sin(pos_y * freq)          # [H, d//2]
    pe_y[:, 1::2] = torch.cos(pos_y * freq)           # [H, d//2]

    pe_x = torch.zeros(W, d)
    pe_x[:, 0::2] = torch.sin(pos_x * freq)           # [W, d//2]
    pe_x[:, 1::2] = torch.cos(pos_x * freq)           # [W, d//2]

    # Broadcast to grid [H, W, dim]
    pe = torch.cat([
        pe_y.unsqueeze(1).expand(H, W, d),   # Y enc tiled over W
        pe_x.unsqueeze(0).expand(H, W, d),   # X enc tiled over H
    ], dim=-1)  # [H, W, dim]

    return pe.reshape(1, H * W, dim)  # [1, H*W, dim]


class CrossAttentionBlock(nn.Module):
    """Pre-LN multi-head cross-attention + FFN block.

    Q comes from F_t (current frame), K and V come from F_prev (cached t-1).
    Sinusoidal 2D pos encoding is added to Q and K (not V) before projection.

    Args:
        in_channels: channel count of the input feature maps (must be divisible
                     by num_heads). For YOLO11s P5 = 512, head_dim = 64 per spec.
        num_heads:   number of attention heads (default 8).
        ffn_ratio:   FFN hidden dim = in_channels * ffn_ratio (default 2).
    """

    def __init__(self, in_channels: int, num_heads: int = 8, ffn_ratio: int = 2):
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # Pre-LN: separate norms for Q source (F_t) and KV source (F_prev)
        self.norm_q = nn.LayerNorm(in_channels)
        self.norm_kv = nn.LayerNorm(in_channels)

        # Q, K, V projections (no bias — common in ViT-style cross-attn)
        self.proj_q = nn.Linear(in_channels, in_channels, bias=False)
        self.proj_k = nn.Linear(in_channels, in_channels, bias=False)
        self.proj_v = nn.Linear(in_channels, in_channels, bias=False)
        self.proj_out = nn.Linear(in_channels, in_channels, bias=False)

        # FFN: 2-layer MLP with GeLU, Pre-LN
        ffn_dim = in_channels * ffn_ratio
        self.norm_ffn = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, in_channels),
        )

    def forward(self, F_t: Tensor, F_prev: Tensor) -> Tensor:
        """
        Args:
            F_t:    [B, C, H, W]  current frame neck features
            F_prev: [B, C, H, W]  cached previous frame neck features
        Returns:
            [B, C, H, W]  temporally-enhanced features for current frame
        """
        B, C, H, W = F_t.shape
        L = H * W

        # [B, C, H, W] → [B, L, C] sequence of spatial tokens
        Ft = F_t.permute(0, 2, 3, 1).reshape(B, L, C)
        Fp = F_prev.permute(0, 2, 3, 1).reshape(B, L, C)

        # 2D sinusoidal pos encoding [1, L, C] — same device/dtype as input
        pos = _make_2d_sinusoidal(H, W, C).to(Ft)   # .to(tensor) copies device+dtype

        # Pre-LN + add positional encoding to Q and K (not V)
        q = self.proj_q(self.norm_q(Ft) + pos)    # [B, L, C]
        k = self.proj_k(self.norm_kv(Fp) + pos)   # [B, L, C]
        v = self.proj_v(self.norm_kv(Fp))          # [B, L, C]  — no pos enc on V

        # Reshape to [B, num_heads, L, head_dim] for SDPA
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA — auto-selects FlashAttn/math backend; no explicit scale needed
        attn = F.scaled_dot_product_attention(q, k, v)   # [B, nh, L, hd]
        attn = attn.transpose(1, 2).reshape(B, L, C)
        attn = self.proj_out(attn)

        Ft = Ft + attn  # cross-attn residual

        # Pre-LN FFN
        Ft = Ft + self.ffn(self.norm_ffn(Ft))

        return Ft.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
