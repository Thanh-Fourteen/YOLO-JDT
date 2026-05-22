"""Pre-LN multi-head cross-attention block — returns the attention DELTA only.

Q = F_t (current frame), K/V = F_prev (cached previous frame).
Uses F.scaled_dot_product_attention — no flash-attn dependency (SDPA auto-selects
FlashAttention-3 backend on Blackwell sm_120 if available).

IMPORTANT — this block returns ONLY the cross-attention contribution
(`proj_out(attention)`), NOT `F_t + attention`.  The residual connection and
its gate are owned by the enclosing `GatedResidual` (Flamingo GATED
XATTN-DENSE structure).  The previous design folded the query residual *and*
the FFN into this block, which — combined with the outer gated residual —
produced a `(1+α)·F_t` scaling that corrupted the pretrained detector at init
(root cause of the Step 5.DE v1/v2 regression).  Keeping this block delta-only
makes the enclosing layer a provable identity at init.
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
    """Pre-LN multi-head cross-attention. Returns the attention delta only.

    Q comes from F_t (current frame), K and V come from F_prev (cached t-1).
    Sinusoidal 2D pos encoding is added to Q and K (not V) before projection.

    Output = proj_out( MHA( norm_q(F_t)+pos , norm_kv(F_prev)+pos , norm_kv(F_prev) ) )
    i.e. the *temporal correction*, with NO query residual added here.

    Args:
        in_channels: channel count of the input feature maps (must be divisible
                     by num_heads). For YOLO11s P5 = 512, head_dim = 64 per spec.
        num_heads:   number of attention heads (default 8).

    Attention capture (for visualization only — not thread-safe):
        Set `CrossAttentionBlock.capture_attention = True` before the forward
        call; the attention weights [B, num_heads, L, L] will be stored in
        `self._last_attn_weights` (CPU tensor, detached).  When True, the
        forward uses a manual softmax path instead of SDPA so the weights are
        accessible.
    """

    # Class-level flag: enable before inference for visualization, disable after.
    capture_attention: bool = False

    def __init__(self, in_channels: int, num_heads: int = 8):
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
        # Zero-init the output projection (ControlNet/LoRA-style). The block
        # returns a pure delta; with proj_out=0 the delta is EXACTLY 0 at init
        # → TAGate is a provable identity (det AND reid == JDE bit-for-bit),
        # giving a hard HOTA≥JDE floor. Unlike a learnable scalar gate (grad ∝
        # tiny attn magnitude → structurally un-trainable, cf. v1–v8), proj_out
        # is a full [C,C] matrix: it receives O(C) healthy gradient terms from
        # the ReID loss and trains away from zero normally.
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, F_t: Tensor, F_prev: Tensor) -> Tensor:
        """
        Args:
            F_t:    [B, C, H, W]  current frame neck features
            F_prev: [B, C, H, W]  cached previous frame neck features
        Returns:
            [B, C, H, W]  attention DELTA (temporal correction, no residual)
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

        if self.capture_attention:
            # Manual softmax attention so weights are accessible for visualization.
            scale = self.head_dim ** -0.5
            raw = (q @ k.transpose(-2, -1)) * scale          # [B, nh, L, L]
            attn_w = raw.float().softmax(dim=-1).to(q.dtype)  # stable in fp32
            self._last_attn_weights = attn_w.detach().cpu()   # [B, nh, L, L]
            attn_out = attn_w @ v
        else:
            # SDPA — auto-selects FlashAttn/math backend; no explicit scale needed
            attn_out = F.scaled_dot_product_attention(q, k, v)   # [B, nh, L, hd]

        attn = attn_out.transpose(1, 2).reshape(B, L, C)
        attn = self.proj_out(attn)

        # Return the DELTA only — residual + gating handled by GatedResidual.
        return attn.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
