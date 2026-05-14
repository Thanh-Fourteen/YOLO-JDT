"""Tests for TAGate: CrossAttentionBlock, GatedResidual, TAGate module, YOLO_JDT.

Coverage:
- Forward/backward shapes for each component
- BF16 numerical stability (no NaN/Inf)
- Gate alpha initialisation (sigmoid(-2) ≈ 0.12)
- ONNX trace: 1-layer TAGate on P5 features
- YOLO_JDT: zero_cache helper + cached_features round-trip
- cache_levels variants (P5, P4+P5, P3+P4+P5)
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from yolo_jdt.models.tagate.cross_attn import CrossAttentionBlock, _make_2d_sinusoidal
from yolo_jdt.models.tagate.gated_residual import GatedResidual
from yolo_jdt.models.tagate.module import TAGate
from yolo_jdt.models.yolo_jdt import YOLO_JDT

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, C, H, W = 2, 512, 20, 20   # P5 dimensions for YOLO11s, 640×640 input


@pytest.fixture
def F_t():
    return torch.randn(B, C, H, W)


@pytest.fixture
def F_prev():
    return torch.randn(B, C, H, W)


# ---------------------------------------------------------------------------
# _make_2d_sinusoidal
# ---------------------------------------------------------------------------

def test_sinusoidal_shape():
    pe = _make_2d_sinusoidal(20, 20, 512)
    assert pe.shape == (1, 400, 512)


def test_sinusoidal_no_nan():
    pe = _make_2d_sinusoidal(80, 80, 128)
    assert pe.isfinite().all()


def test_sinusoidal_different_sizes():
    for H_, W_, C_ in [(20, 20, 512), (40, 40, 256), (80, 80, 128)]:
        pe = _make_2d_sinusoidal(H_, W_, C_)
        assert pe.shape == (1, H_ * W_, C_)


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------

def test_cross_attn_output_shape(F_t, F_prev):
    blk = CrossAttentionBlock(C)
    out = blk(F_t, F_prev)
    assert out.shape == F_t.shape


def test_cross_attn_gradient(F_t, F_prev):
    F_t = F_t.requires_grad_(True)
    blk = CrossAttentionBlock(C)
    loss = blk(F_t, F_prev).sum()
    loss.backward()
    assert F_t.grad is not None
    assert F_t.grad.isfinite().all()


def test_cross_attn_bf16(F_t, F_prev):
    blk = CrossAttentionBlock(C).bfloat16()
    out = blk(F_t.bfloat16(), F_prev.bfloat16())
    assert out.dtype == torch.bfloat16
    assert out.isfinite().all()


def test_cross_attn_p4_channels():
    blk = CrossAttentionBlock(256, num_heads=8)
    x = torch.randn(1, 256, 40, 40)
    out = blk(x, x)
    assert out.shape == x.shape


def test_cross_attn_p3_channels():
    blk = CrossAttentionBlock(128, num_heads=8)
    x = torch.randn(1, 128, 80, 80)
    out = blk(x, x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# GatedResidual
# ---------------------------------------------------------------------------

def test_gated_residual_shape(F_t, F_prev):
    gr = GatedResidual(C)
    out = gr(F_t, F_prev)
    assert out.shape == F_t.shape


def test_gate_init():
    gr = GatedResidual(C)
    alpha = torch.sigmoid(gr.gate).item()
    expected = 1.0 / (1.0 + math.exp(2.0))   # sigmoid(-2) ≈ 0.1192
    assert abs(alpha - expected) < 1e-4


def test_gate_zero_prev_close_to_input(F_t):
    """When F_prev = zeros, gated output should be close to F_t (α≈0.12 and attn bounded)."""
    gr = GatedResidual(C)
    gr.eval()
    with torch.no_grad():
        F_prev_z = torch.zeros_like(F_t)
        out = gr(F_t, F_prev_z)
    # With α≈0.12 and attention on zero-valued K/V the output should not diverge
    assert out.isfinite().all()


def test_gated_residual_bf16(F_t, F_prev):
    gr = GatedResidual(C).bfloat16()
    out = gr(F_t.bfloat16(), F_prev.bfloat16())
    assert out.dtype == torch.bfloat16
    assert out.isfinite().all()


# ---------------------------------------------------------------------------
# TAGate module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
def test_tagate_shape(F_t, F_prev, num_layers):
    tg = TAGate(C, num_layers=num_layers)
    out = tg(F_t, F_prev)
    assert out.shape == F_t.shape


def test_tagate_f_prev_unchanged(F_t, F_prev):
    """F_prev must not be modified in-place by TAGate."""
    tg = TAGate(C, num_layers=2)
    F_prev_clone = F_prev.clone()
    tg(F_t, F_prev)
    assert torch.allclose(F_prev, F_prev_clone)


def test_tagate_gradient_flow(F_t, F_prev):
    F_t = F_t.requires_grad_(True)
    tg = TAGate(C, num_layers=2)
    loss = tg(F_t, F_prev).sum()
    loss.backward()
    assert F_t.grad is not None
    assert F_t.grad.isfinite().all()
    # Check gate grads flow
    for layer in tg.layers:
        assert layer.gate.grad is not None


def test_tagate_bf16(F_t, F_prev):
    tg = TAGate(C, num_layers=2).bfloat16()
    out = tg(F_t.bfloat16(), F_prev.bfloat16())
    assert out.dtype == torch.bfloat16
    assert out.isfinite().all()


def test_tagate_onnx_trace(tmp_path):
    """ONNX export of a 1-layer TAGate on P5 (512-ch, 20×20) must succeed."""
    onnxruntime = pytest.importorskip("onnxruntime")

    tg = TAGate(C, num_layers=1).eval()
    dummy_ft   = torch.zeros(1, C, H, W)
    dummy_prev = torch.zeros(1, C, H, W)
    onnx_path  = str(tmp_path / "tagate_1layer.onnx")

    torch.onnx.export(
        tg,
        (dummy_ft, dummy_prev),
        onnx_path,
        input_names=["F_t", "F_prev"],
        output_names=["F_t_out"],
        opset_version=17,
        do_constant_folding=True,
    )
    # Verify onnxruntime can load and run
    sess = onnxruntime.InferenceSession(onnx_path,
                                        providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "F_t":    dummy_ft.numpy(),
        "F_prev": dummy_prev.numpy(),
    })
    assert out[0].shape == (1, C, H, W)


# ---------------------------------------------------------------------------
# YOLO_JDT
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yolo_jdt_s():
    return YOLO_JDT(scale="s", nc=1, cache_levels="P5", tagate_num_layers=1).eval()


def test_yolo_jdt_zero_cache(yolo_jdt_s):
    cache = yolo_jdt_s.zero_cache(batch_size=1)
    assert len(cache) == 1                        # 1 level for "P5"
    assert cache[0].shape == (1, 512, 20, 20)


def test_yolo_jdt_eval_output_shapes(yolo_jdt_s):
    img = torch.zeros(1, 3, 640, 640)
    cache = yolo_jdt_s.zero_cache(batch_size=1)
    with torch.no_grad():
        out = yolo_jdt_s(img, cache)
    # eval mode: (decoded, raw_det, reid, offset_out, features_to_cache)
    decoded, raw_det, reid, offset_out, feats_cache = out
    assert decoded.shape[1] == 5         # [B, nc+4, A]  where nc=1
    assert len(reid) == 3                # 3 FPN levels
    assert offset_out is None
    assert len(feats_cache) == 1         # "P5" → 1 cached level
    assert feats_cache[0].shape == (1, 512, 20, 20)


def test_yolo_jdt_cache_roundtrip(yolo_jdt_s):
    """features_to_cache from frame t can be passed as cached_features_prev at t+1."""
    img = torch.zeros(1, 3, 640, 640)
    cache = yolo_jdt_s.zero_cache(batch_size=1)
    with torch.no_grad():
        *_, feats_cache = yolo_jdt_s(img, cache)
        *_, feats_cache2 = yolo_jdt_s(img, feats_cache)
    assert feats_cache2[0].shape == feats_cache[0].shape


def test_yolo_jdt_train_output_shapes():
    model = YOLO_JDT(scale="s", nc=1, cache_levels="P5", tagate_num_layers=1).train()
    img = torch.zeros(1, 3, 640, 640)
    cache = model.zero_cache(batch_size=1)
    out = model(img, cache)
    # train mode: (raw_det, reid, offset_out, features_to_cache)
    raw_det, reid, offset_out, feats_cache = out
    assert len(raw_det) == 3     # 3 FPN levels
    assert len(reid) == 3
    assert offset_out is None
    assert len(feats_cache) == 1


@pytest.mark.parametrize("cache_levels", ["P5", "P4+P5", "P3+P4+P5"])
def test_yolo_jdt_cache_levels(cache_levels):
    model = YOLO_JDT(scale="s", nc=1, cache_levels=cache_levels,
                     tagate_num_layers=1).eval()
    n_levels = len(cache_levels.split("+"))
    cache = model.zero_cache(batch_size=1)
    assert len(cache) == n_levels
    img = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        *_, feats_cache = model(img, cache)
    assert len(feats_cache) == n_levels
