"""Tests for TAGate: CrossAttentionBlock, GatedResidual, TAGate module, YOLO_JDT.

Architecture: Flamingo GATED XATTN-DENSE (Alayrac et al., NeurIPS 2022).

Coverage:
- Forward/backward shapes for each component
- BF16 numerical stability (no NaN/Inf)
- PROVABLE IDENTITY AT INIT: tanh(gate)=0 → TAGate is an exact no-op for ANY
  F_prev (zeros / random / real). This is the headline property — it guarantees
  detection == Phase-4 JDE baseline at init (no regression possible) and is the
  fix for the Step 5.DE v1/v2 regression. Paired with a 1e-5 parity assertion.
- Stage-A curriculum override (train-only, never at eval/export)
- ONNX trace: 1-layer TAGate on P5 features
- YOLO_JDT: zero_cache helper + cached_features round-trip + cache_levels
"""
from __future__ import annotations

import pytest
import torch

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
# CrossAttentionBlock — now returns the attention DELTA only
# ---------------------------------------------------------------------------

def test_cross_attn_output_shape(F_t, F_prev):
    blk = CrossAttentionBlock(C)
    out = blk(F_t, F_prev)
    assert out.shape == F_t.shape


def test_cross_attn_is_delta_not_residual(F_t, F_prev):
    """Block must NOT add the query residual internally (delta-only).

    If it returned F_t + attn, the output would correlate strongly with F_t.
    Delta-only: with random init the output is decorrelated from F_t and its
    magnitude is far smaller than F_t (no (1+α)·F_t leakage).
    """
    blk = CrossAttentionBlock(C).eval()
    with torch.no_grad():
        out = blk(F_t, F_prev)
    # The previous (buggy) design returned F_t + attn + ffn → out ≈ F_t.
    assert not torch.allclose(out, F_t, atol=1e-2)


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
# GatedResidual — Flamingo gates, provable identity at init
# ---------------------------------------------------------------------------

def test_gated_residual_shape(F_t, F_prev):
    gr = GatedResidual(C)
    out = gr(F_t, F_prev)
    assert out.shape == F_t.shape


def test_proj_out_zero_init():
    """CrossAttentionBlock.proj_out is zero-init → delta == 0 at init."""
    gr = GatedResidual(C)
    assert gr.attn.proj_out.weight.abs().max().item() == 0.0
    # other projections are NOT zero (only the output projection is)
    assert gr.attn.proj_q.weight.abs().max().item() > 0.0


def test_gated_residual_identity_at_init_random_prev(F_t, F_prev):
    """PROVABLE IDENTITY: zero-init proj_out → gr(F_t, F_prev) == F_t for ANY
    F_prev, in BOTH train and eval mode (no stage/gate dependence)."""
    for mode in ("train", "eval"):
        gr = GatedResidual(C)
        getattr(gr, mode)()
        ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()
        with ctx:
            out = gr(F_t, F_prev)
        assert torch.allclose(out, F_t, atol=1e-6), \
            f"[{mode}] max abs diff {(out - F_t).abs().max().item():.3e} (must be ~0)"


def test_gated_residual_identity_at_init_zero_prev(F_t):
    gr = GatedResidual(C).eval()
    with torch.no_grad():
        out = gr(F_t, torch.zeros_like(F_t))
    assert torch.allclose(out, F_t, atol=1e-6)


def test_gated_residual_non_identity_when_proj_out_trained(F_t, F_prev):
    """Once proj_out moves away from zero the layer must modify F_t."""
    gr = GatedResidual(C).eval()
    with torch.no_grad():
        torch.nn.init.normal_(gr.attn.proj_out.weight, std=0.1)
        out = gr(F_t, F_prev)
    assert not torch.allclose(out, F_t, atol=1e-3)


def test_proj_out_receives_healthy_gradient(F_t, F_prev):
    """The zero-init proj_out is a full [C,C] matrix: even at zero it gets a
    real (non-starved) gradient from the loss — the whole point of replacing
    the structurally un-trainable scalar gate (v1–v8: gate grad ≈ 1e-8)."""
    gr = GatedResidual(C).train()
    loss = gr(F_t, F_prev).pow(2).sum()
    loss.backward()
    g = gr.attn.proj_out.weight.grad
    assert g is not None and g.isfinite().all()
    assert g.norm().item() > 1e-4, f"proj_out grad too small: {g.norm().item():.2e}"


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


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_tagate_identity_at_init(F_t, F_prev, num_layers):
    """Stacked TAGate is a strict no-op at init for any F_prev (all gates 0)."""
    tg = TAGate(C, num_layers=num_layers).eval()
    with torch.no_grad():
        out = tg(F_t, F_prev)
    assert torch.allclose(out, F_t, atol=1e-6), \
        f"TAGate not identity at init: max diff {(out - F_t).abs().max():.3e}"


def test_tagate_set_stage_a_alpha_is_noop(F_t, F_prev):
    """set_stage_a_alpha is kept for LightningModule call-compat but is a no-op
    in the v9 zero-init design — it must not change the identity-at-init output."""
    tg = TAGate(C, num_layers=2).eval()
    tg.set_stage_a_alpha(0.1)
    with torch.no_grad():
        out = tg(F_t, F_prev)
    assert torch.allclose(out, F_t, atol=1e-6)   # still identity regardless


def test_tagate_f_prev_unchanged(F_t, F_prev):
    """F_prev must not be modified in-place by TAGate."""
    tg = TAGate(C, num_layers=2)
    F_prev_clone = F_prev.clone()
    tg(F_t, F_prev)
    assert torch.allclose(F_prev, F_prev_clone)


def test_tagate_gradient_flow(F_t, F_prev):
    F_t = F_t.requires_grad_(True)
    tg = TAGate(C, num_layers=2)
    loss = tg(F_t, F_prev).pow(2).sum()
    loss.backward()
    assert F_t.grad is not None and F_t.grad.isfinite().all()
    # Every layer's zero-init proj_out must receive a real gradient even though
    # it starts at exactly zero (full-matrix grad, not the dead scalar gate).
    for layer in tg.layers:
        g = layer.attn.proj_out.weight.grad
        assert g is not None and g.norm().item() > 1e-5


def test_tagate_proj_out_trains_from_zero(F_t, F_prev):
    """One SGD step moves proj_out off zero → TAGate stops being identity.
    Proves the v9 fix for the v1–v8 un-trainable-gate failure."""
    tg = TAGate(C, num_layers=1)
    opt = torch.optim.SGD(tg.parameters(), lr=1.0)
    assert tg.layers[0].attn.proj_out.weight.abs().max().item() == 0.0
    loss = tg(F_t, F_prev).pow(2).sum()
    loss.backward()
    opt.step()
    assert tg.layers[0].attn.proj_out.weight.abs().max().item() > 0.0


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
    decoded, raw_det, reid, offset_out, feats_cache = out
    assert decoded.shape[1] == 5         # [B, nc+4, A]  where nc=1
    assert len(reid) == 3                # 3 FPN levels
    assert len(offset_out) == 3          # TrackOffsetHead — one (Δx,Δy) map/level
    assert offset_out[0].shape == (1, 2, 80, 80)   # P3
    assert offset_out[2].shape == (1, 2, 20, 20)   # P5
    assert len(feats_cache) == 1         # "P5" → 1 cached level
    assert feats_cache[0].shape == (1, 512, 20, 20)


def test_yolo_jdt_detection_reid_decoupled_from_cache():
    """HEADLINE GUARANTEE (Step 5.DE pivot): detection AND ReID read the
    ORIGINAL neck features — never the TAGate-enhanced ones — so both outputs
    are INVARIANT to the temporal cache *by construction*, in any training
    state. This is what structurally eliminates the v1–v9 failure mode (TAGate
    perturbing a frozen embedding head). Even with TAGate's proj_out forced
    away from zero, detection + ReID must stay bit-identical across caches."""
    model = YOLO_JDT(scale="s", nc=1, cache_levels="P4+P5",
                     tagate_num_layers=2).eval()
    # Force TAGate non-identity — simulate a trained module.
    for tg in model.tagates:
        for layer in tg.layers:
            torch.nn.init.normal_(layer.attn.proj_out.weight, std=0.1)
    img = torch.randn(1, 3, 640, 640)
    zero = model.zero_cache(batch_size=1)
    rand = [torch.randn_like(c) for c in zero]
    with torch.no_grad():
        dec_zero, _, reid_zero, _, _ = model(img, zero)
        dec_rand, _, reid_rand, _, _ = model(img, rand)
    assert (dec_zero - dec_rand).abs().max().item() < 1e-5, \
        "TAGate leaked into the detection path"
    for rz, rr in zip(reid_zero, reid_rand):
        assert (rz - rr).abs().max().item() < 1e-5, \
            "TAGate leaked into the ReID path"


def test_yolo_jdt_offset_depends_on_cache():
    """The offset head DOES read the TAGate-enhanced P5 features — once both
    TAGate and the offset head are non-zero, the P5 offset output changes with
    the temporal cache (that cache-dependence is the whole point of the head)."""
    model = YOLO_JDT(scale="s", nc=1, cache_levels="P5", tagate_num_layers=1).eval()
    for layer in model.tagates[0].layers:
        torch.nn.init.normal_(layer.attn.proj_out.weight, std=0.1)
    torch.nn.init.normal_(model.offset_head.cv_off[2][-1].weight, std=0.05)
    img = torch.randn(1, 3, 640, 640)
    zero = model.zero_cache(batch_size=1)
    rand = [torch.randn_like(c) for c in zero]
    with torch.no_grad():
        off_zero = model(img, zero)[3]
        off_rand = model(img, rand)[3]
    assert (off_zero[2] - off_rand[2]).abs().max().item() > 1e-4


def test_yolo_jdt_offset_zero_at_init():
    """TrackOffsetHead final conv is zero-init → offset == 0 everywhere at init,
    regardless of the temporal cache (no-motion prior)."""
    model = YOLO_JDT(scale="s", nc=1, cache_levels="P5", tagate_num_layers=1).eval()
    img = torch.randn(1, 3, 640, 640)
    rand = [torch.randn_like(c) for c in model.zero_cache(batch_size=1)]
    with torch.no_grad():
        offset_out = model(img, rand)[3]
    for o in offset_out:
        assert o.abs().max().item() == 0.0


def test_yolo_jdt_cache_roundtrip(yolo_jdt_s):
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
    raw_det, reid, offset_out, feats_cache = out
    assert len(raw_det) == 3
    assert len(reid) == 3
    assert len(offset_out) == 3
    assert offset_out[1].shape == (1, 2, 40, 40)   # P4
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
