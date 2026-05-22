"""Tests for the Step 5.DE track-offset pivot.

Covers:
- TrackOffsetHead — output shapes, zero-init (offset == 0 at init).
- JointDetectionReIDLoss offset path — offset_only mode, SmoothL1, masking of
  invalid offsets, gradient flow into the offset head AND TAGate.
- BoT-SORT-ReID `_offset_cost` — motion cost geometry + stale-track gating.
"""
from __future__ import annotations

import numpy as np
import torch

from third_party.ultralytics_extract.loss import decode_raw_outputs
from yolo_jdt.losses.joint_loss import JointDetectionReIDLoss
from yolo_jdt.models.head.offset_head import TrackOffsetHead
from yolo_jdt.models.yolo_jdt import YOLO_JDT
from yolo_jdt.tracker.botsort_reid import _offset_cost
from yolo_jdt.tracker.track import Track

# ---------------------------------------------------------------------------
# TrackOffsetHead
# ---------------------------------------------------------------------------

def test_offset_head_shapes():
    head = TrackOffsetHead(ch=(128, 256, 512), hidden=128)
    feats = [torch.randn(2, 128, 80, 80),
             torch.randn(2, 256, 40, 40),
             torch.randn(2, 512, 20, 20)]
    out = head(feats)
    assert len(out) == 3
    assert out[0].shape == (2, 2, 80, 80)
    assert out[1].shape == (2, 2, 40, 40)
    assert out[2].shape == (2, 2, 20, 20)


def test_offset_head_zero_init():
    """Final 1×1 conv is zero-init → offset == 0 for any input at init."""
    head = TrackOffsetHead(ch=(128, 256, 512))
    feats = [torch.randn(1, 128, 80, 80),
             torch.randn(1, 256, 40, 40),
             torch.randn(1, 512, 20, 20)]
    out = head(feats)
    for o in out:
        assert o.abs().max().item() == 0.0


def test_offset_head_trains_from_zero():
    """One SGD step moves the offset head off zero."""
    head = TrackOffsetHead(ch=(128,))
    opt = torch.optim.SGD(head.parameters(), lr=1.0)
    feats = [torch.randn(1, 128, 20, 20)]
    target = torch.ones(1, 2, 20, 20)
    loss = (head(feats)[0] - target).pow(2).mean()
    loss.backward()
    opt.step()
    assert head(feats)[0].abs().max().item() > 0.0


# ---------------------------------------------------------------------------
# Offset loss (JointDetectionReIDLoss, offset_only path)
# ---------------------------------------------------------------------------

def _synthetic_batch(bs: int = 2):
    """Random image → real raw_det/offset_out from YOLO_JDT + a synthetic GT
    batch with one centred box per image carrying a known offset."""
    model = YOLO_JDT(scale="s", nc=1, cache_levels="P5", tagate_num_layers=1).train()
    img = torch.randn(bs, 3, 640, 640)
    cache = model.zero_cache(batch_size=bs)
    raw_det, _reid, offset_out, _ = model(img, cache)
    preds = decode_raw_outputs(raw_det, nc=1, reg_max=16)
    batch = {
        "batch_idx":    torch.arange(bs),
        "cls":          torch.zeros(bs, 1),
        "bboxes":       torch.tensor([[0.5, 0.5, 0.4, 0.6]] * bs),
        "offsets":      torch.tensor([[0.03, -0.02]] * bs),
        "offset_valid": torch.ones(bs, dtype=torch.bool),
    }
    return model, preds, offset_out, batch


def test_offset_loss_offset_only():
    """offset_only=True → only loss[4] is populated; det/reid components zero."""
    model, preds, offset_out, batch = _synthetic_batch()
    loss_fn = JointDetectionReIDLoss(nc=1, num_track_ids=10, lambda_offset=1.0)
    total, comp = loss_fn(preds, batch, offset_per_level=offset_out, offset_only=True)
    assert comp.shape == (5,)
    assert comp[:4].abs().sum().item() == 0.0       # box/cls/dfl/reid skipped
    assert torch.isfinite(total) and total.item() > 0.0
    assert comp[4].item() > 0.0


def test_offset_loss_gradient_flow():
    """The offset loss reaches the offset head's final conv. Even though that
    conv is zero-init, its WEIGHT still gets gradient (dL/dW = input ⊛ upstream
    grad). Summed over levels — whichever level the assigner picked positives
    on, that level's final conv moves."""
    model, preds, offset_out, batch = _synthetic_batch()
    loss_fn = JointDetectionReIDLoss(nc=1, num_track_ids=10, lambda_offset=1.0)
    total, _ = loss_fn(preds, batch, offset_per_level=offset_out, offset_only=True)
    total.backward()
    total_grad = sum(
        seq[-1].weight.grad.abs().sum().item()
        for seq in model.offset_head.cv_off
        if seq[-1].weight.grad is not None
    )
    assert total_grad > 0.0


def test_offset_head_to_tagate_gradient():
    """Once the offset head's final conv is non-zero (post step-1 warmup),
    gradient from the P5 offset output propagates back through it into TAGate.
    At init the zero final conv blocks this — a deliberate 1-step warmup, the
    same dynamics as ControlNet zero-conv."""
    model = YOLO_JDT(scale="s", nc=1, cache_levels="P5", tagate_num_layers=1).train()
    torch.nn.init.normal_(model.offset_head.cv_off[2][-1].weight, std=0.01)
    img = torch.randn(1, 3, 640, 640)
    cache = [torch.randn_like(c) for c in model.zero_cache(batch_size=1)]
    offset_out = model(img, cache)[2]
    offset_out[2].sum().backward()                     # linear → grad propagates
    g = model.tagates[0].layers[0].attn.proj_out.weight.grad
    assert g is not None and g.abs().sum().item() > 0.0


def test_offset_loss_masks_invalid():
    """offset_valid all False → no anchor contributes → loss[4] == 0."""
    model, preds, offset_out, batch = _synthetic_batch()
    batch["offset_valid"] = torch.zeros_like(batch["offset_valid"])
    loss_fn = JointDetectionReIDLoss(nc=1, num_track_ids=10, lambda_offset=1.0)
    _, comp = loss_fn(preds, batch, offset_per_level=offset_out, offset_only=True)
    assert comp[4].item() == 0.0


# ---------------------------------------------------------------------------
# BoT-SORT-ReID offset cost
# ---------------------------------------------------------------------------

def test_offset_cost_perfect_match():
    """A detection whose offset-predicted previous centre lands exactly on the
    track's last centre → cost ≈ 0."""
    t = Track(measurement_xywh=np.array([100.0, 100.0, 40.0, 80.0]), score=0.9)
    t.time_since_update = 1                      # updated last frame → offset valid
    dets = np.array([[120.0, 110.0, 40.0, 80.0, 0.9]])   # centre (140, 150)
    offsets = np.array([[-20.0, -10.0]])         # predicted prev (120, 140) = track centre
    cost = _offset_cost([t], dets, offsets)
    assert cost.shape == (1, 1)
    assert cost[0, 0] < 0.01


def test_offset_cost_far_match():
    """A wrong offset → predicted prev far from the track → cost saturates at 1."""
    t = Track(measurement_xywh=np.array([100.0, 100.0, 40.0, 80.0]), score=0.9)
    t.time_since_update = 1
    dets = np.array([[120.0, 110.0, 40.0, 80.0, 0.9]])
    offsets = np.array([[300.0, 300.0]])         # predicted prev way off
    cost = _offset_cost([t], dets, offsets)
    assert cost[0, 0] == 1.0


def test_offset_cost_stale_track_gated():
    """A track not updated exactly one frame ago gets no offset penalty (0)."""
    t = Track(measurement_xywh=np.array([100.0, 100.0, 40.0, 80.0]), score=0.9)
    t.time_since_update = 5                      # stale → single-frame offset invalid
    dets = np.array([[120.0, 110.0, 40.0, 80.0, 0.9]])
    offsets = np.array([[300.0, 300.0]])
    cost = _offset_cost([t], dets, offsets)
    assert cost[0, 0] == 0.0


def test_offset_cost_empty():
    assert _offset_cost([], np.empty((0, 5)), np.empty((0, 2))).shape == (0, 0)
