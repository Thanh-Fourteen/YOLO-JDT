"""Parity tests: standalone YOLO11 (ours) vs Ultralytics YOLO11 (upstream).

Loads pretrained weights into both, runs the same input, asserts maximum
absolute element-wise difference is at or below the FP32 numerical
tolerance budget. Skipped if Ultralytics is not installed (CI may not have it).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "weights" / "pretrained"


@pytest.fixture(scope="module")
def ultralytics_module():
    pytest.importorskip("ultralytics")
    import ultralytics
    return ultralytics


@pytest.mark.parametrize("scale,fname", [
    ("s", "yolo11s.pt"),
    ("m", "yolo11m.pt"),
])
def test_state_dict_loads_into_our_model(scale, fname):
    """Loader maps every Ultralytics key into our model with no shape mismatch."""
    weights_path = WEIGHTS / fname
    if not weights_path.is_file():
        pytest.skip(f"missing {weights_path}")
    pytest.importorskip("ultralytics")
    from yolo_jdt.models.yolo11 import YOLO11
    from yolo_jdt.weights.loader import load_yolo11_weights

    m = YOLO11(scale=scale)
    meta = load_yolo11_weights(m, weights_path)
    assert meta.get("version") is not None


@pytest.mark.parametrize("scale,fname", [
    ("s", "yolo11s.pt"),
    ("m", "yolo11m.pt"),
])
def test_forward_parity_fp32(scale, fname, ultralytics_module):
    """Max absolute diff between our forward and Ultralytics' forward must be
    within FP32 numerical tolerance (≤ 1e-5 in eval mode, single batch)."""
    weights_path = WEIGHTS / fname
    if not weights_path.is_file():
        pytest.skip(f"missing {weights_path}")

    from ultralytics.nn.tasks import DetectionModel
    from yolo_jdt.models.yolo11 import YOLO11
    from yolo_jdt.weights.loader import load_yolo11_weights

    # Upstream model
    ckpt = torch.load(str(weights_path), weights_only=False)
    up_model: DetectionModel = ckpt["model"].float()
    up_model.eval()

    # Ours
    ours = YOLO11(scale=scale)
    load_yolo11_weights(ours, weights_path)
    ours.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        # Upstream returns (decoded[B,84,A], preds_dict). Ours returns
        # (decoded[B,84,A], raw_per_level_list). Decoded is the canonical
        # inference output — match that within FP32 element-wise tolerance.
        up_decoded = up_model(x)[0]
        ours_decoded = ours(x)[0]

    diff_decoded = (up_decoded - ours_decoded).abs().max().item()
    assert diff_decoded <= 1e-5, (
        f"YOLO11{scale} decoded diff {diff_decoded:.3e} exceeds 1e-5")
