"""Sanity test for ECC affine recovery."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from yolo_jdt.tracker.cmc import ECCCompensator, warp_kalman_position


@pytest.fixture
def textured_frame():
    """Generate a frame with enough texture for ECC to lock on."""
    rng = np.random.default_rng(0)
    base = (rng.uniform(0, 255, size=(256, 256, 3))).astype(np.uint8)
    # Smooth a bit so ECC has gradients to work with
    return cv2.GaussianBlur(base, (5, 5), 1.0)


def test_first_call_returns_identity(textured_frame):
    cmc = ECCCompensator(downscale=1)
    warp = cmc.update(textured_frame)
    np.testing.assert_allclose(warp, np.eye(2, 3, dtype=np.float32), atol=1e-6)


def test_recovers_pure_translation(textured_frame):
    """Frame translated 8 px in x → recovered warp should have tx ≈ 8."""
    cmc = ECCCompensator(warp_mode=cv2.MOTION_TRANSLATION, downscale=1)
    cmc.update(textured_frame)
    M = np.array([[1, 0, 8], [0, 1, 0]], dtype=np.float32)
    shifted = cv2.warpAffine(textured_frame, M, (256, 256))
    warp = cmc.update(shifted)
    # ECC's `warp` aligns prev → current, so tx should be ≈ 8 (not -8)
    assert abs(warp[0, 2] - 8) < 1.0, f"expected tx≈8, got {warp[0, 2]}"
    assert abs(warp[1, 2]) < 1.0


def test_warp_kalman_position_translation():
    """warp_kalman_position should shift cx, cy by warp's translation."""
    mean = np.array([100.0, 200.0, 0.5, 80.0, 1.0, 1.0, 0.0, 0.0])
    warp = np.array([[1, 0, 5], [0, 1, -3]], dtype=np.float32)
    out = warp_kalman_position(mean, warp)
    assert out[0] == pytest.approx(105.0)
    assert out[1] == pytest.approx(197.0)
    # Other state dims unchanged
    np.testing.assert_allclose(out[2:], mean[2:])


def test_ecc_failure_returns_identity():
    """Passing two completely different frames should ideally recover something
    or fail gracefully — we just verify no exception escapes."""
    cmc = ECCCompensator(downscale=1, max_iters=5, eps=1e-3)
    frame1 = np.zeros((64, 64, 3), dtype=np.uint8)        # uniform — no gradients
    frame2 = np.full((64, 64, 3), 128, dtype=np.uint8)    # uniform — no gradients
    cmc.update(frame1)
    warp = cmc.update(frame2)
    # Should return identity on ECC failure (uniform images have no gradient)
    np.testing.assert_allclose(warp, np.eye(2, 3, dtype=np.float32), atol=1e-3)
