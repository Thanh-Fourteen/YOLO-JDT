"""Sanity tests for the 8-d Kalman filter used by ByteTrack/BoT-SORT."""
from __future__ import annotations

import numpy as np
import pytest

from yolo_jdt.tracker.kalman import KalmanFilter


def test_initiate_shapes():
    kf = KalmanFilter()
    z = np.array([320.0, 240.0, 0.5, 100.0])  # cx, cy, a, h
    mean, cov = kf.initiate(z)
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    # Position component matches measurement, velocity initialized to zero
    np.testing.assert_allclose(mean[:4], z)
    np.testing.assert_allclose(mean[4:], 0.0)


def test_predict_advances_with_velocity():
    kf = KalmanFilter()
    z = np.array([100.0, 100.0, 1.0, 50.0])
    mean, cov = kf.initiate(z)
    # Inject a known velocity (5 px/frame in x)
    mean[4] = 5.0
    new_mean, new_cov = kf.predict(mean, cov)
    np.testing.assert_allclose(new_mean[0], 105.0)  # cx advanced by vcx
    np.testing.assert_allclose(new_mean[1], 100.0)  # cy unchanged (vy=0)
    # Covariance grew (process noise added)
    assert np.trace(new_cov) > np.trace(cov)


def test_update_shrinks_position_uncertainty():
    """Posterior covariance over the position dims should shrink after update."""
    kf = KalmanFilter()
    z0 = np.array([100.0, 100.0, 1.0, 50.0])
    mean, cov = kf.initiate(z0)
    mean, cov = kf.predict(mean, cov)

    pre_pos_var = np.trace(cov[:4, :4])
    z1 = np.array([102.0, 101.0, 1.0, 50.0])
    new_mean, new_cov = kf.update(mean, cov, z1)
    post_pos_var = np.trace(new_cov[:4, :4])
    assert post_pos_var < pre_pos_var, (
        f"update should shrink position uncertainty; got {pre_pos_var:.4f} -> {post_pos_var:.4f}")
    # Updated position should be between predicted and measurement
    assert mean[0] <= new_mean[0] <= z1[0] or z1[0] <= new_mean[0] <= mean[0]


def test_update_idempotent_to_perfect_measurements():
    """If we keep observing the same measurement, the state should converge to it."""
    kf = KalmanFilter()
    z = np.array([100.0, 100.0, 0.4, 80.0])
    mean, cov = kf.initiate(z)
    for _ in range(20):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, z)
    np.testing.assert_allclose(mean[:4], z, atol=0.1)
    # Velocities should converge near zero
    assert np.abs(mean[4:6]).max() < 0.5


def test_velocity_estimation_from_repeated_motion():
    """Feed a constant-velocity sequence of measurements; velocity estimate
    should approach the true velocity."""
    kf = KalmanFilter()
    true_v = 4.0
    z0 = np.array([100.0, 100.0, 1.0, 50.0])
    mean, cov = kf.initiate(z0)
    for t in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        z = np.array([100.0 + t * true_v, 100.0, 1.0, 50.0])
        mean, cov = kf.update(mean, cov, z)
    # Velocity in x should converge near true_v (within 10% after 30 steps)
    assert abs(mean[4] - true_v) < true_v * 0.1, f"velocity estimate {mean[4]:.3f} vs true {true_v}"


def test_predict_does_not_change_aspect_velocity_when_zero():
    kf = KalmanFilter()
    z = np.array([100.0, 100.0, 1.0, 50.0])
    mean, cov = kf.initiate(z)
    mean, cov = kf.predict(mean, cov)
    # va was 0 and aspect is constant in this test; the projected mean
    # should hold the predicted-from-zero-velocity value
    assert mean[6] == pytest.approx(0.0, abs=1e-9)
