"""8-d state Kalman filter for ByteTrack/BoT-SORT.

State:  [cx, cy, a, h, vcx, vcy, va, vh]
            cx, cy: bbox center
            a:      aspect ratio (w / h)
            h:      bbox height
            vcx..vh: velocities

Measurement: [cx, cy, a, h]

Constant-velocity dynamics, additive Gaussian noise. Process- and
measurement-noise covariances are scaled by the current bbox height
(noise grows with object size — same recipe as SORT/DeepSORT/ByteTrack).
"""
from __future__ import annotations

import numpy as np

# Std-weights from SORT/ByteTrack reference impl.
STD_WEIGHT_POSITION = 1.0 / 20.0
STD_WEIGHT_VELOCITY = 1.0 / 160.0


class KalmanFilter:
    """Filter for a single track. Numpy only — fast for single-track step."""

    def __init__(self):
        ndim = 4
        # Constant-velocity state transition: x_{k+1} = F @ x_k
        self._F = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._F[i, ndim + i] = 1.0
        # Measurement matrix: z_k = H @ x_k (selects [cx, cy, a, h])
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float64)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create state mean + covariance from a single bbox measurement.

        Args:
            measurement: shape (4,) = [cx, cy, a, h]

        Returns:
            mean: shape (8,)
            covariance: shape (8, 8)
        """
        mean_pos = measurement.astype(np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2 * STD_WEIGHT_POSITION * h,
            2 * STD_WEIGHT_POSITION * h,
            1e-2,
            2 * STD_WEIGHT_POSITION * h,
            10 * STD_WEIGHT_VELOCITY * h,
            10 * STD_WEIGHT_VELOCITY * h,
            1e-5,
            10 * STD_WEIGHT_VELOCITY * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate state forward 1 step (no measurement)."""
        h = mean[3]
        std_pos = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-2,
            STD_WEIGHT_POSITION * h,
        ]
        std_vel = [
            STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * h,
            1e-5,
            STD_WEIGHT_VELOCITY * h,
        ]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + Q
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space."""
        h = mean[3]
        std = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-1,
            STD_WEIGHT_POSITION * h,
        ]
        R = np.diag(np.square(std))

        proj_mean = self._H @ mean
        proj_cov = self._H @ covariance @ self._H.T + R
        return proj_mean, proj_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bayesian update with a new measurement."""
        proj_mean, proj_cov = self.project(mean, covariance)

        # Kalman gain via Cholesky-solve (more stable than explicit inverse)
        chol_factor, lower = _cho_factor(proj_cov)
        K = _cho_solve_T(chol_factor, lower, covariance @ self._H.T)

        innovation = measurement - proj_mean
        new_mean = mean + K @ innovation
        new_cov = covariance - K @ proj_cov @ K.T
        return new_mean, new_cov


def _cho_factor(A: np.ndarray) -> tuple[np.ndarray, bool]:
    from scipy.linalg import cho_factor
    return cho_factor(A, lower=True)


def _cho_solve_T(L: np.ndarray, lower: bool, B_T_unused: np.ndarray) -> np.ndarray:
    """Solve A K^T = B^T  =>  K = (A^{-1} B^T)^T using cho_solve.

    We pass `B_T_unused` already as `cov @ H.T` (8×4); cho_solve expects
    RHS shape compatible with L (4×4) — so transpose, solve, transpose.
    """
    from scipy.linalg import cho_solve
    # B_T_unused has shape (8, 4); we want K of shape (8, 4)
    # Solve L L^T x = b  where b = B_T_unused.T (shape 4×8)  → x shape (4, 8)
    # K = x.T  (shape 8×4)
    x = cho_solve((L, lower), B_T_unused.T)
    return x.T
