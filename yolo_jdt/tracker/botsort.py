"""BoT-SORT — Aharon, Orfaig, Bobrovsky 2022 (arXiv:2206.14651).

Extends ByteTrack with:
- Camera Motion Compensation (CMC): apply affine warp from frame_{t-1} →
  frame_t to all kalman-predicted positions, undoing camera ego-motion.
- (Future) ReID embedding cost integrated into the matching cost matrix.
  In Step 3.BCD we ship a no-op ReID hook so that downstream code can
  plug in the YOLO11-JDE branch (Phase 4) without touching this file.

Frame is required as input to `update()` (alongside detections) so that
CMC can compute its warp. The frame is the original BGR image (numpy,
H×W×3) — same format `infer_tracking.py` already loads via cv2.imread.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from yolo_jdt.tracker.bytetrack import ByteTrackConfig, ByteTrackTracker
from yolo_jdt.tracker.cmc import ECCCompensator, warp_kalman_position


@dataclass
class BoTSORTConfig(ByteTrackConfig):
    cmc_max_iters: int = 100
    cmc_eps: float = 1e-5
    cmc_downscale: int = 2
    use_cmc: bool = True


class BoTSORTTracker(ByteTrackTracker):
    """BoT-SORT = ByteTrack + CMC. ReID hook is a no-op for now."""

    def __init__(self, config: BoTSORTConfig | None = None):
        super().__init__(config or BoTSORTConfig())
        self.cfg: BoTSORTConfig = self.cfg
        if self.cfg.use_cmc:
            import cv2
            self.cmc = ECCCompensator(
                warp_mode=cv2.MOTION_EUCLIDEAN,
                max_iters=self.cfg.cmc_max_iters,
                eps=self.cfg.cmc_eps,
                downscale=self.cfg.cmc_downscale,
            )
        else:
            self.cmc = None

    def update(self, detections: np.ndarray, frame_id: int,
               frame: np.ndarray | None = None) -> list:
        """Same as ByteTrack.update but with CMC applied before Kalman predict.

        Args:
            detections: (N, 5) [x, y, w, h, score]
            frame_id: 1-indexed
            frame: original BGR image for this frame (H, W, 3). Required
                when use_cmc=True.

        Returns: list of TRACKED Track objects (same as base class).
        """
        # Apply CMC warp to all kalman states BEFORE predict step
        if self.cmc is not None and frame is not None and self.tracks:
            warp = self.cmc.update(frame)
            # Skip warp if it's identity (first frame or ECC failure)
            if not np.allclose(warp, np.eye(2, 3, dtype=np.float32), atol=1e-6):
                for t in self.tracks:
                    if t.mean is not None:
                        t.mean = warp_kalman_position(t.mean, warp)

        # Delegate to ByteTrack 2-stage matching (which calls predict + assign)
        return super().update(detections, frame_id)
