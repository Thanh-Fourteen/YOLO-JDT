"""Single-track state container + lifecycle state machine for TBD trackers.

Lifecycle:
    NEW         → freshly created, not yet confirmed
    TRACKED     → confirmed and matched in the current frame
    LOST        → confirmed but not matched in the current frame
    REMOVED     → was lost for too many frames; will be deleted

A NEW track is promoted to TRACKED after `min_hits` consecutive successful
matches. A TRACKED track that misses one frame becomes LOST; if it stays
LOST for more than `max_lost` frames, it becomes REMOVED. ByteTrack-style
loss counter: increment every frame with no match.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count

import numpy as np

from yolo_jdt.tracker.kalman import KalmanFilter

_id_gen = count(1)


def reset_id_counter():
    """For tests / per-sequence isolation: restart the global ID counter at 1."""
    global _id_gen
    _id_gen = count(1)


def _next_id() -> int:
    return next(_id_gen)


class TrackState(Enum):
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


def _xywh_to_xyah(xywh: np.ndarray) -> np.ndarray:
    """[x, y, w, h] (top-left + size) → [cx, cy, a, h]."""
    x, y, w, h = xywh
    return np.array([x + w / 2.0, y + h / 2.0, w / max(h, 1e-6), h], dtype=np.float64)


def _xyah_to_xywh(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float64)


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh
    return np.array([x, y, x + w, y + h], dtype=np.float64)


@dataclass
class Track:
    """Single-track state. Owns its Kalman filter + lifecycle counters."""

    measurement_xywh: np.ndarray              # last observed bbox [x, y, w, h]
    score: float                              # last detection confidence
    track_id: int = -1                        # assigned on activation
    state: TrackState = TrackState.NEW
    hits: int = 0                             # # successful matches
    age: int = 0                              # frames since first seen
    time_since_update: int = 0                # frames since last successful match
    start_frame: int = 0
    end_frame: int = 0
    kalman: KalmanFilter = field(default_factory=KalmanFilter)
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None

    def activate(self, frame_id: int):
        """Promote a NEW track to TRACKED state and assign a global ID."""
        self.track_id = _next_id()
        self.mean, self.covariance = self.kalman.initiate(_xywh_to_xyah(self.measurement_xywh))
        self.state = TrackState.TRACKED
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.start_frame = frame_id
        self.end_frame = frame_id

    def reactivate(self, det_xywh: np.ndarray, score: float, frame_id: int,
                   new_id: bool = False):
        """Re-attach a LOST track to a fresh detection. ByteTrack does not
        change ID by default; pass `new_id=True` to force a new identity."""
        if new_id:
            self.track_id = _next_id()
        self.mean, self.covariance = self.kalman.update(
            self.mean, self.covariance, _xywh_to_xyah(det_xywh))
        self.measurement_xywh = det_xywh
        self.score = score
        self.state = TrackState.TRACKED
        self.hits += 1
        self.time_since_update = 0
        self.end_frame = frame_id

    def predict(self):
        """Kalman predict step (called every frame regardless of match)."""
        if self.mean is None:
            return
        # Reduce vh velocity to 0 when track was lost (ByteTrack heuristic):
        # if not in TRACKED state, zero the height-velocity to avoid drift.
        if self.state != TrackState.TRACKED:
            self.mean[7] = 0.0
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, det_xywh: np.ndarray, score: float, frame_id: int):
        """Successful match: Kalman update + lifecycle bookkeeping."""
        self.mean, self.covariance = self.kalman.update(
            self.mean, self.covariance, _xywh_to_xyah(det_xywh))
        self.measurement_xywh = det_xywh
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED
        self.end_frame = frame_id

    def mark_lost(self):
        """Called when no detection matched this frame."""
        if self.state == TrackState.TRACKED:
            self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED

    @property
    def predicted_xywh(self) -> np.ndarray:
        """Current Kalman-predicted bbox in xywh format."""
        if self.mean is None:
            return self.measurement_xywh
        return _xyah_to_xywh(self.mean[:4])

    @property
    def predicted_xyxy(self) -> np.ndarray:
        return _xywh_to_xyxy(self.predicted_xywh)

    @property
    def is_confirmed(self) -> bool:
        return self.state in (TrackState.TRACKED, TrackState.LOST)
