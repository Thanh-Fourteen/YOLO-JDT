"""ByteTrack — Zhang et al. ECCV 2022.

Two-stage assignment:
    Stage 1: high-confidence detections (>= track_thresh) ↔ all alive tracks
             via IoU + Hungarian.
    Stage 2: low-confidence detections (low_thresh ≤ score < track_thresh)
             ↔ tracks still unmatched after Stage 1.
    Tracks unmatched after both stages: become LOST.
    Detections unmatched after Stage 1 (high-conf only): start NEW tracks.

Key reference: https://github.com/ifzhang/ByteTrack
Paper: arXiv:2110.06864
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from yolo_jdt.tracker.matcher import iou_distance, linear_assignment
from yolo_jdt.tracker.track import Track, TrackState, _xywh_to_xyxy


@dataclass
class ByteTrackConfig:
    track_thresh: float = 0.6     # high-conf threshold for stage-1 + new track init
    low_thresh: float = 0.1       # min score considered for stage 2
    match_thresh: float = 0.8     # IoU-distance cap for Hungarian (= 1 - 0.2 IoU)
    max_lost: int = 30            # frames before LOST → REMOVED
    min_box_area: float = 100.0   # filter tiny detections (in pixels²)


class ByteTrackTracker:
    """Stateful single-sequence tracker. Reset per sequence (don't reuse across seqs).

    Inputs to `update()`:
        detections: shape (N, 5) — [x, y, w, h, score] in original image coords
        frame_id: int (1-indexed, MOT convention)

    Returns: list of TRACKED Track objects to write out for this frame.
    """

    def __init__(self, config: ByteTrackConfig | None = None):
        self.cfg = config or ByteTrackConfig()
        self.tracks: list[Track] = []         # active (TRACKED + LOST)
        self.removed_tracks: list[Track] = []  # bookkeeping for inspection
        self.frame_id = 0

    def update(self, detections: np.ndarray, frame_id: int,
               frame: np.ndarray | None = None) -> list[Track]:
        # `frame` is unused by base ByteTrack; accepted for API parity with
        # subclasses (e.g. BoT-SORT) so callers can pass it unconditionally.
        del frame
        self.frame_id = frame_id
        cfg = self.cfg

        # Filter by box area first (cheap reject of NMS noise + tiny boxes)
        if detections.size:
            areas = detections[:, 2] * detections[:, 3]
            detections = detections[areas >= cfg.min_box_area]

        # Split high / low conf
        if detections.size:
            scores = detections[:, 4]
            high_mask = scores >= cfg.track_thresh
            low_mask = (scores >= cfg.low_thresh) & (scores < cfg.track_thresh)
            dets_high = detections[high_mask]
            dets_low = detections[low_mask]
        else:
            dets_high = np.empty((0, 5))
            dets_low = np.empty((0, 5))

        # ---------- Predict for all alive tracks ----------
        for t in self.tracks:
            t.predict()

        # ---------- Stage 1: high-conf dets ↔ all alive tracks ----------
        if self.tracks and dets_high.size:
            track_xyxy = np.stack([t.predicted_xyxy for t in self.tracks])
            det_xyxy = np.stack([_xywh_to_xyxy(d[:4]) for d in dets_high])
            cost = iou_distance(track_xyxy, det_xyxy)
            matches, unmatched_t1, unmatched_d1 = linear_assignment(
                cost, thresh=cfg.match_thresh)
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_t1 = np.arange(len(self.tracks))
            unmatched_d1 = np.arange(len(dets_high))

        for ti, di in matches:
            t = self.tracks[ti]
            d = dets_high[di]
            if t.state == TrackState.TRACKED:
                t.update(d[:4], score=float(d[4]), frame_id=frame_id)
            else:
                t.reactivate(d[:4], score=float(d[4]), frame_id=frame_id, new_id=False)

        # ---------- Stage 2: low-conf dets ↔ tracks unmatched in stage 1 ----------
        # Only consider TRACKED tracks for stage 2 (ByteTrack convention; LOST
        # tracks tied to low-conf are too risky and tend to drift).
        unmatched_tracks_s1 = [self.tracks[i] for i in unmatched_t1
                                if self.tracks[i].state == TrackState.TRACKED]
        if unmatched_tracks_s1 and dets_low.size:
            track_xyxy = np.stack([t.predicted_xyxy for t in unmatched_tracks_s1])
            det_xyxy = np.stack([_xywh_to_xyxy(d[:4]) for d in dets_low])
            cost = iou_distance(track_xyxy, det_xyxy)
            # Lower IoU bar for stage 2 (= higher cost cap) is sometimes used,
            # but ByteTrack reference keeps 0.5; we stick with same match_thresh.
            matches2, unmatched_t2, _ = linear_assignment(cost, thresh=0.5)
        else:
            matches2 = np.empty((0, 2), dtype=int)
            unmatched_t2 = np.arange(len(unmatched_tracks_s1))

        for ti, di in matches2:
            t = unmatched_tracks_s1[ti]
            d = dets_low[di]
            t.update(d[:4], score=float(d[4]), frame_id=frame_id)

        # Tracks still unmatched after BOTH stages → LOST
        matched_t1_set = set(matches[:, 0].tolist())
        matched_t2_track_objs = {id(unmatched_tracks_s1[i]) for i in matches2[:, 0]}
        for i, t in enumerate(self.tracks):
            if i in matched_t1_set:
                continue
            if id(t) in matched_t2_track_objs:
                continue
            t.mark_lost()

        # ---------- New tracks from unmatched high-conf detections ----------
        for di in unmatched_d1:
            d = dets_high[di]
            new_track = Track(
                measurement_xywh=d[:4].copy().astype(np.float64),
                score=float(d[4]),
            )
            new_track.activate(frame_id)
            self.tracks.append(new_track)

        # ---------- Remove stale tracks ----------
        active = []
        for t in self.tracks:
            if t.state == TrackState.LOST and t.time_since_update > cfg.max_lost:
                t.mark_removed()
                self.removed_tracks.append(t)
            else:
                active.append(t)
        self.tracks = active

        # Return tracks to write out — only TRACKED in current frame
        return [t for t in self.tracks if t.state == TrackState.TRACKED]
