"""BoT-SORT-ReID — BoT-SORT (CMC) + per-track ReID embedding cost.

Stage-1 cost = w_iou·(1-IoU) + w_reid·(1-cosine_sim) + w_offset·motion_cost.

Per-track embedding maintained via EMA on each successful match (see
`Track.update_embedding`). When a new track is born, its initial embedding
is the first matched detection's embedding.

Track-offset cue (Step 5.DE pivot): when YOLO-JDT supplies per-detection
inter-frame offsets, `motion_cost` measures the distance between each
detection's offset-predicted previous centre and each track's last observed
centre — a learned motion cue that competes with / complements the Kalman
constant-velocity prior. `w_offset=0` (the config default) disables it, so
plain BoT-SORT-ReID (Phase 4 baseline) is byte-for-byte unchanged.

Stage 2 (low-conf detections) keeps the IoU-only cost — appearance is
unreliable for low-quality detections (motion blur, occlusion).

Reference: BoT-SORT §3.3 (Aharon et al. 2022); CenterTrack offset matching.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yolo_jdt.tracker.botsort import BoTSORTConfig, BoTSORTTracker
from yolo_jdt.tracker.matcher import iou_distance, linear_assignment
from yolo_jdt.tracker.track import Track, TrackState, _xywh_to_xyxy


@dataclass
class BoTSORTReIDConfig(BoTSORTConfig):
    w_iou: float = 0.7                # weight on IoU distance in stage-1 cost
    w_reid: float = 0.3               # weight on (1 - cosine sim) in stage-1 cost
    w_offset: float = 0.0             # weight on track-offset motion cost (0 = off)
    reid_match_thresh: float = 0.25   # cap on reid distance alone for sanity
    embedding_alpha: float = 0.9      # EMA smoothing on track embedding


def _offset_cost(tracks: list, dets: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Track-offset motion cost matrix.

    For detection j with predicted inter-frame offset `offsets[j] = (Δx, Δy)`
    (= predicted centre_{t-1} − centre_t, in detection pixel space), its
    offset-predicted previous centre is `det_centre_j + offsets[j]`. The cost
    against track i is the distance between that predicted previous centre and
    track i's last observed centre, normalized by mean box height.

    Args:
        tracks:  list of Track (rows of the cost matrix).
        dets:    (D, 5) [x, y, w, h, score] detections (columns).
        offsets: (D, 2) per-detection offsets, same pixel space as `dets`.
    Returns:
        cost: (T, D) in [0, 1]. Rows for tracks NOT updated exactly one frame
        ago are 0 — the single-frame offset cannot bridge a longer gap, so
        those tracks get no motion penalty and fall back to IoU/ReID.
    """
    T, D = len(tracks), len(dets)
    cost = np.zeros((T, D), dtype=np.float64)
    if T == 0 or D == 0:
        return cost
    track_c = np.array(
        [[t.measurement_xywh[0] + t.measurement_xywh[2] / 2.0,
          t.measurement_xywh[1] + t.measurement_xywh[3] / 2.0] for t in tracks],
        dtype=np.float64)                                            # [T, 2]
    track_h = np.array([t.measurement_xywh[3] for t in tracks], dtype=np.float64)
    track_fresh = np.array([t.time_since_update == 1 for t in tracks])
    det_c = dets[:, :2] + dets[:, 2:4] / 2.0                         # [D, 2]
    det_h = dets[:, 3].astype(np.float64)                            # [D]
    pred_prev = det_c + offsets.astype(np.float64)                   # [D, 2]
    diff = track_c[:, None, :] - pred_prev[None, :, :]               # [T, D, 2]
    dist = np.sqrt((diff ** 2).sum(-1))                              # [T, D]
    sigma = 0.5 * (track_h[:, None] + det_h[None, :]) + 1e-6         # [T, D]
    cost = np.clip(dist / sigma, 0.0, 1.0)
    cost[~track_fresh] = 0.0
    return cost


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise (1 - cosine_similarity) cost matrix.

    Args:
        a: shape (N, D), L2-normalized
        b: shape (M, D), L2-normalized

    Returns:
        cost: shape (N, M), values in [0, 2]. cost = 1 - dot(a, b).
        Zero / empty inputs return shape (N, M) of zeros.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    sim = a.astype(np.float64) @ b.astype(np.float64).T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


class BoTSORTReIDTracker(BoTSORTTracker):
    """BoT-SORT + ReID. Same lifecycle as BoT-SORT but stage-1 cost combines
    IoU and ReID cosine distance.

    `update()` accepts an extra `embeddings` arg: shape (N_dets, reid_dim)
    L2-normalized. If provided, used in stage-1 matching.
    """

    def __init__(self, config: BoTSORTReIDConfig | None = None):
        super().__init__(config or BoTSORTReIDConfig())
        self.cfg: BoTSORTReIDConfig = self.cfg

    def update(self, detections: np.ndarray, frame_id: int,
               frame: np.ndarray | None = None,
               embeddings: np.ndarray | None = None,
               offsets: np.ndarray | None = None) -> list[Track]:
        """Same signature as BoT-SORT.update + `embeddings` for ReID cost +
        `offsets` for the track-offset motion cue.

        Args:
            detections: (N, 5) [x, y, w, h, score]
            frame_id: 1-indexed
            frame: original BGR image for CMC
            embeddings: (N, reid_dim) L2-normalized; None disables ReID cost
                        (falls back to BoT-SORT IoU+CMC behavior).
            offsets: (N, 2) per-detection inter-frame offsets in the same pixel
                     space as `detections`; None (or cfg.w_offset==0) disables
                     the motion cue.
        """
        # CMC + Kalman predict same as BoT-SORT
        if self.cmc is not None and frame is not None and self.tracks:
            warp = self.cmc.update(frame)
            from yolo_jdt.tracker.cmc import warp_kalman_position
            if not np.allclose(warp, np.eye(2, 3, dtype=np.float32), atol=1e-6):
                for t in self.tracks:
                    if t.mean is not None:
                        t.mean = warp_kalman_position(t.mean, warp)
        # Use base ByteTrack matching but override stage-1 cost matrix.
        # We can't just super().update() because that uses pure IoU cost.
        # Reimplement the dispatch with the combined cost.
        cfg = self.cfg
        self.frame_id = frame_id

        if detections.size:
            areas = detections[:, 2] * detections[:, 3]
            keep_area = areas >= cfg.min_box_area
            detections = detections[keep_area]
            if embeddings is not None:
                embeddings = embeddings[keep_area]
            if offsets is not None:
                offsets = offsets[keep_area]
        if detections.size:
            scores = detections[:, 4]
            high_mask = scores >= cfg.track_thresh
            low_mask = (scores >= cfg.low_thresh) & (scores < cfg.track_thresh)
            dets_high = detections[high_mask]
            dets_low = detections[low_mask]
            embs_high = embeddings[high_mask] if embeddings is not None else None
            offs_high = offsets[high_mask] if offsets is not None else None
        else:
            dets_high = np.empty((0, 5))
            dets_low = np.empty((0, 5))
            embs_high = None
            offs_high = None

        for t in self.tracks:
            t.predict()

        # ---- Stage 1: high-conf dets ↔ alive tracks (IoU + ReID combined) ----
        if self.tracks and dets_high.size:
            track_xyxy = np.stack([t.predicted_xyxy for t in self.tracks])
            det_xyxy = np.stack([_xywh_to_xyxy(d[:4]) for d in dets_high])
            cost_iou = iou_distance(track_xyxy, det_xyxy)

            if embs_high is not None:
                # Tracks that have an embedding contribute ReID cost; others
                # fall back to IoU-only (cost_reid = 0 for those rows).
                track_embs = np.zeros((len(self.tracks), embs_high.shape[1]),
                                      dtype=np.float64)
                has_emb = np.zeros(len(self.tracks), dtype=bool)
                for i, t in enumerate(self.tracks):
                    if t.embedding is not None:
                        track_embs[i] = t.embedding
                        has_emb[i] = True
                cost_reid = _cosine_distance(track_embs, embs_high)
                cost_reid[~has_emb] = 0.0   # no penalty for tracks without emb
                # Reject implausible ReID matches outright (cap)
                cost_reid[cost_reid > cfg.reid_match_thresh] = 1.0
                cost = cfg.w_iou * cost_iou + cfg.w_reid * cost_reid
            else:
                cost = cost_iou

            # Track-offset motion cue (Step 5.DE pivot). Additive — w_offset=0
            # leaves the cost (and thus Phase-4 baselines) byte-for-byte intact.
            if offs_high is not None and cfg.w_offset > 0:
                cost = cost + cfg.w_offset * _offset_cost(
                    self.tracks, dets_high, offs_high)

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
            if embs_high is not None:
                t.update_embedding(embs_high[di], alpha=cfg.embedding_alpha)

        # ---- Stage 2: low-conf dets ↔ TRACKED unmatched (IoU-only) -----------
        unmatched_tracks_s1 = [self.tracks[i] for i in unmatched_t1
                                if self.tracks[i].state == TrackState.TRACKED]
        if unmatched_tracks_s1 and dets_low.size:
            track_xyxy = np.stack([t.predicted_xyxy for t in unmatched_tracks_s1])
            det_xyxy = np.stack([_xywh_to_xyxy(d[:4]) for d in dets_low])
            cost = iou_distance(track_xyxy, det_xyxy)
            matches2, unmatched_t2, _ = linear_assignment(cost, thresh=0.5)
        else:
            matches2 = np.empty((0, 2), dtype=int)
            unmatched_t2 = np.arange(len(unmatched_tracks_s1))

        for ti, di in matches2:
            t = unmatched_tracks_s1[ti]
            d = dets_low[di]
            t.update(d[:4], score=float(d[4]), frame_id=frame_id)
            # Note: no ReID embedding update for low-conf — appearance unreliable

        matched_t1_set = set(matches[:, 0].tolist())
        matched_t2_track_objs = {id(unmatched_tracks_s1[i]) for i in matches2[:, 0]}
        for i, t in enumerate(self.tracks):
            if i in matched_t1_set or id(t) in matched_t2_track_objs:
                continue
            t.mark_lost()

        # ---- New tracks from unmatched high-conf detections ------------------
        for di in unmatched_d1:
            d = dets_high[di]
            new_track = Track(
                measurement_xywh=d[:4].copy().astype(np.float64),
                score=float(d[4]),
            )
            new_track.activate(frame_id)
            if embs_high is not None:
                new_track.update_embedding(embs_high[di], alpha=0.0)  # alpha=0 = use new emb directly
            self.tracks.append(new_track)

        # ---- Remove stale ----------------------------------------------------
        active = []
        for t in self.tracks:
            if t.state == TrackState.LOST and t.time_since_update > cfg.max_lost:
                t.mark_removed()
                self.removed_tracks.append(t)
            else:
                active.append(t)
        self.tracks = active

        return [t for t in self.tracks if t.state == TrackState.TRACKED]
