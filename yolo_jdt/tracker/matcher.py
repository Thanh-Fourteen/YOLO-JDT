"""IoU cost computation + Hungarian assignment for tracker matching.

Used by ByteTrack and BoT-SORT (and the future ours/ associator).
"""
from __future__ import annotations

import lap
import numpy as np


def iou_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise 1 - IoU cost matrix between two sets of bboxes (xyxy).

    Args:
        boxes_a: shape (N, 4) — N predicted track boxes
        boxes_b: shape (M, 4) — M detection boxes

    Returns:
        cost: shape (N, M), values in [0, 1]. cost = 1 - IoU.
              Empty input on either side returns shape (N, M) of zeros so
              that downstream `linear_assignment` reports zero matches.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    a = boxes_a.astype(np.float64)
    b = boxes_b.astype(np.float64)
    # Intersection rectangles
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return 1.0 - iou


def linear_assignment(cost: np.ndarray, thresh: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian assignment with a cost cap.

    Args:
        cost: shape (N, M). Values above `thresh` are treated as infeasible.
        thresh: maximum acceptable cost for a match (typical 0.8 = IoU >= 0.2).

    Returns:
        matches: shape (K, 2) int — pairs (row_idx, col_idx) with cost <= thresh
        unmatched_a: shape (P,) int — row indices with no acceptable match
        unmatched_b: shape (Q,) int — col indices with no acceptable match
    """
    if cost.size == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(cost.shape[0], dtype=int),
                np.arange(cost.shape[1], dtype=int))

    # lap.lapjv minimizes total cost. extend_cost handles non-square matrices.
    # cost_limit makes lap return -1 for entries above the threshold.
    _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)

    matches = []
    for r, c in enumerate(x):
        if c >= 0:
            matches.append([r, c])
    matches = np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b
