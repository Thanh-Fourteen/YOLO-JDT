"""Sanity tests for IoU cost + Hungarian assignment."""
from __future__ import annotations

import numpy as np
import pytest

from yolo_jdt.tracker.matcher import iou_distance, linear_assignment


def test_iou_distance_self_zero():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
    cost = iou_distance(boxes, boxes)
    np.testing.assert_allclose(np.diag(cost), 0.0, atol=1e-9)
    # Off-diagonal: no overlap → cost = 1
    assert cost[0, 1] == pytest.approx(1.0)
    assert cost[1, 0] == pytest.approx(1.0)


def test_iou_distance_partial_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[5, 5, 15, 15]], dtype=np.float64)
    cost = iou_distance(a, b)
    # inter = 5x5 = 25, union = 100 + 100 - 25 = 175 → IoU = 25/175 ≈ 0.143
    expected_iou = 25.0 / 175.0
    np.testing.assert_allclose(cost[0, 0], 1.0 - expected_iou, atol=1e-9)


def test_iou_distance_empty_inputs():
    a = np.empty((0, 4))
    b = np.array([[0, 0, 10, 10]], dtype=np.float64)
    assert iou_distance(a, b).shape == (0, 1)
    assert iou_distance(b, a).shape == (1, 0)
    assert iou_distance(a, a).shape == (0, 0)


def test_linear_assignment_perfect_match():
    """3 tracks, 3 detections, all perfectly aligned."""
    cost = np.array([
        [0.0, 0.9, 0.9],
        [0.9, 0.0, 0.9],
        [0.9, 0.9, 0.0],
    ])
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert matches.shape == (3, 2)
    assert sorted(matches.tolist()) == [[0, 0], [1, 1], [2, 2]]
    assert ua.size == 0 and ub.size == 0


def test_linear_assignment_thresh_cuts_high_cost():
    """1 valid match, 1 above threshold should be unmatched."""
    cost = np.array([
        [0.1, 0.9],
        [0.9, 0.95],
    ])
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert matches.shape == (1, 2)
    assert matches.tolist() == [[0, 0]]
    assert sorted(ua.tolist()) == [1]
    assert sorted(ub.tolist()) == [1]


def test_linear_assignment_more_dets_than_tracks():
    """Rectangular cost: 2 tracks, 4 detections."""
    cost = np.array([
        [0.1, 0.9, 0.9, 0.9],
        [0.9, 0.1, 0.9, 0.9],
    ])
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert matches.shape == (2, 2)
    assert sorted(matches.tolist()) == [[0, 0], [1, 1]]
    assert ua.size == 0
    assert sorted(ub.tolist()) == [2, 3]


def test_linear_assignment_empty_cost():
    cost = np.empty((0, 5))
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert matches.shape == (0, 2)
    assert ua.size == 0
    assert ub.tolist() == [0, 1, 2, 3, 4]


def test_linear_assignment_uses_global_optimum():
    """Greedy would pick cost[0, 0] first, but optimum is the diagonal."""
    cost = np.array([
        [0.4, 0.1],
        [0.3, 0.9],
    ])
    matches, ua, ub = linear_assignment(cost, thresh=0.8)
    # Hungarian: total cost 0.4 + 0.9 vs 0.1 + 0.3. The optimum is
    # row0->col1 (0.1) + row1->col0 (0.3) = 0.4 < diag (0.4 + 0.9).
    pairs = sorted(matches.tolist())
    assert pairs == [[0, 1], [1, 0]]
