"""Smoke tests for ByteTrack: synthetic sequences with known motion.

Goal: verify identity assignment, persistence across frames, and recovery
from short occlusions. Not a numerical match against ifzhang/ByteTrack —
just sanity that the lifecycle + matching chain is correct.
"""
from __future__ import annotations

import numpy as np
import pytest

from yolo_jdt.tracker.bytetrack import ByteTrackConfig, ByteTrackTracker
from yolo_jdt.tracker.track import reset_id_counter


@pytest.fixture(autouse=True)
def _reset_ids():
    reset_id_counter()


def _detection(x, y, w=50, h=100, score=0.9):
    return [x, y, w, h, score]


def test_single_track_persists_across_50_frames():
    """A single object moving linearly should keep one stable ID."""
    tracker = ByteTrackTracker()
    seen_ids = set()
    for frame_id in range(1, 51):
        x = 100.0 + frame_id * 2.0
        dets = np.array([_detection(x, 100.0)])
        active = tracker.update(dets, frame_id=frame_id)
        assert len(active) == 1, f"frame {frame_id}: expected 1 active track"
        seen_ids.add(active[0].track_id)
    assert len(seen_ids) == 1, f"expected 1 unique ID, got {seen_ids}"


def test_three_separate_tracks_get_three_ids():
    """3 spatially distinct objects → 3 unique IDs."""
    tracker = ByteTrackTracker()
    for frame_id in range(1, 11):
        dets = np.array([
            _detection(100.0 + frame_id * 2, 100.0),
            _detection(400.0 + frame_id * 1, 100.0),
            _detection(700.0 - frame_id * 1, 100.0),
        ])
        active = tracker.update(dets, frame_id=frame_id)
        assert len(active) == 3
    final_ids = sorted(t.track_id for t in tracker.tracks)
    assert len(set(final_ids)) == 3


def test_track_recovers_after_short_occlusion():
    """Object disappears for 5 frames then comes back; same ID should re-attach."""
    tracker = ByteTrackTracker()
    # Frames 1-10: visible, get an ID
    first_id = None
    for frame_id in range(1, 11):
        x = 100.0 + frame_id * 2.0
        dets = np.array([_detection(x, 100.0)])
        active = tracker.update(dets, frame_id=frame_id)
        first_id = active[0].track_id

    # Frames 11-15: missing — track goes LOST, no active output
    for frame_id in range(11, 16):
        active = tracker.update(np.empty((0, 5)), frame_id=frame_id)
        assert len(active) == 0

    # Frame 16: reappears at predicted position; should reattach to original ID
    x = 100.0 + 16 * 2.0
    dets = np.array([_detection(x, 100.0)])
    active = tracker.update(dets, frame_id=16)
    assert len(active) == 1
    assert active[0].track_id == first_id, f"expected ID {first_id}, got {active[0].track_id}"


def test_low_confidence_only_does_not_create_track():
    """Detections below track_thresh should never start a NEW track."""
    tracker = ByteTrackTracker(ByteTrackConfig(track_thresh=0.6, low_thresh=0.1))
    for frame_id in range(1, 6):
        dets = np.array([_detection(100, 100, score=0.3)])    # below 0.6
        active = tracker.update(dets, frame_id=frame_id)
        assert len(active) == 0


def test_low_confidence_can_keep_existing_track_alive():
    """High-conf seeds a track; subsequent low-conf detections sustain it via stage 2."""
    tracker = ByteTrackTracker(ByteTrackConfig(track_thresh=0.6, low_thresh=0.1))
    # Frame 1: high-conf seed
    dets = np.array([_detection(100, 100, score=0.95)])
    active = tracker.update(dets, frame_id=1)
    assert len(active) == 1
    seed_id = active[0].track_id

    # Frames 2-5: low-conf, near-by detections — stage 2 should recover them
    for frame_id in range(2, 6):
        x = 100.0 + frame_id * 2.0
        dets = np.array([_detection(x, 100.0, score=0.3)])
        active = tracker.update(dets, frame_id=frame_id)
        assert len(active) == 1, f"frame {frame_id}: stage-2 match should keep track"
        assert active[0].track_id == seed_id


def test_tiny_box_filtered_by_min_area():
    """Boxes below min_box_area should be dropped before assignment."""
    tracker = ByteTrackTracker(ByteTrackConfig(min_box_area=200.0))
    # 10x10 box has area 100 < 200 → filtered
    dets = np.array([_detection(100, 100, w=10, h=10, score=0.9)])
    active = tracker.update(dets, frame_id=1)
    assert len(active) == 0
