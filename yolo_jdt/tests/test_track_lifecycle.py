"""Lifecycle tests for the Track state machine."""
from __future__ import annotations

import numpy as np
import pytest

from yolo_jdt.tracker.track import Track, TrackState, reset_id_counter


@pytest.fixture(autouse=True)
def _reset_ids():
    reset_id_counter()


def _make_track(score: float = 0.9) -> Track:
    return Track(measurement_xywh=np.array([100.0, 100.0, 50.0, 100.0]), score=score)


def test_new_track_starts_unassigned():
    t = _make_track()
    assert t.state == TrackState.NEW
    assert t.track_id == -1
    assert t.hits == 0


def test_activate_assigns_id_and_promotes():
    t = _make_track()
    t.activate(frame_id=1)
    assert t.state == TrackState.TRACKED
    assert t.track_id == 1     # first ID after reset
    assert t.hits == 1
    assert t.start_frame == 1
    assert t.end_frame == 1


def test_id_counter_monotonic_across_tracks():
    t1, t2, t3 = _make_track(), _make_track(), _make_track()
    t1.activate(0); t2.activate(0); t3.activate(0)
    assert [t1.track_id, t2.track_id, t3.track_id] == [1, 2, 3]


def test_predict_then_update_keeps_tracked_state():
    t = _make_track()
    t.activate(frame_id=1)
    t.predict()
    assert t.time_since_update == 1
    t.update(np.array([102.0, 101.0, 50.0, 100.0]), score=0.85, frame_id=2)
    assert t.state == TrackState.TRACKED
    assert t.time_since_update == 0
    assert t.hits == 2
    assert t.end_frame == 2


def test_mark_lost_transitions_tracked_to_lost():
    t = _make_track()
    t.activate(frame_id=1)
    t.predict()                  # frame 2 — no match
    t.mark_lost()
    assert t.state == TrackState.LOST


def test_lost_track_kept_alive_until_max_lost():
    """Simulate: confirm, then lose for 30 frames, then mark removed."""
    t = _make_track()
    t.activate(frame_id=1)
    for f in range(2, 32):       # 30 frames lost
        t.predict()
        t.mark_lost()
    assert t.state == TrackState.LOST
    assert t.time_since_update == 30
    # Still confirmed (just lost), not removed
    assert t.is_confirmed
    # Tracker decides when to call mark_removed; do it manually here
    t.mark_removed()
    assert t.state == TrackState.REMOVED


def test_reactivate_keeps_id_by_default():
    t = _make_track()
    t.activate(frame_id=1)
    original_id = t.track_id
    for _ in range(5):
        t.predict()
        t.mark_lost()
    t.reactivate(np.array([110.0, 105.0, 50.0, 100.0]), score=0.9, frame_id=7)
    assert t.track_id == original_id
    assert t.state == TrackState.TRACKED
    assert t.time_since_update == 0


def test_reactivate_with_new_id_changes_id():
    t = _make_track()
    t.activate(frame_id=1)
    original_id = t.track_id
    t.predict(); t.mark_lost()
    t.reactivate(np.array([110.0, 100.0, 50.0, 100.0]), score=0.9, frame_id=2,
                 new_id=True)
    assert t.track_id != original_id


def test_predicted_xywh_matches_initial_measurement_after_activate():
    t = _make_track()
    t.activate(frame_id=1)
    np.testing.assert_allclose(t.predicted_xywh, [100.0, 100.0, 50.0, 100.0], atol=1e-6)
