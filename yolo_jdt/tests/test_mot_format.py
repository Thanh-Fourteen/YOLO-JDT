"""Round-trip tests for MOT format I/O."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_jdt.eval.mot_format import (cache_gt_dataset, gt_json_to_mot,
                                       write_tracker_mot_txt)


@pytest.fixture
def tiny_seq_json(tmp_path):
    """Create a minimal standard-format sequence JSON."""
    p = tmp_path / "seq001.json"
    seq = {
        "name": "seq001",
        "image_size": [1920, 1080],
        "frame_rate": 30,
        "frames": [
            {"frame_id": 1, "image": "seq001/000001.jpg",
             "objects": [
                {"track_id": 1, "class_id": 0,
                 "bbox_xywh": [100.5, 200.3, 50.0, 100.0], "visibility": 0.9},
                {"track_id": 2, "class_id": 0,
                 "bbox_xywh": [400.0, 200.0, 60.0, 110.0], "visibility": 0.5},
                {"track_id": -1, "class_id": 0,
                 "bbox_xywh": [800.0, 300.0, 30.0, 50.0], "visibility": 0.0},  # static, skipped
             ]},
            {"frame_id": 2, "image": "seq001/000002.jpg",
             "objects": [
                {"track_id": 1, "class_id": 0,
                 "bbox_xywh": [102.0, 202.0, 50.0, 100.0], "visibility": 1.0},
             ]},
        ],
    }
    p.write_text(json.dumps(seq))
    return p


def test_gt_json_to_mot_writes_correct_rows(tiny_seq_json, tmp_path):
    out = tmp_path / "gt.txt"
    n = gt_json_to_mot(tiny_seq_json, out)
    assert n == 3       # 2 in frame 1 (one static skipped) + 1 in frame 2
    rows = out.read_text().strip().split("\n")
    assert len(rows) == 3

    parts = rows[0].split(",")
    assert int(parts[0]) == 1                            # frame
    assert int(parts[1]) == 1                            # track_id
    assert float(parts[2]) == pytest.approx(100.5)       # x
    assert float(parts[3]) == pytest.approx(200.3)       # y
    assert float(parts[4]) == pytest.approx(50.0)        # w
    assert float(parts[5]) == pytest.approx(100.0)       # h
    assert int(parts[6]) == 1                            # mark
    assert int(parts[7]) == 1                            # MOT class id (pedestrian)
    assert float(parts[8]) == pytest.approx(0.9)         # visibility


def test_gt_json_to_mot_skips_static_track_id():
    # Already covered by the count assertion in the previous test
    pass


def test_write_tracker_mot_txt_format(tmp_path):
    out = tmp_path / "pred.txt"
    records = [
        (1, 1, 100.5, 200.3, 50.0, 100.0, 0.95),
        (1, 2, 400.0, 200.0, 60.0, 110.0, 0.78),
        (2, 1, 102.0, 202.0, 50.0, 100.0, 0.93),
    ]
    n = write_tracker_mot_txt(records, out)
    assert n == 3
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 3
    parts = lines[0].split(",")
    assert int(parts[0]) == 1
    assert int(parts[1]) == 1
    assert float(parts[2]) == pytest.approx(100.5)
    assert float(parts[6]) == pytest.approx(0.95)
    # Last 3 cols always -1
    assert parts[7] == "-1" and parts[8] == "-1" and parts[9] == "-1"


def test_write_tracker_mot_txt_rejects_zero_indexed_frame(tmp_path):
    """MOT requires frame_id >= 1; 0 should error to catch off-by-one bugs."""
    out = tmp_path / "pred.txt"
    with pytest.raises(AssertionError, match="1-indexed"):
        write_tracker_mot_txt([(0, 1, 0, 0, 10, 10, 1.0)], out)


def test_cache_gt_dataset_layout(tiny_seq_json, tmp_path):
    """Verify TrackEval-style layout: <gt_cache>/<dataset>_<split>/<seq>/gt/gt.txt."""
    # Set up a fake standard-root structure
    standard_root = tmp_path / "standard"
    seq_anno_dir = standard_root / "fakeset" / "annotations" / "val_half"
    seq_anno_dir.mkdir(parents=True)
    # Re-use the fixture content
    src_seq = json.loads(tiny_seq_json.read_text())
    (seq_anno_dir / "seq001.json").write_text(json.dumps(src_seq))

    cache_root = tmp_path / "cache"
    counts = cache_gt_dataset(standard_root, "fakeset", "val_half", cache_root)
    assert counts == {"seq001": 3}

    bench_dir = cache_root / "fakeset_val_half"
    assert (bench_dir / "seq001" / "gt" / "gt.txt").is_file()
    seqinfo = (bench_dir / "seq001" / "seqinfo.ini").read_text()
    assert "imWidth=1920" in seqinfo
    assert "imHeight=1080" in seqinfo
    assert "seqLength=2" in seqinfo
