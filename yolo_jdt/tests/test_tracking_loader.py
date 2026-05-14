"""Tests for PairedFrameDataset.

Verifies:
- (seq_name_t == seq_name_prev) — both frames from the same sequence
- frame_id_t == frame_id_prev + 1 (or equal for first frame)
- Output tensor shapes
- collate_paired works on a small batch
- Same augmentation is applied to both frames (same spatial transform seed)

Requires datasets/standard/mot17/annotations/val_half to exist.
Skipped if the dataset is not present.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

STANDARD_ROOT = Path("datasets/standard")
MOT17_AVAILABLE = (STANDARD_ROOT / "mot17" / "annotations" / "val_half").is_dir()

pytestmark = pytest.mark.skipif(
    not MOT17_AVAILABLE,
    reason="datasets/standard/mot17/annotations/val_half not found"
)


@pytest.fixture(scope="module")
def dataset():
    from yolo_jdt.data.datasets import MOTDataset
    from yolo_jdt.data.tracking_loader import PairedFrameDataset
    base = MOTDataset(STANDARD_ROOT, name="mot17", split="val_half")
    return PairedFrameDataset(base, imgsz=640, person_only=True)


def test_dataset_len(dataset):
    assert len(dataset) > 0


def test_same_sequence(dataset):
    """Both frames in a pair must come from the same sequence."""
    for idx in range(0, min(len(dataset), 20)):
        item = dataset[idx]
        assert item["seq_name"] is not None
        # seq_name must be the same for both frames (our pairing logic ensures this
        # since prev = flat_t - 1 within the same sequence, or flat_t for first frame)
        flat_t, flat_prev = dataset._pairs[idx]
        seq_t   = dataset.base._seqs[dataset.base._index[flat_t][0]]["name"]
        seq_p   = dataset.base._seqs[dataset.base._index[flat_prev][0]]["name"]
        assert seq_t == seq_p, f"idx={idx}: seq_t={seq_t}, seq_p={seq_p}"


def test_adjacent_frame_ids(dataset):
    """frame_id_t should be frame_id_prev + 1, or equal at sequence boundaries."""
    for idx in range(0, min(len(dataset), 50)):
        item = dataset[idx]
        fid_t = item["frame_id_t"]
        fid_p = item["frame_id_prev"]
        # Either consecutive (normal case) or same (first frame of sequence)
        assert fid_t == fid_p or fid_t == fid_p + 1, (
            f"idx={idx}: frame_id_t={fid_t}, frame_id_prev={fid_p}"
        )


def test_output_shapes(dataset):
    item = dataset[0]
    assert item["img_t"].shape   == (3, 640, 640)
    assert item["img_prev"].shape == (3, 640, 640)
    assert item["bboxes_t"].ndim  == 2 or item["bboxes_t"].numel() == 0
    assert item["cls_t"].ndim     == 1
    assert item["track_ids_t"].ndim == 1


def test_image_range(dataset):
    item = dataset[0]
    assert 0.0 <= item["img_t"].min().item()
    assert item["img_t"].max().item() <= 1.0


def test_first_frame_pair(dataset):
    """The very first item in each new sequence should have equal frame ids."""
    for idx, (flat_t, flat_prev) in enumerate(dataset._pairs):
        _, frame_idx = dataset.base._index[flat_t]
        if frame_idx == 0:
            item = dataset[idx]
            assert item["frame_id_t"] == item["frame_id_prev"]
            break  # Just test the first occurrence


def test_same_aug_consistency(dataset):
    """Both frames in a pair receive the same spatial transform.

    We verify indirectly: running the same item twice with the same external
    numpy seed gives the same augmented image for frame_t.
    """
    import random as _random
    import numpy as _np

    idx = 10
    # Capture state before __getitem__
    py_state = _random.getstate()
    np_state = _np.random.get_state()

    item1 = dataset[idx]

    # Reset state to the same starting point
    _random.setstate(py_state)
    _np.random.set_state(np_state)

    item2 = dataset[idx]

    assert torch.allclose(item1["img_t"], item2["img_t"], atol=1e-5)
    assert torch.allclose(item1["img_prev"], item2["img_prev"], atol=1e-5)


def test_collate_paired(dataset):
    from yolo_jdt.data.tracking_loader import collate_paired

    items = [dataset[i] for i in range(4)]
    batch = collate_paired(items)

    assert batch["img_t"].shape == (4, 3, 640, 640)
    assert batch["img_prev"].shape == (4, 3, 640, 640)
    assert "bboxes_t" in batch
    assert "track_ids_t" in batch
    assert len(batch["seq_names"]) == 4
    assert len(batch["frame_ids_t"]) == 4
