"""Tests for ID-preserving mosaic + global ID mapping."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from yolo_jdt.data.mosaic_id import (DetTrainIDDataset, build_global_id_map,
                                       collate_det_with_ids)


@pytest.fixture
def fake_standard_root(tmp_path):
    """Create a tiny standard-format dataset with 2 sequences and known track_ids."""
    root = tmp_path / "standard"
    seq1 = {
        "name": "seq01",
        "image_size": [640, 480],
        "frame_rate": 30,
        "frames": [
            {"frame_id": 1, "image": "seq01/000001.jpg",
             "objects": [
                {"track_id": 1, "class_id": 0, "bbox_xywh": [100, 100, 50, 100], "visibility": 1.0},
                {"track_id": 2, "class_id": 0, "bbox_xywh": [300, 100, 50, 100], "visibility": 1.0},
                {"track_id": -1, "class_id": 0, "bbox_xywh": [500, 100, 30, 60], "visibility": 0.0},
             ]},
            {"frame_id": 2, "image": "seq01/000002.jpg",
             "objects": [
                {"track_id": 1, "class_id": 0, "bbox_xywh": [102, 102, 50, 100], "visibility": 1.0},
             ]},
        ],
    }
    seq2 = {
        "name": "seq02",
        "image_size": [640, 480],
        "frame_rate": 30,
        "frames": [
            {"frame_id": 1, "image": "seq02/000001.jpg",
             "objects": [
                {"track_id": 5, "class_id": 0, "bbox_xywh": [200, 200, 60, 120], "visibility": 1.0},
             ]},
        ],
    }
    anno = root / "fakeset" / "annotations" / "train_half"
    anno.mkdir(parents=True)
    (anno / "seq01.json").write_text(json.dumps(seq1))
    (anno / "seq02.json").write_text(json.dumps(seq2))
    return root


def test_build_global_id_map_unique_per_seq_track_pair(fake_standard_root):
    m = build_global_id_map(fake_standard_root, "fakeset", "train_half")
    # 3 unique (seq, tid) pairs: (seq01,1), (seq01,2), (seq02,5). Static (-1) excluded.
    assert len(m) == 3
    assert ("seq01", 1) in m and ("seq01", 2) in m and ("seq02", 5) in m
    # IDs are contiguous [0, num_classes)
    assert sorted(m.values()) == [0, 1, 2]


def test_build_global_id_map_excludes_negative_track_id(fake_standard_root):
    m = build_global_id_map(fake_standard_root, "fakeset", "train_half")
    # The static (track_id=-1) instance must NOT appear
    assert all(t >= 0 for (_, t) in m.keys())


# ---- Dataset wrapper tests with a synthetic in-memory base ----

class _FakeBase(Dataset):
    """Minimal stand-in for StandardSeqDataset — yields synthetic targets."""

    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        H, W = 480, 640
        img = torch.rand(3, H, W)
        seq_name = "seq01" if idx % 2 == 0 else "seq02"
        # Encode a known track_id per item so we can verify it survives augmentation
        tid = (idx % 3) + 1     # 1, 2, or 3
        # 1 box, person class, normalized cxcywh
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]], dtype=torch.float32)
        cls = torch.tensor([0], dtype=torch.int64)
        tids = torch.tensor([tid], dtype=torch.int64)
        return img, {
            "bboxes": boxes, "class_ids": cls, "track_ids": tids,
            "seq_name": seq_name, "frame_id": idx,
            "image_size": torch.tensor([H, W], dtype=torch.int64),
            "image_id": f"{seq_name}/{idx:06d}.jpg",
            "visibility": torch.ones(1), "iscrowd": torch.zeros(1, dtype=torch.int64),
        }


def test_id_preserving_dataset_track_ids_in_targets():
    base = _FakeBase(n=16)
    # Build a global map covering the synthetic seqs/tids used by _FakeBase
    gid_map = {
        ("seq01", 1): 0, ("seq01", 2): 1, ("seq01", 3): 2,
        ("seq02", 1): 3, ("seq02", 2): 4, ("seq02", 3): 5,
    }
    ds = DetTrainIDDataset(base, gid_map, imgsz=64, mosaic_p=0.0,
                            hsv=(0, 0, 0), flip_p=0.0, scale=0.0, translate=0.0)
    img, tgt = ds[0]
    assert "track_ids" in tgt
    assert tgt["track_ids"].dtype == torch.int64
    # Box survived augmentation; track_id mapped to global id
    if tgt["bboxes"].shape[0] >= 1:
        # idx 0 → seq01, tid=1 → global 0
        assert tgt["track_ids"][0].item() == 0


def test_collate_with_ids_preserves_track_ids():
    base = _FakeBase(n=4)
    gid_map = {
        ("seq01", 1): 0, ("seq01", 2): 1, ("seq01", 3): 2,
        ("seq02", 1): 3, ("seq02", 2): 4, ("seq02", 3): 5,
    }
    ds = DetTrainIDDataset(base, gid_map, imgsz=64, mosaic_p=0.0,
                            hsv=(0, 0, 0), flip_p=0.0, scale=0.0, translate=0.0)
    items = [ds[i] for i in range(4)]
    batch = collate_det_with_ids(items)
    assert "track_ids" in batch
    assert batch["track_ids"].dtype == torch.int64
    # Each item contributes some boxes with their track_id; total length matches batch_idx
    assert batch["track_ids"].shape[0] == batch["batch_idx"].shape[0]


def test_static_instance_maps_to_minus_one():
    """Instances not present in global_id_map should get track_id = -1 sentinel."""
    base = _FakeBase(n=2)
    gid_map = {("seq01", 1): 0}        # only one entry; everything else missing
    ds = DetTrainIDDataset(base, gid_map, imgsz=64, mosaic_p=0.0,
                            hsv=(0, 0, 0), flip_p=0.0, scale=0.0, translate=0.0)
    # idx 1 → seq02, tid=2 → not in map → should be -1
    _, tgt = ds[1]
    if tgt["bboxes"].shape[0] >= 1:
        assert tgt["track_ids"][0].item() == -1


def test_mosaic_preserves_track_ids():
    """Mosaic of 4 frames combines instances from different sources, each
    keeping its own global track_id label."""
    base = _FakeBase(n=20)
    gid_map = {
        ("seq01", 1): 0, ("seq01", 2): 1, ("seq01", 3): 2,
        ("seq02", 1): 3, ("seq02", 2): 4, ("seq02", 3): 5,
    }
    ds = DetTrainIDDataset(base, gid_map, imgsz=128, mosaic_p=1.0,
                            hsv=(0, 0, 0), flip_p=0.0,
                            scale=0.0, translate=0.0)
    _, tgt = ds[0]
    # After mosaic, multiple instances may survive; each must still have a
    # known global track_id from the map (or -1 if mapped is missing — but
    # since all 6 labels are present in the map, expect non-negative).
    if tgt["bboxes"].shape[0] >= 2:
        for tid in tgt["track_ids"]:
            assert tid.item() in {0, 1, 2, 3, 4, 5}, (
                f"unexpected track_id {tid.item()} after mosaic")
