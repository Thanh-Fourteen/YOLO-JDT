"""Sanity tests for standard-format Dataset classes.

Loads a random sample from each dataset/split and asserts:
    - bbox bounds 0 ≤ {cx, cy, w, h} ≤ 1
    - frame_id monotonically increasing within each sequence
    - track_id consistency: a track_id seen at frame f and frame f+1
      points to objects whose box centers are within a per-frame
      distance bound (no obvious id swap in raw GT)
    - returned tensors are well-formed and shapes line up

These are not learning-quality tests — they catch converter / loader
regressions only.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from yolo_jdt.data.datasets import (
    CrowdHumanDataset,
    DanceTrackDataset,
    MOTDataset,
    StandardSeqDataset,
)

ROOT = Path(__file__).resolve().parents[2] / "datasets" / "standard"

# Fixed seed for reproducible sample selection — the test isn't a
# stress test of every frame, just a representative random check.
random.seed(0)
torch.manual_seed(0)


def _maybe_skip_dataset(root: Path, name: str):
    p = root / name / "annotations"
    if not p.is_dir():
        pytest.skip(f"standard dataset not built: {p}")


def _build_cases() -> list[tuple[str, StandardSeqDataset]]:
    """Construct one (label, ds) per (dataset, split). Skips datasets
    not yet converted so this file works incrementally."""
    cases: list[tuple[str, StandardSeqDataset]] = []
    for ds_name, splits in (
        ("mot17", ("train", "train_half", "val_half")),
        ("mot20", ("train", "train_half", "val_half")),
        ("dancetrack", ("train", "val")),
        ("crowdhuman", ("train", "val")),
    ):
        if not (ROOT / ds_name / "annotations").is_dir():
            continue
        for split in splits:
            if not (ROOT / ds_name / "annotations" / split).is_dir():
                continue
            if ds_name == "mot17" or ds_name == "mot20":
                ds = MOTDataset(ROOT, name=ds_name, split=split)
            elif ds_name == "dancetrack":
                ds = DanceTrackDataset(ROOT, split=split)
            else:
                ds = CrowdHumanDataset(ROOT, split=split)
            cases.append((f"{ds_name}/{split}", ds))
    return cases


CASES = _build_cases()


@pytest.mark.parametrize("label,ds", CASES, ids=[c[0] for c in CASES])
def test_random_sample_well_formed(label: str, ds: StandardSeqDataset):
    """One random frame per (dataset, split) must return well-formed tensors."""
    assert len(ds) > 0, f"{label}: empty dataset"

    idx = random.randrange(len(ds))
    image, targets = ds[idx]

    # Image well-formed
    assert image.dtype == torch.float32, f"{label}: image dtype {image.dtype}"
    assert image.ndim == 3 and image.shape[0] == 3, \
        f"{label}: image shape {tuple(image.shape)}"
    assert image.min() >= 0.0 and image.max() <= 1.0, \
        f"{label}: image range [{image.min():.3f}, {image.max():.3f}]"

    H, W = int(targets["image_size"][0]), int(targets["image_size"][1])
    assert image.shape[1] == H and image.shape[2] == W, \
        f"{label}: shape mismatch image {tuple(image.shape)} vs HxW {H}x{W}"

    # Targets well-formed
    n = targets["bboxes"].shape[0]
    assert targets["class_ids"].shape == (n,)
    assert targets["track_ids"].shape == (n,)
    assert targets["visibility"].shape == (n,)
    assert targets["iscrowd"].shape == (n,)

    # Bbox bounds
    if n > 0:
        b = targets["bboxes"]
        assert (b >= 0.0).all() and (b <= 1.0).all(), (
            f"{label}: bbox out of [0,1] — min {b.min():.3f} max {b.max():.3f}")
        # cx ± w/2 and cy ± h/2 also must be in [0, 1]
        cx, cy, w, h = b.unbind(dim=1)
        x1 = cx - w / 2
        x2 = cx + w / 2
        y1 = cy - h / 2
        y2 = cy + h / 2
        assert (x1 >= -1e-4).all() and (x2 <= 1.0 + 1e-4).all(), \
            f"{label}: bbox x corners out of frame"
        assert (y1 >= -1e-4).all() and (y2 <= 1.0 + 1e-4).all(), \
            f"{label}: bbox y corners out of frame"


@pytest.mark.parametrize("label,ds", CASES, ids=[c[0] for c in CASES])
def test_frame_ids_monotonic_per_seq(label: str, ds: StandardSeqDataset):
    """Within each sequence, frame_id must be strictly increasing."""
    by_seq: dict[str, list[int]] = defaultdict(list)
    for seq in ds._seqs:
        prev = -1
        for f in seq["frames"]:
            fid = int(f["frame_id"])
            assert fid > prev, (
                f"{label} seq={seq['name']}: non-monotonic frame_id "
                f"{prev} -> {fid}")
            prev = fid
            by_seq[seq["name"]].append(fid)
    # Sanity: at least one frame per sequence.
    assert all(len(v) > 0 for v in by_seq.values()), \
        f"{label}: empty sequence found"


@pytest.mark.parametrize(
    "label,ds",
    [(l, d) for l, d in CASES if not d.is_static],
    ids=[c[0] for c in CASES if not c[1].is_static],
)
def test_track_id_consistency(label: str, ds: StandardSeqDataset):
    """For tracking datasets, a track_id appearing in adjacent frames
    must move by less than one image diagonal between those frames
    (sanity check — catches gross GT corruption / id swap in conversion)."""
    for seq in ds._seqs:
        last_seen: dict[int, tuple[int, float, float]] = {}
        for frame in seq["frames"]:
            fid = int(frame["frame_id"])
            for o in frame["objects"]:
                tid = int(o["track_id"])
                if tid < 0:
                    continue
                x, y, w, h = o["bbox_xywh"]
                cx = x + w / 2
                cy = y + h / 2
                if tid in last_seen:
                    pf, pcx, pcy = last_seen[tid]
                    if fid - pf == 1:
                        # Adjacent-frame movement: a person can run at
                        # most ~10 m/s; at typical 25-30 fps that's
                        # ~30-40 cm/frame in world coords, projected to
                        # well under one image diagonal even from the
                        # closest viewpoint. Allow a generous bound.
                        Wimg, Himg = ds._frame_image_size(seq, frame)
                        diag = (Wimg ** 2 + Himg ** 2) ** 0.5
                        d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                        assert d < 0.5 * diag, (
                            f"{label} seq={seq['name']} tid={tid} frame "
                            f"{pf}->{fid}: jump {d:.0f}px > 0.5 diag")
                last_seen[tid] = (fid, cx, cy)


def test_count_overall():
    """Cheap aggregate sanity: total dataset sizes are in the right ballpark."""
    expected = {
        "mot17/train": 5316,
        "mot17/train_half": 2658,
        "mot17/val_half": 2658,
        "mot20/train": 8931,
        "dancetrack/train": None,   # variable, just check > 30000
        "crowdhuman/train": 15000,
        "crowdhuman/val": 4370,
    }
    by_label = {label: len(ds) for label, ds in CASES}
    for label, exp in expected.items():
        if label not in by_label:
            continue
        if exp is None:
            assert by_label[label] >= 30000, f"{label}: only {by_label[label]} frames"
        else:
            # MOT17 half-split frame counts depend on N//2 parity
            assert abs(by_label[label] - exp) <= 10, (
                f"{label}: expected ~{exp}, got {by_label[label]}")
