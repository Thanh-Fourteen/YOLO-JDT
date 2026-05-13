"""MOT-format I/O bridge: standard JSON ↔ MOT-Challenge txt.

MOT-Challenge GT format (per row):
    <frame>, <id>, <x>, <y>, <w>, <h>, <mark>, <class>, <visibility>

MOT-Challenge tracker output format (per row):
    <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, -1, -1, -1

`<frame>` is 1-indexed. `<x>, <y>` are top-left in pixel coords.
`<class>` for pedestrian = 1 in MOT convention. Our standard JSON stores
`class_id = 0` for pedestrian, so we re-map.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def compute_frame_offset(json_path: Path) -> int:
    """Frame offset used to renumber val_half frames to 1..N for MOT eval.

    Standard JSON keeps original frame_ids (e.g. val_half of MOT17-02
    runs frames 301..600). TrackEval bounds-checks against `seqLength`,
    so we must subtract `min_frame_id - 1` from every emitted row to put
    the first frame at 1. Returns the integer offset that consumers
    (tracker writer, ECC, etc.) must subtract.
    """
    with open(json_path) as f:
        seq = json.load(f)
    if not seq["frames"]:
        return 0
    return min(int(fr["frame_id"]) for fr in seq["frames"]) - 1


def gt_json_to_mot(json_path: Path, out_txt_path: Path,
                   class_filter: int = 0,
                   class_remap_to_mot: int = 1,
                   frame_offset: int | None = None) -> int:
    """Convert one standard-format sequence JSON to a MOT-Challenge gt.txt.

    Args:
        json_path: path to `datasets/standard/<name>/annotations/<split>/<seq>.json`.
        out_txt_path: where to write the gt.txt.
        class_filter: only keep objects with this `class_id` (0 = pedestrian
                      in our standard format).
        class_remap_to_mot: write this class id in the MOT row (MOT convention 1).
        frame_offset: subtract this from each frame_id before writing. If None,
                      auto-computed via `compute_frame_offset` (renumbers
                      val_half windows to start at 1).

    Returns:
        Number of GT rows written.
    """
    with open(json_path) as f:
        seq = json.load(f)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    if frame_offset is None:
        frame_offset = (min(int(fr["frame_id"]) for fr in seq["frames"]) - 1
                        if seq["frames"] else 0)

    n = 0
    with open(out_txt_path, "w") as fout:
        for frame in seq["frames"]:
            fid = int(frame["frame_id"]) - frame_offset
            assert fid >= 1, (
                f"renumbered frame_id {fid} < 1 — bad frame_offset "
                f"{frame_offset} for original {frame['frame_id']}")
            for obj in frame["objects"]:
                if obj.get("class_id", 0) != class_filter:
                    continue
                tid = int(obj["track_id"])
                if tid < 0:                       # static-only — not for tracking
                    continue
                x, y, w, h = obj["bbox_xywh"]
                vis = float(obj.get("visibility", 1.0))
                mark = 1 if vis > 0 else 0
                fout.write(
                    f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                    f"{mark},{class_remap_to_mot},{vis:.3f}\n"
                )
                n += 1
    return n


def write_tracker_mot_txt(records: Iterable[tuple[int, int, float, float, float, float, float]],
                          out_txt_path: Path) -> int:
    """Write tracker output rows.

    Args:
        records: iterable of (frame_id, track_id, x, y, w, h, conf) tuples.
                 frame_id 1-indexed, x/y top-left pixel coords.
        out_txt_path: destination MOT-format txt.

    Returns:
        Number of rows written.
    """
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_txt_path, "w") as f:
        for fid, tid, x, y, w, h, conf in records:
            assert fid >= 1, f"MOT format requires 1-indexed frames, got {fid}"
            f.write(
                f"{int(fid)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"{float(conf):.4f},-1,-1,-1\n"
            )
            n += 1
    return n


def cache_gt_dataset(standard_root: Path, dataset_name: str, split: str,
                     gt_cache_root: Path,
                     class_filter: int = 0) -> dict[str, int]:
    """Bulk-convert every sequence JSON of a split into MOT-Challenge layout:

        gt_cache_root/<dataset_name>_<split>/<seq>/gt/gt.txt
        gt_cache_root/<dataset_name>_<split>/<seq>/seqinfo.ini

    Args:
        standard_root: e.g. datasets/standard
        dataset_name: e.g. mot17 / mot20 / dancetrack
        split: e.g. val_half / val
        gt_cache_root: e.g. runs/baselines/_gt_mot

    Returns dict {seq_name: row_count}.
    """
    anno_dir = standard_root / dataset_name / "annotations" / split
    if not anno_dir.is_dir():
        raise FileNotFoundError(f"missing split: {anno_dir}")

    bench_dir = gt_cache_root / f"{dataset_name}_{split}"
    counts: dict[str, int] = {}
    for json_path in sorted(anno_dir.glob("*.json")):
        seq_name = json_path.stem
        seq_dir = bench_dir / seq_name
        # Write gt
        gt_txt = seq_dir / "gt" / "gt.txt"
        n = gt_json_to_mot(json_path, gt_txt, class_filter=class_filter)
        counts[seq_name] = n
        # seqinfo.ini for TrackEval (needs seqLength + image_size)
        with open(json_path) as f:
            seq = json.load(f)
        W, H = seq["image_size"]
        n_frames = len(seq["frames"])
        seqinfo_path = seq_dir / "seqinfo.ini"
        seqinfo_path.write_text(
            "[Sequence]\n"
            f"name={seq_name}\n"
            f"imWidth={W}\nimHeight={H}\n"
            f"seqLength={n_frames}\n"
            f"frameRate={seq.get('frame_rate', 30)}\n"
        )
    return counts
