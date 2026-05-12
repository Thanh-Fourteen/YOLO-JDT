"""Convert MOT17 / MOT20 raw layout → standard format.

Usage:
    python -m yolo_jdt.data.converters.mot_to_std \\
        --src datasets/raw/mot17 --dst datasets/standard/mot17 --name mot17
    python -m yolo_jdt.data.converters.mot_to_std \\
        --src datasets/raw/mot20 --dst datasets/standard/mot20 --name mot20

Layout differences handled:
    mot17: <src>/MOT17/images/{train,test}/<seq>/{img1,gt,seqinfo.ini}
    mot20: <src>/MOT20/{train,test}/<seq>/{img1,gt,seqinfo.ini}

Splits emitted: train (full), train_half, val_half, test (no GT).
GT filter: keep `mark == 1` only (always corresponds to class == 1
pedestrian). Other rows (static person, distractor, occluder, vehicles)
are dropped — see datasets/SPLITS.md for rationale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_jdt.data.converters._common import (
    link_seq_images, parse_mot_gt, parse_seqinfo, write_seq_json,
)


def _seq_root(src: Path, name: str, raw_split: str) -> Path:
    """Return the parent dir holding `<seq>/img1/...` for a split."""
    if name == "mot17":
        return src / "MOT17" / "images" / raw_split
    if name == "mot20":
        return src / "MOT20" / raw_split
    raise ValueError(f"unknown name: {name}")


def _build_frames(
    seq_meta: dict,
    img_dir: Path,
    gt_by_frame: dict[int, list[dict]] | None,
    frame_range: range,
    img_rel_names: list[str],
) -> list[dict]:
    """Stitch per-frame symlink paths + filtered GT objects into JSON frames."""
    frames = []
    for frame_id, rel_name in zip(frame_range, img_rel_names):
        objs = []
        for o in (gt_by_frame or {}).get(frame_id, []):
            if o["mark"] != 1:
                continue
            objs.append({
                "track_id": o["track_id"],
                "class_id": 0,  # pedestrian, only class
                "bbox_xywh": o["bbox_xywh"],
                "visibility": o["visibility"],
                "iscrowd": 0,
            })
        frames.append({
            "frame_id": frame_id,
            "image": rel_name,
            "objects": objs,
        })
    return frames


def _convert_seq(
    seq_dir: Path, dst: Path, split: str, frame_range: range | None = None,
) -> tuple[str, int]:
    """Convert one sequence under `seq_dir` (with `img1/`, optional `gt/gt.txt`,
    `seqinfo.ini`). If `frame_range` is None, use full sequence length.
    Returns (seq_name, num_frames_written).
    """
    seq_meta = parse_seqinfo(seq_dir / "seqinfo.ini")
    seq_name = seq_meta["name"]
    img_dir = seq_dir / seq_meta["im_dir"]
    ext = seq_meta["im_ext"]

    full_range = range(1, seq_meta["seq_length"] + 1)
    rng = frame_range if frame_range is not None else full_range

    # Build (frame_id, src_path) pairs in order
    src_frames = [(fid, img_dir / f"{fid:06d}{ext}") for fid in rng]
    rel_names = link_seq_images(dst, split, seq_name, src_frames)

    gt_path = seq_dir / "gt" / "gt.txt"
    gt_by_frame = parse_mot_gt(gt_path) if gt_path.exists() else None

    frames = _build_frames(seq_meta, img_dir, gt_by_frame, rng, rel_names)
    write_seq_json(dst, split, seq_name, seq_meta, frames)
    return seq_name, len(frames)


def convert(src: Path, dst: Path, name: str) -> dict:
    """Convert MOT17 / MOT20. Returns summary dict."""
    summary: dict[str, list[tuple[str, int]]] = {
        "train": [], "train_half": [], "val_half": [], "test": [],
    }

    train_root = _seq_root(src, name, "train")
    test_root = _seq_root(src, name, "test")

    # train / train_half / val_half all derive from the train sequences.
    for seq_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        seq_meta = parse_seqinfo(seq_dir / "seqinfo.ini")
        n = seq_meta["seq_length"]
        half = n // 2
        full_rng = range(1, n + 1)
        train_half_rng = range(1, half + 1)
        val_half_rng = range(half + 1, n + 1)

        summary["train"].append(_convert_seq(seq_dir, dst, "train", full_rng))
        summary["train_half"].append(
            _convert_seq(seq_dir, dst, "train_half", train_half_rng))
        summary["val_half"].append(
            _convert_seq(seq_dir, dst, "val_half", val_half_rng))

    if test_root.exists():
        for seq_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
            summary["test"].append(_convert_seq(seq_dir, dst, "test"))

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--name", choices=["mot17", "mot20"], required=True)
    args = ap.parse_args()

    summary = convert(args.src, args.dst, args.name)
    print(json.dumps(
        {split: [{"seq": s, "frames": n} for s, n in entries]
         for split, entries in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
