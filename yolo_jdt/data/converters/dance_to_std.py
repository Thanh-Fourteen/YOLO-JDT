"""Convert DanceTrack raw layout → standard format.

Raw layout: <src>/{train,val,test}/dancetrackXXXX/{img1,gt/gt.txt,seqinfo.ini}

Splits emitted: train, val, test (test has no gt).

Usage:
    python -m yolo_jdt.data.converters.dance_to_std \\
        --src datasets/raw/dancetrack --dst datasets/standard/dancetrack
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_jdt.data.converters._common import (
    link_seq_images, parse_mot_gt, parse_seqinfo, write_seq_json,
)


def _convert_seq(seq_dir: Path, dst: Path, split: str) -> tuple[str, int]:
    seq_meta = parse_seqinfo(seq_dir / "seqinfo.ini")
    seq_name = seq_meta["name"]
    img_dir = seq_dir / seq_meta["im_dir"]
    ext = seq_meta["im_ext"]

    rng = range(1, seq_meta["seq_length"] + 1)
    src_frames = [(fid, img_dir / f"{fid:08d}{ext}") for fid in rng]
    rel_names = link_seq_images(dst, split, seq_name, src_frames)

    gt_path = seq_dir / "gt" / "gt.txt"
    gt_by_frame = parse_mot_gt(gt_path) if gt_path.exists() else None

    frames = []
    for frame_id, rel_name in zip(rng, rel_names):
        objs = []
        for o in (gt_by_frame or {}).get(frame_id, []):
            # DanceTrack: all rows are class 1, mark 1 — keep as-is.
            objs.append({
                "track_id": o["track_id"],
                "class_id": 0,  # person
                "bbox_xywh": o["bbox_xywh"],
                "visibility": o["visibility"],
                "iscrowd": 0,
            })
        frames.append({
            "frame_id": frame_id,
            "image": rel_name,
            "objects": objs,
        })
    write_seq_json(dst, split, seq_name, seq_meta, frames)
    return seq_name, len(frames)


def convert(src: Path, dst: Path) -> dict:
    summary: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        split_dir = src / split
        if not split_dir.is_dir():
            continue
        for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            summary[split].append(_convert_seq(seq_dir, dst, split))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    summary = convert(args.src, args.dst)
    print(json.dumps(
        {split: [{"seq": s, "frames": n} for s, n in entries]
         for split, entries in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
