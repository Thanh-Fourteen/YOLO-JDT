"""Shared helpers for raw → standard format converters.

Standard format is documented in `datasets/SPLITS.md`. Each converter
reads its raw layout and emits one JSON file per sequence under
`<dst>/annotations/<split>/<seq>.json`, plus image symlinks under
`<dst>/images/<split>/<seq>/<frame>.jpg` pointing at the original raw
files (no copy).
"""
from __future__ import annotations

import configparser
import json
import os
from pathlib import Path
from typing import Iterable


def parse_seqinfo(path: Path) -> dict:
    """Parse a MOT-style seqinfo.ini file.

    Returns dict with keys: name, frame_rate, seq_length, im_width,
    im_height, im_dir, im_ext.
    """
    cp = configparser.ConfigParser()
    cp.read(path)
    s = cp["Sequence"]
    return {
        "name": s["name"],
        "frame_rate": int(s["frameRate"]),
        "seq_length": int(s["seqLength"]),
        "im_width": int(s["imWidth"]),
        "im_height": int(s["imHeight"]),
        "im_dir": s["imDir"],
        "im_ext": s["imExt"],
    }


def parse_mot_gt(path: Path) -> dict[int, list[dict]]:
    """Parse a MOT-format gt.txt into {frame_id: [object dicts]}.

    Format (1-indexed frames):
        frame, track_id, x, y, w, h, mark, class, vis

    Caller is responsible for any class/mark filtering — this helper
    returns every row as-is.
    """
    out: dict[int, list[dict]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(parts[0])
            obj = {
                "track_id": int(parts[1]),
                "bbox_xywh": [float(parts[2]), float(parts[3]),
                              float(parts[4]), float(parts[5])],
                "mark": int(parts[6]),
                "class": int(parts[7]),
                "visibility": float(parts[8]) if len(parts) > 8 else 1.0,
            }
            out.setdefault(frame, []).append(obj)
    return out


def relative_symlink(src: Path, dst: Path) -> None:
    """Create dst → src symlink using a relative path. Idempotent."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    rel = os.path.relpath(src, dst.parent)
    dst.symlink_to(rel)


def write_seq_json(
    dst_root: Path,
    split: str,
    seq_name: str,
    seq_meta: dict,
    frames: list[dict],
) -> Path:
    """Write a per-sequence annotations JSON. Returns the path written."""
    out = dst_root / "annotations" / split / f"{seq_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": seq_name,
        "frame_rate": seq_meta.get("frame_rate", 0),
        "image_size": [seq_meta["im_width"], seq_meta["im_height"]],
        "num_frames": len(frames),
        "frames": frames,
    }
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return out


def link_seq_images(
    dst_root: Path,
    split: str,
    seq_name: str,
    src_frame_paths: Iterable[tuple[int, Path]],
) -> list[str]:
    """Symlink each (frame_id, src_path) into images/<split>/<seq>/<name>.jpg.

    Returns the list of `<seq>/<name>.jpg` strings (relative to
    `images/<split>/`) for embedding in the JSON.
    """
    rel_names: list[str] = []
    for frame_id, src in src_frame_paths:
        fname = src.name
        dst = dst_root / "images" / split / seq_name / fname
        relative_symlink(src, dst)
        rel_names.append(f"{seq_name}/{fname}")
    return rel_names
