"""Convert CrowdHuman odgt → standard format.

Raw layout:
    <src>/images/{train,val}/<id>,<hash>.jpg
    <src>/annotations/annotation_{train,val}.odgt

odgt = one JSON record per line:
    {"ID": "<id>,<hash>", "gtboxes": [{"tag": "person|mask",
        "fbox": [x,y,w,h] full body, "hbox": ..., "vbox": ..., ...}]}

Standard format treats CrowdHuman as a single "sequence" per split (the
images are unrelated, but schema uniformity simplifies the loader). All
boxes get track_id = -1. Filter: keep tag == "person" (drop mask
ignore regions).

Image size is read once per image via Pillow (CrowdHuman has variable
resolution — no shared seqinfo).

Usage:
    python -m yolo_jdt.data.converters.crowdhuman_to_std \\
        --src datasets/raw/crowdhuman --dst datasets/standard/crowdhuman
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from yolo_jdt.data.converters._common import relative_symlink


def _parse_odgt(path: Path) -> dict[str, list[dict]]:
    """Returns {ID: list of person-tagged gtbox dicts}."""
    out: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            persons = []
            for gb in rec["gtboxes"]:
                if gb.get("tag") != "person":
                    continue
                # fbox = full body box, [x_tl, y_tl, w, h] in pixels
                persons.append({"fbox": gb["fbox"], "extra": gb.get("extra", {})})
            out[rec["ID"]] = persons
    return out


def _convert_split(src: Path, dst: Path, split: str) -> int:
    odgt = src / "annotations" / f"annotation_{split}.odgt"
    img_dir = src / "images" / split
    if not odgt.is_file():
        raise FileNotFoundError(odgt)

    anno_by_id = _parse_odgt(odgt)
    seq_name = f"crowdhuman_{split}"
    frames = []

    for frame_id, (img_id, persons) in enumerate(sorted(anno_by_id.items()), start=1):
        src_img = img_dir / f"{img_id}.jpg"
        if not src_img.exists():
            raise FileNotFoundError(f"missing image for ID {img_id}: {src_img}")

        # Symlink under standard images/<split>/<seq>/<frame>.jpg
        # Use the original filename (it is unique within split).
        dst_img = dst / "images" / split / seq_name / src_img.name
        relative_symlink(src_img, dst_img)

        # CrowdHuman has variable image size — read with Pillow (lazy).
        with Image.open(src_img) as im:
            W, H = im.size

        objs = []
        for p in persons:
            x, y, w, h = p["fbox"]
            # fbox can extend outside image bounds (CrowdHuman quirk for occluded
            # full-body boxes). Clip non-destructively here so downstream
            # normalization stays in [0, 1]; preserve original area as much as
            # possible by clipping each side independently.
            x2 = max(0, min(W, x + w))
            y2 = max(0, min(H, y + h))
            x = max(0, min(W, x))
            y = max(0, min(H, y))
            w = max(0, x2 - x)
            h = max(0, y2 - y)
            if w <= 1 or h <= 1:
                continue  # degenerate after clipping — skip
            objs.append({
                "track_id": -1,
                "class_id": 0,
                "bbox_xywh": [x, y, w, h],
                "visibility": 1.0 - float(p["extra"].get("occ", 0)),
                "iscrowd": 0,
            })

        frames.append({
            "frame_id": frame_id,
            "image": f"{seq_name}/{src_img.name}",
            "objects": objs,
            "image_size": [W, H],  # per-frame because variable
        })

    # CrowdHuman is variable-resolution; don't lie about a single image_size at
    # the seq level. Use the first frame as a placeholder; per-frame size is
    # authoritative.
    first_size = frames[0]["image_size"] if frames else [0, 0]
    out = dst / "annotations" / split / f"{seq_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": seq_name,
        "frame_rate": 0,
        "image_size": first_size,    # placeholder; see frame.image_size
        "variable_image_size": True,
        "num_frames": len(frames),
        "frames": frames,
    }
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return len(frames)


def convert(src: Path, dst: Path) -> dict:
    summary = {}
    for split in ("train", "val"):
        summary[split] = _convert_split(src, dst, split)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    summary = convert(args.src, args.dst)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
