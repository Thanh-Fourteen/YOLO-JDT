"""Standard-format dataset base class.

Reads `datasets/standard/<name>/{images,annotations}/<split>/...` produced
by the converters in `yolo_jdt/data/converters/`. Yields per-frame
samples where bounding boxes have been normalized to YOLO `(cx, cy, w, h)`
in `[0, 1]`.

Returned `__getitem__(idx) -> (image_tensor, targets)`:
    image_tensor : float32 Tensor[3, H, W], values in [0, 1]
    targets : {
        "image_id"   : str  (e.g. "MOT17-02-SDP/000301.jpg"),
        "seq_name"   : str,
        "frame_id"   : int,           # 1-indexed (MOT convention)
        "image_size" : Tensor[H, W],  # int64
        "bboxes"     : Tensor[N, 4]   float32 (cx, cy, w, h) normalized,
        "class_ids"  : Tensor[N]      int64,
        "track_ids"  : Tensor[N]      int64  (-1 for static),
        "visibility" : Tensor[N]      float32,
        "iscrowd"    : Tensor[N]      int64,
    }
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def _xywh_pixels_to_yolo(boxes: list[list[float]], W: int, H: int) -> torch.Tensor:
    """Convert N boxes from pixel (x_tl, y_tl, w, h) to normalized (cx, cy, w, h).

    Clips at the corner level (x1, y1, x2, y2) before re-encoding, so the
    final (cx, cy, w, h) box is fully inside [0, 1]^4. Needed because MOT
    and CrowdHuman both contain boxes that extend past image bounds —
    occluded full-body annotations.
    """
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    t = torch.tensor(boxes, dtype=torch.float32)
    x, y, w, h = t.unbind(dim=1)
    x1 = x.clamp(min=0.0, max=float(W))
    y1 = y.clamp(min=0.0, max=float(H))
    x2 = (x + w).clamp(min=0.0, max=float(W))
    y2 = (y + h).clamp(min=0.0, max=float(H))
    wc = (x2 - x1).clamp(min=0.0)
    hc = (y2 - y1).clamp(min=0.0)
    cx = (x1 + wc / 2) / W
    cy = (y1 + hc / 2) / H
    return torch.stack([cx, cy, wc / W, hc / H], dim=1)


class StandardSeqDataset(Dataset):
    """Flat per-frame index over a standard-format dataset split."""

    def __init__(self, root: str | Path, split: str, *, image_dtype=torch.float32):
        self.root = Path(root)
        self.split = split
        self.image_dtype = image_dtype

        anno_dir = self.root / "annotations" / split
        if not anno_dir.is_dir():
            raise FileNotFoundError(f"missing split: {anno_dir}")

        self._seqs: list[dict] = []          # parsed seq JSONs
        self._index: list[tuple[int, int]] = []  # (seq_idx, frame_idx)
        for json_path in sorted(anno_dir.glob("*.json")):
            with open(json_path) as f:
                seq = json.load(f)
            seq_idx = len(self._seqs)
            self._seqs.append(seq)
            for frame_idx in range(len(seq["frames"])):
                self._index.append((seq_idx, frame_idx))

        if not self._index:
            raise RuntimeError(f"no frames found in {anno_dir}")

    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_sequences(self) -> int:
        return len(self._seqs)

    def seq_names(self) -> list[str]:
        return [s["name"] for s in self._seqs]

    def _frame_image_size(self, seq: dict, frame: dict) -> tuple[int, int]:
        """Return (W, H) for this frame. Honors per-frame override for
        variable-resolution datasets like CrowdHuman."""
        if "image_size" in frame:
            W, H = frame["image_size"]
        else:
            W, H = seq["image_size"]
        return W, H

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        seq_idx, frame_idx = self._index[idx]
        seq = self._seqs[seq_idx]
        frame = seq["frames"][frame_idx]
        W, H = self._frame_image_size(seq, frame)

        img_path = self.root / "images" / self.split / frame["image"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            image_tensor = pil_to_tensor(im).to(self.image_dtype) / 255.0

        objs = frame["objects"]
        bboxes_xywh = [o["bbox_xywh"] for o in objs]
        bboxes = _xywh_pixels_to_yolo(bboxes_xywh, W, H)
        class_ids = torch.tensor([o["class_id"] for o in objs], dtype=torch.int64)
        track_ids = torch.tensor([o["track_id"] for o in objs], dtype=torch.int64)
        visibility = torch.tensor(
            [o.get("visibility", 1.0) for o in objs], dtype=torch.float32)
        iscrowd = torch.tensor(
            [o.get("iscrowd", 0) for o in objs], dtype=torch.int64)

        targets = {
            "image_id": frame["image"],
            "seq_name": seq["name"],
            "frame_id": int(frame["frame_id"]),
            "image_size": torch.tensor([H, W], dtype=torch.int64),
            "bboxes": bboxes,
            "class_ids": class_ids,
            "track_ids": track_ids,
            "visibility": visibility,
            "iscrowd": iscrowd,
        }
        return image_tensor, targets

    def get_seq_meta(self, seq_idx: int) -> dict:
        """Return the parsed JSON for a given sequence (without `frames`)."""
        seq = self._seqs[seq_idx]
        return {k: v for k, v in seq.items() if k != "frames"}
