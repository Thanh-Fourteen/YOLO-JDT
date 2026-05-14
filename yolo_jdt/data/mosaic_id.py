"""ID-preserving mosaic + dataset wrapper for ReID training.

Step 4 (Phase 4) requires per-instance global track IDs to flow from the
StandardSeqDataset (which yields `track_ids`) all the way into the loss
function (which assigns each positive anchor a target instance). The
existing `DetTrainDataset` in `yolo_jdt/data/augment.py` drops `track_ids`
because Step 3.A only needed boxes + class.

This module:
1. `build_global_id_map(...)` — scan a dataset (standard format) and assign
   each unique `(seq_name, track_id)` pair a contiguous global ID
   `[0, num_classes)`. CrowdHuman-style static instances (`track_id < 0`)
   are mapped to -1 (sentinel for "no ReID label, mask this anchor out
   of ReID loss").

2. `DetTrainIDDataset` — same augmentation pipeline as `DetTrainDataset`
   but each bbox carries its global track ID through mosaic + perspective.

3. `collate_det_with_ids` — extends `collate_det` to include `track_ids`.

The collated batch dict format (consumed by JDELitModule):

    {
        "img":        [B, 3, H, W],
        "batch_idx":  [N_total],
        "cls":        [N_total, 1],   float
        "bboxes":     [N_total, 4],   normalized cx,cy,w,h
        "track_ids":  [N_total],      int64; -1 = no ReID label
        "extras":     [...],
    }
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from yolo_jdt.data.augment import (hsv_jitter, horizontal_flip, letterbox,
                                     random_perspective)


def build_global_id_map(standard_root: str | Path, dataset_name: str,
                        split: str, class_filter: int = 0) -> dict[tuple[str, int], int]:
    """Scan the JSONs under `<standard_root>/<dataset>/annotations/<split>/`
    and assign each unique `(seq_name, track_id)` a global integer ID
    starting at 0. Negative `track_id` (static, e.g. CrowdHuman) is excluded.

    Returns:
        dict mapping `(seq_name, track_id) → global_id ∈ [0, num_classes)`.
    """
    anno_dir = Path(standard_root) / dataset_name / "annotations" / split
    if not anno_dir.is_dir():
        raise FileNotFoundError(f"missing split: {anno_dir}")

    pairs: set[tuple[str, int]] = set()
    for json_path in sorted(anno_dir.glob("*.json")):
        with open(json_path) as f:
            seq = json.load(f)
        seq_name = seq["name"]
        for frame in seq["frames"]:
            for obj in frame["objects"]:
                if obj.get("class_id", 0) != class_filter:
                    continue
                tid = int(obj["track_id"])
                if tid < 0:
                    continue
                pairs.add((seq_name, tid))

    return {p: i for i, p in enumerate(sorted(pairs))}


class DetTrainIDDataset(Dataset):
    """ID-preserving variant of `DetTrainDataset`.

    Differences from the parent (`yolo_jdt/data/augment.py:DetTrainDataset`):
    - `_load_one` returns a 4th value, `track_ids`, mapped through
      `global_id_map` (or set to -1 if not in the map).
    - `_mosaic` and `_no_mosaic` propagate `track_ids` alongside `boxes`/`cls`
      through the perspective filter.
    - `__getitem__` returns a `track_ids` tensor in the targets dict.

    Behavioral parity: same RNG sequencing as DetTrainDataset so the
    mosaic + augmentations do not silently diverge from Step 3.A.

    Args:
        base: a `StandardSeqDataset` (yields `track_ids` in targets).
        global_id_map: from `build_global_id_map(...)`. Instances not in the
            map (different split, CrowdHuman static, etc.) get track_id = -1.
        person_only: keep only `class_id == 0`.
    """

    def __init__(self, base: Dataset,
                 global_id_map: dict[tuple[str, int], int],
                 imgsz: int = 640, mosaic_p: float = 1.0,
                 hsv: tuple = (0.015, 0.7, 0.4),
                 flip_p: float = 0.5, scale: float = 0.5, translate: float = 0.1,
                 person_only: bool = True):
        self.base = base
        self.global_id_map = global_id_map
        self.imgsz = imgsz
        self.mosaic_p = mosaic_p
        self.hsv = hsv
        self.flip_p = flip_p
        self.scale = scale
        self.translate = translate
        self.person_only = person_only
        self._n = len(self.base)

    def __len__(self) -> int:
        return self._n

    def set_mosaic_p(self, p: float):
        self.mosaic_p = float(p)

    def _load_one(self, idx: int):
        img_t, tgt = self.base[idx]
        im = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1].copy()
        boxes = tgt["bboxes"].numpy().astype(np.float32)
        cls = tgt["class_ids"].numpy().astype(np.int64)
        track_ids_local = tgt["track_ids"].numpy().astype(np.int64)
        seq_name = tgt.get("seq_name", "")

        if self.person_only:
            mask = cls == 0
            boxes = boxes[mask]
            cls = cls[mask]
            track_ids_local = track_ids_local[mask]

        # Map (seq, local_tid) → global_id; -1 if not in map (static / out-of-split).
        gid = np.full_like(track_ids_local, -1, dtype=np.int64)
        for i, t in enumerate(track_ids_local):
            if t < 0:
                continue
            gid[i] = self.global_id_map.get((seq_name, int(t)), -1)
        return im, boxes, cls, gid

    def _mosaic(self, idx: int):
        s = self.imgsz
        cx = int(random.uniform(s * 0.5, s * 1.5))
        cy = int(random.uniform(s * 0.5, s * 1.5))
        idxs = [idx] + [random.randrange(self._n) for _ in range(3)]
        canvas = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        all_boxes, all_cls, all_gid = [], [], []
        for tile_i, sidx in enumerate(idxs):
            im, boxes, cls, gid = self._load_one(sidx)
            h, w = im.shape[:2]
            r = min(s / h, s / w)
            new_w, new_h = int(round(w * r)), int(round(h * r))
            if (h, w) != (new_h, new_w):
                im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if tile_i == 0:
                x1a, y1a, x2a, y2a = max(cx - new_w, 0), max(cy - new_h, 0), cx, cy
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), new_h - (y2a - y1a), new_w, new_h
            elif tile_i == 1:
                x1a, y1a, x2a, y2a = cx, max(cy - new_h, 0), min(cx + new_w, s * 2), cy
                x1b, y1b, x2b, y2b = 0, new_h - (y2a - y1a), x2a - x1a, new_h
            elif tile_i == 2:
                x1a, y1a, x2a, y2a = max(cx - new_w, 0), cy, cx, min(s * 2, cy + new_h)
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), 0, new_w, y2a - y1a
            else:
                x1a, y1a, x2a, y2a = cx, cy, min(cx + new_w, s * 2), min(s * 2, cy + new_h)
                x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a
            canvas[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
            if len(boxes):
                tile_cx_canvas = (x1a + (x1b + x2b) / 2 - x1b) + (boxes[:, 0] - 0.5) * new_w
                tile_cy_canvas = (y1a + (y1b + y2b) / 2 - y1b) + (boxes[:, 1] - 0.5) * new_h
                tile_w_canvas = boxes[:, 2] * new_w
                tile_h_canvas = boxes[:, 3] * new_h
                all_boxes.append(np.stack([
                    tile_cx_canvas / (s * 2),
                    tile_cy_canvas / (s * 2),
                    tile_w_canvas / (s * 2),
                    tile_h_canvas / (s * 2),
                ], axis=1).astype(np.float32))
                all_cls.append(cls)
                all_gid.append(gid)

        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            cls = np.concatenate(all_cls, axis=0)
            gid = np.concatenate(all_gid, axis=0)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.int64)
            gid = np.zeros((0,), dtype=np.int64)

        canvas, boxes, valid = random_perspective(
            canvas, boxes, scale=self.scale, translate=self.translate,
            border=(-s // 2, -s // 2),
        )
        cls = cls[valid]
        gid = gid[valid]
        return canvas, boxes, cls, gid

    def _no_mosaic(self, idx: int):
        im, boxes, cls, gid = self._load_one(idx)
        canvas, r, (pl, pt) = letterbox(im, new_shape=self.imgsz, scaleup=True)
        if len(boxes):
            h0, w0 = im.shape[:2]
            cx_pix = boxes[:, 0] * w0 * r + pl
            cy_pix = boxes[:, 1] * h0 * r + pt
            w_pix = boxes[:, 2] * w0 * r
            h_pix = boxes[:, 3] * h0 * r
            boxes = np.stack([
                cx_pix / self.imgsz, cy_pix / self.imgsz,
                w_pix / self.imgsz, h_pix / self.imgsz,
            ], axis=1).astype(np.float32)
        canvas, boxes, valid = random_perspective(
            canvas, boxes, scale=self.scale * 0.5, translate=self.translate * 0.5,
            border=(0, 0),
        )
        cls = cls[valid]
        gid = gid[valid]
        return canvas, boxes, cls, gid

    def __getitem__(self, idx: int):
        if random.random() < self.mosaic_p:
            im, boxes, cls, gid = self._mosaic(idx)
        else:
            im, boxes, cls, gid = self._no_mosaic(idx)

        im = hsv_jitter(im, *self.hsv)
        if random.random() < self.flip_p:
            im, boxes = horizontal_flip(im, boxes)

        im_rgb = im[:, :, ::-1].copy()
        img_t = torch.from_numpy(im_rgb).permute(2, 0, 1).contiguous().float().div_(255.0)
        return img_t, {
            "bboxes": torch.from_numpy(boxes).float(),
            "class_ids": torch.from_numpy(cls).long(),
            "track_ids": torch.from_numpy(gid).long(),
        }


def collate_det_with_ids(batch: list[tuple[torch.Tensor, dict]]) -> dict:
    """Collate function preserving `track_ids` alongside the standard fields."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    batch_idx, cls, bboxes, tids = [], [], [], []
    extras = []
    for i, (_, tgt) in enumerate(batch):
        n = tgt["bboxes"].shape[0]
        if n:
            batch_idx.append(torch.full((n,), i, dtype=torch.int64))
            cls.append(tgt["class_ids"])
            bboxes.append(tgt["bboxes"])
            tids.append(tgt.get("track_ids",
                                torch.full((n,), -1, dtype=torch.int64)))
        extras.append({k: v for k, v in tgt.items()
                       if k not in ("bboxes", "class_ids", "track_ids")})
    if batch_idx:
        batch_idx = torch.cat(batch_idx, 0)
        cls = torch.cat(cls, 0)
        bboxes = torch.cat(bboxes, 0)
        tids = torch.cat(tids, 0)
    else:
        batch_idx = torch.zeros((0,), dtype=torch.int64)
        cls = torch.zeros((0,), dtype=torch.int64)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
        tids = torch.zeros((0,), dtype=torch.int64)
    return {
        "img": imgs,
        "batch_idx": batch_idx,
        "cls": cls.unsqueeze(-1).float(),
        "bboxes": bboxes,
        "track_ids": tids,
        "extras": extras,
    }
