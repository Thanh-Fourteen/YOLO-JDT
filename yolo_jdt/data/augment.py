"""Detection-training augmentations: mosaic, HSV jitter, horizontal flip,
random affine (scale/translate), letterbox.

Designed to wrap our `StandardSeqDataset` which yields
`(image_tensor [3,H,W] in [0,1], targets dict)`. The augmentation
pipeline operates on uint8 NumPy arrays internally (faster + cv2-friendly)
and re-tensors the output.

Conventions:
- Bbox format throughout: (cx, cy, w, h) normalized to [0, 1].
- Mosaic always renders to a `2 * imgsz`×`2 * imgsz` canvas, then a
  random affine downscales/crops to `imgsz`×`imgsz`. This matches
  Ultralytics' Mosaic + RandomPerspective sequence.
- Mosaic-close: when `mosaic_p == 0.0`, we skip mosaic and use a single
  image with letterbox + the same affine + flip + HSV. This is what gets
  enabled in the last 15% of epochs ("close mosaic" trick).
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------- low-level ops ---------------------------------

def hsv_jitter(im_bgr: np.ndarray, h: float = 0.015, s: float = 0.7, v: float = 0.4) -> np.ndarray:
    """In-place HSV jitter. Inputs/outputs uint8 BGR."""
    if h == s == v == 0:
        return im_bgr
    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV))
    dt = im_bgr.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dt)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dt)
    lut_val = np.clip(x * r[2], 0, 255).astype(dt)
    out_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)


def random_perspective(im: np.ndarray, boxes_xywhn: np.ndarray,
                       degrees: float = 0.0, translate: float = 0.1,
                       scale: float = 0.5, shear: float = 0.0,
                       perspective: float = 0.0,
                       border: tuple[int, int] = (0, 0)
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random affine (rotation+translate+scale+shear) to an image and its boxes.

    Args:
        im: uint8 HWC BGR image.
        boxes_xywhn: (N, 4) normalized (cx, cy, w, h) in [0, 1] of the input image.
        border: (top/bottom, left/right) negative crop after warp.

    Returns:
        (warped image, boxes_xywhn in the warped image's frame).
    """
    h0, w0 = im.shape[:2]
    h = h0 + 2 * border[0]
    w = w0 + 2 * border[1]

    # Center
    C = np.eye(3)
    C[0, 2] = -w0 / 2
    C[1, 2] = -h0 / 2

    # Rotation + scale (no rotation in default detect aug)
    R = np.eye(3)
    a = random.uniform(-degrees, degrees) if degrees else 0
    s_ = random.uniform(1 - scale, 1 + scale) if scale else 1.0
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s_)

    # Shear
    S = np.eye(3)
    S[0, 1] = np.tan(random.uniform(-shear, shear) * np.pi / 180) if shear else 0
    S[1, 0] = np.tan(random.uniform(-shear, shear) * np.pi / 180) if shear else 0

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w if translate else 0.5 * w
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h if translate else 0.5 * h

    M = T @ S @ R @ C
    if (M != np.eye(3)).any() or border != (0, 0):
        im = cv2.warpAffine(im, M[:2], dsize=(w, h),
                            borderValue=(114, 114, 114))

    # Transform boxes
    if boxes_xywhn is None or len(boxes_xywhn) == 0:
        return im, np.zeros((0, 4), dtype=np.float32)

    cx, cy, bw, bh = boxes_xywhn[:, 0] * w0, boxes_xywhn[:, 1] * h0, \
                     boxes_xywhn[:, 2] * w0, boxes_xywhn[:, 3] * h0
    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    # 4 corners per box, transform via M
    corners = np.stack([
        np.stack([x1, y1, np.ones_like(x1)], 1),
        np.stack([x2, y1, np.ones_like(x1)], 1),
        np.stack([x2, y2, np.ones_like(x1)], 1),
        np.stack([x1, y2, np.ones_like(x1)], 1),
    ], axis=1).reshape(-1, 3)
    corners = corners @ M.T
    corners = corners[:, :2].reshape(-1, 4, 2)
    new_x1 = corners[:, :, 0].min(1)
    new_y1 = corners[:, :, 1].min(1)
    new_x2 = corners[:, :, 0].max(1)
    new_y2 = corners[:, :, 1].max(1)

    # Clip to canvas
    new_x1 = np.clip(new_x1, 0, w)
    new_x2 = np.clip(new_x2, 0, w)
    new_y1 = np.clip(new_y1, 0, h)
    new_y2 = np.clip(new_y2, 0, h)

    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1

    # Filter degenerate (too small or too thin after transform)
    valid = (new_w > 2) & (new_h > 2) & ((new_w * new_h) > 0.1 * (bw * bh + 1e-6))
    new_x1, new_y1, new_w, new_h = new_x1[valid], new_y1[valid], new_w[valid], new_h[valid]
    out = np.stack([
        (new_x1 + new_w / 2) / w,
        (new_y1 + new_h / 2) / h,
        new_w / w,
        new_h / h,
    ], axis=1).astype(np.float32)
    return im, out, valid


def horizontal_flip(im: np.ndarray, boxes_xywhn: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Flip image left-right + bboxes."""
    im = im[:, ::-1, :].copy()
    if boxes_xywhn is not None and len(boxes_xywhn):
        boxes_xywhn = boxes_xywhn.copy()
        boxes_xywhn[:, 0] = 1.0 - boxes_xywhn[:, 0]
    return im, boxes_xywhn


def letterbox(im: np.ndarray, new_shape: int = 640,
              color: tuple = (114, 114, 114), scaleup: bool = True
              ) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Pad-to-square letterbox. Returns (canvas, scale_ratio, (pad_left, pad_top))."""
    h0, w0 = im.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    if not scaleup:
        r = min(r, 1.0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    pad_w = new_shape - new_w
    pad_h = new_shape - new_h
    pad_l = pad_w // 2
    pad_t = pad_h // 2
    if (h0, w0) != (new_h, new_w):
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = im
    return canvas, r, (pad_l, pad_t)


# --------------------- Mosaic + augmentation Dataset ----------------

class DetTrainDataset(Dataset):
    """Wraps a base `StandardSeqDataset` to apply detection-training augs.

    `__getitem__` returns:
        image_tensor : float32 [3, imgsz, imgsz] in [0, 1] (RGB)
        targets dict : {
            "bboxes":   float32 [N, 4]  (cx, cy, w, h) normalized in [0, 1]
            "class_ids":int64   [N]
            "extra":    extra metadata (track_ids/visibility/iscrowd dropped here —
                        not needed for detection fine-tune; collated separately
                        if a downstream caller wants them)
        }
    """

    def __init__(self, base: Dataset, imgsz: int = 640,
                 mosaic_p: float = 1.0, hsv: tuple = (0.015, 0.7, 0.4),
                 flip_p: float = 0.5, scale: float = 0.5, translate: float = 0.1,
                 person_only: bool = True):
        self.base = base
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
        """Toggle mosaic probability — used by 'close-mosaic' callback."""
        self.mosaic_p = float(p)

    def _load_one(self, idx: int):
        """Load one base sample and return uint8 BGR image + xywhn-normalized boxes."""
        img_t, tgt = self.base[idx]
        # img_t is float32 [3, H, W] in [0, 1] RGB. Convert to uint8 BGR HWC.
        im = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1].copy()
        boxes = tgt["bboxes"].numpy().astype(np.float32)  # already (cx,cy,w,h) in [0,1]
        cls = tgt["class_ids"].numpy().astype(np.int64)
        if self.person_only:
            mask = cls == 0
            boxes = boxes[mask]
            cls = cls[mask]
        return im, boxes, cls

    def _mosaic(self, idx: int):
        """4-image mosaic at canvas size 2*imgsz, then crop back to imgsz via affine."""
        s = self.imgsz
        # Random center for the 4-tile cut, in [0.5*s, 1.5*s] (Ultralytics default)
        cx = int(random.uniform(s * 0.5, s * 1.5))
        cy = int(random.uniform(s * 0.5, s * 1.5))
        idxs = [idx] + [random.randrange(self._n) for _ in range(3)]

        canvas = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        all_boxes = []
        all_cls = []
        for tile_i, sidx in enumerate(idxs):
            im, boxes, cls = self._load_one(sidx)
            h, w = im.shape[:2]
            # Letterbox each tile to imgsz preserving aspect — keeps small datasets balanced
            r = min(s / h, s / w)
            new_w, new_h = int(round(w * r)), int(round(h * r))
            if (h, w) != (new_h, new_w):
                im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Where to paste in canvas (4 quadrants offset from (cx,cy))
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
            # Boxes for this tile: convert from xywhn-of-tile to xywhn-of-canvas
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

        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            cls = np.concatenate(all_cls, axis=0)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.int64)

        # Random affine to bring 2*imgsz canvas back to imgsz
        canvas, boxes, valid = random_perspective(
            canvas, boxes, scale=self.scale, translate=self.translate,
            border=(-s // 2, -s // 2),
        )
        cls = cls[valid]
        return canvas, boxes, cls

    def _no_mosaic(self, idx: int):
        """Single-image augmentation (no mosaic) — used in close-mosaic phase."""
        im, boxes, cls = self._load_one(idx)
        # Letterbox to imgsz
        canvas, r, (pl, pt) = letterbox(im, new_shape=self.imgsz, scaleup=True)
        # Convert boxes from xywhn-of-original to xywhn-of-canvas
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
        # Random affine
        canvas, boxes, valid = random_perspective(
            canvas, boxes, scale=self.scale * 0.5, translate=self.translate * 0.5,
            border=(0, 0),
        )
        cls = cls[valid]
        return canvas, boxes, cls

    def __getitem__(self, idx: int):
        if random.random() < self.mosaic_p:
            im, boxes, cls = self._mosaic(idx)
        else:
            im, boxes, cls = self._no_mosaic(idx)

        # HSV
        im = hsv_jitter(im, *self.hsv)
        # Horizontal flip
        if random.random() < self.flip_p:
            im, boxes = horizontal_flip(im, boxes)

        # BGR uint8 HWC → RGB float32 CHW [0,1]
        im_rgb = im[:, :, ::-1].copy()
        img_t = torch.from_numpy(im_rgb).permute(2, 0, 1).contiguous().float().div_(255.0)
        return img_t, {
            "bboxes": torch.from_numpy(boxes).float(),
            "class_ids": torch.from_numpy(cls).long(),
        }


class DetValDataset(Dataset):
    """Validation-only wrapper: letterbox to imgsz, no augmentations."""

    def __init__(self, base: Dataset, imgsz: int = 640, person_only: bool = True):
        self.base = base
        self.imgsz = imgsz
        self.person_only = person_only

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img_t, tgt = self.base[idx]
        im = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1].copy()
        boxes = tgt["bboxes"].numpy().astype(np.float32)
        cls = tgt["class_ids"].numpy().astype(np.int64)
        if self.person_only:
            mask = cls == 0
            boxes = boxes[mask]
            cls = cls[mask]
        h0, w0 = im.shape[:2]
        canvas, r, (pl, pt) = letterbox(im, new_shape=self.imgsz, scaleup=True)
        if len(boxes):
            cx_pix = boxes[:, 0] * w0 * r + pl
            cy_pix = boxes[:, 1] * h0 * r + pt
            w_pix = boxes[:, 2] * w0 * r
            h_pix = boxes[:, 3] * h0 * r
            boxes = np.stack([
                cx_pix / self.imgsz, cy_pix / self.imgsz,
                w_pix / self.imgsz, h_pix / self.imgsz,
            ], axis=1).astype(np.float32)
        im_rgb = canvas[:, :, ::-1].copy()
        img_t = torch.from_numpy(im_rgb).permute(2, 0, 1).contiguous().float().div_(255.0)
        return img_t, {
            "bboxes": torch.from_numpy(boxes).float(),
            "class_ids": torch.from_numpy(cls).long(),
            "image_id": tgt.get("image_id", ""),
            "ori_size": torch.tensor([h0, w0], dtype=torch.int64),
            "ratio_pad": torch.tensor([r, pl, pt], dtype=torch.float32),
        }


def collate_det(batch: list[tuple[torch.Tensor, dict]]) -> dict:
    """Collate function: stack images, flatten targets with batch_idx column.

    Returns dict consumable by DetectionLoss:
        {"img": [B, 3, H, W],
         "batch_idx": [N_total],
         "cls":       [N_total],     int64
         "bboxes":    [N_total, 4]   normalized (cx, cy, w, h)}
    """
    imgs = torch.stack([b[0] for b in batch], dim=0)
    batch_idx, cls, bboxes = [], [], []
    extras = []
    for i, (_, tgt) in enumerate(batch):
        n = tgt["bboxes"].shape[0]
        if n:
            batch_idx.append(torch.full((n,), i, dtype=torch.int64))
            cls.append(tgt["class_ids"])
            bboxes.append(tgt["bboxes"])
        extras.append({k: v for k, v in tgt.items() if k not in ("bboxes", "class_ids")})
    if batch_idx:
        batch_idx = torch.cat(batch_idx, 0)
        cls = torch.cat(cls, 0)
        bboxes = torch.cat(bboxes, 0)
    else:
        batch_idx = torch.zeros((0,), dtype=torch.int64)
        cls = torch.zeros((0,), dtype=torch.int64)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
    return {
        "img": imgs,
        "batch_idx": batch_idx,
        "cls": cls.unsqueeze(-1).float(),
        "bboxes": bboxes,
        "extras": extras,
    }
