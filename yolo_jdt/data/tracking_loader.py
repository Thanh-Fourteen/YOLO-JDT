"""Paired-frame dataset for TAGate temporal training.

Yields (frame_t, frame_{t-1}) pairs from the same sequence with a shared
augmentation random state so spatial transforms are temporally coherent:
both frames get IDENTICAL letterbox + affine + flip + HSV jitter.

Design constraints:
- No mosaic: mosaic tiles come from different frames/sequences and would
  break the temporal relationship between paired frames.
- Same-seed augmentation: the caller saves Python + NumPy RNG state before
  augmenting frame_t, then restores it to augment frame_{t-1} identically.
  This preserves relative object positions across the pair while still
  applying random augmentation for training diversity.
- First-frame handling: for the first frame of each sequence, frame_prev =
  frame_t (same frame repeated). This is a "zero-motion" pseudo-pair that
  trains the model to output a near-identity gate when there is no real
  temporal displacement.

Return format per item (flat dict, use collate_paired for batching):
    {
        "img_t":          Tensor[3, H, W]  float32 RGB in [0,1]
        "img_prev":       Tensor[3, H, W]  float32 RGB in [0,1]
        "bboxes_t":       Tensor[N,  4]    normalized (cx,cy,w,h)
        "cls_t":          Tensor[N]        int64 class ids
        "track_ids_t":    Tensor[N]        int64 local track ids (-1 if none)
        "bboxes_prev":    Tensor[M,  4]    same format for frame t-1
        "cls_prev":       Tensor[M]
        "track_ids_prev": Tensor[M]
        "seq_name":       str
        "frame_id_t":     int  (1-indexed MOT convention)
        "frame_id_prev":  int  (frame_id_t - 1, or same at sequence start)
    }
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from yolo_jdt.data.augment import (
    horizontal_flip,
    hsv_jitter,
    letterbox,
    random_perspective,
)
from yolo_jdt.data.datasets._base import StandardSeqDataset

__all__ = ["PairedFrameDataset", "collate_paired"]


class PairedFrameDataset(Dataset):
    """Yields (frame_t, frame_{t-1}) pairs with shared augmentation state.

    Args:
        base:        A StandardSeqDataset exposing (seq_idx, frame_idx) via _index.
        imgsz:       Output spatial resolution (default 640).
        hsv:         (h_gain, s_gain, v_gain) for HSV jitter.
        flip_p:      Probability of horizontal flip.
        scale:       Random scale range for affine augmentation.
        translate:   Random translate fraction for affine augmentation.
        person_only: Keep only class_id == 0 (person) boxes.
    """

    def __init__(
        self,
        base: StandardSeqDataset,
        imgsz: int = 640,
        hsv: tuple = (0.015, 0.7, 0.4),
        flip_p: float = 0.5,
        scale: float = 0.5,
        translate: float = 0.1,
        person_only: bool = True,
    ):
        self.base = base
        self.imgsz = imgsz
        self.hsv = hsv
        self.flip_p = flip_p
        self.scale = scale
        self.translate = translate
        self.person_only = person_only

        # Build pairing index: (flat_idx_t, flat_idx_prev)
        # Sequences appear consecutively in base._index (sorted JSON glob).
        # For frame_idx == 0, prev = same frame (zero-motion pair).
        self._pairs: list[tuple[int, int]] = []
        for flat_idx, (seq_idx, frame_idx) in enumerate(base._index):
            prev = flat_idx if frame_idx == 0 else flat_idx - 1
            self._pairs.append((flat_idx, prev))

    def __len__(self) -> int:
        return len(self._pairs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw(self, flat_idx: int):
        """Load one frame as uint8 BGR HWC + normalized boxes + metadata."""
        img_t, tgt = self.base[flat_idx]
        # float32 [3,H,W] RGB → uint8 HWC BGR
        im = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        im = im[:, :, ::-1].copy()  # RGB → BGR
        boxes = tgt["bboxes"].numpy().astype(np.float32)
        cls = tgt["class_ids"].numpy().astype(np.int64)
        track_ids = tgt["track_ids"].numpy().astype(np.int64)
        if self.person_only:
            mask = cls == 0
            boxes, cls, track_ids = boxes[mask], cls[mask], track_ids[mask]
        return im, boxes, cls, track_ids, tgt["seq_name"], int(tgt["frame_id"])

    def _apply_aug(
        self, im: np.ndarray, boxes: np.ndarray, cls: np.ndarray,
        track_ids: np.ndarray,
    ) -> tuple[Tensor, np.ndarray, np.ndarray, np.ndarray]:
        """Letterbox + affine + flip + HSV on a single frame.

        Uses the *current* Python / NumPy RNG state so two consecutive calls
        with the same state produce identical augmentation decisions.
        """
        h0, w0 = im.shape[:2]
        canvas, r, (pl, pt) = letterbox(im, new_shape=self.imgsz, scaleup=True)
        # Adjust boxes from [0,1]-of-original to [0,1]-of-canvas
        if len(boxes):
            cx_pix = boxes[:, 0] * w0 * r + pl
            cy_pix = boxes[:, 1] * h0 * r + pt
            w_pix = boxes[:, 2] * w0 * r
            h_pix = boxes[:, 3] * h0 * r
            boxes = np.stack([
                cx_pix / self.imgsz, cy_pix / self.imgsz,
                w_pix / self.imgsz,  h_pix / self.imgsz,
            ], axis=1).astype(np.float32)

        # Random affine (uses Python random + no NumPy internally)
        canvas, boxes, valid = random_perspective(
            canvas, boxes,
            scale=self.scale * 0.5, translate=self.translate * 0.5,
        )
        cls = cls[valid]
        track_ids = track_ids[valid]

        # HSV jitter (uses NumPy random)
        canvas = hsv_jitter(canvas, *self.hsv)

        # Horizontal flip (uses Python random)
        if random.random() < self.flip_p:
            canvas, boxes = horizontal_flip(canvas, boxes)

        # BGR uint8 → RGB float32 tensor
        img_tensor = (
            torch.from_numpy(canvas[:, :, ::-1].copy())
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        return img_tensor, boxes, cls, track_ids

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        flat_t, flat_prev = self._pairs[idx]

        im_t,    boxes_t,    cls_t,    tids_t,    seq_t,    fid_t    = self._load_raw(flat_t)
        im_prev, boxes_prev, cls_prev, tids_prev, seq_prev, fid_prev = self._load_raw(flat_prev)

        # Save RNG state; apply identical augmentation to both frames.
        py_state = random.getstate()
        np_state = np.random.get_state()

        img_t, boxes_t, cls_t, tids_t = self._apply_aug(im_t, boxes_t, cls_t, tids_t)

        random.setstate(py_state)
        np.random.set_state(np_state)

        img_prev, boxes_prev, cls_prev, tids_prev = self._apply_aug(
            im_prev, boxes_prev, cls_prev, tids_prev
        )

        return {
            "img_t":          img_t,
            "img_prev":       img_prev,
            "bboxes_t":       torch.from_numpy(boxes_t).float(),
            "cls_t":          torch.from_numpy(cls_t).long(),
            "track_ids_t":    torch.from_numpy(tids_t).long(),
            "bboxes_prev":    torch.from_numpy(boxes_prev).float(),
            "cls_prev":       torch.from_numpy(cls_prev).long(),
            "track_ids_prev": torch.from_numpy(tids_prev).long(),
            "seq_name":       seq_t,
            "frame_id_t":     fid_t,
            "frame_id_prev":  fid_prev,
        }


def collate_paired(batch: list[dict]) -> dict:
    """Collate a list of PairedFrameDataset items into a batch.

    Images are stacked; per-box arrays are concatenated with a batch_idx
    column (same convention as collate_det_with_ids).
    """
    imgs_t    = torch.stack([s["img_t"]    for s in batch])
    imgs_prev = torch.stack([s["img_prev"] for s in batch])

    def _cat_with_idx(key_boxes, key_cls, key_tids):
        batch_idx, boxes_list, cls_list, tids_list = [], [], [], []
        for i, s in enumerate(batch):
            n = len(s[key_boxes])
            batch_idx.append(torch.full((n,), i, dtype=torch.int64))
            boxes_list.append(s[key_boxes])
            cls_list.append(s[key_cls].unsqueeze(1).float())
            tids_list.append(s[key_tids])
        return (
            torch.cat(batch_idx),
            torch.cat(boxes_list),
            torch.cat(cls_list),
            torch.cat(tids_list),
        )

    bi_t,  bx_t,  cl_t,  ti_t  = _cat_with_idx("bboxes_t",  "cls_t",  "track_ids_t")
    bi_p,  bx_p,  cl_p,  ti_p  = _cat_with_idx("bboxes_prev", "cls_prev", "track_ids_prev")

    return {
        "img_t":           imgs_t,
        "img_prev":        imgs_prev,
        # frame t
        "batch_idx_t":     bi_t,
        "bboxes_t":        bx_t,
        "cls_t":           cl_t,
        "track_ids_t":     ti_t,
        # frame t-1
        "batch_idx_prev":  bi_p,
        "bboxes_prev":     bx_p,
        "cls_prev":        cl_p,
        "track_ids_prev":  ti_p,
        # metadata
        "seq_names":       [s["seq_name"]    for s in batch],
        "frame_ids_t":     [s["frame_id_t"]  for s in batch],
        "frame_ids_prev":  [s["frame_id_prev"] for s in batch],
    }
