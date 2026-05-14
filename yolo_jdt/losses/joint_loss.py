"""Joint detection + ReID loss for Phase 4 (YOLO11-JDE replication).

Wraps the existing `DetectionLoss` from `third_party/ultralytics_extract/loss.py`
and adds a ReID head + cross-entropy loss over global track IDs.

Why a subclass instead of composition: TaskAlignedAssigner returns
`(target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx)`
but the parent `DetectionLoss.__call__` discards `target_gt_idx` (the
positive-anchor → GT-instance mapping the ReID loss needs). Re-running
the assigner just to recover this would duplicate ~50% of the per-step
compute. The cleanest fix is to override `__call__`, mirroring the
parent's body, capturing `target_gt_idx`, then doing the ReID work in the
same forward pass.

The ReID head is an `nn.Linear(reid_dim → num_track_ids)` classifier
trained with CE. Anchors with `track_id < 0` (CrowdHuman static, or
instances not in the global ID map) are excluded via `ignore_index=-1`.

Loss math:
    total = det_loss + lambda_reid * reid_loss

`lambda_reid` defaults to 0.1 per user spec — weight at the same order as
`cls_gain` so neither dominates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.ultralytics_extract.loss import (DetectionLoss, dist2bbox,
                                                    make_anchors, xywh2xyxy)

__all__ = ["JointDetectionReIDLoss"]


class JointDetectionReIDLoss(DetectionLoss, nn.Module):
    """DetectionLoss + ReID classifier in one __call__ pass.

    Inherits from `nn.Module` so the classifier head is registered as a
    sub-module (parameters() / state_dict() / .to(device) work as expected).
    """

    def __init__(self, nc: int, reg_max: int = 16,
                 stride: tuple[float, float, float] = (8.0, 16.0, 32.0),
                 box: float = 7.5, cls: float = 0.5, dfl: float = 1.5,
                 tal_topk: int = 10,
                 reid_dim: int = 128, num_track_ids: int = 359,
                 lambda_reid: float = 0.1):
        nn.Module.__init__(self)
        DetectionLoss.__init__(self, nc=nc, reg_max=reg_max, stride=stride,
                                box=box, cls=cls, dfl=dfl, tal_topk=tal_topk)
        self.reid_dim = reid_dim
        self.num_track_ids = num_track_ids
        self.lambda_reid = lambda_reid
        self.classifier = nn.Linear(reid_dim, num_track_ids)
        # CE with ignore_index=-1 → CrowdHuman/static instances contribute zero
        # gradient to the ReID branch.
        self.reid_ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    def to(self, device):
        DetectionLoss.to(self, device)
        nn.Module.to(self, device)
        return self

    def _pack_track_ids(self, batch_idx: torch.Tensor, track_ids: torch.Tensor,
                         batch_size: int, max_n: int, device) -> torch.Tensor:
        """Pack flat (N,) track_ids into padded (B, max_n) tensor — same packing
        scheme as `_preprocess_targets` so indices align with `target_gt_idx`."""
        out = torch.full((batch_size, max_n), -1, dtype=torch.int64, device=device)
        if batch_idx.numel() == 0 or max_n == 0:
            return out
        bidx = batch_idx.long().to(device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        offsets.scatter_add_(0, bidx + 1, torch.ones_like(bidx))
        offsets = offsets.cumsum(0)
        within = torch.arange(batch_idx.numel(), device=device) - offsets[bidx]
        out[bidx, within] = track_ids.to(device).long()
        return out

    def __call__(self, preds: dict, batch: dict,
                 reid_per_level: list[torch.Tensor] | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run detection loss + (optionally) ReID loss.

        Args:
            preds: as produced by `decode_raw_outputs` (dict with boxes/scores/feats).
            batch: must include `batch_idx`, `cls`, `bboxes`. For ReID also `track_ids`.
            reid_per_level: list of [B, reid_dim, H, W] (output of JointHead.cv4).
                If None, behaves like plain DetectionLoss.

        Returns:
            (total_loss * batch_size, components[box, cls, dfl, reid] detached)
        """
        device = preds["scores"].device
        if self.proj.device != device:
            self.to(device)

        loss = torch.zeros(4, device=device)   # box, cls, dfl, reid
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        bs = pred_scores.shape[0]
        imgsz = (torch.tensor(preds["feats"][0].shape[2:], device=device, dtype=dtype)
                 * self.stride[0])

        targets = self._preprocess_targets(
            batch["batch_idx"], batch["cls"], batch["bboxes"], bs,
            scale_tensor=imgsz[[1, 0, 1, 0]], device=device,
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)

        # CAPTURE target_gt_idx (parent drops this).
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # cls — BCE
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # box + dfl
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum, fg_mask,
            )

        loss[0] *= self.gain["box"]
        loss[1] *= self.gain["cls"]
        loss[2] *= self.gain["dfl"]

        # ---- ReID head ----
        if reid_per_level is not None and "track_ids" in batch and fg_mask.sum() > 0:
            max_n = targets.shape[1]
            gt_track_ids = self._pack_track_ids(
                batch["batch_idx"], batch["track_ids"], bs, max_n, device)

            # Per-anchor target track_id via gather along GT-axis. fg_mask[b,a]
            # tells us if anchor `a` in batch `b` was matched at all.
            # gather requires same dtype; clamp -1 won't reach (we mask after).
            gather_idx = target_gt_idx.clamp(min=0)
            target_tids = torch.gather(gt_track_ids, 1, gather_idx)   # [B, A]
            valid = fg_mask & (target_tids >= 0)

            if valid.sum() > 0:
                # Flatten reid_per_level → [B, reid_dim, A] then permute
                reid_flat = torch.cat(
                    [r.view(bs, self.reid_dim, -1) for r in reid_per_level],
                    dim=2,
                )                                          # [B, reid_dim, A]
                reid_flat = reid_flat.permute(0, 2, 1).contiguous()  # [B, A, reid_dim]
                emb = reid_flat[valid]                     # [N_pos, reid_dim]
                target = target_tids[valid]                # [N_pos]
                logits = self.classifier(emb)              # [N_pos, num_track_ids]
                loss[3] = self.reid_ce(logits, target)

        loss[3] *= self.lambda_reid

        return loss.sum() * bs, loss.detach()
