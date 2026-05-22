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
    total = det_loss + lambda_reid * reid_loss + lambda_offset * offset_loss

`lambda_reid` defaults to 0.1 per user spec — weight at the same order as
`cls_gain` so neither dominates.

Step 5.DE pivot (2026-05-22): added an optional track-offset term — SmoothL1 on
per-anchor inter-frame centre displacement (TrackOffsetHead output). With
`offset_only=True` the detection + ReID terms are skipped entirely (their
branches are frozen at JDE quality) and only the offset head is supervised; the
TaskAlignedAssigner still runs to select the positive anchors.
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
                 lambda_reid: float = 0.1, lambda_offset: float = 1.0):
        nn.Module.__init__(self)
        DetectionLoss.__init__(self, nc=nc, reg_max=reg_max, stride=stride,
                                box=box, cls=cls, dfl=dfl, tal_topk=tal_topk)
        self.reid_dim = reid_dim
        self.num_track_ids = num_track_ids
        self.lambda_reid = lambda_reid
        self.lambda_offset = lambda_offset
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

    def _pack_offsets(self, batch_idx: torch.Tensor, offsets: torch.Tensor,
                      valid: torch.Tensor, batch_size: int, max_n: int, device
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack flat (N,2) offsets + (N,) valid mask into padded (B,max_n,2) /
        (B,max_n) tensors — same packing scheme as `_pack_track_ids`."""
        out = torch.zeros((batch_size, max_n, 2), dtype=torch.float32, device=device)
        out_v = torch.zeros((batch_size, max_n), dtype=torch.bool, device=device)
        if batch_idx.numel() == 0 or max_n == 0:
            return out, out_v
        bidx = batch_idx.long().to(device)
        cnt = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        cnt.scatter_add_(0, bidx + 1, torch.ones_like(bidx))
        cnt = cnt.cumsum(0)
        within = torch.arange(batch_idx.numel(), device=device) - cnt[bidx]
        out[bidx, within] = offsets.to(device).float()
        out_v[bidx, within] = valid.to(device).bool()
        return out, out_v

    def __call__(self, preds: dict, batch: dict,
                 reid_per_level: list[torch.Tensor] | None = None,
                 offset_per_level: list[torch.Tensor] | None = None,
                 offset_only: bool = False
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run detection loss + (optionally) ReID loss + (optionally) offset loss.

        Args:
            preds: as produced by `decode_raw_outputs` (dict with boxes/scores/feats).
            batch: must include `batch_idx`, `cls`, `bboxes`. For ReID also
                `track_ids`; for offset also `offsets` (N,2) and `offset_valid` (N,).
            reid_per_level:   list of [B, reid_dim, H, W] (JointHead.cv4 output).
            offset_per_level: list of [B, 2, H, W] (TrackOffsetHead output).
            offset_only: Step 5.DE pivot — when True, ONLY the offset loss is
                computed (detection + ReID branches are frozen so their loss
                terms would produce no trainable gradient). The assigner still
                runs to pick positive anchors. Skips box/cls/dfl/reid compute.

        Returns:
            (total_loss * batch_size, components[box, cls, dfl, reid, offset] detached)
        """
        device = preds["scores"].device
        if self.proj.device != device:
            self.to(device)

        loss = torch.zeros(5, device=device)   # box, cls, dfl, reid, offset
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

        # CAPTURE target_gt_idx (parent drops this) — needed by ReID + offset.
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        # ---- Detection + ReID losses (skipped in offset_only pivot) ----------
        if not offset_only:
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
                gather_idx = target_gt_idx.clamp(min=0)
                target_tids = torch.gather(gt_track_ids, 1, gather_idx)   # [B, A]
                valid = fg_mask & (target_tids >= 0)

                if valid.sum() > 0:
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

        # ---- Track-offset head (Step 5.DE pivot) -----------------------------
        # SmoothL1 between predicted and GT centre displacement, at positive
        # anchors whose GT box has a t-1 correspondence. Computed in pixel units
        # (×imgsz) so SmoothL1 operates in its linear regime — small normalized
        # offsets in the quadratic regime would yield vanishing gradients.
        if offset_per_level is not None and "offsets" in batch and fg_mask.sum() > 0:
            max_n = targets.shape[1]
            gt_off, gt_off_valid = self._pack_offsets(
                batch["batch_idx"], batch["offsets"], batch["offset_valid"],
                bs, max_n, device)
            gather_idx = target_gt_idx.clamp(min=0)
            tgt_off = torch.gather(
                gt_off, 1, gather_idx.unsqueeze(-1).expand(-1, -1, 2))  # [B, A, 2]
            tgt_valid = torch.gather(gt_off_valid, 1, gather_idx)        # [B, A]
            valid = fg_mask.bool() & tgt_valid

            if valid.sum() > 0:
                off_flat = torch.cat(
                    [o.reshape(bs, 2, -1) for o in offset_per_level], dim=2)
                off_flat = off_flat.permute(0, 2, 1).contiguous()        # [B, A, 2]
                pred_o = off_flat[valid].float()
                tgt_o = tgt_off[valid].float()
                scale = float(imgsz[0])
                loss[4] = F.smooth_l1_loss(pred_o * scale, tgt_o * scale,
                                            reduction="mean")

        loss[4] *= self.lambda_offset

        return loss.sum() * bs, loss.detach()
