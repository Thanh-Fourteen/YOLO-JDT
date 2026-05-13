# Vendored from Ultralytics 8.4.48 (https://github.com/ultralytics/ultralytics)
# Sources:
#   ultralytics/utils/loss.py    (DFLoss, BboxLoss, v8DetectionLoss)
#   ultralytics/utils/tal.py     (TaskAlignedAssigner, bbox2dist)
#   ultralytics/utils/metrics.py (bbox_iou)
#   ultralytics/utils/ops.py     (xywh2xyxy, xyxy2xywh)
# Licensed under AGPL-3.0. Original copyright (c) Ultralytics.
# Modifications:
#   - Single-file vendor of all detection-loss dependencies (combined for 1 import).
#   - Stripped rotated/RLE/segment/pose/classify variants — detection only.
#   - Stripped CUDA-OOM CPU fallback and class_weights handling (we always train on GPU,
#     fixed class set per dataset).
#   - Replaced `model.model[-1]` / `model.args` coupling with explicit constructor
#     args (nc, reg_max, stride, hyp_box, hyp_cls, hyp_dfl) so the loss is detached
#     from any specific model wrapper. Adapter `decode_raw_outputs()` converts our
#     standalone YOLO11's per-level list outputs into the upstream dict format.
#   - Class names + forward semantics preserved verbatim — does not affect grad math.
"""Standalone YOLO11 detection loss (CIoU + DFL + BCE)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.ultralytics_extract.head import dist2bbox, make_anchors

__all__ = [
    "bbox_iou", "xywh2xyxy", "xyxy2xywh", "bbox2dist",
    "DFLoss", "BboxLoss", "TaskAlignedAssigner",
    "DetectionLoss", "decode_raw_outputs",
]


# --------------------- bbox utilities ---------------------------------

def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """(x_c, y_c, w, h) → (x1, y1, x2, y2). Last dim must be 4."""
    assert x.shape[-1] == 4
    y = x.new_empty(x.shape)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2) → (x_c, y_c, w, h)."""
    assert x.shape[-1] == 4
    y = x.new_empty(x.shape)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True,
             GIoU: bool = False, DIoU: bool = False, CIoU: bool = False,
             eps: float = 1e-7) -> torch.Tensor:
    """IoU / GIoU / DIoU / CIoU between box1 and box2. Last dim of both is 4."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor,
              reg_max: int | None = None) -> torch.Tensor:
    """xyxy → (l, t, r, b) distance encoding from anchor points."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)
    return dist


# --------------------- DFL + BboxLoss ---------------------------------

class DFLoss(nn.Module):
    """Distribution Focal Loss integral (Generalized Focal Loss)."""

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """CIoU + DFL bounding-box loss."""

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


# --------------------- TaskAlignedAssigner -----------------------------

class TaskAlignedAssigner(nn.Module):
    """Task-aligned label assignment (PPYOLOE-style).

    Pairs each ground-truth box with topk anchors whose alignment metric
    (`cls_score^alpha * iou^beta`) is highest. Returns target labels,
    boxes, and target scores per anchor.
    """

    def __init__(self, topk: int = 10, num_classes: int = 80,
                 alpha: float = 0.5, beta: float = 6.0,
                 stride: list[int] = (8, 16, 32), eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = list(stride)
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = self._select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self._select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self._get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self._select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def _get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na],
                               dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na],
                                  dtype=pd_scores.dtype, device=pd_scores.device)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def _select_topk_candidates(self, metrics, topk_mask=None):
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        count.masked_fill_(count > 1, 0)
        return count.to(metrics.dtype)

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64,
                                 device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_labels.clamp_(0)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    def _select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)
        bs, n_boxes, _ = gt_bboxes.shape
        n_anchors = xy_centers.shape[0]
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2) \
            .view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    def _select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype,
                                          device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos


# --------------------- Adapter + DetectionLoss -------------------------

def decode_raw_outputs(raw_list: list[torch.Tensor], nc: int, reg_max: int) -> dict:
    """Convert YOLO11 head's per-level training outputs (list of [B, 4*reg_max+nc, H, W])
    into the dict format the v8 detection loss consumes:
        {"boxes":  [B, 4*reg_max, total_anchors],
         "scores": [B, nc, total_anchors],
         "feats":  raw_list (kept for anchor generation)}
    """
    bs = raw_list[0].shape[0]
    no = 4 * reg_max + nc
    flat = torch.cat([x.view(bs, no, -1) for x in raw_list], dim=2)
    boxes, scores = flat.split((4 * reg_max, nc), dim=1)
    return {"boxes": boxes, "scores": scores, "feats": raw_list}


class DetectionLoss:
    """Standalone YOLO11 detection loss, decoupled from any model wrapper.

    Usage:
        loss_fn = DetectionLoss(nc=1, reg_max=16, stride=(8, 16, 32),
                                box=7.5, cls=0.5, dfl=1.5)
        # forward returns raw_list in train mode
        raw = model(image)
        preds = decode_raw_outputs(raw, nc=1, reg_max=16)
        # batch dict: {"batch_idx": [N], "cls": [N, 1], "bboxes": [N, 4] in normalized xywh}
        total_loss, components = loss_fn(preds, batch)
    """

    def __init__(self, nc: int, reg_max: int = 16,
                 stride: tuple[float, float, float] = (8.0, 16.0, 32.0),
                 box: float = 7.5, cls: float = 0.5, dfl: float = 1.5,
                 tal_topk: int = 10):
        self.nc = nc
        self.reg_max = reg_max
        self.stride = torch.tensor(stride, dtype=torch.float)
        self.no = nc + reg_max * 4
        self.use_dfl = reg_max > 1

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(reg_max)
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=nc, alpha=0.5, beta=6.0,
            stride=stride,
        )
        self.proj = torch.arange(reg_max, dtype=torch.float)
        self.gain = {"box": box, "cls": cls, "dfl": dfl}

    def to(self, device):
        self.proj = self.proj.to(device)
        self.stride = self.stride.to(device)
        return self

    def _preprocess_targets(self, batch_idx, cls, bboxes, batch_size, scale_tensor, device):
        """Pack flat (N, 6) batch targets into (B, max_n, 5) padded tensor."""
        targets = torch.cat((batch_idx.view(-1, 1), cls.view(-1, 1), bboxes), 1).to(device)
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=device)
        bidx = targets[:, 0].long()
        _, counts = bidx.unique(return_counts=True)
        out = torch.zeros(batch_size, int(counts.max()), ne - 1, device=device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        offsets.scatter_add_(0, bidx + 1, torch.ones_like(bidx))
        offsets = offsets.cumsum(0)
        within = torch.arange(nl, device=device) - offsets[bidx]
        out[bidx, within] = targets[:, 1:]
        # bboxes: normalized (cx, cy, w, h) → xyxy in pixels (scale_tensor is [W, H, W, H])
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul(scale_tensor))
        return out

    def _bbox_decode(self, anchor_points, pred_dist):
        """DFL integral → bbox xyxy (in stride units, i.e. anchor space)."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3) \
                .matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scaled_total_loss * batch_size, [box, cls, dfl] detached)."""
        device = preds["scores"].device
        if self.proj.device != device:
            self.to(device)

        loss = torch.zeros(3, device=device)  # box, cls, dfl
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        bs = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=device, dtype=dtype) * self.stride[0]

        # Targets: batch dict has flat (N, ...) layout
        targets = self._preprocess_targets(
            batch["batch_idx"], batch["cls"], batch["bboxes"], bs,
            scale_tensor=imgsz[[1, 0, 1, 0]], device=device,
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
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
        return loss.sum() * bs, loss.detach()
