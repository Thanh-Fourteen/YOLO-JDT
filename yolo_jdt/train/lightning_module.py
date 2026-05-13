"""PyTorch Lightning module for YOLO11 detection fine-tune.

Conventions:
- BF16 mixed precision (set in trainer config, not here).
- EMA shadow weights, decay 0.9999, used for val + saved checkpoint.
- Channels-last memory format on the model + inputs (Hopper/Blackwell win).
- Sync BN auto-converted when DDP strategy is active.
- mAP tracked per val dataloader via torchmetrics.
"""
from __future__ import annotations

import copy
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms

from third_party.ultralytics_extract.loss import DetectionLoss, decode_raw_outputs
from yolo_jdt.models.yolo11 import YOLO11


class ModelEMA:
    """Exponential moving average of model weights. Standard YOLO recipe."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau   # warmup step constant; effective decay = decay*(1 - exp(-step/tau))
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1 - torch.exp(torch.tensor(-self.updates / self.tau)).item())
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def _multi_label_nms(decoded: torch.Tensor, conf_thr: float = 0.001,
                     iou_thr: float = 0.7, max_det: int = 300,
                     max_wh: int = 7680) -> list[dict]:
    """NMS post-process matching `scripts/eval_coco.py` semantics. Returns list of
    {"boxes":[N,4]xyxy normalized, "scores":[N], "labels":[N]} per batch image."""
    bs, ch, _ = decoded.shape
    nc = ch - 4
    out = []
    for i in range(bs):
        x = decoded[i].T  # [A, 4+nc]
        boxes_xywh = x[:, :4]
        cls_scores = x[:, 4:]
        keep = cls_scores.amax(dim=1) > conf_thr
        boxes_xywh = boxes_xywh[keep]
        cls_scores = cls_scores[keep]
        if boxes_xywh.numel() == 0:
            out.append({"boxes": boxes_xywh.new_zeros((0, 4)),
                        "scores": cls_scores.new_zeros((0,)),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=decoded.device)})
            continue
        cx, cy, w, h = boxes_xywh.unbind(1)
        boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        # multi-label
        if nc > 1:
            ai, cj = torch.where(cls_scores > conf_thr)
            xyxy = boxes_xyxy[ai]
            scores = cls_scores[ai, cj]
            classes = cj.long()
        else:
            scores = cls_scores[:, 0]
            xyxy = boxes_xyxy
            classes = torch.zeros_like(scores, dtype=torch.long)
        if scores.numel() == 0:
            out.append({"boxes": boxes_xywh.new_zeros((0, 4)),
                        "scores": cls_scores.new_zeros((0,)),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=decoded.device)})
            continue
        boxes_offset = xyxy + classes.float().unsqueeze(1) * max_wh
        kept = nms(boxes_offset, scores, iou_thr)[:max_det]
        out.append({"boxes": xyxy[kept], "scores": scores[kept], "labels": classes[kept]})
    return out


class DetLitModule(L.LightningModule):
    """Detection fine-tune for our standalone YOLO11.

    Args mirror Hydra config under `model:` and `optim:`.
    """

    def __init__(self,
                 scale: str = "s",
                 nc: int = 1,
                 reg_max: int = 16,
                 imgsz: int = 640,
                 # loss gains
                 box_gain: float = 7.5,
                 cls_gain: float = 0.5,
                 dfl_gain: float = 1.5,
                 # optim
                 lr0: float = 0.01,
                 lrf: float = 0.01,             # final LR factor (cosine end)
                 momentum: float = 0.937,
                 weight_decay: float = 5e-4,
                 warmup_epochs: float = 3.0,
                 warmup_momentum: float = 0.8,
                 warmup_bias_lr: float = 0.1,
                 # EMA
                 ema_decay: float = 0.9999,
                 ema_tau: float = 2000,
                 # init weights — path to Ultralytics .pt for fine-tune; None = train from scratch
                 pretrained_weights: str | None = None,
                 channels_last: bool = True):
        super().__init__()
        self.save_hyperparameters()

        self.model = YOLO11(scale=scale, nc=nc, reg_max=reg_max,
                            strides=(8.0, 16.0, 32.0))
        if pretrained_weights:
            from yolo_jdt.weights.loader import load_yolo11_weights
            self._load_pretrained_filter_nc(pretrained_weights)

        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.loss_fn = DetectionLoss(
            nc=nc, reg_max=reg_max, stride=(8.0, 16.0, 32.0),
            box=box_gain, cls=cls_gain, dfl=dfl_gain, tal_topk=10,
        )

        self.ema = None  # initialized in on_train_start (after possible DDP wrap)

        # Per-val-loader mAP. Lightning passes dataloader_idx; we need 2 metrics.
        self._val_map = nn.ModuleList([
            MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                 backend="pycocotools"),
            MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                 backend="pycocotools"),
        ])
        self._val_names = ["mot17_val_half", "crowdhuman_val"]

    def _load_pretrained_filter_nc(self, ckpt_path: str):
        """Load Ultralytics pretrained weights, dropping cls-head weights when
        fine-tuning to a different `nc` (80 → 1). Box/DFL/backbone stay."""
        from yolo_jdt.weights.loader import load_yolo11_weights, map_ultralytics_state_dict
        import torch
        ckpt = torch.load(ckpt_path, weights_only=False)
        src = ckpt["model"].float().state_dict()
        dst = self.model.state_dict()
        try:
            mapped = map_ultralytics_state_dict(src, dst)
            self.model.load_state_dict(mapped, strict=True)
            return
        except ValueError:
            pass
        # nc mismatch: drop cls-head final layers
        partial = {}
        for k, v in src.items():
            from yolo_jdt.weights.loader import _key_destination
            new_k = _key_destination(k)
            if new_k not in dst:
                continue
            if v.shape != dst[new_k].shape:
                # cls head final conv: model.23.cv3.X.2.{weight,bias} shape [80,...] vs [nc,...]
                continue
            partial[new_k] = v
        missing = self.model.load_state_dict(partial, strict=False)
        print(f"[DetLitModule] pretrained load: {len(partial)}/{len(dst)} keys, "
              f"missing {len(missing.missing_keys)}, unexpected {len(missing.unexpected_keys)}")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        if self.ema is None:
            self.ema = ModelEMA(self.model, decay=self.hparams.ema_decay,
                                tau=self.hparams.ema_tau)

    def training_step(self, batch: dict, batch_idx: int):
        img = batch["img"]
        if self.hparams.channels_last:
            img = img.to(memory_format=torch.channels_last)
        raw = self.model(img)
        preds = decode_raw_outputs(raw, nc=self.hparams.nc, reg_max=self.hparams.reg_max)
        total, comp = self.loss_fn(preds, batch)
        # comp = [box, cls, dfl] (per-image scaled internally by loss); total already × bs
        self.log_dict({
            "train/loss":     total / max(img.shape[0], 1),
            "train/loss_box": comp[0],
            "train/loss_cls": comp[1],
            "train/loss_dfl": comp[2],
            "train/lr":       self.optimizers().param_groups[0]["lr"],
            "train/instances": float(batch["batch_idx"].numel()),
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        return total

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update(self.model)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        img = batch["img"]
        if self.hparams.channels_last:
            img = img.to(memory_format=torch.channels_last)
        # Use EMA for eval if available
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        with torch.no_grad():
            decoded, _ = eval_model(img)        # [B, 4+nc, A] in pixel units (× stride)
        # decoded in input-pixel coords; normalize to [0, 1] for matching with GT
        H, W = img.shape[2], img.shape[3]
        decoded_norm = decoded.clone()
        decoded_norm[:, [0, 2], :] /= W
        decoded_norm[:, [1, 3], :] /= H
        preds = _multi_label_nms(decoded_norm, conf_thr=0.001, iou_thr=0.7, max_det=300)

        # GT in normalized xyxy
        gts = []
        bs = img.shape[0]
        for i in range(bs):
            mask = batch["batch_idx"] == i
            cls = batch["cls"][mask].view(-1).long()
            xywh = batch["bboxes"][mask]   # normalized cx,cy,w,h
            if xywh.numel() == 0:
                gts.append({"boxes": xywh.new_zeros((0, 4)),
                            "labels": torch.zeros((0,), dtype=torch.int64, device=img.device)})
                continue
            cx, cy, w, h = xywh.unbind(1)
            xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
            gts.append({"boxes": xyxy, "labels": cls})

        self._val_map[dataloader_idx].update(preds, gts)

    # Map torchmetrics' compute() dict keys → human-readable WandB labels
    # (Ultralytics-style: mAP, mAP50, mAP75, mAR100).
    _VAL_METRIC_KEY_MAP = (
        ("map",       "mAP"),         # mAP @ IoU [.50:.05:.95]
        ("map_50",    "mAP50"),
        ("map_75",    "mAP75"),
        ("map_small", "mAP_small"),
        ("map_medium","mAP_medium"),
        ("map_large", "mAP_large"),
        ("mar_100",   "mAR100"),      # max recall @ 100 detections
    )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # ModelEMA is a plain Python object, not a sub-module — Lightning won't
        # auto-persist it. Stash its state alongside the online state_dict so
        # `promote_ckpt` can prefer EMA weights for export.
        if self.ema is not None:
            checkpoint["ema_state_dict"] = {
                k: v.detach().cpu() for k, v in self.ema.ema.state_dict().items()
            }
            checkpoint["ema_updates"] = self.ema.updates

    def on_validation_epoch_end(self):
        out = {}
        for i, name in enumerate(self._val_names):
            res = self._val_map[i].compute()
            sub = {}
            for k_src, k_log in self._VAL_METRIC_KEY_MAP:
                if k_src in res:
                    v = float(res[k_src])
                    self.log(f"val/{name}/{k_log}", v,
                             on_epoch=True, sync_dist=True)
                    sub[k_log] = v
            out[name] = sub
            self._val_map[i].reset()
        # Stash for ValSummaryCallback to print to stdout
        self._last_val_metrics = out
        return out

    def configure_optimizers(self):
        # Standard YOLO 3-group optimizer: weights-with-decay, biases-no-decay, BN-no-decay
        g = [[], [], []]
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                g[2].append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                g[1].append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                g[0].append(v.weight)

        opt = SGD(g[2], lr=self.hparams.lr0, momentum=self.hparams.momentum, nesterov=True)
        opt.add_param_group({"params": g[0], "weight_decay": self.hparams.weight_decay})
        opt.add_param_group({"params": g[1], "weight_decay": 0.0})

        # Cosine LR from lr0 → lr0 * lrf over total epochs
        max_epochs = self.trainer.max_epochs if self.trainer else 50
        warmup = self.hparams.warmup_epochs
        lrf = self.hparams.lrf

        def lr_lambda(epoch: float) -> float:
            if epoch < warmup:
                return ((epoch / warmup) * (1 - lrf) + lrf)
            # cosine
            import math
            t = (epoch - warmup) / max(max_epochs - warmup, 1)
            return ((1 + math.cos(t * math.pi)) / 2) * (1 - lrf) + lrf

        sch = LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "epoch"}]
