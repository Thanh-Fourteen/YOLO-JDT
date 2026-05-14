"""PyTorch Lightning module for YOLO-JDT: temporal attention training via paired frames.

Training procedure:
  1. PairedFrameDataset yields (img_t, img_prev) pairs from the same sequence.
  2. F_prev = backbone+neck(img_prev) computed with no_grad (detached cache).
  3. Forward: model(img_t, cached_features_prev) → loss on frame_t annotations.
  4. Stage A — freeze backbone + neck; TAGate + JointHead train.
     Stage B — unfreeze all at LR * stage_b_lr_scale.

Validation uses zero_cache() (TAGate ≈ identity at init; bounded after training).
"""
from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from third_party.ultralytics_extract.loss import decode_raw_outputs
from yolo_jdt.losses.joint_loss import JointDetectionReIDLoss
from yolo_jdt.models.yolo_jdt import YOLO_JDT
from yolo_jdt.train.lightning_module import ModelEMA, _multi_label_nms

__all__ = ["JDTLitModule"]

_VAL_METRIC_KEY_MAP = (
    ("map",       "mAP"),
    ("map_50",    "mAP50"),
    ("map_75",    "mAP75"),
    ("map_small", "mAP_small"),
    ("map_medium","mAP_medium"),
    ("map_large", "mAP_large"),
    ("mar_100",   "mAR100"),
)


class JDTLitModule(L.LightningModule):
    """YOLO-JDT training module — TAGate 2-stage training."""

    def __init__(
        self,
        scale: str = "s",
        nc: int = 1,
        reg_max: int = 16,
        imgsz: int = 640,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        lr0: float = 0.001,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 5e-4,
        warmup_epochs: float = 1.0,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.01,
        ema_decay: float = 0.9999,
        ema_tau: float = 2000,
        pretrained_weights: str | None = None,
        channels_last: bool = True,
        # ReID
        num_track_ids: int = 359,
        reid_dim: int = 128,
        reid_hidden: int = 256,
        lambda_reid: float = 0.1,
        # TAGate
        cache_levels: str = "P5",
        tagate_num_layers: int = 2,
        tagate_num_heads: int = 8,
        tagate_ffn_ratio: int = 2,
        # 2-stage schedule
        stage: str = "A",
        stage_b_lr_scale: float = 0.1,
    ):
        L.LightningModule.__init__(self)
        self.save_hyperparameters()
        # Mirror as instance attrs — Lightning save_hyperparameters drops PEP-604 unions
        self._scale = scale
        self._nc = nc
        self._reg_max = reg_max
        self._channels_last = channels_last
        self._lr0 = lr0
        self._lrf = lrf
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._warmup_epochs = warmup_epochs
        self._ema_decay = ema_decay
        self._ema_tau = ema_tau
        self._stage = stage
        self._stage_b_lr_scale = stage_b_lr_scale

        self.model = YOLO_JDT(
            scale=scale, nc=nc, reg_max=reg_max,
            strides=(8.0, 16.0, 32.0),
            reid_dim=reid_dim, reid_hidden=reid_hidden,
            cache_levels=cache_levels,
            tagate_num_layers=tagate_num_layers,
            tagate_num_heads=tagate_num_heads,
            tagate_ffn_ratio=tagate_ffn_ratio,
            img_size=imgsz,
        )
        # loss_fn must be built before _load_pretrained (reid_classifier restore)
        self.loss_fn = JointDetectionReIDLoss(
            nc=nc, reg_max=reg_max, stride=(8.0, 16.0, 32.0),
            box=box_gain, cls=cls_gain, dfl=dfl_gain, tal_topk=10,
            reid_dim=reid_dim, num_track_ids=num_track_ids,
            lambda_reid=lambda_reid,
        )
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if stage == "A":
            self._freeze_backbone_neck()

        self.ema = None
        self._val_map = nn.ModuleList([
            MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools"),
            MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools"),
        ])
        self._val_names = ["mot17_val_half", "crowdhuman_val"]

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_pretrained(self, ckpt_path: str) -> None:
        """Load backbone/neck/head weights; tagates.* stay freshly initialized.

        Accepts:
          - Lightning .ckpt (state_dict prefixed "model.")
          - promote_ckpt .pt  (plain state_dict or {state_dict, scale, nc})
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            src = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            src = ckpt
        else:
            src = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else {}

        if any(k.startswith("model.") for k in src):
            src = {k[len("model."):]: v for k, v in src.items() if k.startswith("model.")}

        result = self.model.load_state_dict(src, strict=False)
        tagate_missing = sum(1 for k in result.missing_keys if k.startswith("tagates."))
        print(
            f"[JDTLitModule] loaded: {len(src)} src keys, "
            f"missing {len(result.missing_keys)} (tagate-fresh: {tagate_missing}), "
            f"unexpected {len(result.unexpected_keys)}"
        )
        if isinstance(ckpt, dict) and "reid_classifier_state_dict" in ckpt:
            self.loss_fn.classifier.load_state_dict(ckpt["reid_classifier_state_dict"])
            print("[JDTLitModule] restored reid_classifier from checkpoint")

    # ------------------------------------------------------------------
    # Stage freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_backbone_neck(self) -> None:
        for name, p in self.model.named_parameters():
            if name.startswith("backbone.") or name.startswith("neck."):
                p.requires_grad_(False)
        n_frozen = sum(1 for _, p in self.model.named_parameters() if not p.requires_grad)
        n_total = sum(1 for _ in self.model.parameters())
        print(
            f"[JDTLitModule] Stage A: froze {n_frozen}/{n_total} params "
            f"(backbone+neck); TAGate + JointHead trainable"
        )

    def unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(True)
        print("[JDTLitModule] Stage B: all parameters unfrozen")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_cache(self, img_prev: Tensor) -> list[Tensor]:
        """Compute neck features from img_prev — detached (no grad through prev)."""
        with torch.no_grad():
            p3, p4, p5 = self.model.backbone(img_prev)
            neck_feats = list(self.model.neck(p3, p4, p5))
        return [neck_feats[i].detach() for i in self.model._level_ids]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int):
        img_t = batch["img_t"]
        img_prev = batch["img_prev"]
        if self._channels_last:
            img_t = img_t.to(memory_format=torch.channels_last)
            img_prev = img_prev.to(memory_format=torch.channels_last)

        cache = self._extract_cache(img_prev)

        raw_det, reid_per_level, _offset, _feats_cache = self.model(img_t, cache)
        preds = decode_raw_outputs(raw_det, nc=self._nc, reg_max=self._reg_max)

        det_batch = {
            "batch_idx": batch["batch_idx_t"],
            "cls":       batch["cls_t"],
            "bboxes":    batch["bboxes_t"],
            "track_ids": batch["track_ids_t"],
        }
        total, comp = self.loss_fn(preds, det_batch, reid_per_level=reid_per_level)

        bs = max(img_t.shape[0], 1)
        # Log gate α for first layer of first TAGate (diagnostic metric)
        alpha = torch.sigmoid(self.model.tagates[0].layers[0].gate).item()
        self.log_dict({
            "train/loss":       total / bs,
            "train/loss_box":   comp[0],
            "train/loss_cls":   comp[1],
            "train/loss_dfl":   comp[2],
            "train/loss_reid":  comp[3],
            "train/gate_alpha": alpha,
            "train/lr":         self.optimizers().param_groups[0]["lr"],
            "train/instances":  float(batch["batch_idx_t"].numel()),
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        return total

    def on_train_start(self) -> None:
        if self.ema is None:
            self.ema = ModelEMA(self.model, decay=self._ema_decay, tau=self._ema_tau)

    def on_train_batch_end(self, *args, **kwargs) -> None:
        if self.ema is not None:
            self.ema.update(self.model)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema is not None:
            checkpoint["ema_state_dict"] = {
                k: v.detach().cpu() for k, v in self.ema.ema.state_dict().items()
            }
            checkpoint["ema_updates"] = self.ema.updates
        checkpoint["reid_classifier_state_dict"] = {
            k: v.detach().cpu() for k, v in self.loss_fn.classifier.state_dict().items()
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "reid_classifier_state_dict" in checkpoint:
            self.loss_fn.classifier.load_state_dict(checkpoint["reid_classifier_state_dict"])

    # ------------------------------------------------------------------
    # Validation  (no temporal context — zero cache)
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        img = batch["img"]
        if self._channels_last:
            img = img.to(memory_format=torch.channels_last)
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        cache = eval_model.zero_cache(batch_size=img.shape[0],
                                       device=img.device, dtype=img.dtype)
        with torch.no_grad():
            decoded, _, _, _, _ = eval_model(img, cache)   # 5-tuple eval mode

        H, W = img.shape[2], img.shape[3]
        decoded_norm = decoded.clone()
        decoded_norm[:, [0, 2], :] /= W
        decoded_norm[:, [1, 3], :] /= H
        preds = _multi_label_nms(decoded_norm, conf_thr=0.001, iou_thr=0.7, max_det=300)

        gts = []
        for i in range(img.shape[0]):
            mask = batch["batch_idx"] == i
            cls = batch["cls"][mask].view(-1).long()
            xywh = batch["bboxes"][mask]
            if xywh.numel() == 0:
                gts.append({"boxes": xywh.new_zeros((0, 4)),
                            "labels": torch.zeros((0,), dtype=torch.int64, device=img.device)})
                continue
            cx, cy, w, h = xywh.unbind(1)
            xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)
            gts.append({"boxes": xyxy, "labels": cls})
        self._val_map[dataloader_idx].update(preds, gts)

    def on_validation_epoch_end(self) -> None:
        for di, name in enumerate(self._val_names):
            result = self._val_map[di].compute()
            self._val_map[di].reset()
            for k_src, k_log in _VAL_METRIC_KEY_MAP:
                val = result.get(k_src, torch.tensor(-1.0))
                self.log(f"val/{name}/{k_log}", val,
                         on_epoch=True, prog_bar=(k_log == "mAP"), sync_dist=False)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        g = [[], [], []]   # [weight+decay, BN-no-decay, bias-no-decay]
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                g[2].append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                g[1].append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                g[0].append(v.weight)

        lr_scale = self._stage_b_lr_scale if self._stage == "B" else 1.0
        base_lr = self._lr0 * lr_scale

        opt = SGD(g[2], lr=base_lr, momentum=self._momentum, nesterov=True)
        opt.add_param_group({"params": g[0], "weight_decay": self._weight_decay})
        opt.add_param_group({"params": g[1], "weight_decay": 0.0})
        opt.add_param_group({"params": [self.loss_fn.classifier.weight],
                             "weight_decay": self._weight_decay})
        opt.add_param_group({"params": [self.loss_fn.classifier.bias],
                             "weight_decay": 0.0})

        max_epochs = self.trainer.max_epochs if self.trainer else 20
        warmup = self._warmup_epochs
        lrf = self._lrf

        def lr_lambda(epoch: float) -> float:
            if epoch < warmup:
                return (epoch / warmup) * (1 - lrf) + lrf
            t = (epoch - warmup) / max(max_epochs - warmup, 1)
            return ((1 + math.cos(t * math.pi)) / 2) * (1 - lrf) + lrf

        sch = LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "epoch"}]
