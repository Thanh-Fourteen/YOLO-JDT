"""PyTorch Lightning module for YOLO11-JDE: detection + ReID joint training.

Builds on `DetLitModule` (Step 3.A detection-only) by:
- Replacing `DecoupledDetect` with `JointHead` (adds cv4 ReID branch).
- Replacing `DetectionLoss` with `JointDetectionReIDLoss` (adds ReID head + CE).
- Supporting a 2-stage training schedule:
    Stage A — freeze backbone + neck, train head (cv2/cv3/cv4) + classifier
              ~5 epochs at full LR.
    Stage B — unfreeze everything, fine-tune at LR/10 ~15 epochs.
- Accepts a global track ID mapping (from `build_global_id_map`) at init so
  the classifier head size matches the ID space exactly.

Forward / val / EMA / mAP-tracking are inherited from DetLitModule.
"""
from __future__ import annotations

import copy

import lightning as L
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from third_party.ultralytics_extract.loss import decode_raw_outputs
from yolo_jdt.losses.joint_loss import JointDetectionReIDLoss
from yolo_jdt.models.head.joint_head import JointHead
from yolo_jdt.models.yolo11 import YOLO11
from yolo_jdt.train.lightning_module import DetLitModule, ModelEMA, _multi_label_nms


__all__ = ["JDELitModule"]


class _YOLO11WithJointHead(YOLO11):
    """YOLO11 assembly that swaps DecoupledDetect → JointHead.

    Forward returns:
        train mode: (raw_det_per_level, reid_per_level)
        eval mode:  (decoded, raw_det_per_level, reid_per_level)
    """

    def __init__(self, scale: str = "s", nc: int = 1, reg_max: int = 16,
                 strides: tuple[float, ...] = (8.0, 16.0, 32.0),
                 reid_dim: int = 128, reid_hidden: int = 256):
        # Skip YOLO11.__init__ assembly (it builds DecoupledDetect); rebuild manually
        nn.Module.__init__(self)
        from yolo_jdt.models.backbone.yolo11 import YOLO11Backbone
        from yolo_jdt.models.neck.panet import YOLO11PANet
        self.scale = scale
        self.nc = nc
        self.backbone = YOLO11Backbone(scale)
        self.neck = YOLO11PANet(scale, in_channels=self.backbone.out_channels)
        self.head = JointHead(nc=nc, ch=self.neck.out_channels,
                               reg_max=reg_max, strides=strides,
                               reid_dim=reid_dim, reid_hidden=reid_hidden)


class JDELitModule(DetLitModule):
    """JDE training module — 2-stage freeze/unfreeze schedule + ReID loss."""

    def __init__(self,
                 # detection args (inherited)
                 scale: str = "s", nc: int = 1, reg_max: int = 16,
                 imgsz: int = 640,
                 box_gain: float = 7.5, cls_gain: float = 0.5, dfl_gain: float = 1.5,
                 lr0: float = 0.001, lrf: float = 0.01,
                 momentum: float = 0.937, weight_decay: float = 5e-4,
                 warmup_epochs: float = 1.0, warmup_momentum: float = 0.8,
                 warmup_bias_lr: float = 0.01,
                 ema_decay: float = 0.9999, ema_tau: float = 2000,
                 pretrained_weights: str | None = None,
                 channels_last: bool = True,
                 # ReID args
                 num_track_ids: int = 359, reid_dim: int = 128,
                 reid_hidden: int = 256, lambda_reid: float = 0.1,
                 # 2-stage schedule
                 stage: str = "A",                # "A" = freeze BB+neck, "B" = unfreeze
                 stage_b_lr_scale: float = 0.1):  # LR multiplier for Stage B fine-tune
        # Initialize Lightning bookkeeping (skip parent constructor body — we
        # rebuild the model + loss with JointHead/JointLoss instead).
        L.LightningModule.__init__(self)
        self.save_hyperparameters()
        # Lightning's save_hyperparameters has trouble round-tripping `str | None`
        # PEP-604 union annotations on Python 3.11 → some kwargs go missing from
        # `self.hparams`. Mirror them as plain instance attrs so train/val code
        # never depends on hparams alone.
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

        # Model: YOLO11 with JointHead (cv4 ReID branch)
        self.model = _YOLO11WithJointHead(
            scale=scale, nc=nc, reg_max=reg_max,
            strides=(8.0, 16.0, 32.0),
            reid_dim=reid_dim, reid_hidden=reid_hidden,
        )
        # loss_fn must exist before _load_pretrained_partial (Stage A→B restores classifier)
        self.loss_fn = JointDetectionReIDLoss(
            nc=nc, reg_max=reg_max, stride=(8.0, 16.0, 32.0),
            box=box_gain, cls=cls_gain, dfl=dfl_gain, tal_topk=10,
            reid_dim=reid_dim, num_track_ids=num_track_ids,
            lambda_reid=lambda_reid,
        )
        if pretrained_weights:
            self._load_pretrained_partial(pretrained_weights)
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # Stage A: freeze backbone + neck (all but head + ReID classifier)
        if stage == "A":
            self._freeze_backbone_neck()

        self.ema = None  # initialized in on_train_start

        # Validation: same 2-loader mAP setup as DetLitModule
        self._val_map = nn.ModuleList([
            MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools"),
            MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools"),
        ])
        self._val_names = ["mot17_val_half", "crowdhuman_val"]

    # ---- Pretrained loading: prefer partial (cv4 missing is OK) -------------
    def _load_pretrained_partial(self, ckpt_path: str):
        """Load weights from either:
        - A Step 3.A-promoted standalone .pt (`{state_dict, scale, nc, ...}` payload), or
        - An Ultralytics .pt (raw COCO pretrained from Phase 2).
        - A Stage A Lightning .ckpt (for Stage B resume) — also restores reid_classifier.

        cv4 keys are NEW and stay freshly initialized by the constructor unless the
        checkpoint contains reid_classifier_state_dict (Stage A → Stage B transfer).
        """
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            # Step 3.A-style promoted ckpt or Lightning .ckpt
            src = ckpt["state_dict"]
            # Lightning .ckpt keys are prefixed "model." — strip for self.model
            if any(k.startswith("model.") for k in src):
                src = {k[len("model."):]: v for k, v in src.items() if k.startswith("model.")}
            result = self.model.load_state_dict(src, strict=False)
            n_missing = len(result.missing_keys)
            n_unexpected = len(result.unexpected_keys)
            cv4_missing = sum(1 for k in result.missing_keys if k.startswith("head.cv4"))
            print(f"[JDELitModule] loaded ckpt: {len(src)} src keys, "
                  f"missing {n_missing} (cv4-only: {cv4_missing}), "
                  f"unexpected {n_unexpected}")
            # Restore ReID classifier if this is a Stage A → Stage B transfer.
            if "reid_classifier_state_dict" in ckpt:
                self.loss_fn.classifier.load_state_dict(ckpt["reid_classifier_state_dict"])
                print("[JDELitModule] restored reid_classifier from checkpoint (Stage A → B)")
        else:
            # Ultralytics raw ckpt — go via the existing filter helper
            from yolo_jdt.train.lightning_module import DetLitModule as _Det
            _Det._load_pretrained_filter_nc(self, ckpt_path)

    # ---- Stage A freeze ------------------------------------------------------
    def _freeze_backbone_neck(self):
        """Freeze backbone + neck (all but head + classifier head)."""
        for name, p in self.model.named_parameters():
            if name.startswith("backbone.") or name.startswith("neck."):
                p.requires_grad_(False)
        n_frozen = sum(1 for n, p in self.model.named_parameters() if not p.requires_grad)
        n_total = sum(1 for _ in self.model.named_parameters())
        print(f"[JDELitModule] Stage A: froze {n_frozen}/{n_total} params (backbone + neck)")

    def unfreeze_all(self):
        """Stage B: unfreeze everything for fine-tune."""
        for p in self.model.parameters():
            p.requires_grad_(True)
        print(f"[JDELitModule] Stage B: unfroze all params")

    # ---- Forward / training step --------------------------------------------
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int):
        img = batch["img"]
        if self._channels_last:
            img = img.to(memory_format=torch.channels_last)
        raw_det, reid_per_level = self.model(img)
        preds = decode_raw_outputs(raw_det, nc=self._nc, reg_max=self._reg_max)
        total, comp = self.loss_fn(preds, batch, reid_per_level=reid_per_level)
        # comp = [box, cls, dfl, reid]
        bs = max(img.shape[0], 1)
        self.log_dict({
            "train/loss":      total / bs,
            "train/loss_box":  comp[0],
            "train/loss_cls":  comp[1],
            "train/loss_dfl":  comp[2],
            "train/loss_reid": comp[3],
            "train/lr":        self.optimizers().param_groups[0]["lr"],
            "train/instances": float(batch["batch_idx"].numel()),
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        return total

    def on_train_start(self):
        if self.ema is None:
            self.ema = ModelEMA(self.model, decay=self._ema_decay,
                                tau=self._ema_tau)

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update(self.model)

    # ---- EMA persistence (inherited pattern from DetLitModule fix) ----------
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema is not None:
            checkpoint["ema_state_dict"] = {
                k: v.detach().cpu() for k, v in self.ema.ema.state_dict().items()
            }
            checkpoint["ema_updates"] = self.ema.updates
        # Also persist the classifier head (it's in self.loss_fn, not self.model,
        # so Lightning's default state_dict() doesn't include it)
        checkpoint["reid_classifier_state_dict"] = {
            k: v.detach().cpu() for k, v in self.loss_fn.classifier.state_dict().items()
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "reid_classifier_state_dict" in checkpoint:
            self.loss_fn.classifier.load_state_dict(checkpoint["reid_classifier_state_dict"])

    # ---- Validation: identical to DetLitModule (uses _multi_label_nms) -------
    # Note: model returns (decoded, raw, reid) in eval — we discard reid.
    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        img = batch["img"]
        if self._channels_last:
            img = img.to(memory_format=torch.channels_last)
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        with torch.no_grad():
            decoded, _, _ = eval_model(img)        # discard raw + reid
        H, W = img.shape[2], img.shape[3]
        decoded_norm = decoded.clone()
        decoded_norm[:, [0, 2], :] /= W
        decoded_norm[:, [1, 3], :] /= H
        preds = _multi_label_nms(decoded_norm, conf_thr=0.001, iou_thr=0.7, max_det=300)

        gts = []
        bs = img.shape[0]
        for i in range(bs):
            mask = batch["batch_idx"] == i
            cls = batch["cls"][mask].view(-1).long()
            xywh = batch["bboxes"][mask]
            if xywh.numel() == 0:
                gts.append({"boxes": xywh.new_zeros((0, 4)),
                            "labels": torch.zeros((0,), dtype=torch.int64, device=img.device)})
                continue
            cx, cy, w, h = xywh.unbind(1)
            xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
            gts.append({"boxes": xyxy, "labels": cls})

        self._val_map[dataloader_idx].update(preds, gts)

    # on_validation_epoch_end + _VAL_METRIC_KEY_MAP inherited from DetLitModule

    # ---- Optimizer: include classifier head in param groups ------------------
    def configure_optimizers(self):
        # YOLO 3-group convention + ReID classifier as 4th group.
        g = [[], [], []]   # [weights-with-decay, BN-no-decay, biases-no-decay]
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                g[2].append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                g[1].append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                g[0].append(v.weight)
        # Classifier weight + bias
        clf_w = [self.loss_fn.classifier.weight]
        clf_b = [self.loss_fn.classifier.bias]

        # Apply Stage B LR scale globally if Stage B
        lr_scale = self._stage_b_lr_scale if self._stage == "B" else 1.0
        base_lr = self._lr0 * lr_scale

        opt = SGD(g[2], lr=base_lr, momentum=self._momentum, nesterov=True)
        opt.add_param_group({"params": g[0], "weight_decay": self._weight_decay})
        opt.add_param_group({"params": g[1], "weight_decay": 0.0})
        opt.add_param_group({"params": clf_w, "weight_decay": self._weight_decay})
        opt.add_param_group({"params": clf_b, "weight_decay": 0.0})

        max_epochs = self.trainer.max_epochs if self.trainer else 20
        warmup = self._warmup_epochs
        lrf = self._lrf

        def lr_lambda(epoch: float) -> float:
            if epoch < warmup:
                return ((epoch / warmup) * (1 - lrf) + lrf)
            import math
            t = (epoch - warmup) / max(max_epochs - warmup, 1)
            return ((1 + math.cos(t * math.pi)) / 2) * (1 - lrf) + lrf

        sch = LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "epoch"}]
