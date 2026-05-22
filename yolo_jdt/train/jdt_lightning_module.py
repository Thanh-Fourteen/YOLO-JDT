"""PyTorch Lightning module for YOLO-JDT: temporal track-offset training.

Step 5.DE pivot (2026-05-22). After v1–v9 of TAGate-for-ReID all degraded HOTA
below the JDE baseline, literature research (see project memory) established the
failure is structural: a trainable module upstream of a frozen embedding head
perturbs that head's input and corrupts JDE-quality embeddings. The pivot
re-targets TAGate to a NEW task that has no pretrained representation to corrupt.

Training procedure:
  1. PairedFrameDataset yields (img_t, img_prev) pairs from the same sequence,
     each frame-t box carrying a GT offset (centre displacement to t-1).
  2. F_prev = backbone+neck(img_prev) computed with no_grad (detached cache).
  3. Forward: model(img_t, cache) → TAGate-enhanced features → TrackOffsetHead.
  4. The ENTIRE JDE model (backbone + neck + head.cv2/cv3/cv4) is frozen. Only
     TAGate and the TrackOffsetHead ever update. Detection AND ReID are therefore
     bit-exact JDE — no regression possible.
  5. Single-stage training: loss = SmoothL1 on the offset head at positive
     anchors (offset_only=True — det/reid loss terms skipped).

Validation runs the frozen JDE detector (zero temporal cache) — val mAP is a
constant invariant: if it ever drifts, TAGate has leaked into the detection
path. The real metric (offset → association quality) is HOTA, measured
post-hoc by infer_tracking_jdt + TrackEval; promote from last.ckpt.
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
        tagate_gate_init: float = 0.0,
        # Track-offset head (Step 5.DE pivot)
        offset_hidden: int = 256,
        offset_gain: float = 1.0,
        # schedule (kept for config/ckpt compat — pivot is single-stage)
        stage: str = "A",
        stage_b_lr_scale: float = 0.1,
        tagate_lr_scale: float = 1.0,
        stage_a_alpha: float = 0.1,
        freeze_detection_head: bool = True,
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
        self._tagate_lr_scale = tagate_lr_scale
        self._stage_a_alpha = stage_a_alpha
        self._freeze_detection_head = freeze_detection_head

        self.model = YOLO_JDT(
            scale=scale, nc=nc, reg_max=reg_max,
            strides=(8.0, 16.0, 32.0),
            reid_dim=reid_dim, reid_hidden=reid_hidden,
            cache_levels=cache_levels,
            tagate_num_layers=tagate_num_layers,
            tagate_num_heads=tagate_num_heads,
            tagate_ffn_ratio=tagate_ffn_ratio,
            tagate_gate_init=tagate_gate_init,
            offset_hidden=offset_hidden,
            img_size=imgsz,
        )
        # loss_fn must be built before _load_pretrained (reid_classifier restore)
        self.loss_fn = JointDetectionReIDLoss(
            nc=nc, reg_max=reg_max, stride=(8.0, 16.0, 32.0),
            box=box_gain, cls=cls_gain, dfl=dfl_gain, tal_topk=10,
            reid_dim=reid_dim, num_track_ids=num_track_ids,
            lambda_reid=lambda_reid, lambda_offset=offset_gain,
        )
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        # Step 5.DE pivot: the ENTIRE JDE model is frozen — backbone, neck, and
        # the full JointHead (cv2 box + cv3 cls + cv4 ReID). Detection AND ReID
        # are therefore bit-exact JDE; no regression is possible. Only TAGate and
        # the TrackOffsetHead ever update. (`freeze_detection_head` is kept as a
        # constructor arg for config compat but the head freeze is unconditional.)
        self._freeze_backbone_neck()
        self._freeze_detection_branches()

        self.ema = None
        # Diagnostic buffers — overwritten each training step, read in on_train_epoch_end
        self._diag_img_t = None
        self._diag_img_prev = None
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
        fresh = sum(1 for k in result.missing_keys
                    if k.startswith(("tagates.", "offset_head.")))
        print(
            f"[JDTLitModule] loaded: {len(src)} src keys, "
            f"missing {len(result.missing_keys)} (TAGate+offset-fresh: {fresh}), "
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
            f"[JDTLitModule] backbone+neck frozen ({n_frozen}/{n_total} params); "
            f"TAGate + TrackOffsetHead trainable"
        )

    def _freeze_detection_branches(self) -> None:
        """Freeze the ENTIRE JointHead — cv2 (box) + cv3 (cls) + cv4 (ReID).

        Step 5.DE diagnostic (2026-05-20): a YOLO_JDT with pure JDE weights and
        an identity TAGate, run through the full JDT tracker+eval, reproduced
        the JDE baseline EXACTLY (HOTA 0.5600 / IDs 453). That proved the
        pipeline is correct and that fine-tuning cv4 on the small 2.6k-pair MOT
        set is what *destroyed* JDE-quality ReID embeddings in every v1–v8 run
        (HOTA fell to 0.51–0.55). So cv4 is now frozen too: the JDE embedding
        space is preserved bit-for-bit and TAGate (the only trainable module,
        zero-init → identity at init) can only *add* temporal consistency on
        top of it. Hard floor: HOTA ≥ JDE 0.560.
        """
        for p in self.model.head.parameters():
            p.requires_grad_(False)
        n_det_frozen = sum(1 for p in self.model.head.parameters()
                           if not p.requires_grad)
        print(f"[JDTLitModule] FULL JointHead frozen ({n_det_frozen} params, "
              f"cv2+cv3+cv4); only TAGate (zero-init) trains")

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

        raw_det, _reid, offset_out, _feats_cache = self.model(img_t, cache)
        preds = decode_raw_outputs(raw_det, nc=self._nc, reg_max=self._reg_max)

        # offset_only=True: detection + ReID are frozen JDE — only the offset
        # head is supervised. The assigner still runs (inside loss_fn) to pick
        # the positive anchors at which the offset GT is regressed.
        det_batch = {
            "batch_idx":    batch["batch_idx_t"],
            "cls":          batch["cls_t"],
            "bboxes":       batch["bboxes_t"],
            "offsets":      batch["offsets_t"],
            "offset_valid": batch["offset_valid_t"],
        }
        total, comp = self.loss_fn(preds, det_batch,
                                   offset_per_level=offset_out, offset_only=True)

        bs = max(img_t.shape[0], 1)
        # Diagnostic: ‖proj_out‖ of TAGate layer 0. Zero-init → 0.0 at init;
        # grows as the cross-attention learns temporal correspondence.
        proj_norm = self.model.tagates[0].layers[0].attn.proj_out.weight.detach().norm().item()
        self.log_dict({
            "train/loss":         total / bs,
            "train/loss_offset":  comp[4],
            "train/tagate_pnorm": proj_norm,
            "train/lr":           self.optimizers().param_groups[0]["lr"],
            "train/instances":  float(batch["batch_idx_t"].numel()),
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        # Keep one sample for epoch-end diagnostics (entropy + full gate log)
        self._diag_img_t = img_t[:2].detach()
        self._diag_img_prev = img_prev[:2].detach()
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
    # Diagnostics — gate α + attention entropy + grad norms
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log per-module gradient norms before each optimizer step (averaged per epoch)."""
        def _gnorm(params):
            grads = [p.grad.detach() for p in params if p.grad is not None]
            if not grads:
                return 0.0
            return torch.stack([g.norm() for g in grads]).norm().item()

        tagate_params = [p for n, p in self.model.named_parameters() if "tagates" in n]
        offset_params = [p for n, p in self.model.named_parameters() if "offset_head" in n]
        bb_params     = [p for n, p in self.model.named_parameters() if n.startswith("backbone.")]
        neck_params   = [p for n, p in self.model.named_parameters() if n.startswith("neck.")]
        self.log_dict({
            "diag/grad_norm_tagate":   _gnorm(tagate_params),
            "diag/grad_norm_offset":   _gnorm(offset_params),
            "diag/grad_norm_backbone": _gnorm(bb_params),
            "diag/grad_norm_neck":     _gnorm(neck_params),
        }, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

    def on_train_epoch_end(self) -> None:
        """Log per-layer TAGate ‖proj_out‖ (0 at init) + attention entropy."""
        log_dict = {}
        for i, tagate in enumerate(self.model.tagates):
            for j, layer in enumerate(tagate.layers):
                log_dict[f"diag/tagate_{i}_layer_{j}_pnorm"] = \
                    layer.attn.proj_out.weight.detach().norm().item()
        self.log_dict(log_dict, on_epoch=True, prog_bar=False, sync_dist=False)
        if self._diag_img_t is not None:
            self._log_attention_entropy()

    def _log_attention_entropy(self) -> None:
        """One diagnostic forward with capture_attention=True to compute attention entropy.

        Attention entropy measures how spread the attention is:
          - High entropy (→ log(L) ≈ 6 for P5 20×20) = uniform, not learning pattern.
          - Low entropy = peaked attention = model found temporal correspondences.
        Run once per epoch; uses self.model (online weights, eval mode).
        """
        from yolo_jdt.models.tagate.cross_attn import CrossAttentionBlock
        device = next(self.model.parameters()).device
        img_t    = self._diag_img_t.to(device)
        img_prev = self._diag_img_prev.to(device)

        CrossAttentionBlock.capture_attention = True
        try:
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                cache = self._extract_cache(img_prev)
                self.model(img_t, cache)
            if was_training:
                self.model.train()
        finally:
            CrossAttentionBlock.capture_attention = False

        log_dict = {}
        for i, tagate in enumerate(self.model.tagates):
            for j, layer in enumerate(tagate.layers):
                attn_w = getattr(layer.attn, "_last_attn_weights", None)
                if attn_w is not None:
                    # [B, nh, L, L] — last dim = key distribution per query token
                    p = attn_w.float().clamp(min=1e-9)
                    ent = (-p * p.log()).sum(-1).mean().item()
                    log_dict[f"diag/tagate_{i}_layer_{j}_attn_entropy"] = ent
        if log_dict:
            self.log_dict(log_dict, on_epoch=True, prog_bar=False, sync_dist=False)

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
        # Step 5.DE pivot: the entire JDE model (backbone+neck+JointHead incl.
        # cv2/cv3/cv4) AND the ReID classifier are frozen — only TAGate and the
        # TrackOffsetHead train. The offset head is a fresh task with no
        # pretrained representation to corrupt, so (unlike v1–v9 ReID training)
        # there is no embedding-degradation failure mode.
        #
        # named_parameters() captures every registered Parameter. Decay
        # heuristic: ndim ≥ 2 → weight decay; scalars / 1-D (bias, LN) → none.
        for p in self.loss_fn.classifier.parameters():
            p.requires_grad_(False)            # freeze ReID classifier (unused in pivot)

        decay, no_decay = [], []
        seen: set[int] = set()
        for n, p in self.model.named_parameters():
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            assert ("tagates" in n or "offset_head" in n), \
                f"unexpected trainable param (JDE model must be frozen): {n}"
            (decay if p.ndim >= 2 else no_decay).append(p)

        train_lr = self._lr0 * self._tagate_lr_scale
        opt = SGD(no_decay, lr=train_lr, momentum=self._momentum,
                  nesterov=True, weight_decay=0.0)
        if decay:
            opt.add_param_group({"params": decay,
                                 "weight_decay": self._weight_decay})

        n_train = len(decay) + len(no_decay)
        print(f"[configure_optimizers] lr={train_lr:.2e}  "
              f"trainable_params={n_train}  (TAGate + TrackOffsetHead; "
              f"JDE model + classifier frozen)")

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
