"""Hydra entry point for YOLO-JDT TAGate track-offset training (Step 5.DE pivot).

Single-stage: the entire JDE model is frozen; only TAGate + TrackOffsetHead
train (SmoothL1 on per-anchor inter-frame motion).

Usage:
    python -m yolo_jdt.scripts.train_tagate -cn tagate \\
        run_name=step5_jdt_offset trainer.max_epochs=25 \\
        model.pretrained_weights=weights/ours/yolo11s_jde.pt
"""
from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from yolo_jdt.train.callbacks import ValSummaryCallback
from yolo_jdt.train.jdt_datamodule import JDTDataModule
from yolo_jdt.train.jdt_lightning_module import JDTLitModule


@hydra.main(config_path="../configs", config_name="tagate", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    orig_cwd = Path(hydra.utils.get_original_cwd())
    if cfg.model.pretrained_weights:
        cfg.model.pretrained_weights = str(orig_cwd / cfg.model.pretrained_weights)
    cfg.data.standard_root = str(orig_cwd / cfg.data.standard_root)
    cfg.output_dir = str(orig_cwd / cfg.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    dm = JDTDataModule(
        standard_root=cfg.data.standard_root,
        imgsz=cfg.data.imgsz,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        hsv=tuple(cfg.data.hsv),
        flip_p=cfg.data.flip_p,
        aug_scale=cfg.data.aug_scale,
        aug_translate=cfg.data.aug_translate,
        person_only=cfg.data.person_only,
        mot17_train_split=cfg.data.mot17_train_split,
    )
    dm.setup()
    num_track_ids = (dm.num_track_ids if cfg.model.num_track_ids in (-1, None)
                     else cfg.model.num_track_ids)
    print(f"[train_tagate] num_track_ids = {num_track_ids}")

    lit = JDTLitModule(
        scale=cfg.model.scale,
        nc=cfg.model.nc,
        reg_max=cfg.model.reg_max,
        imgsz=cfg.model.imgsz,
        box_gain=cfg.model.box_gain,
        cls_gain=cfg.model.cls_gain,
        dfl_gain=cfg.model.dfl_gain,
        lr0=cfg.model.lr0,
        lrf=cfg.model.lrf,
        momentum=cfg.model.momentum,
        weight_decay=cfg.model.weight_decay,
        warmup_epochs=cfg.model.warmup_epochs,
        warmup_momentum=cfg.model.warmup_momentum,
        warmup_bias_lr=cfg.model.warmup_bias_lr,
        ema_decay=cfg.model.ema_decay,
        ema_tau=cfg.model.ema_tau,
        pretrained_weights=cfg.model.pretrained_weights,
        channels_last=cfg.model.channels_last,
        num_track_ids=num_track_ids,
        reid_dim=cfg.model.reid_dim,
        reid_hidden=cfg.model.reid_hidden,
        lambda_reid=cfg.model.lambda_reid,
        cache_levels=cfg.model.cache_levels,
        tagate_num_layers=cfg.model.tagate_num_layers,
        tagate_num_heads=cfg.model.tagate_num_heads,
        tagate_ffn_ratio=cfg.model.tagate_ffn_ratio,
        tagate_gate_init=cfg.model.tagate_gate_init,
        offset_hidden=cfg.model.offset_hidden,
        offset_gain=cfg.model.offset_gain,
        stage=cfg.model.stage,
        stage_b_lr_scale=cfg.model.stage_b_lr_scale,
        tagate_lr_scale=cfg.model.tagate_lr_scale,
        stage_a_alpha=cfg.model.stage_a_alpha,
        freeze_detection_head=cfg.model.freeze_detection_head,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.output_dir,
            filename="{epoch:03d}-step{step}",
            save_top_k=1,
            monitor="val/mot17_val_half/mAP",
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ValSummaryCallback(),
    ]

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            name=cfg.run_name,
            notes=cfg.wandb.notes,
            save_dir=cfg.output_dir,
        )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        sync_batchnorm=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        val_check_interval=cfg.trainer.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=cfg.output_dir,
        deterministic=False,
    )

    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
