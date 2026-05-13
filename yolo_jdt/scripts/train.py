"""Hydra entry point for YOLO-JDT detection fine-tune (Step 3.A).

Usage:
    python -m yolo_jdt.scripts.train -cn base
    python -m yolo_jdt.scripts.train -cn base model.scale=m trainer.devices=1
    python -m yolo_jdt.scripts.train -cn base trainer.max_epochs=5 data.batch_size=4
"""
from __future__ import annotations

import os
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from yolo_jdt.train.callbacks import ValSummaryCallback
from yolo_jdt.train.datamodule import DetDataModule
from yolo_jdt.train.lightning_module import DetLitModule


class CloseMosaicCallback(L.Callback):
    """Disable mosaic in the final `1 - fraction` epochs (Ultralytics convention)."""

    def __init__(self, fraction: float = 0.85, max_epochs: int = 50):
        super().__init__()
        self.close_at = int(max_epochs * fraction)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.current_epoch == self.close_at:
            dm: DetDataModule = trainer.datamodule
            dm.set_mosaic_p(0.0)
            print(f"[CloseMosaic] epoch={trainer.current_epoch}: mosaic disabled.")


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # Resolve pretrained_weights to absolute (Hydra changes cwd)
    if cfg.model.pretrained_weights:
        cfg.model.pretrained_weights = str(
            Path(hydra.utils.get_original_cwd()) / cfg.model.pretrained_weights)
    cfg.data.standard_root = str(
        Path(hydra.utils.get_original_cwd()) / cfg.data.standard_root)
    cfg.output_dir = str(Path(hydra.utils.get_original_cwd()) / cfg.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Datamodule
    dm = DetDataModule(
        standard_root=cfg.data.standard_root,
        imgsz=cfg.data.imgsz,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        mosaic_p=cfg.data.mosaic_p,
        hsv=tuple(cfg.data.hsv),
        flip_p=cfg.data.flip_p,
        aug_scale=cfg.data.aug_scale,
        aug_translate=cfg.data.aug_translate,
        person_only=cfg.data.person_only,
        use_crowdhuman=cfg.data.use_crowdhuman,
        use_mot17=cfg.data.use_mot17,
        mot17_train_split=cfg.data.mot17_train_split,
    )

    # Lit module
    lit = DetLitModule(
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
        CloseMosaicCallback(
            fraction=cfg.close_mosaic_fraction,
            max_epochs=cfg.trainer.max_epochs),
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

    # Drop sync_batchnorm for single-device (Lightning would no-op but cleaner explicit)
    sync_bn = bool(cfg.trainer.sync_batchnorm) and int(cfg.trainer.devices) > 1
    strategy = cfg.trainer.strategy if int(cfg.trainer.devices) > 1 else "auto"

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        strategy=strategy,
        sync_batchnorm=sync_bn,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        val_check_interval=cfg.trainer.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=cfg.output_dir,
        deterministic=False,         # set True only when reproducibility is critical
    )

    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
