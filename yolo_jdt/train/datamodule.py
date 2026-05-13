"""Lightning DataModule for detection fine-tune (Step 3.A).

Train mix:
    CrowdHuman train (15,000) + MOT17 train_half (2,657) — person class only.
    train_half (not full train) avoids leaking val_half into training.
Val:
    MOT17 val_half (2,659)  for tracking-domain detection mAP
    CrowdHuman val  (4,370) for general dense-pedestrian mAP
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from yolo_jdt.data.augment import DetTrainDataset, DetValDataset, collate_det
from yolo_jdt.data.datasets import CrowdHumanDataset, MOTDataset


class DetDataModule(L.LightningDataModule):
    def __init__(self,
                 standard_root: str = "datasets/standard",
                 imgsz: int = 640,
                 batch_size: int = 16,
                 num_workers: int = 8,
                 mosaic_p: float = 1.0,
                 hsv: tuple = (0.015, 0.7, 0.4),
                 flip_p: float = 0.5,
                 aug_scale: float = 0.5,         # renamed from `scale` to avoid clash with LitModule.scale
                 aug_translate: float = 0.1,     # renamed from `translate` for symmetry
                 person_only: bool = True,
                 use_crowdhuman: bool = True,
                 use_mot17: bool = True,
                 mot17_train_split: str = "train_half"):
        super().__init__()
        self.save_hyperparameters()
        self.root = Path(standard_root)

    def setup(self, stage: str | None = None):
        self.train_subsets = []
        if self.hparams.use_crowdhuman:
            self.train_subsets.append(DetTrainDataset(
                CrowdHumanDataset(self.root, split="train"),
                imgsz=self.hparams.imgsz, mosaic_p=self.hparams.mosaic_p,
                hsv=self.hparams.hsv, flip_p=self.hparams.flip_p,
                scale=self.hparams.aug_scale, translate=self.hparams.aug_translate,
                person_only=self.hparams.person_only,
            ))
        if self.hparams.use_mot17:
            self.train_subsets.append(DetTrainDataset(
                MOTDataset(self.root, name="mot17", split=self.hparams.mot17_train_split),
                imgsz=self.hparams.imgsz, mosaic_p=self.hparams.mosaic_p,
                hsv=self.hparams.hsv, flip_p=self.hparams.flip_p,
                scale=self.hparams.aug_scale, translate=self.hparams.aug_translate,
                person_only=self.hparams.person_only,
            ))
        if not self.train_subsets:
            raise ValueError("DataModule constructed with no train datasets enabled")
        self.train_set = ConcatDataset(self.train_subsets)

        self.val_sets = {
            "mot17_val_half": DetValDataset(
                MOTDataset(self.root, name="mot17", split="val_half"),
                imgsz=self.hparams.imgsz, person_only=self.hparams.person_only),
            "crowdhuman_val": DetValDataset(
                CrowdHumanDataset(self.root, split="val"),
                imgsz=self.hparams.imgsz, person_only=self.hparams.person_only),
        }

    def set_mosaic_p(self, p: float):
        """Toggle mosaic on all subsets — used by close-mosaic callback."""
        for sub in self.train_subsets:
            sub.set_mosaic_p(p)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_det,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        # Combine both val sets into one DataLoader sequence for Lightning's
        # multi-dataloader val. We return a list so Lightning iterates each.
        return [
            DataLoader(self.val_sets["mot17_val_half"],
                       batch_size=self.hparams.batch_size,
                       shuffle=False,
                       num_workers=self.hparams.num_workers,
                       collate_fn=collate_det,
                       pin_memory=True,
                       persistent_workers=self.hparams.num_workers > 0),
            DataLoader(self.val_sets["crowdhuman_val"],
                       batch_size=self.hparams.batch_size,
                       shuffle=False,
                       num_workers=self.hparams.num_workers,
                       collate_fn=collate_det,
                       pin_memory=True,
                       persistent_workers=self.hparams.num_workers > 0),
        ]

    @property
    def val_set_names(self) -> list[str]:
        return ["mot17_val_half", "crowdhuman_val"]
