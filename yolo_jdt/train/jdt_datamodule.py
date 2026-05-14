"""DataModule for YOLO-JDT temporal training: PairedFrameDataset (train) + DetValDataset (val).

Training uses MOT17 train_half only — PairedFrameDataset requires sequential
sequences, which CrowdHuman (static single images) cannot provide.

Val loaders are identical to JDEDataModule: MOT17 val_half + CrowdHuman val,
both using DetValDataset (no temporal context needed for mAP evaluation).
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from yolo_jdt.data.augment import DetValDataset, collate_det
from yolo_jdt.data.datasets import CrowdHumanDataset, MOTDataset
from yolo_jdt.data.tracking_loader import PairedFrameDataset, collate_paired
from yolo_jdt.data.mosaic_id import build_global_id_map


class JDTDataModule(L.LightningDataModule):
    def __init__(
        self,
        standard_root: str = "datasets/standard",
        imgsz: int = 640,
        batch_size: int = 8,
        num_workers: int = 4,
        hsv: tuple = (0.015, 0.7, 0.4),
        flip_p: float = 0.5,
        aug_scale: float = 0.5,
        aug_translate: float = 0.1,
        person_only: bool = True,
        mot17_train_split: str = "train_half",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root = Path(standard_root)
        self.global_id_map: dict = {}
        self.num_track_ids: int = 0

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "_setup_done"):
            return
        self._setup_done = True

        self.global_id_map = build_global_id_map(
            self.root, "mot17", self.hparams.mot17_train_split
        )
        self.num_track_ids = len(self.global_id_map)
        print(f"[JDTDataModule] global track IDs: {self.num_track_ids}")

        base = MOTDataset(self.root, name="mot17", split=self.hparams.mot17_train_split)
        self.train_set = PairedFrameDataset(
            base,
            imgsz=self.hparams.imgsz,
            hsv=tuple(self.hparams.hsv),
            flip_p=self.hparams.flip_p,
            scale=self.hparams.aug_scale,
            translate=self.hparams.aug_translate,
            person_only=self.hparams.person_only,
            global_id_map=self.global_id_map,
        )
        print(f"[JDTDataModule] train pairs: {len(self.train_set)}")

        self.val_sets = {
            "mot17_val_half": DetValDataset(
                MOTDataset(self.root, name="mot17", split="val_half"),
                imgsz=self.hparams.imgsz, person_only=self.hparams.person_only,
            ),
            "crowdhuman_val": DetValDataset(
                CrowdHumanDataset(self.root, split="val"),
                imgsz=self.hparams.imgsz, person_only=self.hparams.person_only,
            ),
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_paired,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
            timeout=120 if self.hparams.num_workers > 0 else 0,
        )

    def val_dataloader(self) -> list[DataLoader]:
        kw = dict(
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_det,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            timeout=120 if self.hparams.num_workers > 0 else 0,
        )
        return [
            DataLoader(self.val_sets["mot17_val_half"], **kw),
            DataLoader(self.val_sets["crowdhuman_val"], **kw),
        ]

    @property
    def val_set_names(self) -> list[str]:
        return ["mot17_val_half", "crowdhuman_val"]
