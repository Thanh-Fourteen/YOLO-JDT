"""DataModule for JDE training: detection batches + ReID track_ids.

Differences from `DetDataModule`:
- Train wrappers use `DetTrainIDDataset` (preserves track_ids through mosaic)
  instead of `DetTrainDataset` (drops them).
- Uses `collate_det_with_ids` instead of `collate_det` so the batch dict
  carries `track_ids`.
- Builds the global ID map from MOT17 train_half once at setup() and
  exposes `num_track_ids` so the LightningModule can size its classifier.
- Val DataLoaders unchanged (val mAP doesn't need ReID labels).
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from yolo_jdt.data.augment import DetValDataset, collate_det
from yolo_jdt.data.datasets import CrowdHumanDataset, MOTDataset
from yolo_jdt.data.mosaic_id import (DetTrainIDDataset, build_global_id_map,
                                       collate_det_with_ids)


class JDEDataModule(L.LightningDataModule):
    def __init__(self,
                 standard_root: str = "datasets/standard",
                 imgsz: int = 640,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 mosaic_p: float = 1.0,
                 hsv: tuple = (0.015, 0.7, 0.4),
                 flip_p: float = 0.5,
                 aug_scale: float = 0.5,
                 aug_translate: float = 0.1,
                 person_only: bool = True,
                 use_crowdhuman: bool = True,
                 use_mot17: bool = True,
                 mot17_train_split: str = "train_half"):
        super().__init__()
        self.save_hyperparameters()
        self.root = Path(standard_root)
        self.global_id_map: dict[tuple[str, int], int] = {}
        self.num_track_ids: int = 0

    def setup(self, stage: str | None = None):
        # Guard: Lightning calls setup() again during trainer.fit(); skip if already done.
        if hasattr(self, "_setup_done"):
            return
        self._setup_done = True

        # 1. Build global track ID map ONCE from the MOT17 train split.
        #    CrowdHuman static instances (track_id=-1) are excluded by build_global_id_map.
        if self.hparams.use_mot17:
            self.global_id_map = build_global_id_map(
                self.root, "mot17", self.hparams.mot17_train_split)
            self.num_track_ids = len(self.global_id_map)
            print(f"[JDEDataModule] global track IDs: {self.num_track_ids}")
        else:
            self.num_track_ids = 0

        self.train_subsets = []
        if self.hparams.use_crowdhuman:
            # CrowdHuman has track_id=-1 → all instances will hit the
            # ignore_index=-1 path in CE loss; detection loss still works.
            self.train_subsets.append(DetTrainIDDataset(
                CrowdHumanDataset(self.root, split="train"),
                global_id_map=self.global_id_map,    # CH instances absent → -1
                imgsz=self.hparams.imgsz, mosaic_p=self.hparams.mosaic_p,
                hsv=self.hparams.hsv, flip_p=self.hparams.flip_p,
                scale=self.hparams.aug_scale, translate=self.hparams.aug_translate,
                person_only=self.hparams.person_only,
            ))
        if self.hparams.use_mot17:
            self.train_subsets.append(DetTrainIDDataset(
                MOTDataset(self.root, name="mot17", split=self.hparams.mot17_train_split),
                global_id_map=self.global_id_map,
                imgsz=self.hparams.imgsz, mosaic_p=self.hparams.mosaic_p,
                hsv=self.hparams.hsv, flip_p=self.hparams.flip_p,
                scale=self.hparams.aug_scale, translate=self.hparams.aug_translate,
                person_only=self.hparams.person_only,
            ))
        if not self.train_subsets:
            raise ValueError("JDEDataModule built with no train subsets enabled")
        self.train_set = ConcatDataset(self.train_subsets)

        # Val: same as DetDataModule (no ReID labels needed for mAP)
        self.val_sets = {
            "mot17_val_half": DetValDataset(
                MOTDataset(self.root, name="mot17", split="val_half"),
                imgsz=self.hparams.imgsz, person_only=self.hparams.person_only),
            "crowdhuman_val": DetValDataset(
                CrowdHumanDataset(self.root, split="val"),
                imgsz=self.hparams.imgsz, person_only=self.hparams.person_only),
        }

    def set_mosaic_p(self, p: float):
        for sub in self.train_subsets:
            sub.set_mosaic_p(p)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_det_with_ids,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
            # timeout>0: raise after N seconds if a worker dies instead of hanging forever.
            timeout=120 if self.hparams.num_workers > 0 else 0,
        )

    def val_dataloader(self):
        return [
            DataLoader(self.val_sets["mot17_val_half"],
                       batch_size=self.hparams.batch_size,
                       shuffle=False,
                       num_workers=self.hparams.num_workers,
                       collate_fn=collate_det,
                       pin_memory=True,
                       persistent_workers=self.hparams.num_workers > 0,
                       timeout=120 if self.hparams.num_workers > 0 else 0),
            DataLoader(self.val_sets["crowdhuman_val"],
                       batch_size=self.hparams.batch_size,
                       shuffle=False,
                       num_workers=self.hparams.num_workers,
                       collate_fn=collate_det,
                       pin_memory=True,
                       persistent_workers=self.hparams.num_workers > 0,
                       timeout=120 if self.hparams.num_workers > 0 else 0),
        ]

    @property
    def val_set_names(self) -> list[str]:
        return ["mot17_val_half", "crowdhuman_val"]
