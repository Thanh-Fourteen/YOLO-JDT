"""MOT17 / MOT20 standard-format Dataset.

Splits available: train, train_half, val_half, test (test has empty
object lists since GT is private).
"""
from __future__ import annotations

from pathlib import Path

from yolo_jdt.data.datasets._base import StandardSeqDataset

_VALID_NAMES = {"mot17", "mot20"}
_VALID_SPLITS = {"train", "train_half", "val_half", "test"}


class MOTDataset(StandardSeqDataset):
    is_static = False

    def __init__(self, root: str | Path = "datasets/standard", *,
                 name: str = "mot17", split: str = "train_half"):
        if name not in _VALID_NAMES:
            raise ValueError(f"name must be in {_VALID_NAMES}, got {name}")
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be in {_VALID_SPLITS}, got {split}")
        super().__init__(Path(root) / name, split)
        self.name = name
