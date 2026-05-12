"""DanceTrack standard-format Dataset.

Splits: train (40 seq), val (25 seq), test (35 seq, no GT).
"""
from __future__ import annotations

from pathlib import Path

from yolo_jdt.data.datasets._base import StandardSeqDataset

_VALID_SPLITS = {"train", "val", "test"}


class DanceTrackDataset(StandardSeqDataset):
    is_static = False
    name = "dancetrack"

    def __init__(self, root: str | Path = "datasets/standard", *, split: str = "train"):
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be in {_VALID_SPLITS}, got {split}")
        super().__init__(Path(root) / "dancetrack", split)
