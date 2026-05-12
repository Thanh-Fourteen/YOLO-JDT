"""CrowdHuman standard-format Dataset.

Static-image dataset. All track_ids are -1 (no temporal continuity).
Splits: train (15,000), val (4,370). Image resolution is variable —
the base class reads per-frame image_size when present in the JSON.
"""
from __future__ import annotations

from pathlib import Path

from yolo_jdt.data.datasets._base import StandardSeqDataset

_VALID_SPLITS = {"train", "val"}


class CrowdHumanDataset(StandardSeqDataset):
    is_static = True
    name = "crowdhuman"

    def __init__(self, root: str | Path = "datasets/standard", *, split: str = "train"):
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be in {_VALID_SPLITS}, got {split}")
        super().__init__(Path(root) / "crowdhuman", split)
