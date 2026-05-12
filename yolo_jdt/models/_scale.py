"""Per-scale (n/s/m/l/x) configuration for YOLO11.

Mirrors `ultralytics/cfg/models/11/yolo11.yaml` `scales:` table and the
channel/depth scaling logic in `ultralytics/nn/tasks.py::parse_model`.

Usage:
    from yolo_jdt.models._scale import scale_params
    p = scale_params("s")
    # p = {"depth": 0.5, "width": 0.5, "max_channels": 1024,
    #      "force_c3k": False}
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleConfig:
    depth: float
    width: float
    max_channels: int
    force_c3k: bool   # True for m/l/x — overrides per-block c3k=False in yaml


_SCALES: dict[str, ScaleConfig] = {
    "n": ScaleConfig(0.50, 0.25, 1024, False),
    "s": ScaleConfig(0.50, 0.50, 1024, False),
    "m": ScaleConfig(0.50, 1.00,  512, True),
    "l": ScaleConfig(1.00, 1.00,  512, True),
    "x": ScaleConfig(1.00, 1.50,  512, True),
}


def scale_params(scale: str) -> ScaleConfig:
    if scale not in _SCALES:
        raise ValueError(f"unknown scale {scale!r}, expected one of {list(_SCALES)}")
    return _SCALES[scale]


def make_divisible(x, divisor: int = 8) -> int:
    """Round x up to nearest multiple of divisor (matches Ultralytics ops.make_divisible)."""
    import math
    return int(math.ceil(x / divisor) * divisor)


def scale_channels(c: int, cfg: ScaleConfig) -> int:
    """Apply width + max_channels capping."""
    return make_divisible(min(c, cfg.max_channels) * cfg.width, 8)


def scale_repeats(n: int, cfg: ScaleConfig) -> int:
    """Apply depth multiplier (matches `n_ = max(round(n*depth), 1) if n > 1 else n`)."""
    return max(round(n * cfg.depth), 1) if n > 1 else n
