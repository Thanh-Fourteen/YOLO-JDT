"""Map an Ultralytics YOLO11 state_dict to our standalone module structure.

The Ultralytics .pt files store a pickled `DetectionModel` instance whose
`state_dict()` keys follow the `model.<layer_idx>.<tail>` convention. Our
`yolo_jdt.models.YOLO11` splits those layers into `backbone.layer{N}`,
`neck.layer{N}`, and `head.<tail>` namespaces.

The mapping is layer-index renaming only — every Ultralytics key has exactly
one destination key, and shapes are guaranteed identical because our extracted
modules preserve the upstream class definitions verbatim.

Anything that fails to map (unexpected key, shape mismatch, missing target)
raises immediately. There is no silent drop.
"""
from __future__ import annotations

from pathlib import Path

import torch

# Ultralytics' YOLO11 (cfg/models/11/yolo11.yaml) fixed layer-index → namespace.
# Layers 11, 12, 14, 15, 18, 21 are Upsample / Concat — no params, no keys.
_BACKBONE_LAYERS = set(range(0, 11))
_NECK_LAYERS = {13, 16, 17, 19, 20, 22}
_HEAD_LAYER = 23


def _key_destination(src_key: str) -> str:
    """Map a single Ultralytics key to our key. Raises ValueError if unmapped."""
    if not src_key.startswith("model."):
        raise ValueError(f"unexpected key (no 'model.' prefix): {src_key}")
    after = src_key[len("model."):]
    layer_str, _, tail = after.partition(".")
    layer = int(layer_str)
    if layer in _BACKBONE_LAYERS:
        return f"backbone.layer{layer}.{tail}"
    if layer in _NECK_LAYERS:
        return f"neck.layer{layer}.{tail}"
    if layer == _HEAD_LAYER:
        return f"head.{tail}"
    raise ValueError(
        f"key references unmapped Ultralytics layer {layer}: {src_key} — "
        "expected 0-10 (backbone), 13/16/17/19/20/22 (neck), 23 (head)")


def map_ultralytics_state_dict(src: dict[str, torch.Tensor],
                               dst: dict[str, torch.Tensor]
                               ) -> dict[str, torch.Tensor]:
    """Rename keys + verify shapes line up against `dst`.

    Args:
        src: Ultralytics `model.{i}.{...}` state_dict.
        dst: our YOLO11 model's state_dict (used only for shape reference).

    Returns:
        New dict with renamed keys + tensors copied from `src`.

    Raises:
        ValueError: any key cannot be mapped, or any shape mismatches the
            destination, or any destination key would be left unset, or
            unmapped extra keys exist.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in src.items():
        new_k = _key_destination(k)
        if new_k not in dst:
            raise ValueError(f"mapped key not present in target model: {k} -> {new_k}")
        if dst[new_k].shape != v.shape:
            raise ValueError(
                f"shape mismatch for {k} -> {new_k}: src {tuple(v.shape)} vs "
                f"dst {tuple(dst[new_k].shape)}")
        out[new_k] = v
    missing = set(dst.keys()) - set(out.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} target keys not produced by mapping; sample: "
            f"{sorted(missing)[:5]}")
    return out


def load_yolo11_weights(model: torch.nn.Module, ckpt_path: str | Path) -> dict:
    """Load weights from an Ultralytics .pt file into our YOLO11 model.

    The .pt file is a pickled `DetectionModel`; deserializing it requires the
    `ultralytics` package to be importable. After this one-time import, our
    runtime no longer depends on Ultralytics.

    Returns the metadata dict from the checkpoint (date, version, license, etc.)
    for logging.
    """
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    if "model" not in ckpt:
        raise ValueError(f"checkpoint missing 'model' key: {ckpt_path}")
    src = ckpt["model"].float().state_dict()
    dst = model.state_dict()
    mapped = map_ultralytics_state_dict(src, dst)
    model.load_state_dict(mapped, strict=True)
    return {k: v for k, v in ckpt.items() if k != "model"}
