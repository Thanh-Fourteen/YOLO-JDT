"""Promote a Lightning checkpoint to a standalone YOLO11 state_dict.

Lightning ModelCheckpoint(save_top_k=1) leaves two files in the run dir:
    epoch=<N>-stepstep=<S>.ckpt   <- the best (highest val/mot17_val_half/mAP)
    last.ckpt                     <- the most recent

This script picks the non-`last.ckpt`, strips the `model.` Lightning prefix,
prefers EMA weights when present (via DetLitModule.on_save_checkpoint), and
writes a clean state_dict to `--dst` for use in downstream eval, tracker
init, and Phase 4+ extensions.

Usage:
    python -m yolo_jdt.scripts.promote_ckpt \\
        --src runs/baselines/step3a_yolo11s_70ep \\
        --dst weights/ours/yolo11s_det.pt --scale s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def find_best_ckpt(src_dir: Path) -> Path:
    """Return the top-1 checkpoint, or the most recent `last.ckpt` as fallback."""
    epoch_ckpts = sorted(
        [p for p in src_dir.glob("epoch=*.ckpt") if p.name != "last.ckpt"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if epoch_ckpts:
        return epoch_ckpts[0]
    last = src_dir / "last.ckpt"
    if last.exists():
        print(f"[promote_ckpt] WARN: no epoch=*.ckpt found, falling back to {last}",
              file=sys.stderr)
        return last
    raise FileNotFoundError(f"No .ckpt files in {src_dir}")


def extract_val_metrics(ckpt: dict) -> dict | None:
    """Best-effort extract the monitored val metric from a ModelCheckpoint callback."""
    cb = ckpt.get("callbacks", {})
    for key, state in cb.items():
        if "ModelCheckpoint" in key and isinstance(state, dict):
            return {
                "monitor": state.get("monitor"),
                "best_model_score": float(state["best_model_score"])
                    if state.get("best_model_score") is not None else None,
                "best_model_path": state.get("best_model_path"),
                "current_score": float(state["current_score"])
                    if state.get("current_score") is not None else None,
            }
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="Lightning run dir containing epoch=*.ckpt + last.ckpt")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Output path for the standalone state_dict")
    ap.add_argument("--scale", required=True, choices=["n", "s", "m", "l", "x"],
                    help="YOLO11 scale used for this training run")
    ap.add_argument("--nc", type=int, default=1, help="Number of classes (default 1=person-only)")
    ap.add_argument("--prefer", default="ema", choices=["ema", "online"],
                    help="Which weights to prefer when both are present (default: ema)")
    ap.add_argument("--model", default="det", choices=["det", "jde", "jdt"],
                    help="Model type for sanity-load: 'det' = YOLO11, 'jde' = JointHead, 'jdt' = YOLO_JDT")
    ap.add_argument("--cache-levels", default="P5",
                    choices=["P5", "P4+P5", "P3+P4+P5"],
                    help="TAGate cache levels (only used when --model=jdt)")
    ap.add_argument("--tagate-num-layers", type=int, default=2,
                    help="TAGate layers (only used when --model=jdt)")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"[promote_ckpt] --src is not a directory: {args.src}")
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    best = find_best_ckpt(args.src)
    print(f"[promote_ckpt] loading {best}")
    ckpt = torch.load(best, map_location="cpu", weights_only=False)

    val_metrics = extract_val_metrics(ckpt)
    if val_metrics:
        print(f"[promote_ckpt] val metric at save: {val_metrics}")

    has_ema = "ema_state_dict" in ckpt
    if args.prefer == "ema" and has_ema:
        source = "ema"
        raw_sd = ckpt["ema_state_dict"]
        print(f"[promote_ckpt] using EMA weights (updates={ckpt.get('ema_updates')})")
    else:
        if args.prefer == "ema" and not has_ema:
            print("[promote_ckpt] WARN: requested --prefer=ema but ckpt has no "
                  "'ema_state_dict' (older training run before EMA persistence "
                  "was added) — falling back to online weights")
        source = "online"
        raw_sd = ckpt["state_dict"]

    # Strip "model." Lightning prefix when present (online weights case).
    # EMA dict keys are already in YOLO11 namespace (backbone./neck./head.).
    cleaned_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("model."):
            cleaned_sd[k[len("model."):]] = v
        else:
            cleaned_sd[k] = v

    # Sanity-load into a fresh model to catch any mismatch BEFORE we save.
    if args.model == "jde":
        from yolo_jdt.train.jde_lightning_module import _YOLO11WithJointHead
        model = _YOLO11WithJointHead(scale=args.scale, nc=args.nc)
    elif args.model == "jdt":
        from yolo_jdt.models.yolo_jdt import YOLO_JDT
        model = YOLO_JDT(scale=args.scale, nc=args.nc,
                         cache_levels=args.cache_levels,
                         tagate_num_layers=args.tagate_num_layers)
    else:
        from yolo_jdt.models.yolo11 import YOLO11
        model = YOLO11(scale=args.scale, nc=args.nc)
    missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
    if unexpected:
        sys.exit(f"[promote_ckpt] FAIL: {len(unexpected)} unexpected keys: "
                 f"{unexpected[:5]}")
    if missing:
        sys.exit(f"[promote_ckpt] FAIL: {len(missing)} missing keys: "
                 f"{missing[:5]}")
    print(f"[promote_ckpt] sanity load OK: {len(cleaned_sd)} keys, "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    payload = {
        "state_dict": cleaned_sd,
        "scale": args.scale,
        "nc": args.nc,
        "source_ckpt": str(best),
        "source_weights": source,           # "ema" or "online"
        "val_metrics": val_metrics,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
    }
    # Store TAGate metadata when promoting a JDT checkpoint
    if args.model == "jdt":
        payload["cache_levels"] = args.cache_levels
        payload["tagate_num_layers"] = args.tagate_num_layers
    # Carry over reid_classifier for JDE/JDT so infer scripts can use it
    if "reid_classifier_state_dict" in ckpt:
        payload["reid_classifier_state_dict"] = ckpt["reid_classifier_state_dict"]
    torch.save(payload, args.dst)
    print(f"[promote_ckpt] wrote {args.dst} ({args.dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
