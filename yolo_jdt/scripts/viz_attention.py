"""Visualize TAGate cross-attention weights on DanceTrack sequences.

For each frame pair (t, t-1) in 5 selected DanceTrack sequences, extracts
attention weights from all TAGate CrossAttentionBlock layers and saves overlay
heatmaps to runs/viz/tagate_attention/.

Attention heatmap interpretation:
  - Each query position in F_t attends over all key positions in F_prev.
  - We visualize the "total attention received by each F_prev position" =
    sum over query positions of the mean-head attention weights.
  - This shows which regions in the previous frame the model found most
    informative for the current frame's temporal context.

Output per frame:
  <out_dir>/<seq_name>/frame<NNNN>_t.jpg      — current frame
  <out_dir>/<seq_name>/frame<NNNN>_prev.jpg   — previous frame
  <out_dir>/<seq_name>/frame<NNNN>_attn.jpg   — heatmap overlay on prev frame

Usage:
    python -m yolo_jdt.scripts.viz_attention \\
        --weights weights/ours/yolo11s_jdt.pt --scale s \\
        --out-dir runs/viz/tagate_attention \\
        --seqs dancetrack0005 dancetrack0019 dancetrack0035 dancetrack0040 dancetrack0073 \\
        --frames-per-seq 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from yolo_jdt.data.augment import letterbox
from yolo_jdt.models.tagate.cross_attn import CrossAttentionBlock
from yolo_jdt.models.yolo_jdt import YOLO_JDT
from yolo_jdt.scripts.infer_tracking_jdt import load_jdt_model


def _build_heatmap(attn_weights: list[torch.Tensor], H: int, W: int) -> np.ndarray:
    """Aggregate attention from all TAGate layers into a single [H, W] heatmap.

    attn_weights: list of [1, num_heads, L, L] tensors (one per TAGate layer × level).
    Returns float32 [H, W] heatmap normalized to [0, 1].
    """
    L = H * W
    combined = torch.zeros(L)
    for attn in attn_weights:
        # attn: [1, nh, L, L] — mean over heads, then sum over queries → [L]
        per_pos = attn[0].mean(0).sum(0).float()  # [L] attention received per F_prev pos
        combined += per_pos.cpu()
    # Normalize
    mn, mx = combined.min(), combined.max()
    if mx > mn:
        combined = (combined - mn) / (mx - mn)
    return combined.numpy().reshape(H, W)


def _overlay_heatmap(bgr_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55
                      ) -> np.ndarray:
    """Overlay a [H_feat, W_feat] float32 heatmap onto a BGR image (any size)."""
    H_img, W_img = bgr_img.shape[:2]
    hm_u8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    hm_u8 = cv2.resize(hm_u8, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
    colormap = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)   # blue→red
    return cv2.addWeighted(bgr_img, 1 - alpha, colormap, alpha, 0)


def viz_sequence(model: YOLO_JDT, device: torch.device, dtype: torch.dtype,
                 json_path: Path, image_root: Path, split: str,
                 out_dir: Path, frames_per_seq: int = 50, imgsz: int = 640):
    """Generate attention heatmaps for one sequence."""
    with open(json_path) as f:
        seq = json.load(f)
    seq_name = seq["name"]
    seq_out = out_dir / seq_name
    seq_out.mkdir(parents=True, exist_ok=True)

    # Enable attention capture on all CrossAttentionBlock instances
    CrossAttentionBlock.capture_attention = True
    cross_attn_blocks = [m for m in model.modules()
                         if isinstance(m, CrossAttentionBlock)]

    cache = model.zero_cache(batch_size=1, device=device, dtype=dtype)
    n_frames = min(frames_per_seq, len(seq["frames"]))
    saved = 0

    for fi in range(1, len(seq["frames"])):     # start at 1 to have a real prev frame
        if saved >= n_frames:
            break
        frame_cur = seq["frames"][fi]
        frame_prv = seq["frames"][fi - 1]

        def _load(info) -> np.ndarray:
            p = image_root / "images" / split / info["image"]
            img = cv2.imread(str(p))
            return img if img is not None else np.zeros((360, 640, 3), dtype=np.uint8)

        img_cur_bgr = _load(frame_cur)
        img_prv_bgr = _load(frame_prv)

        canvas, _, _ = letterbox(img_cur_bgr, new_shape=imgsz, scaleup=True)
        x = (torch.from_numpy(canvas[:, :, ::-1].copy())
             .permute(2, 0, 1).float().div_(255.0)
             .unsqueeze(0).to(device, dtype=dtype))

        with torch.no_grad():
            _, _, _, _, features_to_cache = model(x, cache)
        cache = features_to_cache

        # Collect attention from all cross-attention blocks (after this forward)
        # _last_attn_weights is set on each CrossAttentionBlock during this forward
        attn_list = []
        for blk in cross_attn_blocks:
            if hasattr(blk, "_last_attn_weights") and blk._last_attn_weights is not None:
                attn_list.append(blk._last_attn_weights)

        if not attn_list:
            continue

        # The P5 level has 20×20 spatial dims for 640-input
        # Use the first block's weight to determine spatial dims
        L = attn_list[0].shape[-1]
        H_feat = W_feat = int(L ** 0.5)
        if H_feat * W_feat != L:
            continue   # non-square; skip

        heatmap = _build_heatmap(attn_list, H_feat, W_feat)  # [H_feat, W_feat]

        fid = int(frame_cur["frame_id"])
        cv2.imwrite(str(seq_out / f"frame{fid:04d}_t.jpg"), img_cur_bgr)
        cv2.imwrite(str(seq_out / f"frame{fid:04d}_prev.jpg"), img_prv_bgr)
        overlay = _overlay_heatmap(img_prv_bgr, heatmap)
        cv2.imwrite(str(seq_out / f"frame{fid:04d}_attn.jpg"), overlay)
        saved += 1

        if saved % 10 == 0:
            print(f"  [{seq_name}] saved {saved}/{n_frames} frames", flush=True)

    CrossAttentionBlock.capture_attention = False
    print(f"  [{seq_name}] done — {saved} frames saved to {seq_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--scale", default="s", choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--out-dir", type=Path, default=Path("runs/viz/tagate_attention"))
    ap.add_argument("--standard-root", type=Path, default=Path("datasets/standard"))
    ap.add_argument("--dataset", default="dancetrack")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seqs", nargs="+",
                    default=["dancetrack0005", "dancetrack0019",
                             "dancetrack0035", "dancetrack0040", "dancetrack0073"],
                    help="5 DanceTrack non-linear sequences for paper figure")
    ap.add_argument("--frames-per-seq", type=int, default=50)
    ap.add_argument("--cache-levels", default="P5",
                    choices=["P5", "P4+P5", "P3+P4+P5"])
    ap.add_argument("--tagate-num-layers", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    model, _ = load_jdt_model(args.weights, args.scale, device,
                               cache_levels=args.cache_levels,
                               tagate_num_layers=args.tagate_num_layers)
    model = model.to(dtype).eval()
    print(f"[viz_attention] model loaded, scale={args.scale}, "
          f"cache={args.cache_levels}, layers={args.tagate_num_layers}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    anno_dir = args.standard_root / args.dataset / "annotations" / args.split
    image_root = args.standard_root / args.dataset
    json_paths = sorted(anno_dir.glob("*.json"))
    if args.seqs:
        json_paths = [p for p in json_paths if p.stem in set(args.seqs)]

    if not json_paths:
        print(f"[viz_attention] no matching sequences found in {anno_dir}")
        return

    for json_path in json_paths:
        print(f"\n[viz_attention] === {json_path.stem} ===")
        viz_sequence(model, device, dtype, json_path, image_root, args.split,
                     args.out_dir, frames_per_seq=args.frames_per_seq,
                     imgsz=args.imgsz)

    print(f"\n[viz_attention] all done — output in {args.out_dir}")


if __name__ == "__main__":
    main()
