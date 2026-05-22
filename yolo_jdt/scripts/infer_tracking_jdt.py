"""Run YOLO-JDT (TAGate + JointHead) + BoT-SORT-ReID over a dataset split.

Key difference from infer_tracking_jde.py: maintains a rolling feature cache
across frames within each sequence.  The neck features from frame t become the
cached_features_prev for frame t+1, giving TAGate genuine temporal context.

Usage:
    python -m yolo_jdt.scripts.infer_tracking_jdt \\
        --weights weights/ours/yolo11s_jdt.pt --scale s \\
        --dataset mot17 --split val_half \\
        --output-dir runs/tagate/step5_mot17/tracker_outputs \\
        --conf 0.05
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from yolo_jdt.data.augment import letterbox
from yolo_jdt.eval.mot_format import compute_frame_offset, write_tracker_mot_txt
from yolo_jdt.models.yolo_jdt import YOLO_JDT
from yolo_jdt.tracker.botsort_reid import BoTSORTReIDConfig, BoTSORTReIDTracker
from yolo_jdt.tracker.track import reset_id_counter


def load_jdt_model(weights_path: Path, scale: str, device: torch.device,
                   cache_levels: str = "P5", tagate_num_layers: int = 2):
    """Load YOLO_JDT from a promoted .pt or Lightning .ckpt."""
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        src = payload["state_dict"]
        nc = payload.get("nc", 1)
        cl = payload.get("cache_levels", cache_levels)
        nl = payload.get("tagate_num_layers", tagate_num_layers)
    else:
        src, nc, cl, nl = payload, 1, cache_levels, tagate_num_layers

    if any(k.startswith("model.") for k in src):
        src = {k[len("model."):]: v for k, v in src.items() if k.startswith("model.")}

    model = YOLO_JDT(scale=scale, nc=nc, cache_levels=cl,
                     tagate_num_layers=nl).to(device).eval()
    result = model.load_state_dict(src, strict=False)
    if result.missing_keys:
        print(f"[infer_jdt] WARN missing {len(result.missing_keys)} keys: "
              f"{result.missing_keys[:5]}")
    # Use EMA weights if present
    if isinstance(payload, dict) and "ema_state_dict" in payload:
        ema_src = payload["ema_state_dict"]
        if any(k.startswith("model.") for k in ema_src):
            ema_src = {k[len("model."):]: v for k, v in ema_src.items()
                       if k.startswith("model.")}
        result2 = model.load_state_dict(ema_src, strict=False)
        n_loaded = len(ema_src) - len(result2.missing_keys)
        print(f"[infer_jdt] applied EMA weights ({n_loaded} keys)")
    return model, nc


def _nms_with_anchor_idx(decoded: torch.Tensor, conf_thr: float = 0.05,
                          iou_thr: float = 0.7, max_det: int = 300) -> list[dict]:
    """NMS for nc=1 with surviving anchor index tracking (for ReID gather)."""
    bs, ch, A = decoded.shape
    assert ch - 4 == 1, "infer_tracking_jdt only supports nc=1 (person-only)"
    device = decoded.device
    out = []
    for i in range(bs):
        x = decoded[i].T    # [A, 5]
        boxes_xywh = x[:, :4]
        scores = x[:, 4]
        keep_mask = scores > conf_thr
        orig_idx = torch.arange(A, device=device)[keep_mask]
        boxes_xywh = boxes_xywh[keep_mask]
        scores = scores[keep_mask]
        if boxes_xywh.numel() == 0:
            out.append({
                "boxes": boxes_xywh.new_zeros((0, 4)),
                "scores": scores.new_zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                "anchor_idx": torch.zeros((0,), dtype=torch.int64, device=device),
            })
            continue
        cx, cy, w, h = boxes_xywh.unbind(1)
        boxes_xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)
        kept = nms(boxes_xyxy, scores, iou_thr)[:max_det]
        out.append({
            "boxes": boxes_xyxy[kept],
            "scores": scores[kept],
            "labels": torch.zeros_like(kept, dtype=torch.int64),
            "anchor_idx": orig_idx[kept],
        })
    return out


def infer_one_seq(model: YOLO_JDT, device: torch.device, dtype: torch.dtype,
                  json_path: Path, image_root: Path, split: str,
                  tracker_factory, conf: float, iou: float,
                  max_det: int, imgsz: int, log_every: int = 100
                  ) -> tuple[list[tuple], dict]:
    """Run JDT pipeline over one sequence with rolling feature cache."""
    with open(json_path) as f:
        seq = json.load(f)
    offset = compute_frame_offset(json_path)

    reset_id_counter()
    tracker = tracker_factory()
    records: list[tuple] = []
    times = {"load": [], "preproc": [], "forward": [], "nms": [],
             "postproc": [], "track": []}
    n_frames = len(seq["frames"])

    # Rolling cache: initialized to zeros at the start of each sequence
    cache = model.zero_cache(batch_size=1, device=device, dtype=dtype)

    for fi, frame in enumerate(seq["frames"]):
        orig_fid = int(frame["frame_id"])
        renumbered_fid = orig_fid - offset

        t0 = time.perf_counter()
        img_path = image_root / "images" / split / frame["image"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARN: failed to load {img_path}")
            continue
        H0, W0 = img_bgr.shape[:2]
        t1 = time.perf_counter()
        times["load"].append(t1 - t0)

        canvas, ratio, (pad_l, pad_t) = letterbox(img_bgr, new_shape=imgsz, scaleup=True)
        x = (torch.from_numpy(canvas[:, :, ::-1].copy())
             .permute(2, 0, 1).float().div_(255.0)
             .unsqueeze(0).to(device, dtype=dtype, non_blocking=True))
        t2 = time.perf_counter()
        times["preproc"].append(t2 - t1)

        with torch.no_grad():
            decoded, _, reid_per_level, offset_out, features_to_cache = model(x, cache)
        # Update rolling cache for next frame
        cache = features_to_cache
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        times["forward"].append(t3 - t2)

        decoded_f = decoded.float()
        decoded_norm = decoded_f.clone()
        decoded_norm[:, [0, 2], :] /= float(imgsz)
        decoded_norm[:, [1, 3], :] /= float(imgsz)
        preds = _nms_with_anchor_idx(decoded_norm, conf_thr=conf, iou_thr=iou,
                                      max_det=max_det)[0]
        t4 = time.perf_counter()
        times["nms"].append(t4 - t3)

        boxes_xyxy_n = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        anchor_idx = preds["anchor_idx"]

        if len(boxes_xyxy_n) > 0:
            reid_flat = torch.cat(
                [reid_per_level[lvl][0].view(reid_per_level[lvl].shape[1], -1)
                 for lvl in range(len(reid_per_level))],
                dim=1,
            )   # [reid_dim, A]
            embs = reid_flat[:, anchor_idx].T.float().cpu().numpy()

            # Track-offset: gather per-detection (Δx, Δy) at the surviving
            # anchors. Offsets are normalized [0,1] in canvas space; a
            # displacement, so letterbox padding cancels — only ÷ratio is
            # needed to reach original-image pixels.
            offset_flat = torch.cat(
                [offset_out[lvl][0].reshape(2, -1)
                 for lvl in range(len(offset_out))],
                dim=1,
            )   # [2, A]
            offs = offset_flat[:, anchor_idx].T.float().cpu().numpy()  # [N, 2]
            offs_orig = offs * float(imgsz) / ratio

            boxes_640 = boxes_xyxy_n * float(imgsz)
            boxes_orig = boxes_640.copy()
            boxes_orig[:, [0, 2]] = (boxes_640[:, [0, 2]] - pad_l) / ratio
            boxes_orig[:, [1, 3]] = (boxes_640[:, [1, 3]] - pad_t) / ratio
            boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]].clip(0, W0)
            boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]].clip(0, H0)
            x1, y1, x2, y2 = boxes_orig.T
            dets = np.stack([x1, y1, x2 - x1, y2 - y1, scores], axis=1)
        else:
            dets = np.empty((0, 5))
            embs = None
            offs_orig = None
        t5 = time.perf_counter()
        times["postproc"].append(t5 - t4)

        active = tracker.update(dets, frame_id=renumbered_fid,
                                 frame=img_bgr, embeddings=embs,
                                 offsets=offs_orig)
        for tr in active:
            xb, yb, wb, hb = tr.measurement_xywh
            records.append((renumbered_fid, tr.track_id,
                             float(xb), float(yb), float(wb), float(hb),
                             float(tr.score)))
        t6 = time.perf_counter()
        times["track"].append(t6 - t5)

        if (fi + 1) % log_every == 0 or fi + 1 == n_frames:
            elapsed = sum(sum(v) for v in times.values())
            fps = (fi + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  [{seq['name']}] frame {fi+1}/{n_frames}  fps={fps:.1f}", flush=True)

    def _stats(arr):
        if not arr:
            return {"p50": 0.0, "p99": 0.0, "mean": 0.0}
        a = np.asarray(arr)
        return {"p50": float(np.percentile(a, 50) * 1000),
                "p99": float(np.percentile(a, 99) * 1000),
                "mean": float(a.mean() * 1000)}

    fps_breakdown = {stage_: _stats(v) for stage_, v in times.items()}
    total_per_frame = sum(np.asarray(v).mean() if v else 0 for v in times.values())
    fps_breakdown["overall_fps"] = (1.0 / total_per_frame) if total_per_frame > 0 else 0.0
    fps_breakdown["n_frames"] = n_frames
    return records, fps_breakdown


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--scale", required=True, choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--dataset", required=True,
                    choices=["mot17", "mot20", "dancetrack"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--standard-root", type=Path, default=Path("datasets/standard"))
    ap.add_argument("--cache-levels", default="P5",
                    choices=["P5", "P4+P5", "P3+P4+P5"])
    ap.add_argument("--tagate-num-layers", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--seqs", nargs="+", default=None)
    # Associator cost weights. --w-offset 0 disables the track-offset cue
    # (→ exact BoT-SORT-ReID, for the A/B ablation).
    ap.add_argument("--w-iou", type=float, default=0.5)
    ap.add_argument("--w-reid", type=float, default=0.3)
    ap.add_argument("--w-offset", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32
    print(f"[infer_jdt] device={device}, dtype={dtype}")
    print(f"[infer_jdt] dataset={args.dataset}/{args.split}, weights={args.weights}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, nc = load_jdt_model(args.weights, args.scale, device,
                                cache_levels=args.cache_levels,
                                tagate_num_layers=args.tagate_num_layers)
    model = model.to(dtype)
    print(f"[infer_jdt] loaded YOLO_JDT scale={args.scale} nc={nc} "
          f"cache={args.cache_levels} layers={args.tagate_num_layers} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    anno_dir = args.standard_root / args.dataset / "annotations" / args.split
    image_root = args.standard_root / args.dataset
    json_paths = sorted(anno_dir.glob("*.json"))
    if args.seqs:
        json_paths = [p for p in json_paths if p.stem in set(args.seqs)]

    print(f"[infer_jdt] associator weights: w_iou={args.w_iou} "
          f"w_reid={args.w_reid} w_offset={args.w_offset}")
    tracker_factory = lambda: BoTSORTReIDTracker(BoTSORTReIDConfig(
        w_iou=args.w_iou, w_reid=args.w_reid, w_offset=args.w_offset))

    all_fps = {}
    for json_path in json_paths:
        seq_name = json_path.stem
        print(f"\n[infer_jdt] === {seq_name} ===")
        records, fps = infer_one_seq(
            model, device, dtype, json_path, image_root, args.split,
            tracker_factory, args.conf, args.iou, args.max_det, args.imgsz)
        out_txt = args.output_dir / f"{seq_name}.txt"
        n = write_tracker_mot_txt(records, out_txt)
        print(f"  wrote {out_txt} ({n} rows, fps={fps['overall_fps']:.1f})")
        all_fps[seq_name] = fps

    fps_out = args.output_dir.parent / "fps_breakdown.json"
    fps_out.write_text(json.dumps(all_fps, indent=2))
    print(f"\n[infer_jdt] wrote {fps_out}")
    print(f"[infer_jdt] DONE")


if __name__ == "__main__":
    main()
