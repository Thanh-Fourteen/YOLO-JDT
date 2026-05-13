"""Run a tracker over a standard-format dataset split and emit MOT-format
prediction files (one txt per sequence) ready to be passed to TrackEval.

Per sequence:
    - Load detector weights once (promoted YOLO11 .pt)
    - Iterate frames in temporal order from the standard JSON
    - For each frame: letterbox -> model.forward -> multi_label NMS ->
      un-letterbox to original pixel coords -> tracker.update
    - Append confirmed tracks to a per-frame record list
    - At seq end: write MOT-format txt to `<output_dir>/<seq_name>.txt`
      with frame_ids renumbered to 1..N (same offset as the cached GT,
      via `compute_frame_offset`)

Usage:
    python -m yolo_jdt.scripts.infer_tracking \\
        --weights runs/baselines/step3a_yolo11s_70ep/weights/promoted.pt \\
        --scale s \\
        --dataset mot17 --split val_half \\
        --tracker bytetrack \\
        --output-dir runs/baselines/step3bcd_yolo11s_bytetrack_mot17/tracker_outputs \\
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

from yolo_jdt.data.augment import letterbox
from yolo_jdt.eval.mot_format import compute_frame_offset, write_tracker_mot_txt
from yolo_jdt.models.yolo11 import YOLO11
from yolo_jdt.tracker.bytetrack import ByteTrackConfig, ByteTrackTracker
from yolo_jdt.tracker.track import reset_id_counter
from yolo_jdt.train.lightning_module import _multi_label_nms


def build_tracker(name: str):
    if name == "bytetrack":
        return lambda: ByteTrackTracker(ByteTrackConfig())
    if name == "botsort":
        from yolo_jdt.tracker.botsort import BoTSORTConfig, BoTSORTTracker  # noqa
        return lambda: BoTSORTTracker(BoTSORTConfig())
    raise ValueError(f"unknown tracker: {name}")


def load_detector(weights_path: Path, scale: str, device: torch.device):
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
        nc = payload.get("nc", 1)
    else:
        sd, nc = payload, 1
    model = YOLO11(scale=scale, nc=nc).to(device).eval()
    model.load_state_dict(sd, strict=True)
    return model, nc


def infer_one_seq(model, device, dtype, json_path: Path, image_root: Path,
                  split: str, tracker_factory, conf: float, iou: float,
                  max_det: int, imgsz: int, log_every: int = 100
                  ) -> tuple[list[tuple], dict]:
    """Run tracker over one sequence. Returns (records, fps_breakdown)."""
    with open(json_path) as f:
        seq = json.load(f)
    offset = compute_frame_offset(json_path)

    # Reset global track ID counter per-sequence so IDs in MOT txt are 1-based per seq
    reset_id_counter()
    tracker = tracker_factory()
    records: list[tuple] = []

    times = {"load": [], "preproc": [], "forward": [], "nms": [],
             "postproc": [], "track": []}
    n_frames = len(seq["frames"])

    for fi, frame in enumerate(seq["frames"]):
        orig_fid = int(frame["frame_id"])
        renumbered_fid = orig_fid - offset

        # 1. load
        t0 = time.perf_counter()
        img_path = image_root / "images" / split / frame["image"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARN: failed to load {img_path}, skipping frame")
            continue
        H0, W0 = img_bgr.shape[:2]
        t1 = time.perf_counter()
        times["load"].append(t1 - t0)

        # 2. letterbox + tensor
        canvas, ratio, (pad_l, pad_t) = letterbox(img_bgr, new_shape=imgsz, scaleup=True)
        # BGR → RGB → CHW float in [0,1]
        x = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1)
        x = x.float().div_(255.0).unsqueeze(0).to(device, dtype=dtype, non_blocking=True)
        t2 = time.perf_counter()
        times["preproc"].append(t2 - t1)

        # 3. forward
        with torch.no_grad():
            decoded, _ = model(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t3 = time.perf_counter()
        times["forward"].append(t3 - t2)

        # 4. NMS in normalized coords (matches DetLitModule.validation_step)
        decoded_f = decoded.float()
        decoded_norm = decoded_f.clone()
        decoded_norm[:, [0, 2], :] /= float(imgsz)
        decoded_norm[:, [1, 3], :] /= float(imgsz)
        preds = _multi_label_nms(decoded_norm, conf_thr=conf, iou_thr=iou,
                                  max_det=max_det)[0]
        t4 = time.perf_counter()
        times["nms"].append(t4 - t3)

        # 5. un-letterbox → original pixel coords → xywh format for tracker
        boxes_xyxy_n = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        if len(boxes_xyxy_n):
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
        t5 = time.perf_counter()
        times["postproc"].append(t5 - t4)

        # 6. tracker update (pass original BGR frame for trackers that need
        #    it for CMC; base ByteTrack ignores it)
        active = tracker.update(dets, frame_id=renumbered_fid, frame=img_bgr)
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

    # Aggregate FPS breakdown
    def _stats(arr):
        if not arr:
            return {"p50": 0.0, "p99": 0.0, "mean": 0.0}
        a = np.asarray(arr)
        return {"p50": float(np.percentile(a, 50) * 1000),  # ms
                "p99": float(np.percentile(a, 99) * 1000),
                "mean": float(a.mean() * 1000)}

    fps_breakdown = {stage: _stats(v) for stage, v in times.items()}
    total_per_frame_s = sum(np.asarray(v).mean() if v else 0 for v in times.values())
    fps_breakdown["overall_fps"] = (1.0 / total_per_frame_s) if total_per_frame_s > 0 else 0.0
    fps_breakdown["n_frames"] = n_frames

    return records, fps_breakdown


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--scale", required=True, choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--dataset", required=True,
                    choices=["mot17", "mot20", "dancetrack"])
    ap.add_argument("--split", required=True,
                    help="e.g. val_half / val")
    ap.add_argument("--tracker", required=True, choices=["bytetrack", "botsort"])
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="dir to write per-seq MOT txt files")
    ap.add_argument("--standard-root", type=Path, default=Path("datasets/standard"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05,
                    help="NMS conf threshold (tracker-typical 0.05; vs 0.001 for mAP)")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="Optional whitelist of seq names (default: all)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    print(f"[infer_tracking] device={device}, dtype={dtype}, tracker={args.tracker}")
    print(f"[infer_tracking] dataset={args.dataset}/{args.split}, weights={args.weights}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load detector once
    model, nc = load_detector(args.weights, args.scale, device)
    model = model.to(dtype)
    print(f"[infer_tracking] loaded YOLO11{args.scale}, nc={nc}, "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Iterate sequences
    anno_dir = args.standard_root / args.dataset / "annotations" / args.split
    image_root = args.standard_root / args.dataset
    json_paths = sorted(anno_dir.glob("*.json"))
    if args.seqs:
        json_paths = [p for p in json_paths if p.stem in set(args.seqs)]

    tracker_factory = build_tracker(args.tracker)

    all_fps = {}
    for json_path in json_paths:
        seq_name = json_path.stem
        print(f"\n[infer_tracking] === {seq_name} ===")
        records, fps = infer_one_seq(
            model, device, dtype, json_path, image_root, args.split,
            tracker_factory, args.conf, args.iou, args.max_det, args.imgsz)
        out_txt = args.output_dir / f"{seq_name}.txt"
        n = write_tracker_mot_txt(records, out_txt)
        print(f"  wrote {out_txt} ({n} rows, fps={fps['overall_fps']:.1f})")
        all_fps[seq_name] = fps

    # Write fps_breakdown.json next to outputs (parent of output_dir)
    fps_out = args.output_dir.parent / "fps_breakdown.json"
    fps_out.write_text(json.dumps(all_fps, indent=2))
    print(f"\n[infer_tracking] wrote {fps_out}")
    print(f"[infer_tracking] DONE")


if __name__ == "__main__":
    main()
