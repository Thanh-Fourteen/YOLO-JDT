"""Step 4.E — COCO val2017 forward integrity test for JointHead.

Verifies that adding the cv4 ReID branch is **purely additive** to the
detection forward path: COCO mAP with a JointHead-equipped YOLO11s
loaded from Ultralytics COCO weights should match the Phase 2 baseline
to within ±0.05% (the "purely additive change" claim for the paper).

Pre-Phase-2 baselines (from `runs/eval/coco_val2017_yolo11{s,m}.json`):
    YOLO11s  46.63 mAP  (target band 47.0 ± 0.5)
    YOLO11m  51.32 mAP  (target band 51.5 ± 0.5)

The cv4 branch reads from the original FPN feature maps `x[i]` BEFORE
cv2/cv3 mutate them (see JointHead.forward), so detection should be
bit-identical. This script proves it empirically.

Usage:
    python -m yolo_jdt.scripts.eval_coco_integrity \\
        --weights weights/pretrained/yolo11s.pt --scale s \\
        --output runs/integrity/step4_coco_yolo11s.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolo_jdt.data.augment import letterbox
from yolo_jdt.models.backbone.yolo11 import YOLO11Backbone
from yolo_jdt.models.head.joint_head import JointHead
from yolo_jdt.models.neck.panet import YOLO11PANet
from yolo_jdt.scripts.eval_coco import COCO_CONTIGUOUS_TO_ID, postprocess


class YOLO11WithJointHead(nn.Module):
    """Same assembly as YOLO11 but with JointHead instead of DecoupledDetect."""

    def __init__(self, scale: str = "s", nc: int = 80, reg_max: int = 16):
        super().__init__()
        self.scale = scale
        self.nc = nc
        self.backbone = YOLO11Backbone(scale)
        self.neck = YOLO11PANet(scale, in_channels=self.backbone.out_channels)
        self.head = JointHead(nc=nc, ch=self.neck.out_channels,
                               reg_max=reg_max, strides=(8.0, 16.0, 32.0))

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        feat16, feat19, feat22 = self.neck(p3, p4, p5)
        return self.head([feat16, feat19, feat22])


def _load_partial_from_ultralytics(model: nn.Module, ckpt_path: str):
    """Load Ultralytics COCO weights into JointHead-equipped model.
    cv4 has no upstream counterpart → stays freshly initialized."""
    from yolo_jdt.weights.loader import _key_destination
    ckpt = torch.load(ckpt_path, weights_only=False)
    src = ckpt["model"].float().state_dict()
    dst = model.state_dict()
    partial = {}
    for k, v in src.items():
        new_k = _key_destination(k)
        if new_k not in dst:
            continue
        if v.shape != dst[new_k].shape:
            continue
        partial[new_k] = v
    result = model.load_state_dict(partial, strict=False)
    n_cv4 = sum(1 for k in result.missing_keys if k.startswith("head.cv4"))
    n_other = len(result.missing_keys) - n_cv4
    print(f"[integrity] loaded {len(partial)} keys; missing {len(result.missing_keys)} "
          f"(head.cv4: {n_cv4}, other: {n_other}), unexpected {len(result.unexpected_keys)}")
    if n_other > 0:
        non_cv4 = [k for k in result.missing_keys if not k.startswith("head.cv4")][:5]
        print(f"[integrity] WARN: non-cv4 missing keys: {non_cv4}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path,
                    help="Ultralytics YOLO11s.pt or YOLO11m.pt (COCO pretrained, nc=80)")
    ap.add_argument("--scale", required=True, choices=["s", "m"])
    ap.add_argument("--coco-root", type=Path, default=Path("datasets/raw/coco"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    img_dir = args.coco_root / "val2017"
    anno_path = args.coco_root / "annotations" / "instances_val2017.json"

    print(f"[integrity] Building YOLO11{args.scale} with JointHead (cv4 random init)")
    model = YOLO11WithJointHead(scale=args.scale, nc=80).to(device).eval()
    _load_partial_from_ultralytics(model, args.weights)

    coco = COCO(str(anno_path))
    img_ids = sorted(coco.imgs.keys())
    if args.limit is not None:
        img_ids = img_ids[:args.limit]
    print(f"[integrity] {len(img_ids)} val images, imgsz={args.imgsz}")

    results = []
    n_done = 0
    for img_id in img_ids:
        info = coco.imgs[img_id]
        path = img_dir / info["file_name"]
        img = np.array(Image.open(path).convert("RGB"))
        h0, w0 = img.shape[:2]
        canvas, r, (pad_w, pad_h) = letterbox(img, args.imgsz)
        x = torch.from_numpy(canvas).permute(2, 0, 1).contiguous().float().div(255.0)
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            decoded = model(x)[0]   # [decoded, raw, reid] → take decoded
        dets = postprocess(decoded, conf_thr=args.conf, iou_thr=args.iou,
                            max_det=args.max_det)[0]
        if dets.numel() == 0:
            n_done += 1
            continue
        dets = dets.cpu()
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_w) / r
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_h) / r
        dets[:, [0, 2]] = dets[:, [0, 2]].clamp(0, w0)
        dets[:, [1, 3]] = dets[:, [1, 3]].clamp(0, h0)
        for det in dets.tolist():
            x1, y1, x2, y2, score, cls = det
            results.append({
                "image_id": img_id,
                "category_id": COCO_CONTIGUOUS_TO_ID[int(cls)],
                "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                "score": round(score, 5),
            })
        n_done += 1
        if n_done % 500 == 0:
            print(f"  [{n_done}/{len(img_ids)}] {len(results)} dets so far", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results))

    coco_dt = coco.loadRes(str(args.output))
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    map_50_95 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])

    # Phase-2 baseline reference
    baseline_map = {"s": 0.4663, "m": 0.5132}[args.scale]
    delta = map_50_95 - baseline_map

    summary = {
        "scale": args.scale,
        "n_imgs": len(img_ids),
        "joint_head_mAP": map_50_95,
        "joint_head_mAP50": map_50,
        "phase2_baseline_mAP": baseline_map,
        "delta_mAP": delta,
        "abs_delta_mAP": abs(delta),
        "pass_threshold_pct": 0.05,
        "pass": abs(delta) <= 0.0005,    # 0.05% threshold
        "weights": str(args.weights),
        "note": "cv4 freshly initialized (no Ultralytics counterpart)",
    }
    summary_path = args.output.parent / f"step4_coco_yolo11{args.scale}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[integrity] === SUMMARY ===")
    print(f"  YOLO11{args.scale} mAP (JointHead): {map_50_95:.4f}")
    print(f"  Phase 2 baseline:              {baseline_map:.4f}")
    print(f"  Delta:                         {delta:+.4f} ({delta*100:+.2f}%)")
    print(f"  Pass (|Δ| ≤ 0.05%):            {'✓' if abs(delta) <= 0.0005 else '✗'}")
    print(f"  Wrote: {summary_path}")


if __name__ == "__main__":
    main()
