"""Evaluate detection mAP of a standalone YOLO11 checkpoint on MOT17 val_half
+ CrowdHuman val, write results to JSON.

Matches the val pipeline of `DetLitModule.validation_step`:
- letterbox via DetValDataset (same as training-time val)
- multi_label NMS (conf=0.001, iou=0.7, max_det=300)
- torchmetrics MeanAveragePrecision with pycocotools backend, default
  max_detection_thresholds=[1, 10, 100]
- 7 metric keys logged: mAP, mAP50, mAP75, mAP_small, mAP_medium, mAP_large, mAR100

Usage:
    python -m yolo_jdt.scripts.eval_det \\
        --output runs/baselines/detection_map.json \\
        --weights weights/ours/yolo11s_det.pt --scale s
    # multiple weights ok (paired by order with --scale)
    python -m yolo_jdt.scripts.eval_det --output ... \\
        --weights ckpt_s.pt --scale s \\
        --weights ckpt_m.pt --scale m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from yolo_jdt.models.yolo11 import YOLO11
from yolo_jdt.train.datamodule import DetDataModule
from yolo_jdt.train.lightning_module import _multi_label_nms


_METRIC_KEYS = (
    ("map",        "mAP"),
    ("map_50",     "mAP50"),
    ("map_75",     "mAP75"),
    ("map_small",  "mAP_small"),
    ("map_medium", "mAP_medium"),
    ("map_large",  "mAP_large"),
    ("mar_100",    "mAR100"),
)


@torch.no_grad()
def eval_one(weights_path: Path, scale: str, dm: DetDataModule,
             device: torch.device) -> dict:
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
        nc = payload.get("nc", 1)
        meta = {k: payload.get(k) for k in ("source_ckpt", "source_weights",
                                             "epoch", "val_metrics")
                if k in payload}
    else:
        sd, nc, meta = payload, 1, {}

    model = YOLO11(scale=scale, nc=nc).to(device).eval()
    model.load_state_dict(sd, strict=True)

    val_loaders = dm.val_dataloader()
    names = dm.val_set_names
    assert len(val_loaders) == len(names) == 2

    out = {"meta": meta, "scale": scale, "nc": nc}
    for loader, name in zip(val_loaders, names):
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                      backend="pycocotools")
        n_batches = len(loader)
        for bi, batch in enumerate(loader):
            img = batch["img"].to(device, non_blocking=True)
            decoded, _ = model(img)
            H, W = img.shape[2], img.shape[3]
            decoded_norm = decoded.clone()
            decoded_norm[:, [0, 2], :] /= W
            decoded_norm[:, [1, 3], :] /= H
            preds = _multi_label_nms(decoded_norm, conf_thr=0.001,
                                      iou_thr=0.7, max_det=300)
            # GT normalized xyxy
            gts = []
            bs = img.shape[0]
            for i in range(bs):
                mask = batch["batch_idx"] == i
                cls = batch["cls"][mask].view(-1).long().to(device)
                xywh = batch["bboxes"][mask].to(device)
                if xywh.numel() == 0:
                    gts.append({"boxes": xywh.new_zeros((0, 4)),
                                "labels": torch.zeros((0,), dtype=torch.int64, device=device)})
                    continue
                cx, cy, w, h = xywh.unbind(1)
                xyxy = torch.stack([cx - w / 2, cy - h / 2,
                                    cx + w / 2, cy + h / 2], dim=1)
                gts.append({"boxes": xyxy, "labels": cls})
            metric.update(preds, gts)
            if (bi + 1) % 50 == 0 or bi + 1 == n_batches:
                print(f"    [{name}] batch {bi+1}/{n_batches}", flush=True)
        res = metric.compute()
        sub = {k_log: float(res[k_src]) for k_src, k_log in _METRIC_KEYS if k_src in res}
        out[name] = sub
        print(f"  [{name}] {sub}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", action="append", required=True, type=Path,
                    help="Path to a promoted .pt — repeat for multiple models")
    ap.add_argument("--scale", action="append", required=True,
                    choices=["n", "s", "m", "l", "x"],
                    help="Scale corresponding to each --weights (in order)")
    ap.add_argument("--output", required=True, type=Path, help="JSON output path")
    ap.add_argument("--standard_root", default="datasets/standard",
                    help="Path to standard datasets directory")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    if len(args.weights) != len(args.scale):
        sys.exit(f"--weights ({len(args.weights)}) and --scale ({len(args.scale)}) "
                 "must have equal length")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_det] device={device}")

    dm = DetDataModule(
        standard_root=args.standard_root,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        person_only=True,
        use_crowdhuman=True,
        use_mot17=True,
    )
    dm.setup()

    results = {}
    for w, s in zip(args.weights, args.scale):
        tag = f"yolo11{s}"
        print(f"\n[eval_det] === {tag} from {w} ===")
        results[tag] = eval_one(w, s, dm, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n[eval_det] wrote {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
