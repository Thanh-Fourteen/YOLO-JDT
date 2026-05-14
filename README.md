# YOLO-JDT

**Joint Detection-Tracking via Temporal Attention Gate (TAGate) on YOLO11.**

Lightweight cross-attention module (2–3 layers) recycles FPN-P5 features
from frame *t-1* into a joint **Detect + ReID + Track-Offset** head on top
of a standalone YOLO11 backbone — single forward pass, learned temporal
reasoning, and end-to-end exportable. Designed for both top-tier
publication and production deployment (ONNX / TensorRT FP16/INT8/FP8).

## Contribution

1. **TAGate** — micro temporal cross-attention gate (2–3 layers) reusing a
   cached neck feature from the previous frame; ~5–10% latency overhead
   over base YOLO, no full Transformer decoder.
2. **Joint Detect-Track head** — extends YOLO11's decoupled head with
   ReID embedding and track-offset branches, trained jointly with
   uncertainty-weighted losses.
3. **First NMS-free JDT** — same TAGate module ports onto YOLO26's
   end-to-end head, producing a fully NMS-free + Kalman-free pipeline
   for MOT (planned, see roadmap).

See [`plans/yolo_jdt_overview.md`](plans/yolo_jdt_overview.md) and
[`YOLO_JDT_Research_Document.md`](YOLO_JDT_Research_Document.md) for the
full literature review, ablation matrix, and architecture.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Conda env + repo skeleton + Blackwell sm_120 verification | done |
| 1 | Datasets (MOT17, MOT20, DanceTrack, CrowdHuman) → unified standard format + Dataset classes + tests | done |
| 2 | Vendor YOLO11 from Ultralytics → standalone backbone/neck/head, weight loader, COCO val mAP within ±0.5 of canonical | done |
| 3 | Detection fine-tune on CrowdHuman + MOT17 + ByteTrack/BoT-SORT baselines | next |
| 4 | ReID branch (replicate YOLO11-JDE) | planned |
| 5 | TAGate module + paired-frame loader | planned |
| 6 | Track-offset head + uncertainty-weighted multi-task training | planned |
| 7 | Association (3-cue: IoU + ReID + offset) + E2E inference | planned |
| 8 | Ablation suite (A1–A10) on MOT17 / MOT20 / DanceTrack | planned |
| 9 | YOLO26 port (NMS-free JDT) | planned |
| 10 | Production: ONNX → TensorRT FP16 / INT8 / FP8 | planned |
| 11 | Paper draft + MOTChallenge submission | planned |

## Quickstart

```bash
# 1. Conda env (one-time). Pins PyTorch ≥ 2.7 cu128 for Blackwell sm_120.
conda env create -f environment.yml
conda activate yolo_jdt

# 2. Verify hardware + drivers + key library versions.
python yolo_jdt/scripts/check_env.py

# 3. Run the test suite (model parity + dataset loaders).
pytest yolo_jdt/tests/ -v

# 4. (When ready to train) log into Weights & Biases.
wandb login
```

## Hardware

- NVIDIA GPU with `sm_120` (Blackwell, e.g. RTX 5090) for primary training.
- PyTorch ≥ 2.7 with CUDA 12.8 wheels, pinned in `environment.yml`.
- BF16 mixed precision throughout — **not** FP16 (Blackwell BF16 throughput
  is higher and avoids `GradScaler`).
- Tested on 2× RTX 5090 (32 GB each, 64 GB pooled via DDP).

Smaller cards (sm_80+ Ampere) should run inference and small ablations,
but training schedules are tuned for 2× RTX 5090.

## Repository layout

```
yolo_jdt/
  data/               unified Dataset classes + raw-format converters
  models/             standalone backbone (YOLO11) + neck (PANet) + heads
  losses/             detection + ReID + offset + uncertainty weighting
  tracker/            ByteTrack, BoT-SORT, our associator, cache manager
  train/              Lightning module + datamodule + callbacks
  eval/               TrackEval (MOT) + pycocotools (COCO) wrappers
  export/             ONNX / TensorRT / INT8 calibration
  scripts/            check_env, eval_coco, train, eval-mot, ...
  tests/              pytest suites
third_party/
  ultralytics_extract/  vendored YOLO11 building blocks (AGPL-3.0)
  yolo26_extract/       (Phase 9) YOLO26 NMS-free head
datasets/
  CHECKSUMS.md          download URLs + SHA256 + per-dataset layout quirks
  SPLITS.md             standard JSON schema + ByteTrack half-split convention
plans/
  yolo_jdt_overview.md  high-level project plan
YOLO_JDT_Research_Document.md  literature, benchmarks, ablation matrix
```

Raw datasets, pretrained weights, training runs, and evaluation outputs
are git-ignored — see [`.gitignore`](.gitignore).

## Datasets

Four datasets are used. **None are redistributed by this repo** — download
each from upstream (URLs and SHA256 in
[`datasets/CHECKSUMS.md`](datasets/CHECKSUMS.md)) into `datasets/raw/<name>/`,
then run the corresponding converter:

```bash
python -m yolo_jdt.data.converters.mot_to_std         --src datasets/raw/mot17       --dst datasets/standard/mot17 --name mot17
python -m yolo_jdt.data.converters.mot_to_std         --src datasets/raw/mot20       --dst datasets/standard/mot20 --name mot20
python -m yolo_jdt.data.converters.dance_to_std       --src datasets/raw/dancetrack  --dst datasets/standard/dancetrack
python -m yolo_jdt.data.converters.crowdhuman_to_std  --src datasets/raw/crowdhuman  --dst datasets/standard/crowdhuman
```

Each emits a per-sequence JSON under `datasets/standard/<name>/annotations/<split>/<seq>.json`
and symlinks images under `datasets/standard/<name>/images/<split>/<seq>/`.
The unified schema and ByteTrack half-train/half-val split convention are
documented in [`datasets/SPLITS.md`](datasets/SPLITS.md).

Frame counts after conversion:
- MOT17: train 5,316 / train_half 2,657 / val_half 2,659 / test 5,919 (7 seq each)
- MOT20: train 8,931 / train_half 4,464 / val_half 4,467 / test 4,479 (4 seq each)
- DanceTrack: train 41,796 (40 seq) / val 25,508 (25 seq) / test 38,551 (35 seq)
- CrowdHuman: train 15,000 / val 4,370 (single-image, treated as one "seq" per split)

## Reproducing results

### COCO val2017 detection (Phase 2 sanity check)

The standalone YOLO11 reproduces upstream numbers within ±0.5 mAP. After
loading Ultralytics' pretrained `.pt` files into our model:

```bash
# Download pretrained weights (Ultralytics releases v8.3.0)
mkdir -p weights/pretrained && cd weights/pretrained
curl -LO https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
curl -LO https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
cd ../..

# Download COCO val2017 (5,000 images + instances JSON)
mkdir -p datasets/raw/coco && cd datasets/raw/coco
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip 'annotations/instances_val2017.json'
cd ../../..

# Run mAP eval
python -m yolo_jdt.scripts.eval_coco --weights weights/pretrained/yolo11s.pt --scale s
python -m yolo_jdt.scripts.eval_coco --weights weights/pretrained/yolo11m.pt --scale m
```

| Model | Ours mAP@[.50:.95] | Ultralytics | Δ |
|-------|---------------------|-------------|---|
| YOLO11s | **46.63** | 47.0 | −0.37 |
| YOLO11m | **51.32** | 51.5 | −0.18 |

Forward parity vs Ultralytics' `DetectionModel` is verified to **≤ 1e-5
element-wise (FP32, eval mode)** by `yolo_jdt/tests/test_yolo11_parity.py`.
Two source-vs-weights divergences caught during vendoring (BatchNorm
defaults; SPPF.cv1 activation) are documented in
[`third_party/ultralytics_extract/VENDORED.md`](third_party/ultralytics_extract/VENDORED.md).

### MOT / DanceTrack training

Phase 3+ — pipeline under construction.

## Development

```bash
pre-commit install        # ruff + format on commit
pytest yolo_jdt/tests/    # 33 tests across dataset loaders + model parity
ruff check .              # lint
ruff format .             # format
```

Architectural decisions and conventions:
- BF16 mixed precision (not FP16).
- Attention via `F.scaled_dot_product_attention` (no `flash_attn` package
  dependency).
- TAGate cache is an explicit input tensor to `forward()`, never a Python
  attribute on the module — guarantees ONNX-traceable.
- Tracker logic stays outside the model graph (cleanly export-able).

## License

Project license: TBD (will be set before paper release).

| Component | License |
|-----------|---------|
| Vendored Ultralytics blocks (`third_party/ultralytics_extract/`) | AGPL-3.0 — see per-file headers and [`VENDORED.md`](third_party/ultralytics_extract/VENDORED.md) |
| MOT17 | CC BY-NC-SA 3.0 (research / non-commercial) |
| MOT20 | CC BY-NC-SA 3.0 (research / non-commercial) |
| DanceTrack | MIT |
| **CrowdHuman** | **Non-commercial research and educational use only** — do not redistribute. Any commercial deployment of weights trained on CrowdHuman requires re-training without it, or a commercial license from the dataset authors. |
| COCO val2017 | CC BY 4.0 (used only for the Phase 2 mAP sanity check) |

## Acknowledgements

Building blocks vendored from [Ultralytics 8.4.48](https://github.com/ultralytics/ultralytics)
under AGPL-3.0. ByteTrack-style half-train/val split convention follows
the original [ByteTrack](https://github.com/ifzhang/ByteTrack) work.

## Citation

A BibTeX entry will be added with the first paper draft. Until then,
please reference this repository URL.
