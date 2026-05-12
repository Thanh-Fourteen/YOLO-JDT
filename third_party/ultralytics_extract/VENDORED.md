# Vendored Ultralytics modules

Per-module log of what we extracted from upstream Ultralytics into our
runtime-independent `third_party/ultralytics_extract/` package.

## Source

- **Upstream package**: `ultralytics`
- **Upstream version**: 8.4.48 (installed via `pip install ultralytics`)
- **License**: AGPL-3.0 (preserved in every extracted file's header)
- **Upstream code path**: `ultralytics/nn/modules/{conv,block,head}.py`
  + helpers from `ultralytics/utils/{tal,torch_utils}.py` and
  `ultralytics/nn/modules/utils.py` (inlined where needed)

Our extracted package contains ONLY the building blocks YOLO11 detection
needs — about 200 lines per file vs 600–2000 in upstream.

## Why we vendor

Per `feedback_code_style.md` (memory): YOLO-JDT must not depend on
`ultralytics` at runtime. The Ultralytics Trainer/CLI is opinionated and
hard to extend for the multi-task joint training (TAGate + ReID +
TrackOffset) we add in Phases 4-6. Vendoring the building blocks keeps
the architecture pinned and gives us a clean `nn.Module` surface for
ONNX export (Phase 10).

## Per-module mapping

### `conv.py` ← `ultralytics/nn/modules/conv.py`
| Symbol | Upstream | Notes |
|--------|----------|-------|
| `autopad` | line 30 | Pure copy (3-line helper) |
| `Conv`    | line 39 | Pure copy. **One modification:** `BatchNorm2d(eps=1e-3, momentum=0.03)` instead of PyTorch defaults — Ultralytics applies these via `initialize_weights()` post-construction; we do it at `__init__` for parity with pretrained weights. |
| `DWConv`  | line 185 | Pure copy. |
| `Concat`  | line 616 | Pure copy. |

Skipped from upstream: `Conv2`, `LightConv`, `DWConvTranspose2d`,
`ConvTranspose`, `Focus`, `GhostConv`, `RepConv`, `ChannelAttention`,
`SpatialAttention`, `CBAM`, `Index`. Not used by YOLO11 detection.

### `block.py` ← `ultralytics/nn/modules/block.py`
| Symbol | Upstream | Notes |
|--------|----------|-------|
| `DFL`        | line 58   | Pure copy. |
| `SPPF`       | line 208  | **One modification:** `cv1` is built with `act=True` (default SiLU) instead of upstream's `act=False`. The released YOLO11 .pt weights were trained with `act=True`; matching upstream source instead would break parity. |
| `Bottleneck` | line 457  | Pure copy. |
| `C2f`        | line 288  | Pure copy. |
| `C3`         | line 322  | Pure copy. |
| `C3k`        | line 1109 | Pure copy. |
| `C3k2`       | line 1069 | Pure copy. |
| `Attention`  | line 1271 | Pure copy. Position-Sensitive multi-head self-attention used inside PSABlock. |
| `PSABlock`   | line 1331 | Pure copy. |
| `C2PSA`      | line 1436 | Pure copy. |

Skipped from upstream: `Proto`, `HGStem`, `HGBlock`, `C2`, `C1`, `C2fAttn`,
`C2fCIB`, `C3x`, `RepC3`, `BottleneckCSP`, `RepBottleneck`, `CIB`,
`RepVGGDW`, `PSA`, `C2fPSA`, `A2C2f`, `SAVPE`, `BNContrastiveHead`,
`ContrastiveHead`, `Proto26`, `RealNVP`, `Residual`, `SwiGLUFFN`, etc.
None are referenced by YOLO11 detection.

Helper `fuse_conv_and_bn` (used by `forward_fuse` / `fuse()` paths) is
not vendored — fusing is a separate pass we run at export time
(Phase 10) using a clean `torch.nn.utils.fusion` workflow.

### `head.py` ← `ultralytics/nn/modules/head.py`
| Symbol | Upstream | Notes |
|--------|----------|-------|
| `Detect`              | line 26                   | **Stripped** the optional `end2end` / `one2one` branches (YOLO11 doesn't use them; YOLO26 will get its own head in `third_party/yolo26_extract/` for Phase 9), `postprocess()` / `get_topk_index()` (replaced by `torchvision.ops.batched_nms` in our eval/inference scripts), and `fuse()` (handled at export). Attribute names (`cv2`, `cv3`, `dfl`, `nc`, `nl`, `reg_max`, `no`, `stride`, `legacy`, `dynamic`, `export`, `shape`, `anchors`, `strides`) preserved verbatim for state_dict compatibility. |
| `make_anchors`        | `utils/tal.py:400`        | Inlined into head.py. |
| `dist2bbox`           | `utils/tal.py:416`        | Inlined into head.py. |
| `bias_init_with_prob` | `nn/modules/utils.py:35`  | Inlined into head.py. |

Skipped from upstream: `Segment`, `OBB`, `Pose`, `Classify`, `RTDETRDecoder`,
`WorldDetect`, `YOLOEDetect`, `YOLOESegment`, `v10Detect`, `LRPCHead`,
`Pose26`, `Segment26`, `OBB26`. Not used by detection.

## Verification

State_dict mapping is `model.{layer_idx}.{...} → backbone.layer{N}.{...} /
neck.layer{N}.{...} / head.{...}`, encoded in
`yolo_jdt/weights/loader.py::_key_destination`. All 499 keys in the
YOLO11s pretrained checkpoint map cleanly with zero shape mismatches.

Forward parity: `yolo_jdt/tests/test_yolo11_parity.py` confirms the
decoded detection output of our `YOLO11(scale='s'|'m')` matches
Ultralytics' `DetectionModel` within 1e-5 element-wise (FP32, eval mode,
random 1×3×640×640 input).

## Bugs / divergences caught during extraction

1. **BN defaults**: `nn.BatchNorm2d` constructed with PyTorch defaults
   (eps=1e-5, momentum=0.1) gave > 348 element-wise diff vs upstream at
   layer 0. Upstream patches every BN to `eps=1e-3, momentum=0.03` via
   `initialize_weights()` after model construction. Fixed by setting
   the same defaults inside our `Conv.__init__`.

2. **SPPF.cv1 activation**: Upstream's current `block.py` source has
   `act=False` on SPPF.cv1, but the released YOLO11 .pt weights were
   trained with `act=True` (SiLU). Forward parity required matching the
   weights, not the source. Same parity diff (~1.0 at SPPF output)
   would silently degrade COCO mAP if not caught.

These two divergences would not have been caught by shape-only checks
(state_dict mapping passed cleanly even with both bugs in place). The
1e-5 forward-parity test was the only mechanism that surfaced them.

## COCO val2017 mAP (Phase 2.D pass criterion)

Final eval config in `yolo_jdt/scripts/eval_coco.py`:
- letterbox 640×640, scaleup=True, cv2.INTER_LINEAR
- multi_label NMS (each anchor → 1 det per class above conf), class-offset
  via `class_id * max_wh=7680` then single `torchvision.ops.nms`
- conf=0.001, iou=0.7, max_det=300

| Model | Ours | Ultralytics docs | Δ | Pass ±0.5 |
|-------|------|------------------|---|-----------|
| YOLO11s | **46.63** | 47.0 | −0.37 | ✓ |
| YOLO11m | **51.32** | 51.5 | −0.18 | ✓ |

Iterations to hit pass band (each = full 5000-image eval per scale):
1. argmax NMS + PIL letterbox: 46.02 / 50.70 — below band by ~1.0
2. + multi_label NMS: 46.64 / 51.33 — both pass; multi_label is the dominant fix
3. + scaleup=False + cv2: 46.40 / 51.06 — scaleup=False hurt s
4. cv2 + scaleup=True (chosen): 46.63 / 51.32

Residual ~0.4 below canonical centerline likely from sub-pixel letterbox
offset (Ultralytics' `scale_boxes` uses `round((... ) / 2 - 0.1)`) and minor
cv2/PIL JPEG decode differences. Within paper-class tolerance (the
±0.5 budget covers these implementation idiosyncrasies).

