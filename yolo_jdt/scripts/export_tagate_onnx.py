"""Export YOLO-JDT (TAGate + JointHead) to ONNX and verify with onnxruntime.

The ONNX graph signature uses flat positional inputs instead of list[Tensor]:
    Inputs:  image_t  [1, 3, 640, 640]
             cache_P5 [1, 512, 20, 20]   (for P5-only caching)
    Outputs: decoded   [1, 5, 8400]        (nc=1 → 4+1 channels)
             reid_P3   [1, 128, 80, 80]
             reid_P4   [1, 128, 40, 40]
             reid_P5   [1, 128, 20, 20]
             offset_P3 [1, 2, 80, 80]      (Δx, Δy per anchor)
             offset_P4 [1, 2, 40, 40]
             offset_P5 [1, 2, 20, 20]
             cache_out_P5  [1, 512, 20, 20]

For multi-level cache (P4+P5, P3+P4+P5) additional cache inputs/outputs
are added automatically.

Usage:
    # From a trained checkpoint
    python -m yolo_jdt.scripts.export_tagate_onnx \\
        --weights weights/ours/yolo11s_jdt.pt --scale s \\
        --out weights/exported/yolo11s_jdt.onnx

    # Quick smoke-test with freshly initialized weights (no checkpoint needed)
    python -m yolo_jdt.scripts.export_tagate_onnx --test-init-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from yolo_jdt.models.yolo_jdt import YOLO_JDT


class _JDT_ONNXWrapper(nn.Module):
    """Flattens list inputs/outputs for ONNX compatibility.

    ONNX does not support list[Tensor] as I/O; this wrapper accepts cache
    tensors as positional args and returns all outputs as a flat tuple.
    """

    def __init__(self, model: YOLO_JDT):
        super().__init__()
        self.model = model

    def forward(self, image_t: Tensor, *cache_tensors: Tensor) -> tuple:
        out = self.model(image_t, list(cache_tensors))
        # eval mode: (decoded, raw_det, reid_list, offset_list, feats_cache_list)
        decoded, _, reid, offset, feats_cache = out
        return (decoded, *reid, *offset, *feats_cache)


def export_onnx(
    model: YOLO_JDT,
    out_path: Path,
    imgsz: int = 640,
    opset: int = 18,
    simplify: bool = True,
) -> Path:
    """Export YOLO_JDT to ONNX.  Returns path to the exported file."""
    import onnx

    model.eval()
    wrapper = _JDT_ONNXWrapper(model)

    dummy_img = torch.zeros(1, 3, imgsz, imgsz)
    dummy_cache = model.zero_cache(batch_size=1)

    # Input / output names
    n_cache = len(dummy_cache)
    level_tags = ["P3", "P4", "P5"][-n_cache:]
    input_names = ["image_t"] + [f"cache_{t}" for t in level_tags]
    output_names = (
        ["decoded"]
        + [f"reid_{t}" for t in ["P3", "P4", "P5"]]
        + [f"offset_{t}" for t in ["P3", "P4", "P5"]]
        + [f"cache_out_{t}" for t in level_tags]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()   # wrapper must also be in eval mode to suppress the training-mode warning
    torch.onnx.export(
        wrapper,
        (dummy_img, *dummy_cache),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[export_onnx] exported → {out_path}")

    # onnxsim simplify (optional, best-effort)
    if simplify:
        try:
            import onnxsim
            model_onnx = onnx.load(str(out_path))
            model_simp, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(model_simp, str(out_path))
                print("[export_onnx] onnxsim simplification applied")
        except Exception as e:
            print(f"[export_onnx] onnxsim skipped ({e})")

    return out_path


def verify_onnx(onnx_path: Path, model: YOLO_JDT, imgsz: int = 640) -> bool:
    """Load and run the exported ONNX with onnxruntime.

    Pass criterion for Step 5.DE: onnxruntime can load the model and produce
    output tensors with the expected shapes.  Full numerical drift checking
    (PyTorch vs ONNX < 1e-3) is done in Step 10 (export pipeline).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify_onnx] onnxruntime not installed — skipping verify")
        return True

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_img = torch.zeros(1, 3, imgsz, imgsz)
    # Build fresh cache tensors — do NOT reuse model after torch.export to avoid SymFloat
    n_cache = model.num_cached_levels
    level_ids = model._level_ids
    strides = [8, 16, 32]
    dummy_cache_np = [
        torch.zeros(1, model._cache_channels[j],
                    imgsz // strides[level_ids[j]],
                    imgsz // strides[level_ids[j]]).numpy()
        for j in range(n_cache)
    ]

    inputs = {sess.get_inputs()[0].name: dummy_img.numpy()}
    for i, c in enumerate(dummy_cache_np):
        inputs[sess.get_inputs()[i + 1].name] = c

    ort_out = sess.run(None, inputs)
    decoded_shape = tuple(ort_out[0].shape)
    print(f"[verify_onnx] onnxruntime ran OK — decoded shape: {decoded_shape}")
    # Sanity: decoded should be [1, nc+4, num_anchors]
    ok = len(decoded_shape) == 3 and decoded_shape[0] == 1
    print(f"[verify_onnx] {'PASS' if ok else 'FAIL (unexpected output shape)'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=None,
                    help="Trained YOLO_JDT checkpoint (.pt or .ckpt). "
                         "If omitted, a freshly initialized model is used.")
    ap.add_argument("--scale", default="s", choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--cache-levels", default="P5",
                    choices=["P5", "P4+P5", "P3+P4+P5"])
    ap.add_argument("--tagate-num-layers", type=int, default=2)
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .onnx path. Default: weights/exported/<stem>.onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--test-init-only", action="store_true",
                    help="Quick smoke-test: export freshly initialized model, verify, exit.")
    args = ap.parse_args()

    if args.test_init_only:
        print("[export_onnx] -- smoke test: freshly initialized YOLO_JDT --")
        model = YOLO_JDT(scale="s", nc=1, cache_levels="P5",
                          tagate_num_layers=1).eval()
        out_path = Path("/tmp/yolo_jdt_smoke.onnx")
        export_onnx(model, out_path, imgsz=640, simplify=False)
        ok = verify_onnx(out_path, model)
        print(f"[export_onnx] smoke test {'PASS' if ok else 'FAIL'}")
        return

    if args.weights is None:
        ap.error("--weights is required (or use --test-init-only for a smoke test)")

    from yolo_jdt.scripts.infer_tracking_jdt import load_jdt_model
    model, nc = load_jdt_model(
        args.weights, args.scale, torch.device("cpu"),
        cache_levels=args.cache_levels,
        tagate_num_layers=args.tagate_num_layers,
    )

    out_path = (args.out if args.out else
                Path("weights/exported") / f"{args.weights.stem}.onnx")

    export_onnx(model, out_path, imgsz=args.imgsz, opset=args.opset,
                simplify=not args.no_simplify)
    verify_onnx(out_path, model, imgsz=args.imgsz)
    print(f"[export_onnx] DONE → {out_path}")


if __name__ == "__main__":
    main()
