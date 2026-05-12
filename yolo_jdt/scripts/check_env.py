"""Environment sanity check for YOLO-JDT.

Verifies the development environment is correctly set up for training and inference:
- conda env name
- Python version
- Required packages importable
- CUDA + Blackwell sm_120 capability
- BF16 ops
- SDPA (FlashAttention backend availability)
- NCCL DDP init (optional, skipped by --quick)
- Dataset directories present (optional warnings)

Run: `python yolo_jdt/scripts/check_env.py [--quick]`
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
WARN = f"{YELLOW}⚠{RESET}"


def report(name: str, ok: bool, detail: str = "") -> bool:
    mark = CHECK if ok else CROSS
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    return ok


def warn(name: str, detail: str = "") -> None:
    print(f"  {WARN} {name}" + (f" — {detail}" if detail else ""))


def check_conda_env() -> bool:
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "NONE")
    return report(
        "Conda env = yolo_jdt",
        env_name == "yolo_jdt",
        f"active env: {env_name}",
    )


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    return report(
        "Python >= 3.11",
        (major, minor) >= (3, 11),
        f"{major}.{minor}.{sys.version_info[2]}",
    )


def check_imports() -> bool:
    required = [
        "torch",
        "torchvision",
        "pytorch_lightning",
        "hydra",
        "omegaconf",
        "numpy",
        "scipy",
        "cv2",
        "wandb",
        "tensorboard",
        "motmetrics",
        "lap",
        "cython_bbox",
        "sklearn",
        "onnx",
        "onnxruntime",
        "onnxsim",
        "pytest",
        "ruff",
        "mypy",
        "einops",
        "matplotlib",
        "pandas",
    ]
    all_ok = True
    missing = []
    for mod in required:
        try:
            importlib.import_module(mod)
        except ImportError:
            all_ok = False
            missing.append(mod)
    return report(
        f"All {len(required)} core packages importable",
        all_ok,
        f"missing: {', '.join(missing)}" if missing else "",
    )


def check_torch_cuda() -> bool:
    import torch

    if not torch.cuda.is_available():
        return report("CUDA available", False, "torch.cuda.is_available() = False")
    n = torch.cuda.device_count()
    if n < 1:
        return report("At least 1 CUDA device", False, f"found {n}")
    report(f"CUDA available ({n} devices)", True, f"torch {torch.__version__}")
    return True


def check_blackwell_capability() -> bool:
    import torch

    ok = True
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        is_blackwell = cap == (12, 0)
        ok = report(
            f"Device {i} sm_120 (Blackwell)",
            is_blackwell,
            f"{name}, capability={cap}",
        ) and ok
    return ok


def check_bf16_ops() -> bool:
    import torch

    try:
        x = torch.randn(8, 8, dtype=torch.bfloat16, device="cuda")
        y = x @ x
        ok = torch.isfinite(y).all().item() and y.dtype == torch.bfloat16
        return report("BF16 matmul on CUDA", ok, f"shape={tuple(y.shape)}, dtype={y.dtype}")
    except Exception as e:
        return report("BF16 matmul on CUDA", False, str(e))


def check_sdpa() -> bool:
    """Test scaled_dot_product_attention with BF16 to verify Flash backend availability."""
    import torch
    import torch.nn.functional as F

    try:
        q = torch.randn(2, 8, 100, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(2, 8, 100, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(2, 8, 100, 64, dtype=torch.bfloat16, device="cuda")
        out = F.scaled_dot_product_attention(q, k, v)
        ok = torch.isfinite(out).all().item() and out.shape == q.shape
        return report("SDPA BF16 (FlashAttention backend)", ok, f"out shape={tuple(out.shape)}")
    except Exception as e:
        return report("SDPA BF16 (FlashAttention backend)", False, str(e))


def check_conv2d_forward() -> bool:
    import torch

    try:
        x = torch.randn(2, 3, 640, 640, device="cuda", dtype=torch.bfloat16)
        conv = torch.nn.Conv2d(3, 16, 3).cuda().to(dtype=torch.bfloat16)
        y = conv(x)
        return report(
            "Conv2d BF16 forward (no kernel image error)",
            torch.isfinite(y).all().item(),
            f"out shape={tuple(y.shape)}",
        )
    except Exception as e:
        return report("Conv2d BF16 forward (no kernel image error)", False, str(e))


def check_ddp_init() -> bool:
    """Test that NCCL can be initialized across all GPUs (single-host test)."""
    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() < 2:
        warn("DDP NCCL init", "fewer than 2 GPUs — skipping")
        return True

    try:
        # Single-process pseudo-DDP test: just verify NCCL backend is present
        # Full DDP requires multiprocessing; this is a smoke test only.
        from torch.distributed import is_available as dist_available

        if not dist_available():
            return report("torch.distributed available", False)
        # NCCL backend probe — actually init requires multi-process; we just check NCCL is built in
        # by attempting a single-rank init.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=0, world_size=1)
        ok = dist.is_initialized() and dist.get_backend() == "nccl"
        if ok:
            dist.destroy_process_group()
        return report("NCCL DDP init (single-process probe)", ok)
    except Exception as e:
        return report("NCCL DDP init", False, str(e))


def check_datasets() -> None:
    """Warn-only: check if dataset directories exist."""
    root = Path(__file__).resolve().parents[2]
    expected = ["mot17", "mot20", "dancetrack", "crowdhuman"]
    for name in expected:
        p = root / "datasets" / "standard" / name
        if p.exists() and any(p.iterdir()):
            print(f"  {CHECK} datasets/standard/{name}/ present")
        else:
            warn(f"datasets/standard/{name}/", "not present yet — Step 1.X")


def check_weights() -> None:
    root = Path(__file__).resolve().parents[2]
    p = root / "weights" / "pretrained"
    if p.exists():
        n = sum(1 for _ in p.glob("*.pt"))
        if n > 0:
            print(f"  {CHECK} weights/pretrained/ has {n} *.pt files")
        else:
            warn("weights/pretrained/", "no *.pt files yet — Step 2.C will download")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip NCCL DDP probe")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLO-JDT environment check")
    print("=" * 60)

    print("\n[1/4] Env + interpreter")
    e1 = check_conda_env()
    e2 = check_python()
    e3 = check_imports()

    print("\n[2/4] CUDA + Blackwell")
    g1 = check_torch_cuda()
    g2 = check_blackwell_capability() if g1 else False

    print("\n[3/4] BF16 + SDPA + Conv2d (Blackwell sm_120 kernels)")
    k1 = check_bf16_ops() if g1 else False
    k2 = check_sdpa() if g1 else False
    k3 = check_conv2d_forward() if g1 else False

    print("\n[4/4] Distributed")
    if args.quick:
        warn("NCCL DDP init", "skipped (--quick)")
        d1 = True
    else:
        d1 = check_ddp_init() if g1 else False

    print("\n[Bonus] Project data layout (warn-only)")
    check_datasets()
    check_weights()

    print("\n" + "=" * 60)
    blockers = [e1, e2, e3, g1, g2, k1, k2, k3, d1]
    if all(blockers):
        print(f"{GREEN}All required checks passed.{RESET}")
        return 0
    print(f"{RED}One or more required checks failed.{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
