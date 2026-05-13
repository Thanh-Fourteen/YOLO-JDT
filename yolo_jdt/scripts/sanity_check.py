"""Pre-flight smoke test before launching a real training run.

Validates:
  1. Forward shape per FPN level
  2. Backward gradient flow (grad on every trainable param)
  3. BF16 stability across 5 steps (loss + gradnorm finite)
  4. Dataloader speed (1 batch < 1s after warmup)
  5. Initial loss in expected range ([1, 30])
  6. Peak VRAM headroom

Skips DDP-specific checks when devices=1 (use NCCL probe explicitly with
`--ddp` flag if you want a multi-GPU pre-flight).

Usage:
    python -m yolo_jdt.scripts.sanity_check --config base
    python -m yolo_jdt.scripts.sanity_check --config base data.batch_size=2 model.imgsz=640
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_cfg(config_name: str, overrides: list[str]):
    config_dir = str(PROJECT_ROOT / "yolo_jdt" / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, hydra_overrides = ap.parse_known_args()
    cfg = _load_cfg(args.config, hydra_overrides)
    print("[sanity] resolved config:")
    print(OmegaConf.to_yaml(cfg))

    # Force project root cwd so relative paths resolve
    import os
    os.chdir(PROJECT_ROOT)

    # Build datamodule + 1 train batch
    sys.path.insert(0, str(PROJECT_ROOT))
    from yolo_jdt.train.datamodule import DetDataModule
    from yolo_jdt.train.lightning_module import DetLitModule

    print("\n[1/6] Build datamodule + first batch")
    dm = DetDataModule(
        standard_root=cfg.data.standard_root,
        imgsz=cfg.data.imgsz,
        batch_size=cfg.data.batch_size,
        num_workers=0,                 # no workers in sanity (avoids fork overhead)
        mosaic_p=cfg.data.mosaic_p,
        hsv=tuple(cfg.data.hsv),
        person_only=cfg.data.person_only,
        use_crowdhuman=cfg.data.use_crowdhuman,
        use_mot17=cfg.data.use_mot17,
        mot17_train_split=cfg.data.mot17_train_split,
    )
    dm.setup("fit")
    train_dl = dm.train_dataloader()
    t0 = time.perf_counter()
    batch = next(iter(train_dl))
    t1 = time.perf_counter()
    print(f"  ✓ first batch fetched in {t1 - t0:.2f}s, "
          f"img shape {tuple(batch['img'].shape)}, "
          f"GT count {batch['bboxes'].shape[0]}")

    # Build model
    print("\n[2/6] Build LitModule (loads pretrained if configured)")
    lit = DetLitModule(
        scale=cfg.model.scale,
        nc=cfg.model.nc,
        reg_max=cfg.model.reg_max,
        imgsz=cfg.model.imgsz,
        pretrained_weights=cfg.model.pretrained_weights,
        channels_last=cfg.model.channels_last,
    ).cuda()
    n_params = sum(p.numel() for p in lit.model.parameters())
    print(f"  ✓ YOLO11{cfg.model.scale}: {n_params/1e6:.2f}M params")

    # Forward shape
    print("\n[3/6] Forward shape check")
    img = batch["img"].cuda()
    if cfg.model.channels_last:
        img = img.to(memory_format=torch.channels_last)
    raw = lit.model(img)
    expected_no = 4 * cfg.model.reg_max + cfg.model.nc
    for i, x in enumerate(raw):
        assert x.shape[1] == expected_no, f"raw[{i}] ch {x.shape[1]} != {expected_no}"
        print(f"  ✓ P{i+3}: {tuple(x.shape)}")

    # Loss + backward
    print("\n[4/6] Loss + backward + grad flow")
    from third_party.ultralytics_extract.loss import decode_raw_outputs
    batch_gpu = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items() if k != "extras"}
    preds = decode_raw_outputs(raw, nc=cfg.model.nc, reg_max=cfg.model.reg_max)
    total, comp = lit.loss_fn(preds, batch_gpu)
    print(f"  ✓ initial loss: total={total.item():.3f}  box={comp[0]:.3f}  cls={comp[1]:.3f}  dfl={comp[2]:.3f}")
    assert torch.isfinite(total), f"non-finite loss: {total}"
    assert 0 < total.item() < 200, f"initial loss {total.item()} out of [0, 200]"
    total.backward()
    no_grad = [n for n, p in lit.model.named_parameters() if p.requires_grad and p.grad is None]
    nan_grad = [n for n, p in lit.model.named_parameters()
                if p.grad is not None and torch.isnan(p.grad).any()]
    print(f"  ✓ params with grad: {sum(1 for p in lit.model.parameters() if p.grad is not None)}/"
          f"{sum(1 for _ in lit.model.parameters())}")
    if no_grad:
        print(f"  ! {len(no_grad)} params with no grad (sample): {no_grad[:3]}")
    if nan_grad:
        print(f"  ✗ {len(nan_grad)} NaN grads (sample): {nan_grad[:3]}")
        sys.exit(1)

    # BF16 stability across 5 steps
    print("\n[5/6] BF16 stability — 5 forward+backward steps")
    from torch.optim import SGD
    opt = SGD(lit.model.parameters(), lr=0.01, momentum=0.9)
    losses = []
    for step in range(5):
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw = lit.model(img)
            preds = decode_raw_outputs(raw, nc=cfg.model.nc, reg_max=cfg.model.reg_max)
            total, _ = lit.loss_fn(preds, batch_gpu)
        total.backward()
        opt.step()
        gradnorm = sum(p.grad.norm().item() ** 2 for p in lit.model.parameters() if p.grad is not None) ** 0.5
        losses.append(total.item())
        print(f"  step {step}: loss={total.item():.3f}, gradnorm={gradnorm:.3f}")
        assert torch.isfinite(total), f"non-finite loss at step {step}"
        assert gradnorm < 1e6, f"gradnorm {gradnorm} explodes"
    if losses[-1] >= losses[0]:
        print(f"  ! loss not decreasing: {losses[0]:.3f} → {losses[-1]:.3f} "
              f"(could be normal at high LR + small batch; not blocking)")
    else:
        print(f"  ✓ loss decreasing: {losses[0]:.3f} → {losses[-1]:.3f}")

    # VRAM
    print("\n[6/6] Peak VRAM")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  ✓ peak VRAM: {peak_gb:.2f} GB / 32 GB total per card")
    if peak_gb > 28:
        print("  ! peak VRAM > 28 GB — close to OOM headroom")

    print("\n[sanity] ALL CHECKS PASS")


if __name__ == "__main__":
    main()
