# YOLO-JDT — Plan Tổng quan

> **Audience:** developer, reviewer, future-self. File này là tour guide cho dự án — đọc xong nắm được mục tiêu, hướng đi, roadmap, và biết tìm chi tiết ở đâu.
> **Source of truth chi tiết:** [`YOLO_JDT_Research_Document.md`](../YOLO_JDT_Research_Document.md) (literature review + benchmarks + đầy đủ luận lý).
> **Execution plan đầy đủ:** `~/.claude/plans/d-ng-th-ng-tin-v-a-mutable-blum.md` (kế hoạch thực thi).
> **Project state (checkpoint):** [`../todos.md`](../todos.md) (status từng step + resume prompts).

---

## 1. Vấn đề & Mục tiêu

Multi-Object Tracking (MOT) hiện tại có 3 paradigm chính (TBD / JDE / E2E), mỗi paradigm đánh đổi giữa **tốc độ**, **độ chính xác**, **temporal reasoning**, và **model size**. Chưa có phương pháp nào đồng thời đạt: (1) 1 lần backbone inference, (2) temporal reasoning tốt hơn Kalman Filter, (3) real-time 30+ FPS, (4) model nhẹ.

**Mục tiêu YOLO-JDT:**
- Xây dựng Joint Detection-Tracking model **nhanh hơn pipeline TBD tách rời**, **accuracy không giảm** so với SOTA TBD, **model nhẹ**, và có **tính novel** rõ ràng.
- Benchmark: MOT17, MOT20, DanceTrack. Target HOTA ≥ 63 (MOT17), MOTA ≥ 80, IDF1 ≥ 77, FPS ≥ 35.

---

## 2. Hướng đi: TAGate (Temporal Attention Gate)

**Ý tưởng cốt lõi:** thêm vào YOLO11 một module cross-attention nhẹ (2–3 layers) tái sử dụng cached feature map từ frame t-1 để bơm temporal context vào detection pipeline. Module này gọi là **TAGate** — output `F'_t = F_t + α·CrossAttn(F_t, F_{t-1})` với α gated learnable.

**Tại sao novel:**
- MO-YOLO/DecoderTracker dùng full RT-DETR decoder → quá nặng.
- YOLO11-JDE có ReID branch nhưng KHÔNG dùng temporal info.
- ByteTrack/BoT-SORT dùng Kalman Filter ngoài model → không learn complex motion.
- Kết hợp CNN backbone + lightweight temporal cross-attention chưa có trong literature.

**Joint Detect-Track Head:** mở rộng decoupled head với 3 outputs đồng thời:
1. Detection (box + class) — giữ nguyên YOLO11.
2. ReID embedding (128-d).
3. Track offset prediction (Δx, Δy) — displacement giữa current và previous frame.

---

## 3. Dual Goal: Paper + Production

| Track | Yêu cầu chính |
|-------|----------------|
| **Paper top-tier** (CVPR/ICCV/ECCV) | Submit MOT17/MOT20/DanceTrack test server với private detection; ≥8 ablations (A1–A8); baseline so sánh ByteTrack, BoT-SORT, FairMOT, MOTR, MOTRv2, MO-YOLO; code release reproducible; 2 contributions: (a) TAGate, (b) First NMS-free JDT với YOLO26. |
| **Production** | ONNX export sạch (cache là input tensor, không Python state); TensorRT FP16/INT8/FP8 (Blackwell native); CUDA Graphs cho streaming inference; latency report p50/p90/p99; INT8 drift HOTA < 1.0. |

Cả hai track chia sẻ cùng codebase. Quyết định kiến trúc phải thoả mãn cả hai — ví dụ: TAGate cache **bắt buộc** là input tensor (không `self.cache`) ngay từ đầu để khỏi rework khi export.

---

## 4. Kiến trúc cao tầng

```
Video Frame (t)
       │
       ▼
┌──────────────────┐
│  YOLO11 Backbone │──── (P3, P4, P5)
│  (C3k2 + SPPF)  │
└──────────────────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────┐
│   YOLO11 Neck    │     │   Feature Cache     │
│   (PANet FPN)    │────▶│   (P5 from t-1)     │
└──────────────────┘     └─────────────────────┘
       │                          │
       ▼                          │
┌──────────────────────────────────────────┐
│           TAGate Module                   │
│  Cross-Attention (Q=F_t, K/V=F_{t-1})    │
│  Gated Residual: F'_t = F_t + α·Attn     │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│        Joint Detect-Track Head            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Detect   │ │ ReID     │ │ Track    │ │
│  │ (Box+Cls)│ │ (128-d)  │ │ Offset   │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└──────────────────────────────────────────┘
       │               │              │
       ▼               ▼              ▼
┌──────────────────────────────────────────┐
│     Lightweight Association Module        │
│  IoU + ReID + Offset → Hungarian → IDs   │
└──────────────────────────────────────────┘
```

Chi tiết module dimensions, attention shapes, loss components — xem `YOLO_JDT_Research_Document.md` §9.

---

## 5. Phase Roadmap (tóm tắt)

| Phase | Mục tiêu | Deliverable chính |
|-------|----------|-------------------|
| **Phase 0** | Env & repo skeleton | Conda env `yolo_jdt`, repo layout, Claude Code workspace |
| **Phase 1** | Dataset acquisition | MOT17/20/DanceTrack/CrowdHuman ở format chuẩn |
| **Phase 2** | Extract YOLO11 standalone | Pure-PyTorch YOLO11, COCO mAP match paper ±0.5 |
| **Phase 3** | Detection + TBD baselines | YOLO11+ByteTrack/BoT-SORT baseline numbers (3 seeds) |
| **Phase 4** | ReID branch (YOLO11-JDE replicate) | JointHead với ReID, HOTA ≥ baseline |
| **Phase 5** | **TAGate** (core contribution) | TAGate module + A1 ablation chứng minh signal |
| **Phase 6** | Track Offset + multi-task training | 3-stage training với uncertainty weighting |
| **Phase 7** | Association algorithm | 3-cue Hungarian, e2e inference, FPS benchmark |
| **Phase 8** | Ablation suite | A1–A11 (paper rigor), LaTeX table |
| **Phase 9** | YOLO26 port (NMS-free JDT) | YOLO26-JDT, A8 vs YOLO11-JDT |
| **Phase 10** | Production | ONNX + TRT FP16/INT8/FP8 + CUDA Graphs, latency report |
| **Phase 11** | Paper writing | MOTChallenge test submission, draft + figures + code release |

Step-by-step granularity: xem [`../todos.md`](../todos.md).

---

## 6. Optimization Stack

Phân loại theo mức độ áp dụng:

| Tier | Items |
|------|-------|
| **Must-have (M1–M11)** | PyTorch Lightning 2.x, Hydra configs, SDPA (auto FlashAttention), BF16 mixed precision, Sync BatchNorm, EMA model weights, Mosaic-close last 15% epochs, deterministic seeds + config hash, TensorRT FP16+INT8, CUDA Graphs streaming, FP8 trên Blackwell |
| **Explore (E1–E4)** | `torch.compile` (mode="reduce-overhead", fallback eager nếu fail), gradient checkpointing (chỉ khi OOM), multi-scale training, channels-last memory format |
| **Novelty-additive (N1–N5)** | Confidence-aware TAGate α (per-spatial, không scalar), adaptive cue weighting MLP, long-term ReID memory bank (N=10 historical embeddings), EMA-teacher self-distillation cho ReID, KD từ MOTRv2 teacher (paper v2) |

---

## 7. Decisions & Why

**YOLO11 là primary backbone, không ablation v8/v12:**
- v8 → v11 cùng nhà phát triển, cùng decoupled head philosophy, YOLO11 dominates v8 ở mọi scale trên COCO (n: 39.5 vs 37.3, s: 47.0 vs 44.9, m: 51.5 vs 50.2). Ablation v11 vs v8 chỉ chứng minh điều Ultralytics paper đã chứng minh — không đóng góp cho TAGate contribution.
- v12 có training instability đã được chính thức thừa nhận; attention-heavy backbone redundant với TAGate (cross-attention); attention-on-attention diminishing returns; loãng claim "lightweight temporal attention trên CNN YOLO".

→ Section trong paper: "Why YOLO11 as primary backbone" giải thích bằng citation + reasoning, không cần empirical ablation.

**Vendor-and-extract from Ultralytics, không runtime dependency:**
- Cần control multi-task loss, stage-wise freezing, uncertainty weighting → Ultralytics Trainer quá opinionated.
- Export ONNX cần sạch — không import `ultralytics` runtime.
- Lấy `nn.Module` source, paste vào `third_party/ultralytics_extract/`, giữ license header, map state_dict.

**SDPA over `flash-attn` package:**
- `torch.nn.functional.scaled_dot_product_attention` tự chọn FlashAttention-2/3 backend khi hardware support.
- Blackwell sm_120 mới — repo `flash-attn` đang catch-up đầu 2026, không pin trực tiếp.

**BF16 over FP16 (Blackwell-specific):**
- Blackwell BF16 throughput cao hơn FP16.
- BF16 không cần GradScaler, tránh loss-scale instability.

**Conda env isolated `yolo_jdt`:**
- Tránh conflict torch/CUDA version với project khác.
- Reproducibility paper: `environment.yml` + `requirements-lock.txt` đính kèm code release.

---

## 8. Datasets

| Dataset | Vai trò | Size | Sequences |
|---------|---------|------|-----------|
| **CrowdHuman** | Pretrain detection + ReID (static, dense pedestrians) | ~15GB | ~15K train + 4.4K val |
| **MOT17** | Primary benchmark (linear motion, pedestrian) | ~5GB | 7 train + 7 test |
| **MOT20** | Crowded scenes (246 ped/frame) | ~5GB | 4 train + 4 test |
| **DanceTrack** | Non-linear complex motion, similar appearance | ~20GB | 40 train + 25 val + 35 test |

MOT Challenge account cần tạo sớm để submit test set evaluation (review 24–48h).

Format chuẩn (unified) trong `datasets/standard/<name>/`:
```
images/<seq>/<frame_id>.jpg
annotations/<seq>.json
```

---

## 9. Hardware & Environment

- **Workstation:** 2× NVIDIA GeForce RTX 5090 (Blackwell sm_120), 32GB VRAM each = 64GB pooled.
- **Driver:** 575.64.03, CUDA 12.9.
- **Python:** 3.11, conda env `yolo_jdt` (isolated).
- **PyTorch:** ≥ 2.7 với cu128 wheel (Blackwell sm_120 kernel requirement).
- **Training:** DDP qua Lightning, BF16 mixed precision.

---

## 10. Repository Layout

```
yolo-jdt/
├── CLAUDE.md                              # Claude Code context
├── YOLO_JDT_Research_Document.md          # Research blueprint (read-only)
├── plans/yolo_jdt_overview.md             # File này
├── todos.md                               # Step checkpoint với resume prompts
├── README.md                              # Setup + usage guide
├── environment.yml                        # Conda env reproducibility
├── requirements-lock.txt                  # Pip freeze
├── pyproject.toml
├── ruff.toml
├── .gitignore
├── .claude/                               # Claude Code workspace config
│   ├── settings.json
│   ├── skills/                            # 8 skills
│   ├── agents/                            # 2 agents
│   └── hooks/
├── yolo_jdt/                              # Main package
│   ├── models/         # backbone, neck, joint head, TAGate
│   ├── data/           # datasets, loaders, converters
│   ├── losses/         # detection, reid, offset, uncertainty
│   ├── tracker/        # ByteTrack, BoT-SORT, our associator
│   ├── train/          # Lightning module, datamodule, callbacks
│   ├── eval/           # MOT (TrackEval), COCO
│   ├── export/         # ONNX, TensorRT, INT8 calibration
│   ├── configs/        # Hydra configs
│   ├── scripts/        # Entry points
│   ├── tests/          # pytest
│   └── utils/
├── third_party/
│   ├── ultralytics_extract/               # Vendored YOLO11 blocks
│   └── yolo26_extract/                    # Phase 9
├── datasets/{raw,standard}/
├── weights/{pretrained,ours,exported}/
└── runs/{baselines,ablation,final}/
```

---

## 11. Risk Register (tóm tắt)

| Rủi ro | Mức độ | Mitigation |
|--------|--------|------------|
| Gradient conflict detect/ReID/offset | Cao | Uncertainty weighting + stage-wise training + gradient logging per task |
| TAGate overhead vượt FPS budget | TB | Giảm xuống 1 layer, cache chỉ P5, SDPA backend |
| YOLO26 source chưa public | TB | Develop chính trên YOLO11 trước, YOLO26 là extension không phải dependency |
| MOT Challenge submission timing | Thấp | Tạo account sớm Phase 1, submit Phase 11 |
| Reviewer hỏi method mới release trong review period | Thấp | Codebase flexible, dễ thêm baseline |
| Blackwell PyTorch wheel issues | TB | Verify cu128 install ngay Phase 0, fallback nightly nếu cần |

Chi tiết → research doc §11.

---

## 12. Tham chiếu

| File | Vai trò |
|------|---------|
| [`../YOLO_JDT_Research_Document.md`](../YOLO_JDT_Research_Document.md) | Literature review, benchmarks, kiến trúc chi tiết, references |
| [`../CLAUDE.md`](../CLAUDE.md) | Claude Code context (facts + pointers) |
| [`../todos.md`](../todos.md) | Step checkpoint với resume prompts |
| `~/.claude/plans/d-ng-th-ng-tin-v-a-mutable-blum.md` | Execution plan đầy đủ (mỗi phase entry/exit criteria) |
| `../.claude/skills/` | Reusable procedures (check-env, train, eval-mot, ...) |
| `../.claude/agents/` | Specialized reviewers (paper-reviewer, production-reviewer) |

---

*Document version: 1.0 — Khởi tạo cùng Step 0.A. Update mỗi khi phase major kết thúc.*
