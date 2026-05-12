# YOLO-JDT

Joint Detection-Tracking via Temporal Attention Gate (TAGate) on YOLO11 / YOLO26.

## Setup

```bash
# 1. Create isolated conda env (one-time)
conda env create -f environment.yml
conda activate yolo_jdt

# 2. Verify environment
python yolo_jdt/scripts/check_env.py

# 3. (Optional, before first training run) login to Weights & Biases
wandb login              # paste API key from https://wandb.ai/authorize
# or, for offline runs:  wandb offline
# or, set env var:       export WANDB_API_KEY=<key>
# project name `yolo-jdt` is created automatically on first wandb.init() call.

# 4. Pre-commit hooks (optional, recommended for contributors)
pre-commit install
```

## Hardware requirement

- NVIDIA GPU with sm_120 (Blackwell, e.g. RTX 5090) for full training.
- PyTorch ≥ 2.7 with CUDA 12.8 wheels (pinned in `environment.yml`).
- For ablation runs with smaller models, sm_80+ (Ampere) should work but is untested.

## Project orientation

| File | Purpose |
|------|---------|
| [`CLAUDE.md`](CLAUDE.md) | Working context for Claude Code (and any contributor) |
| [`plans/yolo_jdt_overview.md`](plans/yolo_jdt_overview.md) | High-level project plan |
| [`todos.md`](todos.md) | Step-by-step checkpoint with resume prompts |
| [`YOLO_JDT_Research_Document.md`](YOLO_JDT_Research_Document.md) | Research blueprint (literature, benchmarks, architecture) |
| `yolo_jdt/` | Main Python package |
| `third_party/` | Vendored upstream code (Ultralytics YOLO11, YOLO26) — AGPL-3.0 |
| `datasets/`, `weights/`, `runs/` | Data / checkpoints / outputs (gitignored) |

## Usage (Claude Code)

```bash
claude              # opens Claude in this workspace
# inside Claude Code:
/check-env          # verify environment is OK
/train base         # start a training run
/eval-mot ...       # evaluate a checkpoint
/ablation A1        # run an ablation experiment
```

Skills are under `.claude/skills/`. See `CLAUDE.md` for the full skill inventory.

## License

Project license: TBD. Vendored Ultralytics code in `third_party/ultralytics_extract/` is AGPL-3.0; see attribution headers per file.

### Dataset licenses

Datasets under `datasets/raw/` are downloaded from upstream sources for research use; they are **not** redistributed by this repository. Each dataset retains its own license:

| Dataset | License / Terms | Notes |
|---------|-----------------|-------|
| MOT17 | CC BY-NC-SA 3.0 | Research / non-commercial |
| MOT20 | CC BY-NC-SA 3.0 | Research / non-commercial |
| DanceTrack | MIT | Research-friendly |
| **CrowdHuman** | **Non-commercial research and educational use only** | Per upstream terms (https://www.crowdhuman.org/download.html). **Do NOT redistribute** images or annotations. Any production / commercial use of weights trained on CrowdHuman requires re-training without CrowdHuman, or obtaining an explicit commercial license from the dataset authors. Note this in the model card before any release. |
| COCO val2017 | CC BY 4.0 | Used only for detection mAP sanity-check (Phase 2). |
