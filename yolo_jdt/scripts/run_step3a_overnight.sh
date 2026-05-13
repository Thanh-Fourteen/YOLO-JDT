#!/usr/bin/env bash
# Step 3.A overnight orchestration (single GPU 0, ~15h budget).
#
# YOLO11s 70 epochs bs=8, CrowdHuman + MOT17 train_half, BF16,
# EMA decay 0.9999, sync to wandb. Promotes best ckpt -> weights/ours/
# then evaluates mAP on MOT17 val_half + CrowdHuman val -> JSON.
#
# Launch detached:
#   nohup setsid bash yolo_jdt/scripts/run_step3a_overnight.sh \
#     > /tmp/step3a.log 2>&1 < /dev/null &
#   disown

set -u   # NOT -e: we want to keep going (promote/eval) past partial failures
exec > >(tee -a /tmp/step3a.log) 2>&1
trap 'echo "[$(date +%H:%M:%S)] ABORTED on signal"; exit 130' INT TERM

PROJECT=/home/tris/thanh/yolo-jdt
cd "$PROJECT"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate yolo_jdt
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

echo "[$(date +%H:%M:%S)] ============================================"
echo "[$(date +%H:%M:%S)]  Step 3.A overnight — YOLO11s 70ep bs=8"
echo "[$(date +%H:%M:%S)] ============================================"

echo "[$(date +%H:%M:%S)] === Pre-flight ==="
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader
df -h "$PROJECT" | tail -1
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MB" -lt 3000 ]; then
  echo "[ABORT] GPU 0 has only ${FREE_MB} MB free, need >= 3000 MB"
  exit 1
fi
echo "[$(date +%H:%M:%S)] pre-flight OK (GPU 0 ${FREE_MB} MB free)"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [1/3] Train YOLO11s 70ep bs=8 ==="
python -m yolo_jdt.scripts.train -cn base \
  run_name=step3a_yolo11s_70ep \
  trainer.devices=1 trainer.max_epochs=70 \
  trainer.check_val_every_n_epoch=1 trainer.val_check_interval=1.0 \
  data.batch_size=8 data.num_workers=4 \
  model.scale=s model.lr0=0.001 model.warmup_epochs=1.0 \
  model.warmup_bias_lr=0.01 \
  model.pretrained_weights=weights/pretrained/yolo11s.pt \
  close_mosaic_fraction=0.85 \
  wandb.enabled=true \
  wandb.notes=step3a_yolo11s_70ep_bs8_singleGPU_overnight \
  || { echo "[$(date +%H:%M:%S)] [1/3] TRAIN FAILED — exiting"; exit 1; }
echo "[$(date +%H:%M:%S)] [1/3] train OK"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [2/3] Promote best ckpt ==="
python -m yolo_jdt.scripts.promote_ckpt \
  --src runs/baselines/step3a_yolo11s_70ep \
  --dst weights/ours/yolo11s_det.pt \
  --scale s \
  || { echo "[$(date +%H:%M:%S)] [2/3] PROMOTE FAILED — leaving best Lightning ckpt in place"; exit 1; }
echo "[$(date +%H:%M:%S)] [2/3] promote OK"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [3/4] Eval mAP -> JSON ==="
python -m yolo_jdt.scripts.eval_det \
  --output runs/baselines/detection_map.json \
  --weights weights/ours/yolo11s_det.pt --scale s \
  --batch_size 8 --num_workers 4 \
  || echo "[$(date +%H:%M:%S)] [3/4] EVAL FAILED — ckpt is still saved, run manually on return"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [4/4] Package run dir (RUN_INFO.md + snapshots) ==="
python -m yolo_jdt.scripts.package_run \
  --run-dir runs/baselines/step3a_yolo11s_70ep \
  --log-source /tmp/step3a.log \
  --eval-json runs/baselines/detection_map.json \
  --promoted weights/ours/yolo11s_det.pt \
  --phase "3.A" \
  --notes "Step 3.A baseline overnight: YOLO11s 70ep single-GPU bs=8 lr0=0.001 BF16 EMA, CrowdHuman + MOT17 train_half (person-only)." \
  || echo "[$(date +%H:%M:%S)] [4/4] PACKAGE FAILED — RUN_INFO.md not generated, but artifacts still in place"

echo ""
echo "[$(date +%H:%M:%S)] === ALL DONE ==="
echo ""
echo "Run dir:"
ls -la runs/baselines/step3a_yolo11s_70ep/ 2>/dev/null
echo ""
echo "See full summary:"
echo "  cat runs/baselines/step3a_yolo11s_70ep/RUN_INFO.md"
