#!/usr/bin/env bash
# Phase 1: YOLO11s JDE (Step 4) + Phase 2: YOLO11m Step 3.A detection
# Budget: 19h | GPU 0 | bs=8 BF16 | Expected: ~17.9h
set -u
exec > >(tee -a /tmp/step4_jde.log) 2>&1
trap 'echo "[$(date +%H:%M:%S)] ABORTED on signal"; exit 130' INT TERM

PROJECT=/home/tris/thanh/yolo-jdt
cd "$PROJECT"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate yolo_jdt
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

echo "[$(date +%H:%M:%S)] ============================================"
echo "[$(date +%H:%M:%S)]  Phase 1: YOLO11s JDE + Phase 2: YOLO11m det"
echo "[$(date +%H:%M:%S)] ============================================"

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader
df -h "$PROJECT" | tail -1
if [ "$FREE_MB" -lt 5000 ]; then
  echo "[ABORT] GPU 0 has only ${FREE_MB} MB free, need >= 5000 MB"; exit 1
fi
echo "[$(date +%H:%M:%S)] pre-flight OK (GPU 0 ${FREE_MB} MB free)"

echo "[$(date +%H:%M:%S)] === [0] pytest sanity (85 tests) ==="
python -m pytest yolo_jdt/tests/ --tb=short -q \
  || { echo "[$(date +%H:%M:%S)] [0] TESTS FAILED — abort"; exit 1; }
echo "[$(date +%H:%M:%S)] [0] all tests pass"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [1/5] Stage A: 5ep, freeze BB+neck ==="
python -m yolo_jdt.scripts.train_jde -cn jde \
  run_name=step4_jde_yolo11s_stageA \
  trainer.devices=1 trainer.max_epochs=5 \
  trainer.check_val_every_n_epoch=1 trainer.val_check_interval=1.0 \
  data.batch_size=8 data.num_workers=4 \
  model.scale=s model.lr0=0.001 model.warmup_epochs=1.0 \
  model.warmup_bias_lr=0.01 model.lambda_reid=0.1 \
  model.pretrained_weights=weights/ours/yolo11s_det.pt \
  model.stage=A close_mosaic_fraction=0.85 \
  wandb.enabled=true wandb.group=phase4 \
  wandb.notes=step4_stageA_5ep_bs8_singleGPU \
  || { echo "[$(date +%H:%M:%S)] [1/5] STAGE A FAILED"; exit 1; }

STAGE_A_CKPT=$(find runs/jde/step4_jde_yolo11s_stageA -name 'epoch*.ckpt' ! -name 'last.ckpt' | sort | tail -1)
[ -z "$STAGE_A_CKPT" ] && STAGE_A_CKPT=runs/jde/step4_jde_yolo11s_stageA/last.ckpt
echo "[$(date +%H:%M:%S)] Stage A best ckpt: $STAGE_A_CKPT"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [2/5] Stage B: 15ep, unfreeze all, lr=0.0001 ==="
python -m yolo_jdt.scripts.train_jde -cn jde \
  run_name=step4_jde_yolo11s_stageB \
  trainer.devices=1 trainer.max_epochs=15 \
  trainer.check_val_every_n_epoch=1 trainer.val_check_interval=1.0 \
  data.batch_size=8 data.num_workers=4 \
  model.scale=s model.lr0=0.001 model.warmup_epochs=1.0 \
  model.warmup_bias_lr=0.01 model.lambda_reid=0.1 \
  "model.pretrained_weights='$STAGE_A_CKPT'" \
  model.stage=B model.stage_b_lr_scale=0.1 \
  close_mosaic_fraction=0.85 \
  wandb.enabled=true wandb.group=phase4 \
  wandb.notes=step4_stageB_15ep_bs8_singleGPU \
  || { echo "[$(date +%H:%M:%S)] [2/5] STAGE B FAILED"; exit 1; }

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [3/5] Promote Stage B best ckpt -> yolo11s_jde.pt ==="
python -m yolo_jdt.scripts.promote_ckpt \
  --src runs/jde/step4_jde_yolo11s_stageB \
  --dst weights/ours/yolo11s_jde.pt \
  --scale s \
  || { echo "[$(date +%H:%M:%S)] [3/5] PROMOTE FAILED"; exit 1; }

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [4/5] Det mAP eval + Step 4.E COCO integrity ==="
python -m yolo_jdt.scripts.eval_det \
  --output runs/jde/step4_jde_yolo11s_stageB/detection_map.json \
  --weights weights/ours/yolo11s_jde.pt --scale s \
  --batch_size 8 --num_workers 4 \
  || echo "[$(date +%H:%M:%S)] [4/5] EVAL DET FAILED"

mkdir -p runs/integrity
python -m yolo_jdt.scripts.eval_coco_integrity \
  --weights weights/pretrained/yolo11s.pt --scale s \
  --output runs/integrity/step4_coco_yolo11s.json \
  || echo "[$(date +%H:%M:%S)] [4.E] COCO INTEGRITY FAILED — non-blocking"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [5/5] Tracking eval x3 datasets ==="
for DATASET in mot17 mot20 dancetrack; do
  SPLIT=val_half; [ "$DATASET" = "dancetrack" ] && SPLIT=val
  RUNDIR=runs/jde/step4_yolo11s_botsort_reid_${DATASET}
  echo "[$(date +%H:%M:%S)] --- ${DATASET} infer ---"
  python -m yolo_jdt.scripts.infer_tracking_jde \
    --weights weights/ours/yolo11s_jde.pt --scale s \
    --dataset "$DATASET" --split "$SPLIT" --tracker botsort_reid \
    --output-dir "$RUNDIR/tracker_outputs" --conf 0.05 \
    || { echo "INFER FAILED ${DATASET}"; continue; }
  echo "[$(date +%H:%M:%S)] --- ${DATASET} trackeval ---"
  python -m yolo_jdt.eval.trackeval_runner \
    --tracker-outputs "$RUNDIR/tracker_outputs" \
    --gt-cache runs/baselines/_gt_mot \
    --dataset "$DATASET" --split "$SPLIT" \
    --tracker-name yolo11s_botsort_reid \
    --out-dir "$RUNDIR" \
    || echo "TRACKEVAL FAILED ${DATASET}"
done

python -m yolo_jdt.scripts.package_run \
  --run-dir runs/jde/step4_jde_yolo11s_stageB \
  --log-source /tmp/step4_jde.log \
  --eval-json runs/jde/step4_jde_yolo11s_stageB/detection_map.json \
  --promoted weights/ours/yolo11s_jde.pt \
  --phase "4" \
  --notes "Step 4 YOLO11s-JDE 2-stage: StageA 5ep freeze BB+neck, StageB 15ep full unfreeze. bs=8 lr=0.001 BF16 EMA. CrowdHuman+MOT17 train_half person-only. COCO integrity + BoT-SORT-ReID tracking eval x3 datasets." \
  || echo "[$(date +%H:%M:%S)] package Phase 1 FAILED — non-blocking"

# ===========================================================================
echo ""
echo "[$(date +%H:%M:%S)] ============================================"
echo "[$(date +%H:%M:%S)]  Phase 2: YOLO11m Step 3.A — 70ep det fine-tune"
echo "[$(date +%H:%M:%S)] ============================================"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [m-1/3] YOLO11m train 70ep (val@every5) ==="
python -m yolo_jdt.scripts.train -cn base \
  run_name=step3a_yolo11m_70ep \
  trainer.devices=1 trainer.max_epochs=70 \
  trainer.check_val_every_n_epoch=5 trainer.val_check_interval=1.0 \
  data.batch_size=8 data.num_workers=4 \
  model.scale=m model.lr0=0.001 model.warmup_epochs=1.0 \
  model.warmup_bias_lr=0.01 \
  model.pretrained_weights=weights/pretrained/yolo11m.pt \
  close_mosaic_fraction=0.85 \
  wandb.enabled=true wandb.group=phase3 \
  wandb.notes=step3a_yolo11m_70ep_bs8_singleGPU_val5 \
  || { echo "[$(date +%H:%M:%S)] [m-1/3] YOLO11m FAILED"; exit 1; }

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [m-2/3] Promote YOLO11m -> yolo11m_det.pt ==="
python -m yolo_jdt.scripts.promote_ckpt \
  --src runs/baselines/step3a_yolo11m_70ep \
  --dst weights/ours/yolo11m_det.pt \
  --scale m \
  || { echo "[$(date +%H:%M:%S)] [m-2/3] PROMOTE FAILED"; exit 1; }

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [m-3/3] Eval YOLO11m det mAP ==="
python -m yolo_jdt.scripts.eval_det \
  --output runs/baselines/detection_map_m.json \
  --weights weights/ours/yolo11m_det.pt --scale m \
  --batch_size 8 --num_workers 4 \
  || echo "[$(date +%H:%M:%S)] [m-3/3] EVAL FAILED"

python -m yolo_jdt.scripts.package_run \
  --run-dir runs/baselines/step3a_yolo11m_70ep \
  --log-source /tmp/step4_jde.log \
  --eval-json runs/baselines/detection_map_m.json \
  --promoted weights/ours/yolo11m_det.pt \
  --phase "3.A-m" \
  --notes "Step 3.A YOLO11m detection fine-tune 70ep from COCO pretrained. bs=8 lr=0.001 BF16 EMA. val@every5ep. CrowdHuman+MOT17 train_half person-only." \
  || echo "[$(date +%H:%M:%S)] package Phase 2 FAILED — non-blocking"

echo ""
echo "[$(date +%H:%M:%S)] === ALL DONE ==="
ls -la weights/ours/
cat runs/jde/step4_jde_yolo11s_stageB/detection_map.json 2>/dev/null
cat runs/baselines/detection_map_m.json 2>/dev/null
