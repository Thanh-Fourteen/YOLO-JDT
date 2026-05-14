#!/usr/bin/env bash
# Step 4 post-training: promote → COCO integrity → tracking eval × 3 datasets
# Expected: ~35-40 min | GPU 0 | BF16
set -u
exec > >(tee -a /tmp/step4_post.log) 2>&1
trap 'echo "[$(date +%H:%M:%S)] ABORTED on signal"; exit 130' INT TERM

PROJECT=/home/tris/thanh/yolo-jdt
cd "$PROJECT"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate yolo_jdt
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

echo "[$(date +%H:%M:%S)] ======================================"
echo "[$(date +%H:%M:%S)]  Step 4 post: promote + COCO + eval×3"
echo "[$(date +%H:%M:%S)] ======================================"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [1/5] Promote StageB → yolo11s_jde.pt ==="
python -m yolo_jdt.scripts.promote_ckpt \
  --src runs/jde/step4_jde_yolo11s_stageB \
  --dst weights/ours/yolo11s_jde.pt \
  --scale s --model jde \
  || { echo "[$(date +%H:%M:%S)] [1/5] PROMOTE FAILED"; exit 1; }
echo "[$(date +%H:%M:%S)] [1/5] DONE — weights/ours/yolo11s_jde.pt"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [2/5] COCO integrity (Step 4.E) ==="
mkdir -p runs/integrity
python -m yolo_jdt.scripts.eval_coco_integrity \
  --weights weights/pretrained/yolo11s.pt --scale s \
  --output runs/integrity/step4_coco_yolo11s.json \
  || { echo "[$(date +%H:%M:%S)] [2/5] COCO INTEGRITY FAILED"; exit 1; }
echo "[$(date +%H:%M:%S)] [2/5] DONE"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [3/5] Tracking infer + eval — MOT17 val_half ==="
RUNDIR=runs/jde/step4_yolo11s_botsort_reid_mot17
python -m yolo_jdt.scripts.infer_tracking_jde \
  --weights weights/ours/yolo11s_jde.pt --scale s \
  --dataset mot17 --split val_half --tracker botsort_reid \
  --output-dir "$RUNDIR/tracker_outputs" --conf 0.05 \
  || { echo "[$(date +%H:%M:%S)] [3/5] MOT17 INFER FAILED"; exit 1; }
python -m yolo_jdt.eval.trackeval_runner \
  --tracker-outputs "$RUNDIR/tracker_outputs" \
  --gt-cache runs/baselines/_gt_mot \
  --dataset mot17 --split val_half \
  --tracker-name yolo11s_botsort_reid \
  --out-dir "$RUNDIR" \
  || { echo "[$(date +%H:%M:%S)] [3/5] MOT17 TRACKEVAL FAILED"; exit 1; }
echo "[$(date +%H:%M:%S)] [3/5] MOT17 DONE"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [4/5] Tracking infer + eval — MOT20 val_half ==="
RUNDIR=runs/jde/step4_yolo11s_botsort_reid_mot20
python -m yolo_jdt.scripts.infer_tracking_jde \
  --weights weights/ours/yolo11s_jde.pt --scale s \
  --dataset mot20 --split val_half --tracker botsort_reid \
  --output-dir "$RUNDIR/tracker_outputs" --conf 0.05 \
  || { echo "[$(date +%H:%M:%S)] [4/5] MOT20 INFER FAILED"; exit 1; }
python -m yolo_jdt.eval.trackeval_runner \
  --tracker-outputs "$RUNDIR/tracker_outputs" \
  --gt-cache runs/baselines/_gt_mot \
  --dataset mot20 --split val_half \
  --tracker-name yolo11s_botsort_reid \
  --out-dir "$RUNDIR" \
  || { echo "[$(date +%H:%M:%S)] [4/5] MOT20 TRACKEVAL FAILED"; exit 1; }
echo "[$(date +%H:%M:%S)] [4/5] MOT20 DONE"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [5/5] Tracking infer + eval — DanceTrack val ==="
RUNDIR=runs/jde/step4_yolo11s_botsort_reid_dancetrack
python -m yolo_jdt.scripts.infer_tracking_jde \
  --weights weights/ours/yolo11s_jde.pt --scale s \
  --dataset dancetrack --split val --tracker botsort_reid \
  --output-dir "$RUNDIR/tracker_outputs" --conf 0.05 \
  || { echo "[$(date +%H:%M:%S)] [5/5] DANCETRACK INFER FAILED"; exit 1; }
python -m yolo_jdt.eval.trackeval_runner \
  --tracker-outputs "$RUNDIR/tracker_outputs" \
  --gt-cache runs/baselines/_gt_mot \
  --dataset dancetrack --split val \
  --tracker-name yolo11s_botsort_reid \
  --out-dir "$RUNDIR" \
  || { echo "[$(date +%H:%M:%S)] [5/5] DANCETRACK TRACKEVAL FAILED"; exit 1; }
echo "[$(date +%H:%M:%S)] [5/5] DANCETRACK DONE"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === SUMMARY ==="
echo "--- COCO integrity ---"
cat runs/integrity/step4_coco_yolo11s.json 2>/dev/null
echo ""
echo "--- Tracking results ---"
for DS in mot17 mot20 dancetrack; do
  RFILE=runs/jde/step4_yolo11s_botsort_reid_${DS}/summary.json
  echo "[$DS]"; cat "$RFILE" 2>/dev/null || echo "(no summary.json)"
done

echo ""
echo "[$(date +%H:%M:%S)] === ALL DONE ==="
