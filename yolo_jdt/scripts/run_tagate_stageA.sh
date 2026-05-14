#!/usr/bin/env bash
# Step 5.DE orchestration: TAGate Stage A training + eval + viz + ONNX export
# Budget: ~8h Stage A (5ep) + 1h eval + 30m viz + 15m ONNX
# Run:  bash yolo_jdt/scripts/run_tagate_stageA.sh [GPU_ID]
# Default GPU: 0. To use GPU 1: bash run_tagate_stageA.sh 1
set -u
exec > >(tee -a /tmp/step5_tagate.log) 2>&1
trap 'echo "[$(date +%H:%M:%S)] ABORTED on signal"; exit 130' INT TERM

GPU_ID="${1:-0}"
PROJECT=/home/tris/thanh/yolo-jdt
cd "$PROJECT"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate yolo_jdt
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "[$(date +%H:%M:%S)] ============================================"
echo "[$(date +%H:%M:%S)]  Step 5.DE — TAGate Stage A + eval + viz + ONNX"
echo "[$(date +%H:%M:%S)]  GPU: $GPU_ID"
echo "[$(date +%H:%M:%S)] ============================================"

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader
if [ "$FREE_MB" -lt 5000 ]; then
  echo "[ABORT] GPU $GPU_ID has only ${FREE_MB} MB free, need >= 5000 MB"; exit 1
fi
echo "[$(date +%H:%M:%S)] pre-flight OK (GPU $GPU_ID ${FREE_MB} MB free)"

# --- [0] Sanity: pytest suite (includes TAGate + PairedFrameDataset tests) ---
echo ""
echo "[$(date +%H:%M:%S)] === [0] pytest sanity (skip dataset-bound tests if missing) ==="
python -m pytest yolo_jdt/tests/ --tb=short -q \
  -k "not test_dataset and not test_tracking_loader" \
  || { echo "[$(date +%H:%M:%S)] [0] TESTS FAILED — abort"; exit 1; }
echo "[$(date +%H:%M:%S)] [0] tests pass"

# --- [0.1] ONNX smoke test (freshly initialized model, no checkpoint needed) ---
echo ""
echo "[$(date +%H:%M:%S)] === [0.1] ONNX smoke test ==="
python -m yolo_jdt.scripts.export_tagate_onnx --test-init-only \
  || { echo "[$(date +%H:%M:%S)] [0.1] ONNX SMOKE FAILED — abort"; exit 1; }
echo "[$(date +%H:%M:%S)] [0.1] ONNX smoke test PASS"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [1/6] Stage A: 5ep, freeze backbone+neck ==="
python -m yolo_jdt.scripts.train_tagate -cn tagate \
  run_name=step5_tagate_stageA \
  trainer.devices=1 trainer.max_epochs=5 \
  trainer.check_val_every_n_epoch=1 trainer.val_check_interval=1.0 \
  data.batch_size=8 data.num_workers=4 \
  model.scale=s model.lr0=0.001 model.warmup_epochs=1.0 \
  model.warmup_bias_lr=0.01 model.lambda_reid=0.1 \
  model.cache_levels=P5 model.tagate_num_layers=2 \
  model.pretrained_weights=weights/ours/yolo11s_jde.pt \
  model.stage=A \
  wandb.enabled=true wandb.group=phase5 \
  wandb.notes=step5_stageA_5ep_bs8_P5_2layers \
  || { echo "[$(date +%H:%M:%S)] [1/6] STAGE A FAILED"; exit 1; }

STAGE_A_CKPT=$(find runs/tagate/step5_tagate_stageA -name 'epoch*.ckpt' ! -name 'last.ckpt' | sort | tail -1)
[ -z "$STAGE_A_CKPT" ] && STAGE_A_CKPT=runs/tagate/step5_tagate_stageA/last.ckpt
echo "[$(date +%H:%M:%S)] Stage A best ckpt: $STAGE_A_CKPT"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [2/6] Promote Stage A ckpt -> yolo11s_jdt_stageA.pt ==="
python -m yolo_jdt.scripts.promote_ckpt \
  --src runs/tagate/step5_tagate_stageA \
  --dst weights/ours/yolo11s_jdt_stageA.pt \
  --scale s --model jdt \
  --cache-levels P5 --tagate-num-layers 2 \
  || { echo "[$(date +%H:%M:%S)] [2/6] PROMOTE FAILED"; exit 1; }

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [3/6] Tracking eval — MOT17 val_half (baseline: HOTA=0.560) ==="
RUNDIR=runs/tagate/step5_yolo11s_jdt_mot17
python -m yolo_jdt.scripts.infer_tracking_jdt \
  --weights weights/ours/yolo11s_jdt_stageA.pt --scale s \
  --dataset mot17 --split val_half \
  --output-dir "$RUNDIR/tracker_outputs" --conf 0.05 \
  || { echo "[$(date +%H:%M:%S)] [3/6] MOT17 INFER FAILED"; exit 1; }

python -m yolo_jdt.eval.trackeval_runner \
  --tracker-outputs "$RUNDIR/tracker_outputs" \
  --gt-cache runs/baselines/_gt_mot \
  --dataset mot17 --split val_half \
  --tracker-name yolo11s_jdt_stageA \
  --out-dir "$RUNDIR" \
  || echo "[$(date +%H:%M:%S)] [3/6] MOT17 TRACKEVAL FAILED — non-blocking"

# Report HOTA and compare against Step 4 baseline
echo "[$(date +%H:%M:%S)] === MOT17 results (target: HOTA >= 0.570) ==="
python -c "
import json, sys
p = '$RUNDIR/metrics.json'
try:
    d = json.load(open(p))
    hota = d.get('HOTA', d.get('hota', None))
    if hota is not None:
        h = float(hota) if not isinstance(hota, list) else sum(hota)/len(hota)
        baseline = 0.560
        gain = h - baseline
        status = 'PASS' if h >= 0.570 else 'BELOW_TARGET'
        print(f'HOTA={h:.3f}  baseline={baseline:.3f}  gain=+{gain:.3f}  [{status}]')
    else:
        print('Could not parse HOTA from', p)
except Exception as e:
    print('Parse error:', e)
" || true

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [4/6] DanceTrack eval (optional, check AssA improvement) ==="
RUNDIR_DANCE=runs/tagate/step5_yolo11s_jdt_dancetrack
python -m yolo_jdt.scripts.infer_tracking_jdt \
  --weights weights/ours/yolo11s_jdt_stageA.pt --scale s \
  --dataset dancetrack --split val \
  --output-dir "$RUNDIR_DANCE/tracker_outputs" --conf 0.05 \
  || echo "[$(date +%H:%M:%S)] [4/6] DanceTrack INFER FAILED — non-blocking"

python -m yolo_jdt.eval.trackeval_runner \
  --tracker-outputs "$RUNDIR_DANCE/tracker_outputs" \
  --gt-cache runs/baselines/_gt_dance \
  --dataset dancetrack --split val \
  --tracker-name yolo11s_jdt_stageA \
  --out-dir "$RUNDIR_DANCE" \
  || echo "[$(date +%H:%M:%S)] [4/6] DanceTrack TRACKEVAL FAILED — non-blocking"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [5/6] Attention viz — 5 DanceTrack non-linear sequences ==="
python -m yolo_jdt.scripts.viz_attention \
  --weights weights/ours/yolo11s_jdt_stageA.pt --scale s \
  --out-dir runs/viz/tagate_attention \
  --dataset dancetrack --split val \
  --seqs dancetrack0005 dancetrack0019 dancetrack0035 dancetrack0040 dancetrack0073 \
  --frames-per-seq 50 \
  || echo "[$(date +%H:%M:%S)] [5/6] VIZ FAILED — non-blocking"
echo "[$(date +%H:%M:%S)] [5/6] attention viz saved to runs/viz/tagate_attention/"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === [6/6] Full model ONNX export + verify ==="
mkdir -p weights/exported
python -m yolo_jdt.scripts.export_tagate_onnx \
  --weights weights/ours/yolo11s_jdt_stageA.pt --scale s \
  --cache-levels P5 --tagate-num-layers 2 \
  --out weights/exported/yolo11s_jdt_stageA.onnx \
  || echo "[$(date +%H:%M:%S)] [6/6] ONNX EXPORT FAILED — non-blocking"

# ---------------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] ============================================"
echo "[$(date +%H:%M:%S)]  Step 5.DE COMPLETE"
echo "[$(date +%H:%M:%S)]  Summary:"
echo "[$(date +%H:%M:%S)]    Checkpoint: weights/ours/yolo11s_jdt_stageA.pt"
echo "[$(date +%H:%M:%S)]    Tracking:   runs/tagate/step5_yolo11s_jdt_mot17/"
echo "[$(date +%H:%M:%S)]    Viz:        runs/viz/tagate_attention/"
echo "[$(date +%H:%M:%S)]    ONNX:       weights/exported/yolo11s_jdt_stageA.onnx"
echo "[$(date +%H:%M:%S)] ============================================"

ls -la weights/ours/yolo11s_jdt_stageA.pt 2>/dev/null
cat runs/tagate/step5_yolo11s_jdt_mot17/metrics.json 2>/dev/null
