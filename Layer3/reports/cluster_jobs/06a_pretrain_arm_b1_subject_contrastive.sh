#!/usr/bin/env bash
# Job 6a — Pretrain Arm B1 (masked + subject contrastive). ABLATION only.
# Prefer --healthy-only. Run AFTER primary B one-seed exists for comparison.
set -euo pipefail
cd "$(dirname "$0")/../../.."

WINDOW_INDEX="${WINDOW_INDEX:-Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
CKPT_DIR="${CKPT_DIR:-Results/layer3/pretrain/mae_subject_contrastive_mitbih_seed0}"
SEED="${SEED:-0}"

mkdir -p "$CKPT_DIR"

python Layer3/tools/pretrain_encoder.py \
  --window-index "$WINDOW_INDEX" \
  --checkpoint-dir "$CKPT_DIR" \
  --ssl-objective mae_subject_contrastive \
  --healthy-only \
  --exclude-records-csv "$GOLD_CSV" \
  --mask-ratio 0.75 \
  --mask-patch-size 25 \
  --subject-contrastive-lambda 0.30 \
  --subject-col record_id \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr 3e-4 \
  --num-workers "${NUM_WORKERS:-4}" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda}"

echo "[OK] Arm B1 (ablation) checkpoint dir -> $CKPT_DIR"
echo "Label results as B1 ablation, not primary B."
