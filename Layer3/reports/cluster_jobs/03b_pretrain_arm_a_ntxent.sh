#!/usr/bin/env bash
# Job 3b — Pretrain Arm A (NT-Xent). Hold gold eval records out of SSL.
set -euo pipefail
cd "$(dirname "$0")/../../.."

WINDOW_INDEX="${WINDOW_INDEX:-Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
CKPT_DIR="${CKPT_DIR:-Results/layer3/pretrain/ntxent_mitbih_seed0}"
SEED="${SEED:-0}"

mkdir -p "$CKPT_DIR"

python Layer3/tools/pretrain_encoder.py \
  --window-index "$WINDOW_INDEX" \
  --checkpoint-dir "$CKPT_DIR" \
  --ssl-objective ntxent \
  --positive-mode same_window \
  --exclude-records-csv "$GOLD_CSV" \
  --augment-fs 125 \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr 3e-4 \
  --num-workers "${NUM_WORKERS:-4}" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda}"

echo "[OK] Arm A checkpoint dir -> $CKPT_DIR"
echo "Check: $CKPT_DIR/encoder_last.pt and $CKPT_DIR/pretrain_records.json"
