#!/usr/bin/env bash
# Job 7a — Pretrain Arm C (Supervised Contrastive, SupCon).
# Uses PUBLIC LABELS at pretraining only; deployment stays label-free.
# REQUIRES the supcon objective to be implemented first — see
#   Layer3/reports/LAYER3_ARM_C_SUPERVISED_SPEC.md  (§7 code changes).
set -euo pipefail
cd "$(dirname "$0")/../../.."

WINDOW_INDEX="${WINDOW_INDEX:-Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
CKPT_DIR="${CKPT_DIR:-Results/layer3/pretrain/supcon_mitbih_seed0_100ep_8s_goldexcluded}"
LABEL_COL="${LABEL_COL:-safety_group}"
LABEL_MAP="${LABEL_MAP:-NORMAL=normal,DANGEROUS=unsafe,NOISE=unsafe,BENIGN_ABNORMAL=benign,AF_CONTEXT=drop}"
SEED="${SEED:-0}"

mkdir -p "$CKPT_DIR"

python Layer3/tools/pretrain_encoder.py \
  --window-index "$WINDOW_INDEX" \
  --checkpoint-dir "$CKPT_DIR" \
  --ssl-objective supcon \
  --label-col "$LABEL_COL" \
  --label-map "$LABEL_MAP" \
  --exclude-records-csv "$GOLD_CSV" \
  --augment-fs 125 \
  --supcon-temperature "${SUPCON_TEMPERATURE:-0.1}" \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr 3e-4 \
  --num-workers "${NUM_WORKERS:-4}" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda}"

echo "[OK] Arm C (SupCon) checkpoint dir -> $CKPT_DIR"
echo "Verify pretrain_records.json: gold excluded + labels_used_in_pretraining_only=true"
