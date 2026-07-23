#!/usr/bin/env bash
# Job 14a - Pretrain Arm C1 (SupCon + Outlier Exposure).
# EXPLORATORY improvement on Arm C. Public labels at PRETRAINING ONLY; deployment
# stays label-free (per-record healthy Mahalanobis/kNN + conformal). Same frozen
# scorer + gold exclusion as every arm - only the objective changes, so any delta
# is attributable to the representation.
# GUARDRAIL: exploratory on the already-inspected 13 gold records; confirm on an
# untouched cohort before any claim. A0 remains the deployable baseline.
set -euo pipefail
cd "$(dirname "$0")/../../.."

WINDOW_INDEX="${WINDOW_INDEX:-Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
CKPT_DIR="${CKPT_DIR:-Results/layer3/pretrain/supcon_oe_mitbih_seed0_100ep_8s_goldexcluded}"
LABEL_COL="${LABEL_COL:-safety_group}"
LABEL_MAP="${LABEL_MAP:-NORMAL=normal,DANGEROUS=unsafe,NOISE=unsafe,BENIGN_ABNORMAL=benign,AF_CONTEXT=drop}"
SEED="${SEED:-0}"

mkdir -p "$CKPT_DIR"

python Layer3/tools/pretrain_encoder.py \
  --window-index "$WINDOW_INDEX" \
  --checkpoint-dir "$CKPT_DIR" \
  --ssl-objective supcon_oe \
  --label-col "$LABEL_COL" \
  --label-map "$LABEL_MAP" \
  --exclude-records-csv "$GOLD_CSV" \
  --augment-fs 125 \
  --supcon-temperature "${SUPCON_TEMPERATURE:-0.1}" \
  --oe-weight "${OE_WEIGHT:-1.0}" \
  --center-init-windows "${CENTER_INIT_WINDOWS:-2048}" \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr 3e-4 \
  --num-workers "${NUM_WORKERS:-4}" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda}"

echo "[OK] Arm C1 (supcon_oe) checkpoint dir -> $CKPT_DIR"
echo "Verify pretrain_records.json: gold excluded + labels_used_in_pretraining_only=true + oe_weight logged"
