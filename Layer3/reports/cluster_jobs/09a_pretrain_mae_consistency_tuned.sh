#!/usr/bin/env bash
# Exploratory B: preserve more QRS morphology with milder masking/consistency.
set -euo pipefail
cd "$(dirname "$0")/../../.."

python Layer3/tools/pretrain_encoder.py \
  --window-index Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv \
  --checkpoint-dir Results/layer3/pretrain/exploratory_mae_consistency_m50_c05_seed0 \
  --ssl-objective mae_consistency --healthy-only \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --mask-ratio 0.50 --mask-patch-size 25 --consistency-lambda 0.50 \
  --vicreg-expander-dims 512,512,512 \
  --epochs "${EPOCHS:-100}" --batch-size "${BATCH_SIZE:-256}" --lr 3e-4 \
  --num-workers "${NUM_WORKERS:-4}" --seed "${SEED:-0}" --device "${DEVICE:-cuda}"

