#!/usr/bin/env bash
# Exploratory A: healthy-filtered NT-Xent with milder nuisance augmentations.
set -euo pipefail
cd "$(dirname "$0")/../../.."

python Layer3/tools/pretrain_encoder.py \
  --window-index Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv \
  --checkpoint-dir Results/layer3/pretrain/exploratory_ntxent_healthy_mild_seed0 \
  --ssl-objective ntxent --positive-mode same_window --healthy-only \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --augment-fs 125 --augment-p-noise 0.30 --augment-p-wander 0.20 --augment-p-crop 0.0 \
  --epochs "${EPOCHS:-100}" --batch-size "${BATCH_SIZE:-256}" --lr 3e-4 \
  --num-workers "${NUM_WORKERS:-4}" --seed "${SEED:-0}" --device "${DEVICE:-cuda}"

