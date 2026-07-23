#!/usr/bin/env bash
# Exploratory A1: healthy-filtered VICReg with weaker invariance pressure.
set -euo pipefail
cd "$(dirname "$0")/../../.."

python Layer3/tools/pretrain_encoder.py \
  --window-index Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv \
  --checkpoint-dir Results/layer3/pretrain/exploratory_vicreg_healthy_mild_seed0 \
  --ssl-objective vicreg --healthy-only \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --augment-fs 125 --augment-p-noise 0.30 --augment-p-wander 0.20 --augment-p-crop 0.0 \
  --vicreg-sim-coeff 10 --vicreg-var-coeff 25 --vicreg-cov-coeff 1 \
  --vicreg-expander-dims 512,512,512 \
  --epochs "${EPOCHS:-100}" --batch-size "${BATCH_SIZE:-256}" --lr 3e-4 \
  --num-workers "${NUM_WORKERS:-4}" --seed "${SEED:-0}" --device "${DEVICE:-cuda}"

