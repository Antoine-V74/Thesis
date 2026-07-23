#!/usr/bin/env bash
# Exploratory 30 s Phase 1. Do not replace the locked 8 s primary result.
set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT="Results/layer3/pretrain/exploratory_vicreg_healthy_30s_avgmax_seed0/encoder_last.pt"
test -f "$CKPT" || { echo "[ERROR] Missing $CKPT" >&2; exit 1; }

python Layer3/validation/run_beat_validation.py \
  --data-dir data --datasets mit_bih_arrhythmia \
  --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --checkpoint "$CKPT" \
  --out-dir Results/layer3/validation/exploratory_vicreg_healthy_30s_avgmax_seed0 \
  --mode oracle --window-s 30 --target-fs 125 --causal-window --lookahead-ms 100 \
  --per-record-calibration --guard-s 30 \
  --encoder-pooling-mode avg_max \
  --l2-normalize-embeddings --pca-dim 32 \
  --phase1-eval --phase1-arms a0,layer3 --phase1-scorers mahalanobis,knn \
  --threshold-method conformal --conformal-alpha 0.10 \
  --no-random-fallback --device "${DEVICE:-cuda}"

