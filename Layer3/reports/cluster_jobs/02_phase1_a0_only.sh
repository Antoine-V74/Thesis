#!/usr/bin/env bash
# Job 2 — Phase 1 A0-only (no SSL checkpoint required).
# Trust ONLY phase1_*.csv in OUT_DIR (ignore encoder/per_beat junk).
set -euo pipefail
cd "$(dirname "$0")/../../.."

DATA_DIR="${DATA_DIR:-data}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
OUT_DIR="${OUT_DIR:-Results/layer3/validation/pilot_mitbih_A0_only_8s}"

mkdir -p "$OUT_DIR"

python Layer3/validation/run_beat_validation.py \
  --data-dir "$DATA_DIR" \
  --datasets mit_bih_arrhythmia \
  --records-csv "$GOLD_CSV" \
  --out-dir "$OUT_DIR" \
  --mode oracle \
  --window-s 8 --target-fs 125 \
  --causal-window --lookahead-ms 100 \
  --per-record-calibration --guard-s 8 \
  --phase1-eval --phase1-arms a0 \
  --phase1-scorers mahalanobis,knn \
  --threshold-method conformal --conformal-alpha 0.10 \
  --device "${DEVICE:-cuda}"

echo "[OK] Phase 1 A0 outputs -> $OUT_DIR"
echo "Read: phase1_metrics_bootstrap.csv, phase1_metrics_overall.csv (arm=a0_layer2_features)"
