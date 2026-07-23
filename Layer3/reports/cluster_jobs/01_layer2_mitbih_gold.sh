#!/usr/bin/env bash
# Job 1 — Full Layer 2 beat validation on MIT-BIH gold (oracle + cadence modes in one run).
# Usage (from repo root on the cluster):
#   bash Layer3/reports/cluster_jobs/01_layer2_mitbih_gold.sh
# Or: sbatch this file if your site wraps bash in #SBATCH headers externally.
set -euo pipefail
cd "$(dirname "$0")/../../.."

DATA_DIR="${DATA_DIR:-data}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
OUT_DIR="${OUT_DIR:-Results/layer2/beat_sync_mitbih_gold_causal}"

mkdir -p "$OUT_DIR"

python Layer2/validation/run_beat_validation.py \
  --data-dir "$DATA_DIR" \
  --datasets mit_bih_arrhythmia \
  --records-csv "$GOLD_CSV" \
  --out-dir "$OUT_DIR" \
  --feature-sets all \
  --feature-window-mode causal \
  --post-r-lookahead-s 0.08 \
  --morphology-window-s 5.0 \
  --rr-lookback-s 30.0 \
  --per-record-calibration \
  --threshold-method conformal \
  --conformal-alpha 0.10 \
  --anomaly-model mahalanobis \
  --guard-s 5.0 \
  --cadence-observation-lookahead-s 0.40 \
  --cadence-min-safe-observations 6

echo "[OK] Layer 2 outputs -> $OUT_DIR"
echo "Read: metrics_overall.csv, metrics_by_safety_group.csv, per_beat.csv (modes include oracle + cadence_1of8)"
