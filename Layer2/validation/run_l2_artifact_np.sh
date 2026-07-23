#!/usr/bin/env bash
# Layer 2 SQI-first audit (CPU only), with optional later steps.
#
# Default: only the artifact / lead-off stress test (decide if SQI is meaningful).
# Full beat validation + Neyman-Pearson are OFF unless you opt in — do those
# after SQI looks useful and thresholds are recalibrated
# (see ALGORITHM_SUMMARY.md → "SQI flip-on gate").
#
# Usage:
#   bash Layer2/validation/run_l2_artifact_np.sh
# Optional later:
#   RUN_BEAT_AND_NP=1 bash Layer2/validation/run_l2_artifact_np.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root (ECG Processing)

PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-data}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
OUT_ROOT="${OUT_ROOT:-Results/layer2_artifact_np}"
DANGER_BUDGET="${DANGER_BUDGET:-0.01}"
RUN_BEAT_AND_NP="${RUN_BEAT_AND_NP:-0}"

STRESS_DIR="$OUT_ROOT/artifact_stress"
mkdir -p "$STRESS_DIR"

echo "[1/1 default] Artifact / lead-off stress test (real MIT-BIH windows) -> $STRESS_DIR"
$PY Layer2/validation/run_artifact_stress_test.py \
  --data-dir "$DATA_DIR" \
  --dataset mit_bih_arrhythmia \
  --out-dir "$STRESS_DIR"

echo
echo "[OK] SQI stress outputs:"
echo "  $STRESS_DIR/artifact_detection_summary.csv  (per-artifact inhibit rate)"
echo "  $STRESS_DIR/artifact_clean_baseline.csv     (clean false-inhibit rate)"
echo
echo "Read ALGORITHM_SUMMARY.md → 'SQI flip-on gate' before enabling the ensemble"
echo "in the frozen decide() path or running full beat validation."

if [[ "$RUN_BEAT_AND_NP" != "1" ]]; then
  echo "Skipping beat validation + NP (set RUN_BEAT_AND_NP=1 after SQI looks good)."
  exit 0
fi

BEAT_DIR="$OUT_ROOT/beat_validation"
NP_DIR="$OUT_ROOT/np_operating_point"
mkdir -p "$BEAT_DIR" "$NP_DIR"

echo "[2/3] Beat-sync validation on MIT-BIH gold (frozen gate) -> $BEAT_DIR"
$PY Layer2/validation/run_beat_validation.py \
  --data-dir "$DATA_DIR" \
  --datasets mit_bih_arrhythmia \
  --records-csv "$GOLD_CSV" \
  --out-dir "$BEAT_DIR" \
  --threshold-method conformal --conformal-alpha 0.10 \
  --anomaly-model mahalanobis --guard-s 5.0

echo "[3/3] Neyman-Pearson operating point (danger budget=$DANGER_BUDGET) -> $NP_DIR"
$PY Layer2/validation/run_np_operating_point.py \
  --per-beat "$BEAT_DIR/per_beat.csv" \
  --out-dir "$NP_DIR" \
  --danger-budget "$DANGER_BUDGET"

echo
echo "[OK] Beat + NP outputs:"
echo "  $NP_DIR/np_operating_point.csv"
echo "  $NP_DIR/worst_record_danger.csv"
echo "  $NP_DIR/np_frontier.csv"
