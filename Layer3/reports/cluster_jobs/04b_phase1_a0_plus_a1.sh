#!/usr/bin/env bash
# Job 4b — Phase 1 A0 + Arm A1. Requires: 04a finished.
set -euo pipefail
cd "$(dirname "$0")/../../.."

DATA_DIR="${DATA_DIR:-data}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
CKPT="${CKPT:-Results/layer3/pretrain/vicreg_mitbih_seed0/encoder_last.pt}"
OUT_DIR="${OUT_DIR:-Results/layer3/validation/pilot_mitbih_vicreg_seed0_8s}"

if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] Checkpoint not found: $CKPT — run 04a_pretrain_arm_a1_vicreg.sh first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python Layer3/validation/run_beat_validation.py \
  --data-dir "$DATA_DIR" \
  --datasets mit_bih_arrhythmia \
  --records-csv "$GOLD_CSV" \
  --checkpoint "$CKPT" \
  --out-dir "$OUT_DIR" \
  --mode oracle \
  --window-s 8 --target-fs 125 \
  --causal-window --lookahead-ms 100 \
  --per-record-calibration --guard-s 8 \
  --l2-normalize-embeddings --pca-dim 32 \
  --phase1-eval --phase1-arms a0,layer3 \
  --phase1-scorers mahalanobis,knn \
  --threshold-method conformal --conformal-alpha 0.10 \
  --no-random-fallback \
  --device "${DEVICE:-cuda}"

echo "[OK] Phase 1 A0+A1 outputs -> $OUT_DIR"
echo "Verify: encoder_info.json checkpoint_loaded=true"
