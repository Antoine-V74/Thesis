#!/usr/bin/env bash
# Job 15b - Phase 1 A0 + Arm C2 (deepsad). Requires: 15a finished (encoder_last.pt).
# Deployment scorer is identical to every other arm (label-free calibration).
set -euo pipefail
cd "$(dirname "$0")/../../.."

DATA_DIR="${DATA_DIR:-data}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
CKPT="${CKPT:-Results/layer3/pretrain/deepsad_mitbih_seed0_100ep_8s_goldexcluded/encoder_last.pt}"
OUT_DIR="${OUT_DIR:-Results/layer3/validation/pilot_mitbih_deepsad_seed0_100ep_8s}"

if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] Checkpoint not found: $CKPT - run 15a_pretrain_arm_c2_deepsad.sh first." >&2
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

echo "[OK] Phase 1 A0+C2 outputs -> $OUT_DIR"
echo "Headline: false-permit DANGEROUS + bootstrap CI + CAV vs A0. Compare against A0, C, C1, and the SSL arms."
