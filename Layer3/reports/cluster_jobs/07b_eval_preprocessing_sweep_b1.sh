#!/usr/bin/env bash
# Exploratory frozen-checkpoint scorer/preprocessing sweep on B1.
# Locked Phase 1 remains PCA32 + L2 + Ledoit-Wolf at conformal alpha=0.10.
set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT="${CKPT:-Results/layer3/pretrain/mae_subject_contrastive_mitbih_seed0/encoder_last.pt}"
GOLD_CSV="${GOLD_CSV:-Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv}"
DEVICE="${DEVICE:-cuda}"

run_eval() {
  local name="$1"
  local pca="$2"
  local norm="$3"
  local covariance="$4"
  local norm_flag=()
  if [[ "$norm" == "l2" ]]; then norm_flag=(--l2-normalize-embeddings); fi
  python Layer3/validation/run_beat_validation.py \
    --data-dir data --datasets mit_bih_arrhythmia \
    --records-csv "$GOLD_CSV" --checkpoint "$CKPT" \
    --out-dir "Results/layer3/validation/exploratory_b1_${name}_seed0_8s" \
    --mode oracle --window-s 8 --target-fs 125 --causal-window --lookahead-ms 100 \
    --per-record-calibration --guard-s 8 \
    "${norm_flag[@]}" --pca-dim "$pca" --covariance-estimator "$covariance" \
    --phase1-eval --phase1-arms a0,layer3 --phase1-scorers mahalanobis,knn \
    --threshold-method conformal --conformal-alpha 0.10 \
    --no-random-fallback --device "$DEVICE"
}

run_eval pca0_l2_lw 0 l2 ledoit_wolf
run_eval pca16_l2_oas 16 l2 oas
run_eval pca32_raw_lw 32 raw ledoit_wolf
run_eval pca64_l2_diag 64 l2 diagonal

