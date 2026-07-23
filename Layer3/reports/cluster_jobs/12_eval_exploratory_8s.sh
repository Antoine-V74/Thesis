#!/usr/bin/env bash
# Evaluate completed exploratory 8 s checkpoints under the locked Phase 1 scorer.
set -euo pipefail
cd "$(dirname "$0")/../../.."

evaluate() {
  local name="$1" ckpt="$2" pooling="$3"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] Missing $ckpt"
    return
  fi
  python Layer3/validation/run_beat_validation.py \
    --data-dir data --datasets mit_bih_arrhythmia \
    --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
    --checkpoint "$ckpt" --out-dir "Results/layer3/validation/${name}" \
    --mode oracle --window-s 8 --target-fs 125 --causal-window --lookahead-ms 100 \
    --per-record-calibration --guard-s 8 --encoder-pooling-mode "$pooling" \
    --l2-normalize-embeddings --pca-dim 32 \
    --phase1-eval --phase1-arms a0,layer3 --phase1-scorers mahalanobis,knn \
    --threshold-method conformal --conformal-alpha 0.10 \
    --no-random-fallback --device "${DEVICE:-cuda}"
}

evaluate exploratory_ntxent_healthy_mild_seed0_8s \
  Results/layer3/pretrain/exploratory_ntxent_healthy_mild_seed0/encoder_last.pt global_avg
evaluate exploratory_vicreg_healthy_mild_seed0_8s \
  Results/layer3/pretrain/exploratory_vicreg_healthy_mild_seed0/encoder_last.pt global_avg
evaluate exploratory_mae_consistency_m50_c05_seed0_8s \
  Results/layer3/pretrain/exploratory_mae_consistency_m50_c05_seed0/encoder_last.pt global_avg
evaluate exploratory_vicreg_healthy_avgmax_seed0_8s \
  Results/layer3/pretrain/exploratory_vicreg_healthy_avgmax_seed0/encoder_last.pt avg_max

