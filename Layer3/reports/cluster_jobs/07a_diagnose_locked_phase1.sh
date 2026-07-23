#!/usr/bin/env bash
# Post-hoc diagnostics only. Does not change thresholds or locked primary results.
set -euo pipefail
cd "$(dirname "$0")/../../.."

python Layer3/tools/diagnose_phase1_embeddings.py \
  Results/layer3/validation/pilot_mitbih_ntxent_seed0_8s \
  Results/layer3/validation/pilot_mitbih_vicreg_seed0_8s \
  Results/layer3/validation/pilot_mitbih_mae_consistency_seed0_8s \
  Results/layer3/validation/pilot_mitbih_mae_subject_contrastive_seed0_8s \
  --out-dir Results/layer3/diagnostics/locked_seed0

