#!/usr/bin/env bash
# Tier 3A - Cross-setup transfer eval. MIT-BIH-pretrained encoders -> Creighton VF
# (cudb), an UNTOUCHED cohort. NO retraining: the encoder is frozen and only the
# per-record healthy baseline is re-fit on Creighton records. This tests whether the
# label-free deploy contract survives a human->human domain shift - the experiment the
# SSL/Arm-C arms were justified "for translation" but never actually ran.
#
# Only the eval cohort changes vs the MIT-BIH Phase 1 (07b/14b/15b/16b): --datasets and
# --records-csv. Everything else is identical, so any delta is transfer, not setup.
#
# A0 CONTROL (comes free via --phase1-arms a0,layer3): if A0 handcrafted features ALSO
# degrade MIT-BIH->Creighton, the gap is danger MODALITY (Creighton is VF-heavy), not a
# broken contract. If A0 holds but the encoder breaks, the REPRESENTATION does not transfer.
#
# Creighton is the never-inspected cohort, so these numbers double as the promised
# untouched-cohort confirmation for the C ladder. Caveat: VF-oriented -> this answers
# "acute-VF transfer", not general domain shift.
set -euo pipefail
cd "$(dirname "$0")/../../.."

DATA_DIR="${DATA_DIR:-data}"
CREIGHTON_CSV="${CREIGHTON_CSV:-Layer3/reports/pilot_lists/pilot_secondary_creighton_gold.csv}"

# Arm name -> pretrain checkpoint dir follows the convention <arm>_mitbih_seed0_100ep_8s_goldexcluded.
for ARM in supcon supcon_oe deepsad supcon_hybrid; do
  CKPT="Results/layer3/pretrain/${ARM}_mitbih_seed0_100ep_8s_goldexcluded/encoder_last.pt"
  OUT="Results/layer3/validation/transfer_creighton_${ARM}_seed0_100ep_8s"
  if [[ ! -f "$CKPT" ]]; then
    echo "[SKIP] $ARM: checkpoint not found ($CKPT) - run its MIT-BIH pretrain first." >&2
    continue
  fi
  echo "=== Transfer eval: ${ARM} (MIT-BIH-pretrained) -> Creighton VF ==="
  mkdir -p "$OUT"
  python Layer3/validation/run_beat_validation.py \
    --data-dir "$DATA_DIR" \
    --datasets creighton_vfib \
    --records-csv "$CREIGHTON_CSV" \
    --checkpoint "$CKPT" \
    --out-dir "$OUT" \
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
done

echo ""
echo "[OK] Transfer eval complete."
echo "A0-on-Creighton is identical across the four runs (read it once, any out-dir)."
echo "For each arm compare Creighton false-permit vs its MIT-BIH false-permit:"
echo "  - A0 degrades too      -> gap is danger modality (VF), NOT a broken deploy contract."
echo "  - A0 holds, L3 breaks  -> the learned representation does not transfer."
echo "  - both hold            -> the personalized veto contract transfers (evidence toward rat)."
