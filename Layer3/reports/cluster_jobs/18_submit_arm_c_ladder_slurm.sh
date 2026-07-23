#!/usr/bin/env bash
# Submit Arm C (SupCon) + the C1/C2/C3 ladder with Slurm dependencies.
# Site-specific wrapper for MIT ORCD (mirrors 13_submit_exploratory_slurm.sh).
# EXPLORATORY beyond the primary C (SupCon) run: C1/C2/C3 are a bounded,
# pre-registered, single-seed ladder - no sweep. See LAYER3_ARM_C_LADDER_SPEC.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
JOBS="$ROOT/Layer3/reports/cluster_jobs"
cd "$ROOT"
mkdir -p logs

submit_gpu() {
  local name="$1" script="$2" time="$3" dependency="${4:-}"
  local dep_args=()
  if [[ -n "$dependency" ]]; then dep_args=(--dependency="afterok:${dependency}"); fi
  sbatch --parsable \
    -p mit_normal_gpu -N 1 -c 4 -G 1 --mem=32GB -t "$time" \
    -J "$name" -o "logs/${name}_%j.out" -e "logs/${name}_%j.err" \
    "${dep_args[@]}" \
    --wrap="module load miniforge/24.3.0-0; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate ecg; cd \"$ROOT\"; bash \"$JOBS/$script\""
}

# Arm C (SupCon) - the supervised ceiling - and the C1/C2/C3 ladder are four
# INDEPENDENT pretrains (each only reads the shared window index + gold CSV),
# so submit them in parallel like the existing 08a/08b/09a/10a pattern - no
# reason to serialize and waste wall-clock. Each *_eval waits only on its own
# *_pre via --dependency=afterok.
c_pre="$(submit_gpu l3_c_pre 07a_pretrain_arm_c_supcon.sh 06:00:00)"
c_eval="$(submit_gpu l3_c_eval 07b_phase1_a0_plus_c.sh 03:00:00 "$c_pre")"

c1_pre="$(submit_gpu l3_c1_pre 14a_pretrain_arm_c1_supcon_oe.sh 06:00:00)"
c1_eval="$(submit_gpu l3_c1_eval 14b_phase1_a0_plus_c1.sh 03:00:00 "$c1_pre")"

c2_pre="$(submit_gpu l3_c2_pre 15a_pretrain_arm_c2_deepsad.sh 06:00:00)"
c2_eval="$(submit_gpu l3_c2_eval 15b_phase1_a0_plus_c2.sh 03:00:00 "$c2_pre")"

c3_pre="$(submit_gpu l3_c3_pre 16a_pretrain_arm_c3_hybrid.sh 06:00:00)"
c3_eval="$(submit_gpu l3_c3_eval 16b_phase1_a0_plus_c3.sh 03:00:00 "$c3_pre")"

printf 'C:  pretrain=%s eval=%s\n' "$c_pre" "$c_eval"
printf 'C1: pretrain=%s eval=%s\n' "$c1_pre" "$c1_eval"
printf 'C2: pretrain=%s eval=%s\n' "$c2_pre" "$c2_eval"
printf 'C3: pretrain=%s eval=%s\n' "$c3_pre" "$c3_eval"
squeue -u "$USER"

echo ""
echo "NOTE: this is a bounded, single-seed, no-sweep run of C + C1 + C2 + C3."
echo "Report false-permit + CAV for all four vs A0. Do not tune weights based on"
echo "these results and re-run - that reopens the multiplicity problem. If none"
echo "beat A0 / add CAV, bank the negative and move to L2 operating point + transfer."
