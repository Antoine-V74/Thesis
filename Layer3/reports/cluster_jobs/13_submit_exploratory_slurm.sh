#!/usr/bin/env bash
# Submit the post-hoc exploratory track with Slurm dependencies.
# Site-specific wrapper for MIT ORCD; science settings live in scripts 07-12.
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

# Cheap frozen-checkpoint diagnostic/sweep.
b1_sweep="$(submit_gpu l3_b1_sweep 07b_eval_preprocessing_sweep_b1.sh 02:00:00)"

# Four independent 8 s representation experiments.
a_h="$(submit_gpu l3_a_h 08a_pretrain_ntxent_healthy_mild.sh 06:00:00)"
a1_h="$(submit_gpu l3_a1_h 08b_pretrain_vicreg_healthy_mild.sh 06:00:00)"
b_tuned="$(submit_gpu l3_b_tuned 09a_pretrain_mae_consistency_tuned.sh 06:00:00)"
avgmax="$(submit_gpu l3_avgmax 10a_pretrain_vicreg_avgmax.sh 06:00:00)"
eval_dep="${a_h}:${a1_h}:${b_tuned}:${avgmax}"
eval_8s="$(submit_gpu l3_eval8 12_eval_exploratory_8s.sh 03:00:00 "$eval_dep")"

# Longer-context branch: build → pretrain → evaluate.
idx30="$(submit_gpu l3_idx30 11a_build_window_index_mitbih_30s.sh 02:00:00)"
train30="$(submit_gpu l3_train30 11b_pretrain_vicreg_healthy_30s_avgmax.sh 06:00:00 "$idx30")"
eval30="$(submit_gpu l3_eval30 11c_phase1_vicreg_30s_avgmax.sh 03:00:00 "$train30")"

printf 'submitted b1_sweep=%s eval_8s=%s eval_30s=%s\n' "$b1_sweep" "$eval_8s" "$eval30"
printf 'pretrains: A_h=%s A1_h=%s B_tuned=%s avgmax=%s\n' "$a_h" "$a1_h" "$b_tuned" "$avgmax"
squeue -u "$USER"

