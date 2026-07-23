#!/usr/bin/env bash
# Master runner — L2 + A0 in parallel-friendly order, then A pretrain + Phase 1.
# On a Slurm site, submit each numbered script as its own job with dependencies:
#   01 and 02 can run in parallel
#   03a -> 03b -> 03c
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
JOBS="$(cd "$(dirname "$0")" && pwd)"

echo "=== 01 Layer 2 (full gate) ==="
bash "$JOBS/01_layer2_mitbih_gold.sh"

echo "=== 02 Phase 1 A0-only ==="
bash "$JOBS/02_phase1_a0_only.sh"

echo "=== 03a window index ==="
bash "$JOBS/03a_build_window_index_mitbih.sh"

echo "=== 03b pretrain Arm A ==="
bash "$JOBS/03b_pretrain_arm_a_ntxent.sh"

echo "=== 03c Phase 1 A0+A ==="
bash "$JOBS/03c_phase1_a0_plus_a.sh"

echo "=== All jobs finished. Fill LAYER3_L2_A0_A_COMPARE.md comparison table. ==="
