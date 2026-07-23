#!/usr/bin/env bash
# Master runner - Arm C ladder (C1 -> C2 -> C3), pretrain + Phase 1 in order.
# EXPLORATORY improvements on Arm C. On a Slurm site, submit each numbered script as
# its own GPU job with dependencies instead of running here on a login node.
# GUARDRAIL: exploratory on the already-inspected 13 gold records; confirm on an
# untouched cohort before any claim. A0 remains the deployable baseline.
set -euo pipefail
JOBS="$(cd "$(dirname "$0")" && pwd)"

echo "=== 14a Pretrain C1 (supcon_oe) ==="
bash "$JOBS/14a_pretrain_arm_c1_supcon_oe.sh"
echo "=== 14b Phase 1 A0+C1 ==="
bash "$JOBS/14b_phase1_a0_plus_c1.sh"

echo "=== 15a Pretrain C2 (deepsad) ==="
bash "$JOBS/15a_pretrain_arm_c2_deepsad.sh"
echo "=== 15b Phase 1 A0+C2 ==="
bash "$JOBS/15b_phase1_a0_plus_c2.sh"

echo "=== 16a Pretrain C3 (supcon_hybrid) ==="
bash "$JOBS/16a_pretrain_arm_c3_hybrid.sh"
echo "=== 16b Phase 1 A0+C3 ==="
bash "$JOBS/16b_phase1_a0_plus_c3.sh"

echo "=== Arm C ladder done. Fill the A0/C/C1/C2/C3 block in LAYER3_L2_A0_A_COMPARE.md. ==="
echo "Report false-permit DANGEROUS + bootstrap CI + CAV vs A0 - not accuracy."
