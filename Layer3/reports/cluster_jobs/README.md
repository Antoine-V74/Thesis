# Cluster jobs — MIT-BIH gold pilot

Ready-to-run bash scripts. Wave 1 = L2 / A0 / A. Wave 2 = A1 / B / B1 after Wave 1 looks sane.

## Wave 1 (run first)

| Script | Role |
| --- | --- |
| `00_run_all_sequential.sh` | Run 01→02→03a→03b→03c in order |
| `01_layer2_mitbih_gold.sh` | Full Layer 2 (oracle + cadence) |
| `02_phase1_a0_only.sh` | Phase 1 A0-only |
| `03a_build_window_index_mitbih.sh` | 8 s @ 125 Hz window index + signal cache |
| `03b_pretrain_arm_a_ntxent.sh` | Arm A NT-Xent pretrain (gold excluded) |
| `03c_phase1_a0_plus_a.sh` | Phase 1 A0+A + CAV |

## Wave 2 (after Wave 1 inspect)

| Script | Role |
| --- | --- |
| `04a_pretrain_arm_a1_vicreg.sh` | Arm A1 VICReg pretrain |
| `04b_phase1_a0_plus_a1.sh` | Phase 1 A0+A1 |
| `05a_pretrain_arm_b_mae_consistency.sh` | Arm B primary masked (`--healthy-only`) |
| `05b_phase1_a0_plus_b.sh` | Phase 1 A0+B |
| `06a_pretrain_arm_b1_subject_contrastive.sh` | Arm B1 ablation (`--healthy-only`) |
| `06b_phase1_a0_plus_b1.sh` | Phase 1 A0+B1 |

## Wave 3 (supervised representation — needs code first)

| Script | Role |
| --- | --- |
| `07a_pretrain_arm_c_supcon.sh` | Arm C supervised contrastive (labels at pretrain only) |
| `07b_phase1_a0_plus_c.sh` | Phase 1 A0+C |

Arm C primary (`supcon`) is implemented — see
[`../LAYER3_ARM_C_SUPERVISED_SPEC.md`](../LAYER3_ARM_C_SUPERVISED_SPEC.md).
C-ce / C-oe remain deferred.

Reuse the Wave-1 window index from `03a` for all later pretrains.

**Order:** `01∥02 → 03a → 03b → 03c → inspect → 04* → 05* → 06*`  
(A1 / B / B1 pretrains can run in parallel after `03a` if you accept the resource cost; still evaluate only after Wave-1 go.)

Framing + results table: [`../LAYER3_L2_A0_A_COMPARE.md`](../LAYER3_L2_A0_A_COMPARE.md)  
Offline while GPU blocked: [`../LAYER3_CONSOLIDATED_SUMMARY.md`](../LAYER3_CONSOLIDATED_SUMMARY.md) §8b  
Arm depth: A0 / A / B / B1 complete-status docs in `../`

Environment overrides (optional): `DATA_DIR`, `GOLD_CSV`, `OUT_DIR`, `CKPT_DIR`,
`CKPT`, `WINDOW_INDEX`, `DEVICE`, `EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `SEED`.

## Post-hoc exploratory improvement track

The locked Wave 1/2 results remain the primary result. Scripts `07`–`12` are
post-hoc diagnostics/ablations and must not be presented as preregistered.

```text
07a  diagnose frozen representations
07b  frozen B1 PCA/normalization/covariance sweep
08a  healthy-only NT-Xent + mild augmentations
08b  healthy-only VICReg + mild augmentations
09a  healthy-only B with mask=0.50, consistency=0.50
10a  healthy VICReg with local-sensitive avg+max pooling
12   evaluate whichever 8 s exploratory checkpoints exist
11a → 11b → 11c  optional 30 s rhythm-context track
13   submit the complete exploratory track with Slurm dependencies
07a → 07b  Arm C (SupCon supervised ceiling): pretrain → Phase 1 A0+C
14a → 14b  Arm C1 (supcon_oe): SupCon + outlier exposure
15a → 15b  Arm C2 (deepsad): SVDD compact normal + outlier exposure
16a → 16b  Arm C3 (supcon_hybrid): SupCon + SVDD + outlier exposure
```

Arm C ladder (07 / 14 / 15 / 16) spec: [`../LAYER3_ARM_C_LADDER_SPEC.md`](../LAYER3_ARM_C_LADDER_SPEC.md).
Env overrides for the ladder sweep: `OE_WEIGHT`, `SVDD_WEIGHT`, `SUPCON_TEMPERATURE`,
`CENTER_INIT_WINDOWS` (in addition to the common ones above).

Keep conformal `alpha=0.10`, gold exclusion, and `--no-random-fallback`.
Choose a candidate on a separate development cohort where possible; because
the MIT-BIH gold outcomes have already been inspected, improvements on those
13 records are exploratory until confirmed on an untouched cohort.
