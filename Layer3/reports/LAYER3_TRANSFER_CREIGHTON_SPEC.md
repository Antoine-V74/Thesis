# Layer 3 — Tier 3A: cross-setup transfer (MIT-BIH → Creighton VF)

**The experiment the SSL / Arm-C arms were justified "for translation" but never ran.**
Every Layer 3 eval so far was in-domain MIT-BIH. This one freezes the MIT-BIH-pretrained
encoders and evaluates them on **Creighton VF (cudb)** — a cohort never seen in
pretraining and never inspected/tuned — re-fitting **only the per-record healthy
baseline**. No retraining.

Date: July 2026. Script: `cluster_jobs/19_transfer_creighton_eval.sh`.
Parent memo: plan file (Tier 1/Tier 3). Ladder: `LAYER3_ARM_C_LADDER_SPEC.md`.

---

## 0. Why this is the highest-value, lowest-cost next experiment

1. **It tests the one thing that must transfer for deployment.** The deploy contract —
   freeze encoder, re-fit a per-record healthy baseline, conformal threshold — is
   *designed* to survive domain shift. That is ZEROSHOT's genuinely transferable result
   (HUP → SWEC, no fine-tuning). We have asserted "for translation" in the docs but never
   demonstrated it.
2. **It bridges to rat — the actual thesis deliverable.** If the contract survives a
   human→human shift (MIT-BIH → Creighton VF), that is evidence it may survive human→rat.
   If it breaks even human→human, that is a crucial, honest deployment limit to report.
3. **It is nearly free.** Checkpoints, harness, Creighton data, and the gold list
   (`pilot_lists/pilot_secondary_creighton_gold.csv`, 33 records / 108 danger beats) all
   already exist. Script `19` only re-points `--datasets`/`--records-csv`; every other flag
   is identical to the MIT-BIH Phase 1 (`07b`), so any delta is transfer, not setup.
4. **Creighton is the never-inspected cohort**, so these numbers double as the
   **untouched-cohort confirmation** we promised for the C ladder.

---

## 1. The A0 control makes the result interpretable (do not skip it)

`--phase1-arms a0,layer3` scores A0 (handcrafted L2 features) alongside the encoder in the
same run, for free. A0 is the confound separator:

| MIT-BIH → Creighton | A0 | Encoder (L3) | Interpretation |
| --- | --- | --- | --- |
| both hold | ~stable | ~stable | **Deploy contract transfers** → evidence toward rat |
| A0 degrades too | worse | worse | Gap is **danger modality** (Creighton is VF-heavy), not a broken contract |
| A0 holds, L3 breaks | ~stable | worse | The **learned representation** does not transfer (handcrafted does) |

Without A0 on the *same* Creighton cohort, a "transfer failed" headline is unearned — you
cannot tell a broken contract from a different danger distribution.

---

## 2. Design (only the eval cohort changes)

| Setting | MIT-BIH Phase 1 (07b/14b/15b/16b) | Transfer (this) |
| --- | --- | --- |
| Encoder checkpoint | MIT-BIH-pretrained, gold-excluded | **same** (frozen, no retrain) |
| `--datasets` | `mit_bih_arrhythmia` | **`creighton_vfib`** |
| `--records-csv` | `pilot_primary_mitbih_gold.csv` | **`pilot_secondary_creighton_gold.csv`** |
| Calibration | per-record healthy, conformal α=0.10 | **same** (re-fit on Creighton records) |
| Scorer / window / mode / everything else | mahalanobis,knn / 8 s @ 125 Hz / oracle | **identical** |

Arms evaluated: **C (supcon)** primary + **C1/C2/C3** (the ladder, as its confirmation-
cohort check). A0 comes free with each.

---

## 3. Commands

```bash
# On the cluster (GPU). Needs the four MIT-BIH checkpoints from 07a/14a/15a/16a present.
bash Layer3/reports/cluster_jobs/19_transfer_creighton_eval.sh
# or sbatch it with the 13/18-style wrapper (mit_normal_gpu, conda ecg).
```

Outputs per arm: `Results/layer3/validation/transfer_creighton_<arm>_seed0_100ep_8s/`
(`phase1_metrics_bootstrap.csv`, `phase1_metrics_overall.csv`, `phase1_cav_l2_l3.csv`,
`encoder_info.json`).

---

## 4. The read (fill after runs)

For each arm, put its **MIT-BIH** false-permit next to its **Creighton** false-permit
(mahalanobis primary; note kNN if it disagrees). A0 row first — it decides the interpretation.

| Arm | MIT-BIH FP (87 danger) | Creighton FP (108 danger) | Δ transfer | Read |
| --- | --- | --- | --- | --- |
| A0 | 14.9% | _fill_ | _fill_ | control — sets the baseline for "did the contract transfer?" |
| C `supcon` | 36.8% | _fill_ | _fill_ | |
| C1 `supcon_oe` | 31.0% | _fill_ | _fill_ | |
| C2 `deepsad` | 43.7% | _fill_ | _fill_ | |
| C3 `supcon_hybrid` | 33.3% | _fill_ | _fill_ | |

(MIT-BIH numbers are the mahalanobis results already banked from `07b/14b/15b/16b`.)

---

## 5. Papers

- **Yu, 2026 — ZEROSHOT** (repo PDF). Cross-center transfer (HUP→SWEC), frozen encoder +
  per-subject Mahalanobis, no fine-tuning — the exact template.
- **Ballas & Diou, 2023 — "Towards Domain Generalization for ECG and EEG"** (arXiv 2303.11338);
  and 2022 IEEE BigDataService "DG for OOD 12-lead ECG." DG methods/benchmarks for ECG.
- **Ruiz-Barroso et al., 2025 — FADE** (arXiv 2502.07389; repo PDF). Normal-ECG forecasting AD
  with explicit domain adaptation to new patients/sensors — MIT-BIH.
- **Carrera et al., 2019** (Pattern Recognition; repo PDF). Per-user normal model + online HR
  adaptation — the non-stationary-baseline precedent that matters most for rat.

---

## 6. Guardrail / caveats
- Creighton is **VF-oriented** → this answers "acute-VF transfer," not general domain shift.
  Report it as a robustness/transfer check, not a universal transfer claim.
- Human→human transfer is **evidence toward**, not proof of, human→rat transfer.
- This is exploratory; A0 remains the deployable baseline; report false-permit + CAV.
- Do **not** tune anything based on Creighton results — it is the confirmation cohort; tuning
  on it would burn its one job.
