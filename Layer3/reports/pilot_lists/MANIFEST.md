# Layer 3 — Danger inventory & pilot Go decision

**Date:** 2026-07-16  
**Status:** **SOFT GO** — one-seed pilot on **MIT-BIH first** (not full multi-seed campaign yet)  
**Script:** `Layer3/tools/count_transition_records.py`  
**CSVs:** `transition_records_by_record.csv`, `transition_summary_by_dataset.csv`, `gold_transition_records_strict_ge30.csv`

---

## What we are doing (for someone new to the project)

We build an **inhibit-only ECG safety veto** for stimulation-gated cardiac assist:

```text
permit only if Layer1 AND Layer2 AND (Layer3 if enabled) are safe
uncertainty → inhibit
```

**Layer 3** (human PhysioNet research only) asks: under a **fixed** personalized Mahalanobis/kNN + conformal scorer, do SSL embeddings beat Layer 2 **features** (arm A0)?  
Thresholds use **healthy** ECG only; DANGEROUS labels are for offline scoring of **false permits**.

This inventory checks whether enough **within-record** healthy→danger transitions exist before spending GPU time.

**Gold record (strict):** ≥30 sinus beats before the first DANGEROUS event in that record.

---

## Inventory results


| Dataset      | Gold records (≥30 baseline) | Danger beats | Role                                                |
| ------------ | --------------------------- | ------------ | --------------------------------------------------- |
| **MIT-BIH**  | **13**                      | **654**      | **PRIMARY** — acute transitions, cleanest labels    |
| LTAFDB       | 28                          | 5603         | **SECONDARY** — AF-heavy; 88% of all danger beats   |
| Creighton VF | 33                          | 108          | **Robustness** — VF-oriented; small beat count      |
| Malignant VA | 0 beat-eligible             | —            | Rhythm-span only; not for beat-level primary tables |
| **Total**    | **74**                      | **6365**     | LTAFDB dominates the pooled total                   |


---

## How we read C1 (power / danger events)


| Claim                            | Assessment                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------- |
| Empty danger set?                | **No** — soft GO is justified                                                         |
| Worst-case “15–40 records only”? | **Less severe** — 74 gold records overall                                             |
| Main remaining risk              | **Skew:** most beats are LTAFDB (AF context), not acute VT/VF                         |
| MIT-BIH alone                    | Modest *n* (13 records) — OK for a **directional** pilot, not for tiny (<5 pp) claims |


**Implication (state explicitly in thesis/results):**  
A pooled “all datasets” false-permit number would mostly answer *AF / AF-variability*, not *acute ventricular catastrophe*. Primary narrative = **MIT-BIH**; LTAFDB = secondary / robustness with caveats; Creighton = VF check.

Rough power (MIT-BIH, directional only): record-level uncertainty on the order of several pp; MDD for confident A0-vs-SSL gaps is **large** (~10 pp class). Small gaps may look like noise — that is acceptable if we say so.

---

## Decision


|             |                                                                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Now**     | One-seed **NT-Xent** pretrain + Phase 1 (`a0` vs that checkpoint) on **MIT-BIH**, 8 s @ 125 Hz, conformal α=0.10, `--no-random-fallback` |
| **Next**    | If pilot looks sane → multi-seed A / A1 / B on same MIT-BIH setup                                                                        |
| **Then**    | Creighton as generalization / VF check; LTAFDB secondary with AF caveat                                                                  |
| **Not yet** | Full multi-dataset “headline” campaign treating LTAFDB as primary                                                                        |


**Pilot pass criteria (engineering, not cherry-picking SSL):**  
(a) checkpoint loads; calibration not degenerate on most records;  
(b) tables write;  
(c) directionally inspect A0 vs A false-permit (no retuning thresholds on danger).

Optional cheap checks (do not block start): note danger mix on MIT-BIH when tables exist; after pilot, A0 vs L3 score correlation on healthy beats if easy.

---

## Locked primary metric (pre-results)

> **Primary:** false-permit rate on **MIT-BIH** DANGEROUS beats/windows, at conformal α=0.10 healthy budget, with **per-record** rates and (when available) record-bootstrap CI.  
> **Secondary:** healthy permit / false-inhibit; other datasets with explicit role labels above.  
> **Not primary:** AUROC alone; pooled all-datasets FP dominated by LTAFDB.

Protocol commands: `Layer3/reports/ZEROSHOT_CLUSTER_RUN_NOTES.md`  
Overview: `Layer3/reports/LAYER3_SUPERVISOR_SUMMARY.md`

---

## One-line stand-up

> Inventory: 74 gold records / 6365 danger beats (88% LTAFDB AF-context). MIT-BIH (13 / 654) is modest but clean → **SOFT GO** one-seed NT-Xent on MIT-BIH; multi-seed and other datasets after pilot; do not headline LTAFDB as acute-arrhythmia proof.

---

## Code support for the pilot


| File                                       | Use                                 |
| ------------------------------------------ | ----------------------------------- |
| `pilot_primary_mitbih_gold.csv`            | 13 MIT-BIH gold records (allowlist) |
| `run_beat_validation.py --records-csv ...` | Score only those records            |
| `ZEROSHOT_CLUSTER_RUN_NOTES.md` §0b        | Copy-paste pilot commands           |


