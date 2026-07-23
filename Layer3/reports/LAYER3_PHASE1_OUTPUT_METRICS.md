# Phase 1 / Layer 2 — what outputs mean

**Audience:** you, before/after a cluster run.  
**Scope:** Layer 3 Phase 1 CSVs (`phase1_*`) and the main Layer 2 beat-validation CSVs.  
**Not** a protocol redesign doc — see end for how this relates to ±0.5 s and 1-in-8.

---

## 0. Two different experiments (do not mix folders)

| Experiment | Script | Question |
| --- | --- | --- |
| **Layer 3 Phase 1** | `Layer3/validation/run_beat_validation.py --phase1-eval` | Under a **fixed** personalized scorer, do SSL embeddings beat A0 (L2 features)? |
| **Layer 2 validation** | `Layer2/validation/run_beat_validation.py` | How good is the **real** Layer 2 gate (features + hard rules + optional **1-in-8 cadence**)? |

Phase 1 is **not** full Layer 2. A0 inside Phase 1 = L2 **features only** + Mahalanobis/kNN (no hard-rule gate, no cadence).

You **can** run both on the cluster. They answer different questions.

---

## 1. Layer 3 Phase 1 — files and columns

Produced when you pass `--phase1-eval`. Written to `--out-dir`.

### 1.1 Files you should open first

| Priority | File | What it answers |
| --- | --- | --- |
| **1** | `phase1_metrics_bootstrap.csv` | Headline false-permit ± **record-cluster** 95% CI |
| **2** | `phase1_metrics_overall.csv` | Pooled false-permit / false-inhibit / AUROC (Wilson CI = transparency only) |
| **3** | `phase1_metrics_by_record.csv` | Is danger concentrated in 1–2 records? |
| **4** | `phase1_metrics_by_danger_subtype.csv` | Rhythm vs morphology vs noise false permits |
| **5** | `phase1_cav_l2_l3.csv` | Does L3 catch A0 mistakes? (needs `--phase1-arms a0,layer3`) |
| **6** | `phase1_thresholds.csv` | Per-record calibration health / conformal status |
| — | `phase1_per_beat.csv` | Raw beat-level decisions (for debugging / plots) |
| — | `phase1_aurocs.csv` | AUROC subset of metrics |
| — | `phase1_offline_operating_points.csv` | **Non-deployable** danger-targeted threshold (uses labels) |
| — | `encoder_info.json` | Must show `checkpoint_loaded: true` for SSL arms |

**Ignore for A0 claims:** `per_beat.csv`, `embeddings.npy`, `metrics_overall.csv` from the same folder — those are the **encoder** path, not A0 Phase 1.

### 1.2 `phase1_metrics_overall.csv` / `phase1_metrics_by_dataset.csv`

One row per `(arm, scorer, threshold_method)` [and dataset].

| Column | Meaning |
| --- | --- |
| `arm` | `a0_layer2_features` or `layer3` |
| `scorer` | `mahalanobis` or `knn` |
| `threshold_method` | `conformal`, `healthy_quantile`, or `danger_2pct_offline` |
| `n` | # scored test beats in this slice |
| `n_NORMAL` / `n_DANGEROUS` | Label counts |
| `false_permit_DANGEROUS` | **Primary safety rate:** permit on DANGEROUS |
| `false_permit_DANGEROUS_n` | Count of those errors |
| `false_permit_DANGEROUS_ci_low/high` | Wilson beat CI (too optimistic — secondary) |
| `false_inhibit_NORMAL` | Inhibit on healthy (availability cost) |
| `false_inhibit_NORMAL_ci_*` | Wilson CI on that |
| `inhibit_rate_BENIGN_ABNORMAL` / `AF_CONTEXT` / `NOISE` | How often we inhibit those groups |
| `auroc_NORMAL_vs_DANGEROUS` | Score ranking quality (not the operating point) |
| `auroc_NORMAL_vs_all_abnormal` | Same vs all non-NORMAL |
| `conformal_alpha_infeasible_n` | How often conformal couldn’t be set → fail-safe inhibit |

### 1.3 `phase1_metrics_bootstrap.csv` (headline uncertainty)

| Column | Meaning |
| --- | --- |
| `false_permit_DANGEROUS` | Same point estimate as overall |
| `boot_ci_low` / `boot_ci_high` | **95% record-cluster bootstrap CI** — use this in the thesis |
| `n_records` / `n_DANGEROUS` | Effective sample size at record vs beat level |
| `per_record_false_permit_mean/median` | Spread across records |
| `ci_method` | `record_cluster_bootstrap` |

**CI is not a performance metric.** It is “how wide is the uncertainty on the false-permit rate.”

### 1.4 `phase1_metrics_by_record.csv`

| Column | Meaning |
| --- | --- |
| `record_key` | e.g. `mit_bih_arrhythmia/200` |
| `n_DANGEROUS` | Danger beats in that record |
| `false_permit_DANGEROUS` | Per-record false-permit rate |

If one record dominates danger mass, pooled rates can look better than they are.

### 1.5 `phase1_metrics_by_danger_subtype.csv`

| Column | Meaning |
| --- | --- |
| `danger_subtype` | `danger_rhythm` / `danger_morphology` / `danger_noise` / `danger_other` |
| `false_permit` | False-permit within that subtype |

### 1.6 `phase1_cav_l2_l3.csv` (only if both arms)

| Column | Meaning |
| --- | --- |
| `CAV_L3_catches_A0_false_permit` | Among DANGEROUS beats A0 permitted, fraction L3 inhibited |
| `CAV_symmetric_A0_catches_L3` | Symmetric |
| `r_healthy_pearson` / `r_healthy_spearman` | Score correlation on healthy beats |
| `redundancy_ratio_vs_independent` | Joint FP vs product of FPs |
| `healthy_extra_inhibit_cost` | Extra healthy inhibits when L3 vetoes A0 permits |
| `A0_false_permit_DANGEROUS` / `L3_...` / `joint_...` | Rates on shared beats |

### 1.7 `phase1_thresholds.csv`

Per `(record, arm, scorer)` calibration metadata:

| Column (typical) | Meaning |
| --- | --- |
| `status` | `ok`, `insufficient_healthy_calibration`, `scorer_error`, … |
| `n_fit` / `n_val` | Healthy beats used to fit baseline / set threshold |
| `healthy_quantile_threshold` / `conformal_threshold` | Operating thresholds |
| `conformal_status` | `ok` or `alpha_infeasible` (→ conformal decisions inhibit) |
| `feature_dim` | A0: # features kept; L3: embedding dim (pre-PCA) |
| `feature_names_json` | A0 only: which features that record used |
| `n_outlier_removed` | Calibration pruning count |

### 1.8 `phase1_per_beat.csv`

Long format: each beat × arm × scorer × threshold_method.

Important columns: `split`, `decision`, `anomaly_score`, `phase1_label_group`, `threshold`, `score_over_threshold_ratio`.

Only rows with `split == test` and `decision ∈ {permit, inhibit}` enter the metric tables.

### 1.9 Offline / do-not-deploy file

`phase1_offline_operating_points.csv` — threshold tuned using **DANGEROUS labels** to hit ~2% false permit. Useful for thesis appendix curves only. **Not** how deployment sets thresholds.

---

## 2. Layer 2 beat validation — main outputs

Script: `Layer2/validation/run_beat_validation.py`  
(Typical modes include oracle, gated, and **`rpeak_adaptive_cadence_1of8`**.)

| File | Meaning |
| --- | --- |
| `per_beat.csv` | One row per trigger/decision (features, gate reason, cadence fields if used) |
| `metrics_overall.csv` | Healthy permit, abnormal inhibit, false-permit-style rates by mode/feature set |
| `metrics_by_label.csv` | By beat symbol |
| `metrics_by_safety_group.csv` | Policy metrics by safety group |
| `per_record.csv` / `per_group.csv` | Breakdowns |
| `inhibit_reasons.csv` | Why the gate inhibited (hard rule name, etc.) |
| `top_inhibit_features.csv` | Which features drove distance inhibits |
| `consec_inhibit_dist.csv` | Runs of consecutive inhibits |

Cadence-specific columns on `per_beat.csv` (when 1-in-8 mode is on) include things like:

- `cadence_is_stimulation_beat`
- `cadence_phase`
- `cadence_reason` (`cadence_permit`, observation unsafe, …)
- `cadence_observed_safe_beats` / `cadence_required_safe_beats`

**Therapy-relevant Layer 2 claim** should use a **cadence** mode, not only per-beat oracle.

---

## 3. What “good” Phase 1 reading looks like (checklist)

```text
1. encoder_info.json → checkpoint_loaded true  (if layer3 arm present)
2. phase1_thresholds.csv → few insufficient_healthy / alpha_infeasible
3. phase1_metrics_bootstrap.csv → read false_permit + boot_ci_*
4. phase1_metrics_by_record.csv → danger not all in 1 record
5. phase1_metrics_by_danger_subtype.csv → where errors live
6. phase1_cav_l2_l3.csv → if a0+layer3: is L3 complementary?
7. Do NOT quote non-phase1 metrics_overall as A0
```

---

## 4. Can you run this on the cluster now?

| Run | Ready? | Notes |
| --- | --- | --- |
| Layer 3 Phase 1 **A0** (`--phase1-arms a0`) | **Yes** | Ignore encoder junk files |
| Layer 3 Phase 1 **A0+A** (`a0,layer3` + NT-Xent ckpt) | **Yes** after one pretrain | Prefer `--no-random-fallback` |
| Layer 2 beat validation (incl. 1-in-8 cadence) | **Yes** | Separate script/folder; not Phase 1 |

Commands: `Layer3/reports/ZEROSHOT_CLUSTER_RUN_NOTES.md` (§0b) for Phase 1; Layer 2 README / `ALGORITHM_SUMMARY.md` for cadence modes.

---
