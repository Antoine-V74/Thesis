# Layer 2 vs A0 vs Arm A — compare guide

**Purpose:** How to run and interpret the three-way comparison on the cluster
without mixing “full Layer 2 gate” with “Phase 1 representation ablation.”

**Cluster scripts:** [`cluster_jobs/`](cluster_jobs/)  
**Metrics column reference:** [`LAYER3_PHASE1_OUTPUT_METRICS.md`](LAYER3_PHASE1_OUTPUT_METRICS.md)  
**Arm A depth:** [`LAYER3_COMPLETE_STATUS_A.md`](LAYER3_COMPLETE_STATUS_A.md)  
**A0 depth:** [`LAYER3_COMPLETE_STATUS_A0_FIRST.md`](LAYER3_COMPLETE_STATUS_A0_FIRST.md)

---

## 1. Three different things (do not collapse)

| Column | What it is | Decision stack |
| --- | --- | --- |
| **Full Layer 2** | Handcrafted features + **hard rules** + optional **1-in-8 cadence** | Real Layer 2 gate |
| **A0** | Same L2 **features only** + Phase 1 Mahalanobis/kNN + conformal | **Not** the full gate |
| **Arm A** | NT-Xent **embeddings** + **same** Phase 1 scorer as A0 | Representation ablation vs A0 |

```text
A0 vs A     → primary Phase 1 claim (matched scorer; inputs differ — say so)
Full L2 vs A0 → “do hard rules / cadence beat features+distance alone?”
Full L2 vs A  → system comparison (different stack) — frame as systems, not encoder-only
```

---

## 2. Normalize / augs (shared Layer 3 path for A)

### Robust median / MAD (ECG **waveform**)

```text
x ← x - median(x)
scale ← 1.4826 * MAD(x)
x ← x / scale
```

Per-window, pretrain **and** Phase 1 eval. Robust to spikes vs mean/std.  
Not a SimCLR paper requirement — ECG preprocessing so train/eval match.

### L2 normalization (**vectors**, not the ECG)

```text
v ← v / ||v||₂
```

| Where | Same as SimCLR? |
| --- | --- |
| Projection outputs in NT-Xent train | **Yes** |
| `--l2-normalize-embeddings` before PCA/scorer at eval | Common SSL practice; protocol on for Arm A |

**MAD ≠ L2.** MAD scales the signal; L2 unit-lengths embeddings/projections.

### Augmentations

Keep the **safe** set (noise, mild wander, light crop). Do **not** add aggressive
augs for primary A (strong warp / polarity / heavy mask) — they can blur
sinus vs danger. Richer augs = later ablation only. Always `--augment-fs 125`.

---

## 3. Cluster job map

| Order | Script | Out-dir (default) | Parallel? |
| --- | --- | --- | --- |
| 1 | [`cluster_jobs/01_layer2_mitbih_gold.sh`](cluster_jobs/01_layer2_mitbih_gold.sh) | `Results/layer2/beat_sync_mitbih_gold_causal` | with job 2 |
| 2 | [`cluster_jobs/02_phase1_a0_only.sh`](cluster_jobs/02_phase1_a0_only.sh) | `Results/layer3/validation/pilot_mitbih_A0_only_8s` | with job 1 |
| 3a | [`cluster_jobs/03a_build_window_index_mitbih.sh`](cluster_jobs/03a_build_window_index_mitbih.sh) | window index CSV | before 3b |
| 3b | [`cluster_jobs/03b_pretrain_arm_a_ntxent.sh`](cluster_jobs/03b_pretrain_arm_a_ntxent.sh) | `Results/layer3/pretrain/ntxent_mitbih_seed0` | after 3a |
| 3c | [`cluster_jobs/03c_phase1_a0_plus_a.sh`](cluster_jobs/03c_phase1_a0_plus_a.sh) | `Results/layer3/validation/pilot_mitbih_ntxent_seed0_8s` | after 3b |

Gold allowlist (all jobs): `Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv`  
(Layer 2 now accepts `--records-csv` for the same list.)

**Shortcut:** if you only care about A0 vs A, skip job 2 and use A0 rows from job 3c.  
Job 2 is still useful as a cheap A0-only sanity check before pretrain.

From repo root on the cluster:

```bash
# parallel-friendly
bash Layer3/reports/cluster_jobs/01_layer2_mitbih_gold.sh &
bash Layer3/reports/cluster_jobs/02_phase1_a0_only.sh &
wait
bash Layer3/reports/cluster_jobs/03a_build_window_index_mitbih.sh
bash Layer3/reports/cluster_jobs/03b_pretrain_arm_a_ntxent.sh
bash Layer3/reports/cluster_jobs/03c_phase1_a0_plus_a.sh
```

Or sequential master: `bash Layer3/reports/cluster_jobs/00_run_all_sequential.sh`

---

## 4. Which files to open after each job

### Full Layer 2 (`Results/layer2/beat_sync_mitbih_gold_causal/`)

| File | Use |
| --- | --- |
| `metrics_overall.csv` | Healthy / abnormal / dangerous rates by **mode** |
| `metrics_by_safety_group.csv` | Policy metrics |
| `per_beat.csv` | Filter `mode == oracle` vs `mode == rpeak_adaptive_cadence_1of8` |
| `inhibit_reasons.csv` | Hard-rule vs distance reasons |

Therapy-shaped claim → prefer **cadence_1of8** mode rows.  
Upper-bound morphology/rhythm gate → **oracle** mode.

### Phase 1 A0-only or A0+A (`Results/layer3/validation/...`)

| File | Use |
| --- | --- |
| `phase1_metrics_bootstrap.csv` | Headline false-permit + record CI |
| `phase1_metrics_overall.csv` | Rates; filter `arm` |
| `phase1_cav_l2_l3.csv` | Only after 3c (`a0,layer3`) |
| `encoder_info.json` | 3c only: `checkpoint_loaded: true` |

For A0-only folder: ignore `per_beat.csv` / `metrics_overall.csv` (encoder junk).

---

## 5. Comparison table (fill after runs)

Copy numbers from the files above. Use **conformal** + preferred scorer
(Mahalanobis primary; note kNN if it disagrees). For Layer 2, state the **mode**.

| Metric | Full L2 (oracle) | Full L2 (cadence 1-of-8) | A0 (Phase 1) | Arm A (Phase 1) |
| --- | --- | --- | --- | --- |
| Cohort | 13 MIT-BIH gold | same | same | same |
| False permit DANGEROUS | _fill_ | _fill_ | _fill_ | _fill_ |
| False permit 95% CI | _(if available)_ | _(if available)_ | bootstrap | bootstrap |
| False inhibit NORMAL / healthy permit | _fill_ | _fill_ | _fill_ | _fill_ |
| AUROC NORMAL vs DANGEROUS | _optional_ | _optional_ | phase1 | phase1 |
| CAV (L3 catches A0 FP) | n/a | n/a | from 3c | from 3c |
| Notes | hard rules on | stim opportunities only | features+distance | NT-Xent emb |

### Supervised arm C ladder (fill after 07 / 14 / 15 / 16)

Same scorer, same gold cohort, same gold exclusion. **Exploratory** — confirm on an
untouched cohort before any claim (spec: [`LAYER3_ARM_C_LADDER_SPEC.md`](LAYER3_ARM_C_LADDER_SPEC.md)).

| Metric | A0 (Phase 1) | C `supcon` | C1 `supcon_oe` | C2 `deepsad` | C3 `supcon_hybrid` |
| --- | --- | --- | --- | --- | --- |
| False permit DANGEROUS | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| False permit 95% CI | bootstrap | bootstrap | bootstrap | bootstrap | bootstrap |
| False inhibit NORMAL / healthy permit | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| CAV (arm catches A0 FP) | n/a | _fill_ | _fill_ | _fill_ | _fill_ |
| Notes | features+distance | +labels (SupCon) | +outlier exposure | AD-native (SVDD+OE) | SupCon+SVDD+OE |

**Read:** does any C-rung push false-permit **below A0** or add **CAV** over A0? If neither,
the supervised-representation ladder does not beat handcrafted features on this pilot →
bank the negative and return to Layer 2 / runtime. If a rung wins, re-run it on an
untouched cohort before promoting the claim.

**Framing sentence for the thesis:**

> We report full Layer 2 (hard-rule gate, with and without 1-in-8 cadence) as the
> deployable-style baseline, and Phase 1 A0 vs Arm A as a matched-scorer
> representation ablation. A0 is not the full Layer 2 gate; Arm A uses 8 s
> embeddings at 125 Hz while A0 uses Layer 2 features (5 s morph context + 30 s RR).

---

## 6. Go / no-go checklist

- [ ] Job 01 finished; cadence + oracle rows present  
- [ ] Job 02 or 3c A0 rows look sane (bootstrap non-degenerate)  
- [ ] Job 03b: `pretrain_records.json` excludes gold  
- [ ] Job 03c: `checkpoint_loaded: true`; `phase1_cav_l2_l3.csv` exists  
- [ ] Table above filled; framing sentence used in any slide/supervisor note  

---

*Update §5 when numbers land. Do not edit cluster scripts mid-campaign without
bumping out-dir names.*
