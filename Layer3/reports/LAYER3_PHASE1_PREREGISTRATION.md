# Layer 3 Phase 1 — Pre-registration & Step-0 Gate Spec

**Status:** Locked before full cluster campaign (July 2026).  
**Scope:** Human PhysioNet proxy only. Layer 3 is not claimed animal-ready.  
**Companion:** `ZEROSHOT_CLUSTER_RUN_NOTES.md`, `LAYER3_SCIENTIFIC_REVIEW_BRIEF.md`.  
**Detailed reviewer expansion (C1–C5):** `LAYER3_REVIEW_WEAKNESSES_C1_C5.md` — read before claiming power or independence.

This file operationalizes the review:

1. **Step 0 / 0.5** — danger-count, effective *n* records, MDD gate, stratification
2. **Metric definitions** — CAV, score correlation, redundancy vs independence
3. **Pre-registration** — primary metric locked before interpreting danger-label results
4. **Conformal threat model (C5)** — healthy-side budget only; danger FP is empirical

Do **not** change primary metric / operating point / CI method after looking at DANGEROUS-conditioned tables from the campaign.

---

## 1. One-paragraph pre-registration (lock this)

> **Primary claim.** Under a fixed personalized anomaly scorer (Mahalanobis and kNN, healthy-only split-conformal threshold with healthy false-inhibit budget α = 0.10), compare representations A0 / A / A1 / B on beat-synchronous **8 s @ 125 Hz** windows.  
> **Primary metric.** False-permit rate on beats/windows labelled **DANGEROUS**, reported with (i) pooled beat-level Wilson 95% CIs for transparency and (ii) **record-cluster bootstrap** 95% CIs (resample records with replacement, B ≥ 2000, report 2.5/97.5 percentiles of the pooled false-permit rate) as the headline uncertainty. Also report per-record rates (mean/median).  
> **Primary operating point.** Deploy-style threshold from held-out **healthy** validation scores only (`--threshold-method conformal --conformal-alpha 0.10`). This sets a **healthy false-inhibit budget** (≈α if exchangeability holds). It does **not** guarantee false-permit on DANGEROUS. Danger-side performance is measured empirically only. DANGEROUS labels are never used to set the deploy threshold. Offline danger-target curves (e.g. 2% FPR) are secondary / appendix only.  
> **Controls.** A0 = Layer 2 **handcrafted features + the same scorer** (not the full Layer 2 hard-rule gate). SSL arms vary only the frozen encoder.  
> **Main vs appendix.** Pipeline-relevant claims use `layer1_adaptive_gated` (+ 1-in-8 policy rates when available). Oracle is upper-bound / appendix.  
> **Secondary (pre-specified).** Healthy permit rate; false-permit **stratified** by danger subtype (rhythm-origin vs morphology-origin vs noise); A0↔L3 score correlation on healthy beats; conditional added value of L3 among A0-permitted DANGEROUS beats.  
> **Go/no-go.** Full multi-seed campaign proceeds only if Step 0 shows enough within-subject DANGEROUS events under the frozen split for non-degenerate record-bootstrap CIs. Otherwise the headline thesis narrative is “clean protocol + public-data power limitation,” not a powered representation ranking.  
> **Out of scope for this claim.** Animal transfer, stimulation artifacts, dual-scale windows, CoRe-ECG / full NERULA / Deep-SVDD-as-primary.

Copy the paragraph above into `Results/layer3/transition_analysis/MANIFEST.md` when Step 0 is frozen.

---

## 2. Step 0 — danger-event gate (concrete)

### 2.1 Commands

```bash
python Layer3/tools/count_transition_records.py \
  --data-dir data \
  --out-dir Results/layer3/transition_analysis
```

Outputs (existing tool):

```text
transition_records_by_record.csv
transition_summary_by_dataset.csv   # name may match tool output
```

### 2.2 What to freeze in `MANIFEST.md`


| Field                     | Definition                                                            |
| ------------------------- | --------------------------------------------------------------------- |
| Eval record list          | Exact `dataset/record` IDs used for Phase 1 tables                    |
| Pretrain record list      | Exact IDs in the window index used for SSL (or “all downloaded”)      |
| Overlap policy            | `record_disjoint` (preferred) or `overlap_allowed_exploratory`        |
| `n_transition_records`    | Records with healthy baseline **before** DANGEROUS in the same record |
| `n_DANGEROUS_beats` total | Sum over eval transition records                                      |
| Per-record danger         | min / median / max `n_danger_beats` among transition records          |
| Stratified counts         | See Step 0.5                                                          |
| Go/No-go decision         | One sentence + date                                                   |


### 2.3 Effective *n* and MDD (from C1)

Do **not** treat total danger beats as the sample size. Prefer:

```text
effective_n_danger_records =
  # of eval transition records with ≥ 5 DANGEROUS beats
  (records with 1–2 danger beats do not inflate n; note them separately)
```

Estimate a rough **minimum detectable difference (MDD)** in false-permit rate between A0 and an SSL arm at ~80% power under record-level thinking (exact binomial / Fisher on a simplified two-rate comparison, or supervisor-agreed rule of thumb). Write in MANIFEST:

```text
Effective n_danger_records ≈ X
Approx MDD ≈ Y percentage points at 80% power
Decision: GO if MDD ≲ 8 pp; supervisor call if MDD ≳ 10 pp
  (expand data vs reframe to protocol + power-limitation narrative)
```

Illustrative pattern (not a substitute for your real inventory): ~23 effective records can imply MDD ~12–15 pp — too large to detect a 3–5 pp true gain.

### 2.4 Go / No-go rule (operational)

**GO** if all of the following hold:

1. Record-disjoint pretrain/eval **or** strongest claims labelled exploratory.
2. `effective_n_danger_records` and MDD written in MANIFEST; MDD ≲ 8 pp **or** explicit supervisor approval to proceed with a power-limited claim.
3. Record-bootstrap CI will not be degenerate (danger mass not concentrated in 1–2 records).

**NO-GO (narrative shift, not project failure):** if (2) or (3) fail → do **not** sell A0-vs-SSL as powered. Thesis leads with architecture + protocol + data limitation. Pilot (1 seed) may still run for engineering sanity.

The **decision must be written in MANIFEST.md before** multi-seed training.

---

## 3. Step 0.5 — danger-type stratification (concrete)

### 3.1 Subtype taxonomy (for metrics, not for thresholding)

Prefer fine labels when annotations allow (C3), then pool:


| Fine label       | Pool                | Intent                                                     |
| ---------------- | ------------------- | ---------------------------------------------------------- |
| `VT_ONSET`       | `danger_rhythm`     | Transition into VT                                         |
| `VT_SUSTAINED`   | `danger_rhythm`     | Mid-run VT                                                 |
| `VF`             | `danger_rhythm`     | Fibrillation / flutter spans                               |
| `PVC_ISO`        | `danger_morphology` | Isolated ventricular ectopy in otherwise normal context    |
| `PAC_ISO`        | `danger_morphology` | Isolated supraventricular ectopy (if treated must-inhibit) |
| `FUSION`         | `danger_morphology` | Mixed activation                                           |
| `NOISE_ARTIFACT` | `danger_noise`      | Untrustworthy signal (nstdb / NOISE group)                 |
| `danger_other`   | `danger_other`      | Residual                                                   |


Coarse pools for tables: **rhythm-origin** vs **morphology-origin** vs **noise** vs **ALL**.

**Honesty rule:** if a dataset only has rhythm-span DANGEROUS (e.g. vfdb), almost all events land in `danger_rhythm`. Do not invent morphology labels.

### 3.2 What to count at Step 0.5

From annotations (read-only), for the **frozen eval transition records**:

```text
For each subtype S in {danger_rhythm, danger_morphology, danger_noise, danger_other}:
  n_records_with_S
  n_beats_S (or n_windows_S)
  seconds_S if rhythm-span based
```

Practical first pass without new code:

1. Use `transition_records_by_record.csv` for totals.
2. Use `scan_annotations.py` / dataset identity:
  - vfdb, cudb VF episodes, creighton, malignant VA → mostly `danger_rhythm`  
  - nstdb → `danger_noise`  
  - mitdb ventricular beats that map to DANGEROUS → `danger_morphology` or mixed (document the rule)
3. Write the mapping rule in MANIFEST.md (one table).

**After Phase 1:** recompute **false-permit rate stratified by subtype** on `phase1_per_beat.csv` using the same mapping. This tests whether 8 s hurts morphology-origin detection (G3) without running dual-scale.

### 3.3 Dual-scale decision rule (pre-registered)

```text
Dual-scale (1 s + 8 s) = FUTURE only.
Promote to motivated work IFF stratified false-permit shows SSL (or all arms)
clearly worse on danger_morphology than danger_rhythm at matched healthy permit.
Do not add dual-scale before the first A0/A/A1/B campaign.
```

---

## 4. Metric definitions — A0 ↔ Layer 3 added value

Assume Phase 1 produces aligned per-beat rows for arm `a0` and arm `layer3` on the same beats (same `record_id`, beat index / time).  
Let `permit = (score <= threshold)` for each arm. Restrict DANGEROUS metrics to rows with `safety_group == DANGEROUS` (or equivalent Phase 1 label). Restrict healthy correlation to `safety_group == NORMAL` and `split == test` (or the agreed eval split — document it).

### 4.1 Score correlation (healthy)

```text
r_healthy = PearsonCorr( score_A0, score_L3 )
           over NORMAL eval beats with finite scores for both arms

Also report Spearman ρ (robust to monotone transforms).
```

**Interpretation:** high |r| ⇒ representations largely agree on healthy unusualness; AND-fusion less likely to add orthogonal information.

### 4.2 Inhibit agreement on DANGEROUS

```text
On DANGEROUS eval beats:
  both_inhibit  = (~permit_A0) & (~permit_L3)
  both_permit   = permit_A0 & permit_L3          # joint false permit
  only_A0_inhib = (~permit_A0) & permit_L3
  only_L3_inhib = permit_A0 & (~permit_L3)       # L3 unique catches
```

Report counts and rates (denominators = n_DANGEROUS with both scores finite).

### 4.3 Conditional added value (primary system-relevant scalar)

```text
Among DANGEROUS beats that A0 *permits* (A0 false permits):

  CAV = n( L3 inhibits | A0 permits, DANGEROUS )
        / n( A0 permits, DANGEROUS )

  = rate at which Layer 3 newly blocks an A0 false permit
```

Also report the symmetric quantity (A0 catching L3 misses) for honesty.

**Heuristics (from C2 notes — not hard science cutoffs):**  
CAV ≳ 15% → useful complementarity; CAV ≲ 5% → near-pure redundancy;  
`r_danger` > 0.7 or agree_danger > 85% → treat diversity claim with suspicion.

### 4.3b Redundancy vs naive independence (optional but useful)

```text
A0_FP = P(A0 permits | DANGEROUS)
L3_FP = P(L3 permits | DANGEROUS)
Joint_FP = P(A0 permits AND L3 permits | DANGEROUS)   # AND-fusion false permit

If independent: Joint_FP ≈ A0_FP × L3_FP
Redundancy_ratio = Joint_FP / (A0_FP × L3_FP + eps)
  ≫ 1 → positively dependent (redundant); ≈ 1 → closer to independence
```

**Interpretation:** CAV ≈ 0 ⇒ L3 adds little beyond A0 under this scorer; CAV high ⇒ L3 earns its place as an optional veto even if average false-permit is similar.

### 4.4 Healthy cost of that gain

```text
Among NORMAL beats that A0 permits:
  healthy_extra_inhibit_rate =
      n( L3 inhibits | A0 permits, NORMAL ) / n( A0 permits, NORMAL )
```

Report CAV together with this cost (therapy availability impact of the optional veto).

### 4.5 Record-cluster bootstrap (headline CI)

```text
For b = 1..B (B ≥ 2000):
  draw a bootstrap sample of records (with replacement) from the eval set
  recompute pooled false-permit_DANGEROUS on all beats from those records
  (weights: each drawn record contributes its beats once per draw;
   if a record is drawn k times, include its beats k times — standard cluster bootstrap)
CI = percentile(false_permit_b, [2.5, 97.5])
```

Apply the same bootstrap to CAV if sample size allows; otherwise report CAV point estimate + per-record CAV distribution.

### 4.6 Where to compute

**Now automated** by `write_phase1_outputs` in `layer3_phase1_eval.py` (written on
every `--phase1-eval` run):


| Metric                        | Output file (automated)                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| False-permit, stratified      | `phase1_metrics_by_danger_subtype.csv` (danger_subtype mapping)  |
| Record-cluster bootstrap CI   | `phase1_metrics_bootstrap.csv` (headline uncertainty)            |
| Per-record false permit       | `phase1_metrics_by_record.csv`                                   |
| Score correlation / CAV       | `phase1_cav_l2_l3.csv` (requires `--phase1-arms a0,layer3`)      |
| System AND with full L2       | `compare_layer2_layer3.py` (different question — full gate)      |

Subtype mapping is `danger_subtype()` in `layer3_phase1_eval.py`; document any
dataset-specific rule change in MANIFEST. Bootstrap resamples set with
`--phase1-bootstrap-n` (default 2000).


---

## 5. Conformal threat model (C5) — say once

```text
Conformal / healthy quantile:
  → sets healthy false-inhibit budget α on early healthy calibration
  → IF exchangeability holds, new healthy false-inhibit ≈ α
  → beats are temporally dependent → guarantee is weaker than i.i.d. textbooks
  → physiology drift / domain shift can break exchangeability

Conformal does NOT bound:
  → false-permit on DANGEROUS (empirical only)
  → animal / hardware generalization

Always report side-by-side:
  healthy false-inhibit (near α?)  |  danger false-permit (empirical + record CI)
```

Never write “conformal ensures false-permit ≤ …”.

---

## 6. Ordered checklist (copy into MANIFEST)

- Pre-registration paragraph (§1) pasted into MANIFEST  
- `count_transition_records.py` run; CSVs archived  
- Eval / pretrain lists frozen; overlap policy stated  
- Step 0.5: per-record danger × type; **effective_n_danger_records**; **MDD** text  
- Go / No-go written (C1 gate)  
- Pilot 1× NT-Xent (only after Go or with explicit “engineering-only” flag)  
- Full campaign only if Go  
- Post-hoc: stratified FP (C3) + CAV/corr/redundancy (C2) + record-bootstrap CI  
- Report healthy false-inhibit next to danger FP (C5)  
- Dual-scale still FUTURE unless morphology gap rule fires

---

*Locked for Phase 1. Change only with a dated amendment note.*  
*Full prose for C1–C5: `LAYER3_REVIEW_WEAKNESSES_C1_C5.md`.*