# Layer 3 — Complete status (A0 first)

**Purpose:** One place that explains *everything* done so far for the Layer 3
Phase 1 campaign, starting with **Arm A0** (the control you should run first on
the cluster). Written so you can hand this to a supervisor, a collaborator, or
yourself in three months and still know what was decided, what was coded, what
was fixed, and what is still open.

**Date:** July 2026  
**Companion short files:** `LAYER3_CONSOLIDATED_SUMMARY.md` (medium overview),
`ZEROSHOT_CLUSTER_RUN_NOTES.md` (commands), `LAYER3_PHASE1_PREREGISTRATION.md`
(locked protocol), `LAYER3_REVIEW_AND_OPEN_ISSUES.md` (bug list).

---

## 0. One-sentence thesis framing

> Layer 3 does **not** learn population-level pathology. SSL learns a useful ECG
> representation space; safety is then **personalized** by fitting a healthy
> baseline per record/session and inhibiting when new windows move too far from
> that baseline.

A0 is the control that asks: *before we even use SSL, how good is the same
personalized scorer on Layer 2 handcrafted features?*

---

## 1. Arm A0 — full detail (start here)

### 1.1 What A0 is

| Item | Definition |
| --- | --- |
| **Name** | A0 — handcrafted-feature control |
| **Representation** | Layer 2 beat features (morphology + RR + energy/spectral proxies) |
| **Scorer** | Same as SSL arms: Mahalanobis and/or kNN on the feature matrix |
| **Threshold** | Same as SSL arms: healthy-only conformal (`α = 0.10`) + healthy quantile report |
| **What it is not** | Not the full Layer 2 hard-rule gate (`gate.py`). Same features, different head (distance-to-baseline) |

**Scientific role:** floor / ceiling reference for the representation ablation.

```text
Question answered by A0:
  "Under the frozen personalized anomaly protocol, how well do interpretable
   Layer 2 features already separate healthy vs DANGEROUS beats?"

Question answered later by A / A1 / B:
  "Do SSL embeddings beat that floor under the *same* scorer and threshold?"
```

If A0 is already excellent on MIT-BIH gold, SSL has a high bar. If A0 fails on
morphology-origin danger that SSL catches, that is the complementarity story
(CAV — see §4).

### 1.2 How A0 is computed in code

Entry point: `Layer3/validation/run_beat_validation.py` with `--phase1-eval`.

Pipeline inside `layer3_phase1_eval.py`:

```text
1. Build beat table (oracle annotations or Layer 1 triggers)
2. For each record_key:
     early healthy beats → fit / val splits (+ guard)
3. compute_a0_feature_dicts():
     load WFDB lead → filter → extract_beat_features() from Layer 2
     (morphology window ≈ 5 s, RR lookback ≈ 30 s)
4. _a0_matrix_for_group():
     keep features with ≥95% finite values on fit beats
     impute remaining NaNs with fit median
     z-score using fit mean/std only
5. Same scorer path as Layer 3 embeddings:
     fit_baseline_with_pruning → score all beats
     conformal + quantile thresholds from healthy val scores
6. Write Phase 1 CSVs (false permit, bootstrap CI, strata, …)
```

**Important framing caveat (do not forget in the thesis):**

| Arm | Observation unit |
| --- | --- |
| A0 | Layer 2 features from ≈5 s morphology + 30 s RR lookback |
| Layer 3 (A/A1/B) | One **8 s** waveform embedding |

Same **downstream scorer**, different **input**. Fair as a *representation*
ablation; **unfair** if you claim “identical inputs except the encoder.”
Documented as limitation #11 in `LAYER3_CONSOLIDATED_SUMMARY.md`.

### 1.3 A0 does not need a pretrained encoder

A0 uses only handcrafted features. You can run **A0 alone** with no SSL
checkpoint:

```bash
python Layer3/validation/run_beat_validation.py \
  --data-dir data \
  --datasets mit_bih_arrhythmia \
  --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --out-dir Results/layer3/validation/pilot_mitbih_A0_only_8s \
  --mode oracle \
  --window-s 8 --target-fs 125 \
  --causal-window --lookahead-ms 100 \
  --per-record-calibration --guard-s 8 \
  --phase1-eval --phase1-arms a0 \
  --phase1-scorers mahalanobis,knn \
  --threshold-method conformal --conformal-alpha 0.10 \
  --device cuda
```

Notes:

- Omit `--checkpoint` and omit `--no-random-fallback` (encoder is unused for A0).
- `--window-s 8` still matters for Layer 3 later; for A0-only the Layer 3
  embedding path is skipped, but beat table / splits stay protocol-aligned.
- For the real pilot comparison, prefer `--phase1-arms a0,layer3` with a real
  A checkpoint (see §6).

### 1.4 What “good A0” looks like (inspection checklist)

After the run, open:

| File | What to look at |
| --- | --- |
| `phase1_metrics_overall.csv` | `false_permit_DANGEROUS`, `false_inhibit_NORMAL` for A0 |
| `phase1_metrics_bootstrap.csv` | **Headline CI** (record-cluster bootstrap) — not Wilson beat CI |
| `phase1_metrics_by_record.csv` | Danger mass concentrated in 1–2 records? |
| `phase1_metrics_by_danger_subtype.csv` | Rhythm vs morphology vs noise |
| `phase1_thresholds.csv` | Any `insufficient_healthy_calibration` / `alpha_infeasible`? |
| `phase1_offline_operating_points.csv` | Secondary only (uses DANGEROUS labels — not deployable) |

**Do not** headline Wilson beat CIs. Beats inside a record are correlated; the
pre-registered uncertainty is the record bootstrap.

### 1.5 A0 success criteria for “send the rest of the branches”

A0 pilot is a **sanity / protocol** gate, not a claim that A0 is the final gate.

Proceed to Arm A (NT-Xent) if:

1. All 13 gold records produce scorable outputs (or failures are explained).
2. Bootstrap CI is non-degenerate (danger not trapped in 1–2 records).
3. Conformal infeasible rate is low; insufficient-calibration records are few.
4. Numbers look physically plausible (not all-permit or all-inhibit everywhere).

Then run A with the same Phase 1 flags so `phase1_cav_l2_l3.csv` appears
(requires both `a0` and `layer3` arms).

---

## 2. Protocol freeze (what A0 inherits)

These settings are locked for Phase 1. A0 uses the same scorer / threshold /
record list as SSL arms.

| Setting | Frozen value | Why |
| --- | --- | --- |
| Eval cohort | 13 MIT-BIH gold transition records | Healthy baseline *before* DANGEROUS in same record |
| Window (L3 path) | 8 s @ 125 Hz beat-sync | Rhythm + morphology in one waveform |
| Trigger mode (primary engineering) | `oracle` for upper bound; `layer1_adaptive_gated` later for pipeline claim | Oracle = annotated beats |
| Calibration | Per-record, early healthy only | Deployment-shaped personalization |
| Guard | `guard_s = window_s` (8 s) | Avoid window overlap into test |
| Threshold | Conformal α = 0.10 on healthy val scores | Healthy false-inhibit *budget*, not danger guarantee |
| Scorers | Mahalanobis + kNN | Two heads, same representation question |
| Primary metric | False-permit on DANGEROUS + record-cluster bootstrap CI | Safety-first |
| Therapy context | Inhibit-only; 1-in-8 stack policy | Software never commands stim |

Pre-registration paragraph: `LAYER3_PHASE1_PREREGISTRATION.md` §1.

---

## 3. Everything implemented / fixed (July 2026 pass)

Grouped by theme. This is the full changelog of the audit that prepared A0 and
the later arms.

### 3.1 Phase 1 analysis (was missing; now automatic)

| Feature | Where | Output |
| --- | --- | --- |
| Record-cluster bootstrap 95% CI | `layer3_phase1_eval._record_cluster_bootstrap` | `phase1_metrics_bootstrap.csv` |
| Per-record false permit | `_false_permit_by_record` | `phase1_metrics_by_record.csv` |
| Danger-type stratification | `danger_subtype()` + `_false_permit_stratified` | `phase1_metrics_by_danger_subtype.csv` |
| A0 ↔ L3 CAV + score correlation | `_compute_cav_l2_l3` | `phase1_cav_l2_l3.csv` |

CLI: `--phase1-bootstrap-n` (default 2000), `--phase1-bootstrap-seed`.

**CAV definition (pre-registered):** among DANGEROUS beats that A0 *permits*,
what fraction does Layer 3 correctly *inhibit*?  
High CAV → L3 adds value beyond A0. Near-zero CAV → near-redundant with A0.

Also reported: healthy score Pearson / Spearman, joint false-permit,
redundancy ratio vs independence, healthy extra-inhibit cost.

### 3.2 Pretraining footguns (fixed)

| Bug | Fix |
| --- | --- |
| Augmentor default `fs=250` while protocol is 125 Hz | `--augment-fs` (default 125) into `AugmentConfig` |
| Pretrain mean/std z-score vs eval median/MAD | Both use `robust_normalize_window` (per-window) |
| `--healthy-only` silently no-op if column missing | **Fails closed** with clear error |
| Pretrain/eval record overlap easy on 13 gold records | `--exclude-records-csv` + `pretrain_records.json` |
| Bare `record` name collision on exclude | Match only `dataset/record` |
| Augmentor RNG ignored `--seed` | Seed passed into `ECGAugmentor` |

**Normalization subtlety (advanced):** the `.npy` signal cache is still
whole-record normalized in `build_window_index.cache_record_signal`. That is
OK because per-window median/MAD is affine-equivariant: re-normalizing each
slice cancels the global scale, so pretrain and eval inputs match. Do **not**
remove per-window norm without also changing the cache.

### 3.3 Checkpoint loading (safety-critical fix)

**Before:** a bad `--checkpoint` path or key mismatch could leave a randomly
initialized `ECGEncoder1D` running, with only a warning. `--no-random-fallback`
blocked only the tiny `RandomConvEncoder` import fallback — not this case.

**After (`build_encoder`):**

- `checkpoint_loaded=true` only if **zero** missing encoder keys.
- With `--no-random-fallback`: **raise** on missing file / bad state dict /
  missing keys.
- Always verify `encoder_info.json` after SSL eval runs.

A0-only runs do not need this; A/A1/B runs **must** use `--no-random-fallback`.

### 3.4 Arm B / B1 redesign (coded; run after A0→A)

| Arm | Objective flag | Idea |
| --- | --- | --- |
| **B** (primary) | `mae_consistency` | Masked recon + non-contrastive same-window consistency (VICReg-style on two masks) |
| **B1** (ablation) | `mae_subject_contrastive` | Masked recon + subject/record contrastive (ZEROSHOT-like) |

Why B is preferred over subject-contrastive as primary: anomaly detection wants
a tight *healthy* cloud, not “all windows from the same patient are close”
(which can pull healthy and VT together). Spec: `LAYER3_ARM_B_B1_SPEC.md`.

### 3.5 Documentation reframed

- `ALGORITHM_SUMMARY.md` — 8 s @ 125 Hz, arm table, corrected CLI examples.
- `ZEROSHOT_CLUSTER_RUN_NOTES.md` — new Phase 1 output files + bootstrap flags.
- `LAYER3_PHASE1_PREREGISTRATION.md` §4.6 — metrics marked as automated.
- `LAYER3_CONSOLIDATED_SUMMARY.md` — A0≠L3 input caveat.
- `AGENTS.local.md` — local Windows: no torch/numpy/pandas; pip SSL-blocked;
  only `py_compile` locally.
- `LAYER3_REVIEW_AND_OPEN_ISSUES.md` — July audit fixed vs open lists.

---

## 4. Advanced open issues (tracked, none block A0)

These are real engineering/scientific details found in the deep audit. They are
**recorded, not forgotten**. None of them block sending the A0 pilot.

### 4.1 Mahalanobis pruning: in-sample vs leave-one-out (LOO)

**Where:** `layer3_embedding_mahalanobis.fit_robust`  
**Issue:** When pruning calibration outliers, Mahalanobis scores each fit point
with a model that *includes* that point. Contaminated “healthy” beats (missed
PVC, noise burst) can look artificially close and survive pruning.  
**Contrast:** kNN pruning already uses LOO-style scores.  
**Risk for A0/pilot:** Low on clean MIT-BIH gold with early NORMAL-only fit.  
**Later fix:** Align Mahalanobis pruning with LOO (mirror kNN).  
**Until then:** Prefer low `--calibration-outlier-frac` or inspect
`n_outlier_removed` in thresholds.

### 4.2 Legacy metrics count benign PVCs as “abnormal”

**Where:** `metrics_legacy_healthy_vs_abnormal.csv` in beat validation.  
**Issue:** Legacy path uses `~is_healthy_window` as abnormal. Isolated PVC /
BENIGN_ABNORMAL therefore inflate “abnormal catch.”  
**What to trust instead:** Phase 1 tables (`phase1_*`) and policy metrics
(`safety_group` / DANGEROUS).  
**Action:** Treat legacy CSV as historical/advisory; do not put it in the thesis
primary table.

### 4.3 SBR (sinus bradycardia) → NORMAL

**Where:** `label_grouping.py`  
**Issue:** Rhythm label `SBR` maps to NORMAL, so bradycardia windows can enter
healthy calibration.  
**Policy question:** Is slow sinus still a valid stimulation baseline for your
assist device? If brady morphology differs from the later “healthy” test beats,
the baseline cloud widens.  
**Action:** Confirm with supervisor / therapy policy. If brady is not
stimulation-safe, remap to BENIGN_ABNORMAL or exclude from calibration.

### 4.4 Window validation defaults to global split

**Where:** `run_window_validation.py` without `--per-record-calibration`.  
**Issue:** Global record split ≠ deployment (per-session early healthy).  
**Pilot impact:** **None** — `run_beat_validation.py` is always per-record.  
**Action:** Prefer `--per-record-calibration` whenever you use window
validation for thesis numbers; consider flipping the default later.

### 4.5 Signal-cache provenance

**Where:** `build_window_index.cache_record_signal`  
**Issue:** On cache hit, code trusts the existing `.npy` and assumes current
`--target-fs` / lead. A stale cache from an old 250 Hz run can silently feed
wrong windows.  
**Action:** If you change fs/lead/window, rebuild with
`--overwrite-signal-cache`. Longer-term: write a sidecar JSON with fs, lead,
resample params and validate on load.

### 4.6 Trigger mode `filtfilt` is offline / non-causal

**Where:** `run_beat_validation.zero_phase_layer1_filter`  
**Issue:** `layer1_adaptive_gated` detects peaks after **zero-phase** filtering
of the whole record. Peak times therefore use future samples.  
**Already labeled:** offline validation filter, not embedded causal firmware.  
**Action:** Never cite trigger-mode results as a real-time latency claim. For
causal claims, use a causal Layer 1 filter path (future work).

### 4.7 Items verified correct (so you know what we checked)

These were inspected and are sound for the pilot:

- Conformal is upper-tail; infeasible α → inhibit on conformal decisions.
- Per-record fit/val/guard temporal split; guard defaults to window length.
- PCA / L2 normalize fit on healthy-fit only, then applied to all (no leakage).
- Scorer fit on fit split only; val used only for thresholds.
- `is_healthy_window = (safety_group == NORMAL)`; AF_CONTEXT → inhibit expected.
- Insufficient healthy calibration → fail-safe inhibit.
- Encoder convolutions themselves are causal-padded; non-causality comes from
  windowing / offline filters, not from the CNN.

---

## 5. Arms after A0 (order of attack)

Do **not** jump to B before A0 and A look sane.

```text
1. A0 alone          ← you are here (control floor, no SSL)
2. A  (ntxent)       ← first SSL arm; A0 vs A comparison + CAV
3. A1 (vicreg)       ← non-contrastive full-window ablation
4. B  (mae_consistency)     ← primary masked arm
5. B1 (mae_subject_contrastive) ← subject-contrastive ablation
```

| Arm | `--ssl-objective` | Needs checkpoint | Prefer `--healthy-only`? |
| --- | --- | --- | --- |
| A0 | *(none — features)* | No | N/A |
| A | `ntxent` | Yes | Optional (broad SSL OK) |
| A1 | `vicreg` | Yes | Optional |
| B | `mae_consistency` | Yes | **Yes** (preferred primary) |
| B1 | `mae_subject_contrastive` | Yes | **Yes** (safer ablation) |

For SSL runs that will be compared to gold eval records, hold gold out of
pretraining:

```bash
--exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv
```

Check `pretrain_records.json` afterward.

---

## 6. Recommended cluster sequence (A0 → A)

### Step A — A0-only sanity (optional but cheap)

Command in §1.3. Confirm Phase 1 CSVs and bootstrap CI.

### Step B — Pretrain A + joint A0/A Phase 1 (main pilot)

Use `ZEROSHOT_CLUSTER_RUN_NOTES.md` §0b, with these additions preferred:

```bash
# During pretrain A:
  --augment-fs 125 \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv

# During beat validation:
  --phase1-arms a0,layer3 \
  --no-random-fallback
```

Verify:

1. `encoder_info.json` → `checkpoint_loaded: true`, empty/acceptable missing keys.
2. `phase1_cav_l2_l3.csv` exists (both arms present).
3. Headline numbers from `phase1_metrics_bootstrap.csv`.

### Step C — Only then A1 / B / B1

Same eval command; swap checkpoint / objective. Keep record list and threshold
frozen.

---

## 7. File map (where truth lives)

| Need | File |
| --- | --- |
| This complete status | `LAYER3_COMPLETE_STATUS_A0_FIRST.md` *(this file)* |
| Medium overview | `LAYER3_CONSOLIDATED_SUMMARY.md` |
| Locked protocol | `LAYER3_PHASE1_PREREGISTRATION.md` |
| Cluster commands | `ZEROSHOT_CLUSTER_RUN_NOTES.md` |
| B complete status | `LAYER3_COMPLETE_STATUS_B.md` |
| B1 complete status | `LAYER3_COMPLETE_STATUS_B1.md` |
| C complete status | `LAYER3_COMPLETE_STATUS_C.md` |
| B/B1 design | `LAYER3_ARM_B_B1_SPEC.md` |
| C design | `LAYER3_ARM_C_SUPERVISED_SPEC.md` |
| Bugs fixed / open | `LAYER3_REVIEW_AND_OPEN_ISSUES.md` |
| Algorithm | `../ALGORITHM_SUMMARY.md` |
| Gold allowlist | `pilot_lists/pilot_primary_mitbih_gold.csv` |
| Local laptop rules | `../../AGENTS.local.md` |
| A0 feature extraction | `validation/layer3_phase1_eval.py` → `compute_a0_feature_dicts` |
| Beat runner | `validation/run_beat_validation.py` |
| Pretrain | `tools/pretrain_encoder.py` |

---

## 8. Go / no-go for sending A0 to the cluster

| Check | Status |
| --- | --- |
| Protocol frozen (8 s / 125 Hz / conformal α=0.10 / 13 gold) | Yes |
| A0 code path (features + same scorer) | Ready |
| Phase 1 bootstrap / strata / CAV writers | Ready |
| Pretrain footguns fixed (for when you start A) | Ready |
| Checkpoint fail-closed under `--no-random-fallback` | Ready |
| Local laptop can run torch jobs | **No** — cluster only |
| Open issues block A0? | **No** |

**Verdict: GO for A0 (and then A).**  
Send A0 first if you want a cheap protocol check; or send the joint
`a0,layer3` pilot after one NT-Xent seed. Do not start multi-seed A1/B/B1 until
that one-seed MIT-BIH pilot looks sane.

---

## 9. What to say in one paragraph (thesis / supervisor)

> We freeze a personalized anomaly protocol (per-record healthy Mahalanobis/kNN
> + conformal healthy false-inhibit budget α=0.10) and compare representations.
> **A0** is the control: Layer 2 handcrafted features under that same scorer —
> not the full Layer 2 hard gate. SSL arms (A/A1/B/B1) change only the frozen
> encoder. Primary metric is false-permit on DANGEROUS beats with
> **record-cluster bootstrap** uncertainty. We also report danger-type
> stratification and conditional added value of Layer 3 among A0 false permits
> (CAV). The first cluster run is MIT-BIH gold (13 transition records), A0 then
> A; later arms follow only if the pilot is sound.

---

---

## 10. Arm A1 — full detail (VICReg)

**Design reference:** `VICREG_A1_IMPLEMENTATION_PLAN.md` (status: **implemented**).  
**Code:** `pipeline/layer3_vicreg.py` + `--ssl-objective vicreg` in `tools/pretrain_encoder.py`.

### 10.1 What A1 is

| Item | Definition |
| --- | --- |
| **Name** | A1 — VICReg non-contrastive SSL |
| **CLI** | `--ssl-objective vicreg` |
| **Paper** | Bardes, Ponce, LeCun 2022 (VICReg) |
| **Input** | Same as A: two **augmented full windows** (8 s @ 125 Hz), no masking |
| **Encoder** | Same `ECGEncoder1D` (128-d) as A / B |
| **Training-only head** | Expander MLP `512,512,512` — **discarded** after pretrain |
| **Runtime** | Identical to A: frozen encoder → L2/PCA → Mahalanobis/kNN → conformal |

**Scientific role:** isolate “do we need contrastive negatives?”  
A uses NT-Xent (push other batch windows away). A1 uses **no negatives**: only
make two views of the *same* window agree, plus variance/covariance so the
embedding does not collapse.

```text
A  = contrastive (NT-Xent)     — needs negatives in the batch
A1 = non-contrastive (VICReg)  — same two-view path, no negatives
B  = masked recon + consistency — different hypothesis (reconstruction)
```

A1 is **not** the same as B: B masks and reconstructs; A1 never masks.

### 10.2 VICReg loss (training only)

Three terms on expander outputs of two augmented views:

| Term | Role |
| --- | --- |
| **Invariance** (sim) | MSE between the two views → same window → similar embedding |
| **Variance** (var) | Keep each dim’s std above a floor → anti-collapse |
| **Covariance** (cov) | Penalize off-diagonal cov → decorrelate dims (good for Mahalanobis) |

Default coeffs (frozen protocol): `sim=25`, `var=25`, `cov=1`, expander `512,512,512`.

**Never** use VICReg loss / expander output as the permit score. Only encoder
embeddings + healthy-baseline distance.

### 10.3 Why A1 after A0 / A

```text
1. A0  — handcrafted floor (no SSL)
2. A   — first SSL arm (NT-Xent)
3. A1  — same data/augs as A, swap objective to VICReg
4. B   — masked family (different hypothesis)
```

Run A1 only after one-seed A looks sane. Same Phase 1 eval command; only the
checkpoint / `--ssl-objective` change.

### 10.4 Cluster commands (A1)

Pretrain (prefer holding gold out):

```bash
python Layer3/tools/pretrain_encoder.py \
  --window-index Results/layer3/window_index/layer3_windows_8s_125hz.csv \
  --checkpoint-dir Results/layer3/pretrain/vicreg_seed0 \
  --ssl-objective vicreg \
  --positive-mode same_window \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --augment-fs 125 \
  --vicreg-expander-dims 512,512,512 \
  --vicreg-sim-coeff 25 --vicreg-var-coeff 25 --vicreg-cov-coeff 1 \
  --epochs 100 --batch-size 256 --lr 3e-4 \
  --seed 0 --device cuda
```

Phase 1 eval (same as A, point at A1 checkpoint):

```bash
python Layer3/validation/run_beat_validation.py \
  --data-dir data \
  --datasets mit_bih_arrhythmia \
  --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --checkpoint Results/layer3/pretrain/vicreg_seed0/encoder_last.pt \
  --out-dir Results/layer3/validation/pilot_mitbih_vicreg_seed0_8s \
  --mode oracle \
  --window-s 8 --target-fs 125 \
  --causal-window --lookahead-ms 100 \
  --per-record-calibration --guard-s 8 \
  --l2-normalize-embeddings --pca-dim 32 \
  --phase1-eval --phase1-arms a0,layer3 \
  --phase1-scorers mahalanobis,knn \
  --threshold-method conformal --conformal-alpha 0.10 \
  --no-random-fallback \
  --device cuda
```

Read the same `phase1_*` files; `arm=layer3` rows are **A1** for this out-dir.
Compare to the NT-Xent out-dir for A vs A1.

### 10.5 A1-specific caveats

| Point | Detail |
| --- | --- |
| Needs a real checkpoint | Unlike A0; always `--no-random-fallback` |
| Same augs / fs as A | Use `--augment-fs 125`; same safe augmentations |
| Expander discarded | Checkpoint should be encoder weights; verify `encoder_info.json` |
| Not B | No masking; if you want masked + VICReg-style consistency, that is **B** (`mae_consistency`) |
| Batch size | Variance/cov terms want a decent batch (256 is the protocol default) |
| Optional `--healthy-only` | Same story as A: broad SSL OK; healthy-only = ablation |

### 10.6 What “good A1” means

A1 wins as an arm if, under the **same** Phase 1 protocol as A:

- false-permit (bootstrap CI) ≤ A, or  
- clearly better CAV / subtype profile without killing healthy availability  

If A1 ≈ A, report “objective family did not matter much.” If A1 ≫ A on
Mahalanobis, that supports the “no-negatives + decorrelated dims” hypothesis.

### 10.7 Status

| Item | Status |
| --- | --- |
| Code implemented | **Yes** |
| Smoke-tested in `smoke_test_layer3.py` | **Yes** (cluster/CI env) |
| Multi-seed campaign | **After** A0 + one-seed A pilot |
| Docs | This §10 + `VICREG_A1_IMPLEMENTATION_PLAN.md` |

---

*End of complete status. Update when A0/A/A1 pilot numbers land or when an open
issue in §4 is closed.*
