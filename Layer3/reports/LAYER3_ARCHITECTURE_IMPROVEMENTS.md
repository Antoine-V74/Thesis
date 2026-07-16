# Layer 3 Architecture: Possible Changes and Improvements

Companion to the **Layer 3 ZEROSHOT-Inspired Implementation Plan** and
`LAYER3_ARCHITECTURE_RATIONALE.md`. That plan defines the arms we are training
now (A0 handcrafted control, A NT-Xent, B ZEROSHOT-style MAE + subject
contrastive). This document is the backlog of concrete architecture changes we
*could* bring to the current Layer 3 design, why each one might help, and how we
would validate it before believing it.

This is a candidate list, not a commitment. Per project rule
(`CLAUDE.md` §13): prefer the smallest interpretable change that solves a
*measured* safety problem. Nothing here should be adopted on intuition alone —
each item lists how to prove it earns its place against the current baseline.

---

## 0. Guardrails every change must respect

Any modification below is only acceptable if it keeps the existing safety
contract intact:

- Layer 3 can only **inhibit**; it never commands stimulation.
- Uncertainty, missing data, model/encoder failure, or insufficient healthy
calibration → **inhibit**.
- Deployable calibration and thresholds use **healthy data only**. Anything that
touches DANGEROUS labels is offline analysis (e.g. `danger_2pct_offline`).
- Runtime code stays **causal** (past samples + explicitly-bounded post-R
lookahead). Non-causal tricks are offline-only and must be labelled as such.
- The downstream decision stays fixed and transparent:
`embedding/features → optional L2 + PCA → Mahalanobis/kNN → threshold → permit/inhibit`. Reconstruction error is never the anomaly score.

If a proposed change cannot preserve these, it is rejected regardless of AUROC.

---

## 1. Current architecture snapshot (what we are improving on)

```text
beat/trigger → causal beat-sync window (primary 8 s @ 125 Hz; 1 s = morphology ablation)
            → ECGEncoder1D (1D ResNet, ~700K params) → 128-dim embedding
            → per-record healthy baseline (Mahalanobis or kNN)
            → conformal threshold from held-out healthy calibration (α≈0.10)
            → permit / inhibit  (AND with Layer 2 at decision time; layers independent)
```

Dual-scale (1 s + 8 s concat) remains a **candidate improvement** below — ask external
reviewers whether it should stay future-only or become primary before trusting results.

Known weak spots already documented in `LAYER3_REVIEW_AND_OPEN_ISSUES.md`:

- single-Gaussian Mahalanobis is sensitive when healthy calibration is small;
- fusion beats are intrinsically hard (mixed normal/ventricular activation);
- per-record calibration starves records with few early healthy beats;
- one beat → one independent decision ignores temporal context;
- window length / causality / quantile strongly move the safety–availability
trade-off, so "improvements" can be artefacts of configuration.

The improvements below target these specifically.

---

## 2. Improvement catalog

Each item: **idea → why → risk/safety → effort → how to validate**.
Effort is rough (S ≤ 1 day, M ≈ few days, L ≈ 1–2 weeks incl. cluster runs).

### 2.1 Representation / encoder

**(a) Multi-scale / dual-window embedding.**
Concatenate a short morphology window (≈1 s, local QRS shape) with a longer
rhythm window (≈8 s, RR/rhythm context) before scoring.

- *Why:* morphology anomalies (wide QRS, fusion) and rhythm anomalies (VT/VFib
onset) live at different time scales; a single window compromises both.
- *Safety:* neutral — still a veto; both windows stay causal.
- *Effort:* M (`build_window_index.py` already has `--dual-scale` metadata).
- *Validate:* Phase 1 false-permit on DANGEROUS beats at fixed healthy-permit,
dual-scale vs single-scale, ≥3 seeds, Wilson CIs.

**(b) GroupNorm/BatchNorm audit for tiny inference batches.**
Confirm norm layers behave identically at batch size 1 (runtime) and at training
batch size.

- *Why:* a train/inference norm mismatch silently shifts embeddings and
therefore every distance/threshold.
- *Safety:* prevents a hidden false-permit source.
- *Effort:* S.
- *Validate:* embedding parity test (batch=1 vs batch=N) already worth a unit
test in `smoke_test_layer3.py`.

**(c) Capacity sweep.**
The encoder is intentionally small (~700K params). Sweep width/depth once to
confirm we are not under-fitting healthy morphology.

- *Effort:* M. *Validate:* AUROC + false-permit vs params curve; keep smallest
model within noise of the best.

### 2.2 SSL objective (extending the B arm)

The B arm (`layer3_masked_ssl.py`, `mae_subject_contrastive`) is the ZEROSHOT
recipe. Natural follow-ons:

**(a) Masking strategy ablation.**
Compare random-patch masking vs the NERULA-style *inverse* masking, and sweep
mask ratio (0.5 / 0.75 / 0.9) and patch size.

- *Why:* ZEROSHOT/NERULA report masking scheme matters more than encoder size.
- *Effort:* M (flags already exist). *Validate:* downstream Phase 1, not
reconstruction loss.

**(b) Non-contrastive SSL as a negative-free arm against NT-Xent.**

- *Why:* avoids negative-pair pitfalls where two beats from one record are
wrongly pushed apart/together; often more stable on small ECG corpora.
- *Plan:* VICReg is the chosen first non-contrastive arm (**A1**), specified in
`VICREG_A1_IMPLEMENTATION_PLAN.md` (invariance + variance + covariance, no
negatives, no EMA teacher). BYOL/DINO stay as a later ablation if VICReg is
unstable or underperforms.
- *Safety:* neutral. *Effort:* M–L. *Validate:* add as arm A1, same scorer.

**(c) Physiology-aware positives (PhysioCLR-style).**
Choose positive pairs by RR/morphology similarity rather than only augmented
views of the same window.

- *Risk:* must **not** pull healthy and abnormal beats together — keep the
existing `same_window` default as the safety-conservative baseline.
- *Effort:* M. *Validate:* explicit check that abnormal-vs-healthy embedding
separation does not shrink.

### 2.3 Baseline / scoring model

**(a) Covariance regularisation sweep for Mahalanobis.**
Compare current empirical shrinkage vs Ledoit-Wolf / OAS, and diagonal-only in
the small-calibration regime.

- *Why:* directly addresses the "sensitive with few healthy beats" issue.
- *Effort:* S–M (sklearn covariance estimators). *Validate:* per-record
threshold stability + false-permit, stratified by calibration size.

**(b) Mixture / multi-mode healthy model.**
Small GMM or kNN-typicality instead of a single Gaussian, for records with
multiple normal submodes (rate changes, posture).

- *Why:* single-Gaussian over-inhibits legitimate healthy variation.
- *Effort:* M. *Validate:* healthy-permit at matched false-permit; watch
overfitting on tiny calibration sets (guard with min-sample fallback →
inhibit).

**(c) Per-subject score standardisation.**
Convert raw distance to a per-record z-score / rank before thresholding so a
single global operating point transfers across records.

- *Why:* reduces per-record threshold variance flagged in the review doc.
- *Effort:* M. *Validate:* cross-record threshold transfer experiment.

### 2.4 Threshold selection & calibration

**(a) Conformal / distribution-free thresholds with guaranteed healthy-permit.**
*(Implemented — now the default.)* `--threshold-method conformal` is the default
main-decision threshold in both `run_window_validation.py` and
`run_beat_validation.py`, with the healthy false-inhibit budget set by
`--conformal-alpha` (default 0.10). Infeasible alpha for the calibration size
fails safe to inhibit. Both scripts now emit `threshold_coverage.csv` (achieved
vs target healthy false-inhibit, per record and overall, with Wilson CIs). The
legacy quantile method stays available via `--threshold-method healthy_quantile`,
and Phase 1 still reports both.

- *Why:* gives a *stated* false-inhibit budget on healthy data instead of a bare
quantile; easier to defend to a supervisor.
- *Safety:* strong fit — bounds therapy loss while staying healthy-only.
- *Remaining:* coverage calibration plot (target vs achieved) across the real
datasets on the cluster; sweep alpha for the operating-point curve.

**(b) Online / drifting recalibration.**
Slow EMA update of the healthy baseline during confirmed-healthy runtime, with a
freeze-on-uncertainty rule.

- *Why:* animal sessions drift (anesthesia depth, electrode contact).
- *Risk:* must never adapt toward abnormal morphology → only update on
high-confidence healthy windows, and inhibit while adapting.
- *Effort:* L. *Validate:* simulated drift on human data before any animal use.

**(c) Signal-quality gate before scoring.**
Cheap SQI (flatline, saturation, HF-noise, lead-off) that forces inhibit and
skips the encoder when the window is untrustworthy.

- *Why:* keeps garbage windows out of both calibration and scoring; a classic
cause of false permits.
- *Effort:* S–M. *Validate:* inject noise-stress records; confirm they inhibit.

### 2.5 Temporal aggregation (multi-beat veto)

**(a) Score smoothing / hysteresis.**
Aggregate the last *k* beat scores (EMA or "inhibit if ≥ m of last n") instead of
one independent decision per beat.

- *Why:* single-beat decisions are noisy; VT/VFib is a *run*, not one beat. This
is the beat-level analogue of the Layer 2 `min5of7` phase policy already used
in `reports/layer2_policy`.
- *Safety:* asymmetric — bias toward faster inhibit, slower re-permit.
- *Effort:* M. *Validate:* onset-detection latency vs false-permit; reuse the
Layer 2 policy-sweep methodology.

### 2.6 Fusion with Layers 1–2

**(a) Keep AND as the safety default; study learned gating offline only.**
A monotone/learned combiner over (Layer2 score, Layer3 score) *may* beat plain
AND, but a learned fusion that can *increase* permit is a safety regression risk.

- *Rule:* any fusion must be provably ≤ the permit set of `AND`.
- *Effort:* M. *Validate:* conditional analysis already in
`compare_layer2_layer3.py` (`layer3_on_layer2_permitted_beats`) — does Layer 3
catch abnormal beats Layer 2 permits, without new false permits?

### 2.7 Multi-lead (C arm, deferred)

**(a) Lead-agnostic encoder / lead dropout.**
Train so the encoder tolerates single-lead runtime even if multi-lead is used
offline (rat deployment is likely single-lead).

- *Effort:* L. *Status:* appendix per the plan; only pursue if single-lead
results plateau.

### 2.8 Runtime / deployment

**(a) Latency + footprint budget.**
Measure encode+score time at batch=1 on CPU; add ONNX/TorchScript export and
optional int8 for the external-server path.

- *Why:* Layer 3 must fit the "optional advisory server, fail → inhibit" model.
- *Effort:* M. *Validate:* p95 latency under the stimulation-decision deadline;
parity check exported vs eager embeddings.

**(b) Explicit failure-path tests.**
Unit tests that a missing/corrupt checkpoint, NaN score, and server timeout all
resolve to inhibit (not random fallback).

- *Effort:* S. *Validate:* extend `smoke_test_layer3.py`; assert
`--no-random-fallback` behaviour.

### 2.9 Evaluation methodology

- Always report **false-permit on DANGEROUS with Wilson CIs**, not just AUROC.
- **Worst-record / worst-fold** rows, not only means.
- **Group by config** (window length, causal mode, quantile, lookahead) so
offline gains are not sold as deployable.
- **Seed variance** (≥3 seeds) for every learned arm.

Most of this is already wired via `layer3_group_metrics.py` and the Phase 1
outputs; the improvement is to make these the *default* reported tables.

---

## 3. Prioritised shortlist

**Do next (low risk, directly targets a measured weakness, testable now):**

1. ~~Conformal threshold as default + coverage report (§2.4a).~~ **Done** —
  default `--threshold-method conformal` + `threshold_coverage.csv`. Remaining:
   coverage plots + alpha sweep on real data (cluster).
2. Signal-quality inhibit gate (§2.4c) and explicit failure-path tests (§2.8b).
3. Mahalanobis covariance-regulariser sweep (§2.3a).
4. Multi-beat hysteresis veto (§2.5a), reusing the Layer 2 policy methodology.

**Research track (needs cluster + more validation):**

1. Masking-strategy and non-contrastive SSL ablations (§2.2a, §2.2b).
2. Multi-scale embedding (§2.1a).
3. Mixture / per-subject-standardised scoring (§2.3b, §2.3c).
4. Online recalibration (§2.4b) — only after human-data drift simulation.

**Deferred:** multi-lead C arm (§2.7), learned fusion (§2.6a, offline study only).

---

## 4. Definition of done for any adopted change

A change graduates from "candidate" to "adopted" only when:

- it beats the current baseline on **false-permit at matched healthy-permit**,
with non-overlapping Wilson CIs, across ≥3 seeds and the worst record;
- it preserves every guardrail in §0 (causal, healthy-only calibration,
fail-to-inhibit);
- `smoke_test_layer3.py` still passes end-to-end;
- the winning config is written to `run_config.json` with the git SHA.

Until then it stays in this backlog.