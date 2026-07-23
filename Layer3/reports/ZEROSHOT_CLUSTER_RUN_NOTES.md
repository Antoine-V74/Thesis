# Layer 3 — Cluster Run Notes (pretrain + Phase 1 eval)

Reproducible command sequence for **A0 / A / A1 / B (+ B1)** arms.  
VICReg **A1** details: `VICREG_A1_IMPLEMENTATION_PLAN.md`.  
Masked family **B / B1**: design freeze `LAYER3_ARM_B_B1_SPEC.md`; full status + caveats `LAYER3_COMPLETE_STATUS_B.md` / `LAYER3_COMPLETE_STATUS_B1.md`.  
Scientific framing: `LAYER3_SCIENTIFIC_REVIEW_BRIEF.md`.  
**Pre-registration + Step 0/0.5 + CAV metrics:** `LAYER3_PHASE1_PREREGISTRATION.md` (lock before campaign).

## Protocol freeze (do not change mid-campaign)

```text
SSL pretrain:     8 s @ 125 Hz
Phase 1 primary:  8 s beat-sync @ 125 Hz  (Layer 3 carries rhythm itself)
Morphology ablation (optional): 1 s
Dual-scale:       FUTURE only (not before first A0/A/A1/B campaign)
Scorer (all arms): optional L2/PCA → Mahalanobis or kNN → conformal α=0.10
Calibration:      per-record healthy only (ZEROSHOT-style)
A0 control:       Layer 2 features + same scorer (NOT full Layer 2 hard-rule gate)
VICReg expander:  512,512,512  (not 64,64 — smoke-test only)
Checkpoints:      --no-random-fallback; verify encoder_info.json checkpoint_loaded
Pretrain/eval:    exclude gold eval records from SSL pretraining for the clean primary claim
Seeds:            ≥3 per objective
Triggers:         oracle (upper bound) AND layer1_adaptive_gated (pipeline claim)
CIs:              beat Wilson + record-level / record-bootstrap
Extra metrics:    false-permit stratified by danger type; A0↔L3 score corr + inhibit overlap
Scope:            Layer 3 = human research / PhysioNet proxy only (not animal-ready)
```

Downstream decision is fixed for all arms. Layer 3 remains a **veto only**.  
Reconstruction error from the B-arm decoder is **never** the anomaly score.  
Layer 2 and Layer 3 are **independent by construction**; AND only at final permit.  
Primary eval is **8 s** (older 1 s primary docs superseded; 1 s = ablation only).

---

## 0. GATE — danger-event count + pre-registration (before any full campaign)

**Make-or-break.** Full spec: [`LAYER3_PHASE1_PREREGISTRATION.md`](LAYER3_PHASE1_PREREGISTRATION.md)
(Step 0, Step 0.5 stratification, CAV / score-correlation definitions, locked
primary metric paragraph).

```bash
python Layer3/tools/count_transition_records.py \
  --data-dir data \
  --out-dir Results/layer3/transition_analysis
```

Then write `Results/layer3/transition_analysis/MANIFEST.md`:

1. Paste the **§1 pre-registration paragraph** from `LAYER3_PHASE1_PREREGISTRATION.md`  
2. Freeze pretrain vs eval record lists + overlap policy  
3. Fill **n_transition_records**, **n_DANGEROUS** totals + per-record min/median/max  
4. Fill **Step 0.5** subtype counts (`danger_rhythm` / `danger_morphology` / `danger_noise`)  
5. Write **Go / No-go** in one sentence + date  

**Go:** non-degenerate record-bootstrap CI possible; strongest claims on disjoint split (or labelled exploratory).  
**No-go:** tiny/leaky danger set → headline = clean protocol + data limitation (still valid MSc).

After Phase 1: stratified false-permit + **CAV** (L3 inhibits | A0 permits, DANGEROUS) +
healthy score correlation — formulas in the pre-registration doc.

The inventory also writes pilot allowlists (after re-run or from gold CSV):

```text
Layer3/reports/pilot_lists/   (tracked in git — use these on the cluster)
  pilot_primary_mitbih_gold.csv      ← PRIMARY Phase 1 allowlist (13 records)
  pilot_secondary_creighton_gold.csv
  pilot_secondary_ltafdb_gold.csv    ← secondary only (AF-heavy)
  pilot_dataset_roles.csv
  MANIFEST.md                        ← SOFT GO decision
```

Full inventory CSVs also under `Results/layer3/transition_analysis/` locally (gitignored).

---

## 0a. L2 vs A0 vs A (ready bash jobs)

Three-way compare (full Layer 2 gate vs Phase 1 A0 vs Arm A): see
[`LAYER3_L2_A0_A_COMPARE.md`](LAYER3_L2_A0_A_COMPARE.md) and scripts in
[`cluster_jobs/`](cluster_jobs/). Layer 2 accepts the same gold
`--records-csv` as Phase 1.

## 0b. MIT-BIH one-seed pilot (do this before full multi-seed)

**Goal:** cheap sanity check on the clean primary cohort only.

```bash
# Window index for MIT-BIH only (or reuse a full index and filter later)
python Layer3/tools/build_window_index.py \
  --data-dir data \
  --datasets mit_bih_arrhythmia \
  --out-csv Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv \
  --window-s 8 --stride-s 2 --target-fs 125 --lead-index 0

python Layer3/tools/pretrain_encoder.py \
  --window-index Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv \
  --checkpoint-dir Results/layer3/pretrain/ntxent_mitbih_seed0 \
  --ssl-objective ntxent \
  --positive-mode same_window \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --augment-fs 125 \
  --epochs 100 --batch-size 256 --lr 3e-4 \
  --seed 0 --device cuda

python Layer3/validation/run_beat_validation.py \
  --data-dir data \
  --datasets mit_bih_arrhythmia \
  --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --checkpoint Results/layer3/pretrain/ntxent_mitbih_seed0/encoder_last.pt \
  --out-dir Results/layer3/validation/pilot_mitbih_ntxent_seed0_8s \
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

`--records-csv` keeps only the **13 gold MIT-BIH transition records**.  
Do **not** headline LTAFDB until after this pilot looks sane.

---

## 1. Build 8 s / 125 Hz window index (pretraining)

```bash
python Layer3/tools/build_window_index.py \
  --data-dir data \
  --datasets mit_bih_arrhythmia normal_sinus_rhythm supraventricular_arrhythmia long_term_atrial_fibrillation noise_stress_test st_petersburg_12lead atrial_fibrillation malignant_ventricular_arrhythmia creighton_vfib \
  --out-csv Results/layer3/window_index/layer3_windows_8s_125hz.csv \
  --window-s 8 \
  --stride-s 2 \
  --target-fs 125 \
  --lead-index 0
```

Output CSV must include `record_id`, `signal_path`, `start_idx`, and `n_samples`.

---

## 2. Train A — NT-Xent / CLOCS-inspired

```bash
for SEED in 0 1 2; do
  python Layer3/tools/pretrain_encoder.py \
    --window-index Results/layer3/window_index/layer3_windows_8s_125hz.csv \
    --checkpoint-dir Results/layer3/pretrain/ntxent_seed${SEED} \
    --ssl-objective ntxent \
    --positive-mode same_window \
    --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
    --augment-fs 125 \
    --epochs 100 \
    --batch-size 256 \
    --lr 3e-4 \
    --num-workers 4 \
    --seed ${SEED} \
    --device cuda
done
```

Optional: `--healthy-only` if pretraining should avoid abnormal/noise morphology.

---

## 3. Train A1 — VICReg (non-contrastive)

```bash
for SEED in 0 1 2; do
  python Layer3/tools/pretrain_encoder.py \
    --window-index Results/layer3/window_index/layer3_windows_8s_125hz.csv \
    --checkpoint-dir Results/layer3/pretrain/vicreg_seed${SEED} \
    --ssl-objective vicreg \
    --vicreg-expander-dims 512,512,512 \
    --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
    --augment-fs 125 \
    --epochs 100 \
    --batch-size 256 \
    --lr 3e-4 \
    --num-workers 4 \
    --seed ${SEED} \
    --device cuda
done
```

---

## 4. Train B (primary) + B1 (ablation) — masked-reconstruction family

Full rationale + per-arm data policy: [`LAYER3_ARM_B_B1_SPEC.md`](LAYER3_ARM_B_B1_SPEC.md).
Deep dive / caveats: [`LAYER3_COMPLETE_STATUS_B.md`](LAYER3_COMPLETE_STATUS_B.md),
[`LAYER3_COMPLETE_STATUS_B1.md`](LAYER3_COMPLETE_STATUS_B1.md).

**B (primary) = masked recon + non-contrastive same-window consistency.**
Prefer `--healthy-only` (reconstruction on mixed data can normalize pathology).

```bash
for SEED in 0 1 2; do
  python Layer3/tools/pretrain_encoder.py \
    --window-index Results/layer3/window_index/layer3_windows_8s_125hz.csv \
    --checkpoint-dir Results/layer3/pretrain/mae_consistency_seed${SEED} \
    --ssl-objective mae_consistency \
    --healthy-only \
    --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
    --mask-ratio 0.75 \
    --mask-patch-size 25 \
    --consistency-lambda 1.0 \
    --vicreg-expander-dims 512,512,512 \
    --epochs 100 \
    --batch-size 256 \
    --lr 3e-4 \
    --num-workers 4 \
    --seed ${SEED} \
    --device cuda
done
```

**B1 (ablation) = masked recon + subject/record contrastive** (Yu / ZEROSHOT-style;
subject positives can pull healthy+abnormal from one record together — kept as
an ablation, preferably `--healthy-only`).

```bash
for SEED in 0 1 2; do
  python Layer3/tools/pretrain_encoder.py \
    --window-index Results/layer3/window_index/layer3_windows_8s_125hz.csv \
    --checkpoint-dir Results/layer3/pretrain/mae_subject_contrastive_seed${SEED} \
    --ssl-objective mae_subject_contrastive \
    --healthy-only \
    --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
    --mask-ratio 0.75 \
    --mask-patch-size 25 \
    --subject-contrastive-lambda 0.3 \
    --subject-col record_id \
    --epochs 100 \
    --batch-size 256 \
    --lr 3e-4 \
    --num-workers 4 \
    --seed ${SEED} \
    --device cuda
done
```

Decoder (and B's expander) discarded at validation time. Anomaly score = encoder
embedding distance only, never reconstruction error.

---

## 5. Beat-synchronous Phase 1 safety tables (primary: 8 s)

For each checkpoint:

```bash
# Repeat with vicreg_seed${SEED}/encoder_last.pt, mae_consistency_seed${SEED}/encoder_last.pt, etc.
CKPT=Results/layer3/pretrain/ntxent_seed0/encoder_last.pt
OUT=Results/layer3/validation/beat_phase1_ntxent_seed0_8s
```

`--phase1-arms a0,layer3` compares Layer 2 features (A0) vs the **checkpoint encoder** (A, A1, or B depending on which `CKPT` you pass). Run §5 once per seed/objective.

```bash
python Layer3/validation/run_beat_validation.py \
  --data-dir data \
  --datasets mit_bih_arrhythmia \
  --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --checkpoint ${CKPT} \
  --out-dir ${OUT} \
  --mode oracle \
  --window-s 8 \
  --target-fs 125 \
  --causal-window \
  --lookahead-ms 100 \
  --per-record-calibration \
  --guard-s 8 \
  --l2-normalize-embeddings \
  --pca-dim 32 \
  --phase1-eval \
  --phase1-arms a0,layer3 \
  --phase1-scorers mahalanobis,knn \
  --threshold-method conformal \
  --conformal-alpha 0.10 \
  --offline-danger-fpr-target 0.02 \
  --no-random-fallback \
  --device cuda
```

Default decision: conformal α = 0.10 (infeasible α → inhibit all).  
Phase 1 also reports `healthy_quantile` for comparison.

Primary Phase 1 uses **MIT-BIH gold only**. After the primary run is sane,
repeat with `--records-csv Layer3/reports/pilot_lists/pilot_secondary_creighton_gold.csv`
for VF robustness, and only then use LTAFDB as AF-heavy secondary evidence.

Key outputs:

```text
threshold_coverage.csv
phase1_per_beat.csv
phase1_thresholds.csv
phase1_offline_operating_points.csv   # offline only — uses DANGEROUS labels
phase1_metrics_overall.csv
phase1_metrics_by_dataset.csv
phase1_metrics_bootstrap.csv          # HEADLINE false-permit CI (record-cluster bootstrap)
phase1_metrics_by_record.csv          # per-record false permit (danger-mass concentration)
phase1_metrics_by_danger_subtype.csv  # rhythm vs morphology vs noise (C3)
phase1_cav_l2_l3.csv                  # A0↔L3 CAV + healthy score correlation (C2); needs both arms
encoder_info.json                     # must show checkpoint_loaded: true
```

**Headline uncertainty is now automated:** use `phase1_metrics_bootstrap.csv`
(record-cluster bootstrap) as the reported CI, not the beat-level Wilson columns.
`phase1_cav_l2_l3.csv` answers the L2 ⊥ L3 (C2) question and only appears when
`--phase1-arms a0,layer3`. Bootstrap resamples: `--phase1-bootstrap-n` (default 2000).

**Morphology ablation (1 s):** repeat §5 with `--window-s 1 --guard-s 1` and `--out-dir` suffix `_1s`. Secondary only.

---

## 6. Trigger-mode (pipeline-relevant) follow-up

```bash
python Layer3/validation/run_beat_validation.py \
  --data-dir data \
  --datasets malignant_ventricular_arrhythmia creighton_vfib \
  --checkpoint ${CKPT} \
  --out-dir Results/layer3/validation/trigger_phase1_ntxent_seed0_8s \
  --mode layer1_adaptive_gated \
  --window-s 8 \
  --target-fs 125 \
  --causal-window \
  --lookahead-ms 100 \
  --per-record-calibration \
  --guard-s 8 \
  --phase1-eval \
  --phase1-arms a0,layer3 \
  --phase1-scorers mahalanobis,knn \
  --conformal-alpha 0.10 \
  --offline-danger-fpr-target 0.02 \
  --no-random-fallback \
  --device cuda
```

Emphasize `layer1_adaptive_gated` for pipeline / stimulation claims. Oracle = upper bound.

---

## 7. Minimum checks before trusting results

```bash
python -m py_compile \
  Layer3/tools/pretrain_encoder.py \
  Layer3/pipeline/layer3_masked_ssl.py \
  Layer3/pipeline/layer3_vicreg.py \
  Layer3/validation/run_beat_validation.py

python Layer3/tools/smoke_test_layer3.py
```

Run torch-dependent checks on the cluster if local Windows lacks CUDA.

**Order:** §0 gate → §1 index → one-seed NT-Xent pilot (§2+§5) → inspect → full 3 seeds × A/A1/B only if §0 passed.
