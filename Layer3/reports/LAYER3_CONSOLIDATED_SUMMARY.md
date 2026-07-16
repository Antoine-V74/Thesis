# Layer 3 — Consolidated medium summary

**One file to read** when the other Layer 3 reports feel too many.  
Last updated: July 2026. Companion short brief: `LAYER3_SUPERVISOR_SUMMARY.md`.  
Pilot decision + gold allowlists: `pilot_lists/MANIFEST.md` (and CSVs in that folder).  
Cluster commands: `ZEROSHOT_CLUSTER_RUN_NOTES.md` (§0b = MIT-BIH pilot).

---

## 1. What we are trying to achieve

**Application:** ECG-triggered cardiac assistance (cardiomyoplasty / MyoNeural Actuator).  
**Software question:** Is the current ECG safe enough to **permit** stimulation, or should we **inhibit**?

Rules that never change:

- Software can only **inhibit** (never command stimulation).  
- Uncertainty → inhibit.  
- **False permit** (stimulate during unsafe ECG) is the primary error.  
- Therapy policy for the stack: **1-in-8** (observe 7 beats, stimulation opportunity on beat 8).

```text
permit = Layer1_safe AND Layer2_safe AND (Layer3_safe OR Layer3_disabled)
```


| Layer | Job                                                   | Thesis scope                                           |
| ----- | ----------------------------------------------------- | ------------------------------------------------------ |
| 1     | R-peaks / timing / blanking                           | Animals ± humans                                       |
| 2     | Handcrafted features vs personalized healthy baseline | Pigs / humans (first real assist path)                 |
| 3     | SSL embedding vs same style of personalized baseline  | **Humans only** (PhysioNet research / optional server) |


Layer 2 and Layer 3 are **independent vetoes** (no shared feature pipeline). AND only at the final decision.

---

## 2. What Layer 3 is (and is not)

**Is:** A controlled **representation ablation** under a **fixed** safety scorer.

> Do SSL embeddings beat Layer 2 handcrafted features for a personalized permit/inhibit veto?

**Is not:** An arrhythmia classifier, a clinical VT/VF detector, or an animal-ready deployable gate.

**Runtime path (ZEROSHOT-style personalization):**

```text
Pretrain encoder on unlabeled ECG (optional, offline)
  → freeze encoder
  → per record: fit Mahalanobis/kNN on early healthy embeddings
  → conformal threshold = healthy false-inhibit budget (α ≈ 0.10)
  → score later beat-sync windows → permit / inhibit
```

DANGEROUS labels are used **offline** to measure false permits. They are **not** used to set the deploy threshold.  
Conformal does **not** guarantee danger-side performance — only a healthy-side operating budget (if exchangeability roughly holds).

**Windows:**


| Stage               | Setting           | Why                                                                   |
| ------------------- | ----------------- | --------------------------------------------------------------------- |
| SSL pretrain        | **8 s @ 125 Hz**  | Rhythm + morphology for representation learning                       |
| Phase 1 primary     | **8 s beat-sync** | Layer 3 must carry rhythm itself (independent of Layer 2 RR features) |
| Morphology ablation | **1 s**           | Local QRS; tests whether 8 s costs morphology sensitivity             |


---

## 3. Why these papers (what we took / what we did not copy)

### Personalization + scoring (the “head”)


| Paper                                                | Idea we use                                                                                  | What we do **not** claim                                     |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **Yu, 2026 (ZEROSHOT)**                              | SSL → frozen encoder → **per-subject healthy Mahalanobis**, zero disease labels for the head | Exact ViT-MAE, iEEG domain, or their seizure metrics as ours |
| **Carrera et al., 2019**                             | Per-user normal model for monitoring                                                         | Their sparse dictionary as our scorer                        |
| **Jiang et al., 2024** / **Kapsecker & Jonas, 2025** | SSL / embedding AD on ECG is feasible                                                        | Same clinical task or federated setup                        |


**Why Yu especially:** Closest template to “no arrhythmia labels at calibration + personalized distance.” Our novelty is **stimulation safety framing**, false-permit metrics, and the layered inhibit-only stack — not inventing Mahalanobis.

### SSL arms (the “encoder”)


| Arm    | Method                                                    | Papers                                                                                                             | Why this arm                                                                 | Honest gap vs paper                                                                                     |
| ------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **A0** | Layer 2 **features** + **same** Mahalanobis/kNN/conformal | de Chazal-style features; Carrera logic                                                                            | Control floor: does SSL beat interpretable features?                         | **Not** the full Layer 2 hard-rule gate                                                                 |
| **A**  | NT-Xent (SimCLR-family)                                   | **Chen et al. 2020 (SimCLR)**; **Kiyasseh et al. 2021 (CLOCS)** inspiration; augs from **Gopal et al. 2021 (3KG)** | Standard contrastive baseline for ECG SSL                                    | Default `same_window` only — **not** full CLOCS CMSC/CMLC/CMPC; single-lead                             |
| **A1** | VICReg                                                    | **Bardes et al. 2022**                                                                                             | Negative-free; may give better-conditioned embeddings for covariance scoring | Same loss family; ECG backbone + safety downstream                                                      |
| **B**  | Masked recon + subject contrastive                        | **Yu 2026**; masking spirit of **Manimaran et al. 2024 (NERULA)**                                                  | Closest to ZEROSHOT recipe                                                   | Conv inpainting, not ViT-MAE; **not** full NERULA dual pathway; recon error **never** the anomaly score |


Other literature (Pan-Tompkins, Roche/Vasilyev assist sync, Ballas/Li domain shift, Clifford SQI, Task Force HRV) motivates **Layers 1–2 and the problem**, not Layer 3 arm choice.

---

## 4. Gold transition records (“golden runs”)

### Why they exist

Personalized veto needs, **in the same recording**:

1. Enough early **healthy** beats to fit a baseline
2. Later **DANGEROUS** content to measure false permits

A pure-VF file with no sinus baseline cannot test that deployment story.

**Gold (strict):** ≥30 NORMAL beats before first DANGEROUS event (`count_transition_records.py`).

### What we found (July 2026 inventory)


| Dataset      | Gold records    | Danger beats | Role in the thesis                                     |
| ------------ | --------------- | ------------ | ------------------------------------------------------ |
| **MIT-BIH**  | **13**          | **654**      | **PRIMARY** — cleanest acute-ish transitions           |
| LTAFDB       | 28              | 5603         | **SECONDARY** — ~88% of all danger beats; **AF-heavy** |
| Creighton VF | 33              | 108          | **Robustness** — VF-oriented; small beat count         |
| Malignant VA | 0 beat-eligible | —            | Rhythm-span labels; not primary beat tables            |
| **Total**    | **74**          | **6365**     | Do **not** pool blindly                                |


**Critical honesty:** If you merge all datasets, results mostly answer “AF / AF-variability,” not “acute VT/VF.”  
State that explicitly. Headline = MIT-BIH; LTAFDB secondary with caveat.

**Code:** Phase 1 can restrict to gold MIT-BIH via  
`--records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv`.

**Soft GO:** one-seed pilot on MIT-BIH gold is justified; full multi-seed after pilot looks sane.  
MIT-BIH alone is **modest** *n* (≈13 records) — directional study, not tiny-pp claims.

---

## 5. Pretrain vs eval (what uses which data)


| Step             | Data                                    | Why                                                  |
| ---------------- | --------------------------------------- | ---------------------------------------------------- |
| SSL **pretrain** | Prefer **whole MIT-BIH** (many windows) | Encoder needs volume; no need for danger transitions |
| Phase 1 **eval** | **Gold MIT-BIH only** (13 records)      | Healthy→danger personalization story                 |
| Later robustness | Creighton (then LTAFDB secondary)       | Different rhythm modality / AF caveat                |


PhysioNet pretrain and eval may **overlap records** → do not claim “unseen patient” unless you freeze a disjoint split. Prefer disjoint when feasible; otherwise label strongest claims exploratory.

---

## 6. Ablations (condensed map)

### Must / main campaign


| Ablation                              | What varies    | Question                                 |
| ------------------------------------- | -------------- | ---------------------------------------- |
| **A0 vs A vs A1 vs B**                | Representation | Does SSL beat features? Which objective? |
| **Mahalanobis vs kNN**                | Scorer         | Distance model sensitivity               |
| **Oracle vs `layer1_adaptive_gated`** | Trigger source | Upper bound vs pipeline-relevant         |
| **MIT-BIH primary**                   | Dataset role   | Clean directional answer                 |


### High value next (after pilot)


| Ablation                    | What varies               | Question                                 |
| --------------------------- | ------------------------- | ---------------------------------------- |
| **8 s vs 1 s**              | Window                    | Rhythm vs morphology tradeoff            |
| **Danger-type reporting**   | Metric split (VT/VF/PVC…) | Does pooling hide morphology fails?      |
| **CAV**                     | Metric: P(L3 inhibits     | A0 permits, DANGEROUS)                   |
| **Creighton after MIT-BIH** | Eval set                  | Generalize to VF-heavy data?             |
| **LTAFDB secondary**        | Eval set                  | AF robustness (not headline acute story) |


### Later / backlog


| Ablation                               | Question                                                   |
| -------------------------------------- | ---------------------------------------------------------- |
| **Dual-scale (1 s + 8 s concat)**      | Fix morph+rhythm if 1 s gap is large                       |
| **Train dataset A → eval B**           | Stronger transfer (ZEROSHOT-style stress); not first pilot |
| `**same_window` vs `same_record` SSL** | CLOCS-like pairing; risk mixing healthy/abnormal           |
| `**--healthy-only` pretrain**          | Cleaner normal encoder?                                    |
| **B-arm mask ratio / inverse mask**    | NERULA/ZEROSHOT masking details                            |
| **Conformal α** (0.05 vs 0.10)         | Availability vs strictness                                 |
| **PCA on/off, dim**                    | Calibration stability                                      |


Full backlog prose: `LAYER3_ARCHITECTURE_IMPROVEMENTS.md`.

---

## 7. Generalization issues (cite these)

These are scientific limitations, not “bugs to hide.”

1. **Proxy data:** PhysioNet ≠ experimental OR / animal ECG; no stimulation artifacts in current L3 claim.
2. **AF skew:** Pooled danger mass is LTAFDB-dominated → wrong clinical narrative if not stratified by dataset role.
3. **Small independent *n* on MIT-BIH (~13 gold records):** Underpowered for tiny A0–SSL gaps; OK for directional / protocol thesis.
4. **Record overlap / leakage:** Shared pretrain+eval records can inflate SSL; freeze lists; prefer disjoint.
5. **Beat correlation:** Beat-level CIs look too tight; prefer per-record summaries (± bootstrap later).
6. **Independence L2 ⊥ L3:** True by construction (different inputs); **not** proven in error space until CAV/correlation is measured.
7. **8 s vs morphology:** Longer windows match many SSL/rhythm papers; may dilute isolated ectopy — hence 1 s / dual-scale later.
8. **Cross-dataset (A→B):** Valuable *after* within-MIT-BIH personalization works; Yu’s main transferable idea is personalized healthy Mahalanobis, not only LOSO dataset transfer.
9. **Oracle vs L1-gated:** Oracle overstates deployability; pipeline claims need Layer-1 triggers.
10. **Conformal:** Healthy false-inhibit budget ≠ danger false-permit guarantee.

---

## 8. What to run now (simple)

```text
1. Inventory done → SOFT GO (see MANIFEST)
2. Cluster §0b:
     - build MIT-BIH 8 s index
     - pretrain NT-Xent seed 0
     - Phase 1: a0 vs checkpoint, --records-csv pilot_primary_mitbih_gold.csv
     - --no-random-fallback; check checkpoint_loaded
3. If sane → multi-seed A / A1 / B on same MIT-BIH gold setup
4. Then Creighton; LTAFDB only as secondary with AF caveat
```

**Phase 1** = evaluation (same scorer, A0 vs SSL), not training.  
**Seed 0** = first reproducible training run; later seeds 1–2 for stability.

---

## 9. Document map (so you can ignore the rest)


| Need                      | File                                             |
| ------------------------- | ------------------------------------------------ |
| **This medium summary**   | `LAYER3_CONSOLIDATED_SUMMARY.md` ← you are here  |
| 2-page overview           | `LAYER3_SUPERVISOR_SUMMARY.md`                   |
| Pilot Go + numbers        | `Results/layer3/transition_analysis/MANIFEST.md` |
| Commands                  | `ZEROSHOT_CLUSTER_RUN_NOTES.md`                  |
| Locked metrics / CAV defs | `LAYER3_PHASE1_PREREGISTRATION.md`               |
| Deep C1–C5 critique notes | `LAYER3_REVIEW_WEAKNESSES_C1_C5.md`              |
| External AI review prompt | `LAYER3_SCIENTIFIC_REVIEW_BRIEF.md`              |
| Algorithm detail          | `../ALGORITHM_SUMMARY.md`                        |
| Backlog ablations         | `LAYER3_ARCHITECTURE_IMPROVEMENTS.md`            |
| Slides                    | `LAYER3_SUPERVISOR_DECK.pptx`                    |
| Archive                   | `archive/` — do not use                          |


---

## 10. Quoteable paragraph

> Layer 3 is an optional human-only research veto: a frozen SSL ECG encoder plus a per-record healthy Mahalanobis/kNN score and conformal healthy-side threshold, in the spirit of Yu’s ZEROSHOT personalization and ECG SSL work (SimCLR/CLOCS-inspired contrastive, VICReg, masked+subject). It is compared to Layer 2 features (A0) under a fixed scorer — a representation ablation for stimulation safety, not a clinical detector. Evaluation uses gold within-record healthy→danger transitions; MIT-BIH (13 gold records) is primary because pooled PhysioNet danger is AF-dominated (LTAFDB). Pretrain may use broader MIT-BIH windows; Phase 1 restricts to gold records. Planned ablations include SSL objective, window length (8 s vs 1 s), trigger mode, scorer, and later cross-dataset checks. Limits — proxy data, modest independent *n*, possible pretrain overlap, and 8 s morphology tradeoffs — are stated up front so negative or mixed SSL results remain scientifically interpretable.

---

*End of consolidated summary.*