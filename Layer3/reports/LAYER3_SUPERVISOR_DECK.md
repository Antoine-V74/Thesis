# Layer 3 — Supervisor Slide Deck (outline)

Generated companion to `LAYER3_SUPERVISOR_DECK.pptx`.

---

## Slide 1 — Title

**Layer 3 — Learned ECG Anomaly Safety Veto**

Master Thesis — ECG-triggered cardiac stimulation  
Supervisor briefing | July 2026

---

## Slide 2 — Project context

- Layer 1: fast R-peak + RR supervisor (always on)
- Layer 2: handcrafted feature gate (primary deploy path)
- Layer 3: learned embedding anomaly veto (optional research)
- Final rule: permit only if L1 AND L2 AND (L3 if enabled)
- Uncertainty / failure → inhibit

---

## Slide 3 — What Layer 3 does

- Input: beat-synchronous ECG window
- Output: permit or inhibit (veto only)
- NOT a clinical classifier
- NOT required for immediate safety loop
- Detects deviation from session healthy baseline

---

## Slide 4 — Runtime pipeline

```
ECG → ECGEncoder1D → 128-d embedding → Mahalanobis/kNN → threshold → permit/inhibit
```

- ~700K param 1D ResNet
- Per-record healthy calibration
- Conformal threshold (default α=10%)
- Optional L2 + PCA(32) + Ledoit-Wolf

---

## Slide 5 — Beat-synchronous evaluation

- One beat → one decision
- Primary window: 8 s @ 125 Hz (rhythm + morphology)
- Morphology ablation: 1 s window (secondary)
- Causal + optional post-R lookahead (50–150 ms)
- Oracle vs Layer 1 gated modes

---

## Slide 6 — Encoder pretraining (SSL + supervised)

- SSL arms (A/A1/B/B1): unlabeled ECG + physiology-aware augmentations
- Arm C: supervised contrastive using public labels **at pretrain only**
- All training heads discarded → deploy is label-free one-class
- Human pretrain → frozen encoder → per-session healthy baseline re-fit

---

## Slide 7 — Phase 1 arms

| Arm | Method |
|-----|--------|
| A0 | Layer 2 handcrafted (control) |
| A | NT-Xent contrastive |
| A1 | VICReg non-contrastive |
| B | MAE + same-window consistency (primary masked) |
| B1 | MAE + subject-contrastive (ablation) |
| C | Supervised contrastive (SupCon) — labels pretrain-only |

Fixed downstream scorer isolates encoder quality. Multi-lead = future appendix, not Arm C.

---

## Slide 8 — Metrics

- **Primary:** false-permit on DANGEROUS (VT/VF/noise)
- **Secondary:** healthy-permit (availability)
- Conformal threshold + Wilson CIs
- Benign ectopy = don't-care for false-permit

---

## Slide 9 — ZEROSHOT literature

- Parallel architecture validated in iEEG (AUROC 0.954)
- Our bar: false-permit + CIs, not AUROC alone
- Personalized baseline > population model

---

## Slide 10 — Status

- Full pipeline implemented (encoder, scorer, validation, conformal threshold)
- SSL arms A / A1 / B / B1 implemented
- Arm C (SupCon supervised) implemented — labels pretrain-only, encoder-only checkpoint
- Exploratory 8 s runs: SSL not yet clearly beating A0 → motivates Arm C + locked pilot
- Smoke test covers index build, pretrain (incl. supcon), validation

---

## Slide 11 — Human → rat plan

- MIT-BIH proxy validation now
- Per-session rat calibration later
- Prospective animal validation required

---

## Slide 12 — Next steps

- Locked gold pilot: A0 / A / A1 / B (+B1 ablation) on MIT-BIH gold
- Add Arm C (SupCon): does using labels at pretrain help the representation?
- Phase 1 safety tables: false-permit + record-bootstrap CIs; CAV vs A0
- Rat baseline when data available

---

## Slide 13 — Questions

1. Healthy rat ECG access?
2. Single vs multi-lead?
3. Baseline recording conditions?
4. Induced arrhythmia data later?
5. External server acceptable?
6. False-permit budget for animals?

---

*See also: `reports/README.md`, `ALGORITHM_SUMMARY.md`, `LAYER3_ARCHITECTURE_RATIONALE.md`*
