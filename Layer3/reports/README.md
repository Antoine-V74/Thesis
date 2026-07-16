# Layer 3 documentation map

One place to find Layer 3 docs. If you are lost, start here.

```text
Lost? Read ONE medium file → LAYER3_CONSOLIDATED_SUMMARY.md
Short overview               → LAYER3_SUPERVISOR_SUMMARY.md
Pilot Go + gold allowlists   → pilot_lists/MANIFEST.md
Cluster / pilot commands     → ZEROSHOT_CLUSTER_RUN_NOTES.md  (§0b)
External AI review prompt    → LAYER3_SCIENTIFIC_REVIEW_BRIEF.md
Slides                       → LAYER3_SUPERVISOR_DECK.pptx
Full algorithm               → ../ALGORITHM_SUMMARY.md
Ablation backlog             → LAYER3_ARCHITECTURE_IMPROVEMENTS.md
```

**Protocol freeze (July 2026):** 8 s pretrain + **8 s** primary Phase 1; Layer 2 ⊥ Layer 3;
VICReg expander `512,512,512`; ZEROSHOT-style per-record healthy Mahalanobis; 1 s = ablation only.
**Pilot:** MIT-BIH gold records primary (see MANIFEST / consolidated summary).

---

## Active files (maintained)

| File | Role |
| --- | --- |
| [`LAYER3_CONSOLIDATED_SUMMARY.md`](LAYER3_CONSOLIDATED_SUMMARY.md) | **Main medium summary** — papers, gold runs, ablations, generalization, next steps (read this first) |
| [`LAYER3_SCIENTIFIC_REVIEW_BRIEF.md`](LAYER3_SCIENTIFIC_REVIEW_BRIEF.md) | Critical peer-review brief for an AI agent / external reader |
| [`LAYER3_PHASE1_PREREGISTRATION.md`](LAYER3_PHASE1_PREREGISTRATION.md) | **Locked Phase 1 protocol:** Step 0/0.5 gate, CAV metrics, pre-registration paragraph |
| [`LAYER3_REVIEW_WEAKNESSES_C1_C5.md`](LAYER3_REVIEW_WEAKNESSES_C1_C5.md) | **Canonical notes** of external review C1–C5 (power, independence, 8 s, redundancy, conformal) |
| [`LAYER3_ISSUES_AND_CODE_CHANGES.md`](LAYER3_ISSUES_AND_CODE_CHANGES.md) | **Issues ↔ mitigations ↔ proposed code** + questions before coding |
| [`LAYER3_SUPERVISOR_SUMMARY.md`](LAYER3_SUPERVISOR_SUMMARY.md) | **Text supervisor brief** — current overview, arms, metrics, status |
| [`LAYER3_SUPERVISOR_DECK.pptx`](LAYER3_SUPERVISOR_DECK.pptx) | **Main slide deck**. Regenerate: `python Layer3/tools/generate_layer3_supervisor_deck.py` |
| [`LAYER3_SUPERVISOR_DECK.md`](LAYER3_SUPERVISOR_DECK.md) | Markdown outline (auto-generated with PPTX) |
| [`../ALGORITHM_SUMMARY.md`](../ALGORITHM_SUMMARY.md) | End-to-end algorithm, metrics, file map |
| [`LAYER3_ARCHITECTURE_RATIONALE.md`](LAYER3_ARCHITECTURE_RATIONALE.md) | Beat-sync windows, causality, Mahalanobis, thresholds |
| [`README_LAYER3_VALIDATION.md`](README_LAYER3_VALIDATION.md) | CLI flags, splits, guard regions, output CSVs |
| [`ZEROSHOT_CLUSTER_RUN_NOTES.md`](ZEROSHOT_CLUSTER_RUN_NOTES.md) | Reproducible pretrain + Phase 1 eval (8 s primary, A0/A/A1/B) |
| [`LAYER3_REVIEW_AND_OPEN_ISSUES.md`](LAYER3_REVIEW_AND_OPEN_ISSUES.md) | Scientific caveats, fixed bugs, reporting rules |
| [`LAYER3_ARCHITECTURE_IMPROVEMENTS.md`](LAYER3_ARCHITECTURE_IMPROVEMENTS.md) | Research backlog (dual-scale etc.; not committed work) |
| [`VICREG_A1_IMPLEMENTATION_PLAN.md`](VICREG_A1_IMPLEMENTATION_PLAN.md) | A1 arm design reference (**implemented**) |

---

## Document tiers

```text
Tier 1 — Thesis / supervisor / external review (read these)
  LAYER3_SCIENTIFIC_REVIEW_BRIEF.md
  LAYER3_SUPERVISOR_SUMMARY.md
  LAYER3_SUPERVISOR_DECK.pptx
  ALGORITHM_SUMMARY.md
  LAYER3_ARCHITECTURE_RATIONALE.md

Tier 2 — Operational (when you run code)
  README_LAYER3_VALIDATION.md
  ZEROSHOT_CLUSTER_RUN_NOTES.md

Tier 3 — Engineering hygiene
  LAYER3_REVIEW_AND_OPEN_ISSUES.md

Tier 4 — Backlog / reference
  LAYER3_ARCHITECTURE_IMPROVEMENTS.md
  VICREG_A1_IMPLEMENTATION_PLAN.md
```

---

## Archive (do not use for meetings or thesis)

See [`archive/README.md`](archive/README.md). Superseded ZEROSHOT-only decks/summaries and internal AI prompts live there only.

| Do not use | Use instead |
| --- | --- |
| `archive/superseded/ZEROSHOT_LAYER3_SUPERVISOR_SUMMARY.md` | `LAYER3_SUPERVISOR_SUMMARY.md` |
| `archive/superseded/ZEROSHOT_LAYER3_SUPERVISOR_DECK.pptx` | `LAYER3_SUPERVISOR_DECK.pptx` |
| `archive/superseded/LAYER3_THESIS_SLIDES_DRAFT.md` | `LAYER3_SUPERVISOR_DECK.pptx` |

---

## Maintenance rules

1. **One critical review brief** — `LAYER3_SCIENTIFIC_REVIEW_BRIEF.md`
2. **One canonical text brief** — `LAYER3_SUPERVISOR_SUMMARY.md`
3. **One canonical deck** — `LAYER3_SUPERVISOR_DECK.pptx` via `generate_layer3_supervisor_deck.py`
4. **One cluster recipe** — `ZEROSHOT_CLUSTER_RUN_NOTES.md` (protocol freeze at top)
5. **Caveats in one place** — `LAYER3_REVIEW_AND_OPEN_ISSUES.md`
6. **Do not update** files under `archive/` except to add a superseded banner

---

## Related paths outside this folder

| Location | Contents |
| --- | --- |
| [`../../CLAUDE.md`](../../CLAUDE.md) | Project-wide safety rules |
| [`../../reports/ECG_SAFETY_PIPELINE_ARCHITECTURE.md`](../../reports/ECG_SAFETY_PIPELINE_ARCHITECTURE.md) | Layers 1–3 architecture diagrams |
| [`../tools/`](../tools/) | pretrain, smoke test, deck generators |
