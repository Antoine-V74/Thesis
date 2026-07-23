# Layer 3 — Tier 1: normalizing-flow density scorer

**Attacks the proven lever (the scorer), not the encoder.** The Arm C ladder showed the
encoder is not the bottleneck; the per-record density model is. Mahalanobis assumes each
patient's healthy embedding cloud is a single Gaussian; healthy ECG has sub-modes (rate,
posture, respiration). A **normalizing flow models an arbitrary multimodal healthy
density**, gives an exact likelihood (→ conformal on NLL), and is a **scorer swap** — so
it runs on the **A0 handcrafted features** too, which is the decisive test.

Date: July 2026. Code: `Layer3/pipeline/layer3_flow_scorer.py` (**implemented + self-test
passing**). Parent memo: plan file (Tier 1/Tier 3).

---

## 0. Status

| Piece | Status |
| --- | --- |
| `EmbeddingFlowBaseline` (RealNVP + NLL score, `fit`/`score` like Mahalanobis) | **Implemented** |
| Population-then-personalize (AltUB: adapt only base distribution) | **Implemented** (`set_population_state` / `export_population_state`) |
| Local self-test (separates anomalies in the gap of a **bimodal** healthy cloud) | **Passing** — healthy NLL 13.7 vs anomaly NLL 17.9 |
| Harness wiring (`--phase1-scorers flow`) | **Not wired** — precise recipe in §3; needs a cluster smoke to verify |
| Cluster ablation vs Mahalanobis/kNN on A0 + best L3 arm | **Pending** |

The self-test is deliberately bimodal because that is the case a single Gaussian
mishandles and the flow should win — passing it is evidence the flow adds what Mahalanobis
structurally cannot.

---

## 1. Argument
1. Scorer is the lever (ladder result). Improving the *density model* is aimed at the
   right target, unlike any encoder arm.
2. Flow → exact likelihood → NLL drops straight into the existing conformal α=0.10
   machinery (score = NLL, higher = more anomalous).
3. **Arm-agnostic ⇒ the decisive experiment: run the flow on A0 features.** If
   flow-on-A0 beats Mahalanobis-on-A0, that proves the scorer was the bottleneck — a
   thesis-level result needing no learned encoder.

## 2. The data-scarcity catch + mitigation (AltUB)
Per-record healthy calibration has few beats → a per-record flow overfits. Mitigation
(**implemented**): train ONE flow on **pooled** healthy embeddings (gold excluded), then
per record adapt **only** the diagonal-Gaussian base distribution (mean / log-std) on that
record's healthy beats — `set_population_state()` freezes the coupling layers. Borrows
statistical strength, stays personalized and label-free. `min_fit` refuses to fit a flow on
fewer than 8 healthy beats (fall back to Mahalanobis there).

## 3. Harness wiring recipe (the remaining, un-blind step)
Three points, mirroring how `mahalanobis`/`knn` are handled:
1. **`layer3_flow_scorer.py`** — done. Same `fit(healthy)/score(all)` contract; exposes
   `n_fit_` / `n_outlier_removed_` so it slots into the pruning wrapper.
2. **`run_window_validation.py` → `fit_baseline_with_pruning(..., anomaly_model=scorer)`** —
   add a `flow` branch returning `EmbeddingFlowBaseline` (optionally seeded with a pooled
   `population_state`).
3. **`layer3_validation/layer3_phase1_eval.py`** (~line 428) — add `flow` to the allowed
   scorer set `{"mahalanobis","knn"}`.
Do this on a box with the cluster embeddings and run a 1-record smoke before trusting it —
I did **not** blind-edit the core eval path.

## 4. Recommended experiment (bounded)
- **Primary:** flow vs Mahalanobis vs kNN on **A0 features**, MIT-BIH gold, identical
  conformal α=0.10 — a clean scorer ablation isolating scorer-vs-representation.
- **Secondary:** same on the best L3 arm (C, mahalanobis).
- **Population-then-personalize:** pooled-flow pretrain on gold-excluded healthy embeddings,
  AltUB per record.

## 5. Papers
- **Rudolph et al., 2021 — DifferNet** (WACV). Flows on frozen features for AD.
- **Gudovskiy et al., 2022 — CFLOW-AD** (WACV; arXiv 2107.12571). Conditional NF, real-time, small.
- **Yu et al., 2021 — FastFlow** (arXiv 2111.07677). Efficient 2D flows.
- **Kim et al., 2022 — AltUB** (arXiv 2210.14913). Update a flow's base distribution — the
  personalization mechanism used here.
- **Dai & Chen, 2022 — GANF** (ICLR; arXiv 2202.07857). Flows for label-free time-series AD.
- **Ibrahim et al., 2026 — Lightweight UAD** (repo PDF). NF among UAD filters under wearable
  compute limits → feasibility + rat/low-power framing.
- **Kamoi & Kobayashi, 2020** (arXiv 2003.00402). Why Mahalanobis works — the baseline to beat.

## 6. Guardrail
Exploratory; A0 remains the deployable baseline; report false-permit + CAV. The flow only
produces a score — threshold stays conformal on healthy scores, never abnormal labels.
