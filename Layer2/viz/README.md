# Layer 2 Figures

Presentation-friendly plots and animations for thesis slides and analysis.

Run validation first, then generate all figures:

```powershell
# After cross-dataset or beat-sync validation
.\.venv\Scripts\python.exe Layer2\viz\make_all.py `
    --per-beat Results\layer2\cross_dataset_causal_100ms\per_beat.csv `
    --out-dir Results\layer2\viz
```

## What each script produces

| Script | Output | Purpose |
|--------|--------|---------|
| `plot_dataset_performance.py` | `dataset_summary.png`, `arrhythmia_breakdown.png`, `worst_records.png` | Per-dataset metrics, arrhythmia classes, outlier records |
| `plot_pareto.py` | `pareto_operating_curves.png`, `pareto_frontier.png` | Safety trade-off curves and post-hoc Pareto frontier |
| `plot_feature_auroc.py` | `feature_auroc.png`, `feature_group_auroc.png`, `top_deviators.png` | Feature separability and inhibit drivers |
| `animate_beat_gate.py` | `layer2_gate_animation.gif` | Step-by-step gate walkthrough (synthetic demo) |
| `make_all.py` | All of the above | One-command figure generation |

## Inputs

- **`per_beat.csv`** — from `validation/run_beat_validation.py` or `validation/run_cross_dataset_validation.py`
- **`pareto_posthoc.csv`** (optional) — from `validation/run_pareto_sweep.py posthoc`
- **`beat_features.csv`** (optional) — raw feature CSV for richer AUROC (e.g. INCART analysis)

## Design

Shared colors and styling live in `plot_style.py`. Figures are saved to `Results/layer2/viz/` by default.

Legacy plot scripts from the old `tools/` and `reports/` folders are archived under `Layer2/archive/old_reports/`.
