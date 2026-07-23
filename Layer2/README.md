# Layer 2 - Feature Safety Gate

Layer 2 computes ECG features, calibrates a healthy baseline, and returns

permit/inhibit decisions. It never commands stimulation.

**Read first:** [ALGORITHM_SUMMARY.md](ALGORITHM_SUMMARY.md) — full algorithm,

justifications, `decide()` vs `decide_hybrid()`.

## Folder Map

```text

Layer2/

  pipeline/     Core runtime code (features + decision/)

  validation/   Rerun pipeline on PhysioNet datasets -> CSV metrics

  viz/          Presentation figures and gate animation

  archive/      Old one-off experiments

```

## Core Pipeline

| Path | Purpose |

|------|---------|

| `pipeline/main_pipeline.py` | Public API: extract, calibrate, decide |

| `pipeline/stimulation_cadence.py` | Prospective 1-in-8 stimulation policy |

| `pipeline/full_features.py` | Feature assembler |

| `pipeline/decision/` | Baseline calibration + permit/inhibit logic |

| `pipeline/README.md` | Short pipeline reference |

## What to run

### Validation (reruns feature extraction + gate)

| Command | Script |

|---------|--------|

| Beat-sync (deployment-like) | `validation/run_beat_validation.py` |

| Shared validation helpers | `validation/layer2_validation_utils.py` |

| Cross-dataset | `validation/run_cross_dataset_validation.py` |

| Pareto / operating point | `validation/run_pareto_sweep.py {quick,full,posthoc}` |

| Neyman-Pearson operating point + worst-record | `validation/run_np_operating_point.py` |

| Artifact / lead-off stress test (SQI gate) | `validation/run_artifact_stress_test.py` |

| Causal lookahead sweep | `validation/run_causal_lookahead_sweep.py` |

The deployment-like scripts also report a prospective cadence mode: observe 7
unstimulated beats with a longer causal lookahead, then stimulate only the 8th
beat if at least 6 of those decisions were safe and the 7th beat was safe.

Optional `--records-csv` (columns `dataset,record`) restricts scoring to a gold
allowlist — use the same MIT-BIH gold CSV as Layer 3 Phase 1 when comparing
full Layer 2 vs A0 vs Arm A (`Layer3/reports/LAYER3_L2_A0_A_COMPARE.md`).

### Figures (CSV in -> plots out)

| Command | Script |

|---------|--------|

| All thesis figures | `viz/make_all.py` |

| Per-dataset / arrhythmia / worst records | `viz/plot_dataset_performance.py` |

| Pareto operating curves | `viz/plot_pareto.py` |

| Feature AUROC | `viz/plot_feature_auroc.py` |

| Gate walkthrough animation | `viz/animate_beat_gate.py` |

See [viz/README.md](viz/README.md) for details.

## Main commands

```powershell

# Beat-synchronous validation (conformal threshold + policy groups)

.\.venv\Scripts\python.exe Layer2\validation\run_beat_validation.py `

    --data-dir data --datasets mit_bih_arrhythmia `

    --out-dir Results\layer2_beat_validation `

    --threshold-method conformal --conformal-alpha 0.10 `

    --guard-s 5.0



# Cross-dataset validation

.\.venv\Scripts\python.exe Layer2\validation\run_cross_dataset_validation.py `

    --data-dir data --out-dir Results\cross_dataset `

    --datasets mit_bih_arrhythmia noise_stress_test supraventricular_arrhythmia



# Unified Pareto sweep

.\.venv\Scripts\python.exe Layer2\validation\run_pareto_sweep.py quick `

    --data-dir data --out-dir Results\pareto_quick_test



# Generate all presentation figures

.\.venv\Scripts\python.exe Layer2\viz\make_all.py `

    --per-beat Results\layer2\cross_dataset_causal_100ms\per_beat.csv



# Neyman-Pearson operating point + worst-record danger leak (offline, label-aware)

.\.venv\Scripts\python.exe Layer2\validation\run_np_operating_point.py `

    --per-beat Results\layer2_beat_validation\per_beat.csv `

    --out-dir Results\layer2_np_operating_point --danger-budget 0.01



# Artifact / lead-off stress test for the SQI gate

.\.venv\Scripts\python.exe Layer2\validation\run_artifact_stress_test.py `

    --data-dir data --dataset mit_bih_arrhythmia `

    --out-dir Results\layer2_artifact_stress

```

### Artifact-robustness feature flags (opt-in)

The SQI ensemble and ICD-style discriminators are OFF by default so the frozen
feature set and existing calibrations are unchanged. Enable them via
`extract_layer2_features(..., compute_sqi_ensemble=True, compute_onset_stability=True)`
and calibrate with `decision.config.hard_rules_with_extensions(include_sqi_ensemble=True)`.

**Do not flip into `DEFAULT_HARD_RULES` until the SQI flip-on gate passes**
([ALGORITHM_SUMMARY.md](ALGORITHM_SUMMARY.md) → "SQI flip-on gate"): stress-test
first → recalibrate cutoffs → only then wire into beat validation / frozen gate.
Cluster default is SQI-only: `bash Layer2/validation/run_l2_artifact_np.sh`.

