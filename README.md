# Master Thesis — ECG Processing

This project is organized by safety layer.

```text
ECG Processing/
  data/       PhysioNet datasets with descriptive folder names
  Layer1/     Fast R-peak detection and rhythm supervision
  Layer2/     Handcrafted feature safety gate
  Layer3/     Learned embedding / anomaly-detection research layer
  Results/    Generated benchmark outputs and plots
```

## Layer 1

Fast deterministic timing layer.

```text
Layer1/
  pipeline/   Production code and benchmark scripts
  tools/      Optional analysis and plots
  archive/    Old detector experiments
```

Start here:

```powershell
.\.venv\Scripts\python.exe Layer1\pipeline\run_benchmark.py --data-dir data --out-dir Results\layer1_benchmark
```

## Layer 2

Interpretable feature gate.

```text
Layer2/
  pipeline/     Feature extraction + baseline gate
  validation/   Beat-sync, cross-dataset, and Pareto validation scripts
  viz/          Presentation figures and gate animation
  archive/      Scratch and old experiments
```

Start here:

```powershell
.\.venv\Scripts\python.exe Layer2\validation\run_beat_validation.py --data-dir data --datasets mit_bih_arrhythmia --out-dir Results\layer2_beat_validation
```

## Layer 3

Research layer for learned ECG embeddings/anomaly detection. Treat as optional
until Layer 1 and Layer 2 are validated.

## Data

See `data/README.md`. Old PhysioNet IDs such as `mitdb` still work as aliases,
but folders now use readable names such as `mit_bih_arrhythmia`.
