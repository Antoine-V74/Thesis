# Layer 3 validation add-ons

These files add the missing validation utilities without rewriting the existing
Layer 3 core files.

Existing files kept/reused:
- `layer3_encoder.py`
- `layer3_augmentations.py`
- `layer3_anomaly.py`

Added / patched files:
- `build_window_index.py`
- `layer3_embedding_mahalanobis.py`
- `layer3_validation_utils.py`
- `layer3_pretrain.py` (patched for determinism, mmap, logger, `encoder_last.pt`)
- `layer3_validate.py`
- `layer3_validate_beat_sync.py`
- `compare_layer2_layer3.py`
- `smoke_test_layer3.py` (synthetic end-to-end test)

## Important behavior

- Layer 3 is an anomaly-veto validation layer, not a clinical arrhythmia
  classifier.
- It never commands stimulation by itself. Final therapy rule (deployment):
  permit iff Layer 1 trigger reliable AND Layer 2 permits AND Layer 3 permits.
- Uncertainty or runtime failure must inhibit.
- Calibration windows are marked `calibration_no_stim` and excluded from therapy
  metrics. They never trigger stimulation regardless of score.
- A **guard region** (default = `--window-s`) is enforced between calibration
  and test windows so no test window overlaps any calibration window. Guarded
  rows are labelled `guard_excluded`.
- The default beat-sync mode is annotated-beat / oracle upper-bound validation;
  this is offline only and not deployable.
- `--mode layer1_adaptive_gated` scores one Layer 3 decision per accepted
  adaptive Layer 1 trigger. Oracle annotations are used only afterward to label
  those triggers for offline metrics. Unmatched triggers are treated as
  non-healthy for conservative safety-gate reporting. The current Layer 1
  trigger helper follows the existing zero-phase offline validation filter, so
  this is a runtime-style comparison mode, not an embedded deployment trace.
- Centered beat windows are offline / non-causal. Use `--causal-window` to
  simulate a strict real-time gate.
- `build_window_index.py` emits **both** validation columns (`start_sample`,
  `end_sample`, etc. in native fs) **and** the pretraining columns required by
  `layer3_pretrain.py`: `record_id`, `signal_path`, `start_idx`, `n_samples`
  (in cached fs).

## New flags worth knowing

| flag | scripts | meaning |
|---|---|---|
| `--seed N`              | pretrain / validate / beat-sync | deterministic seed for sampler, dataset, and torch |
| `--deterministic`       | pretrain / validate / beat-sync | enable PyTorch deterministic algorithms |
| `--guard-s S`           | validate / beat-sync            | seconds of guard between calibration and test (default = `--window-s`) |
| `--num-workers N`       | pretrain                        | DataLoader workers; default 0 (CPU laptop / Windows safe) |
| `--no-mmap`             | pretrain                        | disable mmap loading of cached `.npy` records |
| `--healthy-only`        | pretrain                        | restrict pretraining to windows with `is_healthy_window=True` if column present |
| `--max-windows N`       | pretrain                        | cap pretraining windows for smoke testing |
| `--max-records N`       | validate / beat-sync            | cap records (preserves per-record calibration math); preferred over `--max-windows` |
| `--causal-window`       | beat-sync                       | strict real-time beat window (no future samples) |
| `--mode oracle`         | beat-sync                       | annotated-beat offline upper-bound mode; not runtime deployable |
| `--mode layer1_adaptive_gated` | beat-sync                | score accepted adaptive Layer 1 triggers; annotations used only for offline labels |
| `--annotation-match-tolerance-s S` | beat-sync            | tolerance for offline annotation labels in Layer 1 trigger mode |
| `--no-random-fallback`  | validate / beat-sync            | fail if no real encoder is available, instead of using the random smoke-test encoder |
| `--merge-tolerance-s S` | compare                         | optional nearest-neighbor Layer 2/Layer 3 beat matching tolerance in seconds |
| `--merge-tolerance-samples N` | compare                  | optional nearest-neighbor beat matching tolerance in samples |

Output column naming is consistent with Layer 2:

- `decision` ∈ {`permit`, `inhibit`, `calibration_no_stim`,
  `unscored_insufficient_healthy_calibration`}
- `split` ∈ {`fit`, `val`, `calibration_excluded`, `guard_excluded`, `test`,
  `unscored_insufficient_healthy_calibration`}
- `anomaly_score`, `threshold`, `score_over_threshold_ratio` are present as
  aliases for the Mahalanobis-specific columns so downstream comparisons stay
  generic.

## Commands

```bash
python Layer3/build_window_index.py \
  --data-dir data \
  --datasets mitdb \
  --out-csv Results/layer3_validation/mitdb_windows.csv \
  --window-s 5 \
  --stride-s 1
```

```bash
python Layer3/layer3_pretrain.py \
  --window-index Results/layer3_validation/mitdb_windows.csv \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 4 \
  --seed 0 \
  --checkpoint-dir Results/layer3_pretrain
```

```bash
python Layer3/layer3_validate.py \
  --data-dir data \
  --datasets mitdb \
  --window-index Results/layer3_validation/mitdb_windows.csv \
  --checkpoint Results/layer3_pretrain/encoder_last.pt \
  --out-dir Results/layer3_validation/window_level \
  --per-record-calibration \
  --seed 0
```

```bash
python Layer3/layer3_validate_beat_sync.py \
  --data-dir data \
  --datasets mitdb \
  --checkpoint Results/layer3_pretrain/encoder_last.pt \
  --out-dir Results/layer3_validation/beat_sync \
  --per-record-calibration \
  --seed 0
```

Adaptive Layer 1 trigger mode:

```bash
python Layer3/layer3_validate_beat_sync.py \
  --data-dir data \
  --datasets mitdb \
  --checkpoint Results/layer3_pretrain/encoder_last.pt \
  --out-dir Results/layer3_validation/beat_sync_layer1_adaptive \
  --mode layer1_adaptive_gated \
  --causal-window \
  --per-record-calibration \
  --seed 0
```

```bash
python Layer3/compare_layer2_layer3.py \
  --layer2-dir Results/final_mitbih_validation/beat_sync \
  --layer3-dir Results/layer3_validation/beat_sync \
  --out-dir Results/layer3_validation/comparison
```

If Layer 2 beat-sync rows are accepted Layer 1 triggers rather than the exact
same annotated beat samples used by Layer 3 oracle mode, use a bounded tolerant
merge:

```bash
python Layer3/compare_layer2_layer3.py \
  --layer2-dir Results/final_mitbih_validation/beat_sync \
  --layer3-dir Results/layer3_validation/beat_sync \
  --out-dir Results/layer3_validation/comparison \
  --merge-tolerance-s 0.10
```

## Quick smoke test (no real data needed)

Runs the entire pipeline against synthetic WFDB records in a temp directory.
Takes ~30 s on CPU.

```bash
python Layer3/smoke_test_layer3.py
```

Set `LAYER3_SMOKE_KEEP=1` to keep the temp workspace for inspection.

## Expected outputs

`Results/layer3_validation/window_level/`
- `per_window.csv`, `metrics_overall.csv`, `metrics_by_label.csv`,
  `metrics_by_record.csv`, `embedding_scores.csv`, `thresholds.csv`,
  `false_permits_detail.csv`, `FINAL_LAYER3_SUMMARY.md`,
  `runtime_summary.json`, `encoder_info.json`, `embeddings.npy`

`Results/layer3_validation/beat_sync/`
- same set, with `per_beat.csv` instead of `per_window.csv`.

`Results/layer3_validation/comparison/`
- `combined_per_beat.csv`, `comparison_layer2_layer3.csv`,
  `final_comparison_table.csv`, `false_permits_detail.csv`,
  `FINAL_COMPARISON_SUMMARY.md`

## Dependencies

```bash
pip install wfdb torch pandas numpy scipy scikit-learn
```
