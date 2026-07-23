#!/usr/bin/env bash
# Job 3a — Build 8 s / 125 Hz window index (MIT-BIH) for Arm A pretrain.
set -euo pipefail
cd "$(dirname "$0")/../../.."

DATA_DIR="${DATA_DIR:-data}"
OUT_CSV="${OUT_CSV:-Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv}"
SIGNAL_DIR="${SIGNAL_DIR:-Results/layer3/window_index/signals_mitbih_8s_125hz}"

mkdir -p "$(dirname "$OUT_CSV")" "$SIGNAL_DIR"

python Layer3/tools/build_window_index.py \
  --data-dir "$DATA_DIR" \
  --datasets mit_bih_arrhythmia \
  --out-csv "$OUT_CSV" \
  --signal-dir "$SIGNAL_DIR" \
  --window-s 8 --stride-s 2 --target-fs 125 --lead-index 0 \
  --overwrite-signal-cache

echo "[OK] Window index -> $OUT_CSV"
