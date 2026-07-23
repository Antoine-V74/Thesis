#!/usr/bin/env bash
# Exploratory longer rhythm context. Locked primary remains 8 s.
set -euo pipefail
cd "$(dirname "$0")/../../.."

python Layer3/tools/build_window_index.py \
  --data-dir data --datasets mit_bih_arrhythmia \
  --out-csv Results/layer3/window_index/layer3_windows_mitbih_30s_125hz.csv \
  --signal-dir Results/layer3/window_index/signals_mitbih_30s_125hz \
  --window-s 30 --stride-s 4 --target-fs 125 --lead-index 0 \
  --overwrite-signal-cache

