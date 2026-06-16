"""
Generate all Layer 2 presentation figures in one command.

Run after validation produces per_beat.csv:

    .venv\\Scripts\\python Layer2\\viz\\make_all.py \\
        --per-beat Results\\layer2\\cross_dataset_causal_100ms\\per_beat.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(_HERE / script), *extra]
    print(f"\n=== {script} ===")
    subprocess.run(cmd, check=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-beat", type=Path,
                   default=Path("Results/layer2/cross_dataset_causal_100ms/per_beat.csv"))
    p.add_argument("--posthoc-csv", type=Path, default=None)
    p.add_argument("--beat-features", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/viz"))
    p.add_argument("--skip-animation", action="store_true")
    args = p.parse_args(argv)

    common = ["--per-beat", str(args.per_beat), "--out-dir", str(args.out_dir)]

    _run("plot_dataset_performance.py", common)
    _run("plot_pareto.py", common + (
        ["--posthoc-csv", str(args.posthoc_csv)] if args.posthoc_csv else []
    ))
    feat_extra = common.copy()
    if args.beat_features:
        feat_extra += ["--beat-features", str(args.beat_features)]
    _run("plot_feature_auroc.py", feat_extra)

    if not args.skip_animation:
        _run("animate_beat_gate.py", ["--out", str(args.out_dir / "layer2_gate_animation")])

    print(f"\nAll figures written to: {args.out_dir}")


if __name__ == "__main__":
    main()
