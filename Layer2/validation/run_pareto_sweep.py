"""
Unified Pareto / operating-point sweeps for Layer 2.

Modes
-----
quick
    Fast 10-record MIT-BIH subset. Re-extracts features and runs the gate.

full
    Slow full beat-sync grid. Repeatedly runs run_beat_validation.py.

posthoc
    Seconds-fast threshold rescaling on an existing per_beat.csv.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

sys.path.insert(0, str(_HERE))

from pareto_quick import run_quick  # noqa: E402
from pareto_posthoc import sweep as sweep_posthoc  # noqa: E402


def _run(cmd: List[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        print("Command failed:\n", " ".join(cmd))
        print(p.stdout)
        print(p.stderr)
        raise SystemExit(p.returncode)


def run_full_grid(args: argparse.Namespace) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for tgt in args.targets:
        for fi_cap in args.healthy_fi_caps:
            tag = f"t{tgt:.3f}_fi{fi_cap:.2f}".replace(".", "p")
            out_dir = args.out_root / tag
            cmd = [
                sys.executable,
                "Layer2/validation/run_beat_validation.py",
                "--data-dir",
                str(args.data_dir),
                "--datasets",
                *args.datasets,
                "--out-dir",
                str(out_dir),
                "--per-record-calibration",
                "--feature-sets",
                "all",
                "hybrid_rewarming",
                "--abnormal-target-inhibit",
                f"{tgt:.4f}",
                "--max-healthy-false-inhibit",
                f"{fi_cap:.4f}",
            ]
            if args.all_use_hybrid_gate:
                cmd.append("--all-use-hybrid-gate")

            print(f"\n=== Running {tag} ===")
            _run(cmd, cwd=_ROOT)

            mpath = out_dir / "metrics_overall.csv"
            if not mpath.exists():
                continue
            m = pd.read_csv(mpath)
            sel = m[(m["feature_set"] == "all") & (m["mode"] == args.summary_mode)]
            if sel.empty:
                continue
            r = sel.iloc[0].to_dict()
            r["target_abnormal_inhibit"] = tgt
            r["healthy_fi_cap"] = fi_cap
            r["run_tag"] = tag
            rows.append(r)

    if not rows:
        raise SystemExit("No runs completed.")

    out = pd.DataFrame(rows)
    out["meets_target"] = out["abnormal_inhibit_rate"] >= out["target_abnormal_inhibit"]
    out["score"] = (
        out["healthy_permit_rate"]
        - 0.25 * (out["abnormal_inhibit_rate"] < out["target_abnormal_inhibit"])
    )
    out = out.sort_values(["meets_target", "healthy_permit_rate"], ascending=[False, False])
    out.to_csv(args.out_root / "pareto_summary.csv", index=False)

    best = out[out["meets_target"]].head(1)
    if len(best):
        print("\nBest target-meeting operating point:")
        print(best[[
            "run_tag",
            "healthy_permit_rate",
            "abnormal_inhibit_rate",
            "false_permit_rate",
            "false_inhibit_rate",
            "target_abnormal_inhibit",
            "healthy_fi_cap",
        ]].to_string(index=False))
    else:
        print("\nNo run met its abnormal-inhibit target.")

    print(f"\nWrote: {args.out_root / 'pareto_summary.csv'}")


def run_posthoc(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.per_beat} ...")
    df = pd.read_csv(args.per_beat, low_memory=False)
    df = df[(df["mode"] == args.mode_name) & (df["feature_set"] == args.feature_set)].copy()
    print(f"  {len(df)} rows, labels: {df['label'].value_counts().to_dict()}")

    results = []
    for ignore_morph in (False, True):
        results.append(
            sweep_posthoc(
                df,
                mahal_scales=args.mahal_scales,
                sig_scales=args.signal_scales,
                zscore_scales=args.zscore_scales,
                ignore_morph=ignore_morph,
            )
        )

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(
        ["meets_95", "meets_82hp", "healthy_permit"],
        ascending=[False, False, False],
    )
    out.to_csv(args.out_dir / "pareto_posthoc.csv", index=False)
    print(f"\nWrote: {args.out_dir / 'pareto_posthoc.csv'}")

    target = out[(out["abnormal_inhibit"] >= 0.95) & (out["false_permit"] <= 0.05)]
    if len(target):
        best = target.sort_values("healthy_permit", ascending=False)
        print("\n=== Operating points meeting >=95% abnormal inhibit AND <=5% false permit ===")
        print(best.head(10).to_string(index=False))
    else:
        target93 = out[out["abnormal_inhibit"] >= 0.93].sort_values(
            "healthy_permit",
            ascending=False,
        )
        print("\n=== No point meets both targets. Best >=93% abnormal inhibit ===")
        print(target93.head(10).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    quick = sub.add_parser("quick", help="Fast 10-record MIT-BIH Pareto test.")
    quick.add_argument("--data-dir", type=Path, default=Path("data"))
    quick.add_argument("--out-dir", type=Path, default=Path("Results/pareto_quick_test"))
    quick.add_argument("--abnormal-target", type=float, default=0.95)
    quick.add_argument("--healthy-fi-cap", type=float, default=0.15)
    quick.add_argument("--records", nargs="+", default=None)
    quick.add_argument("--no-hybrid-gate", action="store_true")

    full = sub.add_parser("full", help="Full beat-sync Pareto grid.")
    full.add_argument("--data-dir", type=Path, default=Path("data"))
    full.add_argument(
        "--out-root",
        type=Path,
        default=Path("Results/final_mitbih_validation/pareto_sweep"),
    )
    full.add_argument("--datasets", nargs="+", default=["mitdb"])
    full.add_argument("--targets", nargs="+", type=float, default=[0.95, 0.96, 0.97, 0.98])
    full.add_argument(
        "--healthy-fi-caps",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.60, 0.80],
    )
    full.add_argument("--all-use-hybrid-gate", action="store_true", default=False)
    full.add_argument("--summary-mode", default="layer1_adaptive_gated")

    posthoc = sub.add_parser("posthoc", help="Post-hoc sweep on existing per_beat.csv.")
    posthoc.add_argument(
        "--per-beat",
        type=Path,
        default=Path("Results/final_mitbih_validation/beat_sync/per_beat.csv"),
    )
    posthoc.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Results/final_mitbih_validation/pareto_posthoc"),
    )
    posthoc.add_argument("--mode-name", default="layer1_adaptive_gated")
    posthoc.add_argument("--feature-set", default="all")
    posthoc.add_argument(
        "--mahal-scales",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0],
    )
    posthoc.add_argument(
        "--signal-scales",
        nargs="+",
        type=float,
        default=[0.8, 1.0, 1.2, 1.5, 2.0, 3.0],
    )
    posthoc.add_argument("--zscore-scales", nargs="+", type=float, default=[1.0])
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "quick":
        run_quick(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            abnormal_target=args.abnormal_target,
            healthy_fi_cap=args.healthy_fi_cap,
            records=args.records,
            use_hybrid_gate=not args.no_hybrid_gate,
        )
    elif args.mode == "full":
        run_full_grid(args)
    elif args.mode == "posthoc":
        run_posthoc(args)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
