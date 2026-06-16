"""
Quick diagnostic summary over run_benchmark.py per_record.csv.
"""
import pandas as pd
import sys


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else "benchmark_results/per_record.csv"
    df = pd.read_csv(path)

    print(f"\nLoaded {path}: {len(df)} rows\n")

    cols = ["dataset", "record", "n_ref_beats", "n_candidates", "n_accepted",
            "det_sensitivity", "acc_sensitivity", "acc_fp_per_hour", "polarity"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))

    print("\n--- Summary ---")
    print(f"Total records: {len(df)}")
    for col, label in [("det_sensitivity", "detector"), ("acc_sensitivity", "accepted")]:
        if col in df.columns:
            valid = df[df[col].notna() & (df.get("n_ref_beats", 1) > 0)]
            print(f"  [{label}] mean Se={valid[col].mean():.3f} on {len(valid)} records")


if __name__ == "__main__":
    main()
