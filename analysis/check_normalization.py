"""
Check whether the per-gene z-scoring creates a sum-to-zero constraint
across timepoints. Computes the distribution of partial sums:

  S1  = 1w
  S12 = 1w + 2w
  S124= 1w + 2w + 4w
  S   = 1w + 2w + 4w + 8w

Reports mean ± std for each. If z-scored per gene across timepoints,
S should be ~0 with near-zero std.

Usage:
    python -m ScgptTCPipeline.analysis.check_normalization \
        --data_dir data/TCSeqData_n_20 \
        [--tissue SKMGN] [--omic METAB] [--sex male]
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd


WEEKS = ["1w", "2w", "4w", "8w"]

PARTIAL_SUMS = [
    ("1w",           ["1w"]),
    ("1w+2w",        ["1w", "2w"]),
    ("1w+2w+4w",     ["1w", "2w", "4w"]),
    ("1w+2w+4w+8w",  ["1w", "2w", "4w", "8w"]),
]


def load_all(data_dir: str, tissue: str | None, omic: str | None, sex: str | None) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*", "*.csv")
    frames = []
    for path in sorted(glob.glob(pattern)):
        parts = os.path.basename(os.path.dirname(path)).split("_")
        if len(parts) < 3:
            continue
        t, o, s = parts[0], parts[1], "_".join(parts[2:])
        if tissue and t != tissue:
            continue
        if omic and o != omic:
            continue
        if sex and s != sex:
            continue
        df = pd.read_csv(path)
        df["tissue"] = t
        df["omic"] = o
        df["sex"] = s
        frames.append(df)
    if not frames:
        raise ValueError(f"No data found in {data_dir} matching tissue={tissue}, omic={omic}, sex={sex}")
    return pd.concat(frames, ignore_index=True)


def report(data_dir, tissue, omic, sex):
    df = load_all(data_dir, tissue, omic, sex)

    # Drop rows missing any week
    df = df.dropna(subset=WEEKS)
    n_genes = len(df)

    combos = sorted(df[["tissue", "omic", "sex"]].drop_duplicates().apply(tuple, axis=1))
    print(f"\nLoaded {n_genes} genes/features from {len(combos)} (tissue, omic, sex) combinations:")
    for c in combos:
        print(f"  {c[0]} | {c[1]} | {c[2]}")

    print(f"\n{'Partial sum':<20}  {'Mean':>12}  {'Std':>12}  {'Min':>12}  {'Max':>12}")
    print("-" * 72)
    for label, cols in PARTIAL_SUMS:
        vals = df[cols].sum(axis=1).values
        print(f"{label:<20}  {vals.mean():>12.6f}  {vals.std():>12.6f}  {vals.min():>12.6f}  {vals.max():>12.6f}")

    print()
    print("Interpretation:")
    print("  If '1w+2w+4w+8w' mean≈0 and std≈0 → z-scored per gene across timepoints")
    print("  (8w prediction from 1w,2w,4w context is then trivially determined)")
    print("  If std is large → constraint does not hold, 8w prediction is a real task")


def main():
    parser = argparse.ArgumentParser(description="Check per-gene z-score normalization constraint")
    parser.add_argument("--data_dir", default="data/TCSeqData_n_20")
    parser.add_argument("--tissue",   default=None, help="Filter by tissue (e.g. SKMGN)")
    parser.add_argument("--omic",     default=None, help="Filter by omic (e.g. METAB)")
    parser.add_argument("--sex",      default=None, help="Filter by sex (e.g. male)")
    args = parser.parse_args()

    report(args.data_dir, args.tissue, args.omic, args.sex)


if __name__ == "__main__":
    main()