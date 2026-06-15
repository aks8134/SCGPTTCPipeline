"""
Re-normalize cluster data by timepoint instead of per gene across timepoints.

The original data is z-scored per gene across timepoints, which creates an
exact constraint: 1w + 2w + 4w + 8w = 0 for every gene. This makes 8w
prediction trivial given 1w, 2w, 4w as context.

This script re-normalizes by computing mean and std across all genes within
each timepoint, breaking that constraint:

    value[gene, t] = (value[gene, t] - mean_genes(t)) / std_genes(t)

The normalization is computed globally across all clusters for a given
(tissue, omic, sex) combination so that relative differences between
clusters are preserved.

Output mirrors the original data format exactly, so the existing pipeline
works with --data_dir pointing to the new directory.

Usage:
    python -m ScgptTCPipeline.data.renorm_by_timepoint \\
        --data_dir  data/TCSeqData_n_20 \\
        --output_dir data/TCSeqData_n_20_renorm \\
        [--tissue SKMGN] [--sex male]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ScgptTCPipeline.data.loader import TCSeqLoader, VALID_WEEKS


def renorm_combination(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Re-normalize a full (tissue, omic, sex) dataframe by timepoint.
    Returns normalized df and the per-timepoint (mean, std) stats used.
    """
    df = df.copy()
    stats = {}
    for wk in VALID_WEEKS:
        col = df[wk].dropna()
        mu  = col.mean()
        std = col.std()
        if std < 1e-8:
            std = 1.0  # avoid division by zero for constant columns
        df[wk] = (df[wk] - mu) / std
        stats[wk] = {"mean": mu, "std": std}
    return df, stats


def generate(data_dir: str, output_dir: str, tissue: str | None, sex: str | None):
    loader = TCSeqLoader(data_dir=data_dir)
    combos = loader.available_combinations()

    if tissue:
        combos = [(t, o, s) for t, o, s in combos if t == tissue.upper()]
    if sex:
        combos = [(t, o, s) for t, o, s in combos if s == sex.lower()]

    if not combos:
        raise ValueError(f"No combinations found for tissue={tissue}, sex={sex}")

    print(f"\n[renorm_by_timepoint]")
    print(f"  Source      : {data_dir}")
    print(f"  Output      : {output_dir}")
    print(f"  Combinations: {len(combos)}")
    print()

    for tissue_k, omic_k, sex_k in combos:
        df = loader.load(tissue_k, omic_k, sex_k)

        # Compute normalization stats globally across all clusters for this combo
        norm_df, stats = renorm_combination(df)

        print(f"  {tissue_k} | {omic_k} | {sex_k}  ({len(norm_df)} features)")
        for wk in VALID_WEEKS:
            print(f"    {wk}: mean={stats[wk]['mean']:+.4f}  std={stats[wk]['std']:.4f}")

        # Write out one CSV per cluster, same format as original
        combo_dir = os.path.join(output_dir, f"{tissue_k}_{omic_k}_{sex_k}")
        os.makedirs(combo_dir, exist_ok=True)

        for c_num in sorted(norm_df["cluster"].unique()):
            cluster_df = norm_df[norm_df["cluster"] == c_num].copy()
            out_path   = os.path.join(combo_dir, f"cluster{int(c_num)}.csv")
            cluster_df.to_csv(out_path, index=False)

        print()

    print(f"[renorm_by_timepoint] Done. Output written to: {output_dir}\n")


def main():
    p = argparse.ArgumentParser(
        description="Re-normalize cluster data by timepoint (across genes) instead of per gene."
    )
    p.add_argument("--data_dir",   required=True,  help="Original TCSeqData directory")
    p.add_argument("--output_dir", required=True,  help="Output directory (same structure)")
    p.add_argument("--tissue",     default=None,   help="Restrict to one tissue")
    p.add_argument("--sex",        default=None,   choices=["male", "female"],
                   help="Restrict to one sex")
    args = p.parse_args()

    generate(args.data_dir, args.output_dir, args.tissue, args.sex)


if __name__ == "__main__":
    main()