"""
generate_bootstraps.py — Pre-compute bootstrap centroid replicates for training.

For each (tissue, omic, sex, cluster), resamples the member genes/features
with replacement n_bootstraps times and computes a bootstrap centroid
(mean logFC across resampled members at each timepoint).

Each replicate is saved as its own directory mirroring the original data
format so TCSeqLoader can read it without any modifications.

Output structure:
    <output_dir>/
        bootstrap_0001/
            HEART_TRNSCRPT_male/
                cluster1.csv   ← one row: bootstrap centroid
                cluster2.csv
                ...
            HEART_PROT_male/
                ...
        bootstrap_0002/
            ...
        ...

Each cluster CSV has the same columns as the original (gene, cluster, 1w, 2w, 4w, 8w)
with a single row representing the bootstrap centroid. This makes it fully
compatible with TCSeqLoader.centroids() which just takes the mean.

Usage:
    python -m ScgptTCPipeline.data.generate_bootstraps \\
        --data_dir     data/TCSeqData_n_20 \\
        --output_dir   data/TCSeqData_n_20_bs100 \\
        --n_bootstraps 100 \\
        --seed         42

    # Restrict to one sex or tissue:
    python -m ScgptTCPipeline.data.generate_bootstraps \\
        --data_dir     data/TCSeqData_n_20 \\
        --output_dir   data/TCSeqData_n_20_bs100_male \\
        --n_bootstraps 100 \\
        --sex          male
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ScgptTCPipeline.data.loader import TCSeqLoader, VALID_WEEKS


# ------------------------------------------------------------------
# Core bootstrap logic
# ------------------------------------------------------------------

def _bootstrap_centroid(
    cluster_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Resample rows of cluster_df with replacement and return mean logFC per week.
    NaN values are ignored in the mean (nanmean behaviour).
    """
    n = len(cluster_df)
    sample_idx = rng.integers(0, n, size=n)
    resampled  = cluster_df.iloc[sample_idx]
    return resampled[VALID_WEEKS].apply(lambda col: np.nanmean(col.values))


def generate_bootstraps(
    data_dir:     str,
    output_dir:   str,
    n_bootstraps: int,
    seed:         int  = 42,
    sex:          str  = None,
    tissue:       str  = None,
):
    loader = TCSeqLoader(data_dir=data_dir)
    combos = loader.available_combinations()

    # Filter by sex / tissue if requested
    if sex:
        combos = [(t, o, s) for t, o, s in combos if s == sex.lower()]
    if tissue:
        combos = [(t, o, s) for t, o, s in combos if t == tissue.upper()]

    if not combos:
        raise ValueError(
            f"No data combinations found for sex={sex}, tissue={tissue} "
            f"in {data_dir}."
        )

    print(f"\n[generate_bootstraps] Starting bootstrap generation")
    print(f"  Source       : {data_dir}")
    print(f"  Output       : {output_dir}")
    print(f"  n_bootstraps : {n_bootstraps}")
    print(f"  seed         : {seed}")
    print(f"  Combinations : {len(combos)}")

    # Independent RNG per (bootstrap, combo, cluster) via seed offset
    base_rng = np.random.default_rng(seed)

    for b in range(1, n_bootstraps + 1):
        bs_dir = os.path.join(output_dir, f"bootstrap_{b:04d}")

        for tissue_k, omic_k, sex_k in combos:
            omic_out = os.path.join(
                bs_dir, f"{tissue_k}_{omic_k}_{sex_k}"
            )
            os.makedirs(omic_out, exist_ok=True)

            df = loader.load(tissue_k, omic_k, sex_k)

            for c_num in sorted(df["cluster"].unique()):
                cluster_df = df[df["cluster"] == c_num]

                centroid = _bootstrap_centroid(cluster_df, base_rng)

                row = pd.DataFrame({
                    "gene":    ["bootstrap_centroid"],
                    "cluster": [int(c_num)],
                })
                for wk in VALID_WEEKS:
                    row[wk] = centroid.get(wk, float("nan"))

                out_path = os.path.join(omic_out, f"cluster{int(c_num)}.csv")
                row.to_csv(out_path, index=False)

        if b % 10 == 0 or b == n_bootstraps:
            print(f"  [{b:4d}/{n_bootstraps}] replicates written...")

    print(f"\n[generate_bootstraps] Done.")
    print(f"  {n_bootstraps} replicates saved to: {output_dir}")
    print(
        f"  Use with: python -m ScgptTCPipeline.finetune "
        f"--bootstrap_dir {output_dir} ...\n"
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Pre-compute bootstrap centroid replicates for TCSeqData."
    )
    p.add_argument("--data_dir",      default="data/TCSeqData_n_20",
                   help="Path to original TCSeqData directory.")
    p.add_argument("--output_dir",    required=True,
                   help="Where to save bootstrap replicate directories.")
    p.add_argument("--n_bootstraps",  type=int, default=100,
                   help="Number of bootstrap replicates to generate.")
    p.add_argument("--seed",          type=int, default=42,
                   help="Base random seed for reproducibility.")
    p.add_argument("--sex",           default=None,
                   choices=["male", "female"],
                   help="Restrict to one sex. None = both.")
    p.add_argument("--tissue",        default=None,
                   help="Restrict to one tissue (HEART/LIVER/SKMGN). None = all.")
    args = p.parse_args()

    generate_bootstraps(
        data_dir     = args.data_dir,
        output_dir   = args.output_dir,
        n_bootstraps = args.n_bootstraps,
        seed         = args.seed,
        sex          = args.sex,
        tissue       = args.tissue,
    )


if __name__ == "__main__":
    main()
