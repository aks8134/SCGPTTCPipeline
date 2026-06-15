"""
generate_bootstraps.py — Pre-compute bootstrap centroid replicates for training.

For each (tissue, omic, sex, cluster), resamples the member genes/features
with replacement n_bootstraps times and computes a bootstrap centroid
(mean logFC across resampled members at each timepoint).

Optionally adds Gaussian noise to each gene's values before computing the
centroid, controlled by --noise_std:
  - float (e.g. 0.1): fixed noise std in z-score units, same for all genes
  - "adaptive":       per-gene noise std = std of that gene's values across
                      timepoints, so noise is proportional to each gene's
                      temporal variability

Each replicate is saved as its own directory mirroring the original data
format so TCSeqLoader can read it without any modifications.

Output structure:
    <output_dir>/
        bootstrap_0001/
            HEART_TRNSCRPT_male/
                cluster1.csv   ← one row: bootstrap centroid
                cluster2.csv
                ...
        bootstrap_0002/
            ...

Usage:
    # Bootstrap only (original behaviour)
    python -m ScgptTCPipeline.data.generate_bootstraps \\
        --data_dir     data/TCSeqData_n_20 \\
        --output_dir   data/TCSeqData_n_20_bs100 \\
        --n_bootstraps 100 \\
        --seed         42

    # Bootstrap + fixed Gaussian noise (std=0.1)
    python -m ScgptTCPipeline.data.generate_bootstraps \\
        --data_dir     data/TCSeqData_n_20 \\
        --output_dir   data/TCSeqData_n_20_bs100_noise01 \\
        --n_bootstraps 100 \\
        --noise_std    0.1

    # Bootstrap + adaptive noise (per-gene temporal std)
    python -m ScgptTCPipeline.data.generate_bootstraps \\
        --data_dir     data/TCSeqData_n_20 \\
        --output_dir   data/TCSeqData_n_20_bs100_adaptive \\
        --n_bootstraps 100 \\
        --noise_std    adaptive
"""

import argparse
import os
import sys
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ScgptTCPipeline.data.loader import TCSeqLoader, VALID_WEEKS


# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------

def _add_noise(
    values: np.ndarray,
    rng: np.random.Generator,
    noise_std: float | Literal["adaptive"] | None,
) -> np.ndarray:
    """
    Add Gaussian noise to a [n_genes, n_weeks] array.

    noise_std=None or 0  : no noise
    noise_std=float      : fixed std, same for every gene and timepoint
    noise_std="adaptive" : per-gene std = std of that gene's values across
                           timepoints (genes with larger temporal range get
                           proportionally larger noise)
    """
    if not noise_std:
        return values

    if noise_std == "adaptive":
        per_gene_std = np.nanstd(values, axis=1, keepdims=True)  # [n_genes, 1]
        per_gene_std = np.where(per_gene_std == 0, 1e-6, per_gene_std)
        std_matrix = np.broadcast_to(per_gene_std, values.shape)
    else:
        std_matrix = float(noise_std)

    noise = rng.normal(loc=0.0, scale=std_matrix, size=values.shape)
    return values + noise


def _bootstrap_centroid(
    cluster_df: pd.DataFrame,
    rng: np.random.Generator,
    noise_std: float | Literal["adaptive"] | None,
) -> pd.Series:
    """
    Resample rows with replacement, optionally add noise, return mean per week.
    """
    n = len(cluster_df)
    sample_idx = rng.integers(0, n, size=n)
    resampled = cluster_df.iloc[sample_idx][VALID_WEEKS].values.astype(float)  # [n, n_weeks]

    resampled = _add_noise(resampled, rng, noise_std)

    centroid = np.nanmean(resampled, axis=0)
    return pd.Series(centroid, index=VALID_WEEKS)


def generate_bootstraps(
    data_dir:     str,
    output_dir:   str,
    n_bootstraps: int,
    seed:         int  = 42,
    sex:          str  = None,
    tissue:       str  = None,
    noise_std:    float | Literal["adaptive"] | None = None,
):
    loader = TCSeqLoader(data_dir=data_dir)
    combos = loader.available_combinations()

    if sex:
        combos = [(t, o, s) for t, o, s in combos if s == sex.lower()]
    if tissue:
        combos = [(t, o, s) for t, o, s in combos if t == tissue.upper()]

    if not combos:
        raise ValueError(
            f"No data combinations found for sex={sex}, tissue={tissue} "
            f"in {data_dir}."
        )

    noise_label = f"{noise_std}" if noise_std else "none"
    print(f"\n[generate_bootstraps] Starting bootstrap generation")
    print(f"  Source       : {data_dir}")
    print(f"  Output       : {output_dir}")
    print(f"  n_bootstraps : {n_bootstraps}")
    print(f"  seed         : {seed}")
    print(f"  noise_std    : {noise_label}")
    print(f"  Combinations : {len(combos)}")

    base_rng = np.random.default_rng(seed)

    for b in range(1, n_bootstraps + 1):
        bs_dir = os.path.join(output_dir, f"bootstrap_{b:04d}")

        for tissue_k, omic_k, sex_k in combos:
            omic_out = os.path.join(bs_dir, f"{tissue_k}_{omic_k}_{sex_k}")
            os.makedirs(omic_out, exist_ok=True)

            df = loader.load(tissue_k, omic_k, sex_k)

            for c_num in sorted(df["cluster"].unique()):
                cluster_df = df[df["cluster"] == c_num]

                centroid = _bootstrap_centroid(cluster_df, base_rng, noise_std)

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
    p.add_argument("--noise_std",     default=None,
                   help=(
                       "Gaussian noise to add to gene values before centroid computation. "
                       "Pass a float (e.g. 0.1) for fixed noise, or 'adaptive' to scale "
                       "noise by each gene's temporal std. Default: no noise."
                   ))
    args = p.parse_args()

    # Parse noise_std: "adaptive" stays as string, numeric string → float, None → None
    noise_std = None
    if args.noise_std is not None:
        if args.noise_std.lower() == "adaptive":
            noise_std = "adaptive"
        else:
            noise_std = float(args.noise_std)

    generate_bootstraps(
        data_dir     = args.data_dir,
        output_dir   = args.output_dir,
        n_bootstraps = args.n_bootstraps,
        seed         = args.seed,
        sex          = args.sex,
        tissue       = args.tissue,
        noise_std    = noise_std,
    )


if __name__ == "__main__":
    main()