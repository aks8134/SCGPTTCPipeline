"""
Plot all clusters for a given (tissue, omic, sex) combination.
Each subplot shows individual gene/feature trajectories + centroid.

Usage:
    python -m ScgptTCPipeline.analysis.plot_clusters \
        --data_dir data/TCSeqData_n_20 \
        --tissue   SKMGN \
        --omic     METAB \
        --sex      male \
        [--output  clusters_SKMGN_METAB_male.png]
"""

import argparse
import glob
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WEEKS = ["1w", "2w", "4w", "8w"]


def load_data(data_dir: str, tissue: str, omic: str, sex: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, f"{tissue}_{omic}_{sex}", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def plot_clusters(data_dir: str, tissue: str, omic: str, sex: str, output: str | None):
    df = load_data(data_dir, tissue, omic, sex)

    cluster_nums = sorted(df["cluster"].unique())
    n_clusters = len(cluster_nums)

    ncols = 5
    nrows = int(np.ceil(n_clusters / ncols))
    colors = cm.tab20(np.linspace(0, 1, n_clusters))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharex=True)
    axes = axes.flatten()

    for i, c_num in enumerate(cluster_nums):
        ax = axes[i]
        color = colors[i]
        cluster_df = df[df["cluster"] == c_num]
        vals = cluster_df[WEEKS].values  # [n_features, 4]

        # Individual trajectories
        for row in vals:
            ax.plot(WEEKS, row, color=color, alpha=0.15, linewidth=0.6, zorder=1)

        # Centroid
        centroid = np.nanmean(vals, axis=0)
        ax.plot(WEEKS, centroid, color=color, linewidth=2.5, zorder=3, label="Centroid")

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_title(f"Cluster {c_num}  (n={len(cluster_df)})", fontsize=9, fontweight="bold")
        ax.set_ylabel("z-score (logFC)", fontsize=7)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{tissue} | {omic} | {sex} — {n_clusters} Clusters\nGene/Feature Trajectories + Centroid",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if output is None:
        output = f"clusters_{tissue}_{omic}_{sex}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot all clusters for a given tissue/omic/sex")
    parser.add_argument("--data_dir", required=True, help="Path to TCSeqData directory")
    parser.add_argument("--tissue",   required=True, help="Tissue (e.g. SKMGN)")
    parser.add_argument("--omic",     required=True, help="Omic (e.g. METAB, TRNSCRPT)")
    parser.add_argument("--sex",      required=True, help="Sex (e.g. male, female)")
    parser.add_argument("--output",   default=None,  help="Output PNG path (auto-named if omitted)")
    args = parser.parse_args()

    plot_clusters(args.data_dir, args.tissue, args.omic, args.sex, args.output)


if __name__ == "__main__":
    main()