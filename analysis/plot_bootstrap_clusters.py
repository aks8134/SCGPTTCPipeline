"""
Plot all bootstrap centroid replicates overlaid per cluster.

For each cluster subplot:
  - Thin semi-transparent lines: each bootstrap replicate centroid
  - Shaded band: mean ± std across replicates
  - Thick solid line: real centroid from original data

Usage:
    python -m ScgptTCPipeline.analysis.plot_bootstrap_clusters \
        --data_dir      data/TCSeqData_n_20 \
        --bootstrap_dir data/TCSeqData_n_20_bs100_SKMGN_male \
        --tissue        SKMGN \
        --omic          TRNSCRPT \
        --sex           male \
        [--output       bootstrap_clusters_SKMGN_TRNSCRPT_male.png]
"""

import argparse
import glob
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WEEKS = ["1w", "2w", "4w", "8w"]
X = list(range(len(WEEKS)))


def load_centroids(data_dir: str, tissue: str, omic: str, sex: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, f"{tissue}_{omic}_{sex}", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.groupby("cluster")[WEEKS].mean()  # index=cluster_num, cols=weeks


def load_bootstrap_replicates(bootstrap_dir: str, tissue: str, omic: str, sex: str) -> dict:
    """Returns {cluster_num: np.array of shape [n_replicates, 4]}"""
    replicate_dirs = sorted(glob.glob(os.path.join(bootstrap_dir, "bootstrap_*")))
    if not replicate_dirs:
        raise FileNotFoundError(f"No bootstrap_* subdirs found in {bootstrap_dir}")

    cluster_data: dict[int, list] = {}
    for rep_dir in replicate_dirs:
        pattern = os.path.join(rep_dir, f"{tissue}_{omic}_{sex}", "*.csv")
        files = sorted(glob.glob(pattern))
        for f in files:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                c_num = int(row["cluster"])
                vals = [row[w] for w in WEEKS]
                cluster_data.setdefault(c_num, []).append(vals)

    return {c: np.array(v) for c, v in cluster_data.items()}


def plot(bootstrap_dir, tissue, omic, sex, output, data_dir=None):
    bs_data = load_bootstrap_replicates(bootstrap_dir, tissue, omic, sex)

    cluster_nums = sorted(bs_data.keys())
    n_clusters = len(cluster_nums)
    ncols = 5
    nrows = int(np.ceil(n_clusters / ncols))
    colors = cm.tab20(np.linspace(0, 1, n_clusters))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharex=True)
    axes = axes.flatten()

    for i, c_num in enumerate(cluster_nums):
        ax = axes[i]
        color = colors[i]
        reps = bs_data[c_num]  # [n_replicates, 4]

        # Bootstrap replicate lines
        for rep in reps:
            ax.plot(X, rep, color=color, alpha=0.08, linewidth=0.6, zorder=1)

        # Mean ± std band
        mean = reps.mean(axis=0)
        std  = reps.std(axis=0)
        ax.fill_between(X, mean - std, mean + std, color=color, alpha=0.2, zorder=2)
        ax.plot(X, mean, color=color, linewidth=1.5, linestyle="--", zorder=3, label="BS mean")

        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
        ax.set_xticks(X)
        ax.set_xticklabels(WEEKS, fontsize=8)
        ax.set_title(f"Cluster {c_num}  (n_bs={len(reps)})", fontsize=9, fontweight="bold")
        ax.set_ylabel("z-score (logFC)", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    n_reps = len(sorted(glob.glob(os.path.join(bootstrap_dir, "bootstrap_*"))))
    fig.suptitle(
        f"{tissue} | {omic} | {sex} — Bootstrap Replicates (n={n_reps})\n"
        f"Thin lines: replicates  |  Dashed: BS mean ± std",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if output is None:
        output = f"bootstrap_clusters_{tissue}_{omic}_{sex}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot bootstrap centroid replicates per cluster")
    parser.add_argument("--bootstrap_dir", required=True, help="Bootstrap directory (contains bootstrap_* subdirs)")
    parser.add_argument("--tissue",        required=True)
    parser.add_argument("--omic",          required=True)
    parser.add_argument("--sex",           required=True)
    parser.add_argument("--output",        default=None, help="Output PNG path (auto-named if omitted)")
    args = parser.parse_args()

    plot(args.bootstrap_dir, args.tissue, args.omic, args.sex, args.output)


if __name__ == "__main__":
    main()