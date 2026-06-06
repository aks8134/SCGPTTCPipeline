"""
TCSeqData Loader for ScgptTCPipeline.

Supports all tissues (HEART, LIVER, SKMGN), all sexes (male, female),
all omics (TRNSCRPT, PROT, METAB, ATAC), and both n_20 / n_50 cluster datasets.

Data format (per cluster CSV):
    Columns: gene, cluster, 1w, 2w, 4w, 8w
    Values:  z-scored log-fold-change relative to sedentary control
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple


VALID_TISSUES  = ["HEART", "LIVER", "SKMGN"]
VALID_OMICS    = ["TRNSCRPT", "PROT", "METAB", "ATAC"]
VALID_SEXES    = ["male", "female"]
VALID_WEEKS    = ["1w", "2w", "4w", "8w"]
WEEK_ORDER     = ["1w", "2w", "4w", "8w"]


class TCSeqLoader:
    """
    Unified loader for TCSeqData_n_20 / TCSeqData_n_50 cluster-level omic data.

    Each (tissue, omic, sex) combination lives in its own subdirectory,
    e.g. data/TCSeqData_n_20/HEART_TRNSCRPT_female/, containing cluster1.csv … clusterN.csv.
    """

    def __init__(self, data_dir: str = "data/TCSeqData_n_20"):
        self.data_dir = data_dir
        self._cache: Dict[Tuple, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _folder_name(self, tissue: str, omic: str, sex: str) -> str:
        return f"{tissue.upper()}_{omic.upper()}_{sex.lower()}"

    def _omic_dir(self, tissue: str, omic: str, sex: str) -> str:
        return os.path.join(self.data_dir, self._folder_name(tissue, omic, sex))

    def available_combinations(self) -> List[Tuple[str, str, str]]:
        """Returns all (tissue, omic, sex) combinations present on disk."""
        combos = []
        for tissue in VALID_TISSUES:
            for omic in VALID_OMICS:
                for sex in VALID_SEXES:
                    if os.path.isdir(self._omic_dir(tissue, omic, sex)):
                        combos.append((tissue, omic, sex))
        return combos

    def n_clusters(self, tissue: str, omic: str, sex: str) -> int:
        """Returns number of cluster CSVs present for this combination."""
        d = self._omic_dir(tissue, omic, sex)
        return len(glob.glob(os.path.join(d, "cluster*.csv")))

    # ------------------------------------------------------------------
    # Main loading API
    # ------------------------------------------------------------------

    def load(
        self,
        tissue: str,
        omic: str,
        sex: str,
        clusters: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Load all cluster CSVs for one (tissue, omic, sex) combination.

        Returns a DataFrame with columns: gene, cluster, 1w, 2w, 4w, 8w.
        Results are cached — repeated calls are free.
        """
        key = (tissue.upper(), omic.upper(), sex.lower(),
               tuple(sorted(clusters)) if clusters else None)
        if key in self._cache:
            return self._cache[key].copy()

        omic_dir = self._omic_dir(tissue, omic, sex)
        if not os.path.isdir(omic_dir):
            raise FileNotFoundError(
                f"No data directory found: {omic_dir}\n"
                f"Available: {[self._folder_name(*c) for c in self.available_combinations()]}"
            )

        if clusters:
            csv_files = [os.path.join(omic_dir, f"cluster{c}.csv") for c in clusters]
            for f in csv_files:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Missing cluster file: {f}")
        else:
            csv_files = sorted(glob.glob(os.path.join(omic_dir, "cluster*.csv")))
            if not csv_files:
                raise FileNotFoundError(f"No cluster CSVs in: {omic_dir}")

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            df.columns = [str(c).strip('"') for c in df.columns]
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str).str.strip('"')
            for wk in VALID_WEEKS:
                if wk in df.columns:
                    df[wk] = pd.to_numeric(df[wk], errors="coerce")
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        print(
            f"[TCSeqLoader] {tissue.upper()}|{omic.upper()}|{sex}: "
            f"{len(combined)} features, {combined['cluster'].nunique()} clusters"
        )
        self._cache[key] = combined
        return combined.copy()

    def load_all(
        self,
        sex: str = "male",
        tissue: Optional[str] = None,
        clusters: Optional[List[int]] = None,
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Load every available omic for a given sex (and optionally tissue).
        Returns dict keyed by (tissue, omic).
        """
        result = {}
        for t, o, s in self.available_combinations():
            if s != sex.lower():
                continue
            if tissue and t != tissue.upper():
                continue
            try:
                result[(t, o)] = self.load(t, o, s, clusters=clusters)
            except FileNotFoundError as e:
                print(f"[TCSeqLoader] Warning: {e}")
        return result

    # ------------------------------------------------------------------
    # Centroid helpers
    # ------------------------------------------------------------------

    def centroids(self, tissue: str, omic: str, sex: str) -> pd.DataFrame:
        """
        Mean temporal trajectory per cluster.
        Index = cluster number, columns = [1w, 2w, 4w, 8w].
        """
        df = self.load(tissue, omic, sex)
        return df.groupby("cluster")[VALID_WEEKS].mean()

    def all_centroids(
        self, sex: str = "male", tissue: Optional[str] = None
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Returns centroids for all available (tissue, omic) combinations."""
        result = {}
        for t, o, s in self.available_combinations():
            if s != sex.lower():
                continue
            if tissue and t != tissue.upper():
                continue
            try:
                result[(t, o)] = self.centroids(t, o, s)
            except Exception as e:
                print(f"[TCSeqLoader] Warning centroids {t}|{o}|{s}: {e}")
        return result

    def cluster_members(
        self, tissue: str, omic: str, sex: str, cluster_num: int
    ) -> pd.DataFrame:
        """Returns all feature rows for a specific cluster."""
        df = self.load(tissue, omic, sex)
        return df[df["cluster"] == cluster_num].copy()

    # ------------------------------------------------------------------
    # Context week helpers
    # ------------------------------------------------------------------

    @staticmethod
    def context_weeks(predict_week: str) -> List[str]:
        """
        Returns earlier timepoints available as causal context.
            predict 2w → [1w]
            predict 4w → [1w, 2w]
            predict 8w → [1w, 2w, 4w]
        """
        if predict_week not in WEEK_ORDER:
            raise ValueError(f"predict_week must be one of {WEEK_ORDER}")
        idx = WEEK_ORDER.index(predict_week)
        if idx == 0:
            raise ValueError("Cannot predict 1w — no earlier timepoints available.")
        return WEEK_ORDER[:idx]

    @staticmethod
    def all_week_subsets() -> List[Tuple[List[str], str]]:
        """
        Returns all valid (context_weeks, predict_week) pairs for causal prediction.
        Used during random masking in training.
        """
        pairs = []
        for i in range(1, len(WEEK_ORDER)):
            pairs.append((WEEK_ORDER[:i], WEEK_ORDER[i]))
        return pairs

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def summary(self, sex: str = "male"):
        """Prints a concise summary of all available data."""
        print(f"\n=== TCSeqData Summary (sex={sex}) ===")
        for t, o, s in self.available_combinations():
            if s != sex:
                continue
            try:
                df = self.load(t, o, s)
                n_c = df["cluster"].nunique()
                row = (
                    f"  {t:6s} | {o:8s} | {s:6s} : "
                    f"{len(df):5d} features, {n_c:2d} clusters"
                )
                for wk in VALID_WEEKS:
                    if wk in df.columns:
                        row += f"  {wk}:{df[wk].notna().sum()}"
                print(row)
            except FileNotFoundError:
                pass
        print()
