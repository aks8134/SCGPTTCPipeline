"""
Evaluation metrics for ScgptTCPipeline.

Two evaluation modes:

  1. Point prediction  — single predicted value vs actual for one (cluster, week).
     Metrics: MAE, RMSE, rRMSE, L2-ratio  (backward-compatible with old TCPipeline).

  2. Trajectory prediction — full predicted trajectory [T values] vs actual [T values]
     for one cluster across multiple weeks.
     Metrics: MAE, RMSE, Pearson r, DTW distance, direction accuracy.

Both modes produce a results DataFrame that can be sliced by
tissue / omic / sex / cluster / predict_week.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


# ------------------------------------------------------------------
# Low-level scalar metrics
# ------------------------------------------------------------------

def _safe(x) -> float:
    v = float(x)
    return v if np.isfinite(v) else float("nan")


def point_metrics(
    errors:  Union[np.ndarray, List, pd.Series],
    actuals: Optional[Union[np.ndarray, List, pd.Series]] = None,
) -> Dict[str, float]:
    """
    MAE, RMSE, rRMSE, R², and count over a vector of (predicted - actual) errors.
    rRMSE and R² are only computed when actuals is supplied.
    """
    err = np.asarray(errors, dtype=float)
    valid = err[np.isfinite(err)]
    if len(valid) == 0:
        return {"count": 0, "mae": np.nan, "rmse": np.nan,
                "rrmse": np.nan, "r2": np.nan}

    mae  = float(np.mean(np.abs(valid)))
    rmse = float(np.sqrt(np.mean(valid ** 2)))

    rrmse = np.nan
    r2    = np.nan
    if actuals is not None:
        act = np.asarray(actuals, dtype=float)[np.isfinite(err)]
        denom = float(np.mean(np.abs(act)))
        if denom > 0:
            rrmse = rmse / denom
        # predicted = actual - error  (error = predicted - actual)
        predicted = act - valid
        r2 = r2_score(predicted, act)

    return {"count": len(valid), "mae": mae, "rmse": rmse,
            "rrmse": rrmse, "r2": r2}


def r2_score(
    predicted: Union[np.ndarray, List, pd.Series],
    actuals:   Union[np.ndarray, List, pd.Series],
) -> float:
    """
    Coefficient of determination  R² = 1 - SS_res / SS_tot.

    Interpretation:
      R² = 1.0  → perfect prediction
      R² = 0.0  → model predicts no better than the global mean of actuals
      R² < 0    → model is worse than predicting the global mean (same as rRMSE > 1)

    Relationship to l2_ratio (old TCPipeline):
      l2_ratio uses ||actual - historical_mean|| as denominator (time-local baseline).
      R² uses     ||actual - global_mean||       as denominator (global baseline).
      Both < 1 means you beat your respective baseline; R² is the standard convention.
    """
    pred = np.asarray(predicted, dtype=float)
    act  = np.asarray(actuals,   dtype=float)
    valid = np.isfinite(pred) & np.isfinite(act)
    if valid.sum() < 2:
        return np.nan
    p, a   = pred[valid], act[valid]
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def l2_ratio(
    errors:    Union[np.ndarray, List, pd.Series],
    baselines: Union[np.ndarray, List, pd.Series],
) -> float:
    """
    ||errors||_2 / ||baselines||_2  — the custom metric from old TCPipeline.
    baseline[i] = actual[i] - historical_mean[i]
    Values < 1 mean you beat the historical-mean baseline.
    """
    e = np.asarray(errors,    dtype=float)
    b = np.asarray(baselines, dtype=float)
    valid = np.isfinite(e) & np.isfinite(b)
    if not valid.any():
        return np.nan
    denom = float(np.linalg.norm(b[valid]))
    return float(np.linalg.norm(e[valid])) / denom if denom > 0 else np.nan


# ------------------------------------------------------------------
# Trajectory-level metrics
# ------------------------------------------------------------------

def pearson_r(pred: np.ndarray, actual: np.ndarray) -> float:
    """Pearson correlation between two trajectory vectors."""
    if len(pred) < 2:
        return np.nan
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 2:
        return np.nan
    p, a = pred[mask], actual[mask]
    if p.std() == 0 or a.std() == 0:
        return np.nan
    return float(np.corrcoef(p, a)[0, 1])


def dtw_distance(pred: np.ndarray, actual: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two 1-D trajectories.
    O(T^2) — fine for T=4 timepoints.
    """
    T = len(pred)
    dtw = np.full((T + 1, T + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, T + 1):
            cost = abs(float(pred[i - 1]) - float(actual[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[T, T])


def direction_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """
    Fraction of consecutive pairs where predicted direction (up/down)
    matches actual direction.  Meaningful for ≥2 timepoints.
    """
    if len(pred) < 2:
        return np.nan
    pred_dir   = np.sign(np.diff(pred))
    actual_dir = np.sign(np.diff(actual))
    valid = (pred_dir != 0) & (actual_dir != 0)
    if valid.sum() == 0:
        return np.nan
    return float((pred_dir[valid] == actual_dir[valid]).mean())


def trajectory_metrics(
    pred:   np.ndarray,
    actual: np.ndarray,
) -> Dict[str, float]:
    """All trajectory metrics for one (cluster, predict_weeks) pair."""
    err = pred - actual
    return {
        "mae":          float(np.mean(np.abs(err))),
        "rmse":         float(np.sqrt(np.mean(err ** 2))),
        "pearson_r":    pearson_r(pred, actual),
        "dtw":          dtw_distance(pred, actual),
        "dir_accuracy": direction_accuracy(pred, actual),
    }


# ------------------------------------------------------------------
# Results-DataFrame helpers
# ------------------------------------------------------------------

class TCMetrics:
    """
    Accumulates per-scenario predictions and computes aggregate reports.

    Usage (eval loop):
        metrics = TCMetrics()
        for batch_output in ...:
            metrics.add(predictions, actuals, meta_list)
        metrics.report()
        df = metrics.to_dataframe()
    """

    def __init__(self):
        self._rows: List[Dict] = []

    def add(
        self,
        predictions: np.ndarray,    # [N_query] predicted logFC values
        actuals:     np.ndarray,    # [N_query] actual logFC values
        meta:        Dict,          # tissue, omic, sex, cluster_num, predict_week
        hist_means:  Optional[np.ndarray] = None,  # [N_query] historical means
    ):
        """Record one prediction scenario (one cluster × one predict_week)."""
        for i, (pred_val, act_val) in enumerate(zip(predictions, actuals)):
            hm = float(hist_means[i]) if hist_means is not None else float("nan")
            self._rows.append({
                "tissue":       meta.get("tissue",       ""),
                "omic":         meta.get("omic",         ""),
                "sex":          meta.get("sex",          ""),
                "cluster_num":  meta.get("cluster_num",  -1),
                "predict_week": meta.get("predict_week", ""),
                "predicted":    _safe(pred_val),
                "actual":       _safe(act_val),
                "error":        _safe(pred_val - act_val),
                "hist_mean":    hm,
            })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def report(self, groupby: Optional[List[str]] = None) -> Dict:
        """
        Print a summary report.  groupby slices the report further
        (e.g. groupby=["predict_week"] gives per-week breakdown).
        Returns the top-level aggregate metrics dict.
        """
        df = self.to_dataframe()
        if df.empty:
            print("[TCMetrics] No predictions recorded.")
            return {}

        print("\n" + "=" * 60)
        print("ScgptTCPipeline — Evaluation Report")
        print("=" * 60)

        top = point_metrics(df["error"], df["actual"])
        print(f"  Total predictions : {top['count']}")
        print(f"  MAE               : {top['mae']:.4f}")
        print(f"  RMSE              : {top['rmse']:.4f}")
        if np.isfinite(top["rrmse"]):
            print(f"  rRMSE             : {top['rrmse']:.4f}")
        if np.isfinite(top["r2"]):
            print(f"  R²                : {top['r2']:.4f}"
                  f"  {'(beats mean baseline)' if top['r2'] > 0 else '(worse than mean baseline)'}")

        # l2_ratio: ||errors|| / ||actual - hist_mean||
        if "hist_mean" in df.columns:
            baselines = df["actual"] - df["hist_mean"]
            lr = l2_ratio(df["error"].values, baselines.values)
            if np.isfinite(lr):
                print(f"  l2_ratio          : {lr:.4f}"
                      f"  {'(beats hist-mean baseline)' if lr < 1 else '(worse than hist-mean baseline)'}")

        # Per predict_week breakdown (always shown)
        print("\n  --- Per predict_week ---")
        for wk, grp in df.groupby("predict_week"):
            m = point_metrics(grp["error"], grp["actual"])
            lr_str = ""
            if "hist_mean" in grp.columns:
                baselines = grp["actual"] - grp["hist_mean"]
                lr = l2_ratio(grp["error"].values, baselines.values)
                if np.isfinite(lr):
                    lr_str = f"  l2_ratio={lr:.4f}"
            print(f"    {wk}: N={m['count']:4d}  MAE={m['mae']:.4f}  "
                  f"RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}{lr_str}")

        # Optional extra groupby
        if groupby:
            print(f"\n  --- Per {' × '.join(groupby)} ---")
            for keys, grp in df.groupby(groupby):
                m     = point_metrics(grp["error"], grp["actual"])
                label = keys if isinstance(keys, str) else " | ".join(str(k) for k in keys)
                r2_str = f"  R²={m['r2']:.4f}" if np.isfinite(m["r2"]) else ""
                lr_str = ""
                if "hist_mean" in grp.columns:
                    baselines = grp["actual"] - grp["hist_mean"]
                    lr = l2_ratio(grp["error"].values, baselines.values)
                    if np.isfinite(lr):
                        lr_str = f"  l2_ratio={lr:.4f}"
                print(f"    {label}: N={m['count']:4d}  MAE={m['mae']:.4f}  "
                      f"RMSE={m['rmse']:.4f}{r2_str}{lr_str}")

        # Trajectory metrics: group by (tissue, omic, sex, cluster_num, predict_week)
        traj_rows = []
        grp_keys = ["tissue", "omic", "sex", "cluster_num", "predict_week"]
        for keys, grp in df.groupby(grp_keys):
            if len(grp) < 1:
                continue
            tm = trajectory_metrics(
                grp["predicted"].values, grp["actual"].values
            )
            traj_rows.append(dict(zip(grp_keys, keys)) | tm)

        if traj_rows:
            traj_df = pd.DataFrame(traj_rows)
            print("\n  --- Trajectory Metrics (mean across scenarios) ---")
            for col in ["pearson_r", "dtw", "dir_accuracy"]:
                vals = traj_df[col].dropna()
                if len(vals):
                    print(f"    {col:14s}: {vals.mean():.4f}  (std={vals.std():.4f})")

        print("=" * 60 + "\n")
        return top

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        print(f"[TCMetrics] Results saved to {path}")
