"""
TCSeqDataset — PyTorch Dataset for TCTransformerModel training and evaluation.

Each dataset item is one prediction scenario:

    target:   one cluster in (tissue, omic, sex) at a specific predict_week
    context:  all clusters from one or more (tissue, omic) combinations
              at the earlier (causal) timepoints

Sequence layout per item  [total length ≤ MAX_SEQ_LEN]:
    pos 0           : CLS token  (feature_id=0, time_id=0, modality_id=0,
                                  value=0, is_masked=False)
    pos 1 … C       : context tokens  — real values, is_masked=False
    pos C+1 … C+Q   : query tokens    — value=0,    is_masked=True

The dataset is indexed over all
    (tissue, omic, sex, cluster_num, predict_week)
tuples present in the data, so one full epoch covers every prediction
scenario.  Context modalities are randomly sampled each call during
training (fixed during evaluation).

Returned tensors per item:
    feature_ids  [S]  long
    time_ids     [S]  long
    modality_ids [S]  long
    values       [S]  float   (0.0 at query positions)
    is_masked    [S]  bool    (True at query positions)
    targets      [S]  float   (actual logFC at query positions, 0 elsewhere)
    meta         dict  — tissue / omic / sex / cluster / predict_week (for eval)

collate_fn pads all tensors to the longest sequence in the batch.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ScgptTCPipeline.data.loader import (
    TCSeqLoader, VALID_OMICS, VALID_WEEKS, WEEK_ORDER
)
from ScgptTCPipeline.model.feature_vocab import FeatureVocab
from ScgptTCPipeline.model.token_encoder import MODALITY_TO_ID, TIME_TO_ID

MAX_SEQ_LEN = 512


# ------------------------------------------------------------------
# Index entry
# ------------------------------------------------------------------

class _ScenarioIndex:
    """One prediction scenario entry.
    gene_id=None  → cluster-level mode (predict centroid)
    gene_id=str   → gene-level mode   (predict individual gene)
    """
    __slots__ = ("tissue", "omic", "sex", "cluster_num", "predict_week", "gene_id")

    def __init__(self, tissue, omic, sex, cluster_num, predict_week, gene_id=None):
        self.tissue       = tissue
        self.omic         = omic
        self.sex          = sex
        self.cluster_num  = cluster_num
        self.predict_week = predict_week
        self.gene_id      = gene_id


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class TCSeqDataset(Dataset):
    """
    Args:
        loader:              TCSeqLoader pointing at the data directory.
        vocab:               FeatureVocab already built from the same data.
        sex:                 which sex to use ('male', 'female', 'both').
        tissue:              restrict to one tissue (None = all).
        context_omics:       list of omics to use as context.  None = all available.
        fixed_predict_week:  if set, only generate scenarios for this week
                             (useful for single-task evaluation).
        fixed_context_omics: if set, always use exactly these omics as context
                             (disables random omic sampling — use for eval).
        random_omic_context: during training, randomly drop context omics
                             to force the model to handle missing modalities.
        seed:                base random seed (each __getitem__ derives its
                             own seed from index + seed for reproducibility).
    """

    def __init__(
        self,
        loader:               TCSeqLoader,
        vocab:                FeatureVocab,
        sex:                  str  = "male",
        tissue:               Optional[str]        = None,
        context_omics:        Optional[List[str]]  = None,
        target_omics:         Optional[List[str]]  = None,
        fixed_predict_week:   Optional[str]        = None,
        fixed_context_omics:  Optional[List[str]]  = None,
        random_omic_context:  bool = True,
        gene_level:           bool = False,
        seed:                 int  = 42,
    ):
        self.loader              = loader
        self.vocab               = vocab
        self.sex                 = sex
        self.context_omics       = [o.upper() for o in context_omics] \
                                    if context_omics else \
                                    [o.upper() for o in VALID_OMICS]
        self.target_omics        = [o.upper() for o in target_omics] \
                                    if target_omics else None  # None = all omics
        self.fixed_predict_week  = fixed_predict_week
        self.fixed_context_omics = [o.upper() for o in fixed_context_omics] \
                                    if fixed_context_omics else None
        self.random_omic_context = random_omic_context
        self.gene_level          = gene_level
        self.seed                = seed

        # Pre-load all centroid DataFrames into memory
        sexes = ["male", "female"] if sex == "both" else [sex]
        self._centroids: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        for t, o, s in loader.available_combinations():
            if s not in sexes:
                continue
            if tissue and t != tissue.upper():
                continue
            try:
                self._centroids[(t, o, s)] = loader.centroids(t, o, s)
            except Exception as e:
                print(f"[TCSeqDataset] Warning: could not load {t}|{o}|{s}: {e}")

        # In gene-level mode also cache full gene DataFrames (indexed by gene)
        self._gene_data: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        if gene_level:
            for (t, o, s) in self._centroids:
                try:
                    df = loader.load(t, o, s)
                    df = df.set_index("gene")
                    self._gene_data[(t, o, s)] = df
                except Exception as e:
                    print(f"[TCSeqDataset] Warning gene data {t}|{o}|{s}: {e}")

        # Build flat index.
        # Cluster-level: one entry per (tissue, omic, sex, cluster_num, predict_week).
        # Gene-level:    one entry per (tissue, omic, sex, gene_id,     predict_week).
        # Context always uses cluster centroids; only the query differs.
        predict_weeks = [fixed_predict_week] if fixed_predict_week \
                        else ["2w", "4w", "8w"]
        self._index: List[_ScenarioIndex] = []
        for (t, o, s), centroids_df in self._centroids.items():
            if self.target_omics is not None and o not in self.target_omics:
                continue

            if not gene_level:
                # Cluster-level: one entry per cluster
                for c_num in centroids_df.index:
                    for pw in predict_weeks:
                        if pd.notna(centroids_df.loc[c_num, pw]):
                            self._index.append(
                                _ScenarioIndex(t, o, s, int(c_num), pw))
            else:
                # Gene-level: one entry per gene within each cluster
                gene_df = self._gene_data.get((t, o, s))
                if gene_df is None:
                    continue
                for c_num in centroids_df.index:
                    cluster_genes = gene_df[gene_df["cluster"] == c_num]
                    for gene_id in cluster_genes.index:
                        for pw in predict_weeks:
                            val = cluster_genes.loc[gene_id, pw] \
                                  if pw in cluster_genes.columns else float("nan")
                            if pd.notna(val):
                                self._index.append(
                                    _ScenarioIndex(t, o, s, int(c_num), pw,
                                                   gene_id=gene_id))

        print(f"[TCSeqDataset] {len(self._index)} prediction scenarios "
              f"({len(self._centroids)} tissue|omic|sex combinations)")

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Core item builder
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sc   = self._index[idx]
        rng  = random.Random(self.seed + idx)   # reproducible per-item RNG

        ctx_weeks = TCSeqLoader.context_weeks(sc.predict_week)  # causal earlier weeks

        # --- Choose context omics for this scenario ---
        if self.fixed_context_omics is not None:
            ctx_omics = self.fixed_context_omics
        elif self.random_omic_context:
            # Always include the target omic; randomly include others
            available = [o for o in self.context_omics
                         if (sc.tissue, o, sc.sex) in self._centroids]
            others    = [o for o in available if o != sc.omic]
            n_others  = rng.randint(0, len(others))
            ctx_omics = [sc.omic] + rng.sample(others, n_others)
        else:
            ctx_omics = [o for o in self.context_omics
                         if (sc.tissue, o, sc.sex) in self._centroids]

        # ------------------------------------------------------------------
        # Build token lists
        # ------------------------------------------------------------------
        feature_ids_list  = []
        time_ids_list     = []
        modality_ids_list = []
        values_list       = []
        is_masked_list    = []
        targets_list      = []

        # --- CLS token ---
        feature_ids_list.append(self.vocab.cls_id())
        time_ids_list.append(0)
        modality_ids_list.append(0)
        values_list.append(0.0)
        is_masked_list.append(False)
        targets_list.append(0.0)

        # --- Context tokens: all clusters × all context weeks × context omics ---
        # For gene-level mode: the target gene's own cluster uses the gene's
        # individual historical values instead of the cluster centroid, so each
        # gene gets a distinct context and therefore a distinct prediction.
        for omic in ctx_omics:
            mod_id    = MODALITY_TO_ID.get(omic, 0)
            cent_df   = self._centroids.get((sc.tissue, omic, sc.sex))
            if cent_df is None:
                continue
            for c_num in sorted(cent_df.index):
                try:
                    feat_id = self.vocab.cluster_id(sc.tissue, omic, sc.sex, c_num)
                except KeyError:
                    continue
                for wk in ctx_weeks:
                    # Gene-level: substitute gene's own value for its cluster/omic
                    if (sc.gene_id is not None
                            and c_num == sc.cluster_num
                            and omic == sc.omic):
                        gene_df = self._gene_data.get((sc.tissue, omic, sc.sex))
                        if (gene_df is not None
                                and sc.gene_id in gene_df.index
                                and wk in gene_df.columns):
                            val = gene_df.loc[sc.gene_id, wk]
                        else:
                            val = cent_df.loc[c_num, wk]
                    else:
                        val = cent_df.loc[c_num, wk]
                    val = 0.0 if pd.isna(val) else float(val)
                    feature_ids_list.append(feat_id)
                    time_ids_list.append(TIME_TO_ID[wk])
                    modality_ids_list.append(mod_id)
                    values_list.append(val)
                    is_masked_list.append(False)
                    targets_list.append(0.0)

        # --- Query token: target at predict_week (masked) ---
        # Cluster-level: predict the cluster centroid.
        # Gene-level:    predict the individual gene value.
        tgt_mod_id  = MODALITY_TO_ID.get(sc.omic, 0)
        tgt_time_id = TIME_TO_ID[sc.predict_week]

        if sc.gene_id is None:
            # --- Cluster-level query ---
            tgt_cent_df = self._centroids[(sc.tissue, sc.omic, sc.sex)]
            try:
                feat_id = self.vocab.cluster_id(
                    sc.tissue, sc.omic, sc.sex, sc.cluster_num)
            except KeyError:
                feat_id = self.vocab.cls_id()
            actual    = tgt_cent_df.loc[sc.cluster_num, sc.predict_week]
            ctx_vals  = [tgt_cent_df.loc[sc.cluster_num, w]
                         for w in ctx_weeks if w in tgt_cent_df.columns]
        else:
            # --- Gene-level query ---
            # Use the cluster token (not a gene-specific token) so the model
            # predicts the cluster response; target is the individual gene value.
            # This evaluates how well cluster-level predictions proxy gene-level behaviour.
            gene_df = self._gene_data[(sc.tissue, sc.omic, sc.sex)]
            try:
                feat_id = self.vocab.cluster_id(
                    sc.tissue, sc.omic, sc.sex, sc.cluster_num)
            except KeyError:
                feat_id = self.vocab.cls_id()
            actual   = gene_df.loc[sc.gene_id, sc.predict_week] \
                       if sc.predict_week in gene_df.columns else float("nan")
            ctx_vals = [gene_df.loc[sc.gene_id, w]
                        for w in ctx_weeks if w in gene_df.columns]

        actual    = 0.0 if pd.isna(actual) else float(actual)
        hist_mean = float(np.nanmean(ctx_vals)) if ctx_vals else 0.0

        feature_ids_list.append(feat_id)
        time_ids_list.append(tgt_time_id)
        modality_ids_list.append(tgt_mod_id)
        values_list.append(0.0)             # masked — value unknown at inference
        is_masked_list.append(True)
        targets_list.append(actual)

        # hist_means: same length as sequence, non-zero only at the query position
        # used for l2_ratio = ||actual - predicted|| / ||actual - hist_mean||
        hist_means_list = [0.0] * (len(targets_list) - 1) + [hist_mean]

        # --- Truncate if over MAX_SEQ_LEN ---
        # Keep CLS (pos 0) + all query tokens (end); trim context in the middle
        if len(feature_ids_list) > MAX_SEQ_LEN:
            n_query   = sum(is_masked_list)
            keep_ctx  = MAX_SEQ_LEN - 1 - n_query
            def trim(lst):
                return [lst[0]] + lst[1: 1 + keep_ctx] + lst[-n_query:]
            feature_ids_list  = trim(feature_ids_list)
            time_ids_list     = trim(time_ids_list)
            modality_ids_list = trim(modality_ids_list)
            values_list       = trim(values_list)
            is_masked_list    = trim(is_masked_list)
            targets_list      = trim(targets_list)
            hist_means_list   = trim(hist_means_list)

        return {
            "feature_ids":  torch.tensor(feature_ids_list,  dtype=torch.long),
            "time_ids":     torch.tensor(time_ids_list,     dtype=torch.long),
            "modality_ids": torch.tensor(modality_ids_list, dtype=torch.long),
            "values":       torch.tensor(values_list,       dtype=torch.float),
            "is_masked":    torch.tensor(is_masked_list,    dtype=torch.bool),
            "targets":      torch.tensor(targets_list,      dtype=torch.float),
            "hist_means":   torch.tensor(hist_means_list,   dtype=torch.float),
            "meta": {
                "tissue":       sc.tissue,
                "omic":         sc.omic,
                "sex":          sc.sex,
                "cluster_num":  sc.cluster_num,
                "predict_week": sc.predict_week,
                "gene_id":      sc.gene_id or "",
            },
        }


# ------------------------------------------------------------------
# Collate function for DataLoader
# ------------------------------------------------------------------

def collate_tc(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pads all variable-length sequences in a batch to the same length.
    padding_mask is True at padded positions (transformer ignores them).
    """
    max_len = max(b["feature_ids"].shape[0] for b in batch)

    f_ids, t_ids, m_ids, vals, masked, targets, hist_means, pad_masks = \
        [], [], [], [], [], [], [], []
    metas = [b["meta"] for b in batch]

    for b in batch:
        S   = b["feature_ids"].shape[0]
        pad = max_len - S

        f_ids.append(      torch.nn.functional.pad(b["feature_ids"],  (0, pad), value=0))
        t_ids.append(      torch.nn.functional.pad(b["time_ids"],     (0, pad), value=0))
        m_ids.append(      torch.nn.functional.pad(b["modality_ids"], (0, pad), value=0))
        vals.append(       torch.nn.functional.pad(b["values"],       (0, pad), value=0.0))
        masked.append(     torch.nn.functional.pad(b["is_masked"],    (0, pad), value=False))
        targets.append(    torch.nn.functional.pad(b["targets"],      (0, pad), value=0.0))
        hist_means.append( torch.nn.functional.pad(b["hist_means"],   (0, pad), value=0.0))

        pmask = torch.zeros(max_len, dtype=torch.bool)
        pmask[S:] = True
        pad_masks.append(pmask)

    return {
        "feature_ids":   torch.stack(f_ids),
        "time_ids":      torch.stack(t_ids),
        "modality_ids":  torch.stack(m_ids),
        "values":        torch.stack(vals),
        "is_masked":     torch.stack(masked),
        "targets":       torch.stack(targets),
        "hist_means":    torch.stack(hist_means),
        "padding_mask":  torch.stack(pad_masks),
        "meta":          metas,
    }
