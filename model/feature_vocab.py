"""
FeatureVocab — maps every cluster (and optionally every gene) to an integer
feature_id for use in TCTransformerModel's feature_embedding.

Vocabulary layout
-----------------
  ID 0        : <cls>  — CLS / padding token (padding_idx, kept at zero)
  ID 1 … C    : one entry per (tissue, omic, sex, cluster_num) — cluster tokens
  ID C+1 … N  : one entry per unique gene ID (optional, gene-level mode)

Cluster token key string: "{TISSUE}|{OMIC}|{sex}|{cluster_num}"
  e.g.  "HEART|TRNSCRPT|male|3"

Embedding initialisation from scGPT
------------------------------------
For TRNSCRPT and PROT clusters whose members map to human gene symbols
present in scGPT's vocab, we initialise the cluster embedding as the mean
of its member gene embeddings from the scGPT checkpoint.  This gives the
transformer a biologically meaningful starting point instead of random noise.

METAB and ATAC clusters (no direct scGPT gene mapping) are initialised
from the global mean of all scGPT embeddings + small Gaussian noise.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from ScgptTCPipeline.data.loader import TCSeqLoader


SPECIAL_TOKENS = {"<cls>": 0}


class FeatureVocab:
    """
    Builds and stores the feature vocabulary for TCTransformerModel.

    Usage
    -----
        vocab  = FeatureVocab.build(loader, sex="male")
        feat_id = vocab.cluster_id("HEART", "TRNSCRPT", "male", cluster_num=3)
        vocab.save("output/feature_vocab.json")

        # Later:
        vocab = FeatureVocab.load("output/feature_vocab.json")
    """

    def __init__(self):
        self._token_to_id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self._id_to_token: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        loader: TCSeqLoader,
        sex: str = "male",
        tissue: Optional[str] = None,
        include_genes: bool = False,
    ) -> "FeatureVocab":
        """
        Scan all available (tissue, omic, sex) combinations in the loader
        and register one token per cluster.  Optionally register individual
        gene IDs as additional tokens.

        Args:
            loader:        TCSeqLoader pointing at your data directory.
            sex:           which sex to build vocab for ('male', 'female', or 'both').
            tissue:        restrict to one tissue (None = all).
            include_genes: if True also register each unique gene ID as a token.
        """
        vocab = cls()
        sexes = ["male", "female"] if sex == "both" else [sex]

        for t, o, s in loader.available_combinations():
            if s not in sexes:
                continue
            if tissue and t != tissue.upper():
                continue

            df = loader.load(t, o, s)
            for c_num in sorted(df["cluster"].unique()):
                key = cls._cluster_key(t, o, s, int(c_num))
                vocab._register(key)

            if include_genes:
                for gene in df["gene"].unique():
                    key = f"GENE|{gene}"
                    if key not in vocab._token_to_id:
                        vocab._register(key)

        print(f"[FeatureVocab] Built vocab: {len(vocab)} tokens "
              f"({len(vocab) - len(SPECIAL_TOKENS)} feature tokens)")
        return vocab

    # ------------------------------------------------------------------
    # Token lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_key(tissue: str, omic: str, sex: str, cluster_num: int) -> str:
        return f"{tissue.upper()}|{omic.upper()}|{sex.lower()}|{cluster_num}"

    def cluster_id(
        self, tissue: str, omic: str, sex: str, cluster_num: int
    ) -> int:
        key = self._cluster_key(tissue, omic, sex, cluster_num)
        if key not in self._token_to_id:
            raise KeyError(
                f"Cluster token not in vocab: {key}\n"
                "Call FeatureVocab.build() first or check tissue/omic/sex."
            )
        return self._token_to_id[key]

    def gene_id(self, gene: str) -> int:
        key = f"GENE|{gene}"
        return self._token_to_id.get(key, SPECIAL_TOKENS["<cls>"])

    def cls_id(self) -> int:
        return SPECIAL_TOKENS["<cls>"]

    def __len__(self) -> int:
        return len(self._token_to_id)

    def _register(self, key: str):
        idx = len(self._token_to_id)
        self._token_to_id[key] = idx
        self._id_to_token[idx] = key

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._token_to_id, f, indent=2)
        print(f"[FeatureVocab] Saved {len(self)} tokens to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureVocab":
        with open(path) as f:
            token_to_id = json.load(f)
        vocab = cls()
        vocab._token_to_id = token_to_id
        vocab._id_to_token = {v: k for k, v in token_to_id.items()}
        print(f"[FeatureVocab] Loaded {len(vocab)} tokens from {path}")
        return vocab

    # ------------------------------------------------------------------
    # Embedding initialisation from scGPT
    # ------------------------------------------------------------------

    def init_embeddings_from_scgpt(
        self,
        embedding: torch.nn.Embedding,
        loader: TCSeqLoader,
        scgpt_vocab_path: str,
        scgpt_checkpoint_path: str,
        rat_to_human_cache_path: str = "data/rat_to_human_cache.json",
        sex: str = "male",
    ) -> int:
        """
        For every cluster token whose members have human-gene orthologues present
        in scGPT's vocab, set the cluster embedding = mean(member gene embeddings).

        Returns the number of cluster tokens successfully initialised.
        """
        if not os.path.exists(scgpt_vocab_path):
            print("[FeatureVocab] scGPT vocab not found — skipping embedding init.")
            return 0
        if not os.path.exists(scgpt_checkpoint_path):
            print("[FeatureVocab] scGPT checkpoint not found — skipping embedding init.")
            return 0

        # Load scGPT vocab and weights
        with open(scgpt_vocab_path) as f:
            scgpt_vocab: Dict[str, int] = json.load(f)

        ckpt = torch.load(scgpt_checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        scgpt_emb_key = next(
            (k for k in state if "encoder.embedding.weight" in k or
             k == "encoder.embedding.weight"), None
        )
        if scgpt_emb_key is None:
            print("[FeatureVocab] Could not locate encoder.embedding.weight in checkpoint.")
            return 0

        scgpt_emb_matrix = state[scgpt_emb_key].float()   # [V_scgpt, d_model]
        global_mean = scgpt_emb_matrix.mean(dim=0)         # fallback init

        # Load rat→human cache
        rat_to_human: Dict[str, str] = {}
        if os.path.exists(rat_to_human_cache_path):
            with open(rat_to_human_cache_path) as f:
                rat_to_human = json.load(f)

        n_inited = 0

        with torch.no_grad():
            for token_key, feat_id in self._token_to_id.items():
                if "|" not in token_key or token_key.startswith("GENE|"):
                    continue  # skip special and bare-gene tokens

                parts = token_key.split("|")
                if len(parts) != 4:
                    continue
                tissue, omic, sex_k, c_num_str = parts
                c_num = int(c_num_str)

                # Only TRNSCRPT / PROT have gene members mappable to scGPT
                if omic not in ("TRNSCRPT", "PROT"):
                    # Fallback: global mean + noise
                    noise = torch.randn_like(global_mean) * 0.02
                    embedding.weight.data[feat_id] = global_mean + noise
                    continue

                try:
                    members_df = loader.cluster_members(tissue, omic, sex_k, c_num)
                except Exception:
                    continue

                gene_vecs = []
                for rat_id in members_df["gene"].values:
                    human_sym = rat_to_human.get(rat_id, rat_id.upper())
                    scgpt_id  = scgpt_vocab.get(human_sym)
                    if scgpt_id is not None:
                        gene_vecs.append(scgpt_emb_matrix[scgpt_id])

                if gene_vecs:
                    cluster_vec = torch.stack(gene_vecs).mean(dim=0)
                    embedding.weight.data[feat_id] = cluster_vec
                    n_inited += 1
                else:
                    noise = torch.randn_like(global_mean) * 0.02
                    embedding.weight.data[feat_id] = global_mean + noise

        print(f"[FeatureVocab] Initialised {n_inited}/{len(self._token_to_id)} "
              "cluster embeddings from scGPT gene means.")
        return n_inited

    # ------------------------------------------------------------------
    # Helpers for dataset building
    # ------------------------------------------------------------------

    def cluster_ids_for(
        self, tissue: str, omic: str, sex: str
    ) -> List[Tuple[int, int]]:
        """
        Returns [(cluster_num, feature_id), ...] for all registered clusters
        of a given (tissue, omic, sex).
        """
        prefix = f"{tissue.upper()}|{omic.upper()}|{sex.lower()}|"
        result = []
        for key, fid in self._token_to_id.items():
            if key.startswith(prefix):
                c_num = int(key.split("|")[-1])
                result.append((c_num, fid))
        return sorted(result)
