"""
Token encoding components for TCTransformerModel.

Each position in the transformer sequence represents one
(feature, timepoint, modality) triplet.  The combined token embedding is:

    token = FeatureEmbedding[feature_id]
          + TimeEmbedding[time_id]
          + ModalityEmbedding[modality_id]
          + ScalarValueEncoder(value)   ← real value, OR
          + MaskToken                   ← learned vector when value is unknown

All sub-embeddings share the same dimension d_model so they can be summed.

Modality IDs  (MODALITY_TO_ID):
    0 = TRNSCRPT
    1 = PROT
    2 = METAB
    3 = ATAC

Timepoint IDs (TIME_TO_ID):
    0 = 1w  |  1 = 2w  |  2 = 4w  |  3 = 8w
"""

import torch
import torch.nn as nn
from typing import Optional

# ------------------------------------------------------------------
# Global ID maps  (used here and in tc_dataset.py)
# ------------------------------------------------------------------

MODALITY_TO_ID = {"TRNSCRPT": 0, "PROT": 1, "METAB": 2, "ATAC": 3}
TIME_TO_ID     = {"1w": 0, "2w": 1, "4w": 2, "8w": 3}
N_MODALITIES   = len(MODALITY_TO_ID)
N_TIMEPOINTS   = len(TIME_TO_ID)


# ------------------------------------------------------------------
# Sub-embedding modules
# ------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """Learnable embedding for the 4 exercise timepoints."""

    def __init__(self, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(N_TIMEPOINTS, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, time_ids: torch.Tensor) -> torch.Tensor:
        # time_ids: [B, S]  → [B, S, d_model]
        return self.emb(time_ids)


class ModalityEmbedding(nn.Module):
    """Learnable embedding for omic modality (TRNSCRPT / PROT / METAB / ATAC)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(N_MODALITIES, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        # modality_ids: [B, S]  → [B, S, d_model]
        return self.emb(modality_ids)


class ScalarValueEncoder(nn.Module):
    """
    Projects a single scalar logFC value → d_model.

    Unlike scGPT's ContinuousValueEncoder (which clamps to [0, max] because
    expression counts are non-negative), logFC values are signed, so we skip
    the clamp and use tanh-normalisation instead.

    Architecture: Linear(1 → d_model) → GELU → Linear(d_model → d_model) → LayerNorm
    """

    def __init__(self, d_model: int, logfc_scale: float = 3.0):
        super().__init__()
        # logfc_scale: typical |logFC| range for z-scored data; used for soft normalisation
        self.logfc_scale = logfc_scale
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: [B, S]  (raw logFC floats)
        # soft-normalise to roughly [-1, 1] before projection
        x = torch.tanh(values / self.logfc_scale)   # [B, S]
        x = x.unsqueeze(-1)                          # [B, S, 1]
        return self.norm(self.net(x))                # [B, S, d_model]


# ------------------------------------------------------------------
# Mask token
# ------------------------------------------------------------------

class MaskToken(nn.Module):
    """
    Learned d_model-dimensional vector injected at query (unknown) positions
    instead of ScalarValueEncoder output.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.token, mean=0.0, std=0.02)

    def expand(self, batch: int, seq: int, device: torch.device) -> torch.Tensor:
        # Returns [B, S, d_model] filled with the learned mask vector
        return self.token.unsqueeze(0).unsqueeze(0).expand(batch, seq, -1).to(device)


# ------------------------------------------------------------------
# Combined token encoder
# ------------------------------------------------------------------

class TCTokenEncoder(nn.Module):
    """
    Builds the full [B, S, d_model] input tensor for the transformer.

    For each position i in the sequence:
        token[i] = feature_emb[feature_id[i]]
                 + time_emb[time_id[i]]
                 + modality_emb[modality_id[i]]
                 + ( ScalarValueEncoder(value[i])  if not masked[i]
                     else  MaskToken               )

    Args:
        feature_embedding: the nn.Embedding from TCTransformerModel (shared,
                           so pretrained scGPT weights can be reused for it).
        d_model:           transformer hidden dimension.
        logfc_scale:       soft-normalisation scale for logFC values.
    """

    def __init__(
        self,
        feature_embedding: nn.Embedding,
        d_model: int,
        logfc_scale: float = 3.0,
    ):
        super().__init__()
        self.feature_emb  = feature_embedding          # shared with transformer
        self.time_emb     = TimeEmbedding(d_model)
        self.modality_emb = ModalityEmbedding(d_model)
        self.value_enc    = ScalarValueEncoder(d_model, logfc_scale)
        self.mask_token   = MaskToken(d_model)
        self.norm         = nn.LayerNorm(d_model)

    def forward(
        self,
        feature_ids:  torch.Tensor,   # [B, S]  long
        time_ids:     torch.Tensor,   # [B, S]  long
        modality_ids: torch.Tensor,   # [B, S]  long
        values:       torch.Tensor,   # [B, S]  float  (0.0 at masked positions)
        is_masked:    torch.Tensor,   # [B, S]  bool   (True = query / unknown)
    ) -> torch.Tensor:                # [B, S, d_model]

        B, S = feature_ids.shape
        device = feature_ids.device

        feat  = self.feature_emb(feature_ids)          # [B, S, d_model]
        time  = self.time_emb(time_ids)                # [B, S, d_model]
        mod   = self.modality_emb(modality_ids)        # [B, S, d_model]

        # Value branch: real encoding for context, mask token for queries
        val_enc  = self.value_enc(values)              # [B, S, d_model]
        mask_enc = self.mask_token.expand(B, S, device)# [B, S, d_model]

        # Select per-position: masked positions get mask_token, rest get val_enc
        # is_masked: [B, S] → [B, S, 1] for broadcasting
        sel = is_masked.unsqueeze(-1).float()
        value_part = val_enc * (1.0 - sel) + mask_enc * sel  # [B, S, d_model]

        return self.norm(feat + time + mod + value_part)      # [B, S, d_model]
