"""
TCTransformerModel — scGPT backbone adapted for multimodal time-series prediction.

Design principles:
  - Standalone nn.Module (does NOT subclass scGPT internals to avoid API coupling).
  - Shares transformer architecture with scGPT (d_model=512, nhead=8, nlayers=12).
  - Provides load_scgpt_weights() to copy the pretrained transformer + gene embedding.
  - Input:  structured (feature_id, time_id, modality_id, value, is_masked) per position.
  - Output: scalar logFC prediction at every masked (query) position.

Sequence layout fed to the transformer:
    [CLS]  ctx_tok_0  ctx_tok_1  ...  query_tok_0  query_tok_1  ...
      ↑ position 0 — its output is the global sequence / "cell" embedding
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from ScgptTCPipeline.model.token_encoder import TCTokenEncoder


# ------------------------------------------------------------------
# Output head
# ------------------------------------------------------------------

class TemporalDecoder(nn.Module):
    """
    Maps transformer hidden state [d_model] → scalar logFC prediction.

    Applied only at query (masked) positions during training;
    at all positions during inference if needed.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, S, d_model]  → [B, S]
        return self.net(hidden).squeeze(-1)


# ------------------------------------------------------------------
# Main model
# ------------------------------------------------------------------

class TCTransformerModel(nn.Module):
    """
    Multimodal time-series transformer for exercise-response prediction.

    Args:
        n_features:   vocabulary size (number of distinct cluster / gene IDs + specials).
        d_model:      transformer hidden dimension (512 to match scGPT whole-human).
        nhead:        number of attention heads (8).
        d_hid:        feedforward hidden dimension (512).
        nlayers:      transformer depth (12).
        dropout:      dropout rate.
        logfc_scale:  soft-normalisation scale for logFC values (default 3.0).
    """

    CLS_ID = 0   # feature_id reserved for the CLS token

    def __init__(
        self,
        n_features:  int,
        d_model:     int   = 512,
        nhead:       int   = 8,
        d_hid:       int   = 512,
        nlayers:     int   = 12,
        dropout:     float = 0.1,
        logfc_scale: float = 3.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Feature embedding — shared with TCTokenEncoder
        # padding_idx=0 keeps CLS token gradient-free when used as padding
        self.feature_embedding = nn.Embedding(n_features, d_model, padding_idx=0)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=0.02)

        # Token encoder: combines feature + time + modality + value/mask
        self.token_encoder = TCTokenEncoder(
            feature_embedding=self.feature_embedding,
            d_model=d_model,
            logfc_scale=logfc_scale,
        )

        # Transformer backbone (same architecture as scGPT whole-human)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LayerNorm, matches scGPT default
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayers,
            norm=nn.LayerNorm(d_model),
        )

        # Output decoder
        self.decoder = TemporalDecoder(d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feature_ids:    torch.Tensor,          # [B, S]   long
        time_ids:       torch.Tensor,          # [B, S]   long
        modality_ids:   torch.Tensor,          # [B, S]   long
        values:         torch.Tensor,          # [B, S]   float
        is_masked:      torch.Tensor,          # [B, S]   bool  — True = query
        padding_mask:   Optional[torch.Tensor] = None,  # [B, S] bool — True = pad
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            predictions:  [B, S]      logFC predictions at every position
            query_preds:  [B, S]      predictions zeroed-out at context positions
            cell_emb:     [B, d_model] CLS-token embedding (global sequence repr.)
        """
        # 1. Build token embeddings
        tokens = self.token_encoder(
            feature_ids, time_ids, modality_ids, values, is_masked
        )                                       # [B, S, d_model]

        # 2. Run transformer
        hidden = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask,  # True positions are ignored
        )                                       # [B, S, d_model]

        # 3. Decode at all positions; caller can mask to query positions
        predictions = self.decoder(hidden)      # [B, S]

        # 4. Zero predictions at context positions (only query positions matter for loss)
        query_preds = predictions * is_masked.float()

        # 5. CLS token embedding (position 0)
        cell_emb = hidden[:, 0, :]              # [B, d_model]

        return {
            "predictions": predictions,
            "query_preds": query_preds,
            "cell_emb":    cell_emb,
            "hidden":      hidden,
        }

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def load_scgpt_weights(self, checkpoint_path: str, freeze_n_layers: int = 8):
        """
        Copy compatible weights from a scGPT checkpoint into this model.

        Copies:
          - transformer encoder layers (all nlayers if shapes match)
          - gene embedding weights (for features that overlap with scGPT vocab)

        Args:
            checkpoint_path: path to scGPT best_model.pt
            freeze_n_layers: freeze the first N transformer layers after loading
                             (unfreeze later via unfreeze_layers()).
        """
        if not os.path.exists(checkpoint_path):
            print(f"[TCTransformerModel] Checkpoint not found: {checkpoint_path}. "
                  "Training from scratch.")
            return

        print(f"[TCTransformerModel] Loading scGPT weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)

        # --- Transformer layers ---
        matched, skipped = 0, 0
        own_state = self.state_dict()
        remap = {
            # scGPT key prefix  →  our key prefix
            "transformer_encoder.layers": "transformer.layers",
            "transformer_encoder.norm":   "transformer.norm",
        }
        translated = {}
        for k, v in state.items():
            new_k = k
            for old_pfx, new_pfx in remap.items():
                if k.startswith(old_pfx):
                    new_k = new_pfx + k[len(old_pfx):]
                    break
            if new_k in own_state and own_state[new_k].shape == v.shape:
                translated[new_k] = v
                matched += 1
            else:
                skipped += 1

        self.load_state_dict(translated, strict=False)
        print(f"[TCTransformerModel] Loaded {matched} weight tensors "
              f"({skipped} skipped — shape mismatch or new layers).")

        # --- Freeze first N transformer layers ---
        self.freeze_layers(n=freeze_n_layers)

    def freeze_layers(self, n: int):
        """Freeze the first n transformer encoder layers."""
        for i, layer in enumerate(self.transformer.layers):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False
        print(f"[TCTransformerModel] Froze first {n} transformer layers.")

    def unfreeze_layers(self, n: Optional[int] = None):
        """Unfreeze transformer layers. If n is None, unfreeze all."""
        for i, layer in enumerate(self.transformer.layers):
            if n is None or i >= (len(self.transformer.layers) - n):
                for p in layer.parameters():
                    p.requires_grad = True
        label = "all" if n is None else f"last {n}"
        print(f"[TCTransformerModel] Unfroze {label} transformer layers.")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self):
        total   = self.total_params()
        trainable = self.trainable_params()
        print(f"[TCTransformerModel] Parameters: "
              f"{trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.1f}%)")
