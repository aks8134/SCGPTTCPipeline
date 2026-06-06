"""
ScgptTCPipeline — Fine-tuning script.

Training pipeline:
  1. Build FeatureVocab from disk (all available tissue/omic/sex combos).
  2. Initialise TCTransformerModel.
  3. Load scGPT pretrained transformer weights (optional but recommended).
  4. Initialise cluster embeddings from scGPT gene-mean vectors.
  5. Train with MSE loss on masked (query) positions only.
  6. Freeze/unfreeze schedule: bottom N layers frozen for first K epochs,
     then progressively opened.
  7. Save best checkpoint (lowest val loss).

Usage:
    python -m ScgptTCPipeline.finetune \\
        --data_dir  data/TCSeqData_n_20 \\
        --scgpt_dir models/scgpt-human-whole-body \\
        --output_dir output/scgpt_tc \\
        --epochs 30 --batch_size 32 --sex male

    # Restrict to one tissue for a quick run:
    python -m ScgptTCPipeline.finetune --tissue HEART --epochs 10
"""

import argparse
import os
import sys
import math

import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ScgptTCPipeline.data.loader      import TCSeqLoader
from ScgptTCPipeline.model.feature_vocab  import FeatureVocab
from ScgptTCPipeline.model.tc_transformer import TCTransformerModel
from ScgptTCPipeline.dataset.tc_dataset   import TCSeqDataset, collate_tc
from ScgptTCPipeline.evaluation.metrics   import TCMetrics, point_metrics


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

def masked_mse(predictions: torch.Tensor,
               targets:     torch.Tensor,
               is_masked:   torch.Tensor,
               hist_means:  torch.Tensor = None) -> torch.Tensor:
    """MSE computed only at query (masked) positions."""
    mask = is_masked.float()
    n    = mask.sum().clamp(min=1)
    return ((predictions - targets) ** 2 * mask).sum() / n


def masked_l2_ratio(predictions: torch.Tensor,
                    targets:     torch.Tensor,
                    is_masked:   torch.Tensor,
                    hist_means:  torch.Tensor,
                    eps:         float = 1e-6) -> torch.Tensor:
    """
    Differentiable l2_ratio loss computed only at query (masked) positions.

        loss = ||pred - actual||_2 / ||actual - hist_mean||_2

    The denominator is constant w.r.t. model parameters, so gradients flow
    only through the numerator. eps prevents division by zero when all
    actual values equal their historical mean (flat trajectory clusters).
    Falls back to pure MSE scaling when denominator is degenerate.
    """
    mask   = is_masked.float()                          # [B, S]
    errors = (predictions - targets) * mask             # zero at non-query positions
    baseln = (targets - hist_means) * mask

    num   = torch.sqrt((errors ** 2).sum() + eps)
    denom = torch.sqrt((baseln ** 2).sum() + eps)

    return num / denom


LOSS_FNS = {
    "mse":      masked_mse,
    "l2_ratio": masked_l2_ratio,
}


# ------------------------------------------------------------------
# Training epoch
# ------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool,
              loss_fn=masked_mse) -> float:
    model.train(train)
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            feature_ids  = batch["feature_ids"].to(device)
            time_ids     = batch["time_ids"].to(device)
            modality_ids = batch["modality_ids"].to(device)
            values       = batch["values"].to(device)
            is_masked    = batch["is_masked"].to(device)
            targets      = batch["targets"].to(device)
            hist_means   = batch["hist_means"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            out  = model(feature_ids, time_ids, modality_ids,
                         values, is_masked, padding_mask)
            loss = loss_fn(out["predictions"], targets, is_masked, hist_means)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ------------------------------------------------------------------
# Validation with full metric report
# ------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, loss_fn=masked_mse) -> float:
    """Returns val loss; also computes MAE for logging."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_actuals = [], []

    for batch in loader:
        feature_ids  = batch["feature_ids"].to(device)
        time_ids     = batch["time_ids"].to(device)
        modality_ids = batch["modality_ids"].to(device)
        values       = batch["values"].to(device)
        is_masked    = batch["is_masked"].to(device)
        targets      = batch["targets"].to(device)
        hist_means   = batch["hist_means"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        out  = model(feature_ids, time_ids, modality_ids,
                     values, is_masked, padding_mask)
        loss = loss_fn(out["predictions"], targets, is_masked, hist_means)
        total_loss += loss.item()
        n_batches  += 1

        mask    = is_masked.cpu()
        preds   = out["predictions"].cpu()[mask].numpy()
        actuals = targets.cpu()[mask].numpy()
        all_preds.extend(preds.tolist())
        all_actuals.extend(actuals.tolist())

    avg_loss = total_loss / max(n_batches, 1)
    errors   = [p - a for p, a in zip(all_preds, all_actuals)]
    m        = point_metrics(errors, all_actuals)
    print(f"    Val loss={avg_loss:.4f}  MAE={m['mae']:.4f}  "
          f"RMSE={m['rmse']:.4f}  rRMSE={m['rrmse']:.4f}")
    return avg_loss


# ------------------------------------------------------------------
# Main training entry point
# ------------------------------------------------------------------

def train(args):
    print("=" * 70)
    print("ScgptTCPipeline — Fine-tuning")
    for k, v in vars(args).items():
        print(f"  {k:<22s}: {v}")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n[Info] Device: {device}")

    # ------------------------------------------------------------------
    # 1. Loader + Vocab
    # ------------------------------------------------------------------
    loader = TCSeqLoader(data_dir=args.data_dir)

    vocab_path = os.path.join(args.output_dir, "feature_vocab.json")
    if args.reload_vocab and os.path.exists(vocab_path):
        vocab = FeatureVocab.load(vocab_path)
    else:
        vocab = FeatureVocab.build(loader, sex=args.sex, tissue=args.tissue,
                                   include_genes=args.gene_level)
        os.makedirs(args.output_dir, exist_ok=True)
        vocab.save(vocab_path)

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = TCTransformerModel(
        n_features  = len(vocab),
        d_model     = args.d_model,
        nhead       = args.nhead,
        d_hid       = args.d_hid,
        nlayers     = args.nlayers,
        dropout     = args.dropout,
        logfc_scale = args.logfc_scale,
    ).to(device)

    # ------------------------------------------------------------------
    # 3. Load scGPT pretrained weights
    # ------------------------------------------------------------------
    if args.scgpt_dir:
        ckpt_path  = os.path.join(args.scgpt_dir, "best_model.pt")
        vocab_scgpt = os.path.join(args.scgpt_dir, "vocab.json")
        model.load_scgpt_weights(ckpt_path, freeze_n_layers=args.freeze_layers)

        # 4. Initialise cluster embeddings from scGPT gene-mean vectors
        if os.path.exists(vocab_scgpt):
            vocab.init_embeddings_from_scgpt(
                embedding           = model.feature_embedding,
                loader              = loader,
                scgpt_vocab_path    = vocab_scgpt,
                scgpt_checkpoint_path = ckpt_path,
                rat_to_human_cache_path = args.rat_cache,
                sex                 = args.sex,
            )
    else:
        print("[Info] No scGPT dir supplied — training from scratch.")

    model.param_summary()

    # ------------------------------------------------------------------
    # 5. Dataset + DataLoaders
    # ------------------------------------------------------------------
    context_omics = [o.strip() for o in args.context_omics.split(",")]
    target_omics  = [o.strip() for o in args.target_omics.split(",")] \
                    if getattr(args, "target_omics", None) else None

    bootstrap_dir = getattr(args, "bootstrap_dir", None)

    if bootstrap_dir:
        # ------------------------------------------------------------------
        # Bootstrap mode:
        #   Train on all bootstrap replicates (cluster-level centroids).
        #   Validate on real original data for unbiased metric tracking.
        # ------------------------------------------------------------------
        bs_dirs = sorted(glob.glob(os.path.join(bootstrap_dir, "bootstrap_*")))
        if not bs_dirs:
            raise ValueError(
                f"No bootstrap subdirectories found in {bootstrap_dir}. "
                f"Run generate_bootstraps.py first."
            )
        print(f"[Info] Bootstrap mode: {len(bs_dirs)} replicates from {bootstrap_dir}")

        train_datasets = []
        for bs_dir in bs_dirs:
            bs_loader = TCSeqLoader(data_dir=bs_dir)
            ds = TCSeqDataset(
                loader              = bs_loader,
                vocab               = vocab,
                sex                 = args.sex,
                tissue              = args.tissue,
                context_omics       = context_omics,
                target_omics        = target_omics,
                random_omic_context = True,
                gene_level          = False,   # bootstrap training always cluster-level
                seed                = args.seed,
            )
            train_datasets.append(ds)

        train_ds = ConcatDataset(train_datasets)

        # Validation: real data, fixed context, no gene-level
        val_ds = TCSeqDataset(
            loader              = loader,
            vocab               = vocab,
            sex                 = args.sex,
            tissue              = args.tissue,
            context_omics       = context_omics,
            target_omics        = target_omics,
            fixed_context_omics = context_omics,
            random_omic_context = False,
            gene_level          = False,
            seed                = args.seed,
        )

        print(f"[Info] Train: {len(train_ds)} bootstrap scenarios "
              f"({len(bs_dirs)} replicates × {len(train_datasets[0])} scenarios)")
        print(f"[Info] Val  : {len(val_ds)} real scenarios")

    else:
        # ------------------------------------------------------------------
        # Standard mode: split original data into train / val
        # ------------------------------------------------------------------
        full_ds = TCSeqDataset(
            loader              = loader,
            vocab               = vocab,
            sex                 = args.sex,
            tissue              = args.tissue,
            context_omics       = context_omics,
            target_omics        = target_omics,
            random_omic_context = True,
            gene_level          = args.gene_level,
            seed                = args.seed,
        )

        n_val   = max(1, int(len(full_ds) * args.val_split))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed)
        )
        # Val set uses fixed context (no random omic dropout) for stable metrics
        val_ds.dataset.random_omic_context = False

        print(f"[Info] Train: {n_train} scenarios | Val: {n_val} scenarios")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_tc,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_tc,
                              num_workers=args.num_workers, pin_memory=True)

    # ------------------------------------------------------------------
    # 6. Optimiser + Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    loss_fn       = LOSS_FNS[args.loss]
    best_val_loss = math.inf
    best_ckpt     = os.path.join(args.output_dir, "best_model.pt")

    print(f"[Info] Loss function: {args.loss}")

    for epoch in range(1, args.epochs + 1):

        # Progressive unfreeze schedule
        if epoch == args.unfreeze_epoch:
            model.unfreeze_layers(n=4)
            # Rebuild optimizer to include newly unfrozen params
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.5, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=args.lr * 0.01
            )
            model.param_summary()

        if epoch == args.full_unfreeze_epoch:
            model.unfreeze_layers()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
            model.param_summary()

        train_loss = run_epoch(model, train_loader, optimizer, device,
                               train=True, loss_fn=loss_fn)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_loss = validate(model, val_loader, device, loss_fn=loss_fn)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "vocab_path": vocab_path,
                    "args":       vars(args),
                    "val_loss":   val_loss,
                }, best_ckpt)
                print(f"    ✓ New best model saved (val_loss={val_loss:.4f})")

    # Save final checkpoint regardless
    final_ckpt = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "epoch":      args.epochs,
        "model_state_dict": model.state_dict(),
        "vocab_path": vocab_path,
        "args":       vars(args),
    }, final_ckpt)
    print(f"\n[Done] Best val loss: {best_val_loss:.4f}")
    print(f"       Best checkpoint : {best_ckpt}")
    print(f"       Final checkpoint: {final_ckpt}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Fine-tune TCTransformerModel on TCSeqData"
    )

    # Data
    p.add_argument("--data_dir",      default="data/TCSeqData_n_20")
    p.add_argument("--sex",           default="male",
                   choices=["male","female","both"])
    p.add_argument("--tissue",        default=None,
                   help="Restrict to one tissue (HEART/LIVER/SKMGN). "
                        "None = all.")
    p.add_argument("--context_omics", default="TRNSCRPT,PROT,METAB,ATAC",
                   help="Comma-separated omics to include as context.")
    p.add_argument("--target_omics",  default=None,
                   help="Comma-separated omic(s) to predict "
                        "(e.g. TRNSCRPT or TRNSCRPT,PROT). "
                        "None = all omics.")
    p.add_argument("--gene_level",    action="store_true",
                   help="Train on individual gene predictions within clusters "
                        "instead of cluster centroids.")
    p.add_argument("--bootstrap_dir", default=None,
                   help="Path to pre-computed bootstrap directory (output of "
                        "generate_bootstraps.py). If set, trains on all bootstrap "
                        "replicates and validates on real data. "
                        "Overrides --gene_level for training (always cluster-level).")

    # scGPT weights
    p.add_argument("--scgpt_dir",     default="models/scGPT",
                   help="Path to scGPT checkpoint dir (contains "
                        "best_model.pt and vocab.json). "
                        "Defaults to models/scGPT.")
    p.add_argument("--rat_cache",     default="data/rat_to_human_cache.json")

    # Model
    p.add_argument("--d_model",       type=int,   default=512)
    p.add_argument("--nhead",         type=int,   default=8)
    p.add_argument("--d_hid",         type=int,   default=512)
    p.add_argument("--nlayers",       type=int,   default=12)
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--logfc_scale",   type=float, default=3.0)

    # Training
    p.add_argument("--loss",          default="l2_ratio",
                   choices=["l2_ratio", "mse"],
                   help="Training loss function. l2_ratio normalises by the "
                        "historical-mean baseline; mse is standard MSE.")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--val_split",     type=float, default=0.15)
    p.add_argument("--eval_every",    type=int,   default=2)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--seed",          type=int,   default=42)

    # Freeze schedule
    p.add_argument("--freeze_layers",       type=int, default=8,
                   help="Freeze bottom N transformer layers at start.")
    p.add_argument("--unfreeze_epoch",      type=int, default=10,
                   help="Epoch at which to unfreeze last 4 layers.")
    p.add_argument("--full_unfreeze_epoch", type=int, default=20,
                   help="Epoch at which to unfreeze all layers.")

    # Output
    p.add_argument("--output_dir",    default="output/scgpt_tc")
    p.add_argument("--reload_vocab",  action="store_true",
                   help="Reload existing vocab.json instead of rebuilding.")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
