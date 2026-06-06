"""
ScgptTCPipeline — Evaluation / inference runner.

Sub-commands
------------
  evaluate   Load a checkpoint and run all prediction scenarios, print metrics.
  zero_shot  Build vocab + model from scGPT weights only (no finetuning), evaluate.
  list       Print a data summary (tissues / omics / cluster counts).

Usage examples
--------------
  # Evaluate a finetuned checkpoint
  python -m ScgptTCPipeline.run_pipeline evaluate \\
      --checkpoint output/scgpt_tc/best_model.pt \\
      --data_dir   data/TCSeqData_n_20 \\
      --sex male

  # Single predict-week, two modalities as context
  python -m ScgptTCPipeline.run_pipeline evaluate \\
      --checkpoint output/scgpt_tc/best_model.pt \\
      --predict_week 4w \\
      --context_omics TRNSCRPT,PROT

  # Zero-shot (uses scGPT backbone only, no finetuning)
  python -m ScgptTCPipeline.run_pipeline zero_shot \\
      --scgpt_dir models/scgpt-human-whole-body \\
      --data_dir  data/TCSeqData_n_20

  # List available data
  python -m ScgptTCPipeline.run_pipeline list --data_dir data/TCSeqData_n_20
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ScgptTCPipeline.data.loader         import TCSeqLoader
from ScgptTCPipeline.model.feature_vocab  import FeatureVocab
from ScgptTCPipeline.model.tc_transformer import TCTransformerModel
from ScgptTCPipeline.dataset.tc_dataset   import TCSeqDataset, collate_tc
from ScgptTCPipeline.evaluation.metrics   import TCMetrics


# ------------------------------------------------------------------
# Model loading helpers
# ------------------------------------------------------------------

def _load_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load model + vocab from a finetune.py checkpoint."""
    ckpt  = torch.load(ckpt_path, map_location=device)
    saved = ckpt.get("args", {})

    vocab = FeatureVocab.load(ckpt["vocab_path"])

    model = TCTransformerModel(
        n_features  = len(vocab),
        d_model     = saved.get("d_model",     512),
        nhead       = saved.get("nhead",        8),
        d_hid       = saved.get("d_hid",       512),
        nlayers     = saved.get("nlayers",      12),
        dropout     = 0.0,                          # no dropout at eval
        logfc_scale = saved.get("logfc_scale",  3.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[run_pipeline] Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f})")
    return model, vocab, saved


def _build_zero_shot_model(
    scgpt_dir:   str,
    loader:      TCSeqLoader,
    sex:         str,
    tissue:      Optional[str],
    rat_cache:   str,
    device:      torch.device,
    gene_level:  bool = False,
):
    """Build model + vocab from scGPT weights with no finetuning."""
    vocab = FeatureVocab.build(loader, sex=sex, tissue=tissue,
                               include_genes=gene_level)

    model = TCTransformerModel(
        n_features=len(vocab), d_model=512, nhead=8,
        d_hid=512, nlayers=12, dropout=0.0,
    )
    ckpt_path  = os.path.join(scgpt_dir, "best_model.pt")
    vocab_path = os.path.join(scgpt_dir, "vocab.json")
    model.load_scgpt_weights(ckpt_path, freeze_n_layers=0)

    if os.path.exists(vocab_path):
        vocab.init_embeddings_from_scgpt(
            embedding             = model.feature_embedding,
            loader                = loader,
            scgpt_vocab_path      = vocab_path,
            scgpt_checkpoint_path = ckpt_path,
            rat_to_human_cache_path = rat_cache,
            sex                   = sex,
        )

    model.to(device).eval()
    return model, vocab


# ------------------------------------------------------------------
# Shared evaluation loop
# ------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model:         TCTransformerModel,
    loader:        TCSeqLoader,
    vocab:         FeatureVocab,
    args,
    device:        torch.device,
) -> TCMetrics:

    context_omics = [o.strip().upper() for o in args.context_omics.split(",")]
    raw_target    = getattr(args, "target_omic", None)
    target_omics  = [o.strip().upper() for o in raw_target.split(",")] \
                    if raw_target else None

    gene_level = getattr(args, "gene_level", False)

    dataset = TCSeqDataset(
        loader               = loader,
        vocab                = vocab,
        sex                  = args.sex,
        tissue               = getattr(args, "tissue", None),
        context_omics        = context_omics,
        target_omics         = target_omics,
        fixed_predict_week   = getattr(args, "predict_week", None),
        fixed_context_omics  = context_omics,   # fixed at eval — no random dropout
        random_omic_context  = False,
        gene_level           = gene_level,
        seed                 = 42,
    )

    # Filter to specific cluster(s) if requested
    cluster_filter = getattr(args, "cluster", None)
    if cluster_filter is not None:
        clusters = {int(c) for c in str(cluster_filter).split(",")}
        dataset._index = [s for s in dataset._index if s.cluster_num in clusters]
        print(f"[run_pipeline] Cluster filter {clusters}: "
              f"{len(dataset._index)} scenarios remaining.")

    # Randomly sample N genes per (cluster, predict_week) if requested
    n_genes = getattr(args, "n_genes", None)
    if n_genes is not None and gene_level:
        import random as _random
        from collections import defaultdict
        _random.seed(42)
        # Group index by (cluster_num, predict_week) key
        groups = defaultdict(list)
        for s in dataset._index:
            groups[(s.cluster_num, s.predict_week)].append(s)
        sampled = []
        for entries in groups.values():
            sampled.extend(
                entries if len(entries) <= n_genes
                else _random.sample(entries, n_genes)
            )
        dataset._index = sampled
        print(f"[run_pipeline] Sampled {n_genes} genes/cluster: "
              f"{len(dataset._index)} scenarios remaining.")

    if len(dataset) == 0:
        print("[run_pipeline] No prediction scenarios found. "
              "Check --data_dir, --sex, --tissue, --cluster.")
        return TCMetrics()

    eval_loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = collate_tc,
        num_workers = getattr(args, "num_workers", 2),
        pin_memory  = True,
    )

    metrics = TCMetrics()
    total   = len(dataset)
    done    = 0

    for batch in eval_loader:
        feature_ids  = batch["feature_ids"].to(device)
        time_ids     = batch["time_ids"].to(device)
        modality_ids = batch["modality_ids"].to(device)
        values       = batch["values"].to(device)
        is_masked    = batch["is_masked"].to(device)
        targets      = batch["targets"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        out = model(feature_ids, time_ids, modality_ids,
                    values, is_masked, padding_mask)

        preds_all      = out["predictions"].cpu()   # [B, S]
        targets_all    = targets.cpu()              # [B, S]
        mask_all       = is_masked.cpu()            # [B, S]
        hist_means_all = batch["hist_means"].cpu()  # [B, S]

        # Record per-item (one meta entry per sequence in the batch)
        for i, meta in enumerate(batch["meta"]):
            qmask      = mask_all[i]                  # [S] bool
            preds      = preds_all[i][qmask].numpy()
            actuals    = targets_all[i][qmask].numpy()
            hist_means = hist_means_all[i][qmask].numpy()
            metrics.add(preds, actuals, meta, hist_means=hist_means)

        done += len(batch["meta"])
        if done % 200 == 0 or done == total:
            print(f"  [{done}/{total}] scenarios evaluated...")

    return metrics


# ------------------------------------------------------------------
# Sub-command handlers
# ------------------------------------------------------------------

def cmd_evaluate(args):
    print("=" * 70)
    print("ScgptTCPipeline — Evaluate (finetuned checkpoint)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    loader = TCSeqLoader(data_dir=args.data_dir)
    model, vocab, saved_args = _load_from_checkpoint(args.checkpoint, device)

    # Allow CLI overrides of saved args
    if not hasattr(args, "context_omics") or args.context_omics is None:
        args.context_omics = saved_args.get("context_omics",
                                            "TRNSCRPT,PROT,METAB,ATAC")
    if not hasattr(args, "tissue") or args.tissue is None:
        args.tissue = saved_args.get("tissue", None)

    groupby = [g.strip() for g in args.groupby.split(",")] \
              if getattr(args, "groupby", None) else ["omic", "predict_week"]
    metrics = run_evaluation(model, loader, vocab, args, device)
    metrics.report(groupby=groupby)

    if args.output_file:
        metrics.save(args.output_file)


def cmd_zero_shot(args):
    print("=" * 70)
    print("ScgptTCPipeline — Zero-shot (scGPT backbone only)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    loader = TCSeqLoader(data_dir=args.data_dir)
    model, vocab = _build_zero_shot_model(
        args.scgpt_dir, loader, args.sex,
        getattr(args, "tissue", None), args.rat_cache, device,
        gene_level=getattr(args, "gene_level", False),
    )

    groupby = [g.strip() for g in args.groupby.split(",")] \
              if getattr(args, "groupby", None) else ["omic", "predict_week"]
    metrics = run_evaluation(model, loader, vocab, args, device)
    metrics.report(groupby=groupby)

    if args.output_file:
        metrics.save(args.output_file)


def cmd_list(args):
    loader = TCSeqLoader(data_dir=args.data_dir)
    loader.summary(sex=args.sex)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _add_shared_eval_args(p):
    p.add_argument("--data_dir",       default="data/TCSeqData_n_20")
    p.add_argument("--sex",            default="male",
                   choices=["male","female","both"])
    p.add_argument("--tissue",         default=None,
                   help="Restrict evaluation to one tissue.")
    p.add_argument("--predict_week",   default=None,
                   choices=["2w","4w","8w"],
                   help="Evaluate one specific predict-week only. "
                        "None = all three.")
    p.add_argument("--target_omic",    default=None,
                   help="Comma-separated omic(s) to predict "
                        "(e.g. TRNSCRPT or TRNSCRPT,PROT). "
                        "None = all omics.")
    p.add_argument("--cluster",        default=None,
                   help="Comma-separated cluster number(s) to evaluate "
                        "(e.g. 1 or 1,3,5). None = all clusters.")
    p.add_argument("--gene_level",     action="store_true",
                   help="Predict individual genes within clusters instead of "
                        "cluster centroids. Automatically includes gene tokens "
                        "in the feature vocab.")
    p.add_argument("--n_genes",        type=int, default=None,
                   help="Randomly sample N genes per cluster per predict_week "
                        "(only applies with --gene_level). "
                        "None = use all genes.")
    p.add_argument("--context_omics",  default="TRNSCRPT,PROT,METAB,ATAC")
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--num_workers",    type=int, default=2)
    p.add_argument("--output_file",    default=None,
                   help="Path to save results CSV.")
    p.add_argument("--groupby",        default="cluster_num,predict_week",
                   help="Comma-separated columns for the breakdown report. "
                        "Available: tissue, omic, sex, cluster_num, predict_week. "
                        "Default: cluster_num,predict_week.")


def main():
    parser = argparse.ArgumentParser(
        description="ScgptTCPipeline — Evaluation Runner"
    )
    sub = parser.add_subparsers(dest="command")

    # evaluate
    p_eval = sub.add_parser("evaluate",
                             help="Evaluate a finetuned checkpoint.")
    p_eval.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt from finetune.py.")
    _add_shared_eval_args(p_eval)

    # zero_shot
    p_zs = sub.add_parser("zero_shot",
                           help="Evaluate with scGPT backbone only (no finetuning).")
    p_zs.add_argument("--scgpt_dir",  default="models/scGPT",
                      help="Path to scGPT checkpoint directory. "
                           "Defaults to models/scGPT.")
    p_zs.add_argument("--rat_cache",  default="data/rat_to_human_cache.json")
    _add_shared_eval_args(p_zs)

    # list
    p_list = sub.add_parser("list", help="Print data summary.")
    p_list.add_argument("--data_dir", default="data/TCSeqData_n_20")
    p_list.add_argument("--sex",      default="male")

    args = parser.parse_args()
    if   args.command == "evaluate":  cmd_evaluate(args)
    elif args.command == "zero_shot": cmd_zero_shot(args)
    elif args.command == "list":      cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
