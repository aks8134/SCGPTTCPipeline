# ScgptTCPipeline

A multimodal timeseries prediction pipeline for exercise-response omics data, built on top of the [scGPT](https://github.com/bowang-lab/scGPT) pretrained transformer backbone.

---

## Overview

ScgptTCPipeline predicts the temporal response of biological features (genes, proteins, metabolites, chromatin accessibility) to exercise training. Given observations at earlier timepoints across one or more omics modalities, the model predicts the response at a future timepoint.

**Key design choices:**
- Vector-based input (not language-based) — each token encodes a biological feature with its measurement value, timepoint, and modality
- Adaptive masking — any subset of (timepoint × modality) combinations can serve as context or prediction target
- Signed logFC handling — uses `tanh(logFC / scale)` instead of scGPT's positive-only binning
- Pretrained weights from scGPT initialise the transformer backbone, with cluster embeddings initialised from mean gene embeddings

---

## Data

**TCSeqData_n_20**: Multimodal exercise-response timeseries from the MoTrPAC study.

| Property | Details |
|----------|---------|
| Tissues | HEART, LIVER, SKMGN |
| Omics | TRNSCRPT, PROT, METAB, ATAC |
| Timepoints | 1w, 2w, 4w, 8w post-exercise |
| Clusters | 20 per (tissue, omic, sex) |
| Values | Z-scored log-fold-change vs sedentary control |

Data directory structure:
```
data/TCSeqData_n_20/
    SKMGN_TRNSCRPT_male/
        cluster1.csv   # columns: gene, cluster, 1w, 2w, 4w, 8w
        cluster2.csv
        ...
    SKMGN_PROT_male/
    ...
```

---

## Model Architecture

### Token Construction

Each sequence position encodes one (cluster, timepoint, modality) observation:

```
token = FeatureEmbedding(cluster_id)
      + TimeEmbedding(timepoint)
      + ModalityEmbedding(omic)
      + ScalarValueEncoder(logFC)     ← context tokens
      OR MaskToken                    ← query tokens (prediction targets)
```

**ScalarValueEncoder**: `tanh(logFC / logfc_scale) → Linear → GELU → Linear → LayerNorm`
Handles signed logFC values unlike scGPT's positive-only expression binning.

**MaskToken**: Learned parameter injected at query positions, replacing the value encoder.

### Sequence Layout

```
[CLS] [ctx_cluster_1_1w] [ctx_cluster_1_2w] ... [ctx_cluster_N_Kw] [QUERY_cluster_i_Tw]
```

- **CLS token**: prepended for global representation
- **Context tokens**: all cluster centroids at earlier timepoints (real values, unmasked)
- **Query token**: target cluster at predict timepoint (masked — value to be predicted)

### Transformer

Identical architecture to scGPT:

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Attention heads | 8 |
| Hidden dim | 512 |
| FFN dim | 512 |
| Dropout | 0.2 |
| Max sequence length | 512 |

### Prediction Head (TemporalDecoder)

```
Linear(512→512) → GELU → Linear(512→256) → GELU → Linear(256→1) → scalar logFC
```

Applied only at masked (query) positions.

---

## Pretrained Weights

The transformer backbone is initialised from scGPT pretrained weights (`models/scGPT/best_model.pt`), pretrained on 33M human single cells.

**Weight mapping**: scGPT's `transformer_encoder.layers` → our `transformer.layers`

**Cluster embedding initialisation**:
- TRNSCRPT / PROT clusters: embedding = mean of member gene embeddings from scGPT (rat→human ortholog mapping applied)
- METAB / ATAC clusters: embedding = scGPT global mean + small Gaussian noise

---

## Loss Function

Training uses a differentiable **l2_ratio loss**:

```
loss = ||predicted - actual|| / ||actual - hist_mean||
```

where `hist_mean` is the mean of the target feature's values at context timepoints (historical baseline). A loss < 1 means the model beats predicting the historical mean.

This is more informative than MSE because it normalises by a meaningful biological baseline — if the model achieves l2_ratio = 0.25, its errors are 4× smaller than simply extrapolating from past observations.

Switch to MSE with `--loss mse` if needed.

---

## Bootstrapping

Because the dataset is small (~20 clusters per tissue/omic/sex), we augment training data by bootstrapping cluster centroids.

**Process**: For each cluster, resample its member genes with replacement N times and recompute the mean logFC → N slightly different centroid trajectories per cluster.

**Two-stage training/evaluation**:
1. **Train** on bootstrap centroid replicates (cluster-level, augmented)
2. **Evaluate** on real individual gene/metabolite values (gene-level, real data)

This tests whether cluster-level representations learned from centroid statistics generalise to individual feature predictions.

### Generate Bootstrap Data

```bash
python -m ScgptTCPipeline.data.generate_bootstraps \
    --data_dir     data/TCSeqData_n_20 \
    --output_dir   data/TCSeqData_n_20_bs100_SKMGN_male \
    --n_bootstraps 100 \
    --tissue       SKMGN \
    --sex          male \
    --seed         42
```

Output: one subdirectory per replicate (`bootstrap_0001/`, ..., `bootstrap_0100/`), each mirroring the original data format with one centroid row per cluster.

---

## Finetuning

```bash
python -m ScgptTCPipeline.finetune \
    --data_dir      data/TCSeqData_n_20 \
    --bootstrap_dir data/TCSeqData_n_20_bs100_SKMGN_male \
    --scgpt_dir     models/scGPT \
    --output_dir    output/scgpt_tc_SKMGN_male \
    --context_omics TRNSCRPT \
    --target_omics  TRNSCRPT \
    --tissue        SKMGN \
    --sex           male \
    --epochs        30 \
    --batch_size    32 \
    --loss          l2_ratio
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/TCSeqData_n_20` | Original data directory |
| `--bootstrap_dir` | None | Pre-computed bootstrap directory. If set, trains on all replicates and validates on real data |
| `--scgpt_dir` | `models/scGPT` | Path to scGPT checkpoint directory |
| `--context_omics` | `TRNSCRPT,PROT,METAB,ATAC` | Omics to use as context |
| `--target_omics` | None (all) | Omics to predict |
| `--tissue` | None (all) | Restrict to one tissue |
| `--sex` | `male` | Sex to use |
| `--loss` | `l2_ratio` | Loss function (`l2_ratio` or `mse`) |
| `--gene_level` | False | Train on individual gene predictions |
| `--freeze_layers` | 8 | Freeze bottom N transformer layers at start |
| `--unfreeze_epoch` | 10 | Epoch to unfreeze last 4 layers |
| `--full_unfreeze_epoch` | 20 | Epoch to unfreeze all layers |

### Progressive Unfreeze Schedule

| Phase | Epochs | Trainable |
|-------|--------|-----------|
| Frozen | 1–9 | New heads + embeddings only (~35%) |
| Partial | 10–19 | Last 4 transformer layers unfrozen |
| Full | 20–30 | All parameters |

---

## Evaluation

```bash
python -m ScgptTCPipeline.run_pipeline evaluate \
    --checkpoint    output/scgpt_tc_SKMGN_male/best_model.pt \
    --data_dir      data/TCSeqData_n_20 \
    --context_omics TRNSCRPT \
    --target_omic   TRNSCRPT \
    --tissue        SKMGN \
    --sex           male \
    --predict_week  4w \
    --gene_level \
    --groupby       cluster_num,predict_week \
    --output_file   output/scgpt_tc_SKMGN_male/eval_predictions.csv
```

### Gene-Level Evaluation

With `--gene_level`, each individual gene/metabolite is evaluated separately:
- **Query token**: uses the cluster feature embedding (not a gene-specific token)
- **Context**: the gene's own values at earlier timepoints replace the cluster centroid for that cluster position — giving each gene a unique input sequence and therefore a unique prediction
- **Target**: the individual gene's actual logFC at the predict week

This correctly evaluates generalisation from cluster-level training to individual feature prediction.

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--predict_week` | Evaluate one week only (`2w`/`4w`/`8w`). None = all three |
| `--target_omic` | Omic to predict (e.g. `TRNSCRPT` or `METAB`) |
| `--gene_level` | Evaluate at individual gene/metabolite level |
| `--groupby` | Report breakdown by columns (e.g. `cluster_num,predict_week`) |
| `--output_file` | Save predictions CSV |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean absolute error |
| **RMSE** | Root mean squared error |
| **rRMSE** | RMSE / mean(|actual|) — relative RMSE |
| **R²** | Coefficient of determination. R²=1: perfect; R²=0: no better than global mean |
| **l2_ratio** | `‖errors‖ / ‖actual − hist_mean‖`. < 1: beats historical-mean baseline |

**l2_ratio interpretation**: Values close to 0 are best. A value of 0.25 means the model's errors are 4× smaller than predicting "the next timepoint equals the historical mean of previous timepoints."

---

## Zero-Shot Evaluation

Evaluate without any finetuning using only the scGPT backbone:

```bash
python -m ScgptTCPipeline.run_pipeline zero_shot \
    --scgpt_dir     models/scGPT \
    --data_dir      data/TCSeqData_n_20 \
    --context_omics TRNSCRPT \
    --target_omic   TRNSCRPT \
    --tissue        SKMGN \
    --sex           male \
    --predict_week  4w
```

---

## Example Results (SKMGN, Male)

### TRNSCRPT → TRNSCRPT (gene-level, 4w)

| Metric | Value |
|--------|-------|
| MAE | 0.2331 |
| RMSE | 0.2825 |
| R² | 0.8698 |
| l2_ratio | 0.2540 |

### TRNSCRPT,PROT,METAB,ATAC → METAB (metabolite-level)

| Week | MAE | RMSE | R² | l2_ratio |
|------|-----|------|----|---------|
| 2w | 0.2188 | 0.2721 | 0.8653 | 0.2424 |
| 4w | 0.1964 | 0.2489 | 0.9045 | 0.2153 |
| 8w | 0.1709 | 0.2257 | 0.9540 | 0.1608 |

---

## Project Structure

```
ScgptTCPipeline/
├── data/
│   ├── loader.py               # TCSeqLoader — data loading and centroid computation
│   └── generate_bootstraps.py  # Pre-compute bootstrap centroid replicates
├── model/
│   ├── token_encoder.py        # ScalarValueEncoder, MaskToken, TCTokenEncoder
│   ├── tc_transformer.py       # TCTransformerModel, TemporalDecoder
│   └── feature_vocab.py        # FeatureVocab — cluster/gene token registry
├── dataset/
│   └── tc_dataset.py           # TCSeqDataset, collate_tc
├── evaluation/
│   └── metrics.py              # TCMetrics, point_metrics, l2_ratio, r2_score
├── finetune.py                 # Finetuning script
└── run_pipeline.py             # Evaluation / inference runner
```
