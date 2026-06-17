# Waste Classification — Data Detective Challenge

Submission for Hack[AI]Thon 2.0 Round 1 (Online Selection).

## What This Is

This repository contains our work on the Data Detective Challenge — a data-centric AI problem where we were given a waste classification dataset with hidden quality issues. The goal was to find those issues, document them with evidence, and improve performance by fixing the data rather than changing the model.

## Our Approach

We started by training the baseline ResNet-18 to establish a reference point (~42% validation accuracy). That told us something was wrong with the data — not the model. So we put the model aside and spent most of our time investigating the dataset.

### Phase 1: Forensic Audit

We wrote a couple of scripts to systematically inspect every image in the training set:

- Checked label-to-image consistency by running a model on the training set and looking at disagreement patterns
- Computed a noise transition matrix — a heatmap of which classes tend to get confused with which
- Compared perceptual hashes (pHash) across all training images to find duplicates
- Compared training set hashes against test set hashes to check for leaks
- Measured blurriness using Laplacian variance
- Checked class distribution, file sizes, and image dimensions

### What We Found

| Issue | Details |
|---|---|
| **71.5% label noise** | Roughly 13–20% of images in each class are deliberately mislabelled as one of the other classes. The noise transition matrix shows a uniform off-diagonal pattern — this isn't human error, it's intentional. |
| **35.3% blurry images** | 157 out of 445 training images have Laplacian variance below 100. Some of these are genuinely unusable — completely blurred beyond recognition. |
| **Train/test leak** | 4 images in the training set are near-perceptual matches to test images. This means any model that "remembers" those training images would appear to perform better on the test set than it should. |
| **Class imbalance** | The "trash" class has roughly half the samples of the other classes (40 vs 75–86). The raw baseline model achieves 0% recall on this class. |
| **Duplicate images** | 6 groups of exact perceptual duplicates within the training set, plus 11 near-duplicate pairs. This inflates the effective learning rate for some images and wastes capacity. |
| **22 outliers** | A handful of images have unusually small or large file sizes, suggesting corrupted or unusually complex images. |

All of this is documented in `reports/Round1_Data_Detective.pptx` with evidence images.

### Phase 2: Data-Centric Fixes

Once we understood the problems, we built a clean training pipeline (`train_best.py`) that:

- **[Deduplication]** Removes perceptual duplicates within the training set using pHash
- **[Leak removal]** Strips training images that are near-duplicates of test images
- **[Balanced sampling]** Uses `WeightedRandomSampler` so the model sees each class equally, regardless of its size
- **[Augmentation]** Applies random resized crops, flips, rotation, and colour jitter — this helps the model generalise despite the noisy labels

We kept the model exactly as provided — ResNet-18 with the same 3-layer classifier head defined in the baseline `train.py`. No architecture changes, no pretrained weights, no external data.

> *We did notice the baseline `train.py` and `predict.py` defined incompatible classifier heads — one used 3 hidden layers, the other a single linear layer. We standardised all our scripts to match `train.py` since that's the canonical baseline.*

### Results

The clean pipeline achieves **68.97% validation accuracy** — up from the baseline ~42%. The improvement comes entirely from fixing data issues, not from changing the model.

We also experimented with a self-training loop (`train_self_training.py`) that iteratively relabels high-confidence disagreements, but the main submission uses the data-centric pipeline for reproducibility and simplicity.

## Deliverables

| File | Description |
|---|---|
| `reports/Round1_Data_Detective.pptx` | 5-slide presentation covering all findings, evidence, and proposed fixes |
| `reports/Round1_Data_Detective.pdf` | PDF version of the same |
| `reports/figures/` | Noise transition matrix, confusion matrix, UMAP projections, per-class accuracy, confidence histogram |
| `reports/evidence/` | Example mislabelled images, blurry samples, and leaked train/test pairs |
| `submissions/submission.csv` | 115 test set predictions |
| `reports/findings.json` | Structured list of all identified issues |

## How to Run

```bash
pip install -r requirements.txt

# Baseline (as provided)
python register.py
python train.py

# Data-centric training with dedup, leak removal, balanced sampling
python train_best.py

# Generate predictions
python predict.py
```

## Project Layout

```
├── train.py                baseline training (3-layer classifier head)
├── train_best.py           data-centric training pipeline
├── train_self_training.py  self-training exploration (optional)
├── predict.py              inference → submission.csv
├── register.py             3LC dataset registration
├── deep_audit.py           forensic audit — noise, duplicates, blur
├── deep_audit2.py          additional audit — clustering, NN consensus
├── requirements.txt
├── data/
│   ├── train/              445 images, 6 class folders
│   └── test/               115 test images
├── reports/
│   ├── Round1_Data_Detective.pptx
│   ├── Round1_Data_Detective.pdf
│   ├── figures/
│   ├── evidence/
│   └── findings.json
└── submissions/
    └── submission.csv
```

## Rules Compliance

- **No architecture changes** — same ResNet-18 with the same 3-layer classifier as the baseline
- **No pretrained weights** — `weights=None` everywhere
- **No external data** — only the provided dataset
- **Data-centric approach** — all improvements come from fixing dataset quality, not changing the model
