# Waste Classification — Data Detective Challenge

Submission for Hack[AI]Thon 2.0 Online Selection Round. Clean, train,
and evaluate a waste classifier on a dataset with 71.5% label noise.

## Quick Start

```bash
pip install -r requirements.txt
bash scripts/run_all.sh          # full pipeline
```

## Project Layout

```
├── src/              reusable library (import as `from src import ...`)
│   ├── model.py      WasteClassifier (ResNet-18)
│   ├── dataset.py    datasets, transforms, loading, dedup, relabel
│   ├── trainer.py    training loops, cross-validation
│   ├── analysis.py   data quality inspection, confident learning
│   ├── visualization.py  plotting utilities
│   └── predictor.py  TestDataset + inference
├── scripts/          entry-point scripts
│   ├── inspect_dataset.py        audit for mislabels, blur, dupes
│   ├── train_baseline.py         ResNet-18 on raw folder labels
│   ├── train_improved.py         relabel + dedup + weights + smoothing
│   ├── train_confident_learning.py  3-fold CV + CleanLab correction
│   ├── clean_data.py             compare folder vs filename labels
│   ├── predict.py                generate submission CSV
│   ├── visualize.py              confusion matrix, UMAP, confidence
│   └── run_all.sh                full pipeline
├── configs/
│   └── config.yaml               training configuration
├── reports/
│   ├── index.md                  consolidated findings
│   ├── data_audit.md             per-class metrics
│   └── experiments/              per-experiment results
├── data/                         (not included) train/test images
├── models/                       (gitignored) .pth checkpoints
├── submissions/                  prediction CSVs
├── notebooks/
├── pyproject.toml
└── requirements.txt
```

## Key Findings

| Problem | Impact |
|---------|--------|
| 71.5% label noise | Both folder and filename labels unreliable |
| 49.43% baseline | Best model trains on noisy folder labels directly |
| Confident learning | Corrects only 29/445 labels; matches baseline |
| Self-training | Next step — needs dataset to run |

## Experiments

| Experiment | Val Acc |
|------------|---------|
| Baseline (folder labels) | 49.43% |
| Relabel (filename prefix) | 21.84% |
| All 6 fixes + relabel | 24.14% |
| Confident Learning | 47.19% |
