# Waste Classification — Data Detective Challenge

Submission for Hack[AI]Thon 2.0 Online Selection Round (Round 1).

## Deliverables

- `reports/Round1_Data_Detective.pptx` — 5-slide PPT
- `reports/Round1_Data_Detective.pdf` — 3-page PDF
- `submissions/submission.csv` — test set predictions

## Key Findings

| Issue | Impact |
|-------|--------|
| 71.5% label noise | Model learns folder bias, not visual features |
| 35.3% blurry images | Significant noise in training data |
| 2.15× class imbalance | Trash class at 0% recall |
| 6 duplicate groups + 11 near-duplicates | Train/val leakage |

## Quick Start

```bash
pip install -r requirements.txt
python register.py   # register dataset with 3LC
python train.py      # train ResNet-18 baseline
python predict.py    # generate submission.csv
```

## Project Layout

```
├── train.py           baseline training (ResNet-18)
├── predict.py         inference → submission.csv
├── register.py        3LC dataset registration
├── requirements.txt
├── data/
│   ├── train/         445 images, 6 class folders
│   └── test/          115 test images
├── reports/
│   ├── Round1_Data_Detective.pptx
│   ├── Round1_Data_Detective.pdf
│   ├── figures/       confusion matrix, UMAP, confidence hist
│   └── evidence/      mislabeled & blurry examples
└── submissions/
    └── submission.csv
```
