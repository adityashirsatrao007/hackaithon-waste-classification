# Hack[AI]Thon 2.0 2026 — Waste Classification

**Data-Centric AI Challenge** | Organized by Sphere Hive @ KVGCE, Sullia

A national-level 24-hour AI hackathon focused on **Data-Centric AI** — improving model performance by auditing and fixing the dataset instead of modifying the model architecture.

---

## Challenge Overview

### Round 1: Data Detective (Online — Jun 15 to Jul 25, 2026)

Audit a waste classification image dataset with intentionally introduced data quality bugs.

**Dataset:** 500 labeled + 100 unlabeled images across 5 classes:
- Plastic, Paper, Metal, Glass, Organic Waste

**Introduced bugs:**
- Wrong labels, duplicate images, blurry images, class imbalance, outliers

**Submission:** 5-slide PPT / 3-page PDF + `submission.csv`

### Round 2: The 3LC Retraining Arena (Offline — Aug 8-9, 2026)

Top 16 teams compete on-site at VRIF VTU Belagavi using [3LC.ai](https://3lc.ai) to analyze embeddings, strategically label data, and retrain models.

---

## Repository Structure

```
├── data/
│   ├── raw/              # Original dataset (download manually from Google Drive)
│   └── processed/        # Cleaned/augmented dataset
├── notebooks/            # EDA & analysis notebooks
├── src/                  # Source code (baseline training, evaluation)
├── utils/                # Helper scripts
├── reports/              # Submission PPT/PDF & figures
└── README.md
```

## Getting Started

1. Clone this repo
2. Download the starter kit from the [Google Drive link](https://drive.google.com/file/d/1Wn_UXHZT-YPrLTPdiYjVS_nFewWI1udj/view) and place in `data/raw/`
3. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Dataset

Starter Kit: [Download from Google Drive](https://drive.google.com/file/d/1Wn_UXHZT-YPrLTPdiYjVS_nFewWI1udj/view)

> ⚠️ You need to download this manually — it requires Google sign-in.

## Resources

- [Hack[AI]Thon Website](https://hackaithon.spherehive.in/)
- [Register on Unstop](https://unstop.com/p/hackaithon-20-sphere-hive-kvg-college-of-engineering-sullia-1699309)
- [3LC.ai Platform](https://3lc.ai)
- [Sphere Hive](https://spherehive.in)
