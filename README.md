# 🗑️ Waste Classification — Data Detective Challenge

Welcome to the **Hack[AI]thon 2.0 Online Selection Round**!

To get to know more about 3LC, check out this video: https://youtu.be/zdIq1QpeSI8?si=b5KlwIjsefdQ6WuE

## 🎯 Problem Statement

Participants are provided with a **waste-classification image dataset** containing intentionally introduced data-quality issues and label noise.

Your mission is to act as a **Data Detective**: investigate the dataset, identify hidden issues that negatively impact model performance, provide evidence for your findings, and propose actionable improvements.

Unlike traditional machine learning competitions, **the model architecture is fixed**. Success depends entirely on your ability to analyze and improve the quality of the data.

> **Important:** Do not modify model architectures or training pipelines. Focus exclusively on data-centric AI techniques and dataset quality improvements.

---

## 🎯 Objectives

Your task is to perform a complete audit of the dataset and identify:

### 1. Incorrect Labels

* Detect mislabeled waste images

* Provide evidence supporting label corrections

### 2. Duplicate Images

* Identify exact duplicates

* Identify near-duplicate samples

* Recommend removal or consolidation strategies

### 3. Low-Quality Images

* Locate blurry, corrupted, or low-information samples

* Assess their impact on training performance

### 4. Class Imbalance

* Analyze class distributions

* Identify underrepresented categories

* Recommend balancing strategies

### 5. Outliers & Anomalies

* Detect unusual samples that do not fit their assigned class

* Identify hidden distribution shifts or sampling issues

### 6. Additional Data Quality Issues

* Discover any other problems affecting dataset reliability

* Provide clear documentation and supporting evidence

---

## 🚀 Getting Started

### 1. Install Dependencies

Ensure you have Python 3.10+ installed.

```bash

python -m venv venv

# macOS/Linux

source venv/bin/activate

# Windows

venv\Scripts\activate

pip install -r requirements.txt

```

---

### 2. Setup 3LC Authentication

Go to https://account.3lc.ai/home to get your API key.

After that, run the following command and paste the API key when prompted:

```bash

3lc login

```

Then run:

```bash

3lc service

```

---

### 3. Register the Dataset

Register the dataset with 3LC Tables:

```bash

python register.py

```

---

### 4. Generate Baseline Metrics

Run the baseline training pipeline:

```bash

python train.py

```

This will:

* Train the fixed ResNet-18 model

* Generate validation metrics

* Log predictions and sample-level information to 3LC

---

### 5. Investigate Using the 3LC Dashboard

Open the 3LC Dashboard and navigate to:

```text

Waste-Classification → fixed_image_run

```

Use the dashboard to:

* Inspect model predictions

* Analyze high-loss samples

* Investigate misclassified images

* Explore embeddings and clusters

* Detect anomalies and potential labeling errors

---

## 🏆 Evaluation Criteria

Participants will be evaluated on:

| Category           | Description                                       |
| ------------------ | ------------------------------------------------- |
| Bug Discovery      | Number and quality of issues identified           |
| Evidence           | Strength of supporting analysis                   |
| Data Understanding | Quality of dataset investigation                  |
| Proposed Fixes     | Practicality and effectiveness of recommendations |
| Documentation      | Clarity of findings and report                    |

---

## 💡 What We Are Looking For

Think like a data scientist and dataset auditor.

The goal is not to build a better model.

The goal is to understand:

* Why the model fails

* Which data issues are responsible

* How those issues can be fixed

Good luck, Detective! 🔍♻️

---

## Results

- **Audit script**: `inspect_dataset.py` — checks labels, duplicates, blur, imbalance, and outliers
- **Findings**: `reports/findings.md` with evidence images in `reports/evidence/`
- **Baseline accuracy**: 46.07% (ResNet-18, 1 epoch, on the buggy dataset)
- **Predictions**: `submission.csv` (115 test images)

### Issues Found

| Problem | Count | Severity |
|---------|-------|----------|
| Mislabeled images | 318 / 445 (71.5%) | Critical |
| Class imbalance | 2.15x (trash=40, paper=86) | Critical |
| Blurry images | 157 | Moderate |
| Exact duplicates | 6 groups | Moderate |
| Near-duplicates | 11 pairs | Low |
| File size outliers | 22 | Low |

### How to Run

```bash
# dataset inspection
python inspect_dataset.py

# train baseline
python train.py

# generate submission
python predict.py
```
