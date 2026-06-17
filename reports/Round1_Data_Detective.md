# Waste Classification — Data Detective Report

**Hack[AI]Thon 2.0 · Round 1 · Online Screening**  
**Team:** [Your Team Name] | **Date:** July 2026

---

## Dataset Overview

| Attribute | Value |
|-----------|-------|
| Training images | 445 |
| Test images | 115 |
| Classes | cardboard, glass, metal, paper, plastic, trash |
| Baseline model | ResNet-18 (fixed, not modified) |
| Baseline accuracy | 42.53% (random = 16.7%) |

**Class Distribution:**

| Class | Count | Proportion |
|-------|-------|------------|
| cardboard | 82 | 18.4% |
| glass | 78 | 17.5% |
| metal | 84 | 18.9% |
| paper | 86 | 19.3% |
| plastic | 75 | 16.9% |
| trash | 40 | 9.0% |

Imbalance ratio: **2.15×** (trash most underrepresented)

---

## Problems Found

### 1. Wrong Labels — 71.5% (318/445)
The majority of images are stored in incorrect class folders. Folder labels and filename-prefix labels disagree on most samples. The model learns folder bias (predicts "cardboard" for everything) rather than learning visual features.

### 2. Duplicate Images
- **6 exact duplicate groups** — same image appears in multiple folders with different labels
- **11 near-duplicate pairs** — visually similar images with minor variations
- These inflate dataset size and cause train/validation leakage

### 3. Blurry / Low-Quality Images
- **157 images (35.3%)** have Laplacian variance < 100
- Minimum variance: **0.6** (effectively blank/noise)
- These images contribute noise instead of signal

### 4. Class Imbalance
- **trash** class has only **40 samples** vs 82–86 for other classes
- Model achieves **0% precision, 0% recall, 0% F1** on trash — effectively ignores this class entirely

### 5. Outliers
- **22 file-size outliers** detected (range: 4.7 KB – 43 KB)
- Potential corrupted or anomalous samples

---

## Evidence Summary

| Metric | Value |
|--------|-------|
| Overall validation accuracy | 42.53% |
| Precision range | 0.00 (trash) – 0.80 (cardboard) |
| Recall range | 0.00 (trash) – 0.72 (cardboard) |
| Mislabeled images | 318 (71.5%) |
| Blurry images | 157 (35.3%) |
| Duplicate groups | 6 exact + 11 near-duplicate |

**Key insight:** The confusion matrix shows the model overwhelmingly predicts "cardboard" regardless of input. This is evidence that the model has learned the **folder bias** (most images are in the cardboard folder) rather than any visual features of waste materials.

---

## Suggested Fixes

| Fix | Expected Impact | Effort |
|-----|----------------|--------|
| 1. Correct labels (filename prefix + human review) | +10–15% accuracy | High |
| 2. Remove duplicates (exact + near-duplicate) | +3–5% accuracy | Low |
| 3. Handle blurry images (down-weight or remove) | +2–3% accuracy | Low |
| 4. Balance classes (oversample/augment trash) | +5–8% on trash recall | Medium |
| 5. Stratified train/validation splits | Prevents leakage | Low |
| 6. Self-training / pseudo-labeling (iterative) | +5–10% additional | Medium |

**Projected combined improvement:** 42% → **60–70%** validation accuracy

---

## Expected Impact

### Primary Impact
- **Label correction** alone could improve accuracy by **10–15 percentage points** by removing the dominant source of noise
- **Class balancing** could bring **trash recall from 0% to ~30–40%** , making the model usable for all 6 classes

### Secondary Impact
- **Deduplication** ensures honest evaluation metrics (no data leakage)
- **Blurry image handling** cleans up the bottom 35% of the dataset

### Recommendation
The 71.5% label noise is too severe for automated correction (Confident Learning fixed only 24/445 labels). **Human-in-the-loop verification** of the most ambiguous samples is recommended. The remaining labels can be corrected via **self-training**: predict with current model, filter high-confidence predictions (p > 0.6), add to corrected set, and retrain iteratively.

---

*Supporting evidence: confusion matrix, UMAP embeddings, per-class metrics, and 24 example images available in `reports/figures/` and `reports/evidence/`.*
