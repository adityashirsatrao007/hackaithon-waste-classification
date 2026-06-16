# Waste Classification - Data Quality Findings

## Overview
Total training images: 445
Classes: cardboard, glass, metal, paper, plastic, trash

## 1. Class Distribution
| Class | Count |
|-------|-------|
| cardboard | 82 |
| glass | 78 |
| metal | 84 |
| paper | 86 |
| plastic | 75 |
| trash | 40 |

Imbalance ratio: 2.15x
**Impact**: The trash class has significantly fewer samples. The model will struggle to learn this category.

## 2. Label Errors
Mislabeled images: 318 / 445 (71.5%)
Correctly labeled: 127

The filename pattern `{true_label}_random_id.jpg` reveals the actual class.
Evidence images saved to `reports/evidence/` with filenames showing folder vs actual label.

## 3. Duplicate Images
Exact duplicate groups found: 6
Near-duplicate pairs found: 11

## 4. Image Quality
Blurry images (Laplacian variance < 100): 157
- Min variance: 0.6
- Mean variance: 48.5

Corrupted images: 0
Low resolution images (< 100px): 0

## 5. Outliers
File size outliers: 22
Size range: 4770 bytes - 43213 bytes

## 6. Recommended Fixes
1. Relabel images using the prefix in each filename
2. Remove exact duplicate images (keep one copy per group)
3. Remove or flag blurry images for the organizers
4. Augment the trash class to balance the dataset
5. Use stratified sampling during training to handle imbalance
