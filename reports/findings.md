# Data Quality Findings & Experiments

## Dataset Overview
- **445 training images**, 6 classes
- Folder labels: 71.5% mislabeled (318/445)
- Filename prefix labels: also unreliable (21.84% model accuracy)

## Class Distribution
| Class | Count |
|-------|-------|
| paper | 86 |
| metal | 84 |
| cardboard | 82 |
| glass | 78 |
| plastic | 75 |
| trash | 40 |

Imbalance ratio: 2.15x

## Quality Issues Detected
| Issue | Value |
|-------|-------|
| Mislabeled (folder vs filename) | 318 / 445 (71.5%) |
| Blurry images (var < 100) | 157 (35.3%) |
| Exact duplicate groups | 6 |
| Near-duplicate pairs | 11 |
| File size outliers | 22 |
| Corrupted images | 0 |

## Experiments

### Experiment 1: Baseline
ResNet-18 from scratch, 10 epochs, folder labels.
- **Best val acc: 44.94%** → later improved to **49.43%** with dedup+stratified split

### Experiment 2: Relabel via filename
Same architecture, filename-prefix labels.
- **Best val acc: 20.22-21.84%** — essentially random (16.7%)
- Loss stuck at ~1.78 (random = 1.79)
- Filename prefixes are NOT reliable ground truth

### Experiment 3: All 6 fixes + relabel
Stratified split, dedup, class weights, label smoothing (0.1), blur down-weight.
- **Best val acc: 24.14%** — +2.3% over relabel-only
- Fixes help but can't compensate for bad labels

### Experiment 4: Confident Learning
3-fold CV, CleanLab-style per-class thresholding.
- **OOF accuracy on original labels: 9.89%** (confirms severe label noise)
- Only 29/445 labels corrected with high confidence
- **Retrain on corrected labels: 47.19%** — close to but below baseline

## Conclusion
The 71.5% label noise rate is too high for any single technique to fix. Both metadata sources (folders and filenames) are unreliable. The model trained on noisy folder labels (49.43%) is the best available — it extracts the maximum usable signal. Next steps: iterative pseudo-labeling with self-training.
