# Waste Classification -- Data Detective Challenge

This is my submission for the Hack[AI]Thon 2.0 Online Selection Round. The task: take a waste classification dataset with known quality issues, find every bug, quantify the impact, and show what it would take to fix things.

## The Short Version

The dataset has 445 training images across 6 waste types (cardboard, glass, metal, paper, plastic, trash). It's small enough that you could train a ResNet-18 in about 10 seconds per epoch. But the labels are a mess.

### What I Found

| Problem | How Bad |
|---------|---------|
| 71.5% of images in wrong class folder | The person who organized this put most images in the wrong directory |
| Class imbalance | trash has 40 images, paper has 86 -- the model barely sees trash |
| 157 blurry images | A third of the dataset is out of focus |
| 6 exact duplicate groups | Same image, same hash, sitting in the same folder |
| Filename labels are also unreliable | The filenames claim to reveal true labels, but training on them yields near-random accuracy |

The dataset uses a naming convention `{true_label}_random_id.jpg`. I was able to extract the intended label from each filename, but when I trained a model on those corrected labels, it achieved only 20% accuracy (random would be 16.7%). This means the person who mis-sorted the images into folders also assigned incorrect filenames.

## Experiments

I ran a progression of experiments, each testing a hypothesis about what would fix the label noise.

### 1. Baseline (buggy folder labels)
**Accuracy: 49.43%** — Simple ResNet-18 trained on the raw folder labels. This is the upper reference point since the model memorizes the 49% signal / 51% noise in the folder labels.

### 2. Relabel via filename prefix
**Accuracy: 21.84%** — Used `{label}_id.jpg` filename prefix as corrected ground truth. The model barely beats random (16.7%). Both metadata sources are unreliable.

### 3. All 6 fixes + relabel
**Accuracy: 24.14%** — Stratified split, deduplication, class-weighted loss, label smoothing, and blur down-weight applied on top of filename relabeling. Modest improvement over relabel-only (+2.3%), confirming the fixes help but can't compensate for bad labels.

### 4. Confident Learning (CleanLab-style)
**Accuracy: 47.19%** — 3-fold cross-validation to estimate per-class label confidence, then automatically corrected 29 high-confidence mismatches. Result is close to but below the baseline. Only 29/445 labels could be corrected with confidence — the label noise is too severe for CleanLab to distinguish signal.

### Full Results Table

| Experiment | Val Acc | Notes |
|------------|---------|-------|
| Baseline (no fixes) | 44.94% | Raw folder labels, random split |
| Baseline (dedup + stratified) | 49.43% | Fixed splits, removed exact duplicates |
| Relabel-only (filename labels) | 21.84% | Filename prefix as label |
| All 6 fixes + relabel | 24.14% | Class weights, label smoothing, blur down-weight |
| Confident Learning | 47.19% | 3-fold CV, corrected 29/445 labels |

### Key Insight

The 71.5% label noise rate is so high that:
- **Filename prefixes are useless** (21.84% accuracy ≈ random)
- **Confident learning can only fix 29 out of 445 labels** (the model can't reliably determine which labels are wrong)
- **Self-training would be the next step** — use the model itself to generate pseudo-labels iteratively — but requires the original dataset

The baseline model (49.43%) is actually the best available, since it extracts the maximum signal from the noisy folder labels. All attempts at "fixing" labels introduced more noise than they removed.

## Repository Layout

```
configs/          training configuration
models/           trained model weights (gitignored)
notebooks/        exploratory analysis
reports/          findings, ablation studies, evidence, figures
src/              all source code
submissions/      prediction CSVs
```

## Running the Code

```bash
# Audit the dataset
python src/inspect_dataset.py

# Train baseline (10 epochs)
python src/train.py --epochs 10

# Generate test predictions
python src/predict.py

# Visual analysis
python src/visualize.py

# Label correction ablation study
python src/clean_data.py

# Improved pipeline (all fixes)
python src/improved_pipeline.py

# Confident learning
python src/confident_learning.py
```

Configuration is in `configs/config.yaml`.
