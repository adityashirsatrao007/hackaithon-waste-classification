# Ablation Study: Impact of All Fixes

## Experiment Setup
- ResNet-18 from scratch, 10 epochs, Adam lr=0.0001
- Stratified train/val split (80/20) for all runs
- Label source: filename prefix (true_label portion)

## Fixes Applied per Run

| Fix | Baseline | Relabel Only | Improved |
|-----|----------|-------------|----------|
| Relabel (318/445) | ✗ | ✓ | ✓ |
| Deduplicate | ✓ | ✓ | ✓ |
| Stratified split | ✗ | ✗ | ✓ |
| Class-weighted loss | ✗ | ✗ | ✓ |
| Label smoothing (0.1) | ✗ | ✗ | ✓ |
| Blurry down-weight | ✗ | ✗ | ✓ |

## Results
| Setup | Validation Accuracy |
|-------|-------------------|
| Baseline (folder labels, dedup+stratified) | 49.43% |
| Relabel only (filename labels) | 21.84% |
| **Improved (all fixes + relabel)** | **24.14%** |

## Interpretation
- Additional fixes beyond relabeling are needed
- Weighted loss + label smoothing + blur down-weight adds modest gains (+2.3%)
- The filename labels are too unreliable for any fix to salvage
- Best approach: train on noisy folder labels directly (49.43%) or use self-training
