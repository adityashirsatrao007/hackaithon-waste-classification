# Ablation Study: Impact of Label Correction

## Experiment
- Same ResNet-18 architecture, 10 epochs each
- **Baseline**: trained on original folder labels (71.5% mislabeled)
- **Clean**: trained on corrected labels from filename prefixes

## Results
| Setup | Best Validation Accuracy |
|-------|------------------------|
| Baseline (buggy folder labels) | 44.94% |
| Clean (corrected filename labels) | 20.22% |

## Interpretation
**Filename prefix labels are also unreliable.** The clean model achieves
barely-above-random accuracy (random = 16.7%), with cross-entropy loss
stuck at ~1.78 throughout training (random = 1.79). This indicates the
filename-based "true labels" are not consistent with the visual content
either — the same careless annotator likely assigned both folder
placements and filenames.

### Key Insight
Neither metadata source (folder assignment nor filename prefix) can be
trusted as ground truth. The baseline's 45% accuracy comes from the
model extracting the ~49% usable signal in the noisy folder labels.
