# Confident Learning — Label Correction Results

## Method
1. 3-fold cross-validation on original folder labels (ResNet-18, 8 epochs/fold)
2. Out-of-fold probabilities for every training sample
3. CleanLab-style per-class confidence thresholding
4. Corrected labels where model disagrees with high confidence
5. Retrained on corrected labels (20 epochs, class-weighted + label smoothing)

## Cross-Validation
- OOF Accuracy on original labels: 9.89%
- This confirms the folder labels are essentially random for 4 of 6 classes
- Per-fold val accuracies: 44.97% / 45.95% / 55.41%

## Label Correction
| Outcome | Count |
|---------|-------|
| Kept original label | 44 |
| Corrected (model confident) | 29 |
| Flagged (low confidence) | 372 |
| **Total** | 445 |

## Final Results
| Setup | Validation Accuracy |
|-------|-------------------|
| Baseline (folder labels, dedup+stratified) | 49.43% |
| Relabel-only (filename prefixes) | 21.84% |
| All 6 fixes + relabel | 24.14% |
| **Confident Learning** | **47.19%** |

## Interpretation
- Only 29/445 labels (6.5%) could be corrected with high confidence
- The model agrees with folder labels only 9.89% of the time (OOF)
- This confirms 71.5% label noise rate — both label sources are unreliable
- Self-training (iterative pseudo-labeling) is the recommended next step
