import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from src.model import CLASSES, NUM_CLASSES
from src.dataset import load_folder_labeled
from src.trainer import seed_everything, train_model, cross_val_predict
from src.analysis import confident_learning_correction
from src.predictor import generate_predictions

RETRAIN_EPOCHS = 10
BATCH_SIZE = 16
LR = 0.0001
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")


def main():
    seed_everything(SEED)
    data_dir = PROJECT_ROOT / "data" / "train"

    print("=" * 60)
    print("  CONFIDENT LEARNING LABEL CORRECTION")
    print("=" * 60)

    samples = load_folder_labeled(data_dir)
    print(f"\nLoaded {len(samples)} samples")

    # ── Cross-val predictions ──
    print("\n--- Step 1: Cross-validated predictions ---")
    oof_probs = cross_val_predict(samples, n_folds=3, epochs=8,
                                  batch_size=BATCH_SIZE, lr=LR, device=device)

    orig_labels = np.array([s[1] for s in samples])
    oof_preds = oof_probs.argmax(1)
    cv_acc = accuracy_score(orig_labels, oof_preds)
    print(f"\n  OOF Accuracy: {100*cv_acc:.2f}%")
    cm = confusion_matrix(orig_labels, oof_preds)
    print(f"  Confusion matrix:\n{cm}")

    # ── Apply confident learning ──
    print("\n--- Step 2: Confident Learning correction ---")
    corrected_samples, cl_stats, thresholds = confident_learning_correction(
        samples, oof_probs, NUM_CLASSES)
    print(f"  Kept: {cl_stats['kept']}, Corrected: {cl_stats['corrected']}, Flagged: {cl_stats['flagged']}")
    new_labels = Counter(s[1] for s in corrected_samples)
    for c in CLASSES:
        print(f"    {c}: {new_labels.get(CLASSES.index(c), 0)}")

    # ── Retrain ──
    model_cl, acc_cl, _ = train_model(
        corrected_samples, "CONFIDENT LEARNING", RETRAIN_EPOCHS,
        BATCH_SIZE, LR, label_smoothing=0.1, use_class_weights=True,
        use_blur_weights=False, device=device,
    )

    print(f"\n{'='*60}")
    print(f"  Baseline (folder labels):  49.43% (from previous run)")
    print(f"  Confident Learning:        {acc_cl:.2f}%")

    save_path = PROJECT_ROOT / "models" / "confident_learning_model.pth"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model_cl.state_dict(), save_path)
    print(f"Saved {save_path}")

    print("\nGenerating submission...")
    generate_predictions(
        model_cl,
        PROJECT_ROOT / "data" / "test",
        PROJECT_ROOT / "submissions" / "sample_submission.csv",
        PROJECT_ROOT / "submissions" / "submission_cl.csv",
        device,
    )

    report = f"""# Confident Learning — Results

- OOF Accuracy: {100*cv_acc:.2f}%
- Corrected: {cl_stats['corrected']}/{cl_stats['total']}
- Final Val Acc: {acc_cl:.2f}%
"""
    (PROJECT_ROOT / "reports" / "experiments" / "confident_learning.md").write_text(report)


if __name__ == "__main__":
    main()
