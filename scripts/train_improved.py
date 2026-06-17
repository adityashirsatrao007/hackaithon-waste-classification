import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model import WasteClassifier, NUM_CLASSES
from src.dataset import load_relabeled, remove_duplicates
from src.trainer import seed_everything, train_model
from src.predictor import generate_predictions

EPOCHS = 10
BATCH_SIZE = 16
LR = 0.0001
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")


def main():
    seed_everything(SEED)
    data_dir = PROJECT_ROOT / "data" / "train"

    # ── Relabel + Dedup ──
    print("=" * 60)
    print("  DATA CLEANING")
    print("=" * 60)
    samples_raw = load_relabeled(data_dir)
    print(f"  Total: {len(samples_raw)}")
    samples_clean = remove_duplicates(samples_raw)
    print(f"  After dedup: {len(samples_clean)}")

    # ── Baseline ──
    raw_samples = load_relabeled(data_dir)
    model_base, acc_base, _ = train_model(
        raw_samples, "BASELINE", EPOCHS, BATCH_SIZE, LR,
        label_smoothing=0.0, use_class_weights=False,
        use_blur_weights=False, device=device,
    )
    time.sleep(3)

    # ── Improved ──
    model_imp, acc_imp, _ = train_model(
        samples_clean, "IMPROVED", EPOCHS, BATCH_SIZE, LR,
        label_smoothing=0.1, use_class_weights=True,
        use_blur_weights=True, device=device,
    )
    time.sleep(3)

    # ── Relabel-only ──
    model_relabel, acc_relabel, _ = train_model(
        samples_clean, "RELABEL ONLY", EPOCHS, BATCH_SIZE, LR,
        label_smoothing=0.0, use_class_weights=False,
        use_blur_weights=False, device=device,
    )

    # ── Results ──
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Baseline (buggy labels):    {acc_base:.2f}%")
    print(f"  Relabel only:               {acc_relabel:.2f}%")
    print(f"  Improved (all fixes):       {acc_imp:.2f}%")

    save_path = PROJECT_ROOT / "models" / "improved_model.pth"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model_imp.state_dict(), save_path)
    print(f"\nSaved {save_path}")

    # ── Submission ──
    print("\nGenerating submission...")
    generate_predictions(
        model_imp,
        PROJECT_ROOT / "data" / "test",
        PROJECT_ROOT / "submissions" / "sample_submission.csv",
        PROJECT_ROOT / "submissions" / "submission_improved.csv",
        device,
    )

    # ── Save report ──
    report = f"""# Ablation Study: All Fixes

| Setup | Val Acc |
|-------|---------|
| Baseline | {acc_base:.2f}% |
| Relabel only | {acc_relabel:.2f}% |
| Improved (all 6 fixes) | **{acc_imp:.2f}%** |
"""
    (PROJECT_ROOT / "reports" / "experiments" / "improved.md").write_text(report)


if __name__ == "__main__":
    main()
