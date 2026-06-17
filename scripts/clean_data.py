import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import WasteClassifier, NUM_CLASSES
from src.dataset import load_folder_labeled, stratify_split, make_dataloaders
from src.trainer import seed_everything, train_epochs

EPOCHS = 10
BATCH_SIZE = 16
LR = 0.0001
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")


def main():
    seed_everything(SEED)
    data_dir = PROJECT_ROOT / "data" / "train"

    # ── Baseline: folder labels ──
    print("=" * 60)
    print("  BASELINE: folder labels")
    print("=" * 60)
    raw_samples = load_folder_labeled(data_dir)
    raw_train, raw_val = stratify_split(raw_samples, seed=SEED)
    train_loader, val_loader = make_dataloaders(raw_train, raw_val, BATCH_SIZE)

    model = WasteClassifier(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    _, baseline_acc, _ = train_epochs(
        train_loader, val_loader, model, criterion, optimizer, scheduler, EPOCHS, device)
    time.sleep(3)

    # ── Relabel-only: filename prefixes ──
    print("\n" + "=" * 60)
    print("  RELABEL: filename prefix labels")
    print("=" * 60)
    from src.dataset import load_relabeled
    clean_samples = load_relabeled(data_dir)
    clean_train, clean_val = stratify_split(clean_samples, seed=SEED)
    train_loader2, val_loader2 = make_dataloaders(clean_train, clean_val, BATCH_SIZE)

    model2 = WasteClassifier(NUM_CLASSES).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=LR)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.1)

    _, clean_acc, _ = train_epochs(
        train_loader2, val_loader2, model2, criterion, optimizer2, scheduler2, EPOCHS, device)

    # ── Results ──
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Baseline (folder labels):    {baseline_acc:.2f}%")
    print(f"  Relabel (filename labels):   {clean_acc:.2f}%")
    print(f"  Improvement:                 +{clean_acc - baseline_acc:.2f}%")

    from src.predictor import generate_predictions
    generate_predictions(
        model2,
        PROJECT_ROOT / "data" / "test",
        PROJECT_ROOT / "submissions" / "sample_submission.csv",
        PROJECT_ROOT / "submissions" / "submission_clean.csv",
        device,
    )

    report = f"""# Ablation: Label Correction

| Setup | Val Acc |
|-------|---------|
| Baseline (folder) | {baseline_acc:.2f}% |
| Clean (filename) | {clean_acc:.2f}% |
"""
    (PROJECT_ROOT / "reports" / "experiments" / "baseline.md").write_text(report)


if __name__ == "__main__":
    main()
