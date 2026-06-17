import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model import WasteClassifier, NUM_CLASSES
from src.dataset import load_folder_labeled, stratify_split, make_dataloaders
from src.trainer import seed_everything, train_epochs, torch, nn, optim

EPOCHS = 10
BATCH_SIZE = 16
LR = 0.0001
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}  ResNet-18 from scratch")


def main():
    seed_everything(SEED)
    data_dir = PROJECT_ROOT / "data" / "train"

    print("\nLoading data...")
    samples = load_folder_labeled(data_dir, deduplicate=True)
    print(f"  Total: {len(samples)}")
    train_samps, val_samps = stratify_split(samples, seed=SEED)
    print(f"  Train: {len(train_samps)}  Val: {len(val_samps)}")

    train_loader, val_loader = make_dataloaders(train_samps, val_samps, BATCH_SIZE)

    model = WasteClassifier(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model, best_acc, history = train_epochs(
        train_loader, val_loader, model, criterion, optimizer, scheduler, EPOCHS, device)

    print(f"\nBest val accuracy: {best_acc:.2f}%")
    save_path = PROJECT_ROOT / "models" / "best_model.pth"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
