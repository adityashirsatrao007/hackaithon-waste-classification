import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import WasteClassifier, CLASSES, NUM_CLASSES, IMG_SIZE

EPOCHS = 1
BATCH_SIZE = 16
LR = 0.0001
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
print("ResNet-18 from scratch (no pretrained weights)")

def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(s)

train_tfm = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tfm = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class WasteDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def load_data(data_dir, val_split=0.2):
    data_dir = Path(data_dir)
    samples = []
    for idx, cls in enumerate(CLASSES):
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for f in (data_dir / cls).glob(ext):
                samples.append((str(f.resolve()), idx))
    return samples

def train():
    seed_everything(SEED)

    data_dir = PROJECT_ROOT / "data" / "train"
    print("\nLoading data...")
    samples = load_data(data_dir)
    print(f"  Total: {len(samples)}")
    for cls in CLASSES:
        cnt = sum(1 for _, l in samples if l == CLASSES.index(cls))
        print(f"    {cls}: {cnt}")

    random.shuffle(samples)
    split = int((1 - 0.2) * len(samples))
    train_samps = samples[:split]
    val_samps = samples[split:]
    print(f"\n  Train: {len(train_samps)}  Val: {len(val_samps)}")

    train_loader = DataLoader(WasteDataset(train_samps, train_tfm),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(WasteDataset(val_samps, val_tfm),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = WasteClassifier(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = 100 * correct / total
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Val Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            print("  --> new best")

    print(f"\nBest val accuracy: {best_acc:.2f}%")
    if best_state is not None:
        model.load_state_dict(best_state)
    save_path = PROJECT_ROOT / "models" / "best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}")

    return model, history, (all_preds, all_labels)

if __name__ == "__main__":
    train()
