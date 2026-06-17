"""
Train ResNet-18 for waste classification — data-centric approach.
No architecture modifications. Fixes data issues only.
"""

import torch, torch.nn as nn, torch.optim as optim
import torchvision.models as models, torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from pathlib import Path
from collections import Counter
import random, numpy as np, os, csv, json, shutil
from tqdm import tqdm
import imagehash

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
DATA_DIR = Path("data/train")
TEST_DIR = Path("data/test")

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

def build_clean_dataset():
    samples = []
    for cls in CLASSES:
        folder = DATA_DIR / cls
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".jpg"): continue
            samples.append((folder / fname, CLASS_TO_IDX[cls]))

    seen_phashes = {}
    deduped = []
    dup_count = 0
    for path, label in samples:
        ph = imagehash.phash(Image.open(path))
        if ph in seen_phashes:
            dup_count += 1
            continue
        seen_phashes[ph] = (path, label)
        deduped.append((path, label))
    print(f"Removed {dup_count} intra-train perceptual duplicates")

    train_phashes = {}
    for path, label in deduped:
        train_phashes[imagehash.phash(Image.open(path))] = (path, label)

    test_leaks = set()
    for fname in sorted(os.listdir(TEST_DIR)):
        test_ph = imagehash.phash(Image.open(TEST_DIR / fname))
        for tr_ph in train_phashes:
            if 0 <= test_ph - tr_ph <= 4:
                test_leaks.add(train_phashes[tr_ph])

    cleaned = [(p, l) for p, l in deduped if (p, l) not in test_leaks]
    print(f"Removed {len(deduped) - len(cleaned)} train/test leaked images")
    print(f"Final dataset: {len(cleaned)} images")
    return cleaned

def main():
    print("Building clean dataset...")
    clean_samples = build_clean_dataset()

    random.seed(42)
    random.shuffle(clean_samples)
    split = int(0.8 * len(clean_samples))
    train_samples = clean_samples[:split]
    val_samples = clean_samples[split:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    label_counts = Counter(l for _, l in train_samples)
    weights = [1.0 / label_counts[l] for _, l in train_samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = WasteDataset(train_samples, train_transform)
    val_ds = WasteDataset(val_samples, val_transform)
    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                pred = model(images).argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        scheduler.step()
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  → New best model ({best_acc:.2f}%)")

    print(f"\nBest validation accuracy: {best_acc:.2f}%")

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class TestDataset(Dataset):
        def __init__(self, transform=None):
            self.transform = transform
            self.images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")])
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            name = self.images[idx]
            img = Image.open(TEST_DIR / name).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, name.replace(".jpg", "")

    test_ds = TestDataset(test_transform)
    test_loader = DataLoader(test_ds, BATCH_SIZE, num_workers=0)

    results = []
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1)
            conf, pred = probs.max(1)
            for fid, p, c in zip(ids, pred.cpu().numpy(), conf.cpu().numpy()):
                results.append({"image_id": fid, "prediction": int(p), "confidence": float(c)})

    with open("submission.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        w.writeheader()
        w.writerows(results)

    print(f"\nSaved {len(results)} predictions to submission.csv")
    shutil.copy2("submission.csv", "submissions/submission.csv")
    print("Done!")

if __name__ == "__main__":
    main()
