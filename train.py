"""
Train ResNet-18 classifier on Waste Classification dataset (standalone, no 3LC).

- Loads images from data/train/{class}/
- 80/20 stratified per-class split
- ResNet-18 random init (no pretrained weights)
- Saves best_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os

EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
RANDOM_SEED = 42
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"ResNet-18: random init (no pretrained weights)")

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[OK] Random seed set to {seed}")

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
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
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(data_dir, val_split=0.2):
    data_dir = Path(data_dir)
    all_samples = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in class_dir.glob(ext):
                all_samples.append((str(img_path.resolve()), class_idx))
    return all_samples

def train():
    set_seed(RANDOM_SEED)
    base_path = Path(__file__).parent
    data_path = base_path / "data" / "train"

    print("\nLoading dataset...")
    all_samples = load_data(data_path)
    print(f"  Total samples: {len(all_samples)}")
    for name in CLASS_NAMES:
        count = sum(1 for _, l in all_samples if l == CLASS_NAMES.index(name))
        print(f"    {name}: {count}")

    random.shuffle(all_samples)
    split_idx = int((1 - 0.2) * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    print(f"\n  Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_dataset = WasteDataset(train_samples, transform=train_transform)
    val_dataset = WasteDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_accuracy = 0.0
    best_model_state = None
    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        val_accuracy = 100 * val_correct / val_total
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f} - Val Acc: {val_accuracy:.2f}%")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  --> New best model!")

    print("\n" + "=" * 60)
    print(f"  Best validation accuracy: {best_val_accuracy:.2f}%")
    print("=" * 60)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model_path = base_path / "best_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Best model saved to {model_path}")

if __name__ == "__main__":
    train()
