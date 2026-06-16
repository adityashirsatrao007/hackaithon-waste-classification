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
LR = 0.0001
SEED = 42
NUM_CLASSES = 6
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

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

class ResNet18Classifier(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        feat = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n),
        )

    def forward(self, x):
        return self.classifier(self.resnet(x))

train_tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tfm = transforms.Compose([
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
    base = Path(__file__).parent

    print("\nLoading data...")
    samples = load_data(base / "data" / "train")
    print(f"  Total: {len(samples)}")
    for cls in CLASSES:
        cnt = sum(1 for _, l in samples if l == CLASSES.index(cls))
        print(f"    {cls}: {cnt}")

    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samps = samples[:split]
    val_samps = samples[split:]
    print(f"\n  Train: {len(train_samps)}  Val: {len(val_samps)}")

    train_loader = DataLoader(WasteDataset(train_samps, train_tfm),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(WasteDataset(val_samps, val_tfm),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNet18Classifier(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    best_state = None

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
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        scheduler.step()
        print(f"Epoch {epoch+1} - Loss: {running_loss:.4f} - Val Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            print(f"  --> new best")

    print(f"\nBest val accuracy: {best_acc:.2f}%")
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), base / "best_model.pth")
    print("Saved best_model.pth")

if __name__ == "__main__":
    train()
