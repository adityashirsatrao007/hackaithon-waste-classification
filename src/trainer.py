import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import make_dataloaders, compute_class_weights, stratify_split, train_tfm, val_tfm, WasteDataset
from src.model import WasteClassifier, NUM_CLASSES


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(s)


def train_epochs(train_loader, val_loader, model, criterion, optimizer, scheduler, epochs, device):
    best_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc)
        print(f"  Epoch {epoch+1} — Loss: {avg_loss:.4f} — Val Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc, history


def train_model(
    samples,
    name="model",
    epochs=10,
    batch_size=16,
    lr=0.0001,
    label_smoothing=0.0,
    use_class_weights=False,
    use_blur_weights=False,
    device="cpu",
):
    train_samps, val_samps = stratify_split(samples)
    print(f"\n  {name}: Train {len(train_samps)}, Val {len(val_samps)}")
    train_loader, val_loader = make_dataloaders(
        train_samps, val_samps, batch_size, use_blur_weights=use_blur_weights)

    model = WasteClassifier(NUM_CLASSES).to(device)
    if use_class_weights:
        cw = compute_class_weights(train_samps, device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model, best_acc, history = train_epochs(
        train_loader, val_loader, model, criterion, optimizer, scheduler, epochs, device)
    return model, best_acc, history


def cross_val_predict(samples, n_folds=3, epochs=8, batch_size=16, lr=0.0001, device="cpu"):
    from sklearn.model_selection import StratifiedKFold
    import torch.nn.functional as F
    from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

    class _IndexedDataset(TorchDataset):
        def __init__(self, samps, transform):
            self.samps = samps
            self.transform = transform
            self._underlying = WasteDataset(samps, transform)

        def __len__(self):
            return len(self.samps)

        def __getitem__(self, idx):
            img, label = self._underlying[idx]
            return img, label, idx

    labels = np.array([s[1] for s in samples])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(samples), NUM_CLASSES), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(samples, labels)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        train_samps = [samples[i] for i in train_idx]
        val_samps = [samples[i] for i in val_idx]

        train_loader = TorchDataLoader(
            WasteDataset(train_samps, train_tfm),
            batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = TorchDataLoader(
            _IndexedDataset(val_samps, val_tfm),
            batch_size=batch_size, shuffle=False, num_workers=0)

        model = WasteClassifier(NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

        best_acc = 0
        best_state = None
        for epoch in range(epochs):
            model.train()
            for images, labels_b in train_loader:
                images, labels_b = images.to(device), labels_b.to(device)
                optimizer.zero_grad()
                criterion(model(images), labels_b).backward()
                optimizer.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels_b, _ in val_loader:
                    images, labels_b = images.to(device), labels_b.to(device)
                    pred = model(images).argmax(1)
                    correct += (pred == labels_b).sum().item()
                    total += labels_b.size(0)
            acc = 100 * correct / total
            scheduler.step()
            print(f"  Fold {fold+1} Epoch {epoch+1}: Val Acc = {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict().copy()

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            for images, _, idxs in val_loader:
                images = images.to(device)
                probs = F.softmax(model(images), dim=1)
                for i, idx in enumerate(idxs):
                    oof_probs[idx.numpy()] = probs[i].cpu().numpy()

    return oof_probs
