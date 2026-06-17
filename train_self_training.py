"""
Self-training loop for waste classification with 71.5% label noise.
Strategy: Train -> predict on train set -> relabel high-confidence disagreements -> repeat.
Expected: 75-85% accuracy vs 64% baseline.
"""

import os, sys, json, copy, random, math
from dataclasses import dataclass
from collections import Counter
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# -- Config --
@dataclass
class Config:
    data_root: str = "data/train"
    test_root: str = "data/test"
    class_names: tuple = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
    num_classes: int = 6
    img_size: int = 224
    batch_size: int = 32
    epochs_per_round: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    confidence_threshold: float = 0.70
    max_rounds: int = 10
    min_new_labels: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    results_dir: str = "self_training_output"
    submission_path: str = "submissions/submission.csv"

LEAKED_IMAGES = {
    "test_b201d584.jpg", "test_bd255afc.jpg", "test_f5730887.jpg", "test_fc9b8cc6.jpg"
}
DUPED_TRAIN = {
    "cardboard_hrjs9ws3.jpg", "cardboard_wlvm44zt.jpg",
    "glass_bmbmrr5c.jpg", "glass_mx15crtc.jpg",
    "paper_aucwnlvn.jpg", "paper_tk2xs0i5.jpg",
    "trash_bbyyqot9.jpg",
}

train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class WasteDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
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

def make_model():
    return ResNet18Classifier()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return loss_sum / len(loader), 100.0 * correct / total

def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    all_probs = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total, torch.cat(all_probs)

def get_predictions(model, loader, device):
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, predicted = probs.max(1)
            preds.extend(predicted.cpu().tolist())
            confs.extend(max_probs.cpu().tolist())
    return preds, confs

def load_train_data():
    paths, labels = [], []
    cmap = {c: i for i, c in enumerate(Config.class_names)}
    for cl in Config.class_names:
        cl_dir = os.path.join(Config.data_root, cl)
        for fname in sorted(os.listdir(cl_dir)):
            paths.append(os.path.join(cl_dir, fname))
            labels.append(cmap[cl])
    return paths, labels

def load_test_data():
    cfg = Config
    paths = sorted(glob(os.path.join(cfg.test_root, "*.jpg")))
    return paths  # include all 115 test images

def main():
    cfg = Config
    os.makedirs(cfg.results_dir, exist_ok=True)

    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 60)
    print("SELF-TRAINING WITH CONFIDENT LEARNING")
    print(f"Device: {cfg.device}, Threshold: {cfg.confidence_threshold}")
    print("=" * 60)

    # Load data
    all_paths, folder_labels = load_train_data()
    print(f"\nTotal: {len(all_paths)}")

    # Remove known dupes
    keep = [os.path.basename(p) not in DUPED_TRAIN for p in all_paths]
    all_paths = [p for p, k in zip(all_paths, keep) if k]
    folder_labels = [l for l, k in zip(folder_labels, keep) if k]
    print(f"After dedup: {len(all_paths)}")

    current_labels = list(folder_labels)

    # Stratified split
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
    train_idx, val_idx = next(sss.split(list(range(len(all_paths))), folder_labels))

    train_paths_all = [all_paths[i] for i in train_idx]
    val_paths_all = [all_paths[i] for i in val_idx]
    print(f"Train: {len(train_paths_all)}, Val: {len(val_paths_all)}")

    round_log = []

    for rnd in range(1, cfg.max_rounds + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"ROUND {rnd}", flush=True)
        print("=" * 60, flush=True)

        tr_paths = [all_paths[i] for i in train_idx]
        tr_labels = [current_labels[i] for i in train_idx]
        vl_paths = [all_paths[i] for i in val_idx]
        vl_labels = [current_labels[i] for i in val_idx]

        # Weighted sampler
        counts = Counter(tr_labels)
        sample_weights = torch.DoubleTensor([1.0 / counts[l] for l in tr_labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(WasteDataset(tr_paths, tr_labels, train_tfm),
                                   batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(WasteDataset(vl_paths, vl_labels, val_tfm),
                                 batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        model = make_model().to(cfg.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs_per_round)

        best_val_acc = 0
        best_state = None

        for ep in range(1, cfg.epochs_per_round + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
            val_acc, _ = evaluate(model, val_loader, cfg.device)
            scheduler.step()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
            if ep % 10 == 0 or ep == cfg.epochs_per_round:
                print(f"  Ep {ep:2d}/{cfg.epochs_per_round} | Loss: {train_loss:.4f} | Tr: {train_acc:.2f}% | Val: {val_acc:.2f}%", flush=True)

        print(f"  -> Best val: {best_val_acc:.2f}%")
        ckpt_path = os.path.join(cfg.results_dir, f"round{rnd}_best.pth")
        torch.save(best_state, ckpt_path)
        print(f"  -> Saved: {ckpt_path}")

        # Relabel
        model.load_state_dict(best_state)
        all_loader = DataLoader(WasteDataset(all_paths, current_labels, val_tfm),
                                 batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        preds, confs = get_predictions(model, all_loader, cfg.device)

        new_labels = 0
        samples = []
        for i in range(len(all_paths)):
            if confs[i] >= cfg.confidence_threshold and preds[i] != current_labels[i]:
                samples.append((os.path.basename(all_paths[i]), current_labels[i], preds[i], confs[i]))
                current_labels[i] = preds[i]
                new_labels += 1

        changed = sum(1 for a, b in zip(current_labels, folder_labels) if a != b)
        round_log.append({"round": rnd, "val_acc": best_val_acc, "new_labels": new_labels, "total_changed": changed})
        print(f"  -> New relabels: {new_labels}, Total changed: {changed}/{len(all_paths)}")

        if samples:
            for fname, old, new, conf in samples[:5]:
                print(f"     {fname}: {cfg.class_names[old]} -> {cfg.class_names[new]} (conf={conf:.3f})")

        if new_labels < cfg.min_new_labels:
            print(f"\n Converged! Only {new_labels} new labels.")
            break

    # Final training
    print(f"\n{'='*60}")
    print("FINAL TRAINING ON CORRECTED LABELS")
    print("=" * 60)

    counts = Counter(current_labels)
    sample_weights = torch.DoubleTensor([1.0 / counts[l] for l in current_labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    final_loader = DataLoader(WasteDataset(all_paths, current_labels, train_tfm),
                               batch_size=cfg.batch_size, sampler=sampler, num_workers=0)

    model = make_model().to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs_per_round)

    for ep in range(1, cfg.epochs_per_round + 1):
        train_loss, train_acc = train_epoch(model, final_loader, nn.CrossEntropyLoss(), optimizer, cfg.device)
        scheduler.step()
        if ep % 10 == 0 or ep == cfg.epochs_per_round:
            print(f"  Ep {ep:2d}/{cfg.epochs_per_round} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

    # Submission
    print(f"\n{'='*60}")
    print("GENERATING SUBMISSION")
    print("=" * 60)

    test_paths = load_test_data()
    print(f"Test: {len(test_paths)}")

    test_loader = DataLoader(
        WasteDataset(test_paths, [0] * len(test_paths), val_tfm),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    model.eval()
    results = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(cfg.device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, predicted = probs.max(1)
            for pred, conf in zip(predicted.cpu().tolist(), max_probs.cpu().tolist()):
                results.append((pred, conf))

    all_test_files = sorted(os.listdir(cfg.test_root))

    sub_path = cfg.submission_path
    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
    with open(sub_path, "w") as f:
        f.write("image_id,prediction,confidence\n")
        for fname, (pred, conf) in zip(all_test_files, results):
            image_id = fname.replace(".jpg", "")
            f.write(f"{image_id},{pred},{conf}\n")

    print(f"Saved: {sub_path}")
    torch.save(model.state_dict(), os.path.join(cfg.results_dir, "best_model.pth"))

    # Report
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print("=" * 60)
    final_dist = Counter(current_labels)
    print(f"\nSelf-training: {rnd} rounds")
    cmap = {i: c for c, i in zip(cfg.class_names, range(6))}
    for k in sorted(final_dist):
        print(f"  {cmap[k]:12s}: {final_dist[k]:3d} (orig {Counter(folder_labels)[k]:3d})")

    agree = sum(1 for a, b in zip(current_labels, folder_labels) if a == b)
    print(f"\nChanged: {len(all_paths) - agree}/{len(all_paths)} ({100*(1-agree/len(all_paths)):.1f}%)")
    print(f"Unchanged: {agree}/{len(all_paths)} ({100*agree/len(all_paths):.1f}%)")

    with open(os.path.join(cfg.results_dir, "self_training_log.json"), "w") as f:
        json.dump({
            "rounds": round_log,
            "final_dist": {cmap[k]: v for k, v in final_dist.items()},
            "changed": len(all_paths) - agree,
            "unchanged": agree,
        }, f, indent=2)

    print(f"\nAll outputs: {cfg.results_dir}/")

if __name__ == "__main__":
    main()
