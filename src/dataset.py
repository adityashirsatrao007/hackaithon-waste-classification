import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

import imagehash

from src.model import CLASSES, NUM_CLASSES, IMG_SIZE

train_tfm = transforms.Compose([
    transforms.Resize(256),
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
    def __init__(self, samples, transform=None, weights=None, return_paths=False):
        self.samples = samples
        self.transform = transform
        self.weights = weights
        self.return_paths = return_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.weights is not None:
            return (img, label, self.weights[idx])
        if self.return_paths:
            return img, label, str(path)
        return img, label


def get_true_label_from_filename(path):
    label_str = Path(path).stem.split("_")[0]
    if label_str in CLASSES:
        return CLASSES.index(label_str)
    return None


def load_folder_labeled(data_dir, deduplicate=False):
    samples = []
    for idx, cls in enumerate(CLASSES):
        for f in sorted((Path(data_dir) / cls).glob("*.*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append((str(f.resolve()), idx))
    if deduplicate:
        samples = remove_duplicates(samples)
    return samples


def load_relabeled(data_dir, deduplicate=False):
    data_dir = Path(data_dir)
    samples = []
    for cls in CLASSES:
        for f in (data_dir / cls).glob("*.*"):
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            true_idx = get_true_label_from_filename(f)
            if true_idx is not None:
                samples.append((str(f.resolve()), true_idx))
    if deduplicate:
        samples = remove_duplicates(samples)
    return samples


def remove_duplicates(samples):
    md5_map = {}
    unique = []
    for path, label in samples:
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
        if h not in md5_map:
            md5_map[h] = (path, label)
            unique.append((path, label))

    phash_map = {}
    kept = []
    for path, label in unique:
        try:
            ph = imagehash.phash(Image.open(path))
            is_dup = any(ph - existing_ph <= 8 for existing_ph, _ in phash_map.values())
            if not is_dup:
                phash_map[path] = (ph, label)
                kept.append((path, label))
        except:
            kept.append((path, label))
    return kept


def compute_blur_weights(samples, threshold=100.0):
    weights = []
    blur_count = 0
    for path, _ in samples:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            var = cv2.Laplacian(arr, cv2.CV_64F).var()
            if var < threshold:
                weights.append(0.3)
                blur_count += 1
            else:
                weights.append(1.0)
        else:
            weights.append(1.0)
    return np.array(weights, dtype=np.float32), blur_count


def compute_class_weights(samples, device):
    counts = Counter(s[1] for s in samples)
    total = sum(counts.values())
    weights = [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float).to(device)


def stratify_split(samples, val_size=0.2, seed=42):
    labels = [s[1] for s in samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(samples, labels))
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


def make_dataloaders(train_samps, val_samps, batch_size=16, use_blur_weights=False):
    sample_weights = None
    if use_blur_weights:
        bw, _ = compute_blur_weights(train_samps)
        cw = np.array([1.0] * len(train_samps))
        sample_weights = torch.tensor(bw * cw, dtype=torch.float)

    train_set = WasteDataset(train_samps, train_tfm, weights=sample_weights)
    val_set = WasteDataset(val_samps, val_tfm)

    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
