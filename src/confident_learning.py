"""
Confident Learning Pipeline — CleanLab-style label correction.

Approach:
1. 3-fold cross-validation on noisy folder labels → out-of-fold (OOF) probabilities
2. Apply confident learning to identify and correct label errors
3. Retrain on corrected labels
4. Iterate for further refinement

Confirmed: filename-prefix labels are NOT reliable (21.84% accuracy).
The model itself (at 49%+) contains better signal for label correction.
"""
import hashlib, json, random, os, sys, time, csv
from pathlib import Path
from collections import defaultdict, Counter

import cv2, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import WasteClassifier, CLASSES, NUM_CLASSES, IMG_SIZE

EPOCHS_PER_FOLD = 8
RETRAIN_EPOCHS = 10
BATCH_SIZE = 16
LR = 0.0001
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

train_tfm = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tfm = transforms.Compose([
    transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
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
        if self.transform: img = self.transform(img)
        return img, label, idx

# ─── LOAD DATA ───────────────────────────────────────────────────────────
def load_folder_labeled(data_dir):
    """Load samples with original folder labels."""
    samples = []
    for idx, cls in enumerate(CLASSES):
        for f in sorted((Path(data_dir) / cls).glob("*.*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append((str(f.resolve()), idx))
    return samples

# ─── CROSS-VAL PREDICTION ────────────────────────────────────────────────
def cross_val_predict(samples, n_folds=3):
    """Returns out-of-fold predicted probabilities for all samples."""
    labels = np.array([s[1] for s in samples])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(samples), NUM_CLASSES), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(samples, labels)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        train_samps = [samples[i] for i in train_idx]
        val_samps = [samples[i] for i in val_idx]

        train_loader = DataLoader(WasteDataset(train_samps, train_tfm),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(WasteDataset(val_samps, val_tfm),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = WasteClassifier(NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

        best_acc = 0; best_state = None
        for epoch in range(EPOCHS_PER_FOLD):
            model.train()
            for images, labels_b, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
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

        # Predict on held-out fold
        model.eval()
        with torch.no_grad():
            for images, _, idxs in val_loader:
                images = images.to(device)
                probs = nn.functional.softmax(model(images), dim=1)
                for i, idx in enumerate(idxs):
                    oof_probs[idx.numpy()] = probs[i].cpu().numpy()

    return oof_probs

# ─── CONFIDENT LEARNING ───────────────────────────────────────────────────
def confident_learning_correction(samples, oof_probs):
    """
    CleanLab-style confident learning:
    - For each class, compute the self-confidence threshold as mean predicted prob.
    - Find images where model is confident about a DIFFERENT class than the label.
    - Returns corrected samples + stats.
    """
    probs = oof_probs
    preds = probs.argmax(1)
    orig_labels = np.array([s[1] for s in samples])

    # Per-class confidence threshold = mean predicted prob for that class
    thresholds = {}
    for c in range(NUM_CLASSES):
        mask = orig_labels == c
        if mask.sum() > 0:
            thresholds[c] = probs[mask, c].mean()
        else:
            thresholds[c] = 0.5
    print(f"\n  Per-class confidence thresholds: {thresholds}")

    # Identify label errors:
    # For each sample labeled as c, check if the model assigns high prob (>threshold[d]) to class d != c
    corrected = []
    stats = {"total": len(samples), "kept": 0, "corrected": 0, "flagged": 0}

    for i, (path, orig_label) in enumerate(samples):
        max_prob_class = preds[i]
        max_prob = probs[i].max()
        orig_prob = probs[i, orig_label]

        if max_prob_class == orig_label:
            corrected.append((path, orig_label))
            stats["kept"] += 1
        elif max_prob > thresholds.get(max_prob_class, 0.5) and max_prob > 0.4:
            corrected.append((path, int(max_prob_class)))
            stats["corrected"] += 1
        else:
            # Low confidence → keep original but note it
            corrected.append((path, orig_label))
            stats["flagged"] += 1

    return corrected, stats

# ─── TRAINING ─────────────────────────────────────────────────────────────
def train_model(samples, name, epochs=RETRAIN_EPOCHS):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    from sklearn.model_selection import StratifiedShuffleSplit
    labels = [s[1] for s in samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss.split(samples, labels))
    train_samps = [samples[i] for i in train_idx]
    val_samps = [samples[i] for i in val_idx]
    print(f"  Train: {len(train_samps)}, Val: {len(val_samps)}")

    # Class weights
    counts = Counter(s[1] for s in train_samps)
    total = sum(counts.values())
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float).to(device)

    train_loader = DataLoader(WasteDataset(train_samps, train_tfm),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(WasteDataset(val_samps, val_tfm),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = WasteClassifier(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0; best_state = None; history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels_b, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels_b = images.to(device), labels_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels_b)
            loss.backward(); optimizer.step()
            running_loss += loss.item()

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
        avg_loss = running_loss / len(train_loader)
        history.append((avg_loss, acc))
        print(f"  Epoch {epoch+1} - Loss: {avg_loss:.4f} - Val Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc; best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc, history

# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    seed_everything(SEED)
    data_dir = PROJECT_ROOT / "data" / "train"

    print("=" * 60)
    print("  CONFIDENT LEARNING LABEL CORRECTION")
    print("=" * 60)

    # Step 1: Load data with folder labels
    samples = load_folder_labeled(data_dir)
    print(f"\nLoaded {len(samples)} samples with folder labels")

    # Step 2: Cross-validation predictions
    print("\n--- Step 1: Cross-validated predictions ---")
    oof_probs = cross_val_predict(samples, n_folds=3)

    # Check CV accuracy
    orig_labels = np.array([s[1] for s in samples])
    oof_preds = oof_probs.argmax(1)
    cv_acc = accuracy_score(orig_labels, oof_preds)
    print(f"\n  OOF Accuracy: {100*cv_acc:.2f}%")

    # Show confusion
    cm = confusion_matrix(orig_labels, oof_preds)
    print("  Confusion matrix (rows=true/folder, cols=CV predicted):")
    print(cm)

    # Step 3: Apply confident learning
    print("\n--- Step 2: Confident Learning correction ---")
    corrected_samples, cl_stats = confident_learning_correction(samples, oof_probs)
    print(f"  Kept: {cl_stats['kept']}, Corrected: {cl_stats['corrected']}, Flagged: {cl_stats['flagged']}")

    # Show relabeling distribution
    new_labels = Counter(s[1] for s in corrected_samples)
    print("  New label distribution:")
    for c in CLASSES:
        print(f"    {c}: {new_labels.get(CLASSES.index(c), 0)}")

    # Step 4: Train on corrected labels
    model_cl, acc_cl, hist_cl = train_model(corrected_samples, "CONFIDENT LEARNING CORRECTED")

    # Step 5: Compare with baseline
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    print(f"  Baseline (folder labels):        49.43% (from previous run)")
    print(f"  Confident Learning corrected:    {acc_cl:.2f}%")
    print(f"  Improvement:                     +{acc_cl - 49.43:.2f}%")

    # Save model
    save_path = PROJECT_ROOT / "models" / "confident_learning_model.pth"
    torch.save(model_cl.state_dict(), save_path)
    print(f"\nSaved model to {save_path}")

    # Generate submission
    print("\nGenerating submission...")
    model_cl.eval()
    from predict import TestDataset
    TEST_DIR = PROJECT_ROOT / "data" / "test"
    test_ds = TestDataset(TEST_DIR, val_tfm)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    predictions = []
    with torch.no_grad():
        for images, fnames in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            logits = model_cl(images)
            probs = nn.functional.softmax(logits, dim=1)
            confs, preds = probs.max(1)
            for fid, pred, conf in zip(fnames, preds.cpu().numpy(), confs.cpu().numpy()):
                predictions.append({"image_id": fid, "prediction": int(pred), "confidence": float(conf)})

    SAMPLE_PATH = PROJECT_ROOT / "submissions" / "sample_submission.csv"
    OUTPUT_PATH = PROJECT_ROOT / "submissions" / "submission_cl.csv"
    if SAMPLE_PATH.exists():
        with open(SAMPLE_PATH) as f:
            reader = csv.DictReader(f)
            expected = [row["image_id"] for row in reader]
        pred_map = {p["image_id"]: p for p in predictions}
        predictions = [pred_map.get(e, {"image_id": e, "prediction": 0, "confidence": 0.5}) for e in expected]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        w.writeheader(); w.writerows(predictions)
    print(f"Saved {OUTPUT_PATH} with {len(predictions)} predictions")

    # Report
    report = f"""# Confident Learning — Label Correction Results

## Method
1. 3-fold cross-validation on original folder labels (ResNet-18, 8 epochs/fold)
2. Out-of-fold probabilities for every training sample
3. CleanLab-style per-class confidence thresholding
4. Corrected labels where model disagrees with high confidence
5. Retrained on corrected labels (10 epochs, class-weighted + label smoothing)

## Cross-Validation
- OOF Accuracy on original labels: {100*cv_acc:.2f}%
- This is the upper bound of what the model can learn from folder labels.

## Label Correction
| Outcome | Count |
|---------|-------|
| Kept original label | {cl_stats['kept']} |
| Corrected (model confident) | {cl_stats['corrected']} |
| Flagged (low confidence) | {cl_stats['flagged']} |
| **Total** | {cl_stats['total']} |

## Final Results
| Setup | Validation Accuracy |
|-------|-------------------|
| Baseline (folder labels) | 49.43% |
| Confident Learning | **{acc_cl:.2f}%** |
| Improvement | **+{acc_cl - 49.43:.2f}%** |

## Interpretation
{'- Confident learning successfully identifies and corrects label noise.' if acc_cl > 49.43 else '- Label noise is too severe or model capacity is insufficient.'}
"""
    report_path = PROJECT_ROOT / "reports" / "confident_learning.md"
    report_path.write_text(report)
    print(f"\nSaved report to {report_path}")

if __name__ == "__main__":
    main()
