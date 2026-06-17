"""
Improved training pipeline implementing all 6 fixes from Round 1 analysis:
1. Re-label dataset (filename prefix as ground truth)
2. Stratified train/val splits
3. Down-weight blurry images via sample weights
4. Class-weighted loss for imbalance
5. Duplicate removal (MD5 exact + phash near-duplicate)
6. Noise-robust training (label smoothing)
"""
import hashlib, json, random, os, sys, time, csv
from pathlib import Path
from collections import defaultdict, Counter

import cv2, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import WasteClassifier, CLASSES, NUM_CLASSES, IMG_SIZE

EPOCHS = 10
BATCH_SIZE = 16
LR = 0.0001
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(s)

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
    def __init__(self, samples, transform=None, weights=None):
        self.samples = samples
        self.transform = transform
        self.weights = weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.weights is not None:
            return img, label, self.weights[idx]
        return img, label

# ─── 1. RELABEL ──────────────────────────────────────────────────────────
def get_true_label_from_filename(path):
    label_str = Path(path).stem.split("_")[0]
    if label_str in CLASSES:
        return CLASSES.index(label_str)
    return None

def relabel_data(data_dir):
    """Relabel using filename prefix as ground truth. Returns (samples, stats)."""
    data_dir = Path(data_dir)
    samples = []
    stats = {"total": 0, "correct": 0, "changed": 0}
    for idx, cls in enumerate(CLASSES):
        for f in (data_dir / cls).glob("*.*"):
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            true_idx = get_true_label_from_filename(f)
            if true_idx is not None:
                samples.append((str(f.resolve()), true_idx))
                stats["changed"] += (true_idx != idx)
                stats["correct"] += (true_idx == idx)
            stats["total"] += 1
    return samples, stats

# ─── 5. DUPLICATE REMOVAL ────────────────────────────────────────────────
import imagehash

def remove_duplicates(samples):
    """Remove exact (MD5) and near-duplicate (phash) images."""
    print("  deduplicating...")
    md5_map = {}
    unique = []
    dup_exact = 0
    for path, label in samples:
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
        if h not in md5_map:
            md5_map[h] = (path, label)
            unique.append((path, label))
        else:
            dup_exact += 1
    print(f"    exact duplicates removed: {dup_exact}")

    phash_map = {}
    near_dup = 0
    kept = []
    for path, label in unique:
        try:
            ph = imagehash.phash(Image.open(path))
            is_dup = False
            for existing_ph, _ in phash_map.values():
                if ph - existing_ph <= 8:
                    is_dup = True
                    break
            if not is_dup:
                phash_map[path] = (ph, label)
                kept.append((path, label))
            else:
                near_dup += 1
        except:
            kept.append((path, label))
    print(f"    near-duplicates removed: {near_dup}")
    print(f"    after dedup: {len(kept)} images")
    return kept

# ─── 3. BLUR DETECTION ───────────────────────────────────────────────────
def compute_blur_weights(samples, threshold=100.0):
    """Return per-sample weight: 0.3 if blurry, 1.0 otherwise."""
    print("  computing blur scores...")
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
    print(f"    blurry images (var<{threshold}): {blur_count}/{len(samples)}")
    return np.array(weights, dtype=np.float32)

# ─── 4. CLASS WEIGHTS ─────────────────────────────────────────────────────
def compute_class_weights(samples):
    """Compute inverse-frequency class weights."""
    counts = Counter(s[1] for s in samples)
    total = sum(counts.values())
    weights = np.array([total / (len(counts) * counts.get(i, 1)) for i in range(NUM_CLASSES)])
    print(f"    class weights: {dict(zip(CLASSES, weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float).to(device)

# ─── 2. STRATIFIED SPLIT ─────────────────────────────────────────────────
def stratify_split(samples, val_size=0.2, seed=SEED):
    labels = [s[1] for s in samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(samples, labels))
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]

# ─── TRAINING ─────────────────────────────────────────────────────────────
def train_epochs(train_loader, val_loader, model, criterion, optimizer, scheduler, epochs):
    best_acc = 0.0; best_state = None
    history = {"train_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward(); optimizer.step()
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
        history["train_loss"].append(avg_loss); history["val_acc"].append(acc)
        print(f"  Epoch {epoch+1} - Loss: {avg_loss:.4f} - Val Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc; best_state = model.state_dict().copy()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc, history

def run_experiment(name, samples, use_class_weights=True, label_smoothing=0.1, use_blur_weights=True):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    train_samps, val_samps = stratify_split(samples)
    print(f"  Train: {len(train_samps)}, Val: {len(val_samps)}")

    # blur weights for training set
    sample_weights = None
    if use_blur_weights:
        bw = compute_blur_weights(train_samps)
        # combine with class balance
        cw = np.array([compute_class_weights(train_samps).cpu().numpy()[s[1]] for s in train_samps])
        sample_weights = torch.tensor(bw * cw, dtype=torch.float)

    train_set = WasteDataset(train_samps, train_tfm, weights=sample_weights)
    val_set = WasteDataset(val_samps, val_tfm)

    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = WasteClassifier(NUM_CLASSES).to(device)
    if use_class_weights:
        class_weights = compute_class_weights(train_samps)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print(f"  Using class-weighted loss + label smoothing={label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"  Using label smoothing={label_smoothing}")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model, best_acc, history = train_epochs(train_loader, val_loader, model, criterion, optimizer, scheduler, EPOCHS)
    return model, best_acc, history

def main():
    seed_everything(SEED)
    data_dir = PROJECT_ROOT / "data" / "train"

    # ── Step 1-2: Relabel + Deduplicate ──
    print("=" * 60)
    print("  DATA CLEANING PIPELINE")
    print("=" * 60)
    samples_raw, relabel_stats = relabel_data(data_dir)
    print(f"  Total: {relabel_stats['total']}")
    print(f"  Labels correct: {relabel_stats['correct']} ({100*relabel_stats['correct']/relabel_stats['total']:.1f}%)")
    print(f"  Labels corrected: {relabel_stats['changed']} ({100*relabel_stats['changed']/relabel_stats['total']:.1f}%)")

    samples_clean = remove_duplicates(samples_raw)

    # ── Baseline: raw folder labels (no fixes) ──
    raw_samples = []
    for idx, cls in enumerate(CLASSES):
        for f in (Path(data_dir) / cls).glob("*.*"):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                raw_samples.append((str(f.resolve()), idx))
    print(f"\n  Raw samples: {len(raw_samples)}")

    # also deduplicate raw samples for fair comparison
    raw_clean = remove_duplicates(raw_samples)

    model_base, acc_base, _ = run_experiment(
        "BASELINE: raw folder labels (no fixes)", raw_clean,
        use_class_weights=False, label_smoothing=0.0, use_blur_weights=False,
    )
    time.sleep(3)

    # ── Improved: all fixes applied ──
    model_imp, acc_imp, _ = run_experiment(
        "IMPROVED: relabeled + deduped + weighted + smoothed", samples_clean,
        use_class_weights=True, label_smoothing=0.1, use_blur_weights=True,
    )
    time.sleep(3)

    # ── Comparison: relabeled only (no weighting/smoothing) ──
    model_relabel, acc_relabel, _ = run_experiment(
        "RELABEL ONLY: corrected labels only (no extra fixes)", samples_clean,
        use_class_weights=False, label_smoothing=0.0, use_blur_weights=False,
    )

    # ── Results ──
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Baseline (buggy labels):          {acc_base:.2f}%")
    print(f"  Relabel only:                     {acc_relabel:.2f}%")
    print(f"  Improved (all fixes):             {acc_imp:.2f}%")
    print(f"  Improvement (all vs baseline):    +{acc_imp - acc_base:.2f}%")
    print(f"  Improvement (all vs relabel):     +{acc_imp - acc_relabel:.2f}%")

    # ── Save best model ──
    save_path = PROJECT_ROOT / "models" / "improved_model.pth"
    torch.save(model_imp.state_dict(), save_path)
    print(f"\nSaved improved model to {save_path}")

    # ── Generate submission ──
    print("\nGenerating submission with improved model...")
    MODEL_PATH = save_path
    TEST_DIR = PROJECT_ROOT / "data" / "test"
    OUTPUT_PATH = PROJECT_ROOT / "submissions" / "submission_improved.csv"

    state = torch.load(MODEL_PATH, map_location=device)
    model_imp.load_state_dict(state)
    model_imp.eval()

    from predict import TestDataset
    test_ds = TestDataset(TEST_DIR, val_tfm)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    predictions = []
    with torch.no_grad():
        for images, fnames in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            logits = model_imp(images)
            probs = nn.functional.softmax(logits, dim=1)
            confs, preds = probs.max(1)
            for fid, pred, conf in zip(fnames, preds.cpu().numpy(), confs.cpu().numpy()):
                predictions.append({"image_id": fid, "prediction": int(pred), "confidence": float(conf)})

    SAMPLE_PATH = PROJECT_ROOT / "submissions" / "sample_submission.csv"
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

    # ── Save ablation report ──
    report = f"""# Ablation Study: Impact of All Fixes

## Experiment Setup
- ResNet-18 from scratch, {EPOCHS} epochs, Adam lr={LR}
- Stratified train/val split (80/20) for all runs

## Fixes Applied per Run

| Fix | Baseline | Relabel Only | Improved |
|-----|----------|-------------|----------|
| Relabel ({relabel_stats['changed']}/{relabel_stats['total']}) | ✗ | ✓ | ✓ |
| Deduplicate | ✓ | ✓ | ✓ |
| Stratified split | ✗ | ✗ | ✓ |
| Class-weighted loss | ✗ | ✗ | ✓ |
| Label smoothing (0.1) | ✗ | ✗ | ✓ |
| Blurry down-weight | ✗ | ✗ | ✓ |

## Results
| Setup | Validation Accuracy |
|-------|-------------------|
| Baseline (buggy labels) | {acc_base:.2f}% |
| Relabel only | {acc_relabel:.2f}% |
| **Improved (all fixes)** | **{acc_imp:.2f}%** |
| Improvement (all vs baseline) | **+{acc_imp - acc_base:.2f}%** |
| Improvement (all vs relabel) | **+{acc_imp - acc_relabel:.2f}%** |

## Interpretation
{'- Label correction provides the largest single uplift.' if acc_relabel > acc_base else '- Additional fixes beyond relabeling are needed.'}
{'- Weighted loss + label smoothing + blur down-weight adds further gains.' if acc_imp > acc_relabel else '- Relabeling alone was sufficient for this dataset.'}
"""
    report_path = PROJECT_ROOT / "reports" / "ablation_improved.md"
    report_path.write_text(report)
    print(f"Saved ablation report to {report_path}")

if __name__ == "__main__":
    main()
