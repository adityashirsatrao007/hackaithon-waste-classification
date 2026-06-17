"""
Visual analysis of the waste classification dataset and model performance.
Produces figures for the report: confusion matrix, per-class metrics,
embedding visualization, training curves, and confidence analysis.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import umap
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import WasteClassifier, CLASSES, NUM_CLASSES, IMG_SIZE

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DATA_DIR = PROJECT_ROOT / "data" / "train"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
        return img, label, str(path)

def load_all_data():
    samples = []
    for idx, cls in enumerate(CLASSES):
        for f in sorted((DATA_DIR / cls).glob("*.*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                true_label = f.stem.split("_")[0]
                samples.append({
                    "path": f, "folder_label": cls, "folder_idx": idx,
                    "true_label": true_label,
                })
    return samples

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  saved {filename}")

def plot_umap(embeddings, labels, title, filename, legend=True):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    emb_2d = reducer.fit_transform(embeddings)
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=label_ids,
                          cmap="tab10", s=15, alpha=0.7)
    if legend:
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=le.classes_, title="Class", loc="best")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  saved {filename}")

def plot_confidence_hist(predictions, labels, filename):
    preds = np.array(predictions)
    correct = preds == np.array(labels)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(predictions, bins=20, alpha=0.7, color="steelblue")
    plt.title("Confidence Distribution (All)")
    plt.xlabel("Confidence")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist([p for p, c in zip(predictions, correct) if c],
             bins=20, alpha=0.7, color="green", label="Correct")
    plt.hist([p for p, c in zip(predictions, correct) if not c],
             bins=20, alpha=0.7, color="red", label="Wrong")
    plt.title("Confidence: Correct vs Wrong")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  saved {filename}")

def main():
    print("Loading model...")
    state = torch.load(MODEL_PATH, map_location=device)
    model = WasteClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(state)
    model.eval()

    print("Loading data...")
    samples = load_all_data()
    print(f"  {len(samples)} images")

    ds = WasteDataset(
        [(s["path"], s["folder_idx"]) for s in samples],
        val_tfm,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_features = []
    all_folder_labels = []
    all_true_labels = []
    all_preds = []
    all_confs = []

    print("Extracting features and predictions...")
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            features = model.extract_features(images).cpu().numpy()
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1)
            confs, preds = probs.max(1)

            for i in range(len(labels)):
                fname = Path(paths[i]).stem
                true_label = fname.split("_")[0]
                all_features.append(features[i])
                all_folder_labels.append(CLASSES[labels[i].item()])
                all_true_labels.append(true_label)
                all_preds.append(preds[i].item())
                all_confs.append(confs[i].item())

    all_features = np.array(all_features)
    folder_ids = [CLASSES.index(l) for l in all_folder_labels]
    true_ids = [CLASSES.index(l) if l in CLASSES else -1 for l in all_true_labels]

    # -- Confusion matrix (folder labels vs predictions) --
    print("\n--- Confusion Matrix ---")
    plot_confusion_matrix(folder_ids, all_preds,
                          "Confusion Matrix (Folder Labels)", "confusion_matrix.png")

    # -- Per-class metrics --
    print("\n--- Per-Class Metrics ---")
    report = classification_report(
        folder_ids, all_preds,
        target_names=CLASSES, digits=3, output_dict=False,
    )
    print(report)
    report_dict = classification_report(
        folder_ids, all_preds,
        target_names=CLASSES, digits=3, output_dict=True,
    )
    lines = ["### Per-Class Metrics\n"]
    lines.append("| Class | Precision | Recall | F1-Score | Support |")
    lines.append("|-------|-----------|--------|----------|---------|")
    for cls in CLASSES:
        m = report_dict[cls]
        lines.append(f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {m['support']} |")
    (REPORTS_DIR := PROJECT_ROOT / "reports" / "metrics.md").write_text("\n".join(lines) + "\n")
    print(f"  saved metrics to reports/metrics.md")

    # -- UMAP by folder label --
    print("\n--- UMAP Embeddings ---")
    n_plot = min(500, len(all_features))
    idxs = np.random.RandomState(42).choice(len(all_features), n_plot, replace=False)
    plot_umap(all_features[idxs], [all_folder_labels[i] for i in idxs],
              "Embeddings colored by Folder Label", "umap_by_folder.png")

    # -- UMAP by true label (filename prefix) --
    valid = [i for i in idxs if all_true_labels[i] in CLASSES]
    if valid:
        plot_umap(all_features[valid],
                  [all_true_labels[i] for i in valid],
                  "Embeddings colored by True Label (Filename Prefix)",
                  "umap_by_truelabel.png")

    # -- Confidence histogram --
    print("\n--- Confidence Analysis ---")
    plot_confidence_hist(all_confs, all_preds, "confidence_hist.png")

    # -- Per-class accuracy --
    print("\n--- Per-Class Accuracy ---")
    plt.figure(figsize=(10, 5))
    accuracies = []
    for i, cls in enumerate(CLASSES):
        cls_mask = np.array(folder_ids) == i
        if cls_mask.sum() > 0:
            acc = 100 * (np.array(all_preds)[cls_mask] == i).mean()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    colors = ["#e74c3c" if a < 30 else ("#f39c12" if a < 60 else "#2ecc71")
              for a in accuracies]
    plt.bar(CLASSES, accuracies, color=colors)
    plt.axhline(100 / NUM_CLASSES, color="gray", linestyle="--",
                label=f"Random ({100/NUM_CLASSES:.1f}%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy on Folder Labels")
    plt.legend()
    for i, a in enumerate(accuracies):
        plt.text(i, a + 1, f"{a:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_class_accuracy.png", dpi=150)
    plt.close()
    print("  saved per_class_accuracy.png")

    print(f"\nAll figures saved to {FIGURES_DIR}/")

if __name__ == "__main__":
    main()
