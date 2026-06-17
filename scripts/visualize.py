import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model import WasteClassifier, CLASSES, NUM_CLASSES
from src.dataset import WasteDataset, val_tfm
from src.visualization import (
    plot_confusion_matrix, plot_umap, plot_confidence_hist,
    generate_metrics_report,
)

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DATA_DIR = PROJECT_ROOT / "data" / "train"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading model...")
    state = torch.load(MODEL_PATH, map_location=device)
    model = WasteClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(state)
    model.eval()

    print("Loading data...")
    samples = []
    for idx, cls in enumerate(CLASSES):
        for f in sorted((Path(DATA_DIR) / cls).glob("*.*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append({"path": f, "folder_label": cls, "folder_idx": idx})

    ds = WasteDataset([(s["path"], s["folder_idx"]) for s in samples], val_tfm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_features = []
    all_folder_labels = []
    all_preds = []
    all_confs = []

    print("Extracting features and predictions...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.extract_features(images).cpu().numpy()
            logits = model(images)
            probs = torch.nn.functional.softmax(logits, dim=1)
            confs, preds = probs.max(1)

            for i in range(len(labels)):
                all_features.append(features[i])
                all_folder_labels.append(CLASSES[labels[i].item()])
                all_preds.append(preds[i].item())
                all_confs.append(confs[i].item())

    all_features = np.array(all_features)
    folder_ids = [CLASSES.index(l) for l in all_folder_labels]

    print("\n--- Confusion Matrix ---")
    plot_confusion_matrix(folder_ids, all_preds,
                          "Confusion Matrix (Folder Labels)",
                          FIGURES_DIR / "confusion_matrix.png")

    print("\n--- Per-Class Metrics ---")
    generate_metrics_report(folder_ids, all_preds,
                            PROJECT_ROOT / "reports" / "metrics.md")

    print("\n--- UMAP Embeddings ---")
    n_plot = min(500, len(all_features))
    idxs = np.random.RandomState(42).choice(len(all_features), n_plot, replace=False)
    plot_umap(all_features[idxs], [all_folder_labels[i] for i in idxs],
              "Embeddings colored by Folder Label",
              FIGURES_DIR / "umap_by_folder.png")

    print("\n--- Confidence Histogram ---")
    plot_confidence_hist(all_confs, all_preds,
                         FIGURES_DIR / "confidence_hist.png")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    main()
