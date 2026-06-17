from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import umap

from src.model import CLASSES, NUM_CLASSES


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_umap(embeddings, labels, title, save_path, legend=True):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    emb_2d = reducer.fit_transform(embeddings)
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                          c=label_ids, cmap="tab10", s=15, alpha=0.7)
    if legend:
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=le.classes_, title="Class", loc="best")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_confidence_hist(predictions, labels, save_path):
    preds = np.array(predictions)
    correct_mask = preds == np.array(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.hist(predictions, bins=20, alpha=0.7, color="steelblue")
    ax1.set_title("Confidence Distribution (All)")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Count")

    correct_confs = [p for p, c in zip(predictions, correct_mask) if c]
    wrong_confs = [p for p, c in zip(predictions, correct_mask) if not c]
    ax2.hist(correct_confs, bins=20, alpha=0.7, color="green", label="Correct")
    ax2.hist(wrong_confs, bins=20, alpha=0.7, color="red", label="Wrong")
    ax2.set_title("Confidence: Correct vs Wrong")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def generate_metrics_report(folder_ids, all_preds, save_path):
    report_dict = classification_report(
        folder_ids, all_preds, target_names=CLASSES,
        digits=3, output_dict=True)
    lines = ["### Per-Class Metrics\n"]
    lines.append("| Class | Precision | Recall | F1-Score | Support |")
    lines.append("|-------|-----------|--------|----------|---------|")
    for cls in CLASSES:
        m = report_dict[cls]
        lines.append(
            f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1-score']:.3f} | {m['support']} |")
    Path(save_path).write_text("\n".join(lines) + "\n")
    print(f"  saved metrics to {save_path}")
