"""
Generate audit report with visualizations for Hack[AI]Thon 2.0.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

REPORTS_DIR = Path(__file__).parent / "reports"
DATA_DIR = Path(__file__).parent / "data" / "train"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
REPORTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

def plot_class_distribution():
    distribution = {c: len(list((DATA_DIR / c).glob("*.*"))) for c in CLASS_NAMES}
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#8B4513", "#2E8B57", "#708090", "#4169E1", "#DAA520", "#DC143C"]
    bars = ax.bar(distribution.keys(), distribution.values(), color=colors)
    ax.set_title("Class Distribution (Before Fixing)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    for bar, val in zip(bars, distribution.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
                ha="center", fontweight="bold")
    max_count = max(distribution.values())
    min_count = min(distribution.values())
    ax.text(0.5, -0.15, f"Imbalance Ratio: {max_count/min_count:.2f}x  (trash={min_count} vs paper={max_count})",
            transform=ax.transAxes, ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="salmon", alpha=0.3))
    fig.tight_layout()
    path = REPORTS_DIR / "class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path

def plot_mislabels():
    mislabels = {
        "cardboard": 58, "glass": 53, "metal": 59,
        "paper": 70, "plastic": 50, "trash": 28
    }
    total = {c: len(list((DATA_DIR / c).glob("*.*"))) for c in CLASS_NAMES}
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    totals = [total[c] for c in CLASS_NAMES]
    mis = [mislabels[c] for c in CLASS_NAMES]
    clean = [totals[i] - mis[i] for i in range(len(CLASS_NAMES))]
    ax.bar(x - width/2, clean, width, label="Correctly Labeled", color="green", alpha=0.7)
    ax.bar(x + width/2, mis, width, label="Mislabeled", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Count")
    ax.set_title("Mislabels by Class (71.5% of All Images Mislabeled)", fontsize=14, fontweight="bold")
    ax.legend()
    for i, (t, m) in enumerate(zip(totals, mis)):
        ax.text(i, max(t, m) + 2, f"{m}/{t}", ha="center", fontweight="bold", fontsize=10)
    fig.tight_layout()
    path = REPORTS_DIR / "mislabels.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path

def plot_mislabelled_examples():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    examples = [
        ("cardboard", "cardboard_2ojz4rhf.jpg"),
        ("glass", "glass_eevhunry.jpg"),
        ("metal", "paper_nxksdm5h.jpg"),
        ("paper", "paper_62ewfqzd.jpg"),
        ("plastic", "plastic_zsq20qfy.jpg"),
        ("trash", "metal_odqgqtsl.jpg"),
    ]
    for ax, (folder, filename) in zip(axes.flat, examples):
        path = DATA_DIR / folder / filename
        if path.exists():
            img = Image.open(path)
            ax.imshow(img)
        # Determine true label from filename
        true_label = filename.split("_")[0]
        folder_label = folder
        color = "red" if true_label != folder_label else "green"
        ax.set_title(f"Folder: {folder_label}\nFilename says: {true_label}", fontsize=10, color=color)
        ax.axis("off")
    fig.suptitle("Examples of Mislabeled Images (Filename Reveals True Label)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = REPORTS_DIR / "mislabelled_examples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path

def plot_blurry_examples():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    blurry_images = []
    for c in CLASS_NAMES:
        for img_path in (DATA_DIR / c).glob("*.*"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            if laplacian_var < 100:
                blurry_images.append((img_path, laplacian_var))
    blurry_images.sort(key=lambda x: x[1])
    for ax, (path, var) in zip(axes.flat, blurry_images[:6]):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(f"Var={var:.1f}  {path.parent.name}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Blurriest Images (Laplacian Variance < 100)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = REPORTS_DIR / "blurry_examples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path

def create_html_report(dist_path, mis_path, ex_path, blur_path):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Waste Classification — Data Audit Report</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 1rem; background: #f8f9fa; color: #333; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 8px; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.section {{ background: white; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }}
.badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 14px; }}
.critical {{ background: #ff4444; color: white; }}
.moderate {{ background: #ffa726; color: white; }}
.minor {{ background: #66bb6a; color: white; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #1a1a2e; color: white; }}
tr:hover {{ background: #f1f1f1; }}
</style>
</head>
<body>
<h1>🗑️ Waste Classification — Data-Centric Audit Report</h1>
<p><strong>Hack[AI]Thon 2.0 2026</strong> | Baseline: ResNet-18 (random init) | Validation Accuracy: <strong>46.07%</strong></p>

<div class="section">
<h2>🔍 Bug Summary</h2>
<table>
<tr><th>Issue</th><th>Count</th><th>Severity</th></tr>
<tr><td>Class imbalance (trash=40, paper=86, ratio 2.15×)</td><td>40 vs 86</td><td><span class="badge critical">CRITICAL</span></td></tr>
<tr><td>Mislabeled images (filename vs folder mismatch)</td><td>318 / 445 (71.5%)</td><td><span class="badge critical">CRITICAL</span></td></tr>
<tr><td>Blurry images (Laplacian var < 100)</td><td>65</td><td><span class="badge moderate">MODERATE</span></td></tr>
<tr><td>Exact duplicate images (same MD5)</td><td>25 groups</td><td><span class="badge moderate">MODERATE</span></td></tr>
<tr><td>File size outliers</td><td>21</td><td><span class="badge minor">MINOR</span></td></tr>
<tr><td>Corrupted images</td><td>0</td><td><span class="badge minor">NONE</span></td></tr>
<tr><td>Low resolution (< 100px)</td><td>0</td><td><span class="badge minor">NONE</span></td></tr>
</table>
</div>

<div class="section">
<h2>📊 Class Distribution</h2>
<p><strong>Imbalance ratio: 2.15×</strong> — trash class has only 40 images vs paper with 86.</p>
<img src="{dist_path.name}" alt="Class Distribution">
</div>

<div class="section">
<h2>🏷️ Mislabel Analysis</h2>
<p><strong>71.5% of training images are mislabeled.</strong> The filename convention <code>{{true_label}}_{{random_id}}.jpg</code> reveals the correct ground truth. The 3LC starter kit deliberately placed images in wrong folders to test data-centric AI skills.</p>
<img src="{mis_path.name}" alt="Mislabels">
<img src="{ex_path.name}" alt="Mislabeled Examples">
</div>

<div class="section">
<h2>📸 Blurry Images</h2>
<p><strong>65 images</strong> have Laplacian variance below 100, indicating significant blur.</p>
<img src="{blur_path.name}" alt="Blurry Examples">
</div>

<div class="section">
<h2>🔗 Duplicates</h2>
<p><strong>25 duplicate groups</strong> found via MD5 hash — exact same pixels, different filenames, sometimes different labels.</p>
</div>

<div class="section">
<h2>📈 Dataset Claims vs Reality</h2>
<table>
<tr><th>Claim</th><th>Website</th><th>Actual</th></tr>
<tr><td>Training images</td><td>500 labeled + 100 unlabeled</td><td>445 labeled + 115 test (560 total)</td></tr>
<tr><td>Correctly labeled</td><td>—</td><td>127 / 445 (28.5%)</td></tr>
<tr><td>Balanced classes</td><td>Yes</td><td>No (ratio 2.15×)</td></tr>
<tr><td>No duplicates</td><td>—</td><td>25 duplicate groups</td></tr>
</table>
</div>

<div class="section">
<h2>✅ Recommended Fixes</h2>
<ol>
<li><strong>Relabel by filename</strong> — use <code>{{true_label}}</code> prefix from the filename to assign correct class</li>
<li><strong>Remove duplicates</strong> — deduplicate by MD5 hash, keep one copy per group</li>
<li><strong>Remove or sharpen blurry</strong> — discard images with Laplacian var < 100 or apply sharpening</li>
<li><strong>Augment trash class</strong> — oversample or apply aggressive augmentation to balance classes</li>
<li><strong>Cross-validation</strong> — use stratified k-fold to mitigate imbalance during training</li>
</ol>
</div>
</body>
</html>"""
    report_path = REPORTS_DIR / "audit_report.html"
    report_path.write_text(html)
    print(f"  [OK] {report_path}")
    return report_path

def main():
    print("Generating audit report...")
    print("\n[1/5] Class distribution chart...")
    dist_path = plot_class_distribution()
    print("[2/5] Mislabel chart...")
    mis_path = plot_mislabels()
    print("[3/5] Mislabeled examples...")
    ex_path = plot_mislabelled_examples()
    print("[4/5] Blurry examples...")
    blur_path = plot_blurry_examples()
    print("[5/5] HTML report...")
    html_path = create_html_report(dist_path, mis_path, ex_path, blur_path)
    print(f"\nDone. Report: {html_path}")

if __name__ == "__main__":
    main()
