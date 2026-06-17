import json
import shutil
from pathlib import Path

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import CLASSES
from src.analysis import (
    get_all_images, check_mislabels, check_duplicates,
    check_near_duplicates, check_blurry, check_dims_and_corruption,
    outlier_detection,
)

DATA_DIR = PROJECT_ROOT / "data" / "train"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVIDENCE_DIR = REPORTS_DIR / "evidence"
REPORTS_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR.mkdir(exist_ok=True)


def save_evidence_mislabels(mislabeled, n=12):
    examples = mislabeled[:n]
    for i, img in enumerate(examples):
        true_label = img["path"].stem.split("_")[0]
        name = f"mislabel_{i+1}_{img['folder']}_actually_{true_label}{img['path'].suffix}"
        shutil.copy2(str(img["path"]), str(EVIDENCE_DIR / name))


def save_evidence_blurry(blurry, n=12):
    blurry_sorted = sorted(blurry, key=lambda x: x[1])
    for i, (img, var) in enumerate(blurry_sorted[:n]):
        name = f"blurry_{i+1}_var{var:.0f}_{img['folder']}_{img['path'].name}"
        shutil.copy2(str(img["path"]), str(EVIDENCE_DIR / name))


def write_report(stats):
    class_rows = "\n".join(f"| {c} | {stats['class_counts'][c]} |" for c in CLASSES)
    report = f"""# Waste Classification — Data Quality Findings

## Overview
Total training images: {stats['total']}
Classes: {', '.join(CLASSES)}

## 1. Class Distribution
| Class | Count |
|-------|-------|
{class_rows}

Imbalance ratio: {stats['imbalance_ratio']:.2f}x

## 2. Label Errors
Mislabeled images: {stats['mislabeled_count']} / {stats['total']} ({stats['mislabeled_pct']:.1f}%)
Correctly labeled: {stats['correct_count']}

## 3. Duplicate Images
Exact duplicate groups found: {stats['dup_groups']}
Near-duplicate pairs found: {stats['near_dup_pairs']}

## 4. Image Quality
Blurry images (Laplacian variance < 100): {stats['blurry_count']}
- Min variance: {stats['blurry_stats']['min_var']:.1f}
- Mean variance: {stats['blurry_stats']['mean_var']:.1f}

Corrupted images: {stats['corrupted']}
Low resolution images (< 100px): {stats['low_res']}

## 5. Outliers
File size outliers: {stats['outlier_count']}
Size range: {stats['size_stats']['min']} bytes — {stats['size_stats']['max']} bytes

## 6. Recommended Fixes
1. Relabel images using the prefix in each filename
2. Remove exact duplicate images (keep one copy per group)
3. Remove or flag blurry images
4. Augment the trash class to balance the dataset
5. Use stratified sampling to handle imbalance
"""
    (REPORTS_DIR / "index.md").write_text(report)
    print(f"  report saved to reports/index.md")


def main():
    print("Inspecting dataset...\n")
    images = get_all_images(DATA_DIR)
    total = len(images)
    print(f"Total images found: {total}")

    class_counts = {c: len(list((Path(DATA_DIR) / c).glob("*.*"))) for c in CLASSES}
    imbalance_ratio = max(class_counts.values()) / max(min(class_counts.values()), 1)

    mislabeled, correct = check_mislabels(images)
    mislabeled_count = len(mislabeled)
    correct_count = len(correct)
    dup_groups = check_duplicates(images)
    near_dup_pairs = check_near_duplicates(images)
    blurry = check_blurry(images)
    blurry_vals = [b[1] for b in blurry]
    size_issues = check_dims_and_corruption(images)
    outliers, size_stats = outlier_detection(images)

    print("\nSaving evidence...")
    save_evidence_mislabels(mislabeled)
    save_evidence_blurry(blurry)

    stats = {
        "total": total,
        "class_counts": class_counts,
        "imbalance_ratio": imbalance_ratio,
        "mislabeled_count": mislabeled_count,
        "mislabeled_pct": 100 * mislabeled_count / total if total else 0,
        "correct_count": correct_count,
        "dup_groups": len(dup_groups),
        "near_dup_pairs": len(near_dup_pairs),
        "blurry_count": len(blurry),
        "blurry_stats": {
            "min_var": min(blurry_vals) if blurry_vals else 0,
            "mean_var": np.mean(blurry_vals) if blurry_vals else 0,
        },
        "corrupted": len(size_issues["corrupted"]),
        "low_res": len(size_issues["low_res"]),
        "outlier_count": len(outliers),
        "size_stats": size_stats,
    }

    json.dump(stats, (REPORTS_DIR / "findings.json").open("w"), indent=2)
    write_report(stats)

    print("\n--- Summary ---")
    print(f"  Total: {total}")
    print(f"  Mislabeled: {mislabeled_count}/{total} ({100*mislabeled_count/total:.1f}%)")
    print(f"  Duplicates: {len(dup_groups)} groups, {near_dup_pairs} near-pairs")
    print(f"  Blurry: {len(blurry)}  Outliers: {len(outliers)}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")


if __name__ == "__main__":
    main()
