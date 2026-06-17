"""
Check the waste classification dataset for common data quality issues.
Saves findings to reports/findings.md with supporting evidence.
"""
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "train"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVIDENCE_DIR = REPORTS_DIR / "evidence"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

REPORTS_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR.mkdir(exist_ok=True)

def get_all_images():
    images = []
    for cls in CLASSES:
        for f in (DATA_DIR / cls).glob("*.*"):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                images.append({"path": f, "folder": cls})
    return images

def check_duplicates(images):
    print("  checking for exact duplicates...")
    hash_map = defaultdict(list)
    for img in images:
        h = hashlib.md5(img["path"].read_bytes()).hexdigest()
        hash_map[h].append(img)
    groups = [v for v in hash_map.values() if len(v) > 1]
    return groups

def check_near_duplicates(images):
    print("  checking for near-duplicates (phash)...")
    from PIL import Image as PILImage
    import imagehash

    phashes = {}
    for img in images:
        try:
            pil_img = PILImage.open(img["path"])
            ph = imagehash.phash(pil_img)
            phashes[img["path"].name] = (ph, img)
        except:
            pass

    names = list(phashes.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = phashes[names[i]][0] - phashes[names[j]][0]
            if d <= 8:
                pairs.append((phashes[names[i]][1], phashes[names[j]][1], d))
    return pairs

def check_mislabels(images):
    print("  checking filename vs folder mismatches...")
    mislabeled = []
    correct = []
    for img in images:
        true_label = img["path"].stem.split("_")[0]
        if true_label != img["folder"]:
            mislabeled.append(img)
        else:
            correct.append(img)
    return mislabeled, correct

def check_blurry(images, threshold=100.0):
    print("  checking for blurry images...")
    blurry = []
    for img in images:
        arr = cv2.imread(str(img["path"]), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        var = cv2.Laplacian(arr, cv2.CV_64F).var()
        if var < threshold:
            blurry.append((img, var))
    return blurry

def check_dims_and_corruption(images):
    print("  checking image dimensions and corruption...")
    issues = {"low_res": [], "corrupted": []}
    for img in images:
        try:
            with Image.open(img["path"]) as im:
                if min(im.size) < 100:
                    issues["low_res"].append(img)
        except:
            issues["corrupted"].append(img)
    return issues

def outlier_detection(images):
    print("  checking file size outliers...")
    sizes = []
    for img in images:
        sizes.append((img, img["path"].stat().st_size))
    vals = np.array([s[1] for s in sizes])
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = [s for s in sizes if s[1] < lower or s[1] > upper]
    return outliers, {
        "q1": int(q1), "q3": int(q3), "iqr": int(iqr),
        "min": int(vals.min()), "max": int(vals.max()),
    }

def save_evidence_mislabels(mislabeled, n=12):
    print(f"  saving {n} example mislabeled images...")
    import shutil
    examples = mislabeled[:n]
    for i, img in enumerate(examples):
        true_label = img["path"].stem.split("_")[0]
        name = f"mislabel_{i+1}_{img['folder']}_actually_{true_label}{img['path'].suffix}"
        shutil.copy2(str(img["path"]), str(EVIDENCE_DIR / name))

def save_evidence_blurry(blurry, n=12):
    print(f"  saving {n} blurriest images...")
    import shutil
    blurry_sorted = sorted(blurry, key=lambda x: x[1])
    for i, (img, var) in enumerate(blurry_sorted[:n]):
        name = f"blurry_{i+1}_var{var:.0f}_{img['folder']}_{img['path'].name}"
        shutil.copy2(str(img["path"]), str(EVIDENCE_DIR / name))

def write_report(stats):
    print("  writing findings report...")
    report = f"""# Waste Classification - Data Quality Findings

## Overview
Total training images: {stats['total']}
Classes: {', '.join(CLASSES)}

## 1. Class Distribution
| Class | Count |
|-------|-------|
"""
    for cls in CLASSES:
        report += f"| {cls} | {stats['class_counts'][cls]} |\n"
    report += f"\nImbalance ratio: {stats['imbalance_ratio']:.2f}x\n"

    report += f"""
## 2. Label Errors
Mislabeled images: {stats['mislabeled_count']} / {stats['total']} ({stats['mislabeled_pct']:.1f}%)
Correctly labeled: {stats['correct_count']}

The filename pattern `{{true_label}}_random_id.jpg` reveals the actual class.

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
Size range: {stats['size_stats']['min']} bytes - {stats['size_stats']['max']} bytes

## 6. Recommended Fixes
1. Relabel images using the prefix in each filename
2. Remove exact duplicate images (keep one copy per group)
3. Remove or flag blurry images for the organizers
4. Augment the trash class to balance the dataset
5. Use stratified sampling during training to handle imbalance
"""
    (REPORTS_DIR / "findings.md").write_text(report)
    print(f"  report saved to reports/findings.md")

def main():
    print("Inspecting dataset...\n")

    images = get_all_images()
    total = len(images)
    print(f"Total images found: {total}")

    class_counts = {c: len(list((DATA_DIR / c).glob("*.*"))) for c in CLASSES}
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
    print(f"  Mislabeled: {mislabeled_count} / {total} ({100*mislabeled_count/total:.1f}%)")
    print(f"  Duplicates: {len(dup_groups)} groups, {near_dup_pairs} near-duplicate pairs")
    print(f"  Blurry: {len(blurry)}")
    print(f"  Outliers: {len(outliers)}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    print(f"\nEvidence saved to {EVIDENCE_DIR}/")

if __name__ == "__main__":
    main()
