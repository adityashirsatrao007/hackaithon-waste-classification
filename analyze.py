"""
Data Detective: Comprehensive dataset audit for Hack[AI]Thon 2.0

Finds: mislabels, duplicates, blurry images, class imbalance, outliers, and more.
"""
import hashlib, json, warnings, sys
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageStat, UnidentifiedImageError
import numpy as np
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data" / "train"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NON_TRASH = {"cardboard", "glass", "metal", "paper", "plastic"}

results = {}

def get_images():
    images = []
    for cls in CLASSES:
        cls_dir = DATA_DIR / cls
        if cls_dir.exists():
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    images.append({"path": f, "assigned_label": cls})
    return images

images = get_images()
print(f"[INFO] Total training images: {len(images)}")

# ── 1. CLASS DISTRIBUTION ──
print("\n═══ 1. CLASS DISTRIBUTION ═══")
label_counts = Counter(i["assigned_label"] for i in images)
for cls in CLASSES:
    print(f"  {cls}: {label_counts[cls]}")
min_class = min(label_counts.values())
max_class = max(label_counts.values())
imbalance_ratio = max_class / min_class
results["class_distribution"] = dict(label_counts)
results["class_imbalance_ratio"] = round(imbalance_ratio, 2)
results["severity_class_imbalance"] = (
    "CRITICAL" if imbalance_ratio > 2 else "MODERATE" if imbalance_ratio > 1.5 else "MINOR"
)
print(f"  → Imbalance ratio (max/min): {imbalance_ratio:.2f}")
print(f"  → Severity: {results['severity_class_imbalance']}")

# ── 2. CORRUPTED / UNREADABLE IMAGES ──
print("\n═══ 2. CORRUPTED IMAGES ═══")
corrupted = []
for img in tqdm(images, desc="Checking integrity"):
    try:
        with Image.open(img["path"]) as im:
            im.load()
    except (UnidentifiedImageError, OSError):
        corrupted.append(img["path"])
results["corrupted_count"] = len(corrupted)
results["corrupted_files"] = [str(p) for p in corrupted]
print(f"  Corrupted: {len(corrupted)}")
for p in corrupted:
    print(f"    {p}")

# ── 3. MISLABELED IMAGES (detected via filename convention) ──
print("\n═══ 3. MISLABELED IMAGES ═══")
# Files named {true_label}_*.jpg placed in wrong folder
mislabels_by_folder = {}
total_mislabels = 0
for cls in CLASSES:
    cls_dir = DATA_DIR / cls
    if not cls_dir.exists():
        continue
    for f in cls_dir.iterdir():
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        true_label = f.stem.split("_")[0]
        if true_label in CLASSES and true_label != cls:
            total_mislabels += 1
            if cls not in mislabels_by_folder:
                mislabels_by_folder[cls] = []
            mislabels_by_folder[cls].append(f"{f.name} → should be '{true_label}' (currently in '{cls}')")

results["total_mislabels"] = total_mislabels
results["mislabels_by_folder"] = {k: len(v) for k, v in mislabels_by_folder.items()}
print(f"  Total mislabeled: {total_mislabels}")
for folder, items in mislabels_by_folder.items():
    print(f"  {folder}/: {len(items)} mislabeled")
    for x in items[:3]:
        print(f"    {x}")
    if len(items) > 3:
        print(f"    ... and {len(items)-3} more")

# ── 4. DUPLICATE IMAGES ──
print("\n═══ 4. DUPLICATE IMAGES ═══")
hash_map = defaultdict(list)
for img in tqdm(images, desc="Hashing images"):
    with open(img["path"], "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()
    hash_map[h].append(img)

duplicates_found = 0
dup_groups = []
for h, group in hash_map.items():
    if len(group) > 1:
        duplicates_found += len(group) - 1
        dup_groups.append(
            [{"path": str(g["path"]), "assigned_label": g["assigned_label"]} for g in group]
        )

results["duplicate_groups_count"] = len(dup_groups)
results["duplicate_images_count"] = duplicates_found
results["duplicate_groups"] = dup_groups
print(f"  Duplicate groups: {len(dup_groups)}")
for g in dup_groups:
    print(f"    Group: {[f['path'] for f in g]}")

# ── 5. BLURRY / LOW-QUALITY IMAGES ──
print("\n═══ 5. BLURRY IMAGES ═══")
BLUR_THRESHOLD = 100
blurry = []
sharpness_scores = []
for img in tqdm(images, desc="Computing blur"):
    try:
        arr = cv2.imread(str(img["path"]))
        if arr is None:
            continue
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_scores.append({"file": str(img["path"]), "score": round(score, 2), "label": img["assigned_label"]})
        if score < BLUR_THRESHOLD:
            blurry.append(img["path"])
    except Exception:
        pass

results["blurry_count"] = len(blurry)
results["blurry_files"] = [str(p) for p in blurry]
results["sharpness_stats"] = {
    "mean": round(float(np.mean([s["score"] for s in sharpness_scores])), 2),
    "min": round(float(np.min([s["score"] for s in sharpness_scores])), 2),
    "max": round(float(np.max([s["score"] for s in sharpness_scores])), 2),
}
print(f"  Blurry (Laplacian var < {BLUR_THRESHOLD}): {len(blurry)}")
for p in blurry[:5]:
    print(f"    {p}")
if len(blurry) > 5:
    print(f"    ... and {len(blurry)-5} more")

# ── 6. LOW-RESOLUTION IMAGES ──
print("\n═══ 6. LOW-RESOLUTION IMAGES ═══")
MIN_DIM = 100
low_res = []
for img in tqdm(images, desc="Checking resolution"):
    try:
        with Image.open(img["path"]) as im:
            w, h = im.size
            if w < MIN_DIM or h < MIN_DIM:
                low_res.append({"file": str(img["path"]), "size": f"{w}x{h}", "label": img["assigned_label"]})
    except Exception:
        pass

results["low_res_count"] = len(low_res)
results["low_res_files"] = low_res
print(f"  Low-res (dim < {MIN_DIM}px): {len(low_res)}")
for r in low_res[:5]:
    print(f"    {r['file']} ({r['size']})")

# ── 7. FILE SIZE ANOMALIES (outliers / near-empty images) ──
print("\n═══ 7. FILE SIZE ANOMALIES ═══")
file_sizes = []
for img in images:
    sz = img["path"].stat().st_size
    file_sizes.append({"file": str(img["path"]), "size_bytes": sz, "label": img["assigned_label"]})
sizes = np.array([f["size_bytes"] for f in file_sizes])
q1, q3 = np.percentile(sizes, 25), np.percentile(sizes, 75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers_fs = [f for f in file_sizes if f["size_bytes"] < lower or f["size_bytes"] > upper]
results["file_size_outliers_count"] = len(outliers_fs)
results["file_size_stats"] = {
    "min": int(sizes.min()), "max": int(sizes.max()),
    "mean": int(sizes.mean()), "median": int(np.median(sizes))
}
results["file_size_outliers"] = outliers_fs[:10]
print(f"  File size outliers: {len(outliers_fs)} (beyond 1.5×IQR)")
for o in outliers_fs[:5]:
    print(f"    {o['file']} → {o['size_bytes']} bytes")
if len(outliers_fs) > 5:
    print(f"    ... and {len(outliers_fs)-5} more")

# ── 8. NEAR-DUPLICATES (perceptual hashing) ──
print("\n═══ 8. NEAR-DUPLICATES (perceptual hash) ═══")
import imagehash
near_dups = []
for i in tqdm(range(len(images)), desc="Computing pHash"):
    try:
        with Image.open(images[i]["path"]) as im:
            phash_i = imagehash.phash(im)
    except Exception:
        continue
    for j in range(i + 1, len(images)):
        try:
            with Image.open(images[j]["path"]) as im:
                phash_j = imagehash.phash(im)
        except Exception:
            continue
        hamming = phash_i - phash_j
        if hamming < 10 and hamming > 0:
            near_dups.append({
                "a": str(images[i]["path"]), "b": str(images[j]["path"]),
                "hamming_distance": hamming,
                "label_a": images[i]["assigned_label"],
                "label_b": images[j]["assigned_label"]
            })
results["near_duplicate_count"] = len(near_dups)
results["near_duplicates"] = near_dups[:10]
print(f"  Near-duplicate pairs: {len(near_dups)}")
for nd in near_dups[:5]:
    print(f"    {nd['a']} ↔ {nd['b']} (hamming: {nd['hamming_distance']})")

# ── SUMMARY ──
print("\n" + "=" * 60)
print("  AUDIT SUMMARY")
print("=" * 60)
summary = {
    "total_images": len(images),
    "class_imbalance_ratio": results["class_imbalance_ratio"],
    "severity_class_imbalance": results["severity_class_imbalance"],
    "total_mislabels": total_mislabels,
    "duplicate_images": duplicates_found,
    "blurry_images": len(blurry),
    "low_resolution_images": len(low_res),
    "corrupted_images": len(corrupted),
    "file_size_outliers": len(outliers_fs),
    "near_duplicate_pairs": len(near_dups),
}
for k, v in summary.items():
    print(f"  {k}: {v}")

with open(Path(__file__).parent / "reports" / "audit_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\n[OK] Audit saved to reports/audit_results.json")
