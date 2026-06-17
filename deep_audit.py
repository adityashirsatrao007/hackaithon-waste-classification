"""Deep forensic audit of waste-classification dataset — find hidden patterns."""

import os, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data/train")
TEST = Path("data/test")
CLASSES = sorted(os.listdir(DATA))

# ─── 1. Parse every image: folder label vs filename label ───
rows = []
for cls in CLASSES:
    for fname in os.listdir(DATA / cls):
        if not fname.endswith(".jpg"): continue
        prefix = fname.split("_")[0]
        hash_part = fname.split("_", 1)[1].replace(".jpg", "")
        rows.append({"file": f"{cls}/{fname}", "folder": cls, "prefix": prefix, "hash": hash_part, "match": cls == prefix})

total = len(rows)
mismatch = [r for r in rows if not r["match"]]
match_count = sum(1 for r in rows if r["match"])

print(f"Total images: {total}")
print(f"Folder matches prefix: {match_count} ({100*match_count/total:.1f}%)")
print(f"Folder ≠ prefix: {len(mismatch)} ({100*len(mismatch)/total:.1f}%)")
print()

# ─── 2. Confusion matrix: folder label × prefix label ───
conf = defaultdict(Counter)
for r in rows:
    conf[r["folder"]][r["prefix"]] += 1

print("Confusion: folder label → prefix (filename) label")
print("=" * 60)
print(f"{'Folder↓':>12}", " ".join(f"{c:>8}" for c in CLASSES))
for f in CLASSES:
    vals = [conf[f].get(c, 0) for c in CLASSES]
    total_f = sum(vals)
    pcts = [f"{v:>3d}({100*v/total_f:>4.1f}%)" for v in vals]
    print(f"{f:>12}", " ".join(pcts))

# ─── 3. Class imbalance ───
print()
print("Class distribution (by folder):")
for cls in CLASSES:
    n = sum(1 for r in rows if r["folder"] == cls)
    print(f"  {cls}: {n} ({100*n/total:.1f}%)")

# ─── 4. Train/test duplicate detection ───
print()
print("=" * 60)
print("TRAIN/TEST LEAKAGE DETECTION")

test_hashes = {f.replace("test_", "").replace(".jpg", ""): f for f in os.listdir(TEST)}
train_hashes = {r["hash"]: r["file"] for r in rows}

# Check exact hash matches
overlap = set(test_hashes) & set(train_hashes)
if overlap:
    print(f"  ⚠ Found {len(overlap)} exact hash duplicates between train and test!")
    for h in overlap:
        print(f"    Train: {train_hashes[h]}  ↔  Test: {test_hashes[h]}")
else:
    print("  ✓ No exact hash duplicates between train and test")

# Check perceptual hashing (image similarity)
from PIL import Image
import imagehash

def phash_img(path):
    try:
        return str(imagehash.phash(Image.open(path)))
    except:
        return None

train_phashes = {}
for r in rows:
    ph = phash_img(DATA / r["file"])
    if ph: train_phashes[ph] = r["file"]

test_phashes = {}
for f in os.listdir(TEST):
    ph = phash_img(TEST / f)
    if ph: test_phashes[ph] = f

overlap_ph = set(train_phashes) & set(test_phashes)
if overlap_ph:
    print(f"  ⚠ Found {len(overlap_ph)} perceptual hash duplicates between train and test!")
else:
    print("  ✓ No perceptual hash duplicates found")

# Near-duplicates: hamming distance <= 4
close_pairs = []
for t_ph, t_file in test_phashes.items():
    for tr_ph, tr_file in train_phashes.items():
        hd = bin(int(t_ph, 16) ^ int(tr_ph, 16)).count("1")
        if 0 < hd <= 4:
            close_pairs.append((t_file, tr_file, hd))

if close_pairs:
    print(f"  ⚠ Found {len(close_pairs)} near-duplicate pairs (hamming dist ≤ 4)")
    for tf, trf, hd in close_pairs[:10]:
        print(f"    Test: {tf} ↔ Train: {trf} (dist={hd})")
else:
    print("  ✓ No near-duplicate pairs found")

# ─── 5. Systematic mislabeling patterns ───
print()
print("=" * 60)
print("SYSTEMATIC MISLABEL PATTERNS")

# Check if certain source → target mislabel pairs are over-represented
mismatch_pairs = Counter()
for r in rows:
    if r["folder"] != r["prefix"]:
        mismatch_pairs[(r["folder"], r["prefix"])] += 1

print("Top mislabel paths:")
for (folder, prefix), count in mismatch_pairs.most_common(10):
    pct = 100 * count / total
    print(f"  {folder} → {prefix}: {count} ({pct:.1f}% of all data)")

# ─── 6. Check if mislabeling is random or has structure ───
print()
print("=" * 60)
print("MISLABEL RANDOMNESS TEST")

# H0: each class has equal probability of receiving a wrong prefix
# If some classes are disproportionately used as wrong prefixes, that's systematic
prefixes_in_mismatches = Counter()
for r in mismatch:
    prefixes_in_mismatches[r["prefix"]] += 1

total_mismatch = len(mismatch)
print(f"Wrong-label prefix distribution ({total_mismatch} total mismatches):")
for p in CLASSES:
    count = prefixes_in_mismatches.get(p, 0)
    print(f"  Mislabelled as {p}: {count} ({100*count/total_mismatch:.1f}% of mismatches, {100*count/total:.1f}% of all data)")

# ─── 7. Check for dataset contamination from other source ───
print()
print("=" * 60)
print("FILENAME ANOMALY DETECTION")

# Analyze hash pattern — look for files whose 8-char pattern doesn't match others
hash_lengths = Counter()
for r in rows:
    hash_lengths[len(r["hash"])] += 1
print(f"Hash length distribution: {dict(hash_lengths)}")

# Check for non-standard naming
all_train_files = [r["file"] for r in rows]
prefixes = Counter()
for f in all_train_files:
    fname = os.path.basename(f)
    p = fname.split("_")[0]
    prefixes[p] += 1
print(f"\nUnique filename prefixes in train: {len(prefixes)}")
if set(prefixes.keys()) - set(CLASSES):
    print(f"  ⚠ Unexpected prefixes: {set(prefixes.keys()) - set(CLASSES)}")

# ─── 8. Summary ───
print()
print("=" * 60)
print("SUMMARY OF HIDDEN FINDINGS")

# Calculate per-class mislabel rate
print("\nPer-class label reliability (folder label correct?):")
for cls in CLASSES:
    correct = sum(1 for r in rows if r["folder"] == cls and r["match"])
    total_c = sum(1 for r in rows if r["folder"] == cls)
    print(f"  {cls}: {correct}/{total_c} correct ({100*correct/total_c:.1f}%)")

print(f"\nTotal mislabeled images: {total_mismatch} / {total}")
print(f"Overall label noise rate: {100*total_mismatch/total:.1f}%")
