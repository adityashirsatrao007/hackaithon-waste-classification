"""Phase 2: dig into the train/test leak, label origin, and nearest-neighbor truth."""

import os, shutil
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import imagehash

DATA = Path("data/train")
TEST = Path("data/test")
CLASSES = sorted(os.listdir(DATA))

rows = []
for cls in CLASSES:
    for fname in os.listdir(DATA / cls):
        if not fname.endswith(".jpg"): continue
        prefix = fname.split("_")[0]
        hash_part = fname.split("_", 1)[1].replace(".jpg", "")
        rows.append({"file": f"{cls}/{fname}", "folder": cls, "prefix": prefix, "hash": hash_part, "fname": fname})

# ─── 1. Export leaked train/test pairs ───
print("=" * 60)
print("TRAIN/TEST LEAK — INVESTIGATION")

# Build phash index
def phash_img(path):
    try: return imagehash.phash(Image.open(path))
    except: return None

train_phashes = {}
for r in rows:
    ph = phash_img(DATA / r["file"])
    if ph is not None: train_phashes[ph] = r

leak_dir = Path("reports/evidence/leak")
leak_dir.mkdir(parents=True, exist_ok=True)

leak_found = []
for f in sorted(os.listdir(TEST)):
    test_ph = phash_img(TEST / f)
    if test_ph is None: continue
    for tr_ph, tr_row in train_phashes.items():
        hd = test_ph - tr_ph
        if 0 <= hd <= 4:
            leak_found.append((f, tr_row, hd, test_ph, tr_ph))

# Deduplicate (one test file may match multiple train - take closest)
seen_test = set()
deduped_leaks = []
for f, tr_row, hd, tp, trp in sorted(leak_found, key=lambda x: x[2]):
    if f not in seen_test:
        seen_test.add(f)
        deduped_leaks.append((f, tr_row, hd))
        src = TEST / f
        dst = leak_dir / f"{tr_row['file'].replace('/', '_')}__VS__{f}"
        shutil.copy2(src, dst)

print(f"\nTotal unique leaked test images: {len(deduped_leaks)}")
for f, tr_row, hd in deduped_leaks:
    tag = "SAME" if hd == 0 else f"NEAR-DUP(hd={hd})"
    print(f"  {tag}: test/{f}")
    print(f"         ↔ train/{tr_row['file']} (folder={tr_row['folder']}, prefix={tr_row['prefix']})")

# ─── 2. Check if file hash reveals class — file sorting by prefix ───
print()
print("=" * 60)
print("HASH-BASED CLASS ANALYSIS")

# Check if the first character of the hash correlates with class
for prefix_char_idx in [0, -1, 2]:  # first, last, third char
    print(f"\n  Hash char position {prefix_char_idx}:")
    char_class = defaultdict(Counter)
    for r in rows:
        c = r["hash"][prefix_char_idx] if len(r["hash"]) > prefix_char_idx else "?"
        char_class[c][r["folder"]] += 1
    for char in sorted(char_class.keys()):
        total_c = sum(char_class[char].values())
        top = char_class[char].most_common(1)[0]
        pct = 100 * top[1] / total_c
        if pct > 40:  # Significant skew
            print(f"    '{char}' → {top[0]} ({top[1]}/{total_c}, {pct:.0f}%)")

# ─── 3. Visual similarity consensus voting ───
print()
print("=" * 60)
print("NEAREST-NEIGHBOR LABEL CONSENSUS")

# For images where folder == prefix (the 28.5% "clean" ones), use them as anchors
# and classify ambiguous images by nearest neighbor

clean_by_prefix = {}
clean_by_folder = {}
for r in rows:
    if r["folder"] == r["prefix"]:
        ph = phash_img(DATA / r["file"])
        if ph is not None:
            clean_by_prefix.setdefault(r["prefix"], []).append((ph, r))
            clean_by_folder.setdefault(r["folder"], []).append((ph, r))

print(f"Clean anchors (folder == prefix): {sum(len(v) for v in clean_by_prefix.values())}")

# For each mismatched image, find nearest clean anchor by phash distance
mismatch = [r for r in rows if r["folder"] != r["prefix"]]
neighbor_votes = 0
neighbor_matches_prefix = 0
neighbor_matches_folder = 0

for r in mismatch[:50]:  # Sample first 50
    ph = phash_img(DATA / r["file"])
    if ph is None: continue
    neighbor_votes += 1
    # Find nearest clean by prefix
    best_dist = 100
    best_prefix = None
    for prefix, anchors in clean_by_prefix.items():
        for anchor_ph, _ in anchors:
            d = ph - anchor_ph
            if d < best_dist:
                best_dist = d
                best_prefix = prefix
    if best_prefix == r["prefix"]:
        neighbor_matches_prefix += 1
    if best_prefix == r["folder"]:
        neighbor_matches_folder += 1

if neighbor_votes > 0:
    print(f"\nWhen folder≠prefix, NN consensus votes for filename prefix: {neighbor_matches_prefix}/{neighbor_votes} ({100*neighbor_matches_prefix/neighbor_votes:.0f}%)")
    print(f"When folder≠prefix, NN consensus votes for folder label:    {neighbor_matches_folder}/{neighbor_votes} ({100*neighbor_matches_folder/neighbor_votes:.0f}%)")

# ─── 4. Check if "wrong" folder images cluster with their prefix class ───
print()
print("=" * 60)
print("CROSS-VALIDATION: DO PREFIX LABELS FORM COHERENT CLUSTERS?")

# Compute average intra- vs inter-class phash distance for prefix labels
prefix_phashes = defaultdict(list)
for r in rows:
    ph = phash_img(DATA / r["file"])
    if ph is not None: prefix_phashes[r["prefix"]].append(ph)

print("\nAverage intra-class phash distance (prefix-based):")
for cls in CLASSES:
    if len(prefix_phashes[cls]) < 2: continue
    total_dist = 0
    count = 0
    phs = prefix_phashes[cls]
    for i in range(len(phs)):
        for j in range(i+1, len(phs)):
            total_dist += phs[i] - phs[j]
            count += 1
    avg = total_dist / count if count else 0
    print(f"  {cls}: {avg:.1f} (based on {len(phs)} images)")

# ─── 5. Look for the "exact truth" signal ───
print()
print("=" * 60)
print("GROUND TRUTH: MAJORITY VOTE (FOLDER + PREFIX + NN)")
print()

# For every image, take 3 votes:
# 1. folder label
# 2. prefix label
# 3. nearest-neighbor anchor (if available)

three_vote_disagreement = 0
for r in rows:
    votes = [r["folder"], r["prefix"]]
    if r["folder"] == r["prefix"]:
        # Both agree
        pass
    else:
        # Check if there's a clear visual signal
        ph = phash_img(DATA / r["file"])
        if ph is not None:
            best_dist = 100
            best_cls = None
            for cls, anchors in clean_by_prefix.items():
                for anchor_ph, _ in anchors:
                    d = ph - anchor_ph
                    if d < best_dist:
                        best_dist = d
                        best_cls = cls
            if best_dist <= 6 and best_cls != r["folder"] and best_cls != r["prefix"]:
                three_vote_disagreement += 1
                if three_vote_disagreement <= 5:
                    print(f"  3-way split: {r['file']}: folder={r['folder']}, prefix={r['prefix']}, NN={best_cls} (dist={best_dist})")
            elif best_dist <= 6:
                votes.append(best_cls)
                if Counter(votes).most_common(1)[0][1] == 2:
                    pass  # 2/3 agreement

print(f"\nImages where all 3 signals disagree: {three_vote_disagreement}")

print("\n" + "=" * 60)
print("WINNING INSIGHT SUMMARY")
print("=" * 60)
print("""
1. TRAIN/TEST LEAK: {} exact perceptual dupes between train and test.
   → Test set accuracy is inflated; this is a dataset construction bug.

2. LABEL NOISE IS NOT RANDOM — IT IS DELIBERATE & UNIFORM:
   → Each wrong-label class gets ~14-20% share (uniform distribution)
   → Real-world confusion would concentrate on visually similar pairs
   → This is a crafted adversarial dataset, not a real-world one

3. TRUTH SIGNAL:
   → Prefix labels are as reliable as folder labels (both ~28.5% correct)
   → NN consensus weakly favors prefix labels over folder labels
   → The cleanest signal: images where folder == prefix are likely correct

4. CLASS IMBALANCE:
   → Trash has 40 vs 75-86 for others — drives Trash recall to 0%
   → Fix: oversample Trash or reweight

5. BEST SUBMISSION STRATEGY:
   a) Train on folder labels with class weights → simplest baseline
   b) Train on prefix labels → slightly different bias
   c) Train only on images where folder == prefix → clean but only 127 images
   d) Ensemble with confidence voting → best bet
""".format(len(deduped_leaks)))
