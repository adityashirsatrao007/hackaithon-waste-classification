import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
from PIL import Image

from src.model import CLASSES


def check_duplicates(images):
    hash_map = defaultdict(list)
    for img in images:
        h = hashlib.md5(img["path"].read_bytes()).hexdigest()
        hash_map[h].append(img)
    return [v for v in hash_map.values() if len(v) > 1]


def check_near_duplicates(images):
    import imagehash
    phashes = {}
    for img in images:
        try:
            pil_img = Image.open(img["path"])
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
    sizes = [(img, img["path"].stat().st_size) for img in images]
    vals = np.array([s[1] for s in sizes])
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = [s for s in sizes if s[1] < lower or s[1] > upper]
    return outliers, {
        "q1": int(q1), "q3": int(q3), "iqr": int(iqr),
        "min": int(vals.min()), "max": int(vals.max()),
    }


def confident_learning_correction(samples, oof_probs, num_classes=6):
    probs = oof_probs
    preds = probs.argmax(1)
    orig_labels = np.array([s[1] for s in samples])

    thresholds = {}
    for c in range(num_classes):
        mask = orig_labels == c
        thresholds[c] = probs[mask, c].mean() if mask.sum() > 0 else 0.5

    corrected = []
    stats = {"total": len(samples), "kept": 0, "corrected": 0, "flagged": 0}
    for i, (path, orig_label) in enumerate(samples):
        max_prob_class = preds[i]
        max_prob = probs[i].max()

        if max_prob_class == orig_label:
            corrected.append((path, orig_label))
            stats["kept"] += 1
        elif max_prob > thresholds.get(max_prob_class, 0.5) and max_prob > 0.4:
            corrected.append((path, int(max_prob_class)))
            stats["corrected"] += 1
        else:
            corrected.append((path, orig_label))
            stats["flagged"] += 1

    return corrected, stats, thresholds


def get_all_images(data_dir, classes=None):
    if classes is None:
        classes = CLASSES
    images = []
    for cls in classes:
        for f in (Path(data_dir) / cls).glob("*.*"):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                images.append({"path": f, "folder": cls})
    return images
