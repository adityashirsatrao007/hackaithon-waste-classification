#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1. Dataset Inspection ==="
python scripts/inspect_dataset.py

echo -e "\n=== 2. Baseline Training (folder labels) ==="
python scripts/train_baseline.py

echo -e "\n=== 3. Visual Analysis ==="
python scripts/visualize.py

echo -e "\n=== 4. Label Correction Ablation ==="
python scripts/clean_data.py

echo -e "\n=== 5. Improved Pipeline (all fixes) ==="
python scripts/train_improved.py

echo -e "\n=== 6. Confident Learning ==="
python scripts/train_confident_learning.py

echo -e "\n=== Done ==="
