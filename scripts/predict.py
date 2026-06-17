import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model import WasteClassifier, NUM_CLASSES
from src.predictor import generate_predictions

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return 1

    print("Loading model...")
    state = torch.load(MODEL_PATH, map_location=device)
    model = WasteClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(state)

    generate_predictions(
        model,
        PROJECT_ROOT / "data" / "test",
        PROJECT_ROOT / "submissions" / "sample_submission.csv",
        PROJECT_ROOT / "submissions" / "submission.csv",
        device,
    )
    return 0


if __name__ == "__main__":
    exit(main())
