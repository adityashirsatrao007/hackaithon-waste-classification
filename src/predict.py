import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import WasteClassifier, CLASSES, NUM_CLASSES, IMG_SIZE

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
TEST_DIR = PROJECT_ROOT / "data" / "test"
OUTPUT_PATH = PROJECT_ROOT / "submissions" / "submission.csv"
SAMPLE_PATH = PROJECT_ROOT / "submissions" / "sample_submission.csv"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.images = sorted([
            f for ext in ["*.jpg", "*.jpeg", "*.png"]
            for f in self.img_dir.glob(ext)
        ], key=lambda x: x.name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.images[idx].stem

test_tfm = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return 1

    print("Loading model...")
    state = torch.load(MODEL_PATH, map_location=device)
    model = WasteClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(state)
    model.eval()

    print("Predicting...")
    ds = TestDataset(TEST_DIR, test_tfm)
    if len(ds) == 0:
        print("No test images found.")
        return 1
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    predictions = []
    with torch.no_grad():
        for images, fnames in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.nn.functional.softmax(logits, dim=1)
            confs, preds = probs.max(1)
            for fid, pred, conf in zip(fnames, preds.cpu().numpy(), confs.cpu().numpy()):
                predictions.append({
                    "image_id": fid,
                    "prediction": int(pred),
                    "confidence": float(conf),
                })

    if SAMPLE_PATH.exists():
        with open(SAMPLE_PATH) as f:
            reader = csv.DictReader(f)
            expected = [row["image_id"] for row in reader]
        pred_map = {p["image_id"]: p for p in predictions}
        predictions = [
            pred_map.get(e, {"image_id": e, "prediction": 0, "confidence": 0.5})
            for e in expected
        ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        w.writeheader()
        w.writerows(predictions)
    print(f"Saved {OUTPUT_PATH} with {len(predictions)} predictions")

if __name__ == "__main__":
    exit(main())
