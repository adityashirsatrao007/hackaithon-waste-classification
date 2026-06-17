import csv
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.dataset import val_tfm
from src.model import WasteClassifier, NUM_CLASSES, IMG_SIZE


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


def generate_predictions(model, test_dir, sample_path, output_path, device="cpu", batch_size=32):
    model.eval()
    test_ds = TestDataset(test_dir, val_tfm)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    predictions = []
    with torch.no_grad():
        for images, fnames in tqdm(test_loader, desc="Predicting"):
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

    sample_path = Path(sample_path)
    if sample_path.exists():
        with open(sample_path) as f:
            reader = csv.DictReader(f)
            expected = [row["image_id"] for row in reader]
        pred_map = {p["image_id"]: p for p in predictions}
        predictions = [
            pred_map.get(e, {"image_id": e, "prediction": 0, "confidence": 0.5})
            for e in expected
        ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        w.writeheader()
        w.writerows(predictions)
    print(f"Saved {output_path} with {len(predictions)} predictions")
    return predictions
