import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv

MODEL_PATH = Path("best_model.pth")
TEST_DIR = Path("data/test")
OUTPUT_PATH = Path("submission.csv")
SAMPLE_PATH = Path("sample_submission.csv")
NUM_CLASSES = 6
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
BATCH_SIZE = 32
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

class ResNet18Classifier(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        feat = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n),
        )

    def forward(self, x):
        return self.classifier(self.resnet(x))

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
    model = ResNet18Classifier(NUM_CLASSES).to(device)
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
            probs = torch.nn.functional.softmax(model(images), dim=1)
            confs, preds = probs.max(1)
            for fid, pred, conf in zip(fnames, preds.cpu().numpy(), confs.cpu().numpy()):
                predictions.append({"image_id": fid, "prediction": int(pred), "confidence": float(conf)})

    # align with sample submission order if available
    if SAMPLE_PATH.exists():
        with open(SAMPLE_PATH) as f:
            reader = csv.DictReader(f)
            expected = [row["image_id"] for row in reader]
        pred_map = {p["image_id"]: p for p in predictions}
        predictions = [pred_map.get(e, {"image_id": e, "prediction": 0, "confidence": 0.5}) for e in expected]

    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        w.writeheader()
        w.writerows(predictions)
    print(f"Saved {OUTPUT_PATH} with {len(predictions)} predictions")

if __name__ == "__main__":
    exit(main())
