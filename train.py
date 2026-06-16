"""
Train ResNet-18 classifier on Waste Classification dataset using 3LC.

- 3LC Table loading (train + val). 
- ResNet-18 training.
- Per-sample metrics and embeddings collection.
- Best model saved to best_model.pth (overwritten each run).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tlc
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os
from tlc.integration.torch.samplers import create_weighted_sampler

# ============================================================================
# CONFIGURATION
# ============================================================================

EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
RANDOM_SEED = 42
PROJECT_NAME = "Waste-Classification"
DATASET_NAME = "waste-dataset"
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"ResNet-18: random init (no pretrained weights)")

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[OK] Random seed set to {seed}")

# ============================================================================
# MODEL
# ============================================================================

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)

# ============================================================================
# TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def train_fn(sample):
    image_path = tlc.Url(sample["image"]).to_absolute()
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return train_transform(image), sample["label"]

def val_fn(sample):
    image_path = tlc.Url(sample["image"]).to_absolute()
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return val_transform(image), sample["label"]

# ============================================================================
# METRICS
# ============================================================================

def metrics_fn(batch, predictor_output: tlc.metrics.PredictorOutput):
    labels = batch[1].to(device)
    predictions = predictor_output.forward
    softmax_output = F.softmax(predictions, dim=1)
    predicted_indices = torch.argmax(predictions, dim=1)
    confidence = torch.gather(softmax_output, 1, predicted_indices.unsqueeze(1)).squeeze(1)
    accuracy = (predicted_indices == labels).float()
    valid_labels = labels < predictions.shape[1]
    cross_entropy_loss = torch.ones_like(labels, dtype=torch.float32)
    cross_entropy_loss[valid_labels] = nn.CrossEntropyLoss(reduction="none")(
        predictions[valid_labels], labels[valid_labels]
    )
    return {
        "loss": cross_entropy_loss.cpu().numpy(),
        "predicted": predicted_indices.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }

# ============================================================================
# TRAINING
# ============================================================================

BEST_MODEL_FILENAME = "best_model.pth"

def train():
    set_seed(RANDOM_SEED)
    base_path = Path(__file__).parent


    print("\nLoading 3LC tables...")
    train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train",
    ).latest()
    val_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="val",
    ).latest()

    print(f"  Train: {len(train_table)} samples")
    print(f"  Val:   {len(val_table)} samples")
    class_names = CLASS_NAMES

    train_dataset = train_table.with_transform(train_fn)
    val_dataset = val_table.with_transform(val_fn)
    sampler = create_weighted_sampler(train_table)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    run = tlc.init(
        project_name=PROJECT_NAME,
        run_name="fixed_image_run",
        description="Waste Classification - data-centric workflow",
        if_exists="overwrite",
    )
    metric_schemas = {
        "loss": tlc.schemas.Float32Schema(description="Cross entropy loss"),
        "predicted": tlc.schemas.CategoricalLabelSchema(display_name="predicted label", classes=class_names),
        "accuracy": tlc.schemas.Float32Schema(description="Per-sample accuracy"),
        "confidence": tlc.schemas.Float32Schema(description="Prediction confidence"),
    }
    classification_metrics_collector = tlc.metrics.FunctionalMetricsCollector(
        collection_fn=metrics_fn,
        schema=metric_schemas,
    )
    indices_and_modules = list(enumerate(model.resnet.named_modules()))
    resnet_fc_layer_index = next((i for i, (n, _) in indices_and_modules if n == "fc"), len(indices_and_modules) - 1)
    embeddings_metrics_collector = tlc.metrics.EmbeddingsMetricsCollector(layers=[resnet_fc_layer_index])
    predictor = tlc.metrics.Predictor(model, layers=[resnet_fc_layer_index])

    best_val_accuracy = 0.0
    best_model_state = None
    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        val_accuracy = 100 * val_correct / val_total
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Acc: {val_accuracy:.2f}%")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  --> New best model!")
        tlc.log({"epoch": epoch, "val_accuracy": val_accuracy})

    print("\n" + "=" * 60)
    print(f"  Best validation accuracy: {best_val_accuracy:.2f}%")
    print("=" * 60)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model_path = base_path / BEST_MODEL_FILENAME
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Best model saved to {model_path} (overwrites previous run)")

    print("\nCollecting metrics on train set...")
    model.eval()
    metrics_dataset = train_table.with_transform(val_fn)
    tlc.collect_metrics(
        metrics_dataset,
        predictor=predictor,
        metrics_collectors=[classification_metrics_collector, embeddings_metrics_collector],
        split="train",
        dataloader_args={"batch_size": BATCH_SIZE, "num_workers": 0},
    )
    print("\nReducing embeddings...")
    try:
        run.reduce_embeddings_by_foreign_table_url(
            train_table.url,
            method="umap",
            n_neighbors=15,
            n_components=3,
        )
        print("  [OK] Embeddings reduced.")
    except Exception as e:
        print(f"  WARNING: Embedding reduction failed: {e}")
    run.set_status_completed()
    print("\n[OK] Done. View results at 3LC Dashboard.")

if __name__ == "__main__":
    train()
