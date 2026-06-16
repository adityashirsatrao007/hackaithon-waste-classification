"""
Register Waste Classification dataset in 3LC tables.

Creates 3LC tables for train and val by splitting the images in data/ into 80/20 splits.
"""

import tlc
from pathlib import Path
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
PROJECT_NAME = "Waste-Classification"
DATASET_NAME = "waste-dataset"

schemas = {
    "id": tlc.schemas.Int32Schema(),
    "image": tlc.schemas.ImageSchema(sample_type="url"),
    "label": tlc.schemas.CategoricalLabelSchema(display_name="label", classes=CLASSES),
    "weight": tlc.schemas.SampleWeightSchema(),
}





def register_dataset_to_tables(data_path: Path):
    data_path = Path(data_path)
    train_data = []
    val_data = []

    for class_idx, class_name in enumerate(CLASSES):
        class_folder = data_path / class_name
        if class_folder.exists():
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_files.extend(list(class_folder.glob(ext)))
            
            image_files = sorted(image_files)
            random.seed(42)
            random.shuffle(image_files)
            
            split_idx = int(0.8 * len(image_files))
            for i, img_path in enumerate(image_files):
                # Write the absolute path to the data inside the 3LC project root
                project_data_path = Path.home() / ".local/share/3LC/projects" / PROJECT_NAME / "data"
                rel_path = img_path.relative_to(data_path.parent) # e.g. train/plastic/1.jpg
                abs_path_in_project = project_data_path / rel_path
                data = {"path": str(abs_path_in_project), "label": class_idx}
                if i < split_idx:
                    train_data.append(data)
                else:
                    val_data.append(data)
        else:
            print(f"  [WARN] {class_folder} does not exist")

    print(f"\n  Total images for train: {len(train_data)}")
    print(f"  Total images for val: {len(val_data)}")

    def write_table(image_data, table_name, split_name):
        table_writer = tlc.TableWriter(
            table_name=table_name,
            dataset_name=DATASET_NAME,
            project_name=PROJECT_NAME,
            description=f"Waste Classification {split_name} set with {len(image_data)} images",
            if_exists="overwrite",
            schema=schemas,
        )
        for i, data in enumerate(image_data):
            table_writer.add_row({
                "id": i,
                "image": data["path"],
                "label": data["label"],
                "weight": 1.0,
            })
        return table_writer.finalize()

    print("\nRegistering TRAIN set...")
    train_table = write_table(train_data, "train", "train")
    print(f"  [OK] Train table URL: {train_table.url}")

    print("\nRegistering VAL set...")
    val_table = write_table(val_data, "val", "val")
    print(f"  [OK] Val table URL: {val_table.url}")


def main():
    base_path = Path(__file__).parent
    data_path = base_path / "data" / "train"

    print("=" * 70)
    print("  Registering Waste Classification Dataset in 3LC Tables")
    print("=" * 70)

    if not data_path.exists():
        print(f"\n[ERROR] Train directory not found: {data_path}")
        return

    register_dataset_to_tables(data_path)
    print("\n" + "-" * 70)
    print("  [OK] Successfully registered tables!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
