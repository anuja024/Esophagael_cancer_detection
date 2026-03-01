import os
import shutil
import random
from pathlib import Path

# Paths
SOURCE_DIR = "Kvasir-dataset-v2"   # change this to where your original dataset is
DEST_DIR = "data"                  # output folder with train/ and val/

# Train/val split ratio
SPLIT_RATIO = 0.8   # 80% train, 20% val

# Create train/ and val/ folders
for split in ["train", "val"]:
    split_path = Path(DEST_DIR) / split
    split_path.mkdir(parents=True, exist_ok=True)

# Go through each class folder
for class_name in os.listdir(SOURCE_DIR):
    class_path = Path(SOURCE_DIR) / class_name
    if not class_path.is_dir():
        continue  # skip files

    # Make output folders for this class
    train_class_dir = Path(DEST_DIR) / "train" / class_name
    val_class_dir = Path(DEST_DIR) / "val" / class_name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    # List all images
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)

    # Split into train/val
    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Copy files
    for img in train_imgs:
        shutil.copy(class_path / img, train_class_dir / img)
    for img in val_imgs:
        shutil.copy(class_path / img, val_class_dir / img)

    print(f"Class {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

print("✅ Done! Dataset split into 'data/train' and 'data/val'")
