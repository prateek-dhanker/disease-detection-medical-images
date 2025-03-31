import os
import shutil
import random

# Define paths based on your structure
base_dir = "Skin/2"  # Adjusted path to match your structure
train_dir = os.path.join("Skin", "train")
test_dir = os.path.join("Skin", "test")

# Get category names dynamically (subfolders inside `Skin/2/`)
categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Split ratio
split_ratio = 0.8  # 80% training, 20% testing

# Create train and test directories
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    # Get all images from the original dataset
    category_path = os.path.join(base_dir, category)
    images = [img for img in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, img))]
    random.shuffle(images)

    # Compute split index
    split_index = int(len(images) * split_ratio)

    # Move images to train/test folders
    for i, img in enumerate(images):
        src = os.path.join(category_path, img)
        if i < split_index:
            dest = os.path.join(train_dir, category, img)
        else:
            dest = os.path.join(test_dir, category, img)
        shutil.move(src, dest)

print("Dataset split completed! Train and test sets are ready.")
