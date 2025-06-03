import os
import shutil
import random

def create_val_split(train_dir, val_dir, val_ratio=0.15):
    os.makedirs(val_dir, exist_ok=True)

    for cls in os.listdir(train_dir):
        class_path = os.path.join(train_dir, cls)
        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(files)
        val_count = int(len(files) * val_ratio)
        val_files = files[:val_count]

        val_class_dir = os.path.join(val_dir, cls)
        os.makedirs(val_class_dir, exist_ok=True)

        for f in val_files:
            src = os.path.join(class_path, f)
            dst = os.path.join(val_class_dir, f)
            shutil.copy(src, dst)

if __name__ == "__main__":
    create_val_split("data/train", "data/val", val_ratio=0.15)

