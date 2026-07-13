from pathlib import Path
import shutil
from collections import defaultdict

processed_path = Path("data/processed")
train_path = Path("data/train")
val_path = Path("data/val")
test_path = Path("data/test")

for folder in [train_path, val_path, test_path]:
    folder.mkdir(parents=True, exist_ok=True)

# Group images by object ID
objects = defaultdict(list)

for image_file in processed_path.glob("*.png"):
    name = image_file.stem

    # Example: obj1__45
    object_id = name.split("__")[0]
    angle = int(name.split("__")[1])

    objects[object_id].append((angle, image_file))

# Split each object by angle
for object_id, images in objects.items():
    images.sort(key=lambda x: x[0])

    total = len(images)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for _, img in train_images:
        class_folder = train_path / object_id
        class_folder.mkdir(exist_ok=True)
        shutil.copy(img, class_folder / img.name)

    for _, img in val_images:
        class_folder = val_path / object_id
        class_folder.mkdir(exist_ok=True)
        shutil.copy(img, class_folder / img.name)

    for _, img in test_images:
        class_folder = test_path / object_id
        class_folder.mkdir(exist_ok=True)
        shutil.copy(img, class_folder / img.name)

print("Dataset split completed.")