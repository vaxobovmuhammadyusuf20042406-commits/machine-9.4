from pathlib import Path
from PIL import Image

raw_path = Path("data/raw")
processed_path = Path("data/processed")
processed_path.mkdir(parents=True, exist_ok=True)

image_files = list(raw_path.glob("*.png"))

for image_file in image_files:
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img.save(processed_path / image_file.name)

print("Processed images:", len(list(processed_path.glob('*.png'))))
print("All images resized to 224x224.")