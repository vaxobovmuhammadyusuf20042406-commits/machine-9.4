from pathlib import Path

raw_path = Path("data/raw")

images = list(raw_path.glob("*.png"))

print(f"Total images found: {len(images)}")

if len(images) == 7200:
    print("✅ Dataset is complete.")
else:
    print("❌ Dataset is incomplete.")
    