"""Predict the object in one image from the command line.

Example:
    python scripts/predict.py data/test/obj1/obj1__310.png
    python scripts/predict.py photo.jpg --model custom_cnn --gradcam out.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from modules.gradcam import compute_gradcam, overlay_heatmap
from modules.models import (
    MODEL_SPECS,
    image_transform,
    load_available_models,
    predict_top_k,
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("image", type=Path, help="Path to a PNG or JPG image.")
parser.add_argument(
    "--model",
    choices=list(MODEL_SPECS),
    default="convnexttiny",
    help="Model to use (default: convnexttiny)."
)
parser.add_argument(
    "--gradcam",
    type=Path,
    help="Save the Grad-CAM explanation image to this path."
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = load_available_models(device)

if args.model not in models:
    sys.exit(f"No trained checkpoint found for model: {args.model}")

selected = models[args.model]

image = Image.open(args.image).convert("RGB")
input_tensor = image_transform(image).unsqueeze(0).to(device)

predictions = predict_top_k(selected, input_tensor, k=5)

print(f"Model: {selected.name}")
print(f"Image: {args.image}")
print("Top-5 predictions:")

for rank, prediction in enumerate(predictions, start=1):
    print(
        f"  {rank}. {prediction['label']:<10} "
        f"{prediction['confidence']:6.2f}%"
    )

if args.gradcam:
    heatmap = compute_gradcam(
        selected.model,
        selected.gradcam_layer,
        input_tensor,
        predictions[0]["index"]
    )

    overlay_heatmap(image, heatmap).save(args.gradcam)

    print(f"Grad-CAM saved to: {args.gradcam}")
