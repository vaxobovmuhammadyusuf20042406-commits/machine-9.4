"""Evaluate the trained models on the test set.

Example:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model convnexttiny
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from modules.models import (
    MODEL_SPECS,
    image_transform,
    load_available_models,
)

TEST_DIR = Path("data/test")
BATCH_SIZE = 16

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--model",
    choices=["all", *MODEL_SPECS],
    default="all",
    help="Model to evaluate (default: all)."
)
args = parser.parse_args()

if not TEST_DIR.exists():
    sys.exit(
        f"Test folder not found: {TEST_DIR}. "
        "Run scripts/split_dataset.py first."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = datasets.ImageFolder(TEST_DIR, transform=image_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

models = load_available_models(device)

if args.model != "all":
    if args.model not in models:
        sys.exit(f"No trained checkpoint found for model: {args.model}")

    models = {args.model: models[args.model]}

print(f"Device: {device}")
print(f"Test images: {len(test_dataset)}")
print()
print(f"{'Model':<18} {'Top-1':>8} {'Top-5':>8} {'ms/image':>10}")
print("-" * 47)

for loaded in models.values():
    if loaded.class_names != test_dataset.classes:
        sys.exit(
            f"{loaded.name}: checkpoint classes do not match {TEST_DIR}."
        )

    top_1_correct = 0
    top_5_correct = 0
    inference_seconds = 0.0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            start_time = time.perf_counter()
            outputs = loaded.model(images)
            inference_seconds += time.perf_counter() - start_time

            top_5 = outputs.topk(5, dim=1).indices
            matches = top_5 == labels.unsqueeze(1)

            top_1_correct += matches[:, 0].sum().item()
            top_5_correct += matches.any(dim=1).sum().item()

    total = len(test_dataset)

    print(
        f"{loaded.name:<18} "
        f"{top_1_correct / total:>8.2%} "
        f"{top_5_correct / total:>8.2%} "
        f"{inference_seconds / total * 1000:>10.2f}"
    )
