from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import convnext_tiny, efficientnet_v2_s


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = 224


# =========================================================
# IMAGE PREPROCESSING
# Matches validation/test preprocessing of all three models
# =========================================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# MODEL DEFINITIONS
# These exactly match the training scripts
# =========================================================

class CustomCNN(nn.Module):
    def __init__(self, number_of_output_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.40),
            nn.Linear(128, number_of_output_classes)
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def build_custom_cnn(number_of_classes: int) -> nn.Module:
    return CustomCNN(number_of_classes)


def build_efficientnetv2(number_of_classes: int) -> nn.Module:
    model = efficientnet_v2_s(weights=None)

    classifier_input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(classifier_input_features, number_of_classes)
    )

    return model


def build_convnexttiny(number_of_classes: int) -> nn.Module:
    model = convnext_tiny(weights=None)

    classifier_input_features = model.classifier[2].in_features

    model.classifier[2] = nn.Linear(
        classifier_input_features,
        number_of_classes
    )

    return model


# Display order: best model first.
# "gradcam_layer" picks the last layer that still has a spatial
# feature map, which is what Grad-CAM needs.
MODEL_SPECS = {
    "convnexttiny": {
        "name": "ConvNeXt-Tiny",
        "builder": build_convnexttiny,
        "gradcam_layer": lambda model: model.features[-1],
    },
    "efficientnetv2": {
        "name": "EfficientNetV2-S",
        "builder": build_efficientnetv2,
        "gradcam_layer": lambda model: model.features[-1],
    },
    "custom_cnn": {
        "name": "Custom CNN",
        "builder": build_custom_cnn,
        "gradcam_layer": lambda model: model.features[-2],
    },
}


@dataclass
class LoadedModel:
    key: str
    name: str
    model: nn.Module
    class_names: list
    checkpoint_path: Path

    @property
    def gradcam_layer(self) -> nn.Module:
        return MODEL_SPECS[self.key]["gradcam_layer"](self.model)


# =========================================================
# CHECKPOINT LOADING
# =========================================================

def detect_architecture(state_dict: dict) -> str:
    """Identify the architecture from the layer names in a checkpoint.

    The architecture is read from the weights themselves, so a
    renamed or mislabelled checkpoint file still loads correctly.
    """
    if "classifier.4.weight" in state_dict:
        return "custom_cnn"

    if "classifier.2.weight" in state_dict:
        return "convnexttiny"

    if "classifier.1.weight" in state_dict:
        return "efficientnetv2"

    raise ValueError("Checkpoint does not match any known architecture.")


def load_checkpoint(checkpoint_path: Path, device) -> LoadedModel:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")

    if "class_names" not in checkpoint:
        raise KeyError("Checkpoint is missing 'class_names'.")

    state_dict = checkpoint["model_state_dict"]
    class_names = checkpoint["class_names"]

    key = detect_architecture(state_dict)
    model = MODEL_SPECS[key]["builder"](len(class_names))

    model.load_state_dict(state_dict, strict=True)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.to(device)
    model.eval()

    return LoadedModel(
        key=key,
        name=MODEL_SPECS[key]["name"],
        model=model,
        class_names=class_names,
        checkpoint_path=checkpoint_path
    )


def load_available_models(device, model_dir: Path = MODEL_DIR) -> dict:
    """Load every '*_best.pth' checkpoint, keyed by its real architecture."""
    found = {}

    for checkpoint_path in sorted(model_dir.glob("*_best.pth")):
        loaded = load_checkpoint(checkpoint_path, device)
        found.setdefault(loaded.key, loaded)

    if not found:
        raise FileNotFoundError(
            f"No '*_best.pth' checkpoints found in: {model_dir}"
        )

    return {key: found[key] for key in MODEL_SPECS if key in found}


# =========================================================
# PREDICTION
# =========================================================

def predict_top_k(loaded: LoadedModel, input_tensor, k: int = 5) -> list:
    with torch.inference_mode():
        outputs = loaded.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    top_k = min(k, len(loaded.class_names))

    top_confidences, top_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=1
    )

    return [
        {
            "index": class_index,
            "label": loaded.class_names[class_index],
            "confidence": round(class_confidence * 100, 2)
        }
        for class_index, class_confidence in zip(
            top_indices[0].tolist(),
            top_confidences[0].tolist()
        )
    ]
