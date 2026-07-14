from pathlib import Path
import time
import uuid

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import convnext_tiny
from werkzeug.utils import secure_filename


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"

# IMPORTANT:
# Your ConvNeXt training script saved ConvNeXt checkpoints
# using EfficientNet filenames. Therefore this path is correct.
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnetv2_best.pth"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# HELPERS
# =========================================================

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "High confidence"
    if confidence >= 0.50:
        return "Medium confidence"
    return "Low confidence"


# =========================================================
# MODEL DEFINITION
# This exactly matches train_convnexttiny.py
# =========================================================

def build_model(number_of_classes: int) -> nn.Module:
    model = convnext_tiny(weights=None)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    classifier_input_features = model.classifier[2].in_features

    model.classifier[2] = nn.Linear(
        classifier_input_features,
        number_of_classes
    )

    return model


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")

    if "class_names" not in checkpoint:
        raise KeyError("Checkpoint is missing 'class_names'.")

    class_names = checkpoint["class_names"]
    model = build_model(len(class_names))

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(device)
    model.eval()

    return model, class_names


model, class_names = load_model()


# =========================================================
# IMAGE PREPROCESSING
# Matches validation/test preprocessing
# =========================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")

    if image_file is None or image_file.filename == "":
        return jsonify({
            "success": False,
            "message": "Please select an image."
        }), 400

    if not allowed_file(image_file.filename):
        return jsonify({
            "success": False,
            "message": "Only PNG, JPG, and JPEG files are allowed."
        }), 400

    filename = (
        f"{uuid.uuid4().hex}_"
        f"{secure_filename(image_file.filename)}"
    )

    saved_path = UPLOAD_FOLDER / filename
    image_file.save(saved_path)

    try:
        image = Image.open(saved_path).convert("RGB")
    except Exception:
        saved_path.unlink(missing_ok=True)

        return jsonify({
            "success": False,
            "message": "The uploaded file is not a valid image."
        }), 400

    input_tensor = image_transform(image).unsqueeze(0).to(device)

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    top_k = min(5, len(class_names))

    top_confidences, top_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=1
    )

    top_5_predictions = []

    for class_index, class_confidence in zip(
        top_indices[0].tolist(),
        top_confidences[0].tolist()
    ):
        top_5_predictions.append({
            "label": class_names[class_index],
            "confidence": round(class_confidence * 100, 2)
        })

    best_prediction = top_5_predictions[0]

    return jsonify({
        "success": True,
        "prediction": best_prediction["label"],
        "confidence": best_prediction["confidence"],
        "confidence_level": confidence_level(
            best_prediction["confidence"] / 100
        ),
        "model": "ConvNeXt-Tiny",
        "inference_time_ms": round(inference_time_ms, 2),
        "top_5_predictions": top_5_predictions,
        "message": "Prediction completed successfully."
    })


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "message": "The image is too large. Maximum size is 8 MB."
    }), 413


if __name__ == "__main__":
    print("=" * 60)
    print("Assistive Object Recognition")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Loaded checkpoint: {MODEL_PATH}")
    print("Architecture: ConvNeXt-Tiny")
    print(f"Number of classes: {len(class_names)}")
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
