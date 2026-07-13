from pathlib import Path
import time
import uuid

import cv2
import numpy as np

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from werkzeug.utils import secure_filename


app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnetv2_final.pth"

GRADCAM_FOLDER = PROJECT_ROOT / "static" / "generated"
GRADCAM_FOLDER.mkdir(parents=True, exist_ok=True)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def build_efficientnet_model(number_of_classes: int):
    model = efficientnet_v2_s(weights=None)

    classifier_input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(
            classifier_input_features,
            number_of_classes
        )
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

    class_names = checkpoint["class_names"]
    number_of_classes = len(class_names)

    model = build_efficientnet_model(number_of_classes)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, class_names

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(input_tensor)

        if class_index is None:
            class_index = outputs.argmax(dim=1).item()

        target_score = outputs[0, class_index]
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations or gradients."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (
            weights * self.activations
        ).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)
        cam = cam.squeeze()

        cam_min = cam.min()
        cam_max = cam.max()

        if float(cam_max - cam_min) > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

model, class_names = load_model()
gradcam = GradCAM(
    model=model,
    target_layer=model.features[-1]
)


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "High confidence"
    if confidence >= 0.50:
        return "Medium confidence"
    return "Low confidence"

def create_gradcam_overlay(
    original_image: Image.Image,
    input_tensor: torch.Tensor,
    predicted_index: int
) -> str:
    cam = gradcam.generate(
        input_tensor=input_tensor,
        class_index=predicted_index
    )

    original_array = np.array(
        original_image.convert("RGB")
    )

    original_height, original_width = original_array.shape[:2]

    cam_resized = cv2.resize(
        cam,
        (original_width, original_height)
    )

    heatmap = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_array,
        0.55,
        heatmap,
        0.45,
        0
    )

    output_filename = (
        f"gradcam_{uuid.uuid4().hex}.jpg"
    )

    output_path = GRADCAM_FOLDER / output_filename

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    return f"/static/generated/{output_filename}"

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image was uploaded."
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "success": False,
            "message": "Please select an image."
        }), 400

    if not allowed_file(image_file.filename):
        return jsonify({
            "success": False,
            "message": "Only PNG, JPG, and JPEG files are allowed."
        }), 400

    filename = secure_filename(image_file.filename)
    saved_path = UPLOAD_FOLDER / filename
    image_file.save(saved_path)

    try:
        image = Image.open(saved_path).convert("RGB")
    except Exception:
        return jsonify({
            "success": False,
            "message": "The uploaded file could not be read as an image."
        }), 400

    input_tensor = image_transform(image).unsqueeze(0).to(device)

    inference_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence_tensor, predicted_index_tensor = torch.max(
            probabilities,
            dim=1
        )

    inference_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    predicted_index = predicted_index_tensor.item()
    confidence = confidence_tensor.item()
    prediction = class_names[predicted_index]

    top_5_confidences, top_5_indices = torch.topk(
        probabilities,
        k=5,
        dim=1
    )

    top_5_predictions = []

    for class_index, class_confidence in zip(
        top_5_indices[0].tolist(),
        top_5_confidences[0].tolist()
    ):
        top_5_predictions.append({
            "label": class_names[class_index],
            "confidence": round(class_confidence * 100, 2)
        })
        gradcam_url = create_gradcam_overlay(
    original_image=image,
    input_tensor=input_tensor,
    predicted_index=predicted_index
)

    return jsonify({
        "success": True,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "confidence_level": get_confidence_level(confidence),
        "model": "EfficientNetV2-S",
        "inference_time_ms": round(inference_ms, 2),
        "top_5_predictions": top_5_predictions,
        "gradcam_url": gradcam_url,
        "message": "Prediction completed successfully."
        
    })


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "message": "The image is too large. Maximum size is 8 MB."
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )