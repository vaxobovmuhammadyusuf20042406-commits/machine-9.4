import time

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from modules.gradcam import (
    compute_gradcam,
    image_to_data_url,
    overlay_heatmap,
)
from modules.models import (
    MODEL_DIR,
    image_transform,
    load_available_models,
    predict_top_k,
)


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

PORT = 7860

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Every trained model found in the models folder, best model first.
models = load_available_models(device)
default_model_key = next(iter(models))


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
# ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        models=models.values(),
        default_model_key=default_model_key
    )


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

    model_key = request.form.get("model", default_model_key)

    if model_key not in models:
        return jsonify({
            "success": False,
            "message": "The selected model is not available."
        }), 400

    selected = models[model_key]

    try:
        image = Image.open(image_file.stream).convert("RGB")
    except Exception:
        return jsonify({
            "success": False,
            "message": "The uploaded file is not a valid image."
        }), 400

    input_tensor = image_transform(image).unsqueeze(0).to(device)

    start_time = time.perf_counter()

    top_5_predictions = predict_top_k(selected, input_tensor, k=5)

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    best_prediction = top_5_predictions[0]

    heatmap = compute_gradcam(
        selected.model,
        selected.gradcam_layer,
        input_tensor,
        best_prediction["index"]
    )

    gradcam_url = image_to_data_url(overlay_heatmap(image, heatmap))

    return jsonify({
        "success": True,
        "prediction": best_prediction["label"],
        "confidence": best_prediction["confidence"],
        "confidence_level": confidence_level(
            best_prediction["confidence"] / 100
        ),
        "model": selected.name,
        "inference_time_ms": round(inference_time_ms, 2),
        "top_5_predictions": [
            {
                "label": prediction["label"],
                "confidence": prediction["confidence"]
            }
            for prediction in top_5_predictions
        ],
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
    print("=" * 60)
    print("Assistive Object Recognition")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model folder: {MODEL_DIR}")

    for loaded in models.values():
        print(
            f"Loaded {loaded.name}: {loaded.checkpoint_path.name} "
            f"({len(loaded.class_names)} classes)"
        )

    print(f"Open: http://127.0.0.1:{PORT}")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )
