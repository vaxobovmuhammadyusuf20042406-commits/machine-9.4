import base64
import io

import numpy as np
import torch
from PIL import Image


MAX_OVERLAY_SIDE = 512


def compute_gradcam(model, target_layer, input_tensor, class_index: int):
    """Return a Grad-CAM heatmap (2D array, values 0..1) for one image."""
    captured = {}

    def capture_activation(_module, _inputs, output):
        # The model weights are frozen, so gradients are only
        # tracked from this layer onwards.
        captured["activation"] = output.requires_grad_(True)
        return captured["activation"]

    hook = target_layer.register_forward_hook(capture_activation)

    try:
        with torch.enable_grad():
            outputs = model(input_tensor)
            score = outputs[0, class_index]

            gradients = torch.autograd.grad(
                score,
                captured["activation"]
            )[0]
    finally:
        hook.remove()

    activation = captured["activation"].detach()

    channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((channel_weights * activation).sum(dim=1))[0]

    heatmap -= heatmap.min()
    maximum = heatmap.max()

    if maximum > 0:
        heatmap /= maximum

    return heatmap.cpu().numpy()


def jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map values 0..1 to blue-green-yellow-red RGB colours (0..255)."""
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)

    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45
) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((MAX_OVERLAY_SIDE, MAX_OVERLAY_SIDE))

    heatmap_image = Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    ).resize(image.size, Image.BILINEAR)

    colours = Image.fromarray(
        jet_colormap(np.asarray(heatmap_image) / 255.0)
    )

    return Image.blend(image, colours, alpha)


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"
