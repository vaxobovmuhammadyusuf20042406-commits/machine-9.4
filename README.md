# Explainable Daily Object Recognition for Visually Impaired Users

## Overview

This project presents an explainable deep learning system for recognizing everyday household objects using computer vision. The application is designed to assist visually impaired users by identifying objects from uploaded images through a web interface.

Three deep learning models were implemented and compared:

- Custom CNN
- EfficientNetV2-S
- ConvNeXt-Tiny

The application was developed using Flask and PyTorch.

---

## Features

- Image upload
- Object recognition with a choice of three models
- Confidence score prediction
- Top-5 predictions
- Grad-CAM explanation of each prediction
- Speech output support
- Modern Flask web interface

---

## Dataset

COIL-100 Dataset

- 100 object categories
- 7200 images
- 72 viewpoints per object
- Images resized to 224 × 224 pixels
- Split per object: 5000 training, 1100 validation, 1100 test images

---

## Deep Learning Models

Results on the 1100 test images:

| Model            | Top-1 accuracy | Top-5 accuracy |
| ---------------- | -------------- | -------------- |
| ConvNeXt-Tiny    | 100%           | 100%           |
| EfficientNetV2-S | 99.82%         | 100%           |
| Custom CNN       | 88.91%         | 99.82%         |

---

## Technologies

Python, PyTorch, Flask, Pillow, NumPy, Matplotlib, HTML, CSS, JavaScript

---

## Live demo

https://myusuf04-machine-9-4-object-recognition.hf.space

---

## Installation

```bash
git clone https://github.com/vaxobovmuhammadyusuf20042406-commits/machine-9.4.git
cd machine-9.4
python -m venv venv
venv\Scripts\activate        # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Get the trained models

The trained model files are too large for GitHub. They are hosted with the live demo and can be downloaded into the `models` folder:

```bash
pip install huggingface_hub
hf download Myusuf04/machine-9-4-object-recognition --repo-type=space --include "models/*" --local-dir .
```

This is enough to run the web application. The dataset is only needed for training and evaluation.

## Get the data

1. Download COIL-100 from the Columbia Imaging and Vision Laboratory: https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
2. Extract the 7200 PNG images (`obj1__0.png` … `obj100__355.png`) into `data/raw/`.
3. Prepare the dataset:

```bash
python scripts/check_dataset.py
python scripts/preprocess_images.py
python scripts/split_dataset.py
python scripts/check_split.py
```

This creates `data/processed/` and the `data/train/`, `data/val/` and `data/test/` folders.

## Run the web application

```bash
python app.py
```

Then open http://127.0.0.1:7860

The application loads every `*_best.pth` checkpoint found in the `models` folder.

## Scripts

Run all scripts from the project root.

| Script                             | Purpose                                         |
| ---------------------------------- | ----------------------------------------------- |
| `scripts/check_dataset.py`         | Check that the raw dataset is complete          |
| `scripts/preprocess_images.py`     | Resize the raw images to 224 × 224              |
| `scripts/split_dataset.py`         | Split the images into train, validation, test   |
| `scripts/check_split.py`           | Count the images in each split                  |
| `scripts/train_custom_cnn.py`      | Train the Custom CNN                            |
| `scripts/train_efficientnetv2.py`  | Train EfficientNetV2-S                          |
| `scripts/train_convnexttiny.py`    | Train ConvNeXt-Tiny                             |
| `scripts/evaluate_model.py`        | Evaluate the trained models on the test set     |
| `scripts/predict.py`               | Predict one image from the command line         |

Example:

```bash
python scripts/predict.py data/test/obj1/obj1__305.png --model convnexttiny --gradcam explanation.jpg
```

---

## Repository

Contains

- source code
- Flask application
- training scripts
- frontend
- documentation

Large trained model files are excluded because GitHub limits files to 100 MB.

---

## Author

Muhammadyusuf Vakhobov
