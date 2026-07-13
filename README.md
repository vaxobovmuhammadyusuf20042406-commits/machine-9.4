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
- Object recognition
- Confidence score prediction
- Top-5 predictions
- Modern Flask web interface
- Speech output support
- Model comparison

---

## Dataset

COIL-100 Dataset

- 100 object categories
- 7200 images
- 72 viewpoints per object

Image size

224 × 224 pixels

---

## Deep Learning Models

### Custom CNN

Top-1 Accuracy

88.91%

---

### EfficientNetV2-S

Top-1 Accuracy

99.82%

---

### ConvNeXt-Tiny

Top-1 Accuracy

100%

---

## Technologies

Python

PyTorch

Flask

HTML

CSS

JavaScript

OpenCV

Pillow

NumPy

---

## Installation

```bash
pip install -r requirements.txt
```

Run

```bash
python app.py
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
