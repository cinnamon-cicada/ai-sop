# Construction Task Classifier

A Vision Transformer (ViT) based deep learning model for classifying construction site activities from images.

Imagine you’re onsite, laying foundations for a house. What if you did not have to walk ten seconds across the site to fetch your blueprint—for the twentieth time? That’s what we hope to use spatial intelligence for: smart glasses that read you the exact SOP you need.

Streamed input from the glasses is sent to a server, where ML-based processing happens. The correct SOP is sent back to the glasses, allowing instant, easy access to instructions on the field. 

## Overview

This project uses a fine-tuned Vision Transformer to identify different construction tasks:
- Installing LED lights
- Laying bricks
- Pouring concrete

## Project Structure

```
├── main.py              # Main training and inference pipeline
├── training.py          # Model architecture and training logic
├── inference.py         # Inference predictor class
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
├── construction_dataset/
│   ├── train/          # Training images organized by class
│   └── val/            # Validation images organized by class
├── demo_dataset/       # Images used for demo
└── output/             # Output directory for images fetched via API
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset in the `construction_dataset/` directory with train/val splits.

## Usage

### Training and Inference

Run the main script to train the model (if not already trained) and perform inference:

```bash
python main.py
```

The script will:
- Train a new model if `best_construction_model.pth` doesn't exist
- Save the best model based on validation accuracy
- Run inference on demo images

### Streamlit Web Application

Launch the interactive AI-SOP (Standard Operating Procedure) web app:

```bash
streamlit run app.py
```

The web app features:
- **Image Gallery**: Browse through demo images with thumbnail navigation
- **AI Task Detection**: Identify construction tasks with confidence scores
- **Detected SOPs**: Get detailed safety procedures, required PPE, tools, and step-by-step instructions for the current image by clicking on "Get SOP"
- **Interactive UI**: Modern interface with real-time predictions

### Custom Inference

```python
from inference import ConstructionTaskPredictor

# Initialize predictor
predictor = ConstructionTaskPredictor(
    model_path='best_construction_model.pth',
    classes=['installing_led_lights', 'laying_bricks', 'pouring_concrete'],
    device='cuda'
)

# Predict single image
result = predictor.predict('path/to/image.jpg')
print(f"Task: {result['task']}, Confidence: {result['confidence']:.2%}")
```

## Model Architecture

- **Backbone**: Vision Transformer (ViT-B/16)
- **Pre-training**: ImageNet
- **Optimizer**: AdamW with Cosine Annealing LR
- **Input size**: 224×224 RGB images

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- See `requirements.txt` for full dependencies
