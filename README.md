# Face Anti-Spoofing Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive deep learning framework for **Face Anti-Spoofing (FAS)** detection, comparing state-of-the-art lightweight models against traditional approaches. This project evaluates model performance on the LCC-FASD dataset using various metrics including APCER, NPCER, and ACER.

```
    ______                   ___          __  _    _____                   _____
   / ____/___ _________     /   |  ____  / /_(_)  / ___/____  ____  ____  / __(_)___  ____ _
  / /_  / __ `/ ___/ _ \   / /| | / __ \/ __/ /   \__ \/ __ \/ __ \/ __ \/ /_/ / __ \/ __ `/
 / __/ / /_/ / /__/  __/  / ___ |/ / / / /_/ /   ___/ / /_/ / /_/ / /_/ / __/ / / / / /_/ /
/_/    \__,_/\___/\___/  /_/  |_/_/ /_/\__/_/   /____/ .___/\____/\____/_/ /_/_/ /_/\__, /
                                                    /_/                            /____/
```

## 📋 Table of Contents

- [Overview](#overview)
- [Documentation](#-documentation)
- [Features](#features)
- [Supported Models](#supported-models)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

Face Anti-Spoofing (FAS) detection is important for security of biometric authentication systems. This project aims to train an effectiveness of lightweight neural networks to distinguish between genuine faces and spoofing attacks (e.g., printed photos, videos, masks, digital displays).

**Key Objective**: Compare efficiency and detection accuracy between different models.

- **Input**: Image containing a face
- **Output**: Binary classification (Real or Fake)
- **Dataset**: LCC-FASD (Learning with Conditional Constraint - Face Anti-Spoofing Database)

## Documentation

This project includes comprehensive documentation in the `docs/` folder:

| Document | Purpose |
|----------|---------|
| **[docs/INDEX.md](docs/INDEX.md)** | **Documentation Navigation** - Start here for all docs |
| **[docs/SETUP.md](docs/SETUP.md)** | **Installation & Setup** - Step-by-step guide, troubleshooting |
| **[docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)** | **API Reference** - All modules, classes, and examples |
| **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** | **Contributing** - Code standards, PR process, testing |

**Quick navigation:**
- **First time?** → Start with [docs/SETUP.md](docs/SETUP.md)
- **Need API help?** → Check [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)
- **Want to contribute?** → Read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- **Looking for something?** → Browse [docs/INDEX.md](docs/INDEX.md)

## Features

- ✅ **Multi-model Support**: SEResNeXT50, MobileNetV3, FeatherNet, ShuffleNet
- ✅ **Real-time Detection**: Webcam-based video stream processing
- ✅ **Comprehensive Metrics**: APCER, NPCER, ACER, Accuracy, Recall
- ✅ **Weights & Biases Integration**: Experiment tracking and logging
- ✅ **PyTorch Lightning**: Lightning-fast training with callbacks
- ✅ **Face Detection**: Integrated FaceBoxes detector
- ✅ **Batch Inference**: Process multiple images efficiently
- ✅ **Pre-trained Models**: Download and use pre-trained weights

## Supported Models

| Model | Type | Size | Speed | Accuracy |
|-------|------|------|-------|----------|
| **SEResNeXT50** | Large | 106 MB | Slower | Highest |
| **MobileNetV3** | Medium | 28 MB | Medium | High |
| **FeatherNet** | Lightweight | 4.3 MB | Fast | Good |
| **ShuffleNet** | Lightweight | Small | Fast | Good |

## Performance Results

Results evaluated on LCC-FASD Development set after preprocessing:

| Model | APCER ↓ | NPCER ↓ | ACER ↓ |
|-------|---------|---------|---------|
| SEResNeXT50 | 0.1308 | 0.3037 | 0.2127 |
| MobileNetV3 | 0.1727 | 0.2111 | 0.1917 |
| **FeatherNet** | **0.1994** | **0.1284** | **0.1639** |

*Lower is better for all metrics. FeatherNet shows best balance of accuracy and efficiency.*

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/dtruong46me/face-anti-spoofing.git
cd face-anti-spoofing
```

### Step 2: Install Dependencies

**Using Conda (Recommended):**
```bash
conda create -n fas python=3.9
conda activate fas
bash setup.sh
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import lightning; print(f'Lightning installed')"
```

## Quick Start

### 1. **Single Image Prediction**

```bash
python predict_sample.py \
    --modelname "feathernet" \
    --model_checkpoint "FAS_detector/model/feathernet.ckpt" \
    --image "assets/samples/test_image.jpg"
```

**Output:**
```
============================================================
PREDICTION RESULT
============================================================
Label: REAL
Confidence: 98.45%
Class: 0
============================================================
```

### 2. **Real-time Webcam Detection**

```bash
python run_video_inference.py \
    --modelname "feathernet" \
    --model_checkpoint "FAS_detector/model/feathernet.ckpt" \
    --camera_id 0
```

Press `q` to quit, `s` to save current frame.

### 3. **Train Model**

```bash
python run_training.py \
    --modelname "feathernet" \
    --train_path "data/train" \
    --test_path "data/test" \
    --batch_size 128 \
    --max_epochs 40 \
    --wandb_token "your_wandb_token"
```

## Usage Guide

### Downloading Pre-trained Models

Download model weights and place them in `FAS_detector/model/`:

```bash
# Feathernet (4.3 MB)
wget https://example.com/feathernet.ckpt -O FAS_detector/model/feathernet.ckpt

# MobileNetV3 (28 MB)
wget https://example.com/mobilenetv3.ckpt -O FAS_detector/model/mobilenetv3.ckpt

# SEResNeXT50 (106 MB)
wget https://example.com/seresnext50.ckpt -O FAS_detector/model/seresnext50.ckpt
```

Or download directly from:
- [FeatherNet](https://www.kaggle.com/models/dtruon46/feathernet-face-anti-spoofing)
- [MobileNetV3](https://www.kaggle.com/models/dtruon46/mobilenetv3-face-anti-spoofing)
- [SEResNeXT50](https://www.kaggle.com/models/dtruon46/resnext50-face-anti-spoofing)

### Programmatic Usage

```python
from src.predictor import FASPredictor
from PIL import Image

# Initialize predictor
predictor = FASPredictor(
    model_checkpoint="FAS_detector/model/feathernet.ckpt",
    modelname="feathernet"
)

# Single image prediction
image = Image.open("test.jpg")
result = predictor.predict(image)

print(f"Label: {result['label']}")
print(f"Confidence: {result['probability']:.2%}")

# Batch prediction
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = predictor.batch_predict(images)
for r in results:
    print(r['label'], r['probability'])
```

### Training Configuration

Create a `config.json` for custom training settings:

```json
{
    "modelname": "feathernet",
    "batch_size": 128,
    "max_epochs": 40,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "input_shape": [3, 224, 224],
    "num_classes": 2
}
```

## 📁 Project Structure

```
face-anti-spoofing/
├── src/                          # Main source code
│   ├── config.py                 # Configuration management
│   ├── predictor.py              # FAS predictor class
│   ├── utils.py                  # Utility functions
│   ├── data/                     # Data loading and preprocessing
│   │   ├── dataset.py
│   │   ├── load_data.py
│   │   └── visualize_data.py
│   ├── models/                   # Model architectures
│   │   ├── feathernet.py
│   │   ├── mobilenet.py
│   │   ├── resnext50.py
│   │   └── ln_model.py
│   ├── metrics/                  # Evaluation metrics
│   │   ├── apcer.py
│   │   ├── npcer.py
│   │   ├── acer.py
│   │   └── accuracy.py
│   └── pipelines/                # Training/evaluation pipelines
│       ├── training_pipeline.py
│       └── evaluation_pipeline.py
│
├── FAS_detector/                 # Face detection module
│   ├── main.py                   # Entry point for video detection
│   ├── face_detector.py
│   └── face_det/                 # FaceBoxes implementation
│
├── assets/                       # Sample images and files
│   └── samples/
│
├── run_training.py               # Training script
├── run_inference.py              # Inference utility (FLOPs/speed)
├── run_video_inference.py        # Video stream inference
├── predict_sample.py             # Single image prediction
├── requirements.txt
├── README.md
└── LICENSE
```

## ⚙️ Configuration

### Environment Variables

```bash
# Set GPU ID
export CUDA_VISIBLE_DEVICES=0

# Set number of threads
export OMP_NUM_THREADS=8
```

### Training Arguments

```
--modelname             Model architecture [seresnext50|mobilenetv3|feathernet|shufflenet]
--train_path           Path to training dataset
--test_path            Path to test dataset
--batch_size           Batch size (default: 64)
--max_epochs           Maximum training epochs (default: 10)
--wandb_token          Weights & Biases API token
--wandb_runname        W&B experiment name
--input_shape          Input image shape, e.g., (3,224,224)
--num_classes          Number of output classes (default: 2)
```

## 📚 Dataset

### LCC-FASD Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/faber24/lcc-fasd):

```bash
# Extract to project directory
unzip cv-dataset.zip
mv cv-dataset/ face-anti-spoofing/data/
```

**Dataset Structure:**
```
data/
├── train/
│   ├── fake/
│   └── real/
├── valid/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

## 🔬 Evaluation Metrics

- **APCER** (Attack Presentation Classification Error Rate): Error rate on attack images
- **NPCER** (Normal Presentation Classification Error Rate): Error rate on genuine images  
- **ACER** (Average Classification Error Rate): Average of APCER and NPCER
- **Accuracy**: True positives and true negatives ratio
- **Recall**: True positive rate (sensitivity)

## 🛠️ Development

### Code Style

This project follows PEP 8 guidelines. Format code using:

```bash
pip install black flake8
black src/ --line-length 100
flake8 src/ --max-line-length 100
```

### Type Checking

```bash
pip install mypy
mypy src/
```

### Running Tests

```bash
pytest tests/ -v
```

## 📝 Model Conversion

Convert models to ONNX for deployment:

```bash
python scripts/convert_to_onnx.py \
    --model_checkpoint "model.ckpt" \
    --modelname "feathernet" \
    --output "model.onnx"
```

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and conventions
- Pull request process
- Issue reporting
- Testing requirements

## 👥 Team

**Supervisor**: Prof. Dang Tuan Linh

**Group Members**:

| # | Name | Student ID | Email |
|---|------|-----------|-------|
| 1 | Vu Tuan Minh (Leader) | 20210597 | minh.vt210597@sis.hust.edu.vn |
| 2 | Nguyen Tien Doanh | 20214881 | doanh.nt214881@sis.hust.edu.vn |
| 3 | Nguyen Tung Luong | 20214913 | luong.nt214913@sis.hust.edu.vn |
| 4 | Phan Dinh Truong | 20214937 | truong.pd214937@sis.hust.edu.vn |

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Face Anti-Spoofing Survey](https://arxiv.org/abs/1912.00596)
- [LCC-FASD Dataset Paper](https://arxiv.org/abs/1903.10010)

## 🙏 Acknowledgments

- Original FaceBoxes implementation authors
- PyTorch and PyTorch Lightning communities
- Dataset creators and researchers

## ❓ FAQ

**Q: Can I use this for real-world deployment?**  
A: Yes, FeatherNet model is optimized for edge deployment with minimal computational requirements.

**Q: What GPU do I need?**  
A: Any NVIDIA GPU with CUDA support works. GPU-less CPU-based inference is also supported but slower.

**Q: How do I improve model accuracy?**  
A: Try data augmentation, using SEResNeXT50, increasing training epochs, or ensemble methods.

**Q: Is the code production-ready?**  
A: The codebase is well-structured but should undergo security audits before production deployment.

---

**Last Updated**: March 2026  
**Version**: 2.0
