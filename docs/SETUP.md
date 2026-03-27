# 📦 Installation & Setup Guide

Complete step-by-step guide to install, configure, and run the Face Anti-Spoofing project.

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Directory Setup](#directory-setup)
- [Installation Methods](#installation-methods)
- [Model Setup](#model-setup)
- [Dataset Setup](#dataset-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

### Hardware
- **CPU**: 4+ cores
- **RAM**: 8GB minimum (16GB recommended)
- **GPU** (Optional): NVIDIA GPU with CUDA 11.0+ (recommended for training)
- **Disk**: 20GB free space (including models and datasets)

### Software
- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Git**: For cloning repository
- **CUDA** (Optional): 11.0+ for GPU acceleration

### Check Your System

```bash
# Check Python version
python --version  # Should be 3.8+

# Check GPU (if available)
nvidia-smi  # Shows NVIDIA GPU info

# Check available storage
df -h  # Linux/macOS
wmic logicaldisk get name,size,freespace  # Windows
```

---

## 🗂️ Directory Setup

### 1. Clone Repository

```bash
# Clone from GitHub
git clone https://github.com/dtruong46me/face-anti-spoofing.git
cd face-anti-spoofing

# Or download ZIP and extract
```

### 2. Create Required Directories

```bash
# Create model directory
mkdir -p FAS_detector/model

# Create data directory
mkdir -p data/train/real data/train/fake
mkdir -p data/valid/real data/valid/fake
mkdir -p data/test/real data/test/fake

# Create sample directory
mkdir -p assets/samples

# Create checkpoint directory
mkdir -p checkpoint
```

**Resulting Structure:**
```
face-anti-spoofing/
├── FAS_detector/
│   └── model/                    # ← Place model checkpoints here
├── data/                         # ← Place dataset here
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── valid/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── assets/
│   └── samples/                  # ← Place test images here
├── checkpoint/                   # ← Saved model checkpoints
├── src/                          # Source code
├── docs/                         # Documentation
└── README.md
```

---

## 💻 Installation Methods

### Method 1: Using Conda (Recommended)

```bash
# Step 1: Create conda environment
conda create -n fas python=3.10
conda activate fas

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import lightning; print('Lightning installed')"
```

### Method 2: Using Python venv

```bash
# Step 1: Create virtual environment
python -m venv fas_env

# Step 2: Activate environment
# Windows
fas_env\Scripts\activate
# Linux/macOS
source fas_env/bin/activate

# Step 3: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Method 3: Using pip directly

```bash
# Install dependencies globally
pip install -r requirements.txt

# Note: Not recommended - use virtual environment
```

### Method 4: Using Docker

```bash
# Build Docker image (optional)
docker build -t fas:latest .

# Run container
docker run --gpus all -v $(pwd):/workspace fas:latest
```

### Verify Installation

```bash
# Run quick test
python -c "
import torch
import lightning
from src.config import FASConfig
print('✓ PyTorch:', torch.__version__)
print('✓ Lightning installed')
print('✓ Supported models:', FASConfig.SUPPORTED_MODELS)
"
```

---

## 🤖 Model Setup

### Download Pre-trained Models

Models are available from Kaggle. Choose one or multiple:

#### Option 1: Automatic Download (via script)

Create `download_models.py`:

```python
import os
import urllib.request

models = {
    "feathernet.ckpt": "",  # Add URL
    "mobilenetv3.ckpt": "", # Add URL  
    "seresnext50.ckpt": ""  # Add URL
}

os.makedirs("FAS_detector/model", exist_ok=True)

for model_name, url in models.items():
    if url:
        path = f"FAS_detector/model/{model_name}"
        print(f"Downloading {model_name}...")
        urllib.request.urlretrieve(url, path)
        print(f"✓ Downloaded to {path}")
```

#### Option 2: Manual Download

1. Visit Kaggle Models:
   - [FeatherNet](https://www.kaggle.com/models/dtruon46/feathernet-face-anti-spoofing)
   - [MobileNetV3](https://www.kaggle.com/models/dtruon46/mobilenetv3-face-anti-spoofing)
   - [SEResNeXT50](https://www.kaggle.com/models/dtruon46/resnext50-face-anti-spoofing)

2. Download `.ckpt` files

3. Place in `FAS_detector/model/` directory

**Verify models:**
```bash
ls -la FAS_detector/model/
# Should show:
# - feathernet.ckpt (4.3 MB)
# - mobilenetv3.ckpt (28 MB)
# - seresnext50.ckpt (106 MB)
```

---

## 📊 Dataset Setup

### Download LCC-FASD Dataset

#### Step 1: Download Dataset

**From Kaggle:**
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle account & API token)
kaggle datasets download -d faber24/lcc-fasd

# Extract
unzip lcc-fasd.zip
```

**Or download manually:**
- Visit https://www.kaggle.com/datasets/faber24/lcc-fasd
- Download and extract ZIP file

#### Step 2: Organize Data

```bash
# Extract downloaded data
unzip lcc-fasd.zip

# Create proper directory structure
mkdir -p data/train data/valid data/test

# Move files (adjust based on actual structure)
# From LCC-FASD-training:
cp -r LCC_FASD/LCC_FASD_training/real data/train/
cp -r LCC_FASD/LCC_FASD_training/fake data/train/

# From LCC_FASD_evaluation or development:
cp -r LCC_FASD/LCC_FASD_evaluation/real data/valid/
cp -r LCC_FASD/LCC_FASD_evaluation/fake data/valid/
cp -r LCC_FASD/LCC_FASD_development/real data/test/
cp -r LCC_FASD/LCC_FASD_development/fake data/test/
```

#### Step 3: Verify Dataset

```bash
python -c "
import os
train_real = len(os.listdir('data/train/real'))
train_fake = len(os.listdir('data/train/fake'))
valid_real = len(os.listdir('data/valid/real'))
valid_fake = len(os.listdir('data/valid/fake'))

print('Training:', train_real+train_fake, 
      f'(Real: {train_real}, Fake: {train_fake})')
print('Validation:', valid_real+valid_fake,
      f'(Real: {valid_real}, Fake: {valid_fake})')
"
```

---

## 🧪 Test Images Setup

### Get Sample Images for Testing

#### Option 1: From Dataset

Copy a few images from your downloaded dataset:

```bash
# Copy some validation images for testing
cp data/valid/real/*.jpg assets/samples/
cp data/valid/fake/*.jpg assets/samples/
```

#### Option 2: Use Your Own Images

```bash
# Copy your test images to assets/samples/
cp /path/to/your/face/image.jpg assets/samples/
cp /path/to/another/image.jpg assets/samples/
```

All formats supported: `.jpg`, `.jpeg`, `.png`, `.bmp`

---

## ✅ Verification

### 1. Test Installation

```bash
# Verify all imports work
python verify_install.py
```

Create `verify_install.py`:
```python
"""Verify installation of all components."""

def verify_packages():
    """Check if all packages are installed."""
    packages = [
        ('torch', 'PyTorch'),
        ('lightning', 'PyTorch Lightning'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('sklearn', 'scikit-learn'),
        ('wandb', 'Weights & Biases'),
    ]
    
    print("📦 Checking Packages...")
    for pkg, name in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - Not installed")
    
def verify_directories():
    """Check if required directories exist."""
    import os
    dirs = [
        'FAS_detector/model',
        'data',
        'assets/samples',
        'checkpoint',
    ]
    
    print("\n📁 Checking Directories...")
    for d in dirs:
        if os.path.exists(d):
            print(f"  ✓ {d}")
        else:
            print(f"  ✗ {d} - Missing")

def verify_models():
    """Check if models are downloaded."""
    import os
    model_dir = 'FAS_detector/model'
    models = ['feathernet.ckpt', 'mobilenetv3.ckpt', 'seresnext50.ckpt']
    
    print("\n🤖 Checking Models...")
    for model in models:
        path = os.path.join(model_dir, model)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**2)  # MB
            print(f"  ✓ {model} ({size:.1f} MB)")
        else:
            print(f"  ✗ {model} - Not found")

if __name__ == "__main__":
    verify_packages()
    verify_directories()
    verify_models()
    print("\n" + "="*50)
    print("Verification complete!")
```

Run verification:
```bash
python verify_install.py
```

### 2. Test Model Loading

```bash
python -c "
from src.predictor import FASPredictor
import os

model_path = 'FAS_detector/model/feathernet.ckpt'
if os.path.exists(model_path):
    print('Loading model...')
    predictor = FASPredictor(model_path, 'feathernet')
    print('✓ Model loaded successfully')
else:
    print('✗ Model not found')
"
```

### 3. Test Image Prediction

```bash
# Single image test
python predict_sample.py \
    --modelname "feathernet" \
    --image "assets/samples/sample_image.jpg"
```

### 4. Test GPU (if available)

```bash
python -c "
import torch
if torch.cuda.is_available():
    print('✓ GPU Available')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA Version: {torch.version.cuda}')
    print(f'  cuDNN Version: {torch.backends.cudnn.version()}')
else:
    print('⚠ GPU Not Available - Using CPU')
"
```

---

## 🚀 Quick Start After Setup

### 1. Single Image Prediction

```bash
python predict_sample.py \
    --modelname "feathernet" \
    --model_checkpoint "FAS_detector/model/feathernet.ckpt" \
    --image "assets/samples/test.jpg"
```

### 2. Real-time Webcam Detection

```bash
python run_video_inference.py \
    --modelname "feathernet" \
    --camera_id 0
```

Press `q` to quit, `s` to save current frame.

### 3. Training

```bash
python run_training.py \
    --modelname "feathernet" \
    --train_path "data/train" \
    --test_path "data/valid" \
    --batch_size 64 \
    --max_epochs 30
```

### 4. Batch Inference

Create `batch_predict.py`:
```python
from src.predictor import FASPredictor
import glob

predictor = FASPredictor("FAS_detector/model/feathernet.ckpt", "feathernet")

# Get all images
images = glob.glob("assets/samples/*.jpg")

# Predict
results = predictor.batch_predict(images)

# Print results
for img_path, result in zip(images, results):
    print(f"{img_path}: {result['label']} ({result['probability']:.2%})")
```

Run:
```bash
python batch_predict.py
```

---

## 🔧 Environment Variables

### GPU Configuration

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Disable GPU
export CUDA_VISIBLE_DEVICES=""

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2
```

### Performance Tuning

```bash
# Set number of worker threads
export OMP_NUM_THREADS=8

# Disable OpenMP warnings
export KMP_DUPLICATE_LIB_OK=True
```

---

## ❌ Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install torch torchvision torchaudio
```

### Issue: CUDA out of memory

**Solution:**
```bash
# Reduce batch size
python run_training.py --batch_size 32  # Instead of 64

# Or use CPU only
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Model checkpoint not found

**Solution:**
```bash
# Check model path
ls -la FAS_detector/model/

# Use correct path
python predict_sample.py \
    --model_checkpoint "FAS_detector/model/feathernet.ckpt" \
    --image "assets/samples/test.jpg"
```

### Issue: Dataset directory structure wrong

**Solution:**
```bash
# Verify structure
find data -type d
# Should show:
# data/train/real
# data/train/fake
# data/test/real
# data/test/fake
```

### Issue: Python version incompatible

**Solution:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Create new environment with correct version
conda create -n fas python=3.10
conda activate fas
pip install -r requirements.txt
```

### Issue: Port already in use (TensorBoard/WandB)

**Solution:**
```bash
# Use different port for TensorBoard
tensorboard --logdir=lightning_logs --port=6007
```

---

## 📝 Configuration Files

### Default Paths

Edit `src/config.py` to change default paths:

```python
# Default model checkpoint
DEFAULT_MODEL = "FAS_detector/model/feathernet.ckpt"

# Default data paths
DEFAULT_TRAIN_PATH = "data/train"
DEFAULT_TEST_PATH = "data/test"

# Default sample image
DEFAULT_SAMPLE_IMAGE = "assets/samples/sample.jpg"
```

### Custom Configuration

Create `config.json`:
```json
{
    "modelname": "feathernet",
    "batch_size": 64,
    "max_epochs": 40,
    "input_shape": [3, 224, 224],
    "num_classes": 2,
    "train_path": "data/train",
    "test_path": "data/valid",
    "model_checkpoint": "FAS_detector/model/feathernet.ckpt"
}
```

---

## ✨ Next Steps

1. ✅ Complete setup verification
2. 📊 Download and organize dataset
3. 🤖 Download pre-trained models  
4. 🚀 Run first prediction
5. 📖 Read main README.md for usage
6. 🔬 Start training with your data

---

## 📞 Getting Help

- See [README.md](../README.md) for overview
- Check [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) for API reference
- Review [CONTRIBUTING.md](./CONTRIBUTING.md) for development
- Submit issues on GitHub with detailed description

---

**Last Updated**: March 2026  
**Version**: 1.0
