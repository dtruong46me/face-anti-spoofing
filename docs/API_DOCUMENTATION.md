# Face Anti-Spoofing API Documentation

Complete API reference for the Face Anti-Spoofing detection framework.

## 📚 Table of Contents

- [Core Module: `src.predictor`](#core-module-srcpredictor)
- [Configuration: `src.config`](#configuration-srcconfig)
- [Models: `src.models`](#models-srcmodels)
- [Data: `src.data`](#data-srcdata)
- [Metrics: `src.metrics`](#metrics-srcmetrics)
- [Pipelines: `src.pipelines`](#pipelines-srcpipelines)
- [Detection: `FAS_detector`](#detection-fas_detector)
- [Usage Examples](#usage-examples)
- [Error Handling](#error-handling)
- [Performance Tips](#performance-tips)

---

## Core Module: `src.predictor`

### Class: `FASPredictor`

Main class for running Face Anti-Spoofing inference on images.

#### Constructor

```python
FASPredictor(
    model_checkpoint: str,
    modelname: str,
    input_shape: tuple = (3, 224, 224),
    num_classes: int = 2,
    device: Optional[torch.device] = None
)
```

**Parameters:**
- `model_checkpoint` (str): Path to PyTorch Lightning model checkpoint file
- `modelname` (str): Model architecture name. One of: `["seresnext50", "mobilenetv3", "feathernet", "shufflenet"]`
- `input_shape` (tuple): Input tensor shape as (channels, height, width). Default: (3, 224, 224)
- `num_classes` (int): Number of output classes. Default: 2 (real/fake)
- `device` (torch.device, optional): Computing device. Auto-detected if None

**Raises:**
- `FileNotFoundError`: If checkpoint file doesn't exist
- `ValueError`: If modelname is not supported

**Example:**
```python
from src.predictor import FASPredictor

predictor = FASPredictor(
    model_checkpoint="models/feathernet.ckpt",
    modelname="feathernet"
)
```

#### Methods

##### `predict(image: Image.Image) -> Dict[str, Any]`

Run inference on a single PIL Image.

**Parameters:**
- `image` (PIL.Image): RGB image object

**Returns:**
```python
{
    'class': int,           # Predicted class (0=real, 1=fake)
    'label': str,           # Label string ('real' or 'fake')
    'probability': float,   # Confidence score [0-1]
    'probabilities': list,  # Full probability distribution
    'tensor': list         # One-hot encoded output
}
```

**Example:**
```python
from PIL import Image

image = Image.open("test_face.jpg")
result = predictor.predict(image)
print(f"Label: {result['label']}, Confidence: {result['probability']:.2%}")
```

##### `predict_from_path(image_path: str) -> Dict[str, Any]`

Run inference on an image file.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:** Same as `predict()`

**Raises:**
- `FileNotFoundError`: If image file doesn't exist

**Example:**
```python
result = predictor.predict_from_path("assets/samples/1.jpg")
```

##### `batch_predict(images: list) -> list`

Run inference on multiple images efficiently.

**Parameters:**
- `images` (list): List of PIL Images or file paths (strings)

**Returns:** List of prediction dictionaries

**Example:**
```python
images = ["img1.jpg", "img2.jpg", Image.open("img3.jpg")]
results = predictor.batch_predict(images)

for result in results:
    print(f"{result['label']}: {result['probability']:.2%}")
```

---

## Configuration: `src.config`

### Class: `FASConfig`

Global configuration constants for the project.

**Constants:**
```python
DEFAULT_INPUT_SHAPE = (3, 224, 224)
DEFAULT_NUM_CLASSES = 2
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 10

SUPPORTED_MODELS = ["seresnext50", "mobilenetv3", "feathernet", "shufflenet"]

CLASS_LABELS = {0: "real", 1: "fake"}
```

### Function: `get_training_args() -> argparse.Namespace`

Parse command-line arguments for training.

**Returns:** Namespace with training configuration

**Available Arguments:**
```
--modelname              Model architecture
--input_shape            Input shape as tuple
--num_classes            Number of classes
--train_path             Training data path
--test_path              Test data path
--batch_size             Batch size
--max_epochs             Maximum epochs
--wandb_token            W&B API token
--wandb_runname          W&B run name
```

**Example:**
```python
from src.config import get_training_args

args = get_training_args()
print(f"Training {args.modelname} for {args.max_epochs} epochs")
```

### Function: `get_inference_args() -> argparse.Namespace`

Parse arguments for image inference.

**Available Arguments:**
```
--model_checkpoint       Path to checkpoint
--image                  Path to input image
--modelname             Model architecture
--input_shape           Input shape
--num_classes           Number of classes
```

### Function: `get_video_args() -> argparse.Namespace`

Parse arguments for video/webcam inference.

**Available Arguments:**
```
(same as get_inference_args, plus:)
--camera_id             Camera device ID
```

---

## Models: `src.models`

### Class: `SEResNeXT50`

Squeeze-and-Excitation ResNeXT50 model.

```python
from src.models.resnext50 import SEResNeXT50

backbone = SEResNeXT50(
    input_shape=(3, 224, 224),
    num_classes=2
)
```

**Parameters:**
- `input_shape`: Input tensor shape
- `num_classes`: Number of output classes

### Class: `MobileNetV3`

MobileNetV3 lightweight model.

```python
from src.models.mobilenet import MobileNetV3

backbone = MobileNetV3()
```

### Class: `FeatherNetB`

FeatherNet-B ultra-lightweight model.

```python
from src.models.feathernet import FeatherNetB

backbone = FeatherNetB()
```

**Best for:**
- Mobile deployment
- Edge devices
- Real-time processing
- Minimal memory usage

### Class: `ShuffleNet`

ShuffleNet lightweight model.

```python
from src.models.ShuffleNet import ShuffleNet

backbone = ShuffleNet()
```

### Class: `ModelInterface`

Lightning module wrapping backbone models for training.

**Key Methods:**
```python
# Load from checkpoint
model = ModelInterface.load_from_checkpoint(
    checkpoint_path,
    model=backbone,
    input_shape=(3, 224, 224),
    num_classes=2
)

# Forward pass
outputs = model(images)  # Returns logits
```

### Function: `load_backbone(args: argparse.Namespace) -> torch.nn.Module`

Factory function to load appropriate backbone model.

```python
from src.utils import load_backbone

args = get_training_args()
backbone = load_backbone(args)
```

---

## Data: `src.data`

### Function: `load_data(args: argparse.Namespace) -> dict`

Load image dataset from filesystem.

**Parameters:**
- `args`: Configuration arguments with train_path and test_path

**Returns:**
```python
{
    'train': ImageDataset,
    'val': ImageDataset,
    'test': ImageDataset
}
```

**Example:**
```python
from src.data.dataset import load_data

data = load_data(args)
print(f"Train samples: {len(data['train'])}")
```

### Function: `load_dataloader(data: dict) -> tuple`

Create PyTorch DataLoaders from datasets.

**Parameters:**
- `data`: Dictionary from `load_data()`

**Returns:**
```python
(train_loader, val_loader, test_loader)
```

**Example:**
```python
from src.data.dataset import load_dataloader

train_loader, val_loader, test_loader = load_dataloader(data)
```

### Function: `load_transform_2() -> torchvision.transforms.Compose`

Get preprocessing transforms for images.

```python
from src.utils import load_transform_2

transform = load_transform_2()
# Resizes to (224, 224) and converts to tensor
tensor = transform(image)
```

---

## Metrics: `src.metrics`

### Class: `APCER`

Attack Presentation Classification Error Rate.

```python
from src.metrics.apcer import APCER

apcer_metric = APCER()
error_rate = apcer_metric(predictions, labels)
```

**Lower is better** - Represents error rate on spoofed images.

### Class: `NPCER`

Normal Presentation Classification Error Rate.

```python
from src.metrics.npcer import NPCER

npcer_metric = NPCER()
error_rate = npcer_metric(predictions, labels)
```

**Lower is better** - Represents error rate on genuine images.

### Class: `ACER`

Average Classification Error Rate (APCER + NPCER) / 2.

```python
from src.metrics.acer import ACER

acer_metric = ACER()
avg_error_rate = acer_metric(predictions, labels)
```

**Lower is better** - Overall error rate metric.

### Class: `MyAccuracy`

Total accuracy scoring.

```python
from src.metrics.accuracy import MyAccuracy

accuracy_metric = MyAccuracy()
acc = accuracy_metric(predictions, labels)
```

### Class: `MyRecall`

Recall (True Positive Rate).

```python
from src.metrics.recall import MyRecall

recall_metric = MyRecall()
recall = recall_metric(predictions, labels)
```

---

## Pipelines: `src.pipelines`

### Function: `training_pipeline(args: argparse.Namespace) -> None`

Complete training and evaluation pipeline.

**Parameters:**
- `args`: Training configuration from `get_training_args()`

**Process:**
1. Loads training, validation, and test data
2. Initializes model backbone
3. Sets up W&B logging (if token provided)
4. Trains model with early stopping
5. Evaluates on test set
6. Prints comprehensive metrics

**Example:**
```python
from src.pipelines.training_pipeline import training_pipeline
from src.config import get_training_args

args = get_training_args()
training_pipeline(args)
```

---

## Detection: `FAS_detector`

### Class: `FASDetector`

Video stream processing for real-time detection.

```python
from FAS_detector.main import FASDetector

detector = FASDetector(
    modelname="feathernet",
    model_checkpoint="FAS_detector/model/feathernet.ckpt"
)
```

#### Methods

##### `detect_and_predict(frame: np.ndarray) -> tuple`

Process single frame for detection and liveness check.

**Parameters:**
- `frame` (np.ndarray): OpenCV frame in BGR format

**Returns:**
```python
(annotated_frame, predictions)
```

##### `run(camera_id: int = 0, save_output: bool = False) -> None`

Run real-time detection on webcam stream.

**Parameters:**
- `camera_id` (int): Camera device ID. Default: 0
- `save_output` (bool): Save final frame to fas_output.jpg

**Keyboard Controls:**
- `q` - Quit application
- `s` - Save current frame

**Example:**
```python
detector.run(camera_id=0)
```

### Class: `FaceDetector`

Face detection using FaceBoxes model.

```python
from FAS_detector.face_detector import FaceDetector

face_detector = FaceDetector()
boxes = face_detector(image_array)  # Returns [x1, y1, x2, y2] coordinates
```

---

## Usage Examples

### Example 1: Single Image Classification

```python
from src.predictor import FASPredictor
from PIL import Image

# Initialize
predictor = FASPredictor(
    model_checkpoint="models/feathernet.ckpt",
    modelname="feathernet"
)

# Predict
image = Image.open("face.jpg")
result = predictor.predict(image)

# Display results
print(f"Detected: {result['label'].upper()}")
print(f"Confidence: {result['probability']:.1%}")
```

### Example 2: Batch Processing

```python
import glob
from src.predictor import FASPredictor

predictor = FASPredictor(
    model_checkpoint="models/feathernet.ckpt",
    modelname="feathernet"
)

# Process multiple images
image_paths = glob.glob("test_images/*.jpg")
results = predictor.batch_predict(image_paths)

# Analyze results
real_count = sum(1 for r in results if r['label'] == 'real')
print(f"Real faces: {real_count}/{len(results)}")
```

### Example 3: Real-time Video Detection

```python
from FAS_detector.main import FASDetector

detector = FASDetector(modelname="feathernet")
detector.run(camera_id=0)
```

### Example 4: Custom Training

```python
from src.config import get_training_args
from src.pipelines.training_pipeline import training_pipeline

# Get configuration
args = get_training_args()

# Customize if needed
args.max_epochs = 50
args.batch_size = 128
args.modelname = "seresnext50"

# Train
training_pipeline(args)
```

---

## Return Value Formats

### Prediction Output

```python
{
    'class': 0,                          # Integer (0 or 1)
    'label': 'real',                     # String ('real' or 'fake')
    'probability': 0.9845,               # Float [0.0-1.0]
    'probabilities': [0.9845, 0.0155],   # List of all class probabilities
    'tensor': [1, 0]                     # One-hot encoding
}
```

### Training Metrics

```python
{
    'apcer': 0.1639,     # Attack Presentation Error Rate
    'npcer': 0.1284,     # Normal Presentation Error Rate
    'acer': 0.1461,      # Average Error Rate
    'accuracy': 0.8539,  # Overall accuracy
    'recall': 0.9215     # True positive rate
}
```

---

## Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Model checkpoint missing | Download pre-trained weights |
| `RuntimeError: CUDA out of memory` | Batch size too large | Reduce batch_size argument |
| `ValueError: Unsupported model` | Invalid modelname | Use supported model names |
| `PIL.UnidentifiedImageError` | Invalid image file | Check image format |

---

## Performance Tips

1. **Batch Processing**: Use `batch_predict()` for multiple images
2. **FeatherNet**: Use for real-time/mobile applications
3. **GPU Utilization**: Set `CUDA_VISIBLE_DEVICES` for multi-GPU
4. **Memory Efficiency**: Reduce `batch_size` if OOM occurs
5. **Speed**: Pre-resize images to target shape before prediction

---

*For more information, see the [main README](../README.md), [Setup Guide](SETUP.md), and [Contributing Guidelines](CONTRIBUTING.md).*
