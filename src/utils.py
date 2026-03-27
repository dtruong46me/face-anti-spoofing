"""
Utility functions for Face Anti-Spoofing model loading and image preprocessing.

This module provides helper functions for:
- Loading model backbones
- Image preprocessing and transformation
- Argument parsing (legacy - use src.config instead)
"""

import argparse
import os
import sys
from typing import Callable, Type

import torch.nn as nn
from torchvision import transforms

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from models.feathernet import FeatherNetB
from models.resnext50 import SEResNeXT50
from models.mobilenet import MobileNetV3
from models.ShuffleNet import ShuffleNet


# Available model architectures
MODEL_REGISTRY = {
    "seresnext50": SEResNeXT50,
    "mobilenetv3": MobileNetV3,
    "feathernet": FeatherNetB,
    "shufflenet": ShuffleNet
}


def load_transform_2() -> transforms.Compose:
    """
    Get preprocessing transforms for Face Anti-Spoofing models.
    
    Applies standard preprocessing:
    - Resize to 224x224 (standard input size)
    - Convert to tensor
    - Normalize to [0, 1] range
    
    Returns:
        transforms.Compose: Preprocessing pipeline
        
    Example:
        >>> from PIL import Image
        >>> transform = load_transform_2()
        >>> image = Image.open("face.jpg")
        >>> tensor = transform(image)
        >>> print(tensor.shape)
        torch.Size([3, 224, 224])
    """
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])


def load_backbone(args: argparse.Namespace) -> nn.Module:
    """
    Load backbone model architecture based on configuration.
    
    Factory function that selects and initializes the appropriate
    model architecture (SEResNeXT50, MobileNetV3, FeatherNet, or ShuffleNet)
    based on the modelname argument.
    
    Args:
        args: Argument namespace containing:
          - modelname (str): One of ['seresnext50', 'mobilenetv3', 'feathernet', 'shufflenet']
          - input_shape (tuple): Input tensor shape (C, H, W)
          - num_classes (int): Number of output classes
    
    Returns:
        nn.Module: Backbone neural network instance
        
    Raises:
        ValueError: If modelname is not supported
        TypeError: If required arguments are missing
        
    Example:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     modelname="feathernet",
        ...     input_shape=(3, 224, 224),
        ...     num_classes=2
        ... )
        >>> backbone = load_backbone(args)
        >>> print(type(backbone).__name__)
        FeatherNetB
    """
    modelname = args.modelname.lower()
    
    if modelname not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{modelname}' not supported. "
            f"Available models: {available}"
        )
    
    # Get model class
    ModelClass: Type[nn.Module] = MODEL_REGISTRY[modelname]
    
    # Instantiate model
    if modelname == "feathernet":
        # FeatherNet doesn't take input_shape or num_classes
        model = ModelClass()
    elif modelname == "shufflenet":
        # ShuffleNet doesn't take parameters
        model = ModelClass()
    else:
        # SEResNeXT50 and MobileNetV3 take input_shape and num_classes
        model = ModelClass(args.input_shape, args.num_classes)
    
    return model


def load_argparser() -> argparse.Namespace:
    """
    Legacy argument parser (DEPRECATED).
    
    This function is deprecated. Use src.config module instead:
    - get_training_args()
    - get_inference_args()
    - get_video_args()
    
    Returns:
        argparse.Namespace: Parsed arguments
        
    Warning:
        DEPRECATED: This function maintains duplicate argument parsing.
        For new code, use functions from src.config module.
    """
    parser = argparse.ArgumentParser(
        description="Face Anti-Spoofing Training & Inference (DEPRECATED)"
    )
    
    # Model arguments
    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=(3, 224, 224),
        help="Input tensor shape"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of output classes"
    )
    
    # W&B arguments
    parser.add_argument(
        "--wandb_token",
        type=str,
        default="",
        help="Weights & Biases API token"
    )
    parser.add_argument(
        "--wandb_runname",
        type=str,
        default="model1",
        help="Weights & Biases run name"
    )
    
    # Model selection
    parser.add_argument(
        "--modelname",
        type=str,
        default="seresnext50",
        help="Model architecture to use"
    )
    
    # Data paths
    parser.add_argument(
        "--train_path",
        type=str,
        default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_training",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation",
        help="Path to test dataset"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum training epochs"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="/kaggle/working/checkpoint/cvproject.ckpt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to input image for inference"
    )
    
    args = parser.parse_args()
    return args

