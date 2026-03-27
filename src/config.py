"""
Configuration module for Face Anti-Spoofing project.
Centralizes all argument parsing and configuration management.
"""

import argparse
from typing import Tuple
import os


class FASConfig:
    """Configuration class for Face Anti-Spoofing models and training."""
    
    # Default input/output dimensions
    DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (3, 224, 224)
    DEFAULT_NUM_CLASSES: int = 2
    DEFAULT_BATCH_SIZE: int = 64
    DEFAULT_MAX_EPOCHS: int = 10
    
    # Supported models
    SUPPORTED_MODELS = ["seresnext50", "mobilenetv3", "feathernet", "shufflenet"]
    
    # Class labels
    CLASS_LABELS = {0: "real", 1: "fake"}
    

def get_training_args() -> argparse.Namespace:
    """
    Parse and return arguments for training pipeline.
    
    Returns:
        argparse.Namespace: Training configuration arguments
    """
    parser = argparse.ArgumentParser(
        description="Train Face Anti-Spoofing models"
    )
    
    # Model arguments
    parser.add_argument(
        "--modelname",
        type=str,
        default="seresnext50",
        choices=FASConfig.SUPPORTED_MODELS,
        help="Model architecture to train"
    )
    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=FASConfig.DEFAULT_INPUT_SHAPE,
        help="Input image shape (C, H, W)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=FASConfig.DEFAULT_NUM_CLASSES,
        help="Number of output classes"
    )
    
    # Data arguments
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
        help="Path to test/evaluation dataset"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=FASConfig.DEFAULT_BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=FASConfig.DEFAULT_MAX_EPOCHS,
        help="Maximum training epochs"
    )
    
    # Logging arguments
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
    
    return parser.parse_args()


def get_inference_args() -> argparse.Namespace:
    """
    Parse and return arguments for inference/prediction.
    
    Returns:
        argparse.Namespace: Inference configuration arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Face Anti-Spoofing inference on images"
    )
    
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "FAS_detector",
                "model",
                "seresnext50_v0.ckpt"
            )
        ),
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "assets",
                "samples",
                "1.png"
            )
        ),
        help="Path to input image for inference"
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="seresnext50",
        choices=FASConfig.SUPPORTED_MODELS,
        help="Model architecture used for inference"
    )
    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=FASConfig.DEFAULT_INPUT_SHAPE,
        help="Input image shape (C, H, W)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=FASConfig.DEFAULT_NUM_CLASSES,
        help="Number of output classes"
    )
    
    return parser.parse_args()


def get_video_args() -> argparse.Namespace:
    """
    Parse and return arguments for video/webcam inference.
    
    Returns:
        argparse.Namespace: Video inference configuration arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Face Anti-Spoofing on video stream"
    )
    
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "FAS_detector",
                "model",
                "seresnext50_v0.ckpt"
            )
        ),
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="seresnext50",
        choices=FASConfig.SUPPORTED_MODELS,
        help="Model architecture used for inference"
    )
    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=FASConfig.DEFAULT_INPUT_SHAPE,
        help="Input image shape (C, H, W)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=FASConfig.DEFAULT_NUM_CLASSES,
        help="Number of output classes"
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera device ID (0 for default webcam)"
    )
    
    return parser.parse_args()
