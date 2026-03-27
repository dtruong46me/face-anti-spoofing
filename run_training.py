"""
Training script for Face Anti-Spoofing models.
Supports training with multiple model architectures and W&B logging.
"""

import argparse
import os
import sys

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from src.pipelines.training_pipeline import training_pipeline
from src.config import get_training_args


def print_config(args: argparse.Namespace) -> None:
    """
    Print configuration arguments in a formatted way.
    
    Args:
        args: Configuration arguments
    """
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key:<20} = {value}")
    print("="*60 + "\n")


def run_training():
    """Main training entry point."""
    args = get_training_args()
    
    # Print configuration
    print_config(args)
    
    # Run training pipeline
    training_pipeline(args)


if __name__ == "__main__":
    run_training()