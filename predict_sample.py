"""
Sample prediction script for Face Anti-Spoofing models.
Demonstrates inference on a single image with visualization.
"""

import sys
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from src.predictor import FASPredictor
from src.config import get_inference_args


def predict_sample(args: argparse.Namespace) -> dict:
    """
    Run inference on a single sample image.
    
    Args:
        args: Inference configuration arguments
        
    Returns:
        Dictionary containing prediction results
    """
    # Initialize predictor
    predictor = FASPredictor(
        model_checkpoint=args.model_checkpoint,
        modelname=args.modelname,
        input_shape=args.input_shape,
        num_classes=args.num_classes
    )
    
    # Run prediction
    result = predictor.predict_from_path(args.image)
    
    return result


def visualize_result(image_path: str, result: dict) -> None:
    """
    Visualize prediction result with the input image.
    
    Args:
        image_path: Path to input image
        result: Prediction result dictionary
    """
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(
        f"Prediction: {result['label'].upper()} "
        f"(confidence: {result['probability']:.2%})"
    )
    plt.axis("off")
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.splitext(image_path)[0] + "_prediction.jpg"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Main entry point for sample prediction."""
    args = get_inference_args()
    
    # Validate input image exists
    if not os.path.exists(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        return
    
    print("[INFO] Running Face Anti-Spoofing inference...")
    result = predict_sample(args)
    
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Label: {result['label'].upper()}")
    print(f"Confidence: {result['probability']:.2%}")
    print(f"Class: {result['class']}")
    print(f"Full probabilities: {result['probabilities']}")
    print("="*50 + "\n")
    
    # Visualize result
    visualize_result(args.image, result)


if __name__ == "__main__":
    main()
