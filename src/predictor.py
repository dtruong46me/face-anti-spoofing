"""
FAS Predictor module for running inference with Face Anti-Spoofing models.
Handles image preprocessing, model loading, and prediction.
"""

import sys
import os
import argparse
from typing import Dict, Any, Optional

import torch
from PIL import Image
from torchvision import transforms

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from models.ln_model import ModelInterface
from utils import load_backbone


class FASPredictor:
    """
    Face Anti-Spoofing predictor for running inference on images.
    
    Attributes:
        model: Loaded PyTorch Lightning model
        device: Computing device (CPU or GPU)
        preprocess: Image preprocessing pipeline
        num_classes: Number of output classes
    """
    
    def __init__(
        self,
        model_checkpoint: str,
        modelname: str,
        input_shape: tuple = (3, 224, 224),
        num_classes: int = 2,
        device: Optional[torch.device] = None
    ):
        """
        Initialize FAS predictor.
        
        Args:
            model_checkpoint: Path to model checkpoint file
            modelname: Name of the model architecture
            input_shape: Input image shape (C, H, W)
            num_classes: Number of output classes
            device: Computing device (auto-detected if None)
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If modelname is not supported
        """
        if not os.path.exists(model_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {model_checkpoint}")
        
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Setup preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize([input_shape[1], input_shape[2]]),
            transforms.ToTensor()
        ])
        
        # Load model backbone
        class Args:
            def __init__(self, modelname, input_shape, num_classes):
                self.modelname = modelname
                self.input_shape = input_shape
                self.num_classes = num_classes
        
        args = Args(modelname, input_shape, num_classes)
        backbone = load_backbone(args)
        
        # Load checkpoint
        self.model = ModelInterface.load_from_checkpoint(
            model_checkpoint,
            model=backbone,
            input_shape=input_shape,
            num_classes=num_classes
        )
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image object (RGB)
            
        Returns:
            Dictionary containing:
                - 'class': Predicted class (0=real, 1=fake)
                - 'label': Predicted label string ('real' or 'fake')
                - 'probability': Confidence score for predicted class
                - 'probabilities': Full probability distribution
                - 'tensor': One-hot tensor representation
        """
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get predictions
        predicted_class = torch.argmax(probabilities, dim=1).item()
        probs = probabilities.squeeze().cpu().numpy()
        
        # Format result
        label = "real" if predicted_class == 0 else "fake"
        one_hot = [1, 0] if predicted_class == 0 else [0, 1]
        
        return {
            'class': predicted_class,
            'label': label,
            'probability': float(probs[predicted_class]),
            'probabilities': probs.tolist(),
            'tensor': one_hot
        }
    
    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        return self.predict(image)
    
    def batch_predict(self, images: list) -> list:
        """
        Run inference on multiple images.
        
        Args:
            images: List of PIL Image objects or file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for img in images:
            if isinstance(img, str):
                result = self.predict_from_path(img)
            else:
                result = self.predict(img)
            results.append(result)
        return results
