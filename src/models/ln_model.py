"""
Lightning model interface for Face Anti-Spoofing detection.

This module provides the PyTorch Lightning wrapper for FAS models,
handling training, validation, and testing pipeline.
"""

import os
import sys
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim import Adam

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from metrics.apcer import APCER
from metrics.npcer import NPCER
from metrics.acer import ACER
from metrics.accuracy import MyAccuracy
from metrics.recall import MyRecall


class ModelInterface(LightningModule):
    """
    PyTorch Lightning module for Face Anti-Spoofing model training.
    
    Wraps a backbone architecture and handles:
    - Training loop with metrics computation
    - Validation metrics evaluation
    - Test predictions
    - Weight computation for class imbalance
    
    Attributes:
        input_shape (tuple): Input tensor shape (C, H, W)
        num_classes (int): Number of output classes
        backbone (torch.nn.Module): The backbone network
        
    Metrics tracked:
        - APCER: Attack Presentation Classification Error Rate
        - NPCER: Normal Presentation Classification Error Rate
        - ACER: Average Classification Error Rate
        - Accuracy: Overall classification accuracy
        - Recall: True positive rate (sensitivity)
    """
    
    # Class weights for handling data imbalance
    CLASS_WEIGHTS = [0.85, 0.15]  # Fake, Real
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int],
        num_classes: int = 2
    ) -> None:
        """
        Initialize Lightning model.
        
        Args:
            model: Backbone neural network
            input_shape: Input tensor shape (channels, height, width)
            num_classes: Number of output classes (default: 2)
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Initialize training metrics
        self.train_apcer = APCER()
        self.train_npcer = NPCER()
        self.train_acer = ACER()
        self.train_accuracy = MyAccuracy()
        self.train_recall = MyRecall()
        
        # Initialize validation metrics
        self.val_apcer = APCER()
        self.val_npcer = NPCER()
        self.val_acer = ACER()
        self.val_accuracy = MyAccuracy()
        self.val_recall = MyRecall()
        
        self.backbone = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            logits: Network output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def configure_optimizers(self) -> Adam:
        """
        Configure optimizer for training.
        
        Returns:
            Adam optimizer with predefined learning rate and weight decay
        """
        return Adam(self.parameters(), lr=5e-5, weight_decay=5e-4)
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for one batch.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Computed loss for this batch
        """
        loss, outputs, labels = self._common_step(batch, batch_idx)
        
        # Compute metrics
        acc = self.train_accuracy(outputs, labels)
        rec = self.train_recall(outputs, labels)
        apcer = self.train_apcer(outputs, labels)
        npcer = self.train_npcer(outputs, labels)
        acer = self.train_acer(outputs, labels)
        
        # Log metrics
        metrics = {
            "train/loss": loss,
            "train/accuracy": acc,
            "train/recall": rec,
            "train/apcer": apcer,
            "train/npcer": npcer,
            "train/acer": acer
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False
        )
        
        return loss
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step for one batch.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Computed loss for this batch
        """
        loss, outputs, labels = self._common_step(batch, batch_idx)
        
        # Compute metrics
        acc = self.val_accuracy(outputs, labels)
        rec = self.val_recall(outputs, labels)
        apcer = self.val_apcer(outputs, labels)
        npcer = self.val_npcer(outputs, labels)
        acer = self.val_acer(outputs, labels)
        
        # Log metrics
        metrics = {
            "val/loss": loss,
            "val/accuracy": acc,
            "val/recall": rec,
            "val/apcer": apcer,
            "val/npcer": npcer,
            "val/acer": acer
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False
        )
        
        return loss
    
    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Test prediction step.
        
        Args:
            batch: Tuple of (images, labels)
            
        Returns:
            Predicted class indices
        """
        images, labels = batch
        outputs = self.forward(images)
        _, predictions = torch.max(outputs.data, 1)
        return predictions
    
    def _common_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Common step for training and validation.
        
        Handles forward pass and loss computation with class weights.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Tuple of (loss, outputs, labels)
        """
        images, labels = batch
        labels = labels.squeeze(0).float()
        
        # Setup class weights for imbalanced dataset
        weights = torch.FloatTensor(self.CLASS_WEIGHTS).to(self.device)
        
        # Forward pass
        outputs = self.forward(images)
        
        # Compute weighted cross entropy loss
        loss = nn.CrossEntropyLoss(weight=weights)(outputs, labels)
        
        return loss, outputs, labels


def load_model(
    backbone: nn.Module,
    input_shape: Tuple[int, int, int],
    num_classes: int = 2
) -> ModelInterface:
    """
    Factory function to create a Lightning model.
    
    Args:
        backbone: Backbone neural network architecture
        input_shape: Input tensor shape (channels, height, width)
        num_classes: Number of output classes
        
    Returns:
        ModelInterface: Initialized Lightning module
        
    Raises:
        Exception: If model creation fails
        
    Example:
        >>> from src.models.resnext50 import SEResNeXT50
        >>> backbone = SEResNeXT50((3, 224, 224), 2)
        >>> model = load_model(backbone, (3, 224, 224), 2)
    """
    try:
        model = ModelInterface(backbone, input_shape, num_classes)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}") from e
