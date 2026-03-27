"""
Accuracy metric for Face Anti-Spoofing model evaluation.

Binary classification accuracy measuring percentage of correct predictions
on both real and fake samples.
"""

import torch
from torch import Tensor
from torchmetrics import Metric


class MyAccuracy(Metric):
    """
    Binary classification accuracy metric.
    
    Measures the proportion of correct predictions out of all predictions.
    Provides an overall assessment of model performance but doesn't account
    for class imbalance.
    
    Formula:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        where:
        - TP: True positives (fake correctly classified as fake)
        - TN: True negatives (real correctly classified as real)
        - FP: False positives (real classified as fake)
        - FN: False negatives (fake classified as real)
    
    Label Encoding:
        - Class 0 (Real): [1, 0]
        - Class 1 (Fake/Attack): [0, 1]
        
    Attributes:
        total_correct (Tensor): Count of correct predictions
        total_samples (Tensor): Total number of predictions
        
    Example:
        >>> from torch import Tensor
        >>> metric = MyAccuracy()
        >>> 
        >>> # Batch 1: 32 samples with various predictions
        >>> preds = torch.tensor([
        ...     [0.9, 0.1],    # pred=real (correct if target=real)
        ...     [0.2, 0.8],    # pred=fake (correct if target=fake)
        ...     [0.7, 0.3],    # pred=real
        ...     # ... more predictions
        ... ])
        >>> target = torch.tensor([
        ...     [1.0, 0.0],    # real
        ...     [0.0, 1.0],    # fake
        ...     [1.0, 0.0],    # real
        ...     # ... more labels
        ... ])
        >>> metric.update(preds, target)
        >>>
        >>> # Get accuracy for evaluation set
        >>> accuracy = metric.compute()
        >>> print(f"Accuracy: {accuracy:.4f}")
        Accuracy: 0.9062
        
    Notes:
        - Accuracy alone can be misleading with imbalanced datasets
        - Consider using with APCER/NPCER/ACER for complete evaluation
        - Range: [0.0, 1.0] where 1.0 is perfect
    """
    
    def __init__(self) -> None:
        """Initialize accuracy metric."""
        super().__init__()
        self.add_state(
            "total_correct",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_samples",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum"
        )
    
    def update(
        self,
        preds: Tensor,
        target: Tensor
    ) -> None:
        """
        Update accuracy metric with batch predictions.
        
        Args:
            preds: Model predictions of shape (batch_size, 2) with
                   format [real_prob, fake_prob] or logits
                   - Value > 0.5 in index 1 → predicted fake
                   - Value > 0.5 in index 0 → predicted real
                   
            target: Ground truth labels of shape (batch_size, 2):
                   - [1.0, 0.0]: Real face (honest presentation)
                   - [0.0, 1.0]: Fake face (presentation attack)
                   
        Raises:
            ValueError: If shapes don't match or last dimension != 2
        """
        if preds.shape != target.shape:
            raise ValueError(
                f"Predictions and targets must have same shape. "
                f"Got preds: {preds.shape}, target: {target.shape}"
            )
        
        if preds.shape[1] != 2:
            raise ValueError(
                f"Expected 2 classes, got {preds.shape[1]}"
            )
        
        # Convert to class indices (0=real, 1=fake)
        preds_class = torch.argmax(preds, dim=1)
        target_class = torch.argmax(target, dim=1)
        
        # Calculate correct predictions
        correct = torch.sum(preds_class == target_class)
        
        # Update running totals
        self.total_correct += correct
        self.total_samples += preds.shape[0]
    
    def compute(self) -> Tensor:
        """
        Compute accuracy value.
        
        Returns:
            Tensor: Accuracy score in range [0, 1]
                   0.0 = all predictions incorrect (worst)
                   1.0 = all predictions correct (best)
                   
        Note:
            Returns 0.0 if no samples have been seen
        """
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.total_correct.device)
        
        return self.total_correct.float() / self.total_samples.float()
