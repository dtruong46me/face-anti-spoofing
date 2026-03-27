
"""
Normal Presentation Classification Error Rate (NPCER) metric.

NPCER measures the error rate when classifying genuine (normal) presentations.
Lower NPCER is better (fewer false positives on real images).

See: ISO/IEC 30107-3:2017 for metric definitions
"""

import torch
from torch import Tensor
from torchmetrics import Metric


class NPCER(Metric):
    """
    Normal Presentation Classification Error Rate (NPCER).
    
    Measures the proportion of genuine (real) images that are incorrectly
    classified as spoofed (false positives). Used to evaluate Face Anti-Spoofing
    model performance on legitimate presentations.
    
    Formula:
        NPCER = FP_normal / (TN_normal + FP_normal)
        where:
        - FP_normal: False positives (real classified as fake)
        - TN_normal: True negatives (real correctly classified as real)
    
    Label Encoding:
        - Class 0 (Real): [1, 0]
        - Class 1 (Fake/Attack): [0, 1]
        
    Attributes:
        total_normal_samples (Tensor): Total legitimate presentation samples
        total_normal_error (Tensor): Legitimate presentations misclassified as fake
        
    Example:
        >>> from torch import Tensor
        >>> metric = NPCER()
        >>> 
        >>> # Batch 1: 30 real samples
        >>> preds = torch.tensor([[0.9, 0.1], [0.8, 0.2], ...])
        >>> target = torch.tensor([[1.0, 0.0], [1.0, 0.0], ...])  # All real
        >>> metric.update(preds, target)
        >>>
        >>> # Batch 2: 30 real samples
        >>> preds = torch.tensor([[0.7, 0.3], [0.95, 0.05], ...])
        >>> target = torch.tensor([[1.0, 0.0], [1.0, 0.0], ...])  # All real
        >>> metric.update(preds, target)
        >>>
        >>> # Get NPCER for evaluation set
        >>> npcer_value = metric.compute()
        >>> print(f"NPCER: {npcer_value:.4f}")
        NPCER: 0.0333
        
    Notes:
        - NPCER focuses on false alarms (false positives on real)
        - Lower value indicates fewer false rejections of genuine faces
        - Typical range: [0.0, 1.0]
        - Complements APCER for comprehensive evaluation
    """
    
    def __init__(self) -> None:
        """Initialize Normal Presentation Classification Error Rate metric."""
        super().__init__()
        self.add_state(
            "total_normal_samples",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_normal_error",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum"
        )
    
    def update(
        self,
        preds: Tensor,
        target: Tensor
    ) -> None:
        """
        Update NPCER metric with batch predictions.
        
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
        
        # Calculate metrics for normal presentations (target_class == 0)
        normal_mask = (target_class == 0)
        
        true_negative_normal = torch.sum(
            (preds_class == 0) & normal_mask
        )
        false_positive_normal = torch.sum(
            (preds_class == 1) & normal_mask
        )
        
        # Update running totals
        self.total_normal_error += false_positive_normal
        self.total_normal_samples += (true_negative_normal + false_positive_normal)
    
    def compute(self) -> Tensor:
        """
        Compute NPCER value.
        
        Returns:
            Tensor: NPCER score in range [0, 1]
                   0.0 = perfect real detection (no false alarms)
                   1.0 = all real faces rejected (worst)
                   
        Note:
            Returns 0.0 if no legitimate samples in evaluation set
        """
        if self.total_normal_samples == 0:
            return torch.tensor(0.0, device=self.total_normal_error.device)
        
        return self.total_normal_error.float() / self.total_normal_samples.float()
