
"""
Attack Presentation Classification Error Rate (APCER) metric.

APCER measures the error rate when classifying spoofed (attack) presentations.
Lower APCER is better (fewer false negatives on fake images).

See: ISO/IEC 30107-3:2017 for metric definitions
"""

from typing import Tuple

import torch
from torch import Tensor
from torchmetrics import Metric


class APCER(Metric):
    """
    Attack Presentation Classification Error Rate (APCER).
    
    Measures the proportion of spoofed (fake) images that are incorrectly
    classified as real (false negatives). Used to evaluate Face Anti-Spoofing
    model performance on presentation attacks.
    
    Formula:
        APCER = FN_attacks / (TP_attacks + FN_attacks)
        where:
        - FN_attacks: False negatives (fake classified as real)
        - TP_attacks: True positives (fake correctly classified as fake)
    
    Label Encoding:
        - Class 0 (Real): [1, 0]
        - Class 1 (Fake/Attack): [0, 1]
        
    Attributes:
        total_attack_samples (Tensor): Total attack presentation samples
        total_attack_error (Tensor): Attack presentations misclassified as real
        
    Example:
        >>> from torch import Tensor
        >>> metric = APCER()
        >>> 
        >>> # Batch 1: 30 samples
        >>> preds = torch.tensor([[0.9, 0.1], [0.2, 0.8], ...])  # 32 predictions
        >>> target = torch.tensor([[1.0, 0.0], [0.0, 1.0], ...])  # 32 labels
        >>> metric.update(preds, target)
        >>>
        >>> # Batch 2: 30 samples  
        >>> preds = torch.tensor([[0.8, 0.2], [0.3, 0.7], ...])
        >>> target = torch.tensor([[1.0, 0.0], [0.0, 1.0], ...])
        >>> metric.update(preds, target)
        >>>
        >>> # Get APCER for evaluation set
        >>> apcer_value = metric.compute()
        >>> print(f"APCER: {apcer_value:.4f}")
        APCER: 0.0521
        
    Notes:
        - APCER focuses on attack detection (false negatives on fake)
        - Lower value indicates better attack detection
        - Typical range: [0.0, 1.0]
        - Complements NPCER for comprehensive evaluation
    """
    
    def __init__(self) -> None:
        """Initialize Attack Presentation Classification Error Rate metric."""
        super().__init__()
        self.add_state(
            "total_attack_samples",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_attack_error",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum"
        )
    
    def update(
        self,
        preds: Tensor,
        target: Tensor
    ) -> None:
        """
        Update APCER metric with batch predictions.
        
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
        
        # Calculate metrics for attack presentations (target_class == 1)
        attack_mask = (target_class == 1)
        
        true_positive_attacks = torch.sum(
            (preds_class == 1) & attack_mask
        )
        false_negative_attacks = torch.sum(
            (preds_class == 0) & attack_mask
        )
        
        # Update running totals
        self.total_attack_error += false_negative_attacks
        self.total_attack_samples += (true_positive_attacks + false_negative_attacks)
    
    def compute(self) -> Tensor:
        """
        Compute APCER value.
        
        Returns:
            Tensor: APCER score in range [0, 1]
                   0.0 = perfect attack detection
                   1.0 = all attacks missed (worst)
                   
        Note:
            Returns 0.0 if no attack samples in evaluation set
        """
        if self.total_attack_samples == 0:
            return torch.tensor(0.0, device=self.total_attack_error.device)
        
        return self.total_attack_error.float() / self.total_attack_samples.float()
