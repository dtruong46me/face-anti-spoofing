"""
Average Classification Error Rate (ACER) metric.

ACER is the average of APCER and NPCER, providing a balanced measure
of both attack detection and legitimate face acceptance.

See: ISO/IEC 30107-3:2017 for metric definitions
"""

import torch
from torch import Tensor
from torchmetrics import Metric


class ACER(Metric):
    """
    Average Classification Error Rate (ACER).
    
    Combines APCER (attack detection error) and NPCER (legitimate acceptance
    error) into a single balanced metric. Lower ACER is better overall.
    
    Formula:
        ACER = 0.5 * (APCER + NPCER)
        where:
        - APCER = FN_attacks / (TP_attacks + FN_attacks)
        - NPCER = FP_normal / (TN_normal + FP_normal)
    
    This metric balances:
        - Not rejecting legitimate faces (low NPCER)
        - Detecting presentation attacks (low APCER)
    
    Label Encoding:
        - Class 0 (Real): [1, 0]
        - Class 1 (Fake/Attack): [0, 1]
        
    Attributes:
        total_normal_samples (Tensor): Total legitimate presentation samples
        total_normal_error (Tensor): Legitimate presentations misclassified
        total_attack_samples (Tensor): Total attack presentation samples
        total_attack_error (Tensor): Attack presentations misclassified
        
    Example:
        >>> from torch import Tensor
        >>> metric = ACER()
        >>> 
        >>> # Batch 1: 16 real + 16 fake
        >>> preds = torch.cat([
        ...     torch.rand(16, 2),  # Real samples
        ...     torch.rand(16, 2),  # Fake samples
        ... ])
        >>> target = torch.cat([
        ...     torch.tensor([[1.0, 0.0]] * 16),  # Real labels
        ...     torch.tensor([[0.0, 1.0]] * 16),  # Fake labels
        ... ])
        >>> metric.update(preds, target)
        >>>
        >>> # Batch 2: 16 real + 16 fake
        >>> # ... more batches
        >>> metric.update(preds, target)
        >>>
        >>> # Get ACER for evaluation set
        >>> acer_value = metric.compute()
        >>> print(f"ACER: {acer_value:.4f}")
        ACER: 0.0427
        
    Notes:
        - ACER is the primary evaluation metric for FAS
        - Typical range: [0.0, 1.0]
        - Lower ACER indicates better overall performance
        - More balanced than APCER or NPCER alone
    """
    
    def __init__(self) -> None:
        """Initialize Average Classification Error Rate metric."""
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
        Update ACER metric with batch predictions.
        
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
        true_negative_normal = torch.sum((preds_class == 0) & normal_mask)
        false_positive_normal = torch.sum((preds_class == 1) & normal_mask)
        
        # Calculate metrics for attack presentations (target_class == 1)
        attack_mask = (target_class == 1)
        true_positive_attacks = torch.sum((preds_class == 1) & attack_mask)
        false_negative_attacks = torch.sum((preds_class == 0) & attack_mask)
        
        # Update running totals
        self.total_normal_error += false_positive_normal
        self.total_normal_samples += (true_negative_normal + false_positive_normal)
        
        self.total_attack_error += false_negative_attacks
        self.total_attack_samples += (true_positive_attacks + false_negative_attacks)
    
    def compute(self) -> Tensor:
        """
        Compute ACER value as average of APCER and NPCER.
        
        Returns:
            Tensor: ACER score in range [0, 1]
                   0.0 = perfect detection (no errors)
                   1.0 = worst possible errors
                   
        Note:
            Returns appropriate values even if one component (normal or attack)
            has no samples
        """
        # Compute NPCER
        if self.total_normal_samples == 0:
            npcer = torch.tensor(0.0, device=self.total_normal_error.device)
        else:
            npcer = self.total_normal_error.float() / self.total_normal_samples.float()
        
        # Compute APCER
        if self.total_attack_samples == 0:
            apcer = torch.tensor(0.0, device=self.total_attack_error.device)
        else:
            apcer = self.total_attack_error.float() / self.total_attack_samples.float()
        
        # Return average
        return 0.5 * (apcer + npcer)
