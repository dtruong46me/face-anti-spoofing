"""
Recall (Detection Rate) metric for Face Anti-Spoofing models.

Recall measures the proportion of presentation attacks correctly detected,
also known as the attack detection rate.
"""

import torch
from torch import Tensor
from torchmetrics import Metric


class MyRecall(Metric):
    """
    Recall / Attack Detection Rate metric.
    
    Measures the proportion of presentation attacks (fake images) that are
    correctly identified. Same as True Positive Rate or Sensitivity.
    High recall is important for security - missing attacks is dangerous.
    
    Formula:
        Recall = TP_attacks / (TP_attacks + FN_attacks)
        where:
        - TP_attacks: Fake images correctly classified as fake
        - FN_attacks: Fake images incorrectly classified as real
    
    This is complementary to APCER:
        Recall = 1 - APCER (when calculated on the same data)
    
    Label Encoding:
        - Class 0 (Real): [1, 0]
        - Class 1 (Fake/Attack): [0, 1]
        
    Attributes:
        total_attack_samples (Tensor): Total presentation attack samples
        total_attack_detections (Tensor): Correctly detected attacks
        
    Example:
        >>> from torch import Tensor
        >>> metric = MyRecall()
        >>> 
        >>> # Batch 1: Mix of real and fake samples
        >>> preds = torch.tensor([
        ...     [0.2, 0.8],    # pred=fake (correct attack detection)
        ...     [0.9, 0.1],    # pred=real
        ...     [0.1, 0.9],    # pred=fake (correct attack detection)
        ... ])
        >>> target = torch.tensor([
        ...     [0.0, 1.0],    # fake
        ...     [1.0, 0.0],    # real
        ...     [0.0, 1.0],    # fake
        ... ])
        >>> metric.update(preds, target)
        >>>
        >>> # Get recall on evaluation set
        >>> recall = metric.compute()
        >>> print(f"Recall (Attack Detection): {recall:.4f}")
        Recall (Attack Detection): 1.0000
        
    Notes:
        - Recall focuses on minimizing false negatives (missed attacks)
        - Important for security: missing attacks is very costly
        - Range: [0.0, 1.0] where 1.0 means all attacks detected
        - Inverse of APCER: Recall = 1 - APCER
    """
    
    def __init__(self) -> None:
        """Initialize recall metric."""
        super().__init__()
        self.add_state(
            "total_attack_samples",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_attack_detections",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum"
        )
    
    def update(
        self,
        preds: Tensor,
        target: Tensor
    ) -> None:
        """
        Update recall metric with batch predictions.
        
        Calculates true positives and false negatives for presentation attacks.
        True positives: Attacks correctly detected as fake
        False negatives: Attacks missed (incorrectly classified as real)
        
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
        self.total_attack_detections += true_positive_attacks
        self.total_attack_samples += (true_positive_attacks + false_negative_attacks)
    
    def compute(self) -> Tensor:
        """
        Compute recall/detection rate value.
        
        Returns:
            Tensor: Recall score in range [0, 1]
                   0.0 = no attacks detected (worst, very dangerous)
                   1.0 = all attacks detected (perfect security)
                   
        Note:
            Returns 0.0 if no attack samples in evaluation set
            
        Note:
            Recall = 1 - APCER (they measure the same thing differently)
        """
        if self.total_attack_samples == 0:
            return torch.tensor(0.0, device=self.total_attack_detections.device)
        
        return self.total_attack_detections.float() / self.total_attack_samples.float()