
import torch
from torchmetrics import Metric

class APCER(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_attack_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_attack_error", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        attack_mask = (target == 1).float()
        self.total_attack_samples += torch.sum(attack_mask)
        self.total_attack_error += torch.sum((preds!=target)*attack_mask)

    
    def compute(self):
        total_attack_sample = torch.max(self.total_attack_samples, torch.tensor(1))
        return self.total_attack_error.float() / total_attack_sample