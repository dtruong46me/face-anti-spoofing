
import torch
from torchmetrics import Metric

class APCER(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_attack_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_attack_error", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):        
        preds = 1 - torch.argmax(preds, dim=1)
        target = 1 - torch.argmax(target, dim=1)

        false_pos = torch.sum((preds==1) & (target==0))
        true_neg = torch.sum((preds==0) & (preds==0))

        self.total_attack_error += false_pos
        self.total_attack_samples += (true_neg + false_pos)
    
    def compute(self):
        return self.total_attack_error.float() / self.total_attack_samples.float()
