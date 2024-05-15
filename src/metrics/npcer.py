
import torch
from torchmetrics import Metric

class NPCER(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_normal_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_normal_error", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):        
        preds = torch.argmax(preds, dim=1)
        target = torch.argmax(target, dim=1)

        # true_pos = torch.sum((preds==1) & (target==1))
        true_neg = torch.sum((preds==0) & (target==0))

        false_pos = torch.sum((preds==1) & (target==0))
        # false_neg = torch.sum((preds==0) & (target==1))
        
        self.total_normal_error += false_pos
        self.total_normal_samples += (true_neg + false_pos)
    
    def compute(self):
        return self.total_normal_error.float() / self.total_normal_samples.float()
