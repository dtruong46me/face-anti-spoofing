
import torch
from torchmetrics import Metric

class NPCER(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_normal_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_normal_error", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):        
        preds = 1 - torch.argmax(preds, dim=1)
        target = 1 - torch.argmax(target, dim=1)

        false_neg = torch.sum((preds==0) & (target==1))
        true_pos = torch.sum((preds==1) & (preds==1))

        self.total_normal_error += false_neg
        self.total_normal_samples += (true_pos + false_neg)
    
    def compute(self):
        return self.total_normal_error.float() / self.total_normal_samples.float()
