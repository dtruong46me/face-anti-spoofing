
import torch
from torchmetrics import Metric

class APCER(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_attack_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_attack_error", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds=0: real -> Negative -> [1,0]
        preds=1: fake -> Positive -> [0,1]
        """ 
        preds = torch.argmax(preds, dim=1)
        target = torch.argmax(target, dim=1)

        true_pos = torch.sum((preds==1) & (target==1))
        # true_neg = torch.sum((preds==0) & (target==0))

        # false_pos = torch.sum((preds==1) & (target==0))
        false_neg = torch.sum((preds==0) & (target==1))

        self.total_attack_error += false_neg
        self.total_attack_samples += (true_pos + false_neg)
    
    def compute(self):
        return self.total_attack_error.float() / self.total_attack_samples.float()
