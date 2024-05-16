
import torch
from torchmetrics import Metric

class MyRecall(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_attackk_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_attackk_error", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds=1: fake -> Positive
        preds=0: real -> Negative
        """ 
        preds = torch.argmax(preds, dim=1)
        target = torch.argmax(target, dim=1)

        true_pos = torch.sum((preds==1) & (target==1))
        # true_neg = torch.sum((preds==0) & (target==0))

        # false_pos = torch.sum((preds==1) & (target==0))
        false_neg = torch.sum((preds==0) & (target==1))

        self.total_attackk_error += true_pos
        self.total_attackk_samples += (true_pos + false_neg)
    
    def compute(self):
        return self.total_attackk_error.float() / self.total_attackk_samples.float()