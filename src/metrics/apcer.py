
import torch
from torchmetrics import Metric

class APCER(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_attack_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_attack_error", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):        
        preds = torch.argmax(preds, dim=1)

        self.total_attack_sample += target.numel()

    
    def compute(self):
        total_attack_sample = torch.max(self.total_attack_samples, torch.tensor(1))
        return self.total_attack_error.float() / total_attack_sample
    

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        print(preds, preds.shape)
        preds = torch.argmax(preds, dim=1)
        print(preds)
        print(target)

        self.correct += torch.sum(preds==target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()