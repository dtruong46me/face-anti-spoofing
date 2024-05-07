import torch
from torchmetrics import Metric

class NPCER(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_normal_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_normal_error", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        # Tính số lượng mẫu bình thường và lỗi dự đoán cho mẫu bình thường
        normal_mask = target == 0
        self.total_normal_samples += torch.sum(normal_mask)
        self.total_normal_error += torch.sum((preds != target) * normal_mask)

    def compute(self):
        total_normal_samples = torch.max(self.total_normal_samples, torch.tensor(1))
        return self.total_normal_error.float() / total_normal_samples