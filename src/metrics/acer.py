import torch
from torchmetrics import Metric

import os, sys

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0)

from apcer import APCER
from npcer import NPCER

class ACER(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.apcer = APCER(dist_sync_on_step=dist_sync_on_step)
        self.npcer = NPCER(dist_sync_on_step=dist_sync_on_step)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.apcer.update(preds, target)
        self.npcer.update(preds, target)

    def compute(self):
        apcer = self.apcer.compute()
        npcer = self.npcer.compute()
        acer = (apcer + npcer) / 2
        return acer