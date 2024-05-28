import os, sys

from lightning.pytorch import LightningModule

from torch.optim import Adam
import torch
import torch.nn as nn
from torchsummary import summary

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)


from metrics.apcer import APCER
from metrics.npcer import NPCER
from metrics.acer import ACER
from metrics.accuracy import MyAccuracy
from metrics.recall import MyRecall

"""
    __  ___          __     __   ____      __            ____              
   /  |/  /___  ____/ /__  / /  /  _/___  / /____  _____/ __/___ _________ 
  / /|_/ / __ \/ __  / _ \/ /   / // __ \/ __/ _ \/ ___/ /_/ __ `/ ___/ _ \ 
 / /  / / /_/ / /_/ /  __/ /  _/ // / / / /_/  __/ /  / __/ /_/ / /__/  __/
/_/  /_/\____/\__,_/\___/_/  /___/_/ /_/\__/\___/_/  /_/  \__,_/\___/\___/ 
"""

class ModelInterface(LightningModule):
    def __init__(self, model, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Metric for training
        self.train_apcer = APCER()
        self.train_npcer = NPCER()
        self.train_acer = ACER()
        self.train_accuracy = MyAccuracy()
        self.train_recall = MyRecall()
        
        # Metric for validation
        self.val_apcer = APCER()
        self.val_npcer = NPCER()
        self.val_acer = ACER()
        self.val_accuracy = MyAccuracy()
        self.val_recall = MyRecall()

        self.backbone = model

    def forward(self, x: torch.Tensor):
        output = self.backbone(x)
        return output
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=5e-5, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)

        acc = self.train_accuracy(outputs, labels)
        rec = self.train_recall(outputs, labels)
        apcer = self.train_apcer(outputs, labels)
        npcer = self.train_npcer(outputs, labels)
        acer = self.train_acer(outputs, labels)

        self.log_dict(dictionary={"train/loss": loss, "train/accuracy": acc, "train/recall": rec, "train/apcer": apcer, "train/npcer": npcer, "train/acer": acer},
                      prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)

        acc = self.val_accuracy(outputs, labels)
        rec = self.val_recall(outputs, labels)
        apcer = self.val_apcer(outputs, labels)
        npcer = self.val_npcer(outputs, labels)
        acer = self.val_acer(outputs, labels)

        self.log_dict(dictionary={"val/loss": loss, "val/accuracy": acc, "val/recall": rec, "val/apcer": apcer, "val/npcer": npcer, "val/acer": acer},
                      prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss
    
    def test_step(self, batch):
        images, labels = batch
        outputs = self.forward(images)
        _, preds = torch.max(outputs.data, 1)
        return preds
    
    def _common_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.squeeze(0).float()

        # weights = [0.85, 0.15]
        # weights = torch.FloatTensor(weights).cuda()

        outputs = self.forward(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss, outputs, labels

# Load Lightning Model
def load_model(backbone, input_shape, num_classes):
    try:
        model = ModelInterface(backbone, input_shape, num_classes)
        return model
        
    except Exception as e:
        raise e