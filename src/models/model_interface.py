import os, sys

from lightning.pytorch import LightningModule

from torch.optim import Adam
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, Recall

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from models.resnext50 import SEResNeXT50
from models.feathernet import FeatherNet
from models.mobilenet import MobileNetV2

from metrics.apcer import APCER
from metrics.npcer import NPCER


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

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        # self.train_apcer = APCER()
        
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        # self.val_apcer = APCER()

        self.backbone = model

    def forward(self, x: torch.Tensor):
        output = self.backbone(x)
        return output
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=5e-4, weight_decay=0.05)
    
    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)

        self.train_accuracy(outputs, labels)
        self.train_recall(outputs, labels)
        # self.train_apcer(outputs, labels)

        self.log_dict(dictionary={"train/loss": loss, "train/accuracy": self.train_accuracy, "train/recall": self.train_recall},
                      prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)

        self.val_accuracy(outputs, labels)
        self.val_recall(outputs, labels)
        # self.val_apcer(outputs, labels)

        self.log_dict(dictionary={"val/loss": loss, "val/accuracy": self.val_accuracy, "val/recall": self.val_recall}, 
                      prog_bar=False, logger=True, on_epoch=True, on_step=False)

        return loss
    
    def test_step(self, batch):
        images, labels = batch
        outputs = self.forward(images)
        _, preds = torch.max(outputs.data, 1)
        return preds
    
    def _common_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(0).float()

        outputs = self.forward(images)
        loss = nn.BCELoss()(outputs, labels)
        return loss, outputs, labels
    


def load_model(modelname: str, input_shape, num_classes):
    try:
        # Load SEResNeXT50
        if modelname == "seresnext50":
            print(" > Loading model SE-ResNeXT-50")
            backbone = SEResNeXT50(input_shape, num_classes)
            model = ModelInterface(backbone, input_shape, num_classes)
            return model
        
        # Load MobileNetV2
        if modelname == "mobilenetv2":
            backbone = MobileNetV2(input_shape, num_classes)
            model = ModelInterface(backbone, input_shape, num_classes)
            return model
        
        # Load FeatherNet
        if model == "feathernet":
            backbone = FeatherNet(input_shape, num_classes)
            model = ModelInterface(backbone, input_shape, num_classes)
            return model
        
    except Exception as e:
        raise e