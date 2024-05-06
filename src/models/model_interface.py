from lightning.pytorch import LightningModule

from torch.optim import Adam
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

class SEResNeXT50(LightningModule):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.save_hyperparameters()

        self.train_accuracy = Accuracy(task="binary", num_classes=num_classes)
        self.train_precision = Precision(task="binary", num_classes=num_classes)
        self.train_recall = Recall(task="binary", num_classes=num_classes)
        self.train_f1score = F1Score(task="binary", num_classes=num_classes)

        self.val_accuracy = Accuracy(task="binary", num_classes=num_classes)
        self.val_f1score = F1Score(task="binary", num_classes=num_classes)

        # Clone model backbone ResNeXT50
        self.backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        
        # Freeze all weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.fc.in_features

        # Delete the last layer
        self.backbone.fc = nn.Identity()

        # Linear 1
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.relu =nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        # Classifier
        self.classifier = nn.Linear(in_features=512, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.classifier(out)
        out = self.sigmoid(out)
        return out
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=5e-4, weight_decay=0.05)
    
    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)

        self.train_accuracy(outputs, labels)
        self.train_f1score(outputs, labels)
        self.train_precision(outputs, labels)
        self.train_recall(outputs, labels)

        self.log_dict(dictionary={"train/loss": loss, "train/accuracy": self.train_accuracy, "train/f1_score": self.train_f1score, "train/precicion": self.train_precision, "train/recall": self.train_recall}, 
                      prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)

        self.val_accuracy(outputs, labels)
        self.val_f1score(outputs, labels)

        self.log_dict(dictionary={"val/loss": loss, "val/accuracy": self.val_accuracy, "val/f1_score": self.val_f1score}, 
                      prog_bar=False, logger=True, on_epoch=True, on_step=False)

        return loss
    
    def test_step(self, batch):
        images, labels = batch
        outputs = self.forward(images)
        _, preds = torch.max(outputs.data, 1)
        return preds
    
    def _common_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss, outputs, labels
    
    def on_test_epoch_end(self, outputs) -> None:
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        acc = Accuracy(preds, labels)
        self.log("test_accuracy", acc, on_epoch=True)


def load_model(modelname: str, input_shape, num_classes):
    try:
        # Load SEResNeXT50
        if modelname == "seresnext50":
            model = SEResNeXT50(input_shape, num_classes)
            return model
        
        # Load MobileNetV2
        if modelname == "mobilenetv2":
            pass
        
        # Load FeatherNet
        if model == "feathernet":
            pass
        
    except Exception as e:
        raise e