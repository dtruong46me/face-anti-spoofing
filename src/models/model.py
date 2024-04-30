from lightning.pytorch import LightningModule
from typing import Literal
from torch.optim import Adam
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from pytorch_lightning.utilities.model_summary import summarize


class SEResNeXT50(LightningModule):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.accuracy = Accuracy(task="binary", num_classes=num_classes)
        self.precision = Precision(task="binary", num_classes=num_classes)
        self.recall = Recall(task="binary", num_classes=num_classes)
        self.f1score = F1Score(task="binary", num_classes=num_classes)

        self.backbone = resnext50_32x4d()
        
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        in_features = self.backbone.fc.in_features

        # Delete the last layer
        self.backbone.fc = nn.Identity()

        self.fc = nn.Linear(in_features=in_features, out_features=512)

        self.dropout = nn.Dropout(p=0.5)

        self.classifier = nn.Linear(in_features=512, out_features=num_classes)


    def forward(self, x):
        out = self.backbone(x)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=5e-3, weight_decay=0.05)
    
    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)

        self.accuracy(outputs, labels)
        self.f1score(outputs, labels)
        self.precision(outputs, labels)
        self.recall(outputs, labels)

        self.log_dict(dictionary={"train_loss": loss, "accuracy": self.accuracy, "f1_score": self.f1score, "precicion": self.precision, "recall": self.recall}, 
                      prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)

        self.accuracy(outputs, labels)
        self.f1score(outputs, labels)
        self.precision(outputs, labels)
        self.recall(outputs, labels)

        self.log_dict(dictionary={"val_loss": loss, "accuracy": self.accuracy, "f1_score": self.f1score, "precision": self.precision, "recall": self.recall}, 
                      prog_bar=True, logger=True, on_epoch=True, on_step=False)

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
    
    def on_train_epoch_end(self) -> None:
        self.log("train_acc_epoch", self.accuracy)


def load_model(modelname: str, input_shape, num_classes):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        if modelname == "seresnext50":
            model = SEResNeXT50(input_shape, num_classes)
            model.to(device)
            return model
        
    except Exception as e:
        raise e
    
# if __name__=='__main__':
#     model = load_model("seresnext50", input_shape=(3, 224, 224), num_classes=2)
#     print("++++++++++")
#     for name, param in model.named_parameters():
#         print(name, param.size())
#     print(summarize(model))