import pytorch_lightning as pl
from typing import Literal
from torch.optim import Adam
import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchsummary import summary
from pytorch_lightning.utilities.model_summary import summarize, ModelSummary


class SEResNeXT50(pl.LightningModule):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.dropout1 = nn.Dropout(p=0.5)

        self.fc = nn.Linear(in_features=self.backbone.fc.in_features, 
                            out_features=512)
        
        self.dropout2 = nn.Dropout(p=0.5)

        self.classifier = nn.Linear(in_features=512,
                                    out_features=num_classes)
        

    def forward(self, x):
        x = self.backbone(x)
        output = self.dropout(x)
        output = self.fc(output)
        return output
    
    def config_optimizers(self):
        return Adam(lr=5e-4, weight_decay=0.05)
    
    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch, batch_idx)
        self.log_dict({"val_loss": loss}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
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


def load_model(modelname: Literal["seresnext50", "mobilenet", "feathernet"], input_shape, num_classes):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if modelname == "seresnext50":
            model = SEResNeXT50(input_shape, num_classes)
            model.to(device)
            return model
        
    except Exception as e:
        raise e
    
# if __name__=='__main__':
#     model = load_model("seresnext50", input_shape=(3, 224, 224), num_classes=2)
#     print(summarize(model))