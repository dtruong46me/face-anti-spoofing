import pytorch_lightning as pl
from typing import Literal
from torch.optim import Adam
import torch
import torch.nn as nn
from torchsummary import summary
from pytorch_lightning.utilities.model_summary import summarize, ModelSummary

class OriginResNeXT(nn.Module):
    def __init__(self, input_shape):
        super(OriginResNeXT, self).__init__()

        self.resnext50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)

        # Freeze trainable parameters
        for name, param in self.resnext50.named_parameters():
            print(">>>>>", name, type(name), param.size())
            if name.startswith("fc"):
                param.requires_grad = False
                print("<<<<<<<<<<<<<<<")
    
    def forward(self, x):
        return


class SEResNeXT50(pl.LightningModule):
    def __init__(self, input_shape, num_classes, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

        self.resnext50 = OriginResNeXT(input_shape)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnext50(x)
        output = self.fc(x)
        return output
    
    def config_optimizers(self):
        return Adam(lr=5e-4, weight_decay=0.05)
    
    def training_step(self, batch):
        images, labels = batch
        outputs = self.forward(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def validation_step(self, batch):
        results = self.training_step(batch)
        return results
    
    def test_step(self, batch):
        images, labels = batch
        outputs = self.forward(images)
        _, preds = torch.max(outputs.data, 1)
        return preds


def load_model(modelname: Literal["seresnext50", "mobilenet", "feathernet"], input_shape, num_classes):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if modelname == "seresnext50":
            model = SEResNeXT50(input_shape, num_classes)
            model.to(device)
            return model
        
    except Exception as e:
        raise e
    

if __name__=='__main__':
    # origin = OriginResNeXT(input_shape=(224,224,3))
    model = load_model("seresnext50", input_shape=(3, 224, 224), num_classes=2)

    print(summarize(model))