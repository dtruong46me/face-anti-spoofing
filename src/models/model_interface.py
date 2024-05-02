
from lightning.pytorch import LightningModule
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision import transforms
from torchvision import datasets

class ModelInterface(LightningModule):
    def __init__(self, backbone, input_shape, num_classes, classifier):
        super().__init__()
        self.backbone = backbone
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.classifier = classifier

    def forward(self, x):
        """
        Use: 
            - `forward` for inference/predictions
            - x: torch.Tensor([batch_size, input_shape]) -> 3D or 4D
            - e.g: with batch_size=32 and input_shape=(3, 224, 224) -> x.shape = [32, 3, 224, 224]
        
        Return:
            out: torch.Tensor([batch_size, num_classes])
        """

        out = self.backbone(x)
        out = self.classifier(out)

        return out
    

class SEResNeXT50(ModelInterface):
    def __init__(self, backbone, input_shape, num_classes, classifier):
        super().__init__()

        self.backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2, pretrained=True)


train_set = datasets.STL10('./data',split='train',transform=transforms.ToTensor(),download=True)