
"""
   _____ ______   ____            _   __    _  ________   __________ 
  / ___// ____/  / __ \___  _____/ | / /__ | |/ /_  __/  / ____/ __ \ 
  \__ \/ __/    / /_/ / _ \/ ___/  |/ / _ \|   / / /    /___ \/ / / /
 ___/ / /___   / _, _/  __(__  ) /|  /  __/   | / /    ____/ / /_/ /
/____/_____/  /_/ |_|\___/____/_/ |_/\___/_/|_|/_/    /_____/\____/

"""
from torch import Tensor
import torch.nn as nn
from torchsummary import summary
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

class SEResNeXT50(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.resnext = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

        for param in self.resnext.parameters():
            param.requires_grad = False

        in_features = self.resnext.fc.in_features

        self.resnext.fc = nn.Identity()

        # Linear 1
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.relu =nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        # Classifier
        self.classifier = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: Tensor):
        out = self.resnext(x)
        out = out.view(out.size(0), -1)

        # Linear 1
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        # Classifier
        out = self.classifier(out)
        return out
    
# if __name__=="__main__":
#     INPUT_SHAPE = (3,224,224)
#     NUM_CLASSES = 2

#     model = SEResNeXT50(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
#     summary(model, input_size=(3, 224, 224))

#     import torch
#     tensor1 = torch.rand([1,3,224,224])
#     print(tensor1.shape)

#     output1 = model.forward(tensor1)
#     print(output1, output1.shape)

#     tensor2 = torch.rand([32,3,224,224])
#     print(tensor2.shape)
    
#     output2 = model.forward(tensor2)
#     print(output2, output2.shape)