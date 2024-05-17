
"""
    __  ___      __    _ __     _   __     __     _    _____ 
   /  |/  /___  / /_  (_) /__  / | / /__  / /_   | |  / /__ \ 
  / /|_/ / __ \/ __ \/ / / _ \/  |/ / _ \/ __/   | | / /__/ /
 / /  / / /_/ / /_/ / / /  __/ /|  /  __/ /_     | |/ // __/
/_/  /_/\____/_.___/_/_/\___/_/ |_/\___/\__/     |___//____/

"""
import torch.nn as nn
from torchsummary import summary
from torchvision.models import mobilenet_v3_small

class MobileNetV3(nn.Module):
    def __init__(self, input_shape=(3,224,224), num_classes=2):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.mobilenet = mobilenet_v3_small(pretrained=False)

        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.classifier[0].in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(1024, self.num_classes)
        )

    def forward(self, x):
        x = self.mobilenet.features(x)
        return x

if __name__ == '__main__':
    model = MobileNetV3()
    summary(model, (3, 224, 224))