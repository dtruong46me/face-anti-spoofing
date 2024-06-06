
"""
    __  ___      __    _ __     _   __     __     _    _____ 
   /  |/  /___  / /_  (_) /__  / | / /__  / /_   | |  / /__ \ 
  / /|_/ / __ \/ __ \/ / / _ \/  |/ / _ \/ __/   | | / /__/ /
 / /  / / /_/ / /_/ / / /  __/ /|  /  __/ /_     | |/ // __/
/_/  /_/\____/_.___/_/_/\___/_/ |_/\___/\__/     |___//____/

"""
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import mobilenet_v3_small

class MobileNetV3(nn.Module):
    def __init__(self, input_shape=(3,224,224), num_classes=2):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = mobilenet_v3_small(pretrained=False)

        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


# if __name__ == '__main__':

#     model = MobileNetV3()
#     summary(model, (3, 224, 224))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model.to(device)
#     # Step 2: Create a sample input tensor
#     input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Batch size of 1, 3 color channels, 224x224 image size

#     # Step 3: Pass the input tensor through the model
#     output = model(input_tensor)

#     # Step 4: Print the output tensor
#     print("Model output:", output)
#     print("Output shape:", output.shape)

