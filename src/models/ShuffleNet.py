import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights
from torchsummary import summary


class ShuffleNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=2):
        super(ShuffleNet, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.DEFAULT)

        # Replace the classifier layer with a new one
        num_features = self.model.fc.in_features  # shufflenet_v2_x2_0 uses 'fc' as the classifier layer
        self.model.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    # Step 1: Instantiate the model
    model = ShuffleNet()

    # Print model summary
    print(summary(model, (3, 224, 224)))

    # Step 2: Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Step 3: Create a sample input tensor
    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Batch size of 1, 3 color channels, 224x224 image size

    # Step 4: Pass the input tensor through the model
    output = model(input_tensor)

    # Step 5: Print the output tensor
    print("Model output:", output)
    print("Output shape:", output.shape)