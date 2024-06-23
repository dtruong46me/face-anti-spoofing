import torch
import time
from fvcore.nn import FlopCountAnalysis

# Load your model
model = ...  # Your model here
model.eval()  # Set the model to evaluation mode
model.cuda()  # Move the model to GPU

# Define a dummy input tensor with the appropriate shape for your model
dummy_input = torch.randn(1, 3, 224, 224).cuda()  # Example for a model with input size 224x224 and 3 channels

# Warm up
for _ in range(10):
    _ = model(dummy_input)

# Measure inference time
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)
end_time = time.time()

average_inference_time = (end_time - start_time) / 100
print(f'Average inference time: {average_inference_time:.6f} seconds')

# Calculate FLOPs
flops = FlopCountAnalysis(model, dummy_input)
print(f"Total FLOPs: {flops.total()} FLOPs")