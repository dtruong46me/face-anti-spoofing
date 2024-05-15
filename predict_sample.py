import torch

import sys, os, argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from src.models.ln_model import ModelInterface
from models.resnext50 import SEResNeXT50
from src.utils import load_transform


def predict_sample(args):
    # Define the preprocessing transformations
    preprocess = load_transform()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load backbone model
    backbone = None
    if args.modelname == "seresnext50":
        backbone = SEResNeXT50(args.input_shape, args.num_classes)
    if args.modelname == "mobilenetv2":
        backbone = None
    if args.modelname == "feathernet":
        backbone = None

    # Load model from path
    model = ModelInterface.load_from_checkpoint(args.model_checkpoint, 
                                                model=backbone,
                                                input_shape=args.input_shape, 
                                                num_classes=args.num_classes)

    model.eval()

    if type(args.image)==str:
        image = Image.open(args.image).convert('RGB')
    
    # Apply the transformations to the input image

    image = preprocess(image).unsqueeze(0)

    plt.imshow(image.squeeze().permute(1, 2, 0))  # Convert tensor back to image for visualization
    plt.axis("off")
    plt.show()
    
    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

    # Get the predicted class and the associated probabilities
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_probabilities = probabilities.squeeze().cpu().numpy()

    if predicted_class == 0:
        result = {'tensor': [0, 1], 'class': 0, 'label': 'real', 'probability': predicted_probabilities[1]}
    else:
        result = {'tensor': [1, 0], 'class': 1, 'label': 'fake', 'probability': predicted_probabilities[0]}

    plt.imshow(image)
    plt.title(f"Predict: {result['class']} - {result['label']} - prob: {result['probability']:.4f}")
    plt.axis("off")
    plt.show()
    plt.savefig("predict.jpg")

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="/kaggle/working/checkpoint/cvproject.ckpt")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--modelname", type=str, default="")
    parser.add_argument("--input_shape", type=tuple, default=(3,224,224))
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()

    result = predict_sample(args)
    print(result)

if __name__=="__main__":
    main()
