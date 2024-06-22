from torchvision import transforms
import argparse

import os, sys

from src.models.ShuffleNet import ShuffleNet

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from models.feathernet import FeatherNetB
from models.resnext50 import SEResNeXT50
from models.mobilenet import MobileNetV3


def load_transform_2():
    return transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])


def load_backbone(args: argparse.Namespace):

    # Load SEResNeXT50
    if args.modelname == "seresnext50":
        return SEResNeXT50(args.input_shape, args.num_classes)
    
    # Load MobileNetV2
    if args.modelname == "mobilenetv3":
        return MobileNetV3(args.input_shape, args.num_classes)
    
    # Load FeatherNet
    if args.modelname == "feathernet":
        return FeatherNetB()

    if args.modelname == "shufflenet":
        return ShuffleNet()
    

def load_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shape", type=tuple, default=(3,224,224))
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--wandb_token", type=str, default="")
    parser.add_argument("--wandb_runname", type=str, default="model1")

    parser.add_argument("--modelname", type=str, default="seresnext50")

    parser.add_argument("--train_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_training")
    parser.add_argument("--test_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation")
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)

    parser.add_argument("--model_checkpoint", type=str, default="/kaggle/working/checkpoint/cvproject.ckpt")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--modelname", type=str, default="")

    args = parser.parse_args()

    return args