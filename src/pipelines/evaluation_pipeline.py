import argparse

from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch import LightningDataModule

import os, sys

import torch

path = os.path.abspath(os.path.dirname(__name__))
sys.path.insert(0, path)

from src.models.ln_model import ModelInterface
from src.models.resnext50 import SEResNeXT50
from data.load_data import ingest_data
from utils import load_transform
from metrics.apcer import APCER
from metrics.npcer import NPCER
from metrics.acer import ACER
from metrics.accuracy import MyAccuracy
from metrics.recall import MyRecall

class DataTestFAS(LightningDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        self.test_path = args.test_path
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes

    def prepare_data(self) -> None:
        try:
            preprocess = load_transform()

            self.test = ingest_data(self.test_path, transform=preprocess)

        except Exception as e:
            raise e
        
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)


def evaluation_pipeline(args: argparse.Namespace):
    dataset = DataTestFAS(args)
    dataset.prepare_data()

    test_loader = dataset.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = None

    # Load SEResNeXT50
    if args.modelname == "seresnext50":
        backbone = SEResNeXT50(args.input_shape, args.num_classes)
    
    # Load MobileNetV2
    if args.modelname == "mobilenetv2":
        backbone = None
    
    # Load FeatherNet
    if args.modelname == "feathernet":
        backbone = None

    # Load model from path
    model = ModelInterface.load_from_checkpoint(args.model_checkpoint, 
                                                model=backbone,
                                                input_shape=args.input_shape, 
                                                num_classes=args.num_classes)
    model.to(device)

    model.eval()
    apcer_metric = APCER().to(device)
    npcer_metric = NPCER().to(device)
    acer_metric = ACER().to(device)
    accuracy = MyAccuracy().to(device)
    recall = MyRecall().to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            outputs = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)

            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(all_preds, all_preds.shape)
    print(all_labels, all_labels.shape)

    true_pos = torch.sum((all_preds==1) & (all_labels==1))
    true_neg = torch.sum((all_preds==0) & (all_labels==0))
    false_pos = torch.sum((all_preds==0) & (all_labels==1))
    false_neg = torch.sum((all_preds==1) & (all_labels==0))

    my_apcer = false_neg / (true_pos + false_neg)
    my_npcer = false_pos / (true_neg + false_pos)
    my_acer = 0.5 * (my_apcer + my_npcer)

    print("============")
    print("my_apcer =", my_apcer)
    print("my_npcer =", my_npcer)
    print("my_acer =", my_acer)

    apcer = apcer_metric(all_preds, all_labels)
    npcer = npcer_metric(all_labels, all_labels)

    acer = acer_metric(all_preds, all_labels)
    acc = accuracy(all_preds, all_labels)
    rec = recall(all_preds, all_labels)

    print(f"Test APCER: {apcer}")
    print(f"Test NPCER: {npcer}")
    print(f"Test ACER: {acer}")
    print(f"Test Accuracy: {acc}")
    print(f"Test Recall: {rec}")

