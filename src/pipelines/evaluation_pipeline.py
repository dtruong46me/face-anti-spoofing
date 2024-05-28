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
from utils import load_transform, load_backbone
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


    backbone = load_backbone(args)

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

            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    total_image = all_labels.shape[0]
    positive_image = torch.sum(torch.argmax(all_labels, dim=1)).float()
    negative_image = (total_image - positive_image).float()

    apcer = apcer_metric(all_preds, all_labels)
    npcer = npcer_metric(all_preds, all_labels)

    acer = acer_metric(all_preds, all_labels)
    acc = accuracy(all_preds, all_labels)
    rec = recall(all_preds, all_labels)

    print(f"APCER: {apcer}")
    print(f"NPCER: {npcer}")
    print(f"ACER: {acer}")
    print(f"Accuracy: {acc}")
    print(f"Recall: {rec}")

    true_pos = torch.sum((torch.argmax(all_preds, dim=1)==1) & (torch.argmax(all_labels, dim=1)==1)).float()
    true_neg = torch.sum((torch.argmax(all_preds, dim=1)==0) & (torch.argmax(all_labels, dim=1)==0)).float()
    false_pos = torch.sum((torch.argmax(all_preds, dim=1)==1) & (torch.argmax(all_labels, dim=1)==0)).float()
    false_neg = torch.sum((torch.argmax(all_preds, dim=1)==0) & (torch.argmax(all_labels, dim=1)==1)).float()

    true_pos = int(true_pos.item())
    true_neg = int(true_neg.item())
    false_pos = int(false_pos.item())
    false_neg = int(false_neg.item())

    print("============")
    print("Total images:", total_image)
    print("Positive - Fake (1):", positive_image)
    print("Negative - Real (0)", negative_image)
    print("+TP:", true_pos, 
          "\n+TN:", true_neg,
          "\n+FP:", false_pos,
          "\n+FN:", false_neg)