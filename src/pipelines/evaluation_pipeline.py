import argparse

from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch import LightningDataModule

import os, sys

import torch

path = os.path.abspath(os.path.dirname(__name__))
sys.path.insert(0, path)

from src.models.ln_model import ModelInterface
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

    # Load model from path
    model = ModelInterface.load_from_checkpoint(args.model_checkpoint)
    model.to(device)

    model.eval()
    apcer_metric = APCER()
    npcer_metric = NPCER()
    acer_metric = ACER()
    accuracy = MyAccuracy()
    recall = MyRecall()

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