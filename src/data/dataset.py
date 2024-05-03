
import os
import sys

from lightning.pytorch import LightningDataModule

from torch.utils.data import random_split, DataLoader
from torch import Generator
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from torch.nn import functional as F

from torchvision import datasets, transforms

import argparse

path = os.path.abspath(os.path.dirname(__name__))
sys.path.insert(0, path)
from data.load_data import ingest_data

class LCCFASDataset(LightningDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        self.train_path = args.train_path # args.datapaht="/kaggle/input/lcc-fasd"
        self.test_path = args.test_path
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes

        '''
        Trong đó cấu trúc thư mục của /kaggle/input/lcc-fasd/ như sau:
        /kaggle/input/lcc-fasd/
        |--LCC_FASD
            |--LCC_FASD_development
            |   |--real
            |   |   |--real1.jpg
            |   |   |--...
            |   |--fake
            |       |--fake1.jpg
            |       |--...
            |--LCC_FASD_evaluation
            |   |--real
            |   |--fake
            |--LCC_FASD_training
                |--real
                |--fake
        '''


    def prepare_data(self) -> None:
        try:
            preprocess = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            self.train = ingest_data(self.train_path, transform=preprocess)

            self.test = ingest_data(self.test_path, transform=preprocess)

        except Exception as e:
            raise e

    def setup(self, stage: str="fit") -> None:
        try:
            if stage=="fit":
                TRAIN_RATE = 0.7
                train_size = int(TRAIN_RATE*len(self.train))
                val_size = len(self.train) - train_size
                
                self.train, self.val = random_split(
                    self.train, lengths=[train_size, val_size], generator=Generator().manual_seed(42)
                )

        except Exception as e:
            raise e
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
        
def load_data(args):
    try:
        dataset = LCCFASDataset(args)
        return dataset
    
    except Exception as e:
        raise e
    
def load_dataloader(dataset: LCCFASDataset):
    try:
        dataset.prepare_data()
        dataset.setup()
        train_loader = dataset.train_dataloader()
        val_loader = dataset.train_dataloader()
        test_loader = dataset.train_dataloader()

        return train_loader, val_loader, test_loader

    except Exception as e:
        raise e