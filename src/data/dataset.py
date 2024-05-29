
import os
import random
import sys

from lightning.pytorch import LightningDataModule

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torch import Generator
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from torchvision import transforms

import argparse

path = os.path.abspath(os.path.dirname(__name__))
sys.path.insert(0, path)
from data.load_data import ingest_data
from utils import load_transform_2

class LCCFASDataset(LightningDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        self.train_path = args.train_path # args.datapaht="/kaggle/input/lcc-fasd"
        self.test_path = args.test_path
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes

        '''
            ____        __                  __ 
           / __ \____ _/ /_____ _________  / /_
          / / / / __ `/ __/ __ `/ ___/ _ \/ __/
         / /_/ / /_/ / /_/ /_/ (__  )  __/ /_  
        /_____/\__,_/\__/\__,_/____/\___/\__/  

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
            preprocess = load_transform_2()

            self.train = ingest_data(self.train_path, transform=preprocess)
            print("Load training data:", self.train)

            if self.test_path != "":
                self.test = ingest_data(self.test_path, transform=preprocess)
                print("Load test data:", self.test)

        except Exception as e:
            raise e

    def setup(self, stage: str="fit") -> None:
        try:
            if stage=="fit":
                TRAIN_RATE = 0.7
                train_size = int(TRAIN_RATE*len(self.train))
                val_size = len(self.train) - train_size

                # Seed for reproducibility
                seed = 42
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                self.train, self.val = random_split(
                    self.train, lengths=[train_size, val_size], generator=Generator().manual_seed(seed)
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
        val_loader = dataset.val_dataloader()

        if dataset.test_path == "":
            return train_loader, val_loader, None
        
        test_loader = dataset.test_dataloader()

        return train_loader, val_loader, test_loader

    except Exception as e:
        raise e