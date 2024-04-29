
import logging
import pytorch_lightning as pl

from torch.utils.data import random_split, DataLoader
from torch import Generator
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

from torchvision import datasets, transforms

import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LCCFASDataset(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        self.train_path = args.train_path # args.datapaht="/kaggle/input/lcc-fasd"
        self.test_path = args.test_path
        self.batch_size = args.batch_size
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
        print(1)
        self.train = datasets.ImageFolder(self.train_path, transform=self.get_transform)
        print(2)
        self.test = datasets.ImageFolder(self.test_path, transform=self.get_transform)

    def setup(self, stage: str) -> None:
        print(3)
        self.train, self.val = random_split(
            self.train, lengths=[0.7, 0.3], generator=Generator().manual_seed(42)
        )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        print(4)
        return DataLoader(self.train, batch_size=self.batch_size)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        print(5)
        return DataLoader(self.val, self.batch_size)
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        print(6)
        return DataLoader(self.test, self.batch_size)
    
    def get_transform(self):
        print(7)
        preprocess = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return preprocess
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_training")
    parser.add_argument("--test_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_test")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    dataset = LCCFASDataset(args)
    dataset.setup()
    train_loader = dataset.train_dataloader()
    print(train_loader)
    val_loader = dataset.val_dataloader()
    print(val_loader)
    test_loader = dataset.test_dataloader()
    print(test_loader)