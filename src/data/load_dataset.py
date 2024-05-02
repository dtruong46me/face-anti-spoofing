
import logging

from lightning.pytorch import LightningDataModule

from torch.utils.data import random_split, DataLoader
from torch import Generator
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from torchvision import datasets, transforms

import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LCCFASDataset(LightningDataModule):
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
        try:
            preprocess = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            logger.info(f"Loading image from {self.train_path}")
            self.train = datasets.ImageFolder(self.train_path, transform=preprocess)
            print("Classes of training dataset:", self.train.classes)
            print(self.train[0])
            print(self.train[1])
            
            logger.info(f"Loading image from {self.test_path}")
            self.test = datasets.ImageFolder(self.test_path, transform=preprocess)
            print("Class of test dataset:", self.test.class_to_idx, self.test.classes)

        except Exception as e:
            logger.error(f"Error while loading image: {e}")
            raise e

    def setup(self, stage: str="fit") -> None:
        try:
            if stage=="fit":
                TRAIN_RATE = 0.7
                train_size = int(TRAIN_RATE*len(self.train))
                val_size = len(self.train) - train_size
                
                logger.info(f"Splitting image with size: 70/30")
                self.train, self.val = random_split(
                    self.train, lengths=[train_size, val_size], generator=Generator().manual_seed(42)
                )
        except Exception as e:
            raise e
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        try:
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        except Exception as e:
            raise e
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        try:
            return DataLoader(self.val, self.batch_size, shuffle=False, num_workers=4)
        
        except Exception as e:
            raise e
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        try:
            return DataLoader(self.test, self.batch_size, shuffle=False)
        
        except Exception as e:
            raise e
    
    # def get_transform(self):
    #     try:
    #         preprocess = transforms.Compose([
    #             transforms.Resize([224, 224]),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    #         ])
    #         return preprocess
        
    #     except Exception as e:
    #         raise e
        
def load_data(args):
    try:
        dataset = LCCFASDataset(args)
        return dataset
    
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise e
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_training")
    parser.add_argument("--test_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    dataset = LCCFASDataset(args)
    dataset.prepare_data()
    dataset.setup()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()
    print(dataset.train)
    print(train_loader)
    print(dataset.val_dataloader)
    print(dataset.test_dataloader)