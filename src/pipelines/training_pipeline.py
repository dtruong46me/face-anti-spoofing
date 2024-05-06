import argparse

import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
import torch

import wandb

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from models.model_interface import load_model
from data.dataset import load_data, load_dataloader


def training_pipeline(args: argparse.Namespace):
    # Load dataset
    data = load_data(args)

    # Load dataloader
    train_loader, val_loader, _ = load_dataloader(data)

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" > Device map:", device)

    # Load model
    model = load_model(modelname=args.modelname, 
                       input_shape=args.input_shape, 
                       num_classes=args.num_classes)
    model.to(device)
    

    # Load logger
    wandb.login(key='c74fcec22fbb4be075a981b1f3db3f464b15b089')
    logger = WandbLogger(name="face-anti-spoof", project="cv-project")

    # Load callbacks
    es_callback = EarlyStopping(monitor="val/accuracy", min_delta=0.00, patience=4, verbose=False, mode="max")

    # Load trainer
    trainer = Trainer(max_epochs=args.max_epochs,
                      callbacks=[es_callback],
                      logger=logger)
    
    trainer.fit(model, train_loader, val_loader)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shape", type=tuple, default=(3,224,224))
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--modelname", type=str, default="seresnext50")
    parser.add_argument("--train_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_training")
    parser.add_argument("--test_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()

    training_pipeline(args)