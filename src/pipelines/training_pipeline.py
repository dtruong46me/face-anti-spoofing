import argparse

import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import summarize
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar

import wandb

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from models.model import load_model
from data.dataset import load_data

class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

def training_pipeline(args: argparse.Namespace):
    # Load dataset
    data = load_data(args)
    data.prepare_data()
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    # test_loader = data.test_dataloader()

    print("Complete load data")

    # Load model
    model = load_model(modelname=args.modelname, 
                       input_shape=args.input_shape, 
                       num_classes=args.num_classes)
    
    print("Complete load model")

    summarize(model)

    # Load logger
    wandb.login(key='c74fcec22fbb4be075a981b1f3db3f464b15b089')
    logger = WandbLogger(name="face-anti-spoof", project="cv-project")

    # Load callbacks
    es_callback = EarlyStopping(monitor="accuracy", min_delta=0.00, patience=4, verbose=False, mode="max")
    # tqdm_callback = MyProgressBar()
    print(es_callback)

    # Load trainer
    trainer = Trainer(max_epochs=args.max_epochs, 
                      logger=logger)
    
    trainer.fit(model, train_loader, val_loader)

    print("Complete training")
    
    model.on_save_checkpoint("first_model.ckpt")
    model.on_save_checkpoint("checkpoint_model.pth")
    print("Complete save checkpoint")

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