import argparse

import os
import sys

import pytorch_lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from models.model import load_model
from data.load_dataset import load_data

def training_pipeline(args: argparse.Namespace):
    # Load dataset
    data = load_data(args)

    # Load model
    model = load_model(args.input_shape, args.num_classes)

    # Load callbacks
    es_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
    callbacks = [es_callback]

    # Load trainer
    trainer = pl.Trainer(default_root_dir="/kaggle/working/",
                         max_epochs=10,
                         callbacks=callbacks)
    
    trainer.fit(model, data.train_dataloader(), data.val_dataloader())
    model.on_save_checkpoint("first_model.ckpt")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shape", type=tuple, default=(3,224,224))
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--train_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_training")
    parser.add_argument("--test_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    training_pipeline(args)