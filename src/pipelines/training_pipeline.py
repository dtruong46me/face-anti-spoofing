import argparse

import os
import sys

import pytorch_lightning as pl

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from models.model import load_model

def training_pipeline(args: argparse.Namespace):
    # Load dataset
    data = None
    train_dataloader = None
    val_dataloader = None

    # Load model
    model = load_model(args.input_shape, args.num_classes)

    # Load callbacks
    callbacks = []

    # Load trainer
    trainer = pl.Trainer(max_epochs=10,
                         callbacks=callbacks)
    
    trainer.fit(model, train_dataloader, val_dataloader)

    