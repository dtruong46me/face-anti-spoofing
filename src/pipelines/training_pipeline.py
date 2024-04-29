import argparse

import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from models.model import load_model
from data.load_dataset import load_data

def training_pipeline(args: argparse.Namespace):
    # Load dataset
    data = load_data(args)
    data.prepare_data()
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    print("Complete load data")

    # Load model
    model = load_model(modelname=args.modelname, 
                       input_shape=args.input_shape, 
                       num_classes=args.num_classes)
    
    print("Complete load model")

    # Load logger
    logger = TensorBoardLogger("tb_logs", "my_model")

    # Load callbacks
    es_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
    callbacks = [es_callback]

    # Load trainer
    trainer = Trainer(max_epochs=args.max_epochs, 
                      logger=logger,
                      callbacks=callbacks)
    
    trainer.fit(model, train_loader, val_loader)

    print("Complete training")
    
    model.on_save_checkpoint("first_model.ckpt")
    model.on_save_checkpoint("checkpoint_model.pht")
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