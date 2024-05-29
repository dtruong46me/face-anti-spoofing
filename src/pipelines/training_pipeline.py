import argparse

import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch

import wandb

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from models.ln_model import load_model, ModelInterface
from models.resnext50 import SEResNeXT50
from models.mobilenet import MobileNetV3
from models.feathernet import FeatherNetB
from data.dataset import load_data, load_dataloader

from utils import load_backbone

from metrics.apcer import APCER
from metrics.npcer import NPCER
from metrics.acer import ACER


"""
 ____           __                        __           __                   __      __                                
/\  _`\        /\ \__                    /\ \         /\ \       __        /\ \    /\ \__         __                     
\ \ \L\ \__  __\ \ ,_\   ___   _ __   ___\ \ \___     \ \ \     /\_\     __\ \ \___\ \ ,_\   ___ /\_\    ___      __     
 \ \ ,__/\ \/\ \\ \ \/  / __`\/\`'__\/'___\ \  _ `\    \ \ \  __\/\ \  /'_ `\ \  _ `\ \ \/ /' _ `\/\ \ /' _ `\  /'_ `\   
  \ \ \/\ \ \_\ \\ \ \_/\ \L\ \ \ \//\ \__/\ \ \ \ \    \ \ \L\ \\ \ \/\ \L\ \ \ \ \ \ \ \_/\ \/\ \ \ \/\ \/\ \/\ \L\ \  
   \ \_\ \/`____ \\ \__\ \____/\ \_\\ \____\\ \_\ \_\    \ \____/ \ \_\ \____ \ \_\ \_\ \__\ \_\ \_\ \_\ \_\ \_\ \____ \ 
    \/_/  `/___/> \\/__/\/___/  \/_/ \/____/ \/_/\/_/     \/___/   \/_/\/___L\ \/_/\/_/\/__/\/_/\/_/\/_/\/_/\/_/\/___L\ \ 
             /\___/                                                      /\____/                                  /\____/
             \/__/                                                       \_/__/                                   \_/__/ 
"""


def training_pipeline(args: argparse.Namespace):
    # Load dataset
    data = load_data(args)

    # Load dataloader
    train_loader, val_loader, test_loader = load_dataloader(data)

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" > Device map:", device)


    # Load backbone
    backbone = load_backbone(args)

    # Load model
    model = load_model(backbone=backbone,
                       input_shape=args.input_shape, 
                       num_classes=args.num_classes)
    model.to(device)
    

    # Load logger
    wandb.login(key=args.wandb_token)
    logger = WandbLogger(name=args.wandb_runname, project="cv-project")

    # Load callbacks
    es_callback = EarlyStopping(monitor="val/apcer", min_delta=0.00, patience=4, verbose=True, mode="min")

    ckpt_callback = ModelCheckpoint(
        dirpath='checkpoint',
        filename=args.modelname,
        save_top_k=3,
        verbose=True,
        mode='min',
        monitor="val/npcer"
    )

    # Load trainer
    trainer = Trainer(max_epochs=args.max_epochs,
                      callbacks=[ckpt_callback],
                      logger=logger)
    
    trainer.fit(model, train_loader, val_loader)

    # Save best model checkpoint
    best_model_path = ckpt_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    # Load model from path
    model = ModelInterface.load_from_checkpoint(checkpoint_path=best_model_path, 
                                                model=backbone,
                                                input_shape=args.input_shape, 
                                                num_classes=args.num_classes)
    model.to(device)

    if test_loader is not None:
        model.eval()
        apcer_metric = APCER().to(device)
        npcer_metric = NPCER().to(device)
        acer_metric = ACER().to(device)

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

        print("......................")
        print(f"Test APCER: {apcer}")
        print(f"Test NPCER: {npcer}")
        print(f"Test ACER: {acer}")

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_shape", type=tuple, default=(3,224,224))
#     parser.add_argument("--num_classes", type=int, default=2)
#     parser.add_argument("--modelname", type=str, default="seresnext50")
#     parser.add_argument("--train_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_training")
#     parser.add_argument("--test_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation")
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--max_epochs", type=int, default=10)
#     args = parser.parse_args()

#     print("=========================================")
#     print('\n'.join(f' + {k}={v}' for k, v in vars(args).items()))
#     print("=========================================")

#     training_pipeline(args)