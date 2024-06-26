import argparse

import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch

import wandb

from src.models.feathernet import FeatherNetB
from src.models.mobilenet import MobileNetV3

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from src.models.ln_model import load_model, ModelInterface
from src.models.resnext50 import SEResNeXT50
from src.models.ShuffleNet import ShuffleNet

from models.MobileLiteNet import *
from models.feathernet import *

from data.dataset import load_data, load_dataloader

from utils import load_backbone

from metrics.apcer import APCER
from metrics.npcer import NPCER
from metrics.acer import ACER
from metrics.accuracy import MyAccuracy
from metrics.recall import MyRecall


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


    # Load SEResNeXT50
    if args.modelname == "seresnext50":
        backbone = SEResNeXT50(args.input_shape, args.num_classes)
    
    # Load MobileNetV2
    if args.modelname == "mobilenetv3":
        backbone = MobileNetV3()
    
    # Load FeatherNet
    if args.modelname == "feathernet":
        backbone = FeatherNetB()

    if args.modelname == "shufflenet":
        backbone = ShuffleNet()

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
    es_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=5, verbose=True, mode="min")

    ckpt_callback = ModelCheckpoint(
        dirpath='checkpoint',
        filename=args.modelname,
        save_top_k=2,
        verbose=True,
        mode='min',
        monitor="val/acer"
    )

    # Load trainer
    trainer = Trainer(max_epochs=args.max_epochs,
                      callbacks=[ckpt_callback, es_callback],
                      logger=logger)
    
    trainer.fit(model, train_loader, val_loader)

    # Save best model checkpoint
    best_model_path = ckpt_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    # Load model from path
    model = ModelInterface.load_from_checkpoint(best_model_path, 
                                                model=backbone,
                                                input_shape=args.input_shape, 
                                                num_classes=args.num_classes)
    model.to(device)

    model.eval()
    apcer_metric = APCER().to(device)
    npcer_metric = NPCER().to(device)
    acer_metric = ACER().to(device)
    accuracy = MyAccuracy().to(device)
    recall = MyRecall().to(device)

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

    total_image = all_labels.shape[0]
    positive_image = torch.sum(torch.argmax(all_labels, dim=1)).float()
    negative_image = (total_image - positive_image).float()

    apcer = apcer_metric(all_preds, all_labels)
    npcer = npcer_metric(all_preds, all_labels)
    acer = acer_metric(all_preds, all_labels)
    acc = accuracy(all_preds, all_labels)
    rec = recall(all_preds, all_labels)

    print("+++++++++++++++++++++++++")
    print(f"Test APCER: {apcer}")
    print(f"Test NPCER: {npcer}")
    print(f"Test ACER: {acer}")
    print(f"Test Accuracy: {acc}")
    print(f"Test Recall: {rec}")

    true_pos = torch.sum((torch.argmax(all_preds, dim=1)==1) & (torch.argmax(all_labels, dim=1)==1)).float()
    true_neg = torch.sum((torch.argmax(all_preds, dim=1)==0) & (torch.argmax(all_labels, dim=1)==0)).float()
    false_pos = torch.sum((torch.argmax(all_preds, dim=1)==1) & (torch.argmax(all_labels, dim=1)==0)).float()
    false_neg = torch.sum((torch.argmax(all_preds, dim=1)==0) & (torch.argmax(all_labels, dim=1)==1)).float()

    true_pos = int(true_pos.item())
    true_neg = int(true_neg.item())
    false_pos = int(false_pos.item())
    false_neg = int(false_neg.item())

    print("+++++++++++++++++++++++++")
    print("[+] Total test images:", total_image)
    print("[+] Positive - Fake [1]:", int(positive_image.item()))
    print("[+] Negative - Real [0]:", int(negative_image.item()))
    print("+TP:", true_pos, 
          "\n+TN:", true_neg,
          "\n+FP:", false_pos,
          "\n+FN:", false_neg)
 
