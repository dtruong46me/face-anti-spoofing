"""
Training pipeline for Face Anti-Spoofing models.
Handles model training, validation, and evaluation with metrics.
"""

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
from utils import load_backbone
from data.dataset import load_data, load_dataloader
from metrics.apcer import APCER
from metrics.npcer import NPCER
from metrics.acer import ACER
from metrics.accuracy import MyAccuracy
from metrics.recall import MyRecall


def training_pipeline(args: argparse.Namespace) -> None:
    """
    Complete training pipeline for Face Anti-Spoofing models.
    
    Args:
        args: Training configuration arguments from argparse
    """
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     Face Anti-Spoofing Model Training Pipeline             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # ==================== DATA LOADING ====================
    print("\n[1/5] Loading dataset...")
    data = load_data(args)
    train_loader, val_loader, test_loader = load_dataloader(data)
    print(f"  ✓ Dataset loaded successfully")
    
    # ==================== DEVICE SETUP ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2/5] Device Setup")
    print(f"  ✓ Using device: {device}")
    
    # ==================== MODEL LOADING ====================
    print(f"\n[3/5] Loading Model: {args.modelname}")
    backbone = load_backbone(args)
    model = load_model(
        backbone=backbone,
        input_shape=args.input_shape,
        num_classes=args.num_classes
    )
    model.to(device)
    print(f"  ✓ Model loaded successfully")
    
    # ==================== LOGGING SETUP ====================
    print(f"\n[4/5] Setting up Logging...")
    if args.wandb_token:
        wandb.login(key=args.wandb_token)
        logger = WandbLogger(
            name=args.wandb_runname,
            project="cv-project"
        )
        print(f"  ✓ Weights & Biases configured")
    else:
        logger = None
        print(f"  ⚠ No W&B token provided - logging disabled")
    
    # ==================== TRAINING ====================
    print(f"\n[5/5] Starting Training...")
    print(f"  - Epochs: {args.max_epochs}")
    print(f"  - Batch size: {args.batch_size}")
    
    # Setup callbacks
    es_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode="min"
    )
    
    ckpt_callback = ModelCheckpoint(
        dirpath='checkpoint',
        filename=args.modelname,
        save_top_k=2,
        verbose=True,
        mode='min',
        monitor="val/acer"
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[ckpt_callback, es_callback],
        logger=logger,
        enable_progress_bar=True
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    best_model_path = ckpt_callback.best_model_path
    print(f"\n  ✓ Training completed")
    print(f"  ✓ Best model saved at: {best_model_path}")
    
    # ==================== EVALUATION ====================
    print(f"\n[EVAL] Evaluating on Test Set...")
    
    # Reload best model
    model = ModelInterface.load_from_checkpoint(
        best_model_path,
        model=backbone,
        input_shape=args.input_shape,
        num_classes=args.num_classes
    )
    model.to(device)
    model.eval()
    
    # Initialize metrics
    apcer_metric = APCER().to(device)
    npcer_metric = NPCER().to(device)
    acer_metric = ACER().to(device)
    accuracy_metric = MyAccuracy().to(device)
    recall_metric = MyRecall().to(device)
    
    # Run evaluation on test set
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
    
    # Calculate metrics
    apcer = apcer_metric(all_preds, all_labels)
    npcer = npcer_metric(all_preds, all_labels)
    acer = acer_metric(all_preds, all_labels)
    accuracy = accuracy_metric(all_preds, all_labels)
    recall = recall_metric(all_preds, all_labels)
    
    # Calculate confusion matrix
    pred_labels = torch.argmax(all_preds, dim=1)
    true_labels = torch.argmax(all_labels, dim=1)
    
    true_pos = torch.sum((pred_labels == 1) & (true_labels == 1)).float().item()
    true_neg = torch.sum((pred_labels == 0) & (true_labels == 0)).float().item()
    false_pos = torch.sum((pred_labels == 1) & (true_labels == 0)).float().item()
    false_neg = torch.sum((pred_labels == 0) & (true_labels == 1)).float().item()
    
    total_images = all_labels.shape[0]
    num_fake = torch.sum(true_labels == 1).float().item()
    num_real = torch.sum(true_labels == 0).float().item()
    
    # ==================== RESULTS ====================
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"\n📊 Metrics:")
    print(f"  APCER (Attack Presentation Classification Error Rate): {apcer:.4f}")
    print(f"  NPCER (Normal Presentation Classification Error Rate): {npcer:.4f}")
    print(f"  ACER  (Average Classification Error Rate): {acer:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Recall:   {recall:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"  True Positives (Fake detected):  {int(true_pos)}")
    print(f"  True Negatives (Real detected):  {int(true_neg)}")
    print(f"  False Positives (Real→Fake):     {int(false_pos)}")
    print(f"  False Negatives (Fake→Real):     {int(false_neg)}")
    
    print(f"\n📋 Dataset Statistics:")
    print(f"  Total test images:               {int(total_images)}")
    print(f"  Fake images (class 1):           {int(num_fake)}")
    print(f"  Real images (class 0):           {int(num_real)}")
    
    print("="*60 + "\n")
    
    print("✅ Training pipeline completed successfully!")

 
