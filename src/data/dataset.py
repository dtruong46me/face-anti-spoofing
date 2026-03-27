
"""
Face Anti-Spoofing Dataset Module using PyTorch Lightning.

This module implements a LightningDataModule for Face Anti-Spoofing datasets,
specifically designed for the LCC-FASD dataset with support for train/val/test splits.

Expected Dataset Structure:
    dataset_root/
    ├── LCC_FASD_training/
    │   ├── real/
    │   │   ├── real1.jpg
    │   │   └── ...
    │   └── fake/
    │       ├── fake1.jpg
    │       └── ...
    ├── LCC_FASD_development/
    │   ├── real/
    │   └── fake/
    └── LCC_FASD_evaluation/
        ├── real/
        └── fake/

Classes:
    LCCFASDataset: PyTorch Lightning DataModule for FAS dataset
    
Functions:
    load_data: Initialize dataset from arguments
    load_dataloader: Create dataloaders from dataset
"""

import os
import random
import sys
from typing import Tuple

import argparse
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torch import Generator
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS
)

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from data.load_data import ingest_data
from utils import load_transform_2


class LCCFASDataset(LightningDataModule):
    """
    PyTorch Lightning DataModule for Face Anti-Spoofing dataset.
    
    Handles loading, preprocessing, and splitting of LCC-FASD dataset
    into train/validation/test dataloaders. Implements standard LightningDataModule
    interface for seamless integration with PyTorch Lightning trainers.
    
    Attributes:
        train_path (str): Path to training dataset directory
        test_path (str): Path to test dataset directory
        batch_size (int): Batch size for dataloaders
        num_classes (int): Number of classes (binary: real/fake)
        train (Dataset): Training dataset after prepare_data()
        val (Dataset): Validation dataset after setup()
        test (Dataset): Test dataset after prepare_data()
        
    Example:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     train_path="data/train",
        ...     test_path="data/test",
        ...     batch_size=32,
        ...     num_classes=2
        ... )
        >>> datamodule = LCCFASDataset(args)
        >>> datamodule.prepare_data()
        >>> datamodule.setup(stage="fit")
        >>> train_loader = datamodule.train_dataloader()
    """
    
    # Training/Validation split ratio
    TRAIN_TEST_SPLIT_RATIO = 0.7
    RANDOM_SEED = 42
    NUM_WORKERS = 4
    
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize LCC-FASD DataModule.
        
        Args:
            args: Configuration namespace containing:
                - train_path (str): Path to training dataset
                - test_path (str): Path to test dataset
                - batch_size (int): Batch size for dataloaders
                - num_classes (int): Number of classes (default: 2)
                
        Raises:
            AttributeError: If required arguments are missing
        """
        super().__init__()
        
        try:
            self.train_path = args.train_path
            self.test_path = args.test_path
            self.batch_size = args.batch_size
            self.num_classes = getattr(args, 'num_classes', 2)
        except AttributeError as e:
            raise AttributeError(
                f"Missing required argument: {e}. "
                "Expected: train_path, test_path, batch_size"
            ) from e
        
        # Initialize dataset holders
        self.train = None
        self.val = None
        self.test = None
    
    def prepare_data(self) -> None:
        """
        Download and prepare data (called only once).
        
        This method loads training and test data with preprocessing transforms.
        Called automatically by trainer before training starts.
        
        Raises:
            RuntimeError: If data loading fails or paths are invalid
            
        Note:
            This method runs on single GPU/CPU and doesn't set state.
            Use setup() to create train/val split.
        """
        try:
            # Load preprocessing transforms
            preprocess = load_transform_2()
            
            # Load training data
            self.train = ingest_data(self.train_path, transform=preprocess)
            print(f"✓ Loaded training data: {len(self.train)} samples")
            
            # Load test data if path is provided
            if self.test_path and self.test_path.strip():
                self.test = ingest_data(self.test_path, transform=preprocess)
                print(f"✓ Loaded test data: {len(self.test)} samples")
            else:
                print("⚠ No test path provided, skipping test data")
                
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Dataset path not found: {e}. "
                f"Please check train_path={self.train_path} and test_path={self.test_path}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to prepare data: {e}") from e
    
    def setup(self, stage: str = "fit") -> None:
        """
        Create train/val split and prepare for training.
        
        This method is called after prepare_data() to set up dataloaders.
        It splits the training data into train/validation sets using
        a deterministic seed for reproducibility.
        
        Args:
            stage: Specify which data to use. Options:
                - "fit": Creates train/val split (called during training)
                - "test": Uses test dataset (called during testing)
                - "predict": Prepares prediction data
                Default: "fit"
                
        Raises:
            RuntimeError: If data was not prepared or split fails
            ValueError: If training data size is too small for split
            
        Note:
            Train/Val split ratio: {TRAIN_TEST_SPLIT_RATIO:.0%}/{1-TRAIN_TEST_SPLIT_RATIO:.0%}
            Seed for reproducibility: {RANDOM_SEED}
        """
        try:
            if stage == "fit":
                if self.train is None:
                    raise RuntimeError(
                        "Training data not loaded. "
                        "Call prepare_data() first"
                    )
                
                # Calculate split sizes
                train_size = int(self.TRAIN_TEST_SPLIT_RATIO * len(self.train))
                val_size = len(self.train) - train_size
                
                if train_size < 1 or val_size < 1:
                    raise ValueError(
                        f"Training data too small for split. "
                        f"Got {len(self.train)} samples but need at least 2"
                    )
                
                # Set seeds for reproducibility
                random.seed(self.RANDOM_SEED)
                np.random.seed(self.RANDOM_SEED)
                torch.manual_seed(self.RANDOM_SEED)
                
                # Create train/val split
                self.train, self.val = random_split(
                    self.train,
                    lengths=[train_size, val_size],
                    generator=Generator().manual_seed(self.RANDOM_SEED)
                )
                
                print(f"✓ Created train/val split: {train_size}/{val_size}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to setup data: {e}") from e
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Create training dataloader with shuffling.
        
        Returns:
            DataLoader: Training dataloader with shuffling enabled
            
        Raises:
            RuntimeError: If training data not prepared
            
        Note:
            Shuffles data, uses multiple workers for parallel loading
        """
        if self.train is None:
            raise RuntimeError(
                "Training data not loaded. Call setup() first"
            )
        
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.NUM_WORKERS
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create validation dataloader without shuffling.
        
        Returns:
            DataLoader: Validation dataloader
            
        Raises:
            RuntimeError: If validation data not prepared
            
        Note:
            Does not shuffle data, uses multiple workers
        """
        if self.val is None:
            raise RuntimeError(
                "Validation data not loaded. Call setup() first"
            )
        
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.NUM_WORKERS
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create test dataloader without shuffling.
        
        Returns:
            DataLoader: Test dataloader
            
        Raises:
            RuntimeError: If test data not prepared
            
        Note:
            Does not shuffle data, uses single worker
        """
        if self.test is None:
            raise RuntimeError(
                "Test data not loaded. Call prepare_data() with test_path first"
            )
        
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )


def load_data(args: argparse.Namespace) -> LCCFASDataset:
    """
    Initialize FAS dataset from arguments.
    
    Convenience function to create a LCCFASDataset instance
    with minimal setup.
    
    Args:
        args: Configuration namespace with dataset paths and parameters
        
    Returns:
        LCCFASDataset: Initialized Lightning DataModule
        
    Raises:
        RuntimeError: If initialization fails
        
    Example:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     train_path="data/LCC_FASD_training",
        ...     test_path="data/LCC_FASD_evaluation",
        ...     batch_size=32,
        ...     num_classes=2
        ... )
        >>> dataset = load_data(args)
    """
    try:
        dataset = LCCFASDataset(args)
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to initialize dataset: {e}") from e


def load_dataloader(
    dataset: LCCFASDataset
) -> Tuple[TRAIN_DATALOADERS, EVAL_DATALOADERS, EVAL_DATALOADERS]:
    """
    Create dataloaders from dataset.
    
    Convenience function that prepares data and creates all three
    dataloaders (train, validation, test) in one call.
    
    Args:
        dataset: Initialized LCCFASDataset instance
        
    Returns:
        Tuple containing:
        - train_loader (TRAIN_DATALOADERS): Training dataloader
        - val_loader (EVAL_DATALOADERS): Validation dataloader
        - test_loader (EVAL_DATALOADERS): Test dataloader
        
    Raises:
        RuntimeError: If dataloader creation fails
        
    Example:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     train_path="data/train",
        ...     test_path="data/test",
        ...     batch_size=32,
        ...     num_classes=2
        ... )
        >>> dataset = load_data(args)
        >>> train_loader, val_loader, test_loader = load_dataloader(dataset)
        >>> 
        >>> # Training loop example:
        >>> for batch in train_loader:
        ...     images, labels = batch
        ...     break  # Just get first batch
    """
    try:
        dataset.prepare_data()
        dataset.setup(stage="fit")
        
        train_loader = dataset.train_dataloader()
        val_loader = dataset.val_dataloader()
        test_loader = dataset.test_dataloader()
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        raise RuntimeError(f"Failed to create dataloaders: {e}") from e