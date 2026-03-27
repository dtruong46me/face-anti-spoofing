
"""
Data loading and ingestion module for Face Anti-Spoofing dataset.

This module implements custom Dataset wrappers for loading LCC-FASD
dataset files with preprocessing transforms and label formatting.

Classes:
    IngestData: Custom PyTorch Dataset for image classification with
                binary label encoding
                
Functions:
    ingest_data: Convenience function to create IngestData instance
"""

from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class IngestData(Dataset):
    """
    Custom PyTorch Dataset for Face Anti-Spoofing binary classification.
    
    Loads images from directory structure with categorical subdirectories
    (real/ and fake/) and converts labels to 2-element tensor format:
    - Real faces: [1, 0]
    - Fake/spoofed faces: [0, 1]
    
    This format is compatible with binary cross-entropy loss functions
    and provides explicit class probabilities.
    
    Attributes:
        datapath (str): Root directory containing class subdirectories
        transform: torchvision.transforms composition for preprocessing
        data (ImageFolder): Underlying ImageFolder dataset instance
        
    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>> dataset = IngestData("path/to/data", transform=transform)
        >>> image, label = dataset[0]
        >>> print(image.shape, label.shape)
        torch.Size([3, 224, 224]) torch.Size([2])
        >>> print(label)  # Example: real face
        tensor([1., 0.])
    """
    
    # Class mapping constants
    REAL_LABEL = torch.tensor([1.0, 0.0])  # Real face
    FAKE_LABEL = torch.tensor([0.0, 1.0])  # Spoofed face
    
    # ImageFolder class index convention: 0=real, 1=fake (alphabetical)
    REAL_CLASS_INDEX = 0
    FAKE_CLASS_INDEX = 1
    
    def __init__(
        self,
        datapath: str,
        transform: Optional[object] = None
    ) -> None:
        """
        Initialize IngestData dataset.
        
        Args:
            datapath: Root path to dataset with structure:
                datapath/
                ├── real/
                │   ├── real1.jpg
                │   └── ...
                └── fake/
                    ├── fake1.jpg
                    └── ...
                    
            transform: Optional torchvision transforms composition for
                      image preprocessing (resize, normalize, etc.)
                      
        Raises:
            FileNotFoundError: If datapath doesn't exist
            RuntimeError: If directory structure is invalid
            
        Raises:
            RuntimeError: If unable to load images from directory
        """
        try:
            self.datapath = datapath
            self.transform = transform
            
            # Load images using torchvision's ImageFolder
            # This expects subdirectories named after classes
            self.data = ImageFolder(self.datapath, transform=self.transform)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset path not found: {datapath}. "
                f"Please check directory exists with real/ and fake/ subdirectories"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset from {datapath}: {e}"
            ) from e
    
    def __len__(self) -> int:
        """
        Get total number of samples in dataset.
        
        Returns:
            int: Total number of images in dataset
        """
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Get single sample from dataset.
        
        Loads image at given index and converts class label to
        2-element probability tensor:
        - Index 0: Real face label
        - Index 1: Spoofed face label
        
        Args:
            index: Index of sample to retrieve (0-based)
            
        Returns:
            Tuple containing:
            - image (Tensor): Preprocessed image tensor of shape (C, H, W)
            - label (Tensor): Binary label tensor [1, 0] or [0, 1]
                - [1, 0]: Real face (genuine)
                - [0, 1]: Fake/spoofed face (presentation attack)
                
        Raises:
            IndexError: If index is out of bounds
            
        Example:
            >>> dataset = IngestData("data/train", transform=transform)
            >>> image, label = dataset[0]
            >>> print(f"Label shape: {label.shape}")
            Label shape: torch.Size([2])
            >>> print(f"Is spoofed: {label[1].item()}")
            Is spoofed: 0.0  # Real face
        """
        try:
            # Get image and class index from ImageFolder
            image, class_index = self.data[index]
            
            # Convert class index to 2-element tensor
            # ImageFolder loads classes in alphabetical order:
            # 0: fake (comes first alphabetically)
            # 1: real
            # But we want 0=real, 1=fake in our encoding
            if class_index == self.REAL_CLASS_INDEX:
                label = self.REAL_LABEL.clone()
            else:
                label = self.FAKE_LABEL.clone()
            
            return image, label
            
        except IndexError:
            raise IndexError(
                f"Index {index} out of range. "
                f"Dataset has {len(self)} samples"
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"Failed to load sample at index {index}: {e}"
            ) from e


def ingest_data(
    datapath: str,
    transform: Optional[object] = None
) -> IngestData:
    """
    Convenience function to create IngestData instance.
    
    Wrapper around IngestData constructor for cleaner API.
    
    Args:
        datapath: Path to dataset root directory containing
                 real/ and fake/ subdirectories
        transform: Optional preprocessing transforms composition
        
    Returns:
        IngestData: Initialized dataset instance ready for loading
        
    Raises:
        FileNotFoundError: If datapath doesn't exist
        RuntimeError: If dataset cannot be loaded
        
    Example:
        >>> from torchvision import transforms
        >>> from torch.utils.data import DataLoader
        >>> 
        >>> # Setup transforms for image preprocessing
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize(
        ...         mean=[0.485, 0.456, 0.406],
        ...         std=[0.229, 0.224, 0.225]
        ...     )
        ... ])
        >>> 
        >>> # Load dataset
        >>> dataset = ingest_data("data/LCC_FASD_training", transform=transform)
        >>> print(f"Loaded {len(dataset)} training samples")
        Loaded 5000 training samples
        >>> 
        >>> # Create dataloader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for images, labels in loader:
        ...     print(f"Batch shape: {images.shape}, {labels.shape}")
        ...     break
        Batch shape: torch.Size([32, 3, 224, 224]), torch.Size([32, 2])
    """
    try:
        print(f"✓ Ingesting data from: {datapath}")
        
        ingested_dataset = IngestData(datapath, transform)
        
        print(f"✓ Successfully loaded {len(ingested_dataset)} samples")
        
        return ingested_dataset
        
    except FileNotFoundError as e:
        print(f"✗ Data path not found: {datapath}")
        raise e
    except Exception as e:
        print(f"✗ Failed to ingest data: {e}")
        raise RuntimeError(f"Data ingestion failed: {e}") from e