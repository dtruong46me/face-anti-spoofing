"""
Data visualization module for Face Anti-Spoofing datasets.

Provides utilities for visualizing image samples and labels from
FAS datasets with matplotlib. Useful for EDA and verifying
dataset quality.

Functions:
    visualize_data: Display image samples in grid with labels
    visualize_from_path: Load dataset and visualize samples
"""

import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from data.load_data import IngestData
from utils import load_transform_2


def visualize_data(
    data_loader: DataLoader,
    num_images: int = 5,
    save_path: str = "visualize_images.jpg"
) -> None:
    """
    Visualize image samples from dataloader.
    
    Displays images and their labels in a grid layout using matplotlib.
    Useful for exploring dataset contents and verifying preprocessing.
    
    Args:
        data_loader: PyTorch DataLoader providing (images, labels) tuples
        num_images: Number of images to display (default: 5)
        save_path: Path to save visualization image (default: visualize_images.jpg)
        
    Returns:
        None. Displays plot and saves to file.
        
    Raises:
        ValueError: If num_images <= 0 or greater than batch size
        RuntimeError: If dataloader is empty or data extraction fails
        
    Note:
        - Assumes images are in CHW format (channels, height, width)
        - Converts to HWC format for visualization
        - Closes figures after display to prevent memory leaks
        - Labels shown in format [real_prob, fake_prob]
        
    Example:
        >>> from torch.utils.data import DataLoader
        >>> from torchvision import transforms
        >>> from load_data import IngestData
        >>> 
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>> 
        >>> dataset = IngestData("path/to/data", transform=transform)
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> 
        >>> # Display 10 images
        >>> visualize_data(loader, num_images=10, save_path="my_vis.jpg")
    """
    try:
        if num_images <= 0:
            raise ValueError(f"num_images must be positive, got {num_images}")
        
        # Get iterator from dataloader
        data_iter = iter(data_loader)
        
        # Create subplot grid
        fig, axes = plt.subplots(
            1,
            num_images,
            figsize=(4 * num_images, 5)
        )
        
        # Handle single image case (axes is not array)
        if num_images == 1:
            axes = [axes]
        
        # Visualize samples
        for i in range(num_images):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                print(f"✓ Dataloader contained only {i} samples, displaying {i} images")
                fig.suptitle(f"Visualized {i}/{num_images} samples", fontsize=14)
                break
            
            # Get first image from batch
            image = images[0]
            label = labels[0]
            
            # Convert from CHW tensor to HWC numpy array
            # Handle both normalized and non-normalized images
            image_np = image.permute(1, 2, 0).numpy()
            
            # Clip values to [0, 1] for display (in case of normalization)
            image_np = np.clip(image_np, 0, 1)
            
            # Get class from label (max probability)
            label_np = label.numpy()
            is_real = label_np[0] > label_np[1]
            class_name = "Real" if is_real else "Fake"
            
            # Plot image
            ax = axes[i]
            ax.imshow(image_np)
            ax.set_title(
                f"{i+1}. {class_name}\nLabel: {label_np.round(3)}\n"
                f"Real prob: {label_np[0]:.3f}, Fake prob: {label_np[1]:.3f}",
                fontsize=10
            )
            ax.axis('off')
        
        # Format layout
        fig.suptitle(f"Dataset Visualization ({num_images} samples)", fontsize=14)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
        
        # Display plot
        plt.show()
        plt.close(fig)
        
    except Exception as e:
        raise RuntimeError(f"Failed to visualize data: {e}") from e


def visualize_from_path(
    datapath: str,
    batch_size: int = 32,
    num_images: int = 5,
    save_path: str = "visualize_images.jpg"
) -> None:
    """
    Load dataset and visualize samples in one function.
    
    Convenience function combining data loading and visualization.
    Useful for quick dataset exploration from command line.
    
    Args:
        datapath: Path to dataset root directory with real/ and fake/ subdirs
        batch_size: Batch size for DataLoader (default: 32)
        num_images: Number of images to display (default: 5)
        save_path: Path to save visualization (default: visualize_images.jpg)
        
    Returns:
        None. Displays and saves visualization.
        
    Raises:
        FileNotFoundError: If dataset path doesn't exist
        RuntimeError: If data loading or visualization fails
        
    Example:
        >>> # Quick visualization from command line
        >>> visualize_from_path(
        ...     datapath="data/LCC_FASD_training",
        ...     batch_size=16,
        ...     num_images=8
        ... )
    """
    try:
        print(f"Loading dataset from: {datapath}")
        
        # Load preprocessing transforms
        transform = load_transform_2()
        
        # Create dataset
        dataset = IngestData(datapath, transform)
        print(f"✓ Loaded {len(dataset)} samples")
        
        # Create dataloader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Visualize data
        visualize_data(data_loader, num_images=num_images, save_path=save_path)
        
    except FileNotFoundError as e:
        print(f"✗ Dataset path not found: {datapath}")
        raise e
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        raise RuntimeError(f"Failed to visualize dataset: {e}") from e


if __name__ == "__main__":
    """Command-line interface for dataset visualization."""
    
    parser = argparse.ArgumentParser(
        description="Visualize Face Anti-Spoofing dataset samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_data.py --datapath data/LCC_FASD_training
  python visualize_data.py --datapath data/LCC_FASD_training --num_images 10
  python visualize_data.py --datapath data/train --batch_size 16 --save_path output.png
        """
    )
    
    parser.add_argument(
        '--datapath',
        type=str,
        required=True,
        help='Path to dataset root directory (required)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for DataLoader (default: 32)'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=5,
        help='Number of images to visualize (default: 5)'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='visualize_images.jpg',
        help='Path to save visualization image (default: visualize_images.jpg)'
    )
    
    args = parser.parse_args()
    
    # Run visualization
    visualize_from_path(
        datapath=args.datapath,
        batch_size=args.batch_size,
        num_images=args.num_images,
        save_path=args.save_path
    )
