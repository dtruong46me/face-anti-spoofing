"""
Face detection module using FaceBoxes architecture.

This module provides face detection capabilities using the
FaceBoxes approach optimized for detecting faces at various scales.
"""

import os
import sys
from typing import List

import numpy as np

# Add face detection path
__root__ = os.getcwd()
sys.path.insert(0, __root__)

from FAS_detector.face_det.FaceBoxes import FaceBoxes

# Configuration
USE_GPU = True
USE_ONNX = False

# Initialize face detector
if USE_ONNX:
    # Set environment variables for ONNX
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from face_det.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    _detector = FaceBoxes_ONNX()
else:
    _detector = FaceBoxes(gpu_mode=USE_GPU)


class FaceDetector:
    """
    Face detection using FaceBoxes architecture.
    
    Detects faces in images and returns bounding box coordinates.
    Supports both GPU and CPU inference.
    
    Attributes:
        detector: Internal FaceBoxes detector instance
        
    Example:
        >>> import cv2
        >>> from FAS_detector.face_detector import FaceDetector
        >>> 
        >>> detector = FaceDetector()
        >>> image = cv2.imread("face.jpg")
        >>> boxes = detector(image)
        >>> print(f"Detected {len(boxes)} faces")
        >>> for x1, y1, x2, y2 in boxes:
        ...     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    """
    
    def __init__(self) -> None:
        """Initialize face detector."""
        self.detector = _detector
    
    def __call__(
        self,
        image: np.ndarray,
        dense_flag: bool = False
    ) -> List[List[int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (HxWx3) in BGR or RGB format
            dense_flag: (Deprecated) Not used, kept for compatibility
            
        Returns:
            List of bounding boxes in format [x1, y1, x2, y2] where:
            - (x1, y1) is the top-left corner
            - (x2, y2) is the bottom-right corner
            
        Raises:
            ValueError: If image is empty or invalid
            
        Example:
            >>> import numpy as np
            >>> detector = FaceDetector()
            >>> image = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> boxes = detector(image)
            >>> print(type(boxes))
            <class 'list'>
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid")
        
        # Detect faces using FaceBoxes
        # Output format from FaceBoxes: [left, top, right, bottom, confidence]
        detections = self.detector(image)
        
        # Reformat to [x1, y1, x2, y2] (top-left, bottom-right)
        # Format: [left, top, right, bottom] from [left, top, right, bottom, confidence]
        boxes = np.array([
            [left, top, right, bottom]
            for left, top, right, bottom, _ in detections
        ]).astype(int).tolist()
        
        return boxes
    
    def detect_with_confidence(
        self,
        image: np.ndarray
    ) -> List[tuple]:
        """
        Detect faces with confidence scores.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of tuples (bbox, confidence) where:
            - bbox: [x1, y1, x2, y2]
            - confidence: Detection confidence score [0-1]
            
        Example:
            >>> detector = FaceDetector()
            >>> results = detector.detect_with_confidence(image)
            >>> for bbox, conf in results:
            ...     print(f"Face detected with {conf:.2%} confidence")
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid")
        
        # Get raw detections from FaceBoxes
        detections = self.detector(image)
        
        # Extract boxes and confidence scores
        # Format: [left, top, right, bottom] from [left, top, right, bottom, confidence]
        results = [
            ([left, top, right, bottom], conf)
            for left, top, right, bottom, conf in detections
        ]
        
        return results
