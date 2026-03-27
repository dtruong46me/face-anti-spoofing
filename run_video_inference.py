"""
Real-time Face Anti-Spoofing inference on video stream.
Performs face detection and liveness detection on webcam input.
"""

import sys
import os
import argparse
import numpy as np

import cv2
import torch

sys.path.insert(0, os.getcwd())

from FAS_detector.face_detector import FaceDetector
from FAS_detector.utility.video_utils import VideoUtils
from src.predictor import FASPredictor
from src.config import get_video_args


class VideoProcessor:
    """
    Processes video stream for face anti-spoofing detection.
    
    Attributes:
        face_detector: Face detector for finding faces in frames
        predictor: FAS predictor for liveness detection
    """
    
    def __init__(self, pretrained_model_path: str, modelname: str):
        """
        Initialize video processor.
        
        Args:
            pretrained_model_path: Path to FAS model checkpoint
            modelname: Model architecture name
        """
        print("[INFO] Loading Face Detector...")
        self.face_detector = FaceDetector()
        
        print("[INFO] Loading FAS Predictor...")
        self.predictor = FASPredictor(
            model_checkpoint=pretrained_model_path,
            modelname=modelname
        )
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame for face detection and liveness check.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Tuple of (processed_frame, detection_results)
        """
        # Detect faces
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = self.face_detector(np.array(frame_rgb))
        
        # Get number of detected faces
        num_faces = len(boxes)
        
        # Display face count
        cv2.putText(
            frame,
            f"Faces Detected: {num_faces}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if num_faces > 0 else (0, 0, 255),
            2
        )
        
        results = []
        
        # Process each detected face
        if num_faces > 0:
            face_image = VideoUtils.get_image(frame, boxes)
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Run liveness detection (using first detected face)
            from PIL import Image
            pil_image = Image.fromarray(face_image_rgb)
            result = self.predictor.predict(pil_image)
            results.append(result)
            
            # Draw bounding boxes and predictions
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                color = (0, 255, 0) if result['label'] == 'real' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw prediction label
                label_text = f"{result['label'].upper()}: {result['probability']:.2%}"
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        
        return frame, results
    
    def run(self, camera_id: int = 0) -> None:
        """
        Run video stream processing from webcam.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
        """
        print("[INFO] Starting video stream...")
        video = cv2.VideoCapture(camera_id)
        
        if not video.isOpened():
            print("[ERROR] Cannot open webcam. Please check your camera.")
            return
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video resolution: {width}x{height} @ {fps:.0f} FPS")
        print("[INFO] Press 'q' to exit")
        
        try:
            while True:
                ret, frame = video.read()
                
                if not ret:
                    print("[ERROR] Failed to read frame from video stream")
                    break
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("Face Anti-Spoofing Detection", processed_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Exiting...")
                    break
        
        finally:
            video.release()
            cv2.destroyAllWindows()
            print("[INFO] Video stream closed")


def main():
    """Main entry point for video inference."""
    args = get_video_args()
    
    # Initialize and run processor
    processor = VideoProcessor(
        pretrained_model_path=args.model_checkpoint,
        modelname=args.modelname
    )
    
    processor.run(camera_id=args.camera_id)


if __name__ == "__main__":
    main()
