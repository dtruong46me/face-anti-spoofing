"""
Real-time face anti-spoofing detection module.
Provides video stream processing capabilities for webcam-based liveness detection.
"""

import sys
import os
import argparse
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from FAS_detector.face_detector import FaceDetector
from FAS_detector.utility.video_utils import VideoUtils
from src.predictor import FASPredictor
from src.config import get_video_args


class FASDetector:
    """
    Face Anti-Spoofing detector with video stream support.
    
    Attributes:
        face_detector: Face detector module
        predictor: FAS predictor for liveness detection
    """
    
    def __init__(self, modelname: str = "feathernet", 
                 model_checkpoint: str = "") -> None:
        """
        Initialize FAS detector.
        
        Args:
            modelname: Model architecture name
            model_checkpoint: Path to model checkpoint file
        """
        print("[INFO] Loading Face Detector...")
        self.face_detector = FaceDetector()
        
        if model_checkpoint is None:
            model_checkpoint = os.path.join(
                os.path.dirname(__file__),
                "model",
                f"{modelname}.ckpt"
            )
        
        print("[INFO] Loading FAS Predictor...")
        self.predictor = FASPredictor(
            model_checkpoint=model_checkpoint,
            modelname=modelname
        )
    
    def detect_and_predict(self, frame: np.ndarray) -> tuple:
        """
        Detect faces and predict liveness in a single frame.
        
        Args:
            frame: Input frame in BGR format (OpenCV)
            
        Returns:
            Tuple of (annotated_frame, predictions)
        """
        # Convert BGR to RGB for face detector
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = self.face_detector(np.array(frame_rgb))
        
        # Display face detection count
        num_faces = len(boxes)
        cv2.putText(
            frame,
            f"Faces Detected: {num_faces}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if num_faces > 0 else (0, 0, 255),
            2
        )
        
        predictions = []
        
        if num_faces > 0:
            # Extract face region and run liveness check
            face_image = VideoUtils.get_image(frame, boxes)
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB) # type: ignore
            pil_image = Image.fromarray(face_rgb)
            result = self.predictor.predict(pil_image)
            predictions.append(result)
            
            # Annotate frame with results
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Choose color based on prediction
                color = (0, 255, 0) if result['label'] == 'real' else (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label_text = (
                    f"{result['label'].upper()} "
                    f"({result['probability']:.2%})"
                )
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        
        return frame, predictions
    
    def run(self, camera_id: int = 0, save_output: bool = False) -> None:
        """
        Run real-time face anti-spoofing detection on video stream.
        
        Args:
            camera_id: Camera device ID (0 for default)
            save_output: Whether to save frames on exit
        """
        print("[INFO] Starting video stream...")
        video = cv2.VideoCapture(camera_id)
        
        if not video.isOpened():
            print("[ERROR] Cannot open camera. Check camera connection.")
            return
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video stream: {width}x{height} @ {fps:.0f} FPS")
        print("[INFO] Press 'q' to quit, 's' to save current frame")
        
        output_image = None
        
        try:
            while True:
                ret, frame = video.read()
                
                if not ret:
                    print("[ERROR] Failed to read frame from stream")
                    break
                
                # Process frame
                annotated_frame, predictions = self.detect_and_predict(frame)
                
                # Display result
                cv2.imshow("Face Anti-Spoofing Detection", annotated_frame)
                output_image = annotated_frame
                
                # Handle key presses
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("[INFO] Stopping...")
                    break
                elif key == ord('s'):
                    cv2.imwrite("fas_output.jpg", annotated_frame)
                    print("[INFO] Frame saved to fas_output.jpg")
        
        finally:
            video.release()
            cv2.destroyAllWindows()
            
            if save_output and output_image is not None:
                cv2.imwrite("fas_output.jpg", output_image)
                print("[INFO] Output saved to fas_output.jpg")
            
            print("[INFO] Stream closed")


def main():
    """Main entry point for FAS detector."""
    args = get_video_args()
    
    # Initialize detector
    detector = FASDetector(
        modelname=args.modelname,
        model_checkpoint=args.model_checkpoint
    )
    
    # Run detection
    detector.run(camera_id=args.camera_id)


if __name__ == "__main__":
    main()
