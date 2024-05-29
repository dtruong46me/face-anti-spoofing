import sys
import os
import argparse
sys.path.insert(1, os.getcwd())
from utility.video_utils import VideoUtils
from face_detector import FaceDetector
import numpy as np
import tensorflow as tf
import cv2
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

from src.models.ln_model import ModelInterface
from src.models.resnext50 import SEResNeXT50
from src.utils import load_transform, load_transform_2, load_backbone
import torch
from PIL import Image
preprocess = load_transform()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def load_model(args):

    backbone = load_backbone(args)

    # Load model from path
    model = ModelInterface.load_from_checkpoint(args.model_checkpoint, 
                                                model=backbone,
                                                input_shape=args.input_shape, 
                                                num_classes=args.num_classes)
    
    model.to(device)

    model.eval()

    return model 

def fas(model, image) :
    # Apply the transformations to the input image

    image = preprocess(image).unsqueeze(0)
    image = image.to(device)

    
    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

    # Get the predicted class and the associated probabilities
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_probabilities = probabilities.squeeze().cpu().numpy()

    if predicted_class == 0:
        result = {'tensor': [1, 0], 'class': 0, 'label': 'real', 'probability': predicted_probabilities[0]}
    else:
        result = {'tensor': [0, 1], 'class': 1, 'label': 'fake', 'probability': predicted_probabilities[1]}


    return result



print("[INFO] Loading Face Detector")
face_detector = FaceDetector()

print("[INFO] Starting video stream...")


def face_detection_with_liveness_check(model):
    # Initialize video stream
    video = cv2.VideoCapture(0)
    while True:
        if not video.isOpened():
            print("[ERROR] Cannot find webcam")
            pass

        # Read frame from webcam
        ret, frame = video.read()

        # Call face detector to obtain face image
        frame_bgr = frame[..., ::-1]
        boxes = face_detector(np.array(frame_bgr))

        # Check if face is present
        n = len(boxes)
        if n == 0:
            cv2.putText(frame, "Faces: %s" % (n), (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        else:
            cv2.putText(frame, "Faces: %s" % (n), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            image = VideoUtils.get_image(frame, boxes)
            image = image[..., ::-1]
            face_image = Image.fromarray(image)  
            # Call the function predict_fas and get the result
            result = fas(model, face_image)

            # Get the label and probability from the result
            label = result['label']
            probability = result['probability']

            # Create the display text
            display_text = f"{label} ({probability:.2f})"

            # Determine the color based on the label
            color = (0, 0, 200) if label == 'real' else (0, 200, 0)

            # Display on the frame
            cv2.putText(frame, f"FAS: {display_text}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
            cv2.rectangle(frame, (boxes[0][3], boxes[0][2]), (boxes[0][1], boxes[0][0]), (255, 0, 0), 2)

        # Display frame
        cv2.imshow("FAS Detector", frame)

        # Reduce frame rate for slower detection
        key = cv2.waitKey(15)
        if key == ord('q'):
            face_image.save('output_image.png')
            print("save")
            break

    # Release video stream and close windows
    video.release()
    cv2.destroyAllWindows()

def face_detection_with_liveness(model):
    # Initialize video stream
    video = cv2.VideoCapture(0)
    while True:
        if not video.isOpened():
            print("[ERROR] Cannot find webcam")
            pass

        # Read frame from webcam
        ret, frame = video.read()

        # Call face detector to obtain face image
        frame_bgr = frame[..., ::-1]
        boxes = face_detector(np.array(frame_bgr))

        # Check if face is present
        n = len(boxes)
        if n == 0:
            cv2.putText(frame, "Faces: %s" % (n), (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        else:
            cv2.putText(frame, "Faces: %s" % (n), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            face_image = VideoUtils.get_liveness_score(frame, boxes)
            liveness_score = (np.round(model.predict(face_image)[:, 1].tolist()[0], 3))
            cv2.putText(frame, "Liveness: %s" % liveness_score, (20, 30), cv2.FONT_HERSHEY_DUPLEX,0.7, (0, 0, 200) if liveness_score < 0.7 else (0, 200, 0), 1)
            cv2.rectangle(frame, (boxes[0][3], boxes[0][2]), (boxes[0][1], boxes[0][0]), (255, 0, 0), 2)

        # Display frame
        cv2.imshow("FAS Detector", frame)

        # Reduce frame rate for slower detection
        key = cv2.waitKey(15)
        if key == ord('q'):
            break

    # Release video stream and close windows
    video.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default= r"FAS_detector\model\seresnext50_v0.ckpt")
    parser.add_argument("--modelname", type=str, default="seresnext50")
    parser.add_argument("--input_shape", type=tuple, default=(3,224,224))
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()
    FAS_MODEL_PATH = r"FAS_detector\model\antispoofing.h5"
    #model = VideoUtils.load_keras_model(FAS_MODEL_PATH)
    #face_detection_with_liveness(model)
    model = load_model(args)
    face_detection_with_liveness_check(model)

if __name__=="__main__":
    main()
