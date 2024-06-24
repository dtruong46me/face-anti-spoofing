import cv2
import numpy as np
import os
import sys
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
def add_padding_and_resize(image, target_size=(224, 224), padding_color=(0, 255, 0)):
        # Get original image dimensions
        h, w, _ = image.shape
        max_dim = max(h, w)

        # Create a new image with padding color
        padded_image = np.full((max_dim, max_dim, 3), padding_color, dtype=np.uint8)

        # Calculate starting points
        start_x = (max_dim - w) // 2
        start_y = (max_dim - h) // 2

        # Place the original image in the center
        padded_image[start_y:start_y + h, start_x:start_x + w] = image

        # Resize to target size
        image_resized = cv2.resize(padded_image, target_size)

        return image_resized
class VideoUtils(object):
    
    def get_image(frame, boxes):
        # Extract face locations from bounding boxes
        face_location = [[left, top, right, bottom] for top, right, bottom, left in boxes]
        try:
            # Find the largest face based on area
            xmin, ymin, xmax, ymax = max(face_location, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
        except ValueError as e:
            return None
    
        # Extract the face from the frame
        face_img = frame[max(ymin, 0):ymax, max(xmin, 0):xmax].copy()
        face_img = add_padding_and_resize(face_img)
        return face_img






    