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

class VideoUtils(object):
    def get_liveness_score(frame, boxes):

        face_location = [[left, top, right, bottom] for top, right, bottom, left in boxes]
        try:
            xmin, ymin, xmax, ymax = max(face_location, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
        except ValueError as e:
            return None

        def get_normal_face(img):
            return img/255.

        face_img = frame[max(ymin, 0):ymax, max(xmin, 0):xmax].copy()
        standard_face = np.array(get_normal_face(cv2.resize(face_img, (112, 112))))
        img_array_expanded = np.expand_dims(standard_face, axis=0)
        return img_array_expanded
    
    def get_image(frame, boxes):
        # Extract face locations from bounding boxes
        face_location = [[left, top, right, bottom] for top, right, bottom, left in boxes]
        try:
            # Find the largest face based on area
            xmin, ymin, xmax, ymax = max(face_location, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
        except ValueError as e:
            return None
    
        # Extract the face from the frame
        face_img = frame[max(ymin-10, 0):ymax+10, max(xmin-30, 0):xmax+30].copy()
    
        return face_img



    def load_keras_model(model_path):
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={"lr": lambda x: x, "SparseCategoricalFocalLoss": lambda x: x})
            model.status = 0
        except Exception as e:
            class empty_model(object):
                def __init__(self):
                    self.status = 600
            model = empty_model()
            print('[ERROR] Model failed to load. Check file path.')
            sys.exit()
        return model


    