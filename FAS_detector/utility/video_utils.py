import cv2
import numpy as np
import os
import sys
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import tensorflow as tf
from face_detector import FaceDetector
from face_det.FaceBoxes import FaceBoxes

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class VideoUtils(object):

    def load_facedetector(root_path):

        face_boxes = FaceBoxes(gpu_mode=True)
        return face_boxes

    def detect_faces(img):


        face_boxes = FaceBoxes(gpu_mode=True)
        faces = face_boxes(img)
        faces = np.array([[top, right, bottom, left] for left, top, right, bottom, _ in faces]).astype(int).tolist()
        return faces

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
        face_img = frame[max(ymin, 0):ymax, max(xmin, 0):xmax].copy()
    
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

    def get_normal_face(img):
        return img/255.

    def get_max_face(face_locations):
        return max(face_locations, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))

    def cv_draw_landmark(img_ori, pts, box=None, size=1):
        GREEN = (0, 255, 0)
        img = img_ori.copy()
        n = pts.shape[1]
        if n <= 106:
            for i in range(n):
                cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, GREEN, -1)
        else:
            sep = 1
            for i in range(0, n, sep):
                cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, GREEN, 1)
        if box is not None:
            left, top, right, bottom = np.round(box).astype(np.int32)
            left_top = (left, top)
            right_top = (right, top)
            right_bottom = (right, bottom)
            left_bottom = (left, bottom)
            cv2.line(img, left_top, right_top, GREEN, 1, cv2.LINE_AA)
            cv2.line(img, right_top, right_bottom, GREEN, 1, cv2.LINE_AA)
            cv2.line(img, right_bottom, left_bottom, GREEN, 1, cv2.LINE_AA)
            cv2.line(img, left_bottom, left_top, GREEN, 1, cv2.LINE_AA)

        return img