import os
import sys
root_path = "face_det"
sys.path.append(root_path)
import numpy as np
from face_det.FaceBoxes import FaceBoxes


use_gpu_flag = 1

onnx_flag = False
if onnx_flag:
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from face_det.FaceBoxes_ONNX import FaceBoxes_ONNX
    face_boxes = FaceBoxes_ONNX()
else:
    face_boxes = FaceBoxes(gpu_mode=False if use_gpu_flag == 0 else True)

class FaceDetector(object):

    def __call__(self, img, dense_flag=False):

        faces = face_boxes(img)

        faces = np.array([[top, right, bottom, left] for left, top, right, bottom, _ in faces]).astype(int).tolist()
        return faces
