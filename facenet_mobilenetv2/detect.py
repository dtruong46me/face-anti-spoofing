from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN(image_size=1088)

img = Image.open('C:/Users/VU TUAN MINH/Downloads/fas_face_dataset/fas_face_dataset/fas_face_dataset/real/_168836997114137_2.jpg')

img_cropped = mtcnn(img, save_path='/mtcnn_mobilenetv2/img_detect/_168836997114137_2_detect.jpg')
