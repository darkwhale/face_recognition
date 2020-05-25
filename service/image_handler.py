import numpy as np
import cv2

import dlib
import base64
import os
from PIL import Image
from io import BytesIO

shape_model = os.path.join("model", "shape_predictor_5_face_landmarks.dat")
predict_model = os.path.join("model", "dlib_face_recognition_resnet_model_v1.dat")

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_model)
face_predictor = dlib.face_recognition_model_v1(predict_model)


class ImageHandler(object):
    """
    图片操作类；提供图片从base64解码、人脸检测、人脸特征提取
    备注：opencv读取人脸格式为BGR，输入到dlib时需要转化为RGB；
    """
    def __init__(self,
                 image_base64=None,
                 detect_symbol=True,
                 use_scaled=True):
        self.image = self.base64_to_image_mat(image_base64)
        self.detect_symbol = detect_symbol
        self.use_scaled = use_scaled

        if self.image.ndim == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def base64_to_image_mat(image_base64):
        image_base64 = base64.b64decode(image_base64)
        image_io = BytesIO(image_base64)
        image = Image.open(image_io)
        np_list = np.asarray(image)

        return np_list