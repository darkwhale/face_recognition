import numpy as np
import cv2

import dlib
import base64
import os
from PIL import Image
from io import BytesIO
from service.key_service import generate_key

from service.exception import FaceException

shape_model = os.path.join("model", "shape_predictor_5_face_landmarks.dat")
predict_model = os.path.join("model", "dlib_face_recognition_resnet_model_v1.dat")

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_model)
face_predictor = dlib.face_recognition_model_v1(predict_model)

image_dir = os.path.join("/home", "images")


class ImageHandler(object):
    """
    图片操作类；提供图片从base64解码、人脸检测、人脸特征提取
    备注：opencv读取人脸格式为BGR，输入到dlib时需要转化为RGB；
    """
    def __init__(self,
                 image_base64=None,
                 detect_symbol=True,
                 use_scaled=True,
                 save_symbol=False):
        self.image = self.base64_to_image_mat(image_base64)
        self.detect_symbol = detect_symbol
        self.use_scaled = use_scaled
        self.save_symbol = save_symbol

        if self.image.ndim == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        self.face_chips, self.det_list = self._get_normal_image()

        if self.save_symbol:
            if self.get_face_num() != 1:
                raise FaceException("face num should be 1")
            else:
                self.image_path = self.save_normal_image()

    def get_det_list(self):
        """返回人脸标准化的图像快"""
        return self.det_list

    def get_normal_det_list(self):
        """返回人脸标准化的图像快"""
        return [self.normal_det(det) for det in self.det_list]

    def get_face_num(self):
        return len(self.det_list)

    def get_image_path(self):
        return self.image_path

    def save_normal_image(self):
        """save image"""
        image = Image.fromarray(self.face_chips[0])
        image_name = "{}{}".format(generate_key(), ".png")
        image.save(os.path.join(image_dir, image_name))
        return image_name

    def get_feature(self):
        """获取人脸和对应的人脸特征"""

        face_features = []
        for face_chip in self.face_chips:
            # 注意这里用face_chip[:, :, ::-1]，因为opencv中使用的是BGR格式，需要转化为RGB格式；
            face_features.append(face_predictor.compute_face_descriptor(face_chip))

        return [self.normal_vector(feature) for feature in face_features]

    def _get_normal_image(self):
        """获取标准化后的人脸图片和人脸位置"""

        # 为了提高人脸检测速度，首先进行图片大小resize,最小边在缩放到100；
        if self.use_scaled:
            height, width, channel = self.image.shape
            scale = min(height, width) / 100
            scale_image = cv2.resize(self.image, (int(width / scale), int(height / scale)))

            scale_det_list = detector(scale_image, 1)

            det_list = [dlib.rectangle(
                left=int(scale_det.left() * scale),
                top=int(scale_det.top() * scale),
                right=int(scale_det.right() * scale),
                bottom=int(scale_det.bottom() * scale),
            ) for scale_det in scale_det_list]
        else:
            det_list = detector(self.image, 1)

        faces = dlib.full_object_detections()

        if len(det_list) == 0:
            return [], []

        for det in det_list:
            faces.append(shape_predictor(self.image, det))

        return dlib.get_face_chips(self.image, faces, size=150, padding=0.25), det_list

    @staticmethod
    def base64_to_image_mat(image_base64):
        image_base64 = base64.b64decode(image_base64)
        image_io = BytesIO(image_base64)
        image = Image.open(image_io)

        return np.asarray(image)

    @staticmethod
    def normal_det(det):
        return {
            "left": det.left(),
            "right": det.right(),
            "top": det.top(),
            "bottom": det.bottom()
        }

    @staticmethod
    def normal_vector(vector):
        # return {
        #     "feature": list(vector)
        # }
        return list(vector)

