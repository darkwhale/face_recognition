from service.image_handler import ImageHandler


def get_feature(image_json):
    """根据base64检测出人脸特征，为列表，包含人脸位置，人脸向量"""
    base64_index = image_json.find("base64")
    if base64_index == -1:
        return []
    image_data = image_json[base64_index + 7:]

    image_handler = ImageHandler(image_data)

    return zip(image_handler.get_det_list(), image_handler.get_feature())
