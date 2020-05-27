from service.image_handler import ImageHandler


def make_dict(image_path, det, feature):
    return {
        "image_name": image_path,
        "rectangle": det,
        "feature": feature,
    }


def make_feature(det, feature):
    return {
        "rectangle": det,
        "feature": feature,
    }


def get_feature(image_json, save_symbol=False):
    """根据base64检测出人脸特征，为列表，包含人脸位置，人脸向量"""
    base64_index = image_json.find("base64")
    if base64_index == -1:
        return []
    # noinspection PyBroadException
    try:
        image_data = image_json[base64_index + 7:]

        image_handler = ImageHandler(image_data, save_symbol=save_symbol)

        if save_symbol:
            return make_dict(image_handler.get_image_path(),
                             image_handler.get_normal_det_list()[0],
                             image_handler.get_feature()[0])

        return [make_feature(det, feature) for det, feature in zip(image_handler.get_normal_det_list(), image_handler.get_feature())]
    except Exception:
        return []
