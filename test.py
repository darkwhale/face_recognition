import base64


def image_to_base64(file_path):
    """
    图片文件转bse64；
    :param file_path: 图片路径；
    :return: 返回base64字符串；
    """
    with open(file_path, "rb") as reader:
        image_base_64 = base64.b64encode(reader.read())

    return image_base_64.decode("utf-8")


if __name__ == '__main__':
    print(image_to_base64("1.jpg"))