import json
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from service import face_handler


@csrf_exempt
def face_controller(request):
    """
    解析图片，返回人脸特征
    0: 成功
    1: 未检测到人脸
    2: 其它错误
    """
    if request.method == "POST":
        post_data = json.loads(request.body, strict=False)

        if post_data:
            image_json = post_data.get("image_json", None)
            save_symbol = True if post_data.get("save_symbol", None) == "save" else False

            if image_json is not None:
                response = face_handler.get_feature(image_json, save_symbol)

                code = 0 if response else 1

                response_json = json.dumps({
                    "code": code,
                    "data": response,
                })

                return HttpResponse(response_json, content_type="application/json")

            else:
                response_json = json.dumps({
                    "code": 2,
                    "data": {},
                })

                return HttpResponse(response_json, content_type="application/json")

    response_json = json.dumps({
        "code": 3,
        "data": {},
    })

    return HttpResponse(response_json, content_type="application/json")



