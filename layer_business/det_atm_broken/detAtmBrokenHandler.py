import values.result_code as result_code
from flask import request, jsonify, Blueprint
from datetime import datetime
from layer_infer import det_atm_broken
from utils.ops_image import get_image
from values.class_name_det import atm_broken_class_names


# 创建 Blueprint 实例
det_atm_broken_bp = Blueprint('AtmBrokenDetect', __name__)


@det_atm_broken_bp.route('/', methods=['POST'])
def get_atm_broken_object():
    service_result = {
        'state': result_code.FAILED.SERVICE,  # 状态码
        'is_broken': False,
        'bbox': [],  # 检测框：xyxy int[4]
        'conf': [],  # 置信度：float[1]
        'cls': [],  # 类别：int[1]
        'cost_time': 0,  # 消耗时间：float[4] ms
        'info': '',  # 描述信息：str
        'flow_no': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),  # 流水号：str
    }

    try:
        data = request.json
        image = get_image(data['image_input'])
        result = det_atm_broken(image)

        for k, v in result.items():
            service_result[k] = v

        if set(service_result['cls']).intersection(set(list(atm_broken_class_names.keys())[:2])):
            service_result['is_broken'] = True

        return jsonify(service_result)

    except Exception as e:
        service_result['info'] += f'SERVICE FAILED, OCCUR ERROR: {e}. '

        return jsonify(service_result)
