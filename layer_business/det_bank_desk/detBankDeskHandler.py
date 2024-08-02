import time
import values.result_code as result_code
from flask import request, jsonify, Blueprint
from datetime import datetime
from layer_infer import det_bank_desk
from utils.ops_image import get_image


# 创建 Blueprint 实例
det_bank_desk_bp = Blueprint('bankDeskDetect', __name__)


@det_bank_desk_bp.route('/', methods=['POST'])
def get_bank_desk_object():
    t0 = time.perf_counter()
    service_result = {
        'state': result_code.FAILED.SERVICE,  # 状态码
        'bbox': [],  # 检测框：xyxy int[4]
        'conf': [],  # 置信度：float[1]
        'cls': [],  # 类别：int[1]
        'cost_time': {'total': 0.0, 'image_read': 0.0, 'bank_desk_det': []},
        'info': '',  # 描述信息：str
        'flow_no': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),  # 流水号：str
    }

    try:
        data = request.json
        image = get_image(data['image_input'])
        service_result['cost_time']['image_read'] = round((time.perf_counter() - t0) * 1000, 2)
        result = det_bank_desk(image)

        for k, v in result.items():
            if k == 'cost_time':
                continue
            service_result[k] = v

        service_result['cost_time']['bank_desk_det'] = result['cost_time']
        service_result['cost_time']['total'] = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify(service_result)

    except Exception as e:
        service_result['info'] += f'SERVICE FAILED, OCCUR ERROR: {e}. '
        service_result['cost_time']['total'] = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify(service_result)
