import time
import values.result_code as result_code
from flask import request, jsonify, Blueprint
from datetime import datetime
from layer_infer import det_coco
from utils.ops_image import get_image
from utils.ops_process import filter_detections


# 创建 Blueprint 实例
det_person_bp = Blueprint('personDetect', __name__)


@det_person_bp.route('/', methods=['POST'])
def get_person():
    t0 = time.perf_counter()
    service_result = {
        'state': result_code.FAILED.SERVICE,  # 状态码
        'bbox': [],     # 检测框：xyxy int[4]
        'conf': [],     # 置信度：float[1]
        'cls': [],      # 类别：int[1]
        'cost_time': {'total': 0.0, 'image_read': 0.0, 'person_det': 0.0},
        'info': '',     # 描述信息：str
        'flow_no': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),  # 流水号：str
    }

    try:
        data = request.json
        image = get_image(data['image_input'])
        t1 = time.perf_counter()
        result = det_coco(image)
        person_bbox, person_conf, person_cls = filter_detections(result, (0,))

        service_result['state'] = result['state']
        service_result['bbox'] = person_bbox
        service_result['conf'] = person_conf
        service_result['cls'] = person_cls
        service_result['info'] = result['info']
        service_result['cost_time']['image_read'] = round((t1 - t0) * 1000, 2)
        service_result['cost_time']['person_det'] = round((time.perf_counter() - t0) * 1000, 2)
        service_result['cost_time']['total'] = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify(service_result)

    except Exception as e:
        service_result['info'] += f'SERVICE FAILED, OCCUR ERROR: {e}. '
        service_result['cost_time']['total'] = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify(service_result)
