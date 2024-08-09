import values.result_code as result_code
from flask import request, jsonify, Blueprint
from datetime import datetime
from layer_infer import det_atm_broken
from utils.ops_image import get_image
from values.class_name_det import atm_broken_class_names
from utils.ops_process import filter_detections
import time

# 创建 Blueprint 实例
det_atm_broken_bp = Blueprint('AtmBrokenDetect', __name__)


@det_atm_broken_bp.route('/', methods=['POST'])
def get_atm_broken_object():
    t0 = time.perf_counter()
    service_result = {
        'state': result_code.FAILED.SERVICE,  # 状态码
        'is_broken': False,
        'bbox': [],  # 检测框：xyxy int[4]
        'conf': [],  # 置信度：float[1]
        'cls': [],  # 类别：int[1]
        'cost_time': {'total': 0.0, 'image_read': 0.0, 'obj_det': 0.0},  # 消耗时间：float[4] ms
        'info': '',  # 描述信息：str
        'flow_no': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),  # 流水号：str
    }

    try:
        data = request.json
        image = get_image(data['image_input'])
        t1 = time.perf_counter()
        result = det_atm_broken(image)

        obj_bbox, obj_conf, obj_cls = filter_detections(result, list(atm_broken_class_names.keys())[:2])

        service_result['state'] = result['state']
        service_result['bbox'] = obj_bbox
        service_result['conf'] = obj_conf
        service_result['cls'] = obj_cls
        service_result['info'] = result['info']
        service_result['cost_time']['image_read'] = round((t1 - t0) * 1000, 2)
        service_result['cost_time']['obj_det'] = round((time.perf_counter() - t0) * 1000, 2)
        service_result['cost_time']['total'] = round((time.perf_counter() - t0) * 1000, 2)

        if len(obj_bbox) > 0:
            service_result['is_broken'] = True

        return jsonify(service_result)

    except Exception as e:
        service_result['info'] += f'SERVICE FAILED, OCCUR ERROR: {e}. '

        return jsonify(service_result)
