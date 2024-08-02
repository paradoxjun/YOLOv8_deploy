import time
import values.result_code as result_code
from flask import request, jsonify, Blueprint
from datetime import datetime
from layer_infer import det_hand, det_coco
from utils.ops_image import get_image
from utils.compute import nms
from utils.ops_process import filter_detections, expand_bbox, bbox_offset


# 创建 Blueprint 实例
det_hand_bp = Blueprint('handDetect', __name__)


@det_hand_bp.route('/base', methods=['POST'])
def get_hand():
    t0 = time.perf_counter()
    service_result = {
        'state': result_code.FAILED.SERVICE,  # 状态码
        'bbox': [],     # 检测框：xyxy int[4]
        'conf': [],     # 置信度：float[1]
        'cls': [],      # 类别：int[1]
        'cost_time': {'total': 0.0, 'image_read': 0.0, 'hand_det': []},
        'info': '',     # 描述信息：str
        'flow_no': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),  # 流水号：str
    }

    try:
        data = request.json
        image = get_image(data['image_input'])
        service_result['cost_time']['image_read'] = round((time.perf_counter() - t0) * 1000, 2)
        result = det_hand(image)

        for k, v in result.items():
            if k == 'cost_time':
                continue
            service_result[k] = v

        service_result['cost_time']['hand_det'] = result['cost_time']
        service_result['cost_time']['total'] = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify(service_result)

    except Exception as e:
        service_result['info'] += f'SERVICE FAILED, OCCUR ERROR: {e}. '
        service_result['cost_time']['total'] = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify(service_result)


@det_hand_bp.route('/by_person', methods=['POST'])
def get_hand_by_person():
    t0 = time.perf_counter()
    service_result = {
        'state': result_code.FAILED.SERVICE,  # 状态码
        'bbox': [],  # 检测框：xyxy int[4]
        'conf': [],  # 置信度：float[1]
        'cls': [],  # 类别：int[1]
        'cost_time': {'total': 0.0, 'image_read': 0.0, 'person_det': 0.0, 'hand_det': 0.0},
        'info': '',  # 描述信息：str
        'flow_no': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),  # 流水号：str
    }

    try:
        data = request.json
        image = get_image(data['image_input'])
        height, width, _ = image.shape

        service_result['cost_time']['image_read'] = round((time.perf_counter() - t0) * 1000, 2)
        person_bbox, _, _ = filter_detections(det_coco(image), (0,))
        t1 = time.perf_counter()
        service_result['cost_time']['person_det'] = round((t1 - t0) * 1000, 2)

        hand_bbox = []
        hand_conf = []
        hand_cls = []

        for p_bbox in person_bbox:
            px1, py1, px2, py2 = expand_bbox(p_bbox, width, height, scale=0.12)
            result_hand_det = det_hand(image[py1:py2, px1:px2])
            hand_bbox.extend(bbox_offset(result_hand_det['bbox'], px1, py1))
            hand_conf.extend(result_hand_det['conf'])
            hand_cls.extend(result_hand_det['cls'])

        index = nms(hand_bbox, hand_conf, iou_threshold=0.6)
        for i in index:
            service_result['bbox'].append(hand_bbox[i])
            service_result['conf'].append(hand_conf[i])
            service_result['cls'].append(hand_cls[i])

        service_result['state'] = result_code.SUCCESS.COMMON
        service_result['cost_time']['hand_det'] = round((time.perf_counter() - t1) * 1000, 2)
        service_result['cost_time']['total'] = round((time.perf_counter() - t0) * 1000, 2)
        service_result['info'] = (f"RUN SUCCESS: Total time: {service_result['cost_time']['total']} ms, "
                                  f"Image read time: {service_result['cost_time']['image_read']}, "
                                  f"Person det time: {service_result['cost_time']['person_det']} ms, "
                                  f"Hand det time: {service_result['cost_time']['hand_det']} ms. ")

        return jsonify(service_result)

    except Exception as e:
        service_result['info'] += f'SERVICE FAILED, OCCUR ERROR: {e}. '
        service_result['cost_time']['total'] = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify(service_result)
